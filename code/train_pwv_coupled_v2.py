import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nowcasting.experiments.common import (
    add_model_runtime_args as add_model_args,
    build_generator,
    load_generator_weights,
    make_png_dataloader,
    save_adversarial_checkpoint,
    save_json_args,
)
from nowcasting.models.temporal_discriminator import TemporalDiscriminator
from train_adversarial_custom import (
    append_epoch_log,
    autocast_context,
    discriminator_sequence,
    make_grad_scaler,
    motion_regularization,
    pooled_l1,
    safe_torch_save,
    seed_everything,
    weighted_l1,
    write_training_plot,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train PWV-coupled NowcastNet V2")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--pwv_root", type=str, default="../data/DATA_2025_S/PWV_2025_S")
    parser.add_argument("--save_dir", type=str, default="../checkpoints/pwv_coupled_v2")
    parser.add_argument("--readme_ckpt", type=str, default="../checkpoints/pwv_coupled_v2_model.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="PWVCoupledNowcastNetV2")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--evo_base_channels", type=int, default=32)
    parser.add_argument("--pwv_base_channels", type=int, default=24)
    parser.add_argument("--disc_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--intensity_scale", type=float, default=128.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=1.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--lambda_forecast", type=float, default=1.0)
    parser.add_argument("--lambda_evolution", type=float, default=0.5)
    parser.add_argument("--lambda_advected", type=float, default=0.25)
    parser.add_argument("--lambda_motion", type=float, default=0.02)
    parser.add_argument("--lambda_pool", type=float, default=0.2)
    parser.add_argument("--lambda_adv", type=float, default=0.01)
    parser.add_argument("--lambda_coupling_smooth", type=float, default=0.02)
    parser.add_argument("--lambda_coupling_l1", type=float, default=0.0005)
    parser.add_argument("--lambda_align", type=float, default=0.05)
    parser.add_argument("--lambda_shuffle", type=float, default=0.05)
    parser.add_argument("--shuffle_margin", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init_generator", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser


def make_dataloader(args, split, max_samples):
    return make_png_dataloader(args, split, max_samples)


def coupling_smoothness(coupling):
    dx = coupling[..., :, 1:] - coupling[..., :, :-1]
    dy = coupling[..., 1:, :] - coupling[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def normalize_positive(x):
    x = F.relu(x)
    denom = x.amax(dim=tuple(range(1, x.dim())), keepdim=True).clamp_min(1e-6)
    return x / denom


def pwv_physical_signal(pwv, input_length):
    history = pwv[:, :input_length]
    last = history[:, -1]
    first = history[:, 0]
    delta = F.relu(last - first)
    dx = F.pad(last[..., 1:] - last[..., :-1], (0, 1, 0, 0))
    dy = F.pad(last[..., 1:, :] - last[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx * dx + dy * dy + 1e-6)
    return normalize_positive(delta + gradient)


def coupling_alignment_loss(coupling, target, frames, pwv, args):
    last_radar = frames[:, args.input_length - 1, :, :, 0]
    rain_growth = normalize_positive(target - last_radar.unsqueeze(1))
    pwv_signal = pwv_physical_signal(pwv, args.input_length).unsqueeze(1)
    align_weight = rain_growth * pwv_signal
    return ((1.0 - coupling[:, :, 0]) * align_weight).sum() / align_weight.sum().clamp_min(1e-6)


def make_shuffled_pwv(pwv):
    if pwv.size(0) > 1:
        return pwv[torch.randperm(pwv.size(0), device=pwv.device)]
    return torch.flip(pwv, dims=[1])


def shuffle_contrast_loss(generator, frames, pwv, target, aux, args):
    if args.lambda_shuffle <= 0:
        return target.new_tensor(0.0)
    shuffled_pwv = make_shuffled_pwv(pwv)
    with torch.no_grad():
        shuffled_aux = generator(frames, shuffled_pwv, return_aux=True)
    real_evo = aux["evolution"] * args.intensity_scale
    shuffled_evo = shuffled_aux["evolution"] * args.intensity_scale
    real_loss = weighted_l1(real_evo, target, args.intensity_scale)
    shuffled_loss = weighted_l1(shuffled_evo, target, args.intensity_scale)
    return F.relu(real_loss - shuffled_loss + args.shuffle_margin)


def generator_losses(generator, frames, pwv, aux, target, discriminator, args):
    pred = aux["prediction"][..., 0]
    evo = aux["evolution"] * args.intensity_scale
    advected = aux["advected"]
    coupling = aux["coupling"]

    forecast_loss = weighted_l1(pred, target, args.intensity_scale)
    evolution_loss = weighted_l1(evo, target, args.intensity_scale)
    advected_loss = weighted_l1(advected, target, args.intensity_scale)
    motion_loss = motion_regularization(aux["motion"], target, args.intensity_scale)
    pool_loss = pooled_l1(pred, target)
    fake_logits = discriminator(discriminator_sequence(pred, args.intensity_scale))
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    coupling_smooth = coupling_smoothness(coupling)
    coupling_l1 = coupling.mean()
    align_loss = coupling_alignment_loss(coupling, target, frames, pwv, args)
    shuffle_loss = shuffle_contrast_loss(generator, frames, pwv, target, aux, args)

    total = (
        args.lambda_forecast * forecast_loss
        + args.lambda_evolution * evolution_loss
        + args.lambda_advected * advected_loss
        + args.lambda_motion * motion_loss
        + args.lambda_pool * pool_loss
        + args.lambda_adv * adv_loss
        + args.lambda_coupling_smooth * coupling_smooth
        + args.lambda_coupling_l1 * coupling_l1
        + args.lambda_align * align_loss
        + args.lambda_shuffle * shuffle_loss
    )
    parts = {
        "g_total": total.detach(),
        "forecast": forecast_loss.detach(),
        "evolution": evolution_loss.detach(),
        "advected": advected_loss.detach(),
        "motion": motion_loss.detach(),
        "pool": pool_loss.detach(),
        "g_adv": adv_loss.detach(),
        "c_smooth": coupling_smooth.detach(),
        "c_mean": coupling_l1.detach(),
        "c_align": align_loss.detach(),
        "pwv_shuffle": shuffle_loss.detach(),
    }
    return total, parts


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, scaler_g, scaler_d, args):
    generator.train()
    discriminator.train()
    totals = {}
    seen = 0

    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)

        opt_d.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            with torch.no_grad():
                fake = generator(frames, pwv)[..., 0]
            real_logits = discriminator(discriminator_sequence(target, args.intensity_scale))
            fake_logits = discriminator(discriminator_sequence(fake.detach(), args.intensity_scale))
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_loss = 0.5 * (real_loss + fake_loss)
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        for param in discriminator.parameters():
            param.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            aux = generator(frames, pwv, return_aux=True)
            g_loss, parts = generator_losses(generator, frames, pwv, aux, target, discriminator, args)
        scaler_g.scale(g_loss).backward()
        if args.grad_clip > 0:
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
        scaler_g.step(opt_g)
        scaler_g.update()
        for param in discriminator.parameters():
            param.requires_grad_(True)

        batch_size = frames.size(0)
        seen += batch_size
        parts["d_total"] = d_loss.detach()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size

        if step % args.log_interval == 0:
            msg = "step {:05d}".format(step)
            for key in ("g_total", "d_total", "forecast", "evolution", "c_mean", "c_align", "pwv_shuffle"):
                msg += " {} {:.5f}".format(key, totals.get(key, 0.0) / max(seen, 1))
            print(msg, flush=True)

    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(generator, loader, args):
    generator.eval()
    total = 0.0
    seen = 0
    c_mean = 0.0
    c_std = 0.0
    c_align = 0.0
    for batch in loader:
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        aux = generator(frames, pwv, return_aux=True)
        pred = aux["prediction"][..., 0]
        loss = weighted_l1(pred, target, args.intensity_scale)
        total += loss.item() * frames.size(0)
        c_mean += aux["coupling"].mean().item() * frames.size(0)
        c_std += aux["coupling"].std().item() * frames.size(0)
        c_align += coupling_alignment_loss(aux["coupling"], target, frames, pwv, args).item() * frames.size(0)
        seen += frames.size(0)
    return total / max(seen, 1), c_mean / max(seen, 1), c_std / max(seen, 1), c_align / max(seen, 1)


def save_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args):
    save_adversarial_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args)


def main():
    args = add_model_args(build_parser().parse_args())
    if args.amp:
        print("PWV V2 currently uses full precision for numerical stability; ignoring --amp.", flush=True)
        args.amp = False
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    Path(args.readme_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_json_args(args, save_dir)

    train_loader = make_dataloader(args, "train", args.max_train_samples)
    val_loader = make_dataloader(args, "val", args.max_val_samples)
    print("train windows: {} val windows: {}".format(len(train_loader.dataset), len(val_loader.dataset)), flush=True)

    generator = build_generator(args)
    if args.init_generator:
        load_generator_weights(generator, args.init_generator, args.device)
        print("initialized generator from {}".format(args.init_generator), flush=True)
    discriminator = TemporalDiscriminator(args.gen_oc, base_channels=args.disc_channels).to(args.device)
    opt_g = torch.optim.Adam((p for p in generator.parameters() if p.requires_grad), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    scaler_g = make_grad_scaler(args.device, args.amp)
    scaler_d = make_grad_scaler(args.device, args.amp)

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        generator.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_g.load_state_dict(checkpoint["optimizer_g"])
        opt_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("val_loss", best_val))

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, scaler_g, scaler_d, args)
        val_loss, val_c_mean, val_c_std, val_c_align = validate(generator, val_loader, args)
        metrics["val_c_mean"] = val_c_mean
        metrics["val_c_std"] = val_c_std
        metrics["val_c_align"] = val_c_align
        append_epoch_log(save_dir / "train_log.csv", epoch, val_loss, metrics)
        write_training_plot(save_dir / "train_log.csv", save_dir / "train_log.png")
        metric_text = " ".join("{} {:.5f}".format(k, v) for k, v in sorted(metrics.items()))
        print("epoch {:03d} val_weighted_l1 {:.5f} {}".format(epoch, val_loss, metric_text), flush=True)

        save_checkpoint(save_dir / "latest.ckpt", generator, discriminator, opt_g, opt_d, epoch, val_loss, args)
        safe_torch_save(generator.state_dict(), save_dir / "latest_state_dict.ckpt")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(save_dir / "best.ckpt", generator, discriminator, opt_g, opt_d, epoch, val_loss, args)
            safe_torch_save(generator.state_dict(), save_dir / "best_state_dict.ckpt")
            safe_torch_save(generator.state_dict(), args.readme_ckpt)
            print("saved best generator to {}".format(args.readme_ckpt), flush=True)


if __name__ == "__main__":
    main()
