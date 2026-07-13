import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nowcasting.models.nowcastnet_pwv_v3 import PWVCoupledNetV3
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
from train_pwv_coupled_v2 import (
    add_model_args,
    build_parser as build_v2_parser,
    coupling_alignment_loss,
    coupling_smoothness,
    make_dataloader,
    pwv_physical_signal,
    shuffle_contrast_loss,
)


def build_parser():
    parser = build_v2_parser()
    parser.description = "Train PWV-coupled NowcastNet V3 with false-alarm control"
    parser.set_defaults(
        model_name="PWVCoupledNowcastNetV3",
        save_dir="../checkpoints/pwv_coupled_v3",
        readme_ckpt="../checkpoints/pwv_coupled_v3_model.ckpt",
    )
    parser.add_argument("--fusion_channels", type=int, default=32)
    parser.add_argument("--lambda_false_alarm", type=float, default=0.25)
    parser.add_argument("--lambda_support_dry", type=float, default=0.05)
    parser.add_argument("--lambda_support_l1", type=float, default=0.01)
    parser.add_argument("--false_alarm_threshold", type=float, default=2.0)
    return parser


def dry_unsupported_mask(target, frames, pwv, args):
    last_rain = frames[:, args.input_length - 1, :, :, 0]
    dry = (target < args.false_alarm_threshold).float()
    no_recent_rain = (last_rain < args.false_alarm_threshold).float().unsqueeze(1)
    pwv_signal = pwv_physical_signal(pwv, args.input_length).unsqueeze(1)
    unsupported = (1.0 - pwv_signal).clamp(0.0, 1.0)
    return dry * no_recent_rain * unsupported


def false_alarm_loss(pred, target, frames, pwv, args):
    mask = dry_unsupported_mask(target, frames, pwv, args)
    excess = F.relu(pred - args.false_alarm_threshold)
    return (excess * mask).sum() / mask.sum().clamp_min(1.0)


def support_dry_loss(support_gate, target, frames, pwv, args):
    support = support_gate[:, :, 0]
    mask = dry_unsupported_mask(target, frames, pwv, args)
    return (support * mask).sum() / mask.sum().clamp_min(1.0)


def generator_losses(generator, frames, pwv, aux, target, discriminator, args):
    pred = aux["prediction"][..., 0]
    evo = aux["evolution"] * args.intensity_scale
    advected = aux["advected"]
    coupling = aux["coupling"]
    support_gate = aux["support_gate"]

    forecast_loss = weighted_l1(pred, target, args.intensity_scale)
    evolution_loss = weighted_l1(evo, target, args.intensity_scale)
    advected_loss = weighted_l1(advected, target, args.intensity_scale)
    motion_loss = motion_regularization(aux["motion"], target, args.intensity_scale)
    pool_loss = pooled_l1(pred, target)
    fake_logits = discriminator(discriminator_sequence(pred, args.intensity_scale))
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    coupling_smooth = coupling_smoothness(coupling)
    coupling_l1 = coupling.mean()
    support_l1 = support_gate.mean()
    align_loss = coupling_alignment_loss(coupling * support_gate, target, frames, pwv, args)
    shuffle_loss = shuffle_contrast_loss(generator, frames, pwv, target, aux, args)
    fa_loss = false_alarm_loss(pred, target, frames, pwv, args)
    dry_support_loss = support_dry_loss(support_gate, target, frames, pwv, args)

    total = (
        args.lambda_forecast * forecast_loss
        + args.lambda_evolution * evolution_loss
        + args.lambda_advected * advected_loss
        + args.lambda_motion * motion_loss
        + args.lambda_pool * pool_loss
        + args.lambda_adv * adv_loss
        + args.lambda_coupling_smooth * coupling_smooth
        + args.lambda_coupling_l1 * coupling_l1
        + args.lambda_support_l1 * support_l1
        + args.lambda_align * align_loss
        + args.lambda_shuffle * shuffle_loss
        + args.lambda_false_alarm * fa_loss
        + args.lambda_support_dry * dry_support_loss
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
        "support_mean": support_l1.detach(),
        "c_align": align_loss.detach(),
        "pwv_shuffle": shuffle_loss.detach(),
        "false_alarm": fa_loss.detach(),
        "support_dry": dry_support_loss.detach(),
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
            for key in ("g_total", "d_total", "forecast", "false_alarm", "support_mean", "support_dry"):
                msg += " {} {:.5f}".format(key, totals.get(key, 0.0) / max(seen, 1))
            print(msg, flush=True)

    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(generator, loader, args):
    generator.eval()
    total = 0.0
    seen = 0
    c_mean = 0.0
    s_mean = 0.0
    fa = 0.0
    for batch in loader:
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        aux = generator(frames, pwv, return_aux=True)
        pred = aux["prediction"][..., 0]
        loss = weighted_l1(pred, target, args.intensity_scale)
        total += loss.item() * frames.size(0)
        c_mean += aux["coupling"].mean().item() * frames.size(0)
        s_mean += aux["support_gate"].mean().item() * frames.size(0)
        fa += false_alarm_loss(pred, target, frames, pwv, args).item() * frames.size(0)
        seen += frames.size(0)
    return total / max(seen, 1), c_mean / max(seen, 1), s_mean / max(seen, 1), fa / max(seen, 1)


def save_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args):
    safe_torch_save(
        {
            "model": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )


def main():
    args = add_model_args(build_parser().parse_args())
    if args.amp:
        print("PWV V3 currently uses full precision for numerical stability; ignoring --amp.", flush=True)
        args.amp = False
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(args.readme_ckpt).parent.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_loader = make_dataloader(args, "train", args.max_train_samples)
    val_loader = make_dataloader(args, "val", args.max_val_samples)
    print("train windows: {} val windows: {}".format(len(train_loader.dataset), len(val_loader.dataset)), flush=True)

    generator = PWVCoupledNetV3(args).to(args.device)
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
        val_loss, val_c_mean, val_support_mean, val_false_alarm = validate(generator, val_loader, args)
        metrics["val_c_mean"] = val_c_mean
        metrics["val_support_mean"] = val_support_mean
        metrics["val_false_alarm"] = val_false_alarm
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
