import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from nowcasting.experiments.common import (
    add_model_runtime_args as add_model_args,
    build_generator,
    load_generator_weights,
    make_png_dataloader,
    safe_torch_save,
    save_adversarial_checkpoint,
    save_json_args,
    seed_everything,
)
from nowcasting.losses import fourier_amplitude_and_correlation_loss
from nowcasting.models.temporal_discriminator import TemporalDiscriminator

try:
    from torch.amp import GradScaler as TorchGradScaler
    from torch.amp import autocast as torch_autocast

    def make_grad_scaler(device, enabled):
        return TorchGradScaler(device.split(":")[0], enabled=enabled)

    def autocast_context(device, enabled):
        return torch_autocast(device_type=device.split(":")[0], enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler
    from torch.cuda.amp import autocast as cuda_autocast

    def make_grad_scaler(device, enabled):
        return TorchGradScaler(enabled=enabled)

    def autocast_context(device, enabled):
        return cuda_autocast(enabled=enabled)


def build_parser():
    parser = argparse.ArgumentParser(description="Adversarial NowcastNet training for custom PNG radar data")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--save_dir", type=str, default="../checkpoints/custom_nowcastnet_adv")
    parser.add_argument("--readme_ckpt", type=str, default="../checkpoints/mrms_model.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="NowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--disc_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--lambda_forecast", type=float, default=1.0)
    parser.add_argument("--lambda_evolution", type=float, default=0.5)
    parser.add_argument("--lambda_advected", type=float, default=0.25)
    parser.add_argument("--lambda_motion", type=float, default=0.02)
    parser.add_argument("--lambda_pool", type=float, default=0.2)
    parser.add_argument("--lambda_adv", type=float, default=0.01)
    parser.add_argument("--lambda_facl", type=float, default=0.0)
    parser.add_argument("--facl_fal_probability", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init_generator", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser


def make_dataloader(args, split, max_samples):
    return make_png_dataloader(args, split, max_samples)


def weighted_l1(pred, target, intensity_scale):
    scale = max(float(intensity_scale), 1.0)
    weight = torch.ones_like(target)
    weight = weight + 2.0 * (target >= 0.125 * scale).float()
    weight = weight + 4.0 * (target >= 0.25 * scale).float()
    weight = weight + 8.0 * (target >= 0.5 * scale).float()
    weight = weight * (1.0 + target / scale)
    return (weight * (pred - target).abs()).mean()


def pooled_l1(pred, target):
    loss = 0.0
    for kernel in (2, 4, 8):
        pred_pool = F.avg_pool2d(pred.reshape(-1, 1, pred.shape[-2], pred.shape[-1]), kernel, stride=kernel)
        target_pool = F.avg_pool2d(target.reshape(-1, 1, target.shape[-2], target.shape[-1]), kernel, stride=kernel)
        loss = loss + F.l1_loss(pred_pool, target_pool)
    return loss / 3.0


def motion_regularization(motion, target, intensity_scale):
    dx = motion[..., :, 1:] - motion[..., :, :-1]
    dy = motion[..., 1:, :] - motion[..., :-1, :]
    rain_x = target[..., :, 1:] / max(float(intensity_scale), 1.0)
    rain_y = target[..., 1:, :] / max(float(intensity_scale), 1.0)
    return (dx.abs() * rain_x.unsqueeze(2)).mean() + (dy.abs() * rain_y.unsqueeze(2)).mean()


def discriminator_sequence(seq, intensity_scale):
    return torch.clamp(seq / max(float(intensity_scale), 1.0), 0.0, 1.0)


def facl_reconstruction_loss(pred, target, args):
    if getattr(args, "lambda_facl", 0.0) <= 0:
        return target.new_tensor(0.0)
    pred_norm = discriminator_sequence(pred, args.intensity_scale)
    target_norm = discriminator_sequence(target, args.intensity_scale)
    return fourier_amplitude_and_correlation_loss(
        pred_norm,
        target_norm,
        fal_probability=getattr(args, "facl_fal_probability", 0.5),
    )


def generator_losses(aux, target, discriminator, args):
    pred = aux["prediction"][..., 0]
    evo = aux["evolution"] * args.intensity_scale
    advected = aux["advected"]

    forecast_loss = weighted_l1(pred, target, args.intensity_scale)
    evolution_loss = weighted_l1(evo, target, args.intensity_scale)
    advected_loss = weighted_l1(advected, target, args.intensity_scale)
    motion_loss = motion_regularization(aux["motion"], target, args.intensity_scale)
    pool_loss = pooled_l1(pred, target)
    fake_logits = discriminator(discriminator_sequence(pred, args.intensity_scale))
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    facl_loss = facl_reconstruction_loss(pred, target, args)

    total = (
        args.lambda_forecast * forecast_loss
        + args.lambda_evolution * evolution_loss
        + args.lambda_advected * advected_loss
        + args.lambda_motion * motion_loss
        + args.lambda_pool * pool_loss
        + args.lambda_adv * adv_loss
        + args.lambda_facl * facl_loss
    )
    parts = {
        "g_total": total.detach(),
        "forecast": forecast_loss.detach(),
        "evolution": evolution_loss.detach(),
        "advected": advected_loss.detach(),
        "motion": motion_loss.detach(),
        "pool": pool_loss.detach(),
        "g_adv": adv_loss.detach(),
        "facl": facl_loss.detach(),
    }
    return total, parts


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, scaler_g, scaler_d, args):
    generator.train()
    discriminator.train()
    totals = {}
    seen = 0

    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)

        opt_d.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            with torch.no_grad():
                fake = generator(frames)[..., 0]
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
            aux = generator(frames, return_aux=True)
            g_loss, parts = generator_losses(aux, target, discriminator, args)
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
            for key in ("g_total", "d_total", "forecast", "evolution", "g_adv"):
                msg += " {} {:.5f}".format(key, totals.get(key, 0.0) / max(seen, 1))
            print(msg, flush=True)

    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(generator, loader, args):
    generator.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        pred = generator(frames)[..., 0]
        loss = weighted_l1(pred, target, args.intensity_scale)
        total += loss.item() * frames.size(0)
        seen += frames.size(0)
    return total / max(seen, 1)


def save_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args):
    save_adversarial_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args)


def append_epoch_log(path, epoch, val_loss, metrics):
    row = {"epoch": epoch, "val_weighted_l1": val_loss}
    row.update(metrics)
    fieldnames = ["epoch", "val_weighted_l1"] + sorted(metrics.keys())
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_training_plot(csv_path, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = []
    val_losses = []
    g_losses = []
    d_losses = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            val_losses.append(float(row["val_weighted_l1"]))
            g_losses.append(float(row.get("g_total", 0.0)))
            d_losses.append(float(row.get("d_total", 0.0)))

    if not epochs:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, val_losses, label="val_weighted_l1")
    plt.plot(epochs, g_losses, label="g_total")
    plt.plot(epochs, d_losses, label="d_total")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = add_model_args(build_parser().parse_args())
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
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
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
        val_loss = validate(generator, val_loader, args)
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
