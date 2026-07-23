"""Train end-to-end radar/PWV latent fusion from a matched radar checkpoint."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from nowcasting.birth_growth import apply_pwv_control
from nowcasting.experiments.common import (
    add_model_runtime_args,
    build_generator,
    load_radar_backbone_weights,
    make_png_dataloader,
    save_adversarial_checkpoint,
    save_dataset_provenance,
    save_json_args,
)
from nowcasting.facl import build_facl_loss
from nowcasting.models.temporal_discriminator import TemporalDiscriminator
from train.pwv import build_parser as build_shared_parser
from train.radar import (
    append_epoch_log,
    autocast_context,
    discriminator_sequence,
    generator_losses,
    make_grad_scaler,
    safe_torch_save,
    seed_everything,
    weighted_l1,
    write_training_plot,
)


def build_parser():
    parser = build_shared_parser()
    parser.description = "Train two-stream radar/PWV latent-state fusion"
    parser.set_defaults(
        model_name="PWVLatentFusionNowcastNet",
        pwv_control="real",
        lambda_forecast=1.0,
        lambda_evolution=0.5,
        lambda_advected=0.25,
        lambda_motion=0.02,
        lambda_pool=0.2,
        lambda_adv=0.01,
    )
    parser.add_argument("--pwv_latent_channels", type=int, default=8)
    parser.add_argument("--pwv_latent_heads", type=int, default=4)
    parser.add_argument("--pwv_latent_dropout", type=float, default=0.0)
    parser.add_argument("--lambda_pwv_aux", type=float, default=0.1)
    parser.add_argument("--radar_lr_scale", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--matched_discriminator_seed", type=int, default=-1)
    return parser


def controlled_training_batch(pwv, mode, input_length):
    if mode not in ("real", "spatial_shift"):
        raise ValueError(
            "Latent pilot supports only real or train-time spatial_shift PWV."
        )
    if mode == "real":
        return pwv, pwv[:, input_length:]
    shifted = []
    height, width = pwv.shape[-2:]
    for sample in pwv:
        dy = int(
            torch.randint(
                max(height // 4, 1),
                max(3 * height // 4, height // 4 + 1),
                (1,),
                device=pwv.device,
            )
        )
        dx = int(
            torch.randint(
                max(width // 4, 1),
                max(3 * width // 4, width // 4 + 1),
                (1,),
                device=pwv.device,
            )
        )
        shifted.append(torch.roll(sample, shifts=(dy, dx), dims=(-2, -1)))
    controlled = torch.stack(shifted, dim=0)
    return controlled, controlled[:, input_length:]


def pwv_auxiliary_loss(prediction, target, scale):
    scale = max(float(scale), 1e-6)
    return F.smooth_l1_loss(prediction / scale, target / scale)


def train_one_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    scaler_g,
    scaler_d,
    args,
    facl_criterion,
):
    generator.train()
    discriminator.train()
    totals = {}
    seen = 0
    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv_all = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        pwv, pwv_target = controlled_training_batch(
            pwv_all, args.pwv_control, args.input_length
        )
        target = batch["target_frames"].float().to(
            args.device, non_blocking=True
        )

        opt_d.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            with torch.no_grad():
                fake = generator(frames, pwv)[..., 0]
            real_logits = discriminator(
                discriminator_sequence(target, args.intensity_scale)
            )
            fake_logits = discriminator(
                discriminator_sequence(fake.detach(), args.intensity_scale)
            )
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                + F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
            )
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            aux = generator(frames, pwv, return_aux=True)
            base_loss, parts = generator_losses(
                aux,
                target,
                discriminator,
                args,
                facl_criterion=facl_criterion,
                global_step=getattr(args, "global_step", 0),
            )
            aux_loss = pwv_auxiliary_loss(
                aux["pwv_prediction"],
                pwv_target,
                args.pwv_intensity_scale,
            )
            total = base_loss + args.lambda_pwv_aux * aux_loss
        scaler_g.scale(total).backward()
        if args.grad_clip > 0:
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
        scaler_g.step(opt_g)
        scaler_g.update()
        args.global_step = getattr(args, "global_step", 0) + 1
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        batch_size = frames.size(0)
        seen += batch_size
        parts["g_total"] = total.detach()
        parts["pwv_aux"] = aux_loss.detach()
        parts["d_total"] = d_loss.detach()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        if step % args.log_interval == 0:
            print(
                "step {:05d} {}".format(
                    step,
                    " ".join(
                        "{} {:.5f}".format(key, totals[key] / seen)
                        for key in (
                            "g_total",
                            "d_total",
                            "forecast",
                            "pwv_aux",
                        )
                    ),
                ),
                flush=True,
            )
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(generator, loader, args):
    generator.eval()
    rain_total = 0.0
    pwv_total = 0.0
    seen = 0
    for batch_id, batch in enumerate(loader):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv_all = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        torch.manual_seed(args.seed + batch_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + batch_id)
        pwv, pwv_target = controlled_training_batch(
            pwv_all, args.pwv_control, args.input_length
        )
        target = batch["target_frames"].float().to(
            args.device, non_blocking=True
        )
        aux = generator(frames, pwv, return_aux=True)
        rain_loss = weighted_l1(
            aux["prediction"][..., 0], target, args.intensity_scale
        )
        aux_loss = pwv_auxiliary_loss(
            aux["pwv_prediction"], pwv_target, args.pwv_intensity_scale
        )
        batch_size = frames.size(0)
        rain_total += float(rain_loss) * batch_size
        pwv_total += float(aux_loss) * batch_size
        seen += batch_size
    return {
        "rain": rain_total / max(seen, 1),
        "pwv_aux": pwv_total / max(seen, 1),
    }


def main():
    args = add_model_runtime_args(build_parser().parse_args())
    if args.model_name != "PWVLatentFusionNowcastNet":
        raise ValueError("This trainer only supports PWVLatentFusionNowcastNet.")
    if not args.init_radar_checkpoint:
        raise ValueError("--init_radar_checkpoint is required.")
    if args.freeze_radar_backbone:
        raise ValueError(
            "The latent-fusion protocol fine-tunes the radar path at a lower LR; "
            "do not pass --freeze_radar_backbone."
        )
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(args.readme_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_json_args(args, save_dir)

    train_loader = make_png_dataloader(args, "train", args.max_train_samples)
    val_loader = make_png_dataloader(args, "val", args.max_val_samples)
    save_dataset_provenance(
        {"train": train_loader, "val": val_loader},
        save_dir / "data_manifest.json",
    )
    print(
        "train windows: {} val windows: {}".format(
            len(train_loader.dataset), len(val_loader.dataset)
        ),
        flush=True,
    )

    generator = build_generator(args)
    report = load_radar_backbone_weights(
        generator, args.init_radar_checkpoint, args.device
    )
    fusion_parameters = list(generator.fusion_parameters())
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    radar_parameters = [
        parameter
        for parameter in generator.parameters()
        if id(parameter) not in fusion_ids
    ]
    print(
        "initialized radar {} radar_params={} fusion_params={}".format(
            report,
            sum(parameter.numel() for parameter in radar_parameters),
            sum(parameter.numel() for parameter in fusion_parameters),
        ),
        flush=True,
    )
    if args.matched_discriminator_seed >= 0:
        seed_everything(args.matched_discriminator_seed)
    discriminator = TemporalDiscriminator(
        args.gen_oc, base_channels=args.disc_channels
    ).to(args.device)
    if args.matched_discriminator_seed >= 0:
        seed_everything(args.seed + 1)
    opt_g = torch.optim.Adam(
        [
            {
                "params": radar_parameters,
                "lr": args.lr_g * args.radar_lr_scale,
            },
            {"params": fusion_parameters, "lr": args.lr_g},
        ],
        betas=(args.beta1, args.beta2),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2),
    )
    scaler_g = make_grad_scaler(args.device, args.amp)
    scaler_d = make_grad_scaler(args.device, args.amp)
    facl_criterion = build_facl_loss(
        args, max(args.epochs * len(train_loader), 1)
    )
    args.global_step = 0
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            opt_g,
            opt_d,
            scaler_g,
            scaler_d,
            args,
            facl_criterion,
        )
        val_metrics = validate(generator, val_loader, args)
        train_metrics["val_pwv_aux"] = val_metrics["pwv_aux"]
        append_epoch_log(
            save_dir / "train_log.csv",
            epoch,
            val_metrics["rain"],
            train_metrics,
        )
        write_training_plot(
            save_dir / "train_log.csv", save_dir / "train_log.png"
        )
        print(
            "epoch {:03d} val_weighted_l1 {:.5f} val_pwv_aux {:.5f}".format(
                epoch, val_metrics["rain"], val_metrics["pwv_aux"]
            ),
            flush=True,
        )
        save_adversarial_checkpoint(
            save_dir / "latest.ckpt",
            generator,
            discriminator,
            opt_g,
            opt_d,
            epoch,
            val_metrics["rain"],
            args,
        )
        safe_torch_save(
            generator.state_dict(), save_dir / "latest_state_dict.ckpt"
        )
        if val_metrics["rain"] < best_val:
            best_val = val_metrics["rain"]
            stale_epochs = 0
            save_adversarial_checkpoint(
                save_dir / "best.ckpt",
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                best_val,
                args,
            )
            safe_torch_save(
                generator.state_dict(), save_dir / "best_state_dict.ckpt"
            )
            safe_torch_save(generator.state_dict(), args.readme_ckpt)
            (save_dir / "best_validation.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_weighted_l1": best_val,
                        "val_pwv_aux": val_metrics["pwv_aux"],
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(
                    "early stopping after {} stale epochs".format(
                        stale_epochs
                    ),
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
