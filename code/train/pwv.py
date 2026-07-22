import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nowcasting.experiments.common import (
    add_model_runtime_args as add_model_args,
    build_generator,
    load_generator_weights,
    load_radar_backbone_weights,
    make_png_dataloader,
    save_adversarial_checkpoint,
    save_dataset_provenance,
    save_json_args,
)
from nowcasting.birth_growth import apply_pwv_control, birth_growth_losses
from nowcasting.facl import (
    add_facl_args,
    build_facl_loss,
    compute_forecast_reconstruction_loss,
)
from nowcasting.models.pwv_features import (
    parse_tendency_windows,
    positive_growth_signal,
    pwv_tendency_maps,
)
from nowcasting.models.temporal_discriminator import TemporalDiscriminator
from nowcasting.object_targets import compute_object_loss
from train.radar import (
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


# ---------------------------------------------------------------------------
# Shared utilities (formerly in train_pwv_coupled_v2.py)
# ---------------------------------------------------------------------------

def _make_dataloader(args, split, max_samples):
    return make_png_dataloader(args, split, max_samples)


def coupling_smoothness(coupling):
    dx = coupling[..., :, 1:] - coupling[..., :, :-1]
    dy = coupling[..., 1:, :] - coupling[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def _normalize_positive(x):
    x = F.relu(x)
    denom = x.amax(dim=tuple(range(1, x.dim())), keepdim=True).clamp_min(1e-6)
    return x / denom


def pwv_physical_signal(pwv, input_length, args=None):
    history = pwv[:, :input_length]
    last = history[:, -1]
    first = history[:, 0]
    if args is None:
        delta = F.relu(last - first)
    else:
        delta = positive_growth_signal(history, args)
    dx = F.pad(last[..., 1:] - last[..., :-1], (0, 1, 0, 0))
    dy = F.pad(last[..., 1:, :] - last[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx * dx + dy * dy + 1e-6)
    return _normalize_positive(delta + gradient)


def coupling_alignment_loss(coupling, target, frames, pwv, args):
    last_radar = frames[:, args.input_length - 1, :, :, 0]
    rain_growth = _normalize_positive(target - last_radar.unsqueeze(1))
    pwv_signal_ = pwv_physical_signal(pwv, args.input_length, args).unsqueeze(1)
    align_weight = rain_growth * pwv_signal_
    return ((1.0 - coupling[:, :, 0]) * align_weight).sum() / align_weight.sum().clamp_min(1e-6)


def _make_shuffled_pwv(pwv):
    if pwv.size(0) > 1:
        return pwv[torch.randperm(pwv.size(0), device=pwv.device)]
    return torch.flip(pwv, dims=[1])


def shuffle_contrast_loss(generator, frames, pwv, target, aux, args):
    if args.lambda_shuffle <= 0:
        return target.new_tensor(0.0)
    shuffled_pwv = _make_shuffled_pwv(pwv)
    with torch.no_grad():
        shuffled_aux = generator(frames, shuffled_pwv, return_aux=True)
    real_evo = aux["evolution"] * args.intensity_scale
    shuffled_evo = shuffled_aux["evolution"] * args.intensity_scale
    real_loss = weighted_l1(real_evo, target, args.intensity_scale)
    shuffled_loss = weighted_l1(shuffled_evo, target, args.intensity_scale)
    return F.relu(real_loss - shuffled_loss + args.shuffle_margin)


def build_parser():
    parser = argparse.ArgumentParser(description="Train PWV-coupled NowcastNet V3 with false-alarm control")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--pwv_root", type=str, default="../data/DATA_2025_S/PWV_2025_S")
    parser.add_argument("--save_dir", type=str, default="../checkpoints/pwv")
    parser.add_argument("--readme_ckpt", type=str, default="../checkpoints/pwv_model.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="PWVCoupledNowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--lead_time_embed_dim", type=int, default=16)
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
    parser.add_argument("--split_manifest", type=str, default="")
    parser.add_argument("--require_contiguous", action="store_true")
    parser.add_argument("--strict_pwv", action="store_true")
    parser.add_argument("--pwv_control", choices=["real", "zero", "temporal_reverse"], default="real")
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
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--pwv_tendency_windows", type=str, default="")
    parser.add_argument("--pwv_tendency_mode", choices=["diff", "slope", "both"], default="slope")
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
    parser.add_argument("--init_radar_checkpoint", type=str, default="")
    parser.add_argument("--freeze_radar_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_interval", type=int, default=20)
    add_facl_args(parser)
    # V3-specific args
    parser.add_argument("--fusion_channels", type=int, default=32)
    parser.add_argument("--pwv_source_type", choices=["cnn", "attention"], default="cnn",
                        help="PWV source generator: cnn=LightweightUNet, attention=TemporalPWVCrossAttentionSource")
    parser.add_argument("--pwv_attn_dim", type=int, default=64)
    parser.add_argument("--pwv_attn_heads", type=int, default=4)
    parser.add_argument("--pwv_attn_downsample", type=int, default=4)
    parser.add_argument("--pwv_attn_source_scale", type=float, default=0.0)
    parser.add_argument("--lambda_false_alarm", type=float, default=0.25)
    parser.add_argument("--lambda_support_dry", type=float, default=0.05)
    parser.add_argument("--lambda_support_l1", type=float, default=0.01)
    parser.add_argument("--false_alarm_threshold", type=float, default=2.0)
    parser.add_argument("--object_head_base_channels", type=int, default=24)
    parser.add_argument("--object_loss_threshold", type=float, default=16.0)
    parser.add_argument("--object_loss_min_area", type=int, default=4)
    parser.add_argument("--object_center_sigma", type=float, default=2.0)
    parser.add_argument("--object_center_pos_weight", type=float, default=20.0)
    parser.add_argument("--object_mask_pos_weight", type=float, default=3.0)
    parser.add_argument("--lambda_object_center", type=float, default=0.0)
    parser.add_argument("--lambda_object_mask", type=float, default=0.0)
    parser.add_argument("--lambda_object_area", type=float, default=0.0)
    parser.add_argument("--lambda_object_intensity", type=float, default=0.0)
    parser.add_argument("--lambda_object_dice", type=float, default=0.0)
    parser.add_argument("--lambda_object_count", type=float, default=0.0)
    parser.add_argument("--lambda_object_centroid", type=float, default=0.0)
    parser.add_argument("--lambda_object_consistency", type=float, default=0.0)
    parser.add_argument("--object_consistency_temperature", type=float, default=2.0)
    parser.add_argument("--birth_low_threshold", type=float, default=2.0)
    parser.add_argument("--birth_high_threshold", type=float, default=10.0)
    parser.add_argument("--growth_delta", type=float, default=5.0)
    parser.add_argument("--birth_focal_alpha", type=float, default=0.75)
    parser.add_argument("--birth_focal_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_birth", type=float, default=0.5)
    parser.add_argument("--lambda_growth", type=float, default=0.5)
    parser.add_argument("--lambda_positive_source", type=float, default=0.5)
    parser.add_argument("--lambda_source_sparse", type=float, default=0.05)
    parser.add_argument("--source_active_weight", type=float, default=4.0)
    return parser


def _use_v3_tendency_signal(args):
    return (
        getattr(args, "model_name", "") in ("PWVCoupledNowcastNet", "PWVBirthGrowthNowcastNet")
        and bool(parse_tendency_windows(getattr(args, "pwv_tendency_windows", "")))
    )


def v3_tendency_physical_signal(pwv, input_length, args):
    history = pwv[:, :input_length]
    last = history[:, -1]
    raw_growth = F.relu(last - history[:, 0])
    tendency_maps = pwv_tendency_maps(
        history,
        getattr(args, "frame_minutes", 6.0),
        getattr(args, "pwv_tendency_windows", ""),
        getattr(args, "pwv_tendency_mode", "slope"),
    )
    if tendency_maps:
        tendency_growth = torch.stack([F.relu(item[:, -1]) for item in tendency_maps], dim=1).mean(dim=1)
        growth = torch.maximum(raw_growth, tendency_growth)
    else:
        growth = raw_growth

    dx = F.pad(last[..., 1:] - last[..., :-1], (0, 1, 0, 0))
    dy = F.pad(last[..., 1:, :] - last[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx * dx + dy * dy + 1e-6)
    positive_state = F.relu(last - last.mean(dim=(1, 2), keepdim=True))
    return _normalize_positive(growth + gradient + 0.5 * positive_state)


def physical_signal_for_model(pwv, input_length, args):
    if _use_v3_tendency_signal(args):
        return v3_tendency_physical_signal(pwv, input_length, args)
    return pwv_physical_signal(pwv, input_length, args)


def v3_coupling_alignment_loss(coupling, target, frames, pwv, args):
    last_radar = frames[:, args.input_length - 1, :, :, 0]
    rain_growth = _normalize_positive(target - last_radar.unsqueeze(1))
    pwv_signal = physical_signal_for_model(pwv, args.input_length, args).unsqueeze(1)
    align_weight = rain_growth * pwv_signal
    return ((1.0 - coupling[:, :, 0]) * align_weight).sum() / align_weight.sum().clamp_min(1e-6)


def dry_unsupported_mask(target, frames, pwv, args):
    last_rain = frames[:, args.input_length - 1, :, :, 0]
    dry = (target < args.false_alarm_threshold).float()
    no_recent_rain = (last_rain < args.false_alarm_threshold).float().unsqueeze(1)
    pwv_signal = physical_signal_for_model(pwv, args.input_length, args).unsqueeze(1)
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


def is_finite_tensor(x):
    return bool(torch.isfinite(x).all().item())


def generator_losses(generator, frames, pwv, aux, target, discriminator, args, facl_criterion=None, global_step=0):
    pred = aux["prediction"][..., 0]
    evo = aux["evolution"] * args.intensity_scale
    advected = aux["advected"]
    coupling = aux["coupling"]
    support_gate = aux["support_gate"]

    forecast_loss, forecast_parts = compute_forecast_reconstruction_loss(
        pred,
        target,
        args,
        weighted_l1,
        facl_criterion=facl_criterion,
        global_step=global_step,
    )
    evolution_loss = weighted_l1(evo, target, args.intensity_scale)
    advected_loss = weighted_l1(advected, target, args.intensity_scale)
    motion_loss = motion_regularization(aux["motion"], target, args.intensity_scale)
    pool_loss = pooled_l1(pred, target)
    fake_logits = discriminator(discriminator_sequence(pred, args.intensity_scale))
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    coupling_smooth = coupling_smoothness(coupling)
    coupling_l1 = coupling.mean()
    support_l1 = support_gate.mean()
    birth_growth_mode = getattr(args, "model_name", "") == "PWVBirthGrowthNowcastNet"
    if birth_growth_mode:
        align_loss = target.new_tensor(0.0)
    elif _use_v3_tendency_signal(args):
        align_loss = v3_coupling_alignment_loss(coupling * support_gate, target, frames, pwv, args)
    else:
        align_loss = coupling_alignment_loss(coupling * support_gate, target, frames, pwv, args)
    shuffle_loss = shuffle_contrast_loss(generator, frames, pwv, target, aux, args)
    fa_loss = false_alarm_loss(pred, target, frames, pwv, args)
    dry_support_loss = target.new_tensor(0.0) if birth_growth_mode else support_dry_loss(support_gate, target, frames, pwv, args)
    birth_growth_loss = target.new_tensor(0.0)
    birth_growth_parts = {}
    if birth_growth_mode:
        birth_growth_loss, birth_growth_parts = birth_growth_losses(aux, target, args)
    object_loss = pred.new_tensor(0.0)
    object_parts = {}
    if "object" in aux and (
        args.lambda_object_center
        + args.lambda_object_mask
        + args.lambda_object_area
        + args.lambda_object_intensity
        + args.lambda_object_dice
        + args.lambda_object_count
        + args.lambda_object_centroid
        + args.lambda_object_consistency
    ) > 0:
        object_loss, object_parts = compute_object_loss(aux["object"], target, args, pred_rain=pred)

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
        + birth_growth_loss
        + object_loss
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
    parts.update(object_parts)
    parts.update(birth_growth_parts)
    parts.update({key: value.detach() for key, value in forecast_parts.items()})
    return total, parts


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, scaler_g, scaler_d, args, facl_criterion=None):
    generator.train()
    discriminator.train()
    totals = {}
    seen = 0

    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        pwv = apply_pwv_control(pwv, args.pwv_control)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)

        opt_d.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            with torch.no_grad():
                fake = generator(frames, pwv)[..., 0]
            if not is_finite_tensor(fake):
                print("step {:05d} skipped non-finite generator output before D update".format(step), flush=True)
                opt_d.zero_grad(set_to_none=True)
                opt_g.zero_grad(set_to_none=True)
                continue
            real_logits = discriminator(discriminator_sequence(target, args.intensity_scale))
            fake_logits = discriminator(discriminator_sequence(fake.detach(), args.intensity_scale))
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_loss = 0.5 * (real_loss + fake_loss)
        if not torch.isfinite(d_loss):
            print("step {:05d} skipped non-finite discriminator loss".format(step), flush=True)
            opt_d.zero_grad(set_to_none=True)
            opt_g.zero_grad(set_to_none=True)
            continue
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        for param in discriminator.parameters():
            param.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            aux = generator(frames, pwv, return_aux=True)
            global_step = getattr(args, "global_step", 0)
            g_loss, parts = generator_losses(
                generator,
                frames,
                pwv,
                aux,
                target,
                discriminator,
                args,
                facl_criterion=facl_criterion,
                global_step=global_step,
            )
        if not torch.isfinite(g_loss):
            print("step {:05d} skipped non-finite generator loss".format(step), flush=True)
            opt_g.zero_grad(set_to_none=True)
            for param in discriminator.parameters():
                param.requires_grad_(True)
            continue
        scaler_g.scale(g_loss).backward()
        if args.grad_clip > 0:
            scaler_g.unscale_(opt_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                print("step {:05d} skipped non-finite generator grad norm".format(step), flush=True)
                opt_g.zero_grad(set_to_none=True)
                for param in discriminator.parameters():
                    param.requires_grad_(True)
                continue
        scaler_g.step(opt_g)
        scaler_g.update()
        args.global_step = global_step + 1
        for param in discriminator.parameters():
            param.requires_grad_(True)

        batch_size = frames.size(0)
        seen += batch_size
        parts["d_total"] = d_loss.detach()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size

        if step % args.log_interval == 0:
            msg = "step {:05d}".format(step)
            for key in ("g_total", "d_total", "forecast", "object_total", "false_alarm", "support_mean", "support_dry"):
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
    obj = 0.0
    for batch in loader:
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        pwv = apply_pwv_control(pwv, args.pwv_control)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        aux = generator(frames, pwv, return_aux=True)
        pred = aux["prediction"][..., 0]
        loss = weighted_l1(pred, target, args.intensity_scale)
        total += loss.item() * frames.size(0)
        c_mean += aux["coupling"].mean().item() * frames.size(0)
        s_mean += aux["support_gate"].mean().item() * frames.size(0)
        fa += false_alarm_loss(pred, target, frames, pwv, args).item() * frames.size(0)
        if "object" in aux and (
            args.lambda_object_center
            + args.lambda_object_mask
            + args.lambda_object_area
            + args.lambda_object_intensity
            + args.lambda_object_dice
            + args.lambda_object_count
            + args.lambda_object_centroid
            + args.lambda_object_consistency
        ) > 0:
            object_loss, _ = compute_object_loss(aux["object"], target, args, pred_rain=pred)
            obj += object_loss.item() * frames.size(0)
        seen += frames.size(0)
    return total / max(seen, 1), c_mean / max(seen, 1), s_mean / max(seen, 1), fa / max(seen, 1), obj / max(seen, 1)


def save_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args):
    save_adversarial_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args)


def main():
    args = add_model_args(build_parser().parse_args())
    if args.amp:
        print("PWV V3 currently uses full precision for numerical stability; ignoring --amp.", flush=True)
        args.amp = False
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    Path(args.readme_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_json_args(args, save_dir)

    train_loader = _make_dataloader(args, "train", args.max_train_samples)
    val_loader = _make_dataloader(args, "val", args.max_val_samples)
    save_dataset_provenance({"train": train_loader, "val": val_loader}, save_dir / "data_manifest.json")
    print("train windows: {} val windows: {}".format(len(train_loader.dataset), len(val_loader.dataset)), flush=True)

    generator = build_generator(args)
    if args.init_radar_checkpoint:
        if args.model_name != "PWVBirthGrowthNowcastNet":
            raise ValueError("--init_radar_checkpoint is reserved for PWVBirthGrowthNowcastNet")
        report = load_radar_backbone_weights(generator, args.init_radar_checkpoint, args.device)
        print("initialized fixed radar backbone from {} ({})".format(args.init_radar_checkpoint, report), flush=True)
    if args.freeze_radar_backbone:
        if not hasattr(generator, "freeze_radar_backbone"):
            raise ValueError("Selected model does not support --freeze_radar_backbone")
        generator.freeze_radar_backbone()
        trainable = sum(parameter.numel() for parameter in generator.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in generator.parameters() if not parameter.requires_grad)
        print("radar backbone frozen; trainable_params={} frozen_params={}".format(trainable, frozen), flush=True)
    if args.init_generator:
        load_generator_weights(
            generator,
            args.init_generator,
            args.device,
            strict=not hasattr(generator, "object_head"),
        )
        print("initialized generator from {}".format(args.init_generator), flush=True)
    discriminator = TemporalDiscriminator(args.gen_oc, base_channels=args.disc_channels).to(args.device)
    opt_g = torch.optim.Adam((p for p in generator.parameters() if p.requires_grad), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    scaler_g = make_grad_scaler(args.device, args.amp)
    scaler_d = make_grad_scaler(args.device, args.amp)
    facl_criterion = build_facl_loss(args, max(args.epochs * len(train_loader), 1))

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
    args.global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            opt_g,
            opt_d,
            scaler_g,
            scaler_d,
            args,
            facl_criterion=facl_criterion,
        )
        val_loss, val_c_mean, val_support_mean, val_false_alarm, val_object = validate(generator, val_loader, args)
        metrics["val_c_mean"] = val_c_mean
        metrics["val_support_mean"] = val_support_mean
        metrics["val_false_alarm"] = val_false_alarm
        metrics["val_object"] = val_object
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
