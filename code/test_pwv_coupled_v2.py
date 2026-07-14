import argparse
import json
from pathlib import Path

import torch

from nowcasting.experiments.common import (
    add_model_runtime_args as add_model_args,
    build_generator,
    load_model_state,
    make_png_dataloader,
)
from test_custom import (
    average_neighborhood_score,
    finalize_event_metrics,
    finalize_horizon_metrics,
    finalize_lead_metrics,
    finalize_neighborhood_metrics,
    finalize_psd_metrics,
    finalize_scalar_totals,
    init_event_counts,
    init_horizon_totals,
    init_lead_totals,
    init_psd_totals,
    init_scalar_totals,
    parse_float_list,
    parse_horizon_bins,
    parse_thresholds,
    save_sequence,
    update_event_counts,
    update_lead_and_horizon,
    update_neighborhood_event_counts,
    update_psd_totals,
    update_scalar_totals,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Test PWV-coupled NowcastNet V2")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--pwv_root", type=str, default="../data/DATA_2025_S/PWV_2025_S")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../results/pwv_coupled_v2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="PWVCoupledNowcastNetV2")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--evo_base_channels", type=int, default=32)
    parser.add_argument("--pwv_base_channels", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--num_save_samples", type=int, default=10)
    parser.add_argument("--intensity_scale", type=float, default=128.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=1.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--metric_thresholds", type=str, default="1,5,10,20,40")
    parser.add_argument("--neighborhood_metric_thresholds", type=str, default="")
    parser.add_argument("--neighborhood_size", type=int, default=5)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3,3-6")
    parser.add_argument("--psd_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--psd_wavelengths", type=str, default="4,8,16,32,64")
    parser.add_argument("--grid_km", type=float, default=1.0)
    return parser


load_state = load_model_state


def main():
    args = add_model_args(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_png_dataloader(args, args.split, args.max_samples, shuffle=False, drop_last=False)

    model = build_generator(args)
    model.load_state_dict(load_state(args.checkpoint, args.device))
    model.eval()

    model_totals = init_scalar_totals()
    persistence_totals = init_scalar_totals()
    thresholds = parse_thresholds(args.metric_thresholds)
    neighborhood_thresholds = parse_thresholds(args.neighborhood_metric_thresholds or args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    psd_lead_minutes = parse_float_list(args.psd_lead_minutes)
    psd_wavelengths = parse_float_list(args.psd_wavelengths)
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    persistence_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    model_lead_totals = init_lead_totals(args.gen_oc)
    persistence_lead_totals = init_lead_totals(args.gen_oc)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)
    psd_totals = init_psd_totals(psd_lead_minutes, psd_wavelengths)
    coupling_sum = 0.0
    coupling_sq_sum = 0.0
    coupling_count = 0
    support_sum = 0.0
    support_sq_sum = 0.0
    support_count = 0
    saved = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
            pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
            target = batch["target_frames"].float().to(args.device, non_blocking=True)
            aux = model(frames, pwv, return_aux=True)
            pred = aux["prediction"][..., 0]
            coupling = aux["coupling"][:, :, 0]
            last_input = frames[:, args.input_length - 1, :, :, 0]
            persistence = last_input.unsqueeze(1).repeat(1, args.gen_oc, 1, 1)

            update_scalar_totals(model_totals, pred, target)
            update_scalar_totals(persistence_totals, persistence, target)
            update_lead_and_horizon(model_lead_totals, model_horizon_totals, pred, target, args.frame_minutes, horizon_bins)
            update_lead_and_horizon(persistence_lead_totals, persistence_horizon_totals, persistence, target, args.frame_minutes, horizon_bins)
            update_event_counts(model_event_counts, pred, target, thresholds)
            update_event_counts(persistence_event_counts, persistence, target, thresholds)
            update_neighborhood_event_counts(model_neighborhood_counts, pred, target, neighborhood_thresholds, args.neighborhood_size)
            update_neighborhood_event_counts(persistence_neighborhood_counts, persistence, target, neighborhood_thresholds, args.neighborhood_size)
            update_psd_totals(psd_totals, pred, target, persistence, args.frame_minutes, psd_lead_minutes, psd_wavelengths, args.grid_km)
            coupling_sum += coupling.sum().item()
            coupling_sq_sum += (coupling * coupling).sum().item()
            coupling_count += coupling.numel()
            if "support_gate" in aux:
                support = aux["support_gate"][:, :, 0]
                support_sum += support.sum().item()
                support_sq_sum += (support * support).sum().item()
                support_count += support.numel()

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            input_np = frames.detach().cpu().numpy()[..., 0]
            pwv_np = pwv.detach().cpu().numpy()
            persistence_np = persistence.detach().cpu().numpy()
            coupling_np = coupling.detach().cpu().numpy()
            support_np = aux["support_gate"][:, :, 0].detach().cpu().numpy() if "support_gate" in aux else None

            for i in range(pred_np.shape[0]):
                if saved >= args.num_save_samples:
                    continue
                sample_dir = output_dir / "sample_{:04d}".format(saved)
                save_sequence(sample_dir, "input_", input_np[i, :args.input_length], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "gt_", target_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "pd_", pred_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "ps_", persistence_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "pwv_", pwv_np[i, :args.input_length], args.pwv_intensity_scale, args.pwv_pixel_min, args.pwv_pixel_max, args.pwv_invert)
                save_sequence(sample_dir, "c_", coupling_np[i], 1.0, 0.0, 255.0, False)
                if support_np is not None:
                    save_sequence(sample_dir, "s_", support_np[i], 1.0, 0.0, 255.0, False)
                saved += 1

            print("tested batch {}".format(batch_id + 1), flush=True)

    coupling_mean = coupling_sum / max(coupling_count, 1)
    coupling_var = coupling_sq_sum / max(coupling_count, 1) - coupling_mean * coupling_mean
    support_mean = support_sum / max(support_count, 1) if support_count else None
    support_var = support_sq_sum / max(support_count, 1) - support_mean * support_mean if support_count else None
    model_neighborhood_metrics = finalize_neighborhood_metrics(model_neighborhood_counts)
    persistence_neighborhood_metrics = finalize_neighborhood_metrics(persistence_neighborhood_counts)
    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(loader.dataset),
        "saved_samples": saved,
        "coupling_mean": coupling_mean,
        "coupling_std": max(coupling_var, 0.0) ** 0.5,
        "support_mean": support_mean,
        "support_std": max(support_var, 0.0) ** 0.5 if support_var is not None else None,
        "units": {
            "prediction": "mm/h",
            "thresholds": "mm/h",
            "pwv": "mm",
            "pixel_mapping": "255->0, 0->scale when invert is true",
            "intensity_scale": args.intensity_scale,
            "pwv_intensity_scale": args.pwv_intensity_scale,
        },
        "thresholds": thresholds,
        "neighborhood_thresholds": neighborhood_thresholds,
        "neighborhood_size": args.neighborhood_size,
        "frame_minutes": args.frame_minutes,
        "lead_time_metrics": {
            "model": finalize_lead_metrics(model_lead_totals, args.frame_minutes),
            "persistence": finalize_lead_metrics(persistence_lead_totals, args.frame_minutes),
        },
        "horizon_metrics": {
            "model": finalize_horizon_metrics(model_horizon_totals),
            "persistence": finalize_horizon_metrics(persistence_horizon_totals),
        },
        "event_metrics": {
            "model": finalize_event_metrics(model_event_counts),
            "persistence": finalize_event_metrics(persistence_event_counts),
        },
        "neighborhood_event_metrics": {
            "model": model_neighborhood_metrics,
            "persistence": persistence_neighborhood_metrics,
        },
        "neighborhood_score": {
            "model": average_neighborhood_score(model_neighborhood_metrics),
            "persistence": average_neighborhood_score(persistence_neighborhood_metrics),
        },
        "psd": finalize_psd_metrics(psd_totals, psd_wavelengths, args.grid_km),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
