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
from nowcasting.cli.custom.test import (
    build_intensity_bins,
    compute_target_quantile_thresholds,
    finalize_event_metrics,
    finalize_fss_metrics,
    finalize_horizon_metrics,
    finalize_intensity_bin_metrics,
    finalize_labeled_event_metrics,
    finalize_lead_metrics,
    finalize_scalar_totals,
    init_extreme_cases,
    init_event_counts,
    init_fss_totals,
    init_horizon_totals,
    init_intensity_bin_totals,
    init_labeled_event_counts,
    init_lead_totals,
    init_scalar_totals,
    parse_float_list,
    parse_horizon_bins,
    parse_int_list,
    parse_thresholds,
    quantile_label,
    save_extreme_cases,
    save_sequence,
    threshold_items_from_quantiles,
    update_event_counts,
    update_extreme_cases,
    update_fss_totals,
    update_intensity_bin_totals,
    update_labeled_event_counts,
    update_lead_and_horizon,
    update_scalar_totals,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Test PWV-coupled NowcastNet")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--pwv_root", type=str, default="../data/DATA_2025_S/PWV_2025_S")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../results/pwv_coupled")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="PWVCoupledNowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--lead_time_embed_dim", type=int, default=16)
    parser.add_argument("--evo_base_channels", type=int, default=32)
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
    parser.add_argument("--extreme_quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument("--extreme_rain_min", type=float, default=0.1)
    parser.add_argument("--quantile_bins", type=int, default=4096)
    parser.add_argument("--intensity_bin_quantiles", type=str, default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--fss_quantiles", type=str, default="0.95,0.99")
    parser.add_argument("--fss_neighborhood_sizes", type=str, default="1,3,5,9,15")
    parser.add_argument("--num_extreme_cases", type=int, default=5)
    parser.add_argument("--grid_km", type=float, default=1.0)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3,3-6")
    return parser


load_state = load_model_state


def main():
    args = add_model_args(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_png_dataloader(args, args.split, args.max_samples, shuffle=False, drop_last=False)
    extreme_quantiles = parse_float_list(args.extreme_quantiles)
    intensity_bin_quantiles = parse_float_list(args.intensity_bin_quantiles)
    quantile_info = compute_target_quantile_thresholds(
        loader,
        sorted(set(extreme_quantiles + intensity_bin_quantiles)),
        args.extreme_rain_min,
        args.intensity_scale,
        args.quantile_bins,
    )
    extreme_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in extreme_quantiles]
    )
    fss_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in parse_float_list(args.fss_quantiles)]
    )
    intensity_bins = build_intensity_bins(quantile_info, args.extreme_rain_min)
    fss_neighborhood_sizes = parse_int_list(args.fss_neighborhood_sizes)

    model = build_generator(args)
    model.load_state_dict(load_state(args.checkpoint, args.device))
    model.eval()

    model_totals = init_scalar_totals()
    persistence_totals = init_scalar_totals()
    thresholds = parse_thresholds(args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_extreme_event_counts = init_labeled_event_counts(extreme_items)
    persistence_extreme_event_counts = init_labeled_event_counts(extreme_items)
    model_lead_totals = init_lead_totals(args.gen_oc)
    persistence_lead_totals = init_lead_totals(args.gen_oc)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)
    model_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    persistence_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    model_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    persistence_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    extreme_case_threshold = quantile_info.get("thresholds", {}).get("P99", 0.0)
    extreme_cases = init_extreme_cases(args.num_extreme_cases)
    coupling_sum = 0.0
    coupling_count = 0
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
            update_labeled_event_counts(model_extreme_event_counts, pred, target, extreme_items)
            update_labeled_event_counts(persistence_extreme_event_counts, persistence, target, extreme_items)
            update_intensity_bin_totals(model_intensity_bin_totals, pred, target, intensity_bins)
            update_intensity_bin_totals(persistence_intensity_bin_totals, persistence, target, intensity_bins)
            update_fss_totals(model_fss_totals, pred, target, fss_items, fss_neighborhood_sizes)
            update_fss_totals(persistence_fss_totals, persistence, target, fss_items, fss_neighborhood_sizes)
            coupling_sum += coupling.sum().item()
            coupling_count += coupling.numel()

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            input_np = frames.detach().cpu().numpy()[..., 0]
            pwv_np = pwv.detach().cpu().numpy()
            persistence_np = persistence.detach().cpu().numpy()
            coupling_np = coupling.detach().cpu().numpy()
            update_extreme_cases(
                extreme_cases,
                args.num_extreme_cases,
                batch,
                pred,
                target,
                persistence,
                {
                    "input": input_np[:, :args.input_length],
                    "pwv": pwv_np[:, :args.input_length],
                    "coupling": coupling_np,
                },
                extreme_case_threshold,
            )

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
                saved += 1

            print("tested batch {}".format(batch_id + 1), flush=True)

    save_extreme_cases(output_dir, extreme_cases, quantile_info.get("thresholds", {}), args, not args.no_invert)
    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(loader.dataset),
        "saved_samples": saved,
        "coupling_mean": coupling_sum / max(coupling_count, 1),
        "units": {
            "prediction": "mm/h",
            "thresholds": "mm/h",
            "pwv": "mm",
            "pixel_mapping": "255->0, 0->scale when invert is true",
            "intensity_scale": args.intensity_scale,
            "pwv_intensity_scale": args.pwv_intensity_scale,
        },
        "thresholds": thresholds,
        "extreme_thresholds": quantile_info,
        "fss_neighborhood_sizes": fss_neighborhood_sizes,
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
        "extreme_event_metrics": {
            "model": finalize_labeled_event_metrics(model_extreme_event_counts),
            "persistence": finalize_labeled_event_metrics(persistence_extreme_event_counts),
        },
        "intensity_bin_metrics": {
            "model": finalize_intensity_bin_metrics(model_intensity_bin_totals),
            "persistence": finalize_intensity_bin_metrics(persistence_intensity_bin_totals),
        },
        "fss": {
            "model": finalize_fss_metrics(model_fss_totals, fss_items, args.grid_km),
            "persistence": finalize_fss_metrics(persistence_fss_totals, fss_items, args.grid_km),
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
