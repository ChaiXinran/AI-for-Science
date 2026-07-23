import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch

from nowcasting.birth_growth import BirthGrowthAccumulator, apply_pwv_control

from nowcasting.experiments.common import (
    add_model_runtime_args as add_model_args,
    build_generator,
    load_model_state,
    make_png_dataloader,
    save_dataset_provenance,
    sanitize_json_numbers,
    seed_everything,
)
from test.radar import (
    average_neighborhood_score,
    build_intensity_bins,
    compute_target_quantile_thresholds,
    event_metrics_for_arrays,
    finalize_event_metrics,
    finalize_fss_metrics,
    finalize_horizon_metrics,
    finalize_horizon_event_metrics,
    finalize_intensity_bin_metrics,
    finalize_labeled_event_metrics,
    finalize_lead_metrics,
    finalize_neighborhood_metrics,
    finalize_pearson_totals,
    finalize_psd_metrics,
    finalize_scalar_totals,
    init_extreme_cases,
    init_cra_store,
    init_event_counts,
    init_eventwise_store,
    init_fss_totals,
    init_horizon_totals,
    init_horizon_event_counts,
    init_intensity_bin_totals,
    init_labeled_event_counts,
    init_lead_totals,
    init_object_store,
    init_pearson_totals,
    init_psd_totals,
    init_scalar_totals,
    parse_float_list,
    parse_horizon_bins,
    parse_int_list,
    parse_thresholds,
    quantile_label,
    save_extreme_cases,
    save_sequence,
    scalar_metrics_for_arrays,
    threshold_items_from_quantiles,
    summarize_cra_store,
    summarize_eventwise_store,
    summarize_object_store,
    update_event_counts,
    update_horizon_event_counts,
    update_cra_store,
    update_eventwise_store,
    update_extreme_cases,
    update_fss_totals,
    update_intensity_bin_totals,
    update_labeled_event_counts,
    update_lead_and_horizon,
    update_neighborhood_event_counts,
    update_object_store,
    update_pearson_totals,
    update_psd_totals,
    update_scalar_totals,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Test a PWV-coupled NowcastNet model")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--pwv_root", type=str, default="../data/DATA_2025_S/PWV_2025_S")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../results/pwv_coupled")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
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
    parser.add_argument("--pwv_base_channels", type=int, default=24)
    parser.add_argument("--fusion_channels", type=int, default=32)
    parser.add_argument("--pwv_source_type", choices=["cnn", "attention"], default="cnn",
                        help="PWV source generator: cnn=LightweightUNet, attention=TemporalPWVCrossAttentionSource")
    parser.add_argument("--pwv_attn_dim", type=int, default=64)
    parser.add_argument("--pwv_attn_heads", type=int, default=4)
    parser.add_argument("--pwv_attn_downsample", type=int, default=4)
    parser.add_argument("--pwv_attn_source_scale", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_manifest", type=str, default="")
    parser.add_argument("--require_contiguous", action="store_true")
    parser.add_argument("--strict_pwv", action="store_true")
    parser.add_argument(
        "--pwv_control",
        choices=["real", "zero", "temporal_reverse", "level_only", "spatial_shift"],
        default="real",
    )
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--max_samples_strategy", choices=["head", "uniform"], default="head")
    parser.add_argument("--num_save_samples", type=int, default=10)
    parser.add_argument("--intensity_scale", type=float, default=128.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=1.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--pwv_tendency_windows", type=str, default="")
    parser.add_argument("--pwv_tendency_mode", choices=["diff", "slope", "both"], default="slope")
    parser.add_argument("--metric_thresholds", type=str, default="1,5,10,20,40")
    parser.add_argument("--neighborhood_metric_thresholds", type=str, default="")
    parser.add_argument("--neighborhood_size", type=int, default=5)
    parser.add_argument("--extreme_quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument("--extreme_rain_min", type=float, default=0.1)
    parser.add_argument("--quantile_bins", type=int, default=4096)
    parser.add_argument("--intensity_bin_quantiles", type=str, default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--fss_quantiles", type=str, default="0.95,0.99")
    parser.add_argument("--fss_neighborhood_sizes", type=str, default="1,3,5,9,15")
    parser.add_argument("--num_extreme_cases", type=int, default=5)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3,3-6")
    parser.add_argument("--psd_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--psd_wavelengths", type=str, default="4,8,16,32,64")
    parser.add_argument("--grid_km", type=float, default=1.0)
    parser.add_argument("--cra_thresholds", type=str, default="16")
    parser.add_argument("--cra_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--cra_max_shift", type=int, default=12)
    parser.add_argument("--object_thresholds", type=str, default="16")
    parser.add_argument("--object_min_area", type=int, default=4)
    parser.add_argument("--object_iou_threshold", type=float, default=0.1)
    parser.add_argument("--birth_low_threshold", type=float, default=2.0)
    parser.add_argument("--birth_high_threshold", type=float, default=10.0)
    parser.add_argument("--growth_delta", type=float, default=5.0)
    parser.add_argument("--birth_probability_threshold", type=float, default=0.5)
    parser.add_argument("--pwv_candidate_threshold", type=float, default=0.5)
    parser.add_argument("--pwv_candidate_radius", type=int, default=2)
    parser.add_argument("--pwv_climatology_path", type=str, default="")
    parser.add_argument("--signed_use_tendency", action="store_true")
    parser.add_argument("--signed_residual_scale", type=float, default=0.25)
    parser.add_argument("--deterministic_noise", action="store_true")
    return parser


load_state = load_model_state


def main():
    args = add_model_args(build_parser().parse_args())
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_png_dataloader(args, args.split, args.max_samples, shuffle=False, drop_last=False)
    save_dataset_provenance({args.split: loader}, output_dir / "data_manifest.json")
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
    neighborhood_thresholds = parse_thresholds(args.neighborhood_metric_thresholds or args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    psd_lead_minutes = parse_float_list(args.psd_lead_minutes)
    psd_wavelengths = parse_float_list(args.psd_wavelengths)
    cra_thresholds = parse_thresholds(args.cra_thresholds)
    cra_lead_minutes = parse_float_list(args.cra_lead_minutes)
    object_thresholds = parse_thresholds(args.object_thresholds)
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_horizon_event_counts = init_horizon_event_counts(thresholds, horizon_bins)
    persistence_horizon_event_counts = init_horizon_event_counts(thresholds, horizon_bins)
    model_extreme_event_counts = init_labeled_event_counts(extreme_items)
    persistence_extreme_event_counts = init_labeled_event_counts(extreme_items)
    model_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    persistence_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    model_lead_totals = init_lead_totals(args.gen_oc)
    persistence_lead_totals = init_lead_totals(args.gen_oc)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)
    model_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    persistence_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    model_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    persistence_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    psd_totals = init_psd_totals(psd_lead_minutes, psd_wavelengths)
    model_pearson_totals = init_pearson_totals(args.gen_oc)
    persistence_pearson_totals = init_pearson_totals(args.gen_oc)
    model_eventwise = init_eventwise_store(thresholds)
    persistence_eventwise = init_eventwise_store(thresholds)
    model_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    persistence_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    model_objects = init_object_store(object_thresholds, args.gen_oc)
    persistence_objects = init_object_store(object_thresholds, args.gen_oc)
    extreme_case_threshold = quantile_info.get("thresholds", {}).get("P99", 0.0)
    extreme_cases = init_extreme_cases(args.num_extreme_cases)
    coupling_sum = 0.0
    coupling_sq_sum = 0.0
    coupling_count = 0
    support_sum = 0.0
    support_sq_sum = 0.0
    support_count = 0
    attention_sum = None
    attention_count = 0
    saved = 0
    birth_growth_metrics = None
    eventwise_records = []
    if args.model_name in ("PWVBirthGrowthNowcastNet", "PWVContrastiveTriggerNowcastNet"):
        birth_growth_metrics = BirthGrowthAccumulator(args.birth_probability_threshold)

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
            pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
            pwv = apply_pwv_control(pwv, args.pwv_control, args.input_length)
            target = batch["target_frames"].float().to(args.device, non_blocking=True)
            if args.deterministic_noise:
                torch.manual_seed(args.seed + batch_id)
                torch.cuda.manual_seed_all(args.seed + batch_id)
            aux = model(frames, pwv, return_aux=True)
            pred = aux["prediction"][..., 0]
            if birth_growth_metrics is not None:
                birth_growth_metrics.update(aux, target, args)
            coupling = aux["coupling"][:, :, 0]
            last_input = frames[:, args.input_length - 1, :, :, 0]
            persistence = last_input.unsqueeze(1).repeat(1, args.gen_oc, 1, 1)

            update_scalar_totals(model_totals, pred, target)
            update_scalar_totals(persistence_totals, persistence, target)
            update_lead_and_horizon(model_lead_totals, model_horizon_totals, pred, target, args.frame_minutes, horizon_bins)
            update_lead_and_horizon(persistence_lead_totals, persistence_horizon_totals, persistence, target, args.frame_minutes, horizon_bins)
            update_event_counts(model_event_counts, pred, target, thresholds)
            update_event_counts(persistence_event_counts, persistence, target, thresholds)
            update_horizon_event_counts(
                model_horizon_event_counts, pred, target, thresholds, args.frame_minutes, horizon_bins
            )
            update_horizon_event_counts(
                persistence_horizon_event_counts, persistence, target, thresholds,
                args.frame_minutes, horizon_bins
            )
            update_labeled_event_counts(model_extreme_event_counts, pred, target, extreme_items)
            update_labeled_event_counts(persistence_extreme_event_counts, persistence, target, extreme_items)
            update_neighborhood_event_counts(model_neighborhood_counts, pred, target, neighborhood_thresholds, args.neighborhood_size)
            update_neighborhood_event_counts(persistence_neighborhood_counts, persistence, target, neighborhood_thresholds, args.neighborhood_size)
            update_intensity_bin_totals(model_intensity_bin_totals, pred, target, intensity_bins)
            update_intensity_bin_totals(persistence_intensity_bin_totals, persistence, target, intensity_bins)
            update_fss_totals(model_fss_totals, pred, target, fss_items, fss_neighborhood_sizes)
            update_fss_totals(persistence_fss_totals, persistence, target, fss_items, fss_neighborhood_sizes)
            update_psd_totals(psd_totals, pred, target, persistence, args.frame_minutes, psd_lead_minutes, psd_wavelengths, args.grid_km)
            update_pearson_totals(model_pearson_totals, pred, target)
            update_pearson_totals(persistence_pearson_totals, persistence, target)
            update_eventwise_store(model_eventwise, pred, target, thresholds)
            update_eventwise_store(persistence_eventwise, persistence, target, thresholds)
            update_cra_store(model_cra, pred, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)
            update_cra_store(persistence_cra, persistence, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)
            update_object_store(model_objects, pred, target, object_thresholds, args.object_min_area, args.object_iou_threshold, args.grid_km)
            update_object_store(persistence_objects, persistence, target, object_thresholds, args.object_min_area, args.object_iou_threshold, args.grid_km)
            for sample_index in range(pred.shape[0]):
                sample_id = batch.get("sample_id", [""] * pred.shape[0])[sample_index]
                case_name = batch.get("case_name", [""] * pred.shape[0])[sample_index]
                start_file = batch.get("start_file", [""] * pred.shape[0])[sample_index]
                eventwise_records.append(
                    {
                        "sample_id": str(sample_id),
                        "case_name": str(case_name),
                        "start_file": str(start_file),
                        "model_scalar": scalar_metrics_for_arrays(
                            pred[sample_index], target[sample_index]
                        ),
                        "model_events": event_metrics_for_arrays(
                            pred[sample_index], target[sample_index], thresholds
                        ),
                    }
                )
            coupling_sum += coupling.sum().item()
            coupling_sq_sum += (coupling * coupling).sum().item()
            coupling_count += coupling.numel()
            if "support_gate" in aux:
                support = aux["support_gate"][:, :, 0]
                support_sum += support.sum().item()
                support_sq_sum += (support * support).sum().item()
                support_count += support.numel()
            if "pwv_temporal_attention" in aux:
                attention = aux["pwv_temporal_attention"]
                item = attention.mean(dim=(0, 2, 3)).detach().cpu()
                if attention_sum is None:
                    attention_sum = torch.zeros_like(item)
                attention_sum += item * attention.size(0)
                attention_count += attention.size(0)

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            input_np = frames.detach().cpu().numpy()[..., 0]
            pwv_np = pwv.detach().cpu().numpy()
            persistence_np = persistence.detach().cpu().numpy()
            coupling_np = coupling.detach().cpu().numpy()
            support_np = aux["support_gate"][:, :, 0].detach().cpu().numpy() if "support_gate" in aux else None
            attention_np = aux["pwv_temporal_attention"].detach().cpu().numpy() if "pwv_temporal_attention" in aux else None
            object_center_np = None
            object_mask_np = None
            birth_probability_np = None
            growth_probability_np = None
            growth_amount_np = None
            radar_evolution_np = None
            if birth_growth_metrics is not None:
                birth_probability_np = aux["birth_probability"][:, :, 0].detach().cpu().numpy()
                growth_probability_np = aux["growth_probability"][:, :, 0].detach().cpu().numpy()
                growth_amount_np = aux["growth_amount"][:, :, 0].detach().cpu().numpy()
                radar_evolution_np = aux["radar_evolution"].detach().cpu().numpy()
            if "object" in aux:
                object_center_np = torch.sigmoid(aux["object"]["center_logits"]).detach().cpu().numpy()
                object_mask_np = torch.sigmoid(aux["object"]["mask_logits"]).detach().cpu().numpy()
            extreme_arrays = {
                "input": input_np[:, :args.input_length],
                "pwv": pwv_np[:, :args.input_length],
                "coupling": coupling_np,
            }
            if support_np is not None:
                extreme_arrays["support"] = support_np
            if attention_np is not None:
                extreme_arrays["attention"] = attention_np
            if object_center_np is not None:
                extreme_arrays["object_center"] = object_center_np
                extreme_arrays["object_mask"] = object_mask_np
            update_extreme_cases(
                extreme_cases,
                args.num_extreme_cases,
                batch,
                pred,
                target,
                persistence,
                extreme_arrays,
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
                if support_np is not None:
                    save_sequence(sample_dir, "s_", support_np[i], 1.0, 0.0, 255.0, False)
                if attention_np is not None:
                    save_sequence(sample_dir, "a_", attention_np[i], 1.0, 0.0, 255.0, False)
                if object_center_np is not None:
                    save_sequence(sample_dir, "oc_", object_center_np[i], 1.0, 0.0, 255.0, False)
                    save_sequence(sample_dir, "om_", object_mask_np[i], 1.0, 0.0, 255.0, False)
                if birth_probability_np is not None:
                    save_sequence(sample_dir, "birth_p_", birth_probability_np[i], 1.0, 0.0, 255.0, False)
                    save_sequence(sample_dir, "growth_p_", growth_probability_np[i], 1.0, 0.0, 255.0, False)
                    save_sequence(sample_dir, "growth_amount_", growth_amount_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, False)
                    save_sequence(sample_dir, "radar_evolution_", radar_evolution_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                saved += 1

            print("tested batch {}".format(batch_id + 1), flush=True)

    coupling_mean = coupling_sum / max(coupling_count, 1)
    coupling_var = coupling_sq_sum / max(coupling_count, 1) - coupling_mean * coupling_mean
    support_mean = support_sum / max(support_count, 1) if support_count else None
    support_var = support_sq_sum / max(support_count, 1) - support_mean * support_mean if support_count else None
    attention_mean = None
    if attention_sum is not None and attention_count:
        attention_mean = (attention_sum / attention_count).tolist()
    model_neighborhood_metrics = finalize_neighborhood_metrics(model_neighborhood_counts)
    persistence_neighborhood_metrics = finalize_neighborhood_metrics(persistence_neighborhood_counts)
    save_extreme_cases(output_dir, extreme_cases, quantile_info.get("thresholds", {}), args, not args.no_invert)
    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(loader.dataset),
        "saved_samples": saved,
        "pwv_control": args.pwv_control,
        "pwv_control_scope": "observed_input_only",
        "coupling_mean": coupling_mean,
        "coupling_std": max(coupling_var, 0.0) ** 0.5,
        "support_mean": support_mean,
        "support_std": max(support_var, 0.0) ** 0.5 if support_var is not None else None,
        "pwv_temporal_attention_mean": attention_mean,
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
        "neighborhood_thresholds": neighborhood_thresholds,
        "neighborhood_size": args.neighborhood_size,
        "fss_neighborhood_sizes": fss_neighborhood_sizes,
        "cra_thresholds": cra_thresholds,
        "cra_lead_minutes": cra_lead_minutes,
        "cra_max_shift": args.cra_max_shift,
        "object_thresholds": object_thresholds,
        "object_min_area": args.object_min_area,
        "object_iou_threshold": args.object_iou_threshold,
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
        "horizon_event_metrics": {
            "model": finalize_horizon_event_metrics(model_horizon_event_counts),
            "persistence": finalize_horizon_event_metrics(persistence_horizon_event_counts),
        },
        "extreme_event_metrics": {
            "model": finalize_labeled_event_metrics(model_extreme_event_counts),
            "persistence": finalize_labeled_event_metrics(persistence_extreme_event_counts),
        },
        "intensity_bin_metrics": {
            "model": finalize_intensity_bin_metrics(model_intensity_bin_totals),
            "persistence": finalize_intensity_bin_metrics(persistence_intensity_bin_totals),
        },
        "neighborhood_event_metrics": {
            "model": model_neighborhood_metrics,
            "persistence": persistence_neighborhood_metrics,
        },
        "fss": {
            "model": finalize_fss_metrics(model_fss_totals, fss_items, args.grid_km),
            "persistence": finalize_fss_metrics(persistence_fss_totals, fss_items, args.grid_km),
        },
        "neighborhood_score": {
            "model": average_neighborhood_score(model_neighborhood_metrics),
            "persistence": average_neighborhood_score(persistence_neighborhood_metrics),
        },
        "psd": finalize_psd_metrics(psd_totals, psd_wavelengths, args.grid_km),
        "pearson": {
            "model": finalize_pearson_totals(model_pearson_totals, args.frame_minutes),
            "persistence": finalize_pearson_totals(persistence_pearson_totals, args.frame_minutes),
        },
        "eventwise": {
            "model": summarize_eventwise_store(model_eventwise),
            "persistence": summarize_eventwise_store(persistence_eventwise),
        },
        "cra": {
            "model": summarize_cra_store(model_cra),
            "persistence": summarize_cra_store(persistence_cra),
        },
        "object_metrics": {
            "model": summarize_object_store(model_objects, args.frame_minutes),
            "persistence": summarize_object_store(persistence_objects, args.frame_minutes),
        },
        "birth_growth": birth_growth_metrics.finalize() if birth_growth_metrics is not None else None,
    }
    metrics = sanitize_json_numbers(metrics)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
    with open(output_dir / "eventwise_records.json", "w", encoding="utf-8") as f:
        json.dump(
            sanitize_json_numbers(eventwise_records),
            f,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
