import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


def _connected_components(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [
            (labels == label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, count)
        ]

    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comp_mask = np.zeros(mask.shape, dtype=bool)
            ys = np.array([item[0] for item in pixels], dtype=np.int64)
            xs = np.array([item[1] for item in pixels], dtype=np.int64)
            comp_mask[ys, xs] = True
            components.append((comp_mask, len(pixels)))
    return components


def _draw_gaussian(target, cx, cy, sigma):
    if sigma <= 0:
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= y < target.shape[0] and 0 <= x < target.shape[1]:
            target[y, x] = 1.0
        return
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    height, width = target.shape
    x0 = max(0, int(math.floor(cx)) - radius)
    x1 = min(width, int(math.floor(cx)) + radius + 1)
    y0 = max(0, int(math.floor(cy)) - radius)
    y1 = min(height, int(math.floor(cy)) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], heat.astype(np.float32))


def build_object_targets(target, threshold, min_area, center_sigma, intensity_scale):
    target_np = target.detach().float().cpu().numpy()
    batch, leads, height, width = target_np.shape
    center = np.zeros((batch, leads, height, width), dtype=np.float32)
    mask = np.zeros_like(center)
    area = np.zeros_like(center)
    mean_intensity = np.zeros_like(center)
    max_intensity = np.zeros_like(center)
    count = np.zeros((batch, leads), dtype=np.float32)
    area_den = float(max(height * width, 1))
    intensity_den = float(max(intensity_scale, 1e-6))

    for b in range(batch):
        for t in range(leads):
            field = target_np[b, t]
            for comp_mask, comp_area in _connected_components(field >= threshold):
                if comp_area < min_area:
                    continue
                count[b, t] += 1.0
                values = field[comp_mask]
                ys, xs = np.nonzero(comp_mask)
                weight_sum = float(values.sum())
                if weight_sum > 1e-12:
                    cx = float((xs * values).sum() / weight_sum)
                    cy = float((ys * values).sum() / weight_sum)
                else:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                _draw_gaussian(center[b, t], cx, cy, center_sigma)
                mask[b, t, comp_mask] = 1.0
                area[b, t, comp_mask] = min(comp_area / area_den, 1.0)
                mean_intensity[b, t, comp_mask] = min(float(values.mean()) / intensity_den, 1.0)
                max_intensity[b, t, comp_mask] = min(float(values.max()) / intensity_den, 1.0)

    device = target.device
    return {
        "center": torch.from_numpy(center).to(device=device, dtype=target.dtype),
        "mask": torch.from_numpy(mask).to(device=device, dtype=target.dtype),
        "area": torch.from_numpy(area).to(device=device, dtype=target.dtype),
        "mean_intensity": torch.from_numpy(mean_intensity).to(device=device, dtype=target.dtype),
        "max_intensity": torch.from_numpy(max_intensity).to(device=device, dtype=target.dtype),
        "count": torch.from_numpy(count).to(device=device, dtype=target.dtype),
    }


def weighted_bce_with_logits(logits, target, positive_weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = 1.0 + positive_weight * target
    return (loss * weight).mean()


def masked_l1(pred, target, mask):
    return (pred - target).abs().mul(mask).sum() / mask.sum().clamp_min(1.0)


def soft_dice_loss(pred, target, eps=1e-6):
    dims = tuple(range(2, pred.dim()))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_count_loss(center_prob, target_count, center_sigma):
    gaussian_area = max(2.0 * math.pi * center_sigma * center_sigma, 1.0)
    pred_count = center_prob.sum(dim=(-2, -1)) / gaussian_area
    return F.smooth_l1_loss(pred_count, target_count)


def soft_centroid_loss(center_prob, target_center, target_count, grid_km):
    valid = target_count > 0.0
    if not bool(valid.any()):
        return center_prob.new_tensor(0.0)
    height, width = center_prob.shape[-2:]
    y = torch.linspace(0.0, height - 1.0, height, device=center_prob.device, dtype=center_prob.dtype)
    x = torch.linspace(0.0, width - 1.0, width, device=center_prob.device, dtype=center_prob.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    pred_sum = center_prob.sum(dim=(-2, -1)).clamp_min(1e-6)
    target_sum = target_center.sum(dim=(-2, -1)).clamp_min(1e-6)
    pred_x = (center_prob * xx).sum(dim=(-2, -1)) / pred_sum
    pred_y = (center_prob * yy).sum(dim=(-2, -1)) / pred_sum
    target_x = (target_center * xx).sum(dim=(-2, -1)) / target_sum
    target_y = (target_center * yy).sum(dim=(-2, -1)) / target_sum
    distance = torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2 + 1e-6)
    return (distance[valid] * grid_km).mean()


def rain_object_consistency_loss(object_mask_prob, pred_rain, threshold, temperature):
    rain_event = torch.sigmoid((pred_rain - threshold) / max(float(temperature), 1e-6))
    return soft_dice_loss(object_mask_prob, rain_event)


def compute_object_loss(object_pred, target, args, pred_rain=None):
    targets = build_object_targets(
        target,
        getattr(args, "object_loss_threshold", 16.0),
        getattr(args, "object_loss_min_area", 4),
        getattr(args, "object_center_sigma", 2.0),
        getattr(args, "intensity_scale", 128.0),
    )
    center_loss = weighted_bce_with_logits(
        object_pred["center_logits"],
        targets["center"],
        getattr(args, "object_center_pos_weight", 20.0),
    )
    mask_loss = weighted_bce_with_logits(
        object_pred["mask_logits"],
        targets["mask"],
        getattr(args, "object_mask_pos_weight", 3.0),
    )
    area_loss = masked_l1(object_pred["area"], targets["area"], targets["mask"])
    mean_loss = masked_l1(object_pred["mean_intensity"], targets["mean_intensity"], targets["mask"])
    max_loss = masked_l1(object_pred["max_intensity"], targets["max_intensity"], targets["mask"])
    intensity_loss = 0.5 * (mean_loss + max_loss)
    center_prob = torch.sigmoid(object_pred["center_logits"])
    mask_prob = torch.sigmoid(object_pred["mask_logits"])
    dice_loss = soft_dice_loss(mask_prob, targets["mask"])
    count_loss = soft_count_loss(
        center_prob,
        targets["count"],
        getattr(args, "object_center_sigma", 2.0),
    )
    centroid_loss = soft_centroid_loss(
        center_prob,
        targets["center"],
        targets["count"],
        getattr(args, "grid_km", 1.0),
    )
    consistency_loss = target.new_tensor(0.0)
    if pred_rain is not None:
        consistency_loss = rain_object_consistency_loss(
            mask_prob,
            pred_rain,
            getattr(args, "object_loss_threshold", 16.0),
            getattr(args, "object_consistency_temperature", 2.0),
        )

    total = (
        getattr(args, "lambda_object_center", 0.0) * center_loss
        + getattr(args, "lambda_object_mask", 0.0) * mask_loss
        + getattr(args, "lambda_object_area", 0.0) * area_loss
        + getattr(args, "lambda_object_intensity", 0.0) * intensity_loss
        + getattr(args, "lambda_object_dice", 0.0) * dice_loss
        + getattr(args, "lambda_object_count", 0.0) * count_loss
        + getattr(args, "lambda_object_centroid", 0.0) * centroid_loss
        + getattr(args, "lambda_object_consistency", 0.0) * consistency_loss
    )
    parts = {
        "object_total": total.detach(),
        "object_center": center_loss.detach(),
        "object_mask": mask_loss.detach(),
        "object_area": area_loss.detach(),
        "object_intensity": intensity_loss.detach(),
        "object_dice": dice_loss.detach(),
        "object_count": count_loss.detach(),
        "object_centroid": centroid_loss.detach(),
        "object_consistency": consistency_loss.detach(),
    }
    return total, parts
