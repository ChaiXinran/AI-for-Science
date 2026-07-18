import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def connected_components(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [
            (labels == label, int(stats[label, cv2.CC_STAT_AREA]))
            for label in range(1, count)
        ]
    return _connected_components_numpy(mask)


def _connected_components_numpy(mask):
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


def mask_perimeter(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    exposed = (
        ~padded[:-2, 1:-1]
        + ~padded[2:, 1:-1]
        + ~padded[1:-1, :-2]
        + ~padded[1:-1, 2:]
    )
    return int(exposed[center].sum())


def object_shape(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) <= 1:
        return 1.0, 0.0
    coords = np.stack([xs.astype("float64"), ys.astype("float64")], axis=1)
    cov = np.cov(coords, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = math.sqrt(max(float(eigvals[0]), 1e-6))
    minor = math.sqrt(max(float(eigvals[-1]), 1e-6))
    aspect = major / max(minor, 1e-6)
    eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-6)))
    return aspect, eccentricity


def extract_frame_objects(field, threshold=16.0, min_area=4, grid_km=1.0):
    """Extract connected precipitation objects from one calibrated frame."""
    objects = []
    for local_id, (comp_mask, area_pixels) in enumerate(connected_components(field >= threshold)):
        if area_pixels < min_area:
            continue
        values = field[comp_mask]
        ys, xs = np.nonzero(comp_mask)
        weight_sum = float(values.sum())
        if weight_sum > 1e-12:
            centroid_x = float((xs * values).sum() / weight_sum)
            centroid_y = float((ys * values).sum() / weight_sum)
        else:
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
        aspect, eccentricity = object_shape(comp_mask)
        objects.append(
            {
                "local_id": local_id,
                "mask": comp_mask,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "area_pixels": int(area_pixels),
                "area_km2": float(area_pixels) * float(grid_km) * float(grid_km),
                "mean_intensity": float(values.mean()),
                "max_intensity": float(values.max()),
                "intensity_sum": float(values.sum()),
                "perimeter_pixels": mask_perimeter(comp_mask),
                "aspect_ratio": aspect,
                "eccentricity": eccentricity,
            }
        )
    return objects

