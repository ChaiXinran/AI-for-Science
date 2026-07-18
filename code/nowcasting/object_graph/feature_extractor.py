import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def dilate_mask(mask, radius):
    mask = np.asarray(mask, dtype=np.uint8)
    radius = int(max(radius, 0))
    if radius == 0:
        return mask.astype(bool)
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask, kernel, iterations=1).astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            if (dy - radius) ** 2 + (dx - radius) ** 2 <= radius * radius:
                out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def _safe_stats(values):
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.max()), float(values.std())


def add_pwv_features(objects_by_frame, pwv_frames, ring_radius=5, lag_frames=(2, 5, 10)):
    """Attach direct PWV statistics to object dictionaries in place."""
    if pwv_frames is None:
        for objects in objects_by_frame:
            for obj in objects:
                _fill_missing_pwv(obj, lag_frames)
        return

    gradients = [np.gradient(frame.astype("float32")) for frame in pwv_frames]
    for frame_index, objects in enumerate(objects_by_frame):
        pwv = pwv_frames[frame_index]
        grad_y, grad_x = gradients[frame_index]
        for obj in objects:
            mask = obj["mask"]
            ring = np.logical_and(dilate_mask(mask, ring_radius), ~mask)
            inside_mean, inside_max, inside_std = _safe_stats(pwv[mask])
            ring_mean, ring_max, ring_std = _safe_stats(pwv[ring])
            obj["pwv_inside_mean"] = inside_mean
            obj["pwv_inside_max"] = inside_max
            obj["pwv_inside_std"] = inside_std
            obj["pwv_ring_mean"] = ring_mean
            obj["pwv_ring_max"] = ring_max
            obj["pwv_ring_std"] = ring_std
            obj["pwv_inner_outer_diff"] = inside_mean - ring_mean
            obj["pwv_gradient_x"] = float(grad_x[mask].mean()) if mask.any() else 0.0
            obj["pwv_gradient_y"] = float(grad_y[mask].mean()) if mask.any() else 0.0
            for lag in lag_frames:
                key = "pwv_delta_{}".format(int(lag))
                if frame_index >= lag:
                    delta = pwv - pwv_frames[frame_index - lag]
                    obj[key] = float(delta[mask].mean()) if mask.any() else 0.0
                else:
                    obj[key] = 0.0


def _fill_missing_pwv(obj, lag_frames):
    for key in (
        "pwv_inside_mean",
        "pwv_inside_max",
        "pwv_inside_std",
        "pwv_ring_mean",
        "pwv_ring_max",
        "pwv_ring_std",
        "pwv_inner_outer_diff",
        "pwv_gradient_x",
        "pwv_gradient_y",
    ):
        obj[key] = 0.0
    for lag in lag_frames:
        obj["pwv_delta_{}".format(int(lag))] = 0.0


def add_front_pwv_features(objects_by_frame, pwv_frames, distance_pixels=8, radius_pixels=4):
    """Attach velocity-front PWV statistics after tracks have been assigned."""
    if pwv_frames is None:
        for objects in objects_by_frame:
            for obj in objects:
                obj["pwv_front_mean"] = 0.0
                obj["pwv_gradient_parallel"] = 0.0
        return

    yy, xx = np.indices(pwv_frames[0].shape)
    for frame_index, objects in enumerate(objects_by_frame):
        pwv = pwv_frames[frame_index]
        for obj in objects:
            vx = float(obj.get("velocity_x", 0.0))
            vy = float(obj.get("velocity_y", 0.0))
            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1e-6:
                obj["pwv_front_mean"] = 0.0
                obj["pwv_gradient_parallel"] = 0.0
                continue
            ux = vx / speed
            uy = vy / speed
            cx = float(obj["centroid_x"]) + ux * float(distance_pixels)
            cy = float(obj["centroid_y"]) + uy * float(distance_pixels)
            front = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius_pixels) ** 2
            obj["pwv_front_mean"] = float(pwv[front].mean()) if front.any() else 0.0
            obj["pwv_gradient_parallel"] = (
                float(obj.get("pwv_gradient_x", 0.0)) * ux
                + float(obj.get("pwv_gradient_y", 0.0)) * uy
            )

