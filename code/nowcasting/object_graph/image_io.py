from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def read_grayscale_frame(
    path,
    img_height,
    img_width,
    intensity_scale,
    pixel_min=0.0,
    pixel_max=255.0,
    invert=True,
):
    """Read one calibrated grayscale PNG into a float array."""
    path = Path(path)
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image: {}".format(path))
    else:
        img = np.array(Image.open(path).convert("L"))

    img = fit_canvas(img, img_height, img_width, pixel_max if invert else pixel_min)
    img = img.astype("float32")
    img = np.clip(img, pixel_min, pixel_max)
    denom = max(float(pixel_max) - float(pixel_min), 1e-6)
    if invert:
        img = pixel_max - img
    else:
        img = img - pixel_min
    return np.clip(img / denom * float(intensity_scale), 0.0, float(intensity_scale))


def fit_canvas(img, img_height, img_width, fill):
    h, w = img.shape[:2]
    if h > img_height or w > img_width:
        if cv2 is not None:
            return cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(img)
        return np.array(pil.resize((img_width, img_height), Image.Resampling.BILINEAR))

    canvas = np.full((img_height, img_width), fill, dtype=img.dtype)
    top = (img_height - h) // 2
    left = (img_width - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def find_event_dirs(data_root):
    """Return sorted leaf directories that contain PNG frames."""
    data_root = Path(data_root)
    event_dirs = []
    for root, _, files in _walk(data_root):
        pngs = [name for name in files if name.lower().endswith(".png")]
        if pngs:
            event_dirs.append(Path(root))
    event_dirs.sort()
    return event_dirs


def _walk(path):
    import os

    for root, dirs, files in os.walk(path):
        yield Path(root), dirs, files


def relative_event_id(event_dir, data_root):
    rel = Path(event_dir).relative_to(Path(data_root))
    return "/".join(rel.parts)
