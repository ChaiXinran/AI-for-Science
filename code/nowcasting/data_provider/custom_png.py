import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None


class PngSequenceDataset(Dataset):
    """Sliding-window dataset for chronological radar PNG folders."""

    def __init__(
        self,
        data_root,
        pwv_root="",
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        input_length=9,
        total_length=29,
        img_height=96,
        img_width=96,
        stride=1,
        max_samples=0,
        intensity_scale=128.0,
        pixel_min=0.0,
        pixel_max=255.0,
        invert=True,
        pwv_intensity_scale=1.0,
        pwv_pixel_min=0.0,
        pwv_pixel_max=255.0,
        pwv_invert=False,
    ):
        self.data_root = Path(data_root)
        self.pwv_root = Path(pwv_root) if pwv_root else None
        self.split = split
        self.input_length = input_length
        self.total_length = total_length
        self.img_height = img_height
        self.img_width = img_width
        self.intensity_scale = float(intensity_scale)
        self.pixel_min = float(pixel_min)
        self.pixel_max = float(pixel_max)
        self.invert = invert
        self.pwv_intensity_scale = float(pwv_intensity_scale)
        self.pwv_pixel_min = float(pwv_pixel_min)
        self.pwv_pixel_max = float(pwv_pixel_max)
        self.pwv_invert = pwv_invert
        if self.pixel_max <= self.pixel_min:
            raise ValueError("pixel_max must be greater than pixel_min.")
        if self.pwv_pixel_max <= self.pwv_pixel_min:
            raise ValueError("pwv_pixel_max must be greater than pwv_pixel_min.")

        if img_height % 32 != 0 or img_width % 32 != 0:
            raise ValueError("img_height and img_width must be multiples of 32 for NowcastNet.")
        if not self.data_root.exists():
            raise FileNotFoundError("data_root does not exist: {}".format(self.data_root))

        windows = self._build_windows(stride)
        if not windows:
            raise ValueError("No {}-frame windows found under {}".format(total_length, self.data_root))

        n = len(windows)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        if split == "train":
            windows = windows[:train_end]
        elif split == "val":
            windows = windows[train_end:val_end]
        elif split == "test":
            windows = windows[val_end:]
        elif split == "all":
            pass
        else:
            raise ValueError("Unknown split: {}".format(split))

        if max_samples and max_samples > 0:
            windows = windows[:max_samples]
        if not windows:
            raise ValueError("Split '{}' is empty. Adjust ratios or max_samples.".format(split))
        self.windows = windows

    def _build_windows(self, stride):
        day_dirs = []
        for root, _, files in os.walk(self.data_root):
            pngs = [f for f in files if f.lower().endswith(".png")]
            if pngs:
                day_dirs.append(Path(root))
        day_dirs.sort()

        windows = []
        for day_dir in day_dirs:
            files = sorted(day_dir.glob("*.png"))
            for start in range(0, len(files) - self.total_length + 1, stride):
                windows.append(files[start:start + self.total_length])
        return windows

    def _fit_canvas(self, img, fill):
        h, w = img.shape[:2]
        if h > self.img_height or w > self.img_width:
            if cv2 is not None:
                return cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(img)
            return np.array(pil.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR))

        canvas = np.full((self.img_height, self.img_width), fill, dtype=img.dtype)
        top = (self.img_height - h) // 2
        left = (self.img_width - w) // 2
        canvas[top:top + h, left:left + w] = img
        return canvas

    def _read_frame(self, path, intensity_scale, pixel_min, pixel_max, invert):
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not read image: {}".format(path))
        else:
            img = np.array(Image.open(path).convert("L"))
        fill = pixel_max if invert else pixel_min
        img = self._fit_canvas(img, fill).astype("float32")
        img = np.clip(img, pixel_min, pixel_max)
        denom = max(pixel_max - pixel_min, 1e-6)
        if invert:
            img = pixel_max - img
        else:
            img = img - pixel_min
        img = np.clip(img / denom * intensity_scale, 0.0, intensity_scale)
        return img

    def _read_radar_frame(self, path):
        return self._read_frame(path, self.intensity_scale, self.pixel_min, self.pixel_max, self.invert)

    def _read_pwv_frame(self, radar_path):
        rel_path = radar_path.relative_to(self.data_root)
        pwv_path = self.pwv_root / rel_path
        if not pwv_path.exists():
            return np.zeros((self.img_height, self.img_width), dtype="float32")
        return self._read_frame(
            pwv_path,
            self.pwv_intensity_scale,
            self.pwv_pixel_min,
            self.pwv_pixel_max,
            self.pwv_invert,
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        frames = [self._read_radar_frame(path) for path in self.windows[index]]
        data = np.stack(frames, axis=0)
        mask = np.ones_like(data, dtype="float32")
        vid = np.stack([data, mask], axis=-1).astype("float32")
        target = data[self.input_length:self.total_length].astype("float32")
        sample = {
            "radar_frames": torch.from_numpy(vid),
            "target_frames": torch.from_numpy(target),
            "case_name": str(self.windows[index][0].parent.name),
            "start_file": self.windows[index][0].name,
        }
        if self.pwv_root is not None:
            pwv_frames = [self._read_pwv_frame(path) for path in self.windows[index]]
            sample["pwv_frames"] = torch.from_numpy(np.stack(pwv_frames, axis=0).astype("float32"))
        return sample
