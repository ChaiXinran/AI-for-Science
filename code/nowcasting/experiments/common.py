import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nowcasting.data_provider.custom_png import PngSequenceDataset
from nowcasting.models.registry import build_model


def sanitize_json_numbers(value):
    """Replace NaN/Inf recursively so artifacts are strict RFC-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_numbers(item) for item in value]
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_model_runtime_args(args):
    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10
    return args


def save_json_args(args, folder, filename="train_args.json"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / filename, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def _max_samples_for_split(args, split, max_samples):
    if max_samples is not None:
        return max_samples
    if split == "train":
        return getattr(args, "max_train_samples", 0)
    if split == "val":
        return getattr(args, "max_val_samples", 0)
    return getattr(args, "max_samples", 0)


def make_png_dataloader(args, split, max_samples=None, shuffle=None, drop_last=None):
    has_pwv = bool(getattr(args, "pwv_root", ""))
    dataset_kwargs = {
        "data_root": args.data_root,
        "split": split,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "input_length": args.input_length,
        "total_length": args.total_length,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "stride": args.stride,
        "max_samples": _max_samples_for_split(args, split, max_samples),
        "intensity_scale": args.intensity_scale,
        "pixel_min": args.pixel_min,
        "pixel_max": args.pixel_max,
        "invert": not getattr(args, "no_invert", False),
        "split_manifest": getattr(args, "split_manifest", ""),
        "frame_minutes": getattr(args, "frame_minutes", 6.0),
        "require_contiguous": getattr(args, "require_contiguous", False),
        "max_samples_strategy": getattr(args, "max_samples_strategy", "head"),
    }
    if has_pwv:
        dataset_kwargs.update(
            {
                "pwv_root": args.pwv_root,
                "pwv_intensity_scale": getattr(args, "pwv_intensity_scale", 1.0),
                "pwv_pixel_min": getattr(args, "pwv_pixel_min", 0.0),
                "pwv_pixel_max": getattr(args, "pwv_pixel_max", 255.0),
                "pwv_invert": getattr(args, "pwv_invert", False),
                "strict_pwv": getattr(args, "strict_pwv", False),
            }
        )
    dataset = PngSequenceDataset(**dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )


def save_dataset_provenance(loaders, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, loader in loaders.items():
        payload[name] = loader.dataset.provenance()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_generator(args):
    return build_model(args).to(args.device)


def load_model_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_generator_weights(model, path, device, strict=True):
    state = load_model_state(path, device)
    model.load_state_dict(state, strict=strict)
    return model


def load_radar_backbone_weights(model, path, device):
    source = load_model_state(path, device)
    target = model.state_dict()
    mapped = {}
    for key, value in source.items():
        target_key = "radar_evo_net." + key[len("evo_net."):] if key.startswith("evo_net.") else key
        if target_key in target and tuple(target[target_key].shape) == tuple(value.shape):
            mapped[target_key] = value
    required_prefixes = ("radar_evo_net.", "gen_enc.", "gen_dec.", "proj.")
    missing_required = [
        key for key in target
        if key.startswith(required_prefixes) and key not in mapped
    ]
    if missing_required:
        raise ValueError(
            "Radar checkpoint is incompatible; {} required backbone tensors missing (first: {}).".format(
                len(missing_required), missing_required[0]
            )
        )
    model.load_state_dict(mapped, strict=False)
    return {"loaded_tensors": len(mapped), "missing_required": 0}


def safe_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_adversarial_checkpoint(path, generator, discriminator, opt_g, opt_d, epoch, val_loss, args):
    safe_torch_save(
        {
            "model": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        path,
    )
