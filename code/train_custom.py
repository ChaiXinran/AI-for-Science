import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nowcasting.experiments.common import (
    add_model_runtime_args,
    build_generator,
    load_generator_weights,
    make_png_dataloader,
    safe_torch_save,
    save_json_args,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train NowcastNet on custom PNG radar sequences")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--save_dir", type=str, default="../checkpoints/custom_nowcastnet")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="NowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--intensity_scale", type=float, default=128.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--loss", choices=["l1", "mse", "huber"], default="l1")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init_generator", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser


def make_dataloader(args, split, max_samples):
    return make_png_dataloader(args, split, max_samples)


def compute_loss(pred, target, name):
    pred = pred[..., 0]
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "huber":
        return F.smooth_l1_loss(pred, target)
    return F.l1_loss(pred, target)


def run_epoch(model, loader, optimizer, args, train):
    model.train(train)
    total_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(frames)
            loss = compute_loss(pred, target, args.loss)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size

        if train and step % args.log_interval == 0:
            print("step {:05d} train_loss {:.6f}".format(step, total_loss / max(seen, 1)), flush=True)

    return total_loss / max(seen, 1)


def save_checkpoint(path, model, optimizer, epoch, val_loss, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "args": vars(args),
    }
    safe_torch_save(payload, path)


def main():
    args = add_model_runtime_args(build_parser().parse_args())

    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_json_args(args, save_dir)

    print("Building datasets...", flush=True)
    train_loader = make_dataloader(args, "train", args.max_train_samples)
    val_loader = make_dataloader(args, "val", args.max_val_samples)
    print("train windows: {} val windows: {}".format(len(train_loader.dataset), len(val_loader.dataset)), flush=True)

    print("Initializing {} on {}".format(args.model_name, args.device), flush=True)
    model = build_generator(args)
    if args.init_generator:
        load_generator_weights(model, args.init_generator, args.device)
        print("initialized generator from {}".format(args.init_generator), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("val_loss", best_val))

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, args, train=False)
        print(
            "epoch {:03d} train_loss {:.6f} val_loss {:.6f}".format(epoch, train_loss, val_loss),
            flush=True,
        )

        save_checkpoint(save_dir / "latest.ckpt", model, optimizer, epoch, val_loss, args)
        safe_torch_save(model.state_dict(), save_dir / "latest_state_dict.ckpt")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(save_dir / "best.ckpt", model, optimizer, epoch, val_loss, args)
            safe_torch_save(model.state_dict(), save_dir / "best_state_dict.ckpt")


if __name__ == "__main__":
    main()
