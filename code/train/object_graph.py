import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.experiments.common import seed_everything


RADAR_FEATURES = [
    "centroid_x",
    "centroid_y",
    "area_pixels",
    "mean_intensity",
    "max_intensity",
    "intensity_sum",
    "perimeter_pixels",
    "aspect_ratio",
    "eccentricity",
    "velocity_x",
    "velocity_y",
    "area_change",
    "intensity_change",
    "object_age",
]

PWV_FEATURES = [
    "pwv_inside_mean",
    "pwv_inside_max",
    "pwv_inside_std",
    "pwv_ring_mean",
    "pwv_ring_max",
    "pwv_ring_std",
    "pwv_inner_outer_diff",
    "pwv_front_mean",
    "pwv_delta_2",
    "pwv_delta_5",
    "pwv_delta_10",
    "pwv_gradient_x",
    "pwv_gradient_y",
    "pwv_gradient_parallel",
]

TASKS = ["label_survive", "label_decay", "label_split", "label_merge"]


class ObjectMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, dropout=0.1):
        super(ObjectMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ObjectLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ObjectLinear, self).__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def build_parser():
    parser = argparse.ArgumentParser(description="Train Pilot-1 object lifecycle table baseline.")
    parser.add_argument("--objects_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../outputs/object_graph_baseline")
    parser.add_argument("--feature_set", choices=("radar", "radar_pwv"), default="radar")
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def as_float(row, key):
    value = row.get(key, "")
    if value in ("", "nan", "None"):
        return 0.0
    return float(value)


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if int(float(row.get("label_valid_next", 0))) != 1:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No valid lifecycle rows found in {}".format(path))
    return rows


def select_features(feature_set):
    if feature_set == "radar":
        return list(RADAR_FEATURES)
    return list(RADAR_FEATURES) + list(PWV_FEATURES)


def rows_to_arrays(rows, feature_names):
    x = np.asarray([[as_float(row, key) for key in feature_names] for row in rows], dtype="float32")
    y = np.asarray([[as_float(row, key) for key in TASKS] for row in rows], dtype="float32")
    splits = np.asarray([row.get("split", "train") for row in rows])
    return x, y, splits


def normalize(train_x, *arrays):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = [(array - mean) / std for array in arrays]
    return mean, std, normalized


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x.astype("float32")), torch.from_numpy(y.astype("float32")))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    positives = float(y_true.sum())
    if positives <= 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-6)
    recall_step = y_sorted / positives
    return float((precision * recall_step).sum())


def binary_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(np.float32)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": average_precision(y_true, y_score),
        "positives": int(y_true.sum()),
        "support": int(len(y_true)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_prob = []
    all_target = []
    total_loss = 0.0
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        prob = torch.sigmoid(logits)
        all_prob.append(prob.cpu().numpy())
        all_target.append(y.cpu().numpy())
        total_loss += float(loss.item()) * x.size(0)
        seen += x.size(0)
    y_score = np.concatenate(all_prob, axis=0)
    y_true = np.concatenate(all_target, axis=0)
    metrics = {"loss": total_loss / max(seen, 1), "tasks": {}}
    f1_values = []
    ap_values = []
    for index, task in enumerate(TASKS):
        task_metrics = binary_metrics(y_true[:, index], y_score[:, index])
        metrics["tasks"][task] = task_metrics
        f1_values.append(task_metrics["f1"])
        if task_metrics["ap"] is not None:
            ap_values.append(task_metrics["ap"])
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0.0
    metrics["macro_ap"] = float(np.mean(ap_values)) if ap_values else 0.0
    return metrics


def train_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total = 0.0
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        seen += x.size(0)
    return total / max(seen, 1)


def build_model(args, in_dim):
    if args.model == "linear":
        return ObjectLinear(in_dim, len(TASKS))
    return ObjectMLP(in_dim, len(TASKS), args.hidden_dim, args.dropout)


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = select_features(args.feature_set)
    rows = load_rows(args.objects_csv)
    x, y, splits = rows_to_arrays(rows, feature_names)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    if not train_mask.any():
        raise ValueError("Train split is empty.")
    if not val_mask.any():
        val_mask = test_mask if test_mask.any() else train_mask

    mean, std, (train_x, val_x, test_x) = normalize(x[train_mask], x[train_mask], x[val_mask], x[test_mask])
    train_y = y[train_mask]
    val_y = y[val_mask]
    test_y = y[test_mask]
    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, args.batch_size, False)
    test_loader = make_loader(test_x, test_y, args.batch_size, False) if len(test_y) else None

    model = build_model(args, train_x.shape[1]).to(device)
    positives = torch.from_numpy(train_y.sum(axis=0)).float().to(device)
    negatives = torch.from_numpy((1.0 - train_y).sum(axis=0)).float().to(device)
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_ap": val_metrics["macro_ap"],
        }
        history.append(row)
        score = val_metrics["macro_ap"] + val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "feature_names": feature_names,
                    "tasks": TASKS,
                    "mean": mean.astype("float32"),
                    "std": std.astype("float32"),
                    "args": vars(args),
                },
                output_dir / "best.ckpt",
            )
            with open(output_dir / "val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2, ensure_ascii=False)
        print(
            "epoch {:03d} train_loss {:.5f} val_macro_f1 {:.4f} val_macro_ap {:.4f}".format(
                epoch, train_loss, val_metrics["macro_f1"], val_metrics["macro_ap"]
            ),
            flush=True,
        )

    with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_ap"])
        writer.writeheader()
        writer.writerows(history)

    final = {"val": evaluate(model, val_loader, device)}
    if test_loader is not None:
        final["test"] = evaluate(model, test_loader, device)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()

