import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "horizon_comparison"
OUT.mkdir(parents=True, exist_ok=True)


RESULTS = {
    "3h Radar-only": ROOT / "results" / "quick_3h_radar" / "metrics.json",
    "3h PWV-coupled": ROOT / "results" / "quick_3h_pwv" / "metrics.json",
    "6h Radar-only": ROOT / "results" / "quick_6h_radar" / "metrics.json",
    "6h PWV-coupled": ROOT / "results" / "quick_6h_pwv" / "metrics.json",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_results():
    return {name: read_json(path) for name, path in RESULTS.items() if path.exists()}


def save_horizon_bars(results):
    for metric in ("mae", "rmse"):
        labels = []
        values = []
        for name, data in results.items():
            for horizon, item in data.get("horizon_metrics", {}).get("model", {}).items():
                labels.append("{}\n{}".format(name.replace(" ", "\n"), horizon))
                values.append(item[metric])
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(values) * 0.8), 4.8), dpi=180)
        bars = ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title("Lead-horizon {} comparison".format(metric.upper()))
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "{:.2f}".format(bar.get_height()),
                    ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "horizon_{}.png".format(metric))
        plt.close(fig)


def save_lead_curves(results):
    for metric in ("mae", "rmse"):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
        for name, data in results.items():
            lead_items = data.get("lead_time_metrics", {}).get("model", [])
            if not lead_items:
                continue
            x = [item["lead_minutes"] / 60.0 for item in lead_items]
            y = [item[metric] for item in lead_items]
            ax.plot(x, y, linewidth=2, label=name)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(metric.upper())
        ax.set_title("{} by lead time".format(metric.upper()))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "lead_{}.png".format(metric))
        plt.close(fig)


def save_summary(results):
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "model": data.get("model"),
            "persistence": data.get("persistence"),
            "horizon_metrics": data.get("horizon_metrics", {}).get("model", {}),
            "coupling_mean": data.get("coupling_mean"),
        }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    results = available_results()
    if not results:
        raise SystemExit("No long-horizon metrics found. Run quick_test_3h/6h scripts first.")
    save_horizon_bars(results)
    save_lead_curves(results)
    save_summary(results)
    print(OUT)


if __name__ == "__main__":
    main()
