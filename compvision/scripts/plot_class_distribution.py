"""
plot_class_distribution.py — Histogramas de distribución de clases YOLO.

Genera tres figuras:
  1. Total (train + val)
  2. Train vs Val lado a lado
  3. Porcentaje val/train por clase

Uso:
    python3 compvision/scripts/plot_class_distribution.py --dataset_dir dataset_yolo
    python3 compvision/scripts/plot_class_distribution.py --dataset_dir dataset_yolo --output_dir compvision/plots
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml


def count_split(labels_dir: Path) -> Counter:
    counts = Counter()
    for f in labels_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def load_names(dataset_dir: Path) -> list:
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("names", [])
    return []


def plot_total(names, total, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, [total[i] for i in range(len(names))], color="#4C72B0", edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de clases — Dataset completo (train + val)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_train_val(names, train, val, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    w = 0.38
    b1 = ax.bar(x - w / 2, [train[i] for i in range(len(names))], width=w,
                label="Train", color="#4C72B0", edgecolor="white")
    b2 = ax.bar(x + w / 2, [val[i] for i in range(len(names))], width=w,
                label="Val", color="#DD8452", edgecolor="white")
    ax.bar_label(b1, fmt="%d", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%d", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Instancias")
    ax.set_title("Distribución de clases — Train vs Val")
    ax.legend(framealpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_pct(names, train, val, out_path):
    totals = [train[i] + val[i] for i in range(len(names))]
    pct_train = [train[i] / t * 100 if t else 0 for i, t in enumerate(totals)]
    pct_val   = [val[i]   / t * 100 if t else 0 for i, t in enumerate(totals)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.5
    ax.bar(x, pct_train, width=w, label="Train %", color="#4C72B0", edgecolor="white")
    ax.bar(x, pct_val,   width=w, bottom=pct_train, label="Val %", color="#DD8452", edgecolor="white")
    ax.axhline(80, color="grey", linestyle="--", linewidth=0.8, label="80 %")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("% instancias")
    ax.set_ylim(0, 105)
    ax.set_title("Split estratificado por clase (% Train / Val)")
    ax.legend(framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    for i in range(len(names)):
        ax.text(i, pct_train[i] / 2, f"{pct_train[i]:.1f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(i, pct_train[i] + pct_val[i] / 2, f"{pct_val[i]:.1f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset_yolo")
    parser.add_argument("--output_dir",  default="compvision/plots")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(dataset_dir)
    train = count_split(dataset_dir / "labels" / "train")
    val   = count_split(dataset_dir / "labels" / "val")
    total = train + val

    if not names:
        names = [str(i) for i in sorted(total.keys())]

    print(f"\nClases: {names}")
    print(f"{'Clase':<12} {'Total':>7} {'Train':>7} {'Val':>7} {'Val%':>7}")
    print("-" * 48)
    for i, name in enumerate(names):
        t = total[i]; tr = train[i]; v = val[i]
        print(f"{name:<12} {t:>7,} {tr:>7,} {v:>7,} {v/t*100 if t else 0:>6.1f}%")

    plot_total(names, total, out_dir / "class_dist_total.png")
    plot_train_val(names, train, val, out_dir / "class_dist_train_val.png")
    plot_pct(names, train, val, out_dir / "class_dist_pct.png")

    print(f"\n3 figuras guardadas en {out_dir}/")


if __name__ == "__main__":
    main()
