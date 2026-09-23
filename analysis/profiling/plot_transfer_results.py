#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save(fig, out_dir: Path, stem: str):
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregate_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("transfer_learning_graphs"))
    args = ap.parse_args()

    df = pd.read_csv(args.aggregate_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [5, 10, 20, 50]
    x = np.arange(len(horizons))
    xlabels = [f"H{h}" for h in horizons]
    labels = {"no_transfer": "No Transfer", "coupled": "Coupled", "decoupled": "Decoupled"}

    main_df = df[df["campaign"] == "transfer_main"].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in ["no_transfer", "coupled", "decoupled"]:
        d = main_df[main_df["variant"] == variant].set_index("horizon_requests").loc[horizons]
        ax.errorbar(
            x,
            d["cumulative_duration_mean_s"].to_numpy(),
            yerr=d["cumulative_duration_stdev_ms"].to_numpy() / 1000.0,
            marker="o", linewidth=2, capsize=4, label=labels[variant],
        )
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Orizzonte di richieste")
    ax.set_ylabel("Durata cumulativa di esecuzione (s)")
    ax.set_title("Transfer learning: confronto completo")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(0.01, -0.17, "Media su R01–R03; barre = deviazione standard tra repliche. c = 0.8.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, args.output_dir, "01_transfer_confronto_completo")

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in ["no_transfer", "decoupled"]:
        d = main_df[main_df["variant"] == variant].set_index("horizon_requests").loc[horizons]
        ax.errorbar(
            x,
            d["cumulative_duration_mean_s"].to_numpy(),
            yerr=d["cumulative_duration_stdev_ms"].to_numpy() / 1000.0,
            marker="o", linewidth=2, capsize=4, label=labels[variant],
        )
    dec = main_df[main_df["variant"] == "decoupled"].set_index("horizon_requests").loc[horizons]
    for i, h in enumerate(horizons):
        gain = float(dec.loc[h, "gain_vs_baseline_mean_pct"])
        y = float(dec.loc[h, "cumulative_duration_mean_s"])
        ax.annotate(
            f"−{gain:.2f}% costo" if gain >= 0 else f"+{abs(gain):.2f}% costo",
            (x[i], y),
            xytext=(0, -22 if i in (0, 2) else 14),
            textcoords="offset points", ha="center", fontsize=9, fontweight="bold"
        )
    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Orizzonte di richieste")
    ax.set_ylabel("Durata cumulativa di esecuzione (s)")
    ax.set_title("Beneficio del transfer learning: Decoupled vs No Transfer")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(0.01, -0.19,
            "Metrica: riduzione della cumulative execution duration durante l'adattamento. "
            "Media su R01–R03; c = 0.8.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, args.output_dir, "02_transfer_baseline_vs_decoupled")

    ab = df[(df["campaign"] == "exploration_ablation") & (df["horizon_requests"] == 50)].copy()
    order = [("coupled", 0.0), ("coupled", 0.8), ("decoupled", 0.0), ("decoupled", 0.8)]
    cats = ["Coupled\nc=0", "Coupled\nc=0.8", "Decoupled\nc=0", "Decoupled\nc=0.8"]
    vals, errs, arm_means = [], [], []
    for variant, c in order:
        r = ab[(ab["variant"] == variant) & (ab["c"] == c)].iloc[0]
        vals.append(float(r["cumulative_duration_mean_s"]))
        errs.append(float(r["cumulative_duration_stdev_ms"]) / 1000.0)
        arm_means.append(float(r["arm64_executions_mean"]))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(np.arange(len(cats)), vals, yerr=errs, capsize=4)
    for bar, v, arm in zip(bars, vals, arm_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.3,
                f"{v:.2f} s\nARM {arm:.1f}/50", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(cats)), cats)
    ax.set_ylabel("Durata cumulativa H50 (s)")
    ax.set_title("Ablation sull'esplorazione: effetto di c e del prior decoupled")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.01, -0.18,
            "Media su R02–R03; barre = deviazione standard. "
            "Il caso Decoupled c=0 mostra il lock-in greedy su amd64.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, args.output_dir, "03_exploration_ablation_h50")

if __name__ == "__main__":
    main()
