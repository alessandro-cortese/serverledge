#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZONS = [5, 10, 20, 50]

FAMILIES = {
    "A": {
        "title_suffix": "A: (.01, 2, .2)",
        "baseline": "no_transfer_c02",
        "difference": "difference_A",
        "ratio": "ratio_A",
        "baseline_label": "No Transfer (c=.2)",
        "difference_label": "Difference (.01, 2, .2)",
        "ratio_label": "Ratio (.01, 2, .2)",
    },
    "B": {
        "title_suffix": "B: (.05, 64, .2)",
        "baseline": "no_transfer_c02",
        "difference": "difference_B",
        "ratio": "ratio_B",
        "baseline_label": "No Transfer (c=.2)",
        "difference_label": "Difference (.05, 64, .2)",
        "ratio_label": "Ratio (.05, 64, .2)",
    },
    "C": {
        "title_suffix": "C: (.01, 64, .4)",
        "baseline": "no_transfer_c04",
        "difference": "difference_C",
        "ratio": "ratio_C",
        "baseline_label": "No Transfer (c=.4)",
        "difference_label": "Difference (.01, 64, .4)",
        "ratio_label": "Ratio (.01, 64, .4)",
    },
}


def load_series(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    x = df[df["strategy"] == strategy].copy().sort_values("horizon")
    x = x[x["horizon"].isin(HORIZONS)].copy()
    got = x["horizon"].astype(int).tolist()
    if got != HORIZONS:
        raise ValueError(f"{strategy}: expected horizons {HORIZONS}, got {got}")
    return x


def gain_percent(baseline_ms: np.ndarray, strategy_ms: np.ndarray) -> np.ndarray:
    return 100.0 * (baseline_ms - strategy_ms) / baseline_ms


def add_gain_labels(ax, xs, ys, gains, prefix: str) -> None:
    offsets = {
        5: (-2, -18 if prefix == "D" else 8),
        10: (0, -16 if prefix == "D" else 8),
        20: (6, -16 if prefix == "D" else 8),
        50: (18, -10 if prefix == "D" else 8),
    }
    for x, y, g in zip(xs, ys, gains):
        dx, dy = offsets[int(x)]
        ax.annotate(
            f"{prefix} +{g:.2f}%",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )


def plot_family(df: pd.DataFrame, family_key: str, outdir: Path):
    cfg = FAMILIES[family_key]

    baseline = load_series(df, cfg["baseline"])
    difference = load_series(df, cfg["difference"])
    ratio = load_series(df, cfg["ratio"])

    x = baseline["horizon"].to_numpy(int)
    y0 = np.zeros(len(x), dtype=float)

    b_ms = baseline["cumulative_latency_ms"].to_numpy(float)
    d_ms = difference["cumulative_latency_ms"].to_numpy(float)
    r_ms = ratio["cumulative_latency_ms"].to_numpy(float)

    d_gain = gain_percent(b_ms, d_ms)
    r_gain = gain_percent(b_ms, r_ms)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    ax.plot(x, y0, marker="o", linewidth=2.5, markersize=7, label=cfg["baseline_label"])
    ax.plot(x, d_gain, marker="o", linewidth=2.5, markersize=7, label=cfg["difference_label"])
    ax.plot(x, r_gain, marker="o", linewidth=2.5, markersize=7, label=cfg["ratio_label"])

    ax.set_title(f"Monte Carlo — latency gain | {cfg['title_suffix']}")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Latency gain vs matched No-Transfer (%)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")

    add_gain_labels(ax, x, d_gain, d_gain, "D")
    add_gain_labels(ax, x, r_gain, r_gain, "R")

    ymin = min(0.0, float(d_gain.min()), float(r_gain.min()))
    ymax = max(0.0, float(d_gain.max()), float(r_gain.max()))
    span = max(ymax - ymin, 0.5)
    ax.set_ylim(ymin - 0.25 * span, ymax + 0.30 * span)

    fig.tight_layout()

    stem = f"mc_gain_family_{family_key}"
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for h, dg, rg in zip(x, d_gain, r_gain):
        rows.append({
            "family": family_key,
            "horizon": int(h),
            "baseline_strategy": cfg["baseline"],
            "difference_strategy": cfg["difference"],
            "ratio_strategy": cfg["ratio"],
            "difference_gain_percent": float(dg),
            "ratio_gain_percent": float(rg),
        })
    return png, pdf, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Monte Carlo gain charts per parameter family.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    created = []
    all_rows = []

    for fam in ["A", "B", "C"]:
        png, pdf, rows = plot_family(summary, fam, outdir)
        created.extend([str(png), str(pdf)])
        all_rows.extend(rows)

    values = pd.DataFrame(all_rows)
    csv_path = outdir / "mc_gain_family_values.csv"
    values.to_csv(csv_path, index=False)

    notes = outdir / "MC_GAIN_FAMILY_NOTES.md"
    notes.write_text(
        "# Monte Carlo gain charts by parameter family\n\n"
        "Each figure shows, for one parameter family (A/B/C):\n"
        "- the matched No-Transfer baseline as a 0% gain line;\n"
        "- the Difference formula gain curve;\n"
        "- the Ratio formula gain curve.\n\n"
        "Gain is computed as:\n\n"
        "gain(H) = 100 * (Latency_NoTL(H) - Latency_TL(H)) / Latency_NoTL(H)\n\n"
        "Families:\n"
        "- A = (.01, 2, .2), baseline c=.2\n"
        "- B = (.05, 64, .2), baseline c=.2\n"
        "- C = (.01, 64, .4), baseline c=.4\n",
        encoding="utf-8",
    )

    print("PASS — Monte Carlo gain figures by family generated")
    print()
    print(values.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    for p in created:
        print(p)
    print(csv_path)
    print(notes)


if __name__ == "__main__":
    main()
