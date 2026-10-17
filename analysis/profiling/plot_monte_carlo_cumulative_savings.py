#!/usr/bin/env python3
"""
Serverledge — Monte Carlo cumulative savings figure.

STYLE:
- same visual language as the cumulative-execution-time plots already approved;
- no zero reference line;
- no error bars in the main slide figure;
- two rising curves over H;
- labels show both cumulative seconds saved and relative aggregate gain.

Frozen configurations:
- Difference (.05,64,.2) vs matched No-TL c=.2
- Ratio      (.01,64,.4) vs matched No-TL c=.4

The plotted quantity is:
    saved_seconds(H) =
        (mean cumulative latency NoTL - mean cumulative latency TL) / 1000

The percentage label is:
    100 * (mean NoTL - mean TL) / mean NoTL

This is intentionally the aggregate-latency view used by the cumulative curves,
not the macro mean of per-target percentage gains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]


def get_strategy(summary: pd.DataFrame, name: str) -> pd.DataFrame:
    x = (
        summary[summary["strategy"] == name]
        .copy()
        .sort_values("horizon")
    )
    x = x[x["horizon"].isin(HORIZONS)].copy()

    got = x["horizon"].astype(int).tolist()
    if got != HORIZONS:
        raise SystemExit(
            f"{name}: expected horizons {HORIZONS}, got {got}"
        )
    return x


def calc(base: pd.DataFrame, tl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    b = base["cumulative_latency_ms"].to_numpy(float)
    t = tl["cumulative_latency_ms"].to_numpy(float)

    saved_s = (b - t) / 1000.0
    gain_pct = 100.0 * (b - t) / b

    return saved_s, gain_pct


def annotate(ax, xs, ys, pct, prefix):
    # Hand-tuned offsets to avoid overlaps in a slide.
    offsets = [(0, 11), (0, 11), (0, 11), (0, 11)]
    for x, y, p, (dx, dy) in zip(xs, ys, pct, offsets):
        ax.annotate(
            f"{prefix}: {y:.2f}s ({p:+.2f}%)",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)

    nt02 = get_strategy(summary, "no_transfer_c02")
    nt04 = get_strategy(summary, "no_transfer_c04")
    d = get_strategy(summary, "difference_B")
    r = get_strategy(summary, "ratio_C")

    d_saved, d_pct = calc(nt02, d)
    r_saved, r_pct = calc(nt04, r)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.plot(
        HORIZONS,
        d_saved,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Difference (.05, 64, .2)",
    )
    ax.plot(
        HORIZONS,
        r_saved,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Ratio (.01, 64, .4)",
    )

    ax.set_title("Monte Carlo — cumulative execution time saved")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Cumulative execution time saved vs matched No-TL (s)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")

    annotate(ax, HORIZONS, d_saved, d_pct, "D")
    annotate(ax, HORIZONS, r_saved, r_pct, "R")

    # Give labels breathing room without drawing a fixed zero line.
    ymin = min(float(d_saved.min()), float(r_saved.min()))
    ymax = max(float(d_saved.max()), float(r_saved.max()))
    span = max(ymax - ymin, 0.5)
    ax.set_ylim(max(0.0, ymin - 0.18 * span), ymax + 0.30 * span)

    fig.tight_layout()

    png = out / "05_mc_cumulative_time_saved_frozen.png"
    pdf = out / "05_mc_cumulative_time_saved_frozen.pdf"
    csv = out / "05_mc_cumulative_time_saved_frozen_values.csv"

    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    values = pd.DataFrame({
        "horizon": HORIZONS,
        "difference_saved_s": d_saved,
        "difference_aggregate_gain_percent": d_pct,
        "ratio_saved_s": r_saved,
        "ratio_aggregate_gain_percent": r_pct,
    })
    values.to_csv(csv, index=False)

    print("PASS — cumulative-savings Monte Carlo figure generated")
    print()
    print(values.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))
    print()
    print(png)
    print(pdf)
    print(csv)


if __name__ == "__main__":
    main()
