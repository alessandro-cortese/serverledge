#!/usr/bin/env python3
"""
Serverledge — Monte Carlo gain-only plot for the frozen configurations.

This figure uses the PRIMARY macro-average endpoint:
mean of each target's percentage latency gain vs its matched No-TL baseline.

Difference frozen:
    (.05, 64, .2) vs No-TL c=.2

Ratio frozen:
    (.01, 64, .4) vs No-TL c=.4

The 95% CIs come from the paired target-level bootstrap already produced by
transfer_formula_matched_multiseed.py.

Important:
This is NOT the same quantity as:
    100 * (mean baseline latency - mean TL latency) / mean baseline latency
which weights slower functions more heavily.

Output:
    05_mc_gain_only_frozen.png
    05_mc_gain_only_frozen.pdf
    05_mc_gain_only_frozen_values.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]

PAIRS = {
    "Difference (.05, 64, .2)": "B_difference_vs_notl_c02",
    "Ratio (.01, 64, .4)": "C_ratio_vs_notl_c04",
}


def load_curve(
        paired: pd.DataFrame,
        comparison: str,
) -> pd.DataFrame:
    x = paired[
        (paired["comparison"] == comparison)
        & (paired["metric"] == "latency_reduction_percent")
        & (paired["horizon"].isin(HORIZONS))
        ].copy().sort_values("horizon")

    got = x["horizon"].astype(int).tolist()
    if got != HORIZONS:
        raise SystemExit(
            f"{comparison}: expected horizons {HORIZONS}, got {got}"
        )

    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    paired = pd.read_csv(args.paired)

    curves = {
        label: load_curve(paired, comparison)
        for label, comparison in PAIRS.items()
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    saved_rows = []

    for label, x in curves.items():
        y = x["point_delta"].to_numpy(float)
        lo = x["ci95_low"].to_numpy(float)
        hi = x["ci95_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            HORIZONS,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.3,
            markersize=7,
            capsize=4,
            label=label,
        )

        for h, yy, l, u in zip(HORIZONS, y, lo, hi):
            ax.annotate(
                f"{yy:+.2f}%",
                xy=(h, yy),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
            saved_rows.append({
                "strategy": label,
                "horizon": h,
                "macro_mean_gain_percent": yy,
                "ci95_low": l,
                "ci95_high": u,
            })

    ax.axhline(0.0, linewidth=1.2)
    ax.set_title("Monte Carlo — aggregate latency gain")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Mean per-target latency gain vs matched No-TL (%)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")

    fig.tight_layout()

    png = out / "05_mc_gain_only_frozen.png"
    pdf = out / "05_mc_gain_only_frozen.pdf"
    csv = out / "05_mc_gain_only_frozen_values.csv"

    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    values = pd.DataFrame(saved_rows)
    values.to_csv(csv, index=False)

    print("PASS — gain-only Monte Carlo figure generated")
    print()
    print(values.to_string(
        index=False,
        float_format=lambda v: f"{v:.6f}",
    ))
    print()
    print(png)
    print(pdf)
    print(csv)


if __name__ == "__main__":
    main()
