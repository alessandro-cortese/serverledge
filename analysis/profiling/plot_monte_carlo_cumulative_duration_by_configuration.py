#!/usr/bin/env python3
"""
Serverledge — Monte Carlo cumulative-duration plots by parameter configuration.

Goal
----
Generate three plots in the same visual style as the GCP slide:
for each parameter family, plot cumulative execution duration for:

    No Transfer
    Difference
    Ratio

The y-axis is cumulative execution time (seconds), NOT gain.
The gain percentages are shown only as annotations relative to the matched
No-Transfer baseline.

Families
--------
A = (.01, 2, .2), baseline No-TL c=.2
B = (.05, 64, .2), baseline No-TL c=.2
C = (.01, 64, .4), baseline No-TL c=.4

Input
-----
matched_multiseed_summary.csv

Output
------
mc_cumulative_duration_A.png/.pdf
mc_cumulative_duration_B.png/.pdf
mc_cumulative_duration_C.png/.pdf
mc_cumulative_duration_values.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]

FAMILIES = {
    "A": {
        "params": "(.01, 2, .2)",
        "baseline": "no_transfer_c02",
        "difference": "difference_A",
        "ratio": "ratio_A",
        "baseline_label": "No Transfer (c=.2)",
        "difference_label": "Difference (.01, 2, .2)",
        "ratio_label": "Ratio (.01, 2, .2)",
    },
    "B": {
        "params": "(.05, 64, .2)",
        "baseline": "no_transfer_c02",
        "difference": "difference_B",
        "ratio": "ratio_B",
        "baseline_label": "No Transfer (c=.2)",
        "difference_label": "Difference (.05, 64, .2)",
        "ratio_label": "Ratio (.05, 64, .2)",
    },
    "C": {
        "params": "(.01, 64, .4)",
        "baseline": "no_transfer_c04",
        "difference": "difference_C",
        "ratio": "ratio_C",
        "baseline_label": "No Transfer (c=.4)",
        "difference_label": "Difference (.01, 64, .4)",
        "ratio_label": "Ratio (.01, 64, .4)",
    },
}


def get_strategy(df: pd.DataFrame, name: str) -> pd.DataFrame:
    x = df[df["strategy"] == name].copy()
    x = x[x["horizon"].isin(HORIZONS)].sort_values("horizon")

    got = x["horizon"].astype(int).tolist()
    if got != HORIZONS:
        raise SystemExit(
            f"{name}: expected horizons {HORIZONS}, got {got}"
        )
    return x


def gain_percent(base_ms: np.ndarray, alt_ms: np.ndarray) -> np.ndarray:
    return 100.0 * (base_ms - alt_ms) / base_ms


def annotate_curve(
        ax,
        xs,
        ys,
        gains,
        prefix: str,
        *,
        above: bool,
) -> None:
    """
    Put gain labels near Difference/Ratio points.
    Offsets alternate so the two TL labels remain readable.
    """
    for x, y, g in zip(xs, ys, gains):
        if above:
            xytext = (0, 10)
            va = "bottom"
        else:
            xytext = (0, -18)
            va = "top"

        ax.annotate(
            f"{prefix} {g:+.2f}%",
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )


def plot_family(
        summary: pd.DataFrame,
        family: str,
        outdir: Path,
) -> list[dict]:
    cfg = FAMILIES[family]

    baseline = get_strategy(summary, cfg["baseline"])
    difference = get_strategy(summary, cfg["difference"])
    ratio = get_strategy(summary, cfg["ratio"])

    x = baseline["horizon"].to_numpy(int)

    base_ms = baseline["cumulative_latency_ms"].to_numpy(float)
    diff_ms = difference["cumulative_latency_ms"].to_numpy(float)
    ratio_ms = ratio["cumulative_latency_ms"].to_numpy(float)

    base_s = base_ms / 1000.0
    diff_s = diff_ms / 1000.0
    ratio_s = ratio_ms / 1000.0

    diff_gain = gain_percent(base_ms, diff_ms)
    ratio_gain = gain_percent(base_ms, ratio_ms)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.plot(
        x,
        base_s,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["baseline_label"],
    )
    ax.plot(
        x,
        diff_s,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["difference_label"],
    )
    ax.plot(
        x,
        ratio_s,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["ratio_label"],
    )

    annotate_curve(
        ax,
        x,
        diff_s,
        diff_gain,
        "D",
        above=False,
    )
    annotate_curve(
        ax,
        x,
        ratio_s,
        ratio_gain,
        "R",
        above=True,
    )

    ax.set_title(
        f"Monte Carlo — cumulative execution duration | {family}: {cfg['params']}",
        fontsize=18,
        pad=12,
    )
    ax.set_xlabel("Request horizon H", fontsize=14)
    ax.set_ylabel("Cumulative execution time (s)", fontsize=14)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11)

    # Give labels enough room, but keep the GCP-like cumulative shape.
    ymin = min(base_s.min(), diff_s.min(), ratio_s.min())
    ymax = max(base_s.max(), diff_s.max(), ratio_s.max())
    span = ymax - ymin
    ax.set_ylim(max(0.0, ymin - 0.08 * span), ymax + 0.10 * span)

    fig.tight_layout()

    stem = f"mc_cumulative_duration_{family}"
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"

    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for h, bs, ds, rs, dg, rg in zip(
            x, base_s, diff_s, ratio_s, diff_gain, ratio_gain
    ):
        rows.append({
            "family": family,
            "parameters": cfg["params"],
            "horizon": int(h),
            "no_transfer_cumulative_s": float(bs),
            "difference_cumulative_s": float(ds),
            "ratio_cumulative_s": float(rs),
            "difference_gain_percent": float(dg),
            "ratio_gain_percent": float(rg),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cumulative-duration Monte Carlo plots A/B/C."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)

    required = {
        "strategy",
        "horizon",
        "cumulative_latency_ms",
    }
    missing = required - set(summary.columns)
    if missing:
        raise SystemExit(
            f"summary missing required columns: {sorted(missing)}"
        )

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for family in ("A", "B", "C"):
        all_rows.extend(
            plot_family(summary, family, outdir)
        )

    values = pd.DataFrame(all_rows)
    csv_path = outdir / "mc_cumulative_duration_values.csv"
    values.to_csv(csv_path, index=False)

    print("PASS — cumulative-duration Monte Carlo plots generated")
    print()
    print(
        values.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()

    for family in ("A", "B", "C"):
        print(outdir / f"mc_cumulative_duration_{family}.png")
        print(outdir / f"mc_cumulative_duration_{family}.pdf")
    print(csv_path)


if __name__ == "__main__":
    main()
