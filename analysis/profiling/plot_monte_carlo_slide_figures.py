#!/usr/bin/env python3
"""
Generate the four slide-ready Monte Carlo figures for Serverledge.

Input:
  matched_multiseed_summary.csv
  matched_multiseed_paired_bootstrap.csv

Figures:
  01_notl_vs_difference_frozen.png/pdf
  02_notl_vs_ratio_frozen.png/pdf
  03_notl_vs_difference_vs_ratio_matched_A.png/pdf
  04_ratio_vs_difference_matched_families.png/pdf

Interpretation:
  Fig. 1 uses the frozen Difference configuration (.05,64,.2) and matched NoTL c=.2.
  Fig. 2 uses the frozen Ratio configuration (.01,64,.4) and matched NoTL c=.4.
  Fig. 3 compares Difference vs Ratio with identical parameters (.01,2,.2),
         together with the same NoTL c=.2 baseline.
  Fig. 4 isolates the formula effect under all three identical-parameter families:
         A=(.01,2,.2), B=(.05,64,.2), C=(.01,64,.4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]


def save_both(fig, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_zero_line(ax) -> None:
    ax.axhline(0.0, linewidth=1.0)


def annotate_h10(ax, x, y, text, *, dx=0, dy=8) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )


def subset(summary: pd.DataFrame, strategy: str) -> pd.DataFrame:
    x = summary[summary["strategy"] == strategy].copy()
    x = x[x["horizon"].isin(HORIZONS)].sort_values("horizon")
    if x["horizon"].tolist() != HORIZONS:
        raise SystemExit(
            f"{strategy}: expected horizons {HORIZONS}, got {x['horizon'].tolist()}"
        )
    return x


def plot_notl_vs_difference(summary: pd.DataFrame, out_dir: Path) -> None:
    d = subset(summary, "difference_B")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(
        HORIZONS,
        np.zeros(len(HORIZONS)),
        marker="o",
        label="No-TL (c=.2)",
    )
    ax.plot(
        d["horizon"],
        d["gain_percent_vs_matched_notl"],
        marker="o",
        label="Difference (.05,64,.2)",
    )

    add_zero_line(ax)
    ax.set_title("Monte Carlo — No-TL vs Difference")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Latency gain vs matched No-TL (%)")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(True, alpha=0.25)

    h10 = d[d["horizon"] == 10].iloc[0]
    annotate_h10(
        ax,
        10,
        h10["gain_percent_vs_matched_notl"],
        f"H10: +{h10['gain_percent_vs_matched_notl']:.2f}%",
    )

    save_both(fig, out_dir, "01_notl_vs_difference_frozen")


def plot_notl_vs_ratio(summary: pd.DataFrame, out_dir: Path) -> None:
    r = subset(summary, "ratio_C")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(
        HORIZONS,
        np.zeros(len(HORIZONS)),
        marker="o",
        label="No-TL (c=.4)",
    )
    ax.plot(
        r["horizon"],
        r["gain_percent_vs_matched_notl"],
        marker="o",
        label="Ratio (.01,64,.4)",
    )

    add_zero_line(ax)
    ax.set_title("Monte Carlo — No-TL vs Ratio")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Latency gain vs matched No-TL (%)")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(True, alpha=0.25)

    h10 = r[r["horizon"] == 10].iloc[0]
    annotate_h10(
        ax,
        10,
        h10["gain_percent_vs_matched_notl"],
        f"H10: +{h10['gain_percent_vs_matched_notl']:.2f}%",
    )

    save_both(fig, out_dir, "02_notl_vs_ratio_frozen")


def plot_three_way_matched_A(summary: pd.DataFrame, out_dir: Path) -> None:
    d = subset(summary, "difference_A")
    r = subset(summary, "ratio_A")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(
        HORIZONS,
        np.zeros(len(HORIZONS)),
        marker="o",
        label="No-TL (c=.2)",
    )
    ax.plot(
        d["horizon"],
        d["gain_percent_vs_matched_notl"],
        marker="o",
        label="Difference (.01,2,.2)",
    )
    ax.plot(
        r["horizon"],
        r["gain_percent_vs_matched_notl"],
        marker="o",
        label="Ratio (.01,2,.2)",
    )

    add_zero_line(ax)
    ax.set_title("Monte Carlo — matched Difference vs Ratio")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Latency gain vs matched No-TL (%)")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(True, alpha=0.25)

    d10 = d[d["horizon"] == 10].iloc[0]
    r10 = r[r["horizon"] == 10].iloc[0]
    annotate_h10(
        ax,
        10,
        d10["gain_percent_vs_matched_notl"],
        f"D: +{d10['gain_percent_vs_matched_notl']:.2f}%",
        dx=-28,
        dy=10,
    )
    annotate_h10(
        ax,
        10,
        r10["gain_percent_vs_matched_notl"],
        f"R: +{r10['gain_percent_vs_matched_notl']:.2f}%",
        dx=28,
        dy=10,
    )

    save_both(fig, out_dir, "03_notl_vs_difference_vs_ratio_matched_A")


def plot_formula_delta(paired: pd.DataFrame, out_dir: Path) -> None:
    families = [
        ("A_ratio_vs_difference", "A: (.01,2,.2)"),
        ("B_ratio_vs_difference", "B: (.05,64,.2)"),
        ("C_ratio_vs_difference", "C: (.01,64,.4)"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for comparison, label in families:
        x = paired[
            (paired["comparison"] == comparison)
            & (paired["metric"] == "latency_reduction_percent")
            & (paired["horizon"].isin(HORIZONS))
            ].copy().sort_values("horizon")

        if x["horizon"].tolist() != HORIZONS:
            raise SystemExit(
                f"{comparison}: expected horizons {HORIZONS}, "
                f"got {x['horizon'].tolist()}"
            )

        y = x["point_delta"].to_numpy(float)
        lo = x["ci95_low"].to_numpy(float)
        hi = x["ci95_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            x["horizon"],
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=label,
        )

    add_zero_line(ax)
    ax.set_title("Monte Carlo — Ratio vs Difference at matched parameters")
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel("Ratio − Difference latency advantage (%)")
    ax.set_xticks(HORIZONS)
    ax.legend(title="Identical parameter family")
    ax.grid(True, alpha=0.25)

    save_both(fig, out_dir, "04_ratio_vs_difference_matched_families")


def print_tables(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    print("===== FIGURE 1 — DIFFERENCE FROZEN =====")
    print(
        subset(summary, "difference_B")[
            ["horizon", "gain_percent_vs_matched_notl"]
        ].to_string(index=False)
    )

    print()
    print("===== FIGURE 2 — RATIO FROZEN =====")
    print(
        subset(summary, "ratio_C")[
            ["horizon", "gain_percent_vs_matched_notl"]
        ].to_string(index=False)
    )

    print()
    print("===== FIGURE 3 — MATCHED A =====")
    tmp = pd.concat(
        [
            subset(summary, "difference_A")[
                ["strategy", "horizon", "gain_percent_vs_matched_notl"]
            ],
            subset(summary, "ratio_A")[
                ["strategy", "horizon", "gain_percent_vs_matched_notl"]
            ],
        ],
        ignore_index=True,
    )
    print(tmp.to_string(index=False))

    print()
    print("===== FIGURE 4 — DIRECT FORMULA DELTA =====")
    x = paired[
        paired["comparison"].isin(
            [
                "A_ratio_vs_difference",
                "B_ratio_vs_difference",
                "C_ratio_vs_difference",
            ]
        )
        & (paired["metric"] == "latency_reduction_percent")
        & (paired["horizon"].isin(HORIZONS))
        ][
        [
            "comparison",
            "horizon",
            "point_delta",
            "ci95_low",
            "ci95_high",
        ]
    ].sort_values(["comparison", "horizon"])
    print(x.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--paired", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    summary = pd.read_csv(args.summary)
    paired = pd.read_csv(args.paired)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_notl_vs_difference(summary, out_dir)
    plot_notl_vs_ratio(summary, out_dir)
    plot_three_way_matched_A(summary, out_dir)
    plot_formula_delta(paired, out_dir)
    print_tables(summary, paired)

    print()
    print("PASS — Monte Carlo slide figures generated")
    for pth in sorted(out_dir.glob("*")):
        print(pth)


if __name__ == "__main__":
    main()
