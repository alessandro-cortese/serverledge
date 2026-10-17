#!/usr/bin/env python3
"""
Serverledge — corrected final GCP function-breakdown plots.

Changes vs previous version
---------------------------
1. Preserve the approved visual style and data.
2. Prevent H5/H50 annotations from being clipped or pushed outside the axes.
3. Separate Difference and Ratio labels when curves are very close.
4. Use explicit parameter names in legends:
      Difference: w_R=.05, w_E=64, c=.2
      Ratio:      w_R=.01, w_E=64, c=.4
   so tuple ordering is never ambiguous.
5. Generate both:
      - 8 large single-function plots;
      - 2 larger 2x2 criterion overview figures.

Input:
    gcp_final_plot_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]

TARGET_ORDER = [
    "billibench-knn",
    "dynamic-html-py",
    "functionbench-gzip",
    "graph-bfs",
]

CONDITION_META = {
    "difference": {
        "baseline_condition": "notl-difference",
        "transfer_condition": "difference",
        "baseline_label": "No Transfer (c=.2)",
        "transfer_label": r"Difference ($w_R$=.05, $w_E$=64, $c$=.2)",
        "title_name": "Difference",
    },
    "ratio": {
        "baseline_condition": "notl-ratio",
        "transfer_condition": "ratio",
        "baseline_label": "No Transfer (c=.4)",
        "transfer_label": r"Ratio ($w_R$=.01, $w_E$=64, $c$=.4)",
        "title_name": "Ratio",
    },
}


def load_plot_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "target_function",
        "condition",
        "horizon",
        "cumulative_duration_ms",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )
    return df


def build_pair_df(df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    meta = CONDITION_META[criterion]

    baseline = (
        df[df["condition"] == meta["baseline_condition"]][
            ["target_function", "horizon", "cumulative_duration_ms"]
        ]
        .rename(columns={"cumulative_duration_ms": "baseline_ms"})
        .copy()
    )

    transfer = (
        df[df["condition"] == meta["transfer_condition"]][
            ["target_function", "horizon", "cumulative_duration_ms"]
        ]
        .rename(columns={"cumulative_duration_ms": "transfer_ms"})
        .copy()
    )

    merged = baseline.merge(
        transfer,
        on=["target_function", "horizon"],
        validate="one_to_one",
    )

    merged["gain_percent"] = (
            100.0
            * (merged["baseline_ms"] - merged["transfer_ms"])
            / merged["baseline_ms"]
    )
    merged["criterion"] = criterion

    return merged.sort_values(
        ["target_function", "horizon"]
    ).reset_index(drop=True)


def annotation_offset(
        horizon: int,
        gain: float,
        *,
        compact: bool,
) -> tuple[float, float, str, str]:
    """
    Boundary-aware annotation placement.

    H5  -> move right, never outside left/bottom edge.
    H50 -> move left, never outside right edge.
    Positive gain -> label mostly below TL point.
    Negative gain -> label mostly above TL point.
    """
    dx_edge = 12 if compact else 18
    dy = 8 if compact else 11

    if horizon == 5:
        dx = dx_edge
        ha = "left"
    elif horizon == 50:
        dx = -dx_edge
        ha = "right"
    else:
        dx = 0
        ha = "center"

    if gain >= 0:
        yoff = -dy
        va = "top"
    else:
        yoff = dy
        va = "bottom"

    return dx, yoff, ha, va


def annotate_gain(
        ax,
        x,
        y_transfer,
        gains,
        *,
        compact: bool = False,
) -> None:
    fontsize = 8 if compact else 10

    for h, yt, g in zip(x, y_transfer, gains):
        dx, dy, ha, va = annotation_offset(
            int(h),
            float(g),
            compact=compact,
        )

        ax.annotate(
            f"{g:+.2f}%",
            xy=(h, yt),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight="bold",
            annotation_clip=False,
        )


def set_safe_limits(ax, y_baseline, y_transfer) -> None:
    ymin = min(min(y_baseline), min(y_transfer))
    ymax = max(max(y_baseline), max(y_transfer))
    span = max(ymax - ymin, 0.5)

    lower = max(0.0, ymin - 0.12 * span)
    upper = ymax + 0.12 * span
    ax.set_ylim(lower, upper)

    # Extra horizontal breathing room for H5/H50 labels.
    ax.set_xlim(2.6, 52.4)


def plot_single_function(
        pair_df: pd.DataFrame,
        criterion: str,
        target: str,
        out_png: Path,
        out_pdf: Path,
) -> None:
    meta = CONDITION_META[criterion]
    sub = pair_df[
        pair_df["target_function"] == target
        ].copy()

    if sub.empty:
        raise ValueError(
            f"No data for target={target}, criterion={criterion}"
        )

    x = sub["horizon"].to_numpy(int)
    y_baseline = (
            sub["baseline_ms"].to_numpy(float) / 1000.0
    )
    y_transfer = (
            sub["transfer_ms"].to_numpy(float) / 1000.0
    )
    gains = sub["gain_percent"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    ax.plot(
        x,
        y_baseline,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=meta["baseline_label"],
    )
    ax.plot(
        x,
        y_transfer,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=meta["transfer_label"],
    )

    annotate_gain(
        ax,
        x,
        y_transfer,
        gains,
        compact=False,
    )

    ax.set_title(
        f"GCP — {target} | {meta['title_name']}",
        fontsize=19,
        pad=14,
    )
    ax.set_xlabel(
        "Request horizon H",
        fontsize=14,
    )
    ax.set_ylabel(
        "Cumulative execution time (s)",
        fontsize=14,
    )
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(
        loc="upper left",
        fontsize=10.5,
    )

    set_safe_limits(
        ax,
        y_baseline,
        y_transfer,
    )

    fig.tight_layout()
    fig.savefig(
        out_png,
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_criterion_grid(
        pair_df: pd.DataFrame,
        criterion: str,
        out_png: Path,
        out_pdf: Path,
) -> None:
    meta = CONDITION_META[criterion]

    # Larger than the previous grid: keeps each panel legible.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.0, 10.8),
    )
    axes = axes.flatten()

    for ax, target in zip(
            axes,
            TARGET_ORDER,
    ):
        sub = pair_df[
            pair_df["target_function"] == target
            ].copy()

        x = sub["horizon"].to_numpy(int)
        y_baseline = (
                sub["baseline_ms"].to_numpy(float) / 1000.0
        )
        y_transfer = (
                sub["transfer_ms"].to_numpy(float) / 1000.0
        )
        gains = sub[
            "gain_percent"
        ].to_numpy(float)

        ax.plot(
            x,
            y_baseline,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=meta["baseline_label"],
        )
        ax.plot(
            x,
            y_transfer,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=meta["transfer_label"],
        )

        annotate_gain(
            ax,
            x,
            y_transfer,
            gains,
            compact=True,
        )

        ax.set_title(
            target,
            fontsize=14,
            pad=8,
        )
        ax.set_xlabel(
            "Request horizon H",
            fontsize=10.5,
        )
        ax.set_ylabel(
            "Cumulative time (s)",
            fontsize=10.5,
        )
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.22)

        set_safe_limits(
            ax,
            y_baseline,
            y_transfer,
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.965),
    )

    fig.suptitle(
        f"GCP — per-target breakdown | {meta['title_name']}",
        fontsize=21,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    fig.savefig(
        out_png,
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-data",
        type=Path,
        required=True,
        help="gcp_final_plot_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    df = load_plot_data(
        args.plot_data
    )

    out = args.output_dir.resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_function_dir = (
            out / "by_function"
    )
    by_function_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_criterion_dir = (
            out / "by_criterion"
    )
    by_criterion_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diff_df = build_pair_df(
        df,
        "difference",
    )
    ratio_df = build_pair_df(
        df,
        "ratio",
    )

    for target in TARGET_ORDER:
        plot_single_function(
            diff_df,
            "difference",
            target,
            by_function_dir
            / f"{target}_difference.png",
            by_function_dir
            / f"{target}_difference.pdf",
            )

        plot_single_function(
            ratio_df,
            "ratio",
            target,
            by_function_dir
            / f"{target}_ratio.png",
            by_function_dir
            / f"{target}_ratio.pdf",
            )

    plot_criterion_grid(
        diff_df,
        "difference",
        by_criterion_dir
        / "criterion_difference_grid.png",
        by_criterion_dir
        / "criterion_difference_grid.pdf",
        )

    plot_criterion_grid(
        ratio_df,
        "ratio",
        by_criterion_dir
        / "criterion_ratio_grid.png",
        by_criterion_dir
        / "criterion_ratio_grid.pdf",
        )

    print(
        "PASS — corrected final GCP breakdown plots generated"
    )
    print()
    print(
        "Parameter notation:"
    )
    print(
        "Difference: w_R=.05, w_E=64, c=.2"
    )
    print(
        "Ratio:      w_R=.01, w_E=64, c=.4"
    )
    print()

    for p in sorted(
            out.rglob("*")
    ):
        if p.is_file():
            print(p)


if __name__ == "__main__":
    main()
