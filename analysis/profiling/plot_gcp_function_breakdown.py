# analysis/profiling/plot_gcp_function_breakdown.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
        "transfer_label": "Difference (.05, 64, .2)",
        "short_name": "difference",
        "title_name": "Difference",
    },
    "ratio": {
        "baseline_condition": "notl-ratio",
        "transfer_condition": "ratio",
        "baseline_label": "No Transfer (c=.4)",
        "transfer_label": "Ratio (.01, 64, .4)",
        "short_name": "ratio",
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
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
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
            100.0 * (merged["baseline_ms"] - merged["transfer_ms"]) / merged["baseline_ms"]
    )
    merged["criterion"] = criterion
    return merged.sort_values(["target_function", "horizon"]).reset_index(drop=True)


def _annotate_gain(ax, x, y_transfer, gains):
    for h, yt, g in zip(x, y_transfer, gains):
        label = f"{g:+.2f}%"
        ax.annotate(
            label,
            xy=(h, yt),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


def plot_single_function(
        pair_df: pd.DataFrame,
        criterion: str,
        target: str,
        out_png: Path,
        out_pdf: Path,
) -> None:
    meta = CONDITION_META[criterion]
    sub = pair_df[pair_df["target_function"] == target].copy()

    if sub.empty:
        raise ValueError(f"No data for target={target}, criterion={criterion}")

    x = sub["horizon"].tolist()
    y_baseline = (sub["baseline_ms"] / 1000.0).tolist()
    y_transfer = (sub["transfer_ms"] / 1000.0).tolist()
    gains = sub["gain_percent"].tolist()

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

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

    _annotate_gain(ax, x, y_transfer, gains)

    ax.set_title(
        f"GCP — {target} | {meta['title_name']}",
        fontsize=20,
        pad=14,
    )
    ax.set_xlabel("Request horizon H", fontsize=14)
    ax.set_ylabel("Cumulative execution time (s)", fontsize=14)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_criterion_grid(
        pair_df: pd.DataFrame,
        criterion: str,
        out_png: Path,
        out_pdf: Path,
) -> None:
    meta = CONDITION_META[criterion]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = pair_df[pair_df["target_function"] == target].copy()

        x = sub["horizon"].tolist()
        y_baseline = (sub["baseline_ms"] / 1000.0).tolist()
        y_transfer = (sub["transfer_ms"] / 1000.0).tolist()
        gains = sub["gain_percent"].tolist()

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

        for h, yt, g in zip(x, y_transfer, gains):
            ax.annotate(
                f"{g:+.2f}%",
                xy=(h, yt),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(target, fontsize=14, pad=8)
        ax.set_xlabel("H", fontsize=11)
        ax.set_ylabel("Cum. time (s)", fontsize=11)
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.suptitle(
        f"GCP — per-target breakdown | {meta['title_name']}",
        fontsize=22,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path) -> None:
    path.write_text(
        "# GCP function breakdown plots\n\n"
        "Outputs generated:\n"
        "- 8 single-function plots: 4 targets x 2 criteria;\n"
        "- 2 criterion grids: one for Difference, one for Ratio.\n\n"
        "Each single-function plot follows the same style as the desired "
        "'left chart': cumulative duration on Y, request horizon on X, "
        "baseline vs transfer curves, with gain annotations.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-data",
        type=Path,
        required=True,
        help="CSV produced by plot_gcp_final_overview.py (gcp_final_plot_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    df = load_plot_data(args.plot_data)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    by_function_dir = out / "by_function"
    by_function_dir.mkdir(parents=True, exist_ok=True)

    by_criterion_dir = out / "by_criterion"
    by_criterion_dir.mkdir(parents=True, exist_ok=True)

    diff_df = build_pair_df(df, "difference")
    ratio_df = build_pair_df(df, "ratio")

    # 8 single-function plots
    for target in TARGET_ORDER:
        plot_single_function(
            diff_df,
            "difference",
            target,
            by_function_dir / f"{target}_difference.png",
            by_function_dir / f"{target}_difference.pdf",
            )
        plot_single_function(
            ratio_df,
            "ratio",
            target,
            by_function_dir / f"{target}_ratio.png",
            by_function_dir / f"{target}_ratio.pdf",
            )

    # 2 criterion-level grids
    plot_criterion_grid(
        diff_df,
        "difference",
        by_criterion_dir / "criterion_difference_grid.png",
        by_criterion_dir / "criterion_difference_grid.pdf",
        )
    plot_criterion_grid(
        ratio_df,
        "ratio",
        by_criterion_dir / "criterion_ratio_grid.png",
        by_criterion_dir / "criterion_ratio_grid.pdf",
        )

    notes_path = out / "GCP_FUNCTION_BREAKDOWN_NOTES.md"
    write_notes(notes_path)

    print("PASS — GCP function breakdown plots generated")
    print()
    print("===== OUTPUTS =====")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(p)


if __name__ == "__main__":
    main()