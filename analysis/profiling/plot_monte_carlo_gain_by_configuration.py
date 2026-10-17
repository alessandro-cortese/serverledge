from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HORIZONS = [5, 10, 20, 50]

FAMILY_META = {
    "A": {
        "label": "A: (.01, 2, .2)",
        "difference": "difference_A",
        "ratio": "ratio_A",
        "baseline_label": "No Transfer (c=.2)",
    },
    "B": {
        "label": "B: (.05, 64, .2)",
        "difference": "difference_B",
        "ratio": "ratio_B",
        "baseline_label": "No Transfer (c=.2)",
    },
    "C": {
        "label": "C: (.01, 64, .4)",
        "difference": "difference_C",
        "ratio": "ratio_C",
        "baseline_label": "No Transfer (c=.4)",
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Monte Carlo gain-only charts by configuration (No Transfer vs Difference vs Ratio)."
    )
    p.add_argument(
        "--summary",
        required=True,
        help="Path to matched_multiseed_summary.csv",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures will be saved",
    )
    return p


def pick_column(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find column for {what}. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)

    strategy_col = pick_column(df, ["strategy", "variant", "name"], "strategy")
    horizon_col = pick_column(df, ["horizon", "H", "request_horizon"], "horizon")
    gain_col = pick_column(
        df,
        ["latency_gain_percent", "matched_c_gain_percent", "gain_percent_vs_matched_notl"],
        "gain",
    )

    out = df[[strategy_col, horizon_col, gain_col]].copy()
    out.columns = ["strategy", "horizon", "gain_percent"]

    out["strategy"] = out["strategy"].astype(str)
    out["horizon"] = out["horizon"].astype(int)
    out["gain_percent"] = out["gain_percent"].astype(float)

    return out


def annotate_points(ax, xvals, yvals, prefix: str) -> None:
    for x, y in zip(xvals, yvals):
        ax.annotate(
            f"{prefix} +{y:.2f}%",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def make_plot_for_family(df: pd.DataFrame, family_key: str, output_dir: Path) -> Path:
    meta = FAMILY_META[family_key]
    diff_name = meta["difference"]
    ratio_name = meta["ratio"]

    diff_df = df[df["strategy"] == diff_name].sort_values("horizon").copy()
    ratio_df = df[df["strategy"] == ratio_name].sort_values("horizon").copy()

    if diff_df.empty or ratio_df.empty:
        raise ValueError(
            f"Missing data for family {family_key}: "
            f"difference={diff_name} rows={len(diff_df)}, "
            f"ratio={ratio_name} rows={len(ratio_df)}"
        )

    x = diff_df["horizon"].tolist()
    y_diff = diff_df["gain_percent"].tolist()
    y_ratio = ratio_df["gain_percent"].tolist()
    y_notl = [0.0 for _ in x]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x, y_notl,
        marker="o",
        linewidth=2.8,
        markersize=9,
        label=meta["baseline_label"],
    )
    ax.plot(
        x, y_diff,
        marker="o",
        linewidth=2.8,
        markersize=9,
        label=f"Difference {meta['label'].split(':', 1)[1].strip()}",
    )
    ax.plot(
        x, y_ratio,
        marker="o",
        linewidth=2.8,
        markersize=9,
        label=f"Ratio {meta['label'].split(':', 1)[1].strip()}",
    )

    annotate_points(ax, x, y_diff, "D")
    annotate_points(ax, x, y_ratio, "R")

    ax.set_title(
        f"Monte Carlo — gain only | {meta['label']}",
        fontsize=20,
        pad=14,
    )
    ax.set_xlabel("Request horizon H", fontsize=15)
    ax.set_ylabel("Latency gain vs matched No-Transfer (%)", fontsize=15)

    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)

    ymin = min(min(y_diff), min(y_ratio), 0.0)
    ymax = max(max(y_diff), max(y_ratio), 0.0)
    margin = max(0.20, (ymax - ymin) * 0.18 if ymax > ymin else 0.5)
    ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()

    out_path = output_dir / f"mc_gain_only_{family_key}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    args = build_parser().parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)

    outputs = []
    for family_key in ["A", "B", "C"]:
        outputs.append(make_plot_for_family(df, family_key, output_dir))

    print("PASS — Monte Carlo gain-by-configuration charts generated")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()