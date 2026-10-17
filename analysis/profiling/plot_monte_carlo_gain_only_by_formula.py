from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


FAMILY_LABELS = {
    "A": "A: (.01, 2, .2)",
    "B": "B: (.05, 64, .2)",
    "C": "C: (.01, 64, .4)",
}

STRATEGY_TO_FAMILY = {
    "difference_A": "A",
    "difference_B": "B",
    "difference_C": "C",
    "ratio_A": "A",
    "ratio_B": "B",
    "ratio_C": "C",
}

HORIZONS = [5, 10, 20, 50]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Monte Carlo gain-only charts (Difference and Ratio) with families A/B/C."
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


def load_and_normalize(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)

    strategy_col = pick_column(
        df,
        ["strategy", "variant", "name"],
        "strategy",
    )
    horizon_col = pick_column(
        df,
        ["horizon", "H", "request_horizon"],
        "horizon",
    )
    gain_col = pick_column(
        df,
        [
            "latency_gain_percent",
            "matched_c_gain_percent",
            "gain_percent_vs_matched_notl",
        ],
        "gain",
    )

    out = df[[strategy_col, horizon_col, gain_col]].copy()
    out.columns = ["strategy", "horizon", "gain_percent"]

    out["strategy"] = out["strategy"].astype(str)
    out["horizon"] = out["horizon"].astype(int)
    out["gain_percent"] = out["gain_percent"].astype(float)

    out = out[out["strategy"].isin(STRATEGY_TO_FAMILY.keys())].copy()
    out["family"] = out["strategy"].map(STRATEGY_TO_FAMILY)
    out["formula"] = out["strategy"].apply(
        lambda s: "difference" if s.startswith("difference") else "ratio"
    )

    return out


def annotate_series(ax, xvals, yvals, prefix=""):
    offsets = {
        5: (0, 10),
        10: (0, 10),
        20: (0, 10),
        50: (0, 10),
    }
    for x, y in zip(xvals, yvals):
        dx, dy = offsets.get(int(x), (0, 10))
        ax.annotate(
            f"{prefix}+{y:.2f}%",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def make_gain_plot(df: pd.DataFrame, formula: str, out_path: Path) -> None:
    sub = df[df["formula"] == formula].copy()

    fig, ax = plt.subplots(figsize=(12, 7))

    for family in ["A", "B", "C"]:
        fam = sub[sub["family"] == family].copy()
        fam = fam.sort_values("horizon")

        if fam.empty:
            continue

        x = fam["horizon"].tolist()
        y = fam["gain_percent"].tolist()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.8,
            markersize=9,
            label=FAMILY_LABELS[family],
        )
        annotate_series(ax, x, y)

    pretty_formula = "Difference" if formula == "difference" else "Ratio"

    ax.set_title(
        f"Monte Carlo — gain only | {pretty_formula}",
        fontsize=20,
        pad=14,
    )
    ax.set_xlabel("Request horizon H", fontsize=15)
    ax.set_ylabel("Latency gain vs matched No-Transfer (%)", fontsize=15)

    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)

    ymin = sub["gain_percent"].min()
    ymax = sub["gain_percent"].max()
    margin = max(0.15, (ymax - ymin) * 0.18 if ymax > ymin else 0.5)
    ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize(summary_path)

    out_diff = output_dir / "mc_gain_only_difference_ABC.png"
    out_ratio = output_dir / "mc_gain_only_ratio_ABC.png"

    make_gain_plot(df, "difference", out_diff)
    make_gain_plot(df, "ratio", out_ratio)

    print("PASS — gain-only Monte Carlo charts generated")
    print(out_diff)
    print(out_ratio)


if __name__ == "__main__":
    main()