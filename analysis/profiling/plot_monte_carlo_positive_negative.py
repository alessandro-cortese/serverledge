#!/usr/bin/env python3
"""
Serverledge — Monte Carlo positive-only / negative-only cumulative-duration plots.

For each matched parameter family A/B/C, split target functions into two FIXED
cohorts using the primary endpoint H=10:

Positive cohort:
    both Difference and Ratio improve over matched No-TL at H=10.

Negative cohort:
    both Difference and Ratio worsen vs matched No-TL at H=10.

Then plot, for the SAME cohort across H={5,10,20,50}:
    - No Transfer
    - Difference
    - Ratio

This keeps the visual style of the cumulative-duration plots and avoids changing
the function set at every horizon.

Families:
A = (.01, 2, .2),   No-TL c=.2
B = (.05, 64, .2), No-TL c=.2
C = (.01, 64, .4), No-TL c=.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]
COHORT_HORIZON = 10

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


def non_synthetic_mask(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.lower().isin(["true", "1"])


def load_primary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "target_function",
        "target_is_synthetic",
        "strategy",
        "horizon",
        "cumulative_latency_ms",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"per-target CSV missing columns: {sorted(missing)}"
        )

    df = df[non_synthetic_mask(df["target_is_synthetic"])].copy()
    df["horizon"] = df["horizon"].astype(int)

    n_targets = df["target_function"].nunique()
    if n_targets != 57:
        raise SystemExit(
            f"expected 57 primary non-synthetic targets, got {n_targets}"
        )

    return df


def strategy_frame(
        df: pd.DataFrame,
        strategy: str,
        horizon: int | None = None,
) -> pd.DataFrame:
    x = df[df["strategy"] == strategy].copy()
    if horizon is not None:
        x = x[x["horizon"] == horizon].copy()
    return x


def classify_cohorts(
        df: pd.DataFrame,
        family: str,
) -> tuple[list[str], list[str], pd.DataFrame]:
    cfg = FAMILIES[family]

    baseline = strategy_frame(df, cfg["baseline"], COHORT_HORIZON)[
        ["target_function", "cumulative_latency_ms"]
    ].rename(columns={"cumulative_latency_ms": "baseline_ms"})

    difference = strategy_frame(df, cfg["difference"], COHORT_HORIZON)[
        ["target_function", "cumulative_latency_ms"]
    ].rename(columns={"cumulative_latency_ms": "difference_ms"})

    ratio = strategy_frame(df, cfg["ratio"], COHORT_HORIZON)[
        ["target_function", "cumulative_latency_ms"]
    ].rename(columns={"cumulative_latency_ms": "ratio_ms"})

    merged = (
        baseline
        .merge(difference, on="target_function", validate="one_to_one")
        .merge(ratio, on="target_function", validate="one_to_one")
    )

    if len(merged) != 57:
        raise SystemExit(
            f"family {family}: expected 57 H10 paired targets, got {len(merged)}"
        )

    merged["difference_gain_h10"] = (
            100.0
            * (merged["baseline_ms"] - merged["difference_ms"])
            / merged["baseline_ms"]
    )
    merged["ratio_gain_h10"] = (
            100.0
            * (merged["baseline_ms"] - merged["ratio_ms"])
            / merged["baseline_ms"]
    )

    merged["cohort"] = "mixed_or_neutral"

    merged.loc[
        (merged["difference_gain_h10"] > 0)
        & (merged["ratio_gain_h10"] > 0),
        "cohort",
    ] = "positive_both"

    merged.loc[
        (merged["difference_gain_h10"] < 0)
        & (merged["ratio_gain_h10"] < 0),
        "cohort",
    ] = "negative_both"

    positive = sorted(
        merged.loc[
            merged["cohort"] == "positive_both",
            "target_function",
        ].tolist()
    )
    negative = sorted(
        merged.loc[
            merged["cohort"] == "negative_both",
            "target_function",
        ].tolist()
    )

    merged.insert(0, "family", family)
    merged.insert(1, "parameters", cfg["params"])

    return positive, negative, merged


def aggregate_cohort(
        df: pd.DataFrame,
        family: str,
        targets: list[str],
        cohort_name: str,
) -> pd.DataFrame:
    cfg = FAMILIES[family]

    if not targets:
        return pd.DataFrame()

    rows = []

    for h in HORIZONS:
        values = {}

        for role, strategy in (
                ("no_transfer", cfg["baseline"]),
                ("difference", cfg["difference"]),
                ("ratio", cfg["ratio"]),
        ):
            x = df[
                (df["strategy"] == strategy)
                & (df["horizon"] == h)
                & (df["target_function"].isin(targets))
                ].copy()

            if x["target_function"].nunique() != len(targets):
                raise SystemExit(
                    f"{family}/{cohort_name}/H{h}/{strategy}: "
                    f"expected {len(targets)} targets, got "
                    f"{x['target_function'].nunique()}"
                )

            values[role] = float(
                x["cumulative_latency_ms"].mean()
            )

        d_gain = (
                100.0
                * (values["no_transfer"] - values["difference"])
                / values["no_transfer"]
        )
        r_gain = (
                100.0
                * (values["no_transfer"] - values["ratio"])
                / values["no_transfer"]
        )

        rows.append({
            "family": family,
            "parameters": cfg["params"],
            "cohort": cohort_name,
            "cohort_horizon": COHORT_HORIZON,
            "target_count": len(targets),
            "horizon": h,
            "no_transfer_cumulative_ms": values["no_transfer"],
            "difference_cumulative_ms": values["difference"],
            "ratio_cumulative_ms": values["ratio"],
            "difference_gain_percent": d_gain,
            "ratio_gain_percent": r_gain,
        })

    return pd.DataFrame(rows)


def annotate_gain(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        gain: np.ndarray,
        prefix: str,
        *,
        above: bool,
) -> None:
    for h, yy, gg in zip(x, y, gain):
        dy = 10 if above else -18
        va = "bottom" if above else "top"

        ax.annotate(
            f"{prefix} {gg:+.2f}%",
            xy=(h, yy),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )


def plot_cohort(
        agg: pd.DataFrame,
        family: str,
        cohort_name: str,
        outdir: Path,
) -> tuple[Path, Path]:
    cfg = FAMILIES[family]

    if agg.empty:
        raise SystemExit(
            f"family {family}: cohort {cohort_name} contains zero targets"
        )

    x = agg["horizon"].to_numpy(int)
    no_tl = agg["no_transfer_cumulative_ms"].to_numpy(float) / 1000.0
    diff = agg["difference_cumulative_ms"].to_numpy(float) / 1000.0
    ratio = agg["ratio_cumulative_ms"].to_numpy(float) / 1000.0

    d_gain = agg["difference_gain_percent"].to_numpy(float)
    r_gain = agg["ratio_gain_percent"].to_numpy(float)

    n = int(agg["target_count"].iloc[0])

    pretty = (
        "positive transfer only"
        if cohort_name == "positive_both"
        else "negative transfer only"
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.plot(
        x,
        no_tl,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["baseline_label"],
    )
    ax.plot(
        x,
        diff,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["difference_label"],
    )
    ax.plot(
        x,
        ratio,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=cfg["ratio_label"],
    )

    annotate_gain(
        ax, x, diff, d_gain, "D",
        above=False,
    )
    annotate_gain(
        ax, x, ratio, r_gain, "R",
        above=True,
    )

    ax.set_title(
        f"Monte Carlo — {pretty} | {family}: {cfg['params']} | n={n}",
        fontsize=18,
        pad=12,
    )
    ax.set_xlabel("Request horizon H", fontsize=14)
    ax.set_ylabel("Mean cumulative execution time (s)", fontsize=14)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11)

    ymin = min(no_tl.min(), diff.min(), ratio.min())
    ymax = max(no_tl.max(), diff.max(), ratio.max())
    span = max(ymax - ymin, 1.0)
    ax.set_ylim(
        max(0.0, ymin - 0.07 * span),
        ymax + 0.10 * span,
        )

    fig.tight_layout()

    stem_prefix = (
        "mc_positive_only"
        if cohort_name == "positive_both"
        else "mc_negative_only"
    )
    stem = f"{stem_prefix}_{family}"

    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"

    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-target",
        type=Path,
        required=True,
        help="matched_multiseed_per_target.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    df = load_primary(args.per_target)

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cohort_frames = []
    plot_frames = []
    created = []

    print(
        f"cohort_definition=H{COHORT_HORIZON}; "
        "positive=both formulas gain>0; "
        "negative=both formulas gain<0"
    )
    print()

    for family in ("A", "B", "C"):
        positive, negative, membership = classify_cohorts(
            df, family
        )
        cohort_frames.append(membership)

        print(
            f"{family} {FAMILIES[family]['params']}: "
            f"positive_both={len(positive)}/57 "
            f"negative_both={len(negative)}/57 "
            f"mixed_or_neutral={57-len(positive)-len(negative)}/57"
        )

        pos_agg = aggregate_cohort(
            df, family, positive, "positive_both"
        )
        neg_agg = aggregate_cohort(
            df, family, negative, "negative_both"
        )

        plot_frames.extend([pos_agg, neg_agg])

        if not pos_agg.empty:
            created.extend(
                plot_cohort(
                    pos_agg,
                    family,
                    "positive_both",
                    outdir,
                )
            )

        if not neg_agg.empty:
            created.extend(
                plot_cohort(
                    neg_agg,
                    family,
                    "negative_both",
                    outdir,
                )
            )

    cohorts = pd.concat(
        cohort_frames,
        ignore_index=True,
    )
    cohort_path = (
            outdir / "mc_positive_negative_cohorts.csv"
    )
    cohorts.to_csv(cohort_path, index=False)

    nonempty = [x for x in plot_frames if not x.empty]
    plot_values = pd.concat(
        nonempty,
        ignore_index=True,
    )
    values_path = (
            outdir / "mc_positive_negative_plot_values.csv"
    )
    plot_values.to_csv(values_path, index=False)

    notes = outdir / "MC_POSITIVE_NEGATIVE_NOTES.md"
    notes.write_text(
        "# Monte Carlo positive/negative transfer plots\n\n"
        "Cohorts are fixed at the primary endpoint H=10.\n\n"
        "- positive_both: both Difference and Ratio have positive latency gain at H=10;\n"
        "- negative_both: both Difference and Ratio have negative latency gain at H=10;\n"
        "- mixed_or_neutral: formulas disagree in sign or one is exactly neutral.\n\n"
        "The same target cohort is retained at H=5,10,20,50 so each line is longitudinally comparable.\n\n"
        "Each plot shows mean cumulative execution time across the fixed cohort. "
        "Gain annotations are computed against the matched No-Transfer baseline for that same cohort.\n",
        encoding="utf-8",
    )

    print()
    print("PASS — positive/negative cohort plots generated")
    print()

    print(
        plot_values.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()

    for p in created:
        print(p)
    print(cohort_path)
    print(values_path)
    print(notes)


if __name__ == "__main__":
    main()
