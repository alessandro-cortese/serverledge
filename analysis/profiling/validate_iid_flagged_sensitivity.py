#!/usr/bin/env python3
"""
Serverledge — follow-up validation for the two real warm series flagged by the IID audit.

This script does NOT alter or rerun the Monte Carlo experiment.

It performs two conservative checks:
1) Monte Carlo permutation test of the observed Spearman trend for the two flagged
   x86 warm traces (200,000 permutations by default).
2) Target-level sensitivity analysis excluding the two flagged functions from the
   already-completed matched-parameter multi-seed Monte Carlo results.

The whole target is excluded (both architectures) in the sensitivity analysis,
which is deliberately conservative.

Outputs
-------
flagged_real_sequences.csv
flagged_permutation_tests.csv
iid_sensitivity_excluding_flagged_summary.csv
iid_sensitivity_excluding_flagged_formula_pairs.csv
IID_FLAGGED_SENSITIVITY_REPORT.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.profiling.transfer_prior_formula_comparison import (
    build_performance,
    load_eligible_durations,
)

FLAGGED = (
    ("billibench-linear-regression", "x86"),
    ("graph-mst-py", "x86"),
)

FORMULA_PAIRS = (
    ("A_ratio_vs_difference", "difference_A", "ratio_A"),
    ("B_ratio_vs_difference", "difference_B", "ratio_B"),
    ("C_ratio_vs_difference", "difference_C", "ratio_C"),
    (
        "frozen_ratio_vs_frozen_difference",
        "difference_B",
        "ratio_C",
    ),
)


def acf_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = x - np.mean(x)
    denom = float(np.dot(y, y))
    if len(x) < 2 or denom == 0:
        return math.nan
    return float(np.dot(y[:-1], y[1:]) / denom)


def permutation_spearman(
        values: np.ndarray,
        *,
        permutations: int,
        seed: int,
        batch_size: int = 5000,
) -> tuple[float, float, int]:
    """
    Two-sided Monte Carlo permutation test for Spearman correlation against
    acquisition order. Uses rank correlation and random permutations of the
    observed values. +1 correction gives a valid Monte Carlo p-value.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    x_rank = rankdata(np.arange(n, dtype=float))
    y_rank = rankdata(values)

    x0 = x_rank - x_rank.mean()
    y0 = y_rank - y_rank.mean()
    denom = math.sqrt(float(np.dot(x0, x0) * np.dot(y0, y0)))
    if denom == 0:
        return math.nan, math.nan, 0

    observed = float(np.dot(x0, y0) / denom)
    rng = np.random.default_rng(seed)

    extreme = 0
    done = 0

    while done < permutations:
        b = min(batch_size, permutations - done)

        # Each row becomes an independent random permutation.
        keys = rng.random((b, n))
        order = np.argsort(keys, axis=1)
        perm_y = y0[order]

        rho = (perm_y @ x0) / denom
        extreme += int(np.sum(np.abs(rho) >= abs(observed) - 1e-15))
        done += b

    p = (extreme + 1.0) / (permutations + 1.0)
    return observed, float(p), extreme


def bootstrap_mean_ci(
        values: np.ndarray,
        *,
        replicates: int,
        seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    inds = rng.integers(
        0,
        len(values),
        size=(replicates, len(values)),
    )
    means = values[inds].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def strategy_summary(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []

    for (strategy, family, formula, horizon), g in df.groupby(
            ["strategy", "family", "formula", "horizon"]
    ):
        rows.append({
            "scope": scope,
            "strategy": strategy,
            "family": family,
            "formula": formula,
            "horizon": int(horizon),
            "target_count": int(g["target_function"].nunique()),
            "w_r": float(g["w_r"].iloc[0]),
            "w_e": float(g["w_e"].iloc[0]),
            "c": float(g["c"].iloc[0]),
            "gain_percent_vs_matched_notl":
                float(g["gain_percent_vs_matched_notl"].mean()),
            "gain_percent_vs_common_c08":
                float(g["gain_percent_vs_common_c08"].mean()),
            "optimal_arm_rate":
                float(g["optimal_arm_rate"].mean()),
            "wrong_selections":
                float(g["wrong_selections"].mean()),
            "cumulative_reward_pseudo_regret":
                float(g["cumulative_reward_pseudo_regret"].mean()),
        })

    return pd.DataFrame(rows)


def formula_pair_summary(
        df: pd.DataFrame,
        *,
        scope: str,
        bootstrap_replicates: int,
        bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []

    for comparison, a_name, b_name in FORMULA_PAIRS:
        for horizon in sorted(df["horizon"].unique()):
            a = df[
                (df["strategy"] == a_name)
                & (df["horizon"] == horizon)
                ].set_index("target_function")
            b = df[
                (df["strategy"] == b_name)
                & (df["horizon"] == horizon)
                ].set_index("target_function")

            targets = sorted(set(a.index) & set(b.index))
            a_lat = a.loc[
                targets, "cumulative_latency_ms"
            ].to_numpy(float)
            b_lat = b.loc[
                targets, "cumulative_latency_ms"
            ].to_numpy(float)

            # Positive = strategy_b (Ratio) lower latency than strategy_a.
            delta = 100.0 * (a_lat - b_lat) / a_lat

            seed = (
                    int(bootstrap_seed)
                    + 1009 * int(horizon)
                    + sum(ord(ch) for ch in comparison)
                    + (0 if scope == "all_57" else 100000)
            )
            point, lo, hi = bootstrap_mean_ci(
                delta,
                replicates=bootstrap_replicates,
                seed=seed,
            )

            rows.append({
                "scope": scope,
                "comparison": comparison,
                "strategy_a": a_name,
                "strategy_b": b_name,
                "horizon": int(horizon),
                "paired_targets": int(len(targets)),
                "ratio_minus_difference_latency_percent":
                    point,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    multiseed = args.multiseed_per_target.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"

    for p in (raw_x, raw_a, multiseed):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    performance = build_performance(
        load_eligible_durations(raw_x),
        load_eligible_durations(raw_a),
    )

    # ------------------------------------------------------------------
    # 1. Inspect the exact ordered warm sequences + permutation tests.
    # ------------------------------------------------------------------
    seq_rows = []
    perm_rows = []

    for i, (target, arm) in enumerate(FLAGGED):
        values = np.asarray(
            performance[target]["durations"][arm],
            dtype=float,
        )

        if len(values) != 12:
            raise SystemExit(
                f"{target}/{arm}: expected 12 observations, got {len(values)}"
            )

        rho_scipy, p_scipy = spearmanr(
            np.arange(len(values)),
            values,
        )

        rho_perm, p_perm, extreme = permutation_spearman(
            values,
            permutations=args.permutations,
            seed=args.permutation_seed + i,
        )

        for order, value in enumerate(values, start=1):
            seq_rows.append({
                "target_function": target,
                "arm": arm,
                "acquisition_order": int(order),
                "duration_ms": float(value),
            })

        perm_rows.append({
            "target_function": target,
            "arm": arm,
            "n": int(len(values)),
            "lag1_acf": acf_lag1(values),
            "spearman_rho": float(rho_scipy),
            "spearman_asymptotic_p": float(p_scipy),
            "permutation_replicates": int(args.permutations),
            "permutation_extreme_count": int(extreme),
            "spearman_permutation_p": float(p_perm),
            "first_duration_ms": float(values[0]),
            "last_duration_ms": float(values[-1]),
            "min_duration_ms": float(values.min()),
            "max_duration_ms": float(values.max()),
        })

    seq_df = pd.DataFrame(seq_rows)
    perm_df = pd.DataFrame(perm_rows)

    seq_path = out / "flagged_real_sequences.csv"
    perm_path = out / "flagged_permutation_tests.csv"
    seq_df.to_csv(seq_path, index=False)
    perm_df.to_csv(perm_path, index=False)

    # ------------------------------------------------------------------
    # 2. Conservative target-level sensitivity: remove entire functions.
    # ------------------------------------------------------------------
    df = pd.read_csv(multiseed)

    # Use only the 57 non-synthetic primary targets.
    primary = df[
        ~df["target_is_synthetic"].astype(bool)
    ].copy()

    flagged_targets = {x[0] for x in FLAGGED}
    reduced = primary[
        ~primary["target_function"].isin(flagged_targets)
    ].copy()

    n_all = primary["target_function"].nunique()
    n_reduced = reduced["target_function"].nunique()

    if n_all != 57 or n_reduced != 55:
        raise SystemExit(
            f"unexpected target counts: all={n_all}, reduced={n_reduced}"
        )

    sum_all = strategy_summary(primary, "all_57")
    sum_red = strategy_summary(
        reduced,
        "exclude_flagged_55",
    )
    summary = pd.concat(
        [sum_all, sum_red],
        ignore_index=True,
    )
    summary_path = (
            out / "iid_sensitivity_excluding_flagged_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    pairs_all = formula_pair_summary(
        primary,
        scope="all_57",
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    pairs_red = formula_pair_summary(
        reduced,
        scope="exclude_flagged_55",
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    pairs = pd.concat(
        [pairs_all, pairs_red],
        ignore_index=True,
    )
    pairs_path = (
            out / "iid_sensitivity_excluding_flagged_formula_pairs.csv"
    )
    pairs.to_csv(pairs_path, index=False)

    # H10 concise console/report table.
    h10_sum = summary[
        (summary["horizon"] == 10)
        & summary["strategy"].isin([
            "difference_A", "ratio_A",
            "difference_B", "ratio_B",
            "difference_C", "ratio_C",
        ])
        ].copy()

    h10_pairs = pairs[
        pairs["horizon"] == 10
        ].copy()

    report = [
        "# IID follow-up: flagged real warm traces and robustness sensitivity",
        "",
        "This analysis follows the structural IID/RNG validation. It does not claim to prove that the physical GCP process is IID.",
        "",
        "## Flagged warm traces",
        "",
        "The original audit identified two x86 series with an acquisition-order Spearman trend surviving Benjamini–Hochberg FDR correction:",
        "",
        perm_df.to_markdown(
            index=False,
            floatfmt=".8f",
        ),
        "",
        "The permutation p-values above are obtained by random permutation of the 12 observed runtime values against acquisition order. They are a small-sample follow-up to the asymptotic Spearman p-values.",
        "",
        "## Conservative robustness sensitivity",
        "",
        "For the sensitivity analysis the two affected **functions** are removed entirely, on both architectures, leaving 55 independent target functions. No Monte Carlo is rerun and no parameter is retuned; the already-computed target-level results are simply re-aggregated.",
        "",
        "### H=10 strategy results",
        "",
        h10_sum[
            [
                "scope",
                "strategy",
                "family",
                "gain_percent_vs_matched_notl",
                "gain_percent_vs_common_c08",
                "optimal_arm_rate",
                "wrong_selections",
                "cumulative_reward_pseudo_regret",
            ]
        ].to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "### H=10 direct formula comparisons",
        "",
        h10_pairs[
            [
                "scope",
                "comparison",
                "paired_targets",
                "ratio_minus_difference_latency_percent",
                "ci95_low",
                "ci95_high",
            ]
        ].to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Interpretation rule",
        "",
        "If the main conclusions remain materially unchanged after excluding these two functions, the Monte Carlo conclusions are not driven by the only real-input series that showed FDR-significant monotonic acquisition-order trends. This is a robustness check, not evidence that the original physical cloud measurements are globally IID.",
        "",
    ]

    report_path = out / "IID_FLAGGED_SENSITIVITY_REPORT.md"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("PASS — flagged-series follow-up completed")

    print()
    print("===== FLAGGED SERIES PERMUTATION TESTS =====")
    print(
        perm_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print()
    print("===== H=10 SENSITIVITY SUMMARY =====")
    print(
        h10_sum[
            [
                "scope",
                "strategy",
                "family",
                "gain_percent_vs_matched_notl",
                "gain_percent_vs_common_c08",
                "optimal_arm_rate",
                "wrong_selections",
                "cumulative_reward_pseudo_regret",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("===== H=10 FORMULA PAIRS =====")
    print(
        h10_pairs[
            [
                "scope",
                "comparison",
                "paired_targets",
                "ratio_minus_difference_latency_percent",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(f"sequences={seq_path}")
    print(f"permutation_tests={perm_path}")
    print(f"sensitivity_summary={summary_path}")
    print(f"formula_pairs={pairs_path}")
    print(f"report={report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--multiseed-per-target",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=200000,
    )
    p.add_argument(
        "--permutation-seed",
        type=int,
        default=2026101216,
    )
    p.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=20000,
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026101217,
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
