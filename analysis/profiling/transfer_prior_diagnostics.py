#!/usr/bin/env python3
"""
Diagnostic experiment for the CURRENT Serverledge UCB1 transfer formula.

Purpose
-------
Separate two possible causes of weak transfer gains:

1. donor-selection quality;
2. the UCB1 transfer formula itself.

The experiment compares, using the same empirical target latency samples and
the same UCB1 implementation:

- selected-kmeans:
    prior means from the LOFO K-Means donor;
- selected-dbscan:
    prior means from the LOFO DBSCAN donor, with no-transfer when DBSCAN abstains;
- oracle:
    prior means equal to the target's own true empirical mean rewards;
- wrong:
    adversarial prior obtained by swapping the target's x86 and ARM mean rewards.

Oracle and wrong are DIAGNOSTIC bounds only. They are not deployable strategies.

No c_t dynamics and no prior reset are introduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    DEFAULT_HORIZONS,
    DEFAULT_WEIGHTS,
    empirical_stats,
    load_raw_durations,
    simulate_policy,
    stable_seed,
)

STRATEGY_ORDER = (
    "selected-kmeans",
    "selected-dbscan",
    "oracle",
    "wrong",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prior_mean_for_strategy(
    strategy: str,
    target_stats: dict[str, Any],
    donor_row: dict[str, str] | None,
    performance: dict[str, dict[str, Any]],
) -> tuple[dict[str, float] | None, str, bool]:
    """
    Return:
      prior_mean_reward, prior_function_label, transfer_applied
    """
    if strategy == "oracle":
        return (
            dict(target_stats["mean_reward"]),
            "TARGET_ORACLE",
            True,
        )

    if strategy == "wrong":
        # Deliberately reverse the target's architecture preference while
        # preserving exactly the same two empirical reward levels.
        return (
            {
                "x86": float(target_stats["mean_reward"]["arm64"]),
                "arm64": float(target_stats["mean_reward"]["x86"]),
            },
            "TARGET_REVERSED",
            True,
        )

    if strategy not in {"selected-kmeans", "selected-dbscan"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if donor_row is None:
        raise ValueError(f"Missing donor row for strategy {strategy}")

    donor = donor_row.get("donor_function", "")
    selected = (
        donor_row.get("selection_status") == "selected"
        and bool(donor)
    )
    if not selected:
        return None, "NO_TRANSFER", False

    return (
        dict(performance[donor]["mean_reward"]),
        donor,
        True,
    )


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = 5000,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        draw = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(draw)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def make_sequences(
    target: str,
    target_stats: dict[str, Any],
    horizon: int,
    replicates: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(stable_seed(seed, target))
    return [
        {
            arm: rng.choice(
                target_stats["durations"][arm],
                size=horizon,
                replace=True,
            )
            for arm in ARMS
        }
        for _ in range(replicates)
    ]


def simulate_with_prior_mean(
    target_stats: dict[str, Any],
    sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    weight: float,
    prior_mean: dict[str, float] | None,
) -> dict[str, np.ndarray]:
    # simulate_policy expects an object with ["mean_reward"] for its prior.
    donor_stats = (
        {"mean_reward": prior_mean}
        if prior_mean is not None
        else None
    )
    return simulate_policy(
        target_stats=target_stats,
        arm_sequences=sequences,
        horizon=horizon,
        c=c,
        prior_weight=(weight if prior_mean is not None else 0.0),
        donor_stats=donor_stats,
    )


def donor_rows_by_target(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for row in rows:
        algorithm = row["algorithm"]
        if algorithm not in {"kmeans", "dbscan"}:
            continue
        result[row["target_function"]][algorithm] = row

    return dict(result)


def simulate_target(
    target: str,
    target_stats: dict[str, Any],
    donor_rows: dict[str, dict[str, str]],
    performance: dict[str, dict[str, Any]],
    strategies: list[str],
    weights: list[float],
    horizons: list[int],
    max_horizon: int,
    replicates: int,
    c: float,
    seed: int,
) -> list[dict[str, Any]]:
    sequences_by_rep = make_sequences(
        target=target,
        target_stats=target_stats,
        horizon=max_horizon,
        replicates=replicates,
        seed=seed,
    )

    baseline = [
        simulate_with_prior_mean(
            target_stats=target_stats,
            sequences=sequences,
            horizon=max_horizon,
            c=c,
            weight=0.0,
            prior_mean=None,
        )
        for sequences in sequences_by_rep
    ]

    output: list[dict[str, Any]] = []

    for strategy in strategies:
        source_algorithm = None
        donor_row = None
        if strategy == "selected-kmeans":
            source_algorithm = "kmeans"
            donor_row = donor_rows["kmeans"]
        elif strategy == "selected-dbscan":
            source_algorithm = "dbscan"
            donor_row = donor_rows["dbscan"]

        prior_mean, prior_label, transfer_applied = prior_mean_for_strategy(
            strategy=strategy,
            target_stats=target_stats,
            donor_row=donor_row,
            performance=performance,
        )

        if prior_mean is not None:
            prior_best_arm = max(
                ARMS,
                key=lambda arm: (
                    prior_mean[arm],
                    -ARMS.index(arm),
                ),
            )
            prior_best_arm_matches_target = (
                prior_best_arm == target_stats["best_reward_arm"]
            )
            prior_reward_gap = (
                float(prior_mean["x86"])
                - float(prior_mean["arm64"])
            )
        else:
            prior_best_arm = ""
            prior_best_arm_matches_target = ""
            prior_reward_gap = ""

        for weight in weights:
            strategy_results = [
                simulate_with_prior_mean(
                    target_stats=target_stats,
                    sequences=sequences_by_rep[i],
                    horizon=max_horizon,
                    c=c,
                    weight=weight,
                    prior_mean=prior_mean,
                )
                for i in range(replicates)
            ]

            for horizon in horizons:
                idx = horizon - 1

                baseline_latency = np.asarray(
                    [
                        r["cumulative_latency"][idx]
                        for r in baseline
                    ],
                    dtype=float,
                )
                strategy_latency = np.asarray(
                    [
                        r["cumulative_latency"][idx]
                        for r in strategy_results
                    ],
                    dtype=float,
                )
                latency_gain = 100.0 * (
                    baseline_latency - strategy_latency
                ) / baseline_latency

                baseline_reward = np.asarray(
                    [
                        r["cumulative_reward"][idx]
                        for r in baseline
                    ],
                    dtype=float,
                )
                strategy_reward = np.asarray(
                    [
                        r["cumulative_reward"][idx]
                        for r in strategy_results
                    ],
                    dtype=float,
                )
                reward_gain = strategy_reward - baseline_reward

                reward_regret = np.asarray(
                    [
                        r["cumulative_reward_regret"][idx]
                        for r in strategy_results
                    ],
                    dtype=float,
                )
                latency_regret = np.asarray(
                    [
                        r["cumulative_latency_regret"][idx]
                        for r in strategy_results
                    ],
                    dtype=float,
                )
                optimal_rate = np.asarray(
                    [
                        r["optimal_reward_pulls"][idx] / horizon
                        for r in strategy_results
                    ],
                    dtype=float,
                )
                first_optimal = np.asarray(
                    [
                        1.0
                        if r["chosen_arms"][0]
                        == target_stats["best_reward_arm"]
                        else 0.0
                        for r in strategy_results
                    ],
                    dtype=float,
                )

                gain_lo, gain_hi = percentile_ci(latency_gain)
                optimal_lo, optimal_hi = percentile_ci(optimal_rate)

                output.append(
                    {
                        "target_function": target,
                        "strategy": strategy,
                        "source_algorithm": source_algorithm or "",
                        "prior_label": prior_label,
                        "transfer_applied": transfer_applied,
                        "equivalent_observation_weight": weight,
                        "c": c,
                        "horizon": horizon,
                        "replicates": replicates,
                        "target_best_reward_arm": target_stats[
                            "best_reward_arm"
                        ],
                        "prior_best_reward_arm": prior_best_arm,
                        "prior_best_arm_matches_target": (
                            prior_best_arm_matches_target
                        ),
                        "target_reward_gap_x86_minus_arm": (
                            float(target_stats["mean_reward"]["x86"])
                            - float(target_stats["mean_reward"]["arm64"])
                        ),
                        "prior_reward_gap_x86_minus_arm": prior_reward_gap,
                        "mean_cumulative_latency_ms": float(
                            np.mean(strategy_latency)
                        ),
                        "mean_baseline_cumulative_latency_ms": float(
                            np.mean(baseline_latency)
                        ),
                        "mean_latency_gain_pct_vs_no_transfer": float(
                            np.mean(latency_gain)
                        ),
                        "median_latency_gain_pct_vs_no_transfer": float(
                            np.median(latency_gain)
                        ),
                        "latency_gain_pct_ci95_low": gain_lo,
                        "latency_gain_pct_ci95_high": gain_hi,
                        "positive_latency_gain_probability": float(
                            np.mean(latency_gain > 0)
                        ),
                        "mean_reward_gain_vs_no_transfer": float(
                            np.mean(reward_gain)
                        ),
                        "mean_reward_pseudo_regret": float(
                            np.mean(reward_regret)
                        ),
                        "mean_latency_pseudo_regret_ms": float(
                            np.mean(latency_regret)
                        ),
                        "mean_optimal_reward_arm_selection_rate": float(
                            np.mean(optimal_rate)
                        ),
                        "optimal_rate_ci95_low": optimal_lo,
                        "optimal_rate_ci95_high": optimal_hi,
                        "mean_suboptimal_pull_count": float(
                            horizon - np.mean(optimal_rate) * horizon
                        ),
                        "first_arm_optimal_probability": float(
                            np.mean(first_optimal)
                        ),
                    }
                )

    return output


def summarize(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    keys = sorted(
        {
            (
                r["strategy"],
                float(r["equivalent_observation_weight"]),
                int(r["horizon"]),
            )
            for r in rows
        },
        key=lambda x: (
            STRATEGY_ORDER.index(x[0]),
            x[1],
            x[2],
        ),
    )

    for strategy, weight, horizon in keys:
        group = [
            r
            for r in rows
            if r["strategy"] == strategy
            and float(r["equivalent_observation_weight"]) == weight
            and int(r["horizon"]) == horizon
        ]
        applied = [r for r in group if r["transfer_applied"]]

        gains_all = np.asarray(
            [
                float(r["mean_latency_gain_pct_vs_no_transfer"])
                for r in group
            ],
            dtype=float,
        )
        gains_applied = np.asarray(
            [
                float(r["mean_latency_gain_pct_vs_no_transfer"])
                for r in applied
            ],
            dtype=float,
        )
        optimal_rate = np.asarray(
            [
                float(r["mean_optimal_reward_arm_selection_rate"])
                for r in group
            ],
            dtype=float,
        )

        all_lo, all_hi = bootstrap_mean_ci(
            gains_all,
            stable_seed(
                seed,
                f"{strategy}-{weight}-{horizon}-all",
            ),
        )
        applied_lo, applied_hi = bootstrap_mean_ci(
            gains_applied,
            stable_seed(
                seed,
                f"{strategy}-{weight}-{horizon}-applied",
            ),
        )

        output.append(
            {
                "strategy": strategy,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(group),
                "transfer_applied_count": len(applied),
                "transfer_coverage": (
                    len(applied) / len(group)
                    if group
                    else float("nan")
                ),
                "prior_best_arm_agreement_rate": float(
                    np.mean(
                        [
                            1.0
                            if r["prior_best_arm_matches_target"] is True
                            else 0.0
                            for r in applied
                        ]
                    )
                )
                if applied
                else float("nan"),
                "macro_mean_latency_gain_pct_all_targets": float(
                    np.mean(gains_all)
                ),
                "macro_median_latency_gain_pct_all_targets": float(
                    np.median(gains_all)
                ),
                "macro_latency_gain_ci95_low_all_targets": all_lo,
                "macro_latency_gain_ci95_high_all_targets": all_hi,
                "target_positive_gain_rate_all_targets": float(
                    np.mean(gains_all > 0)
                ),
                "macro_mean_latency_gain_pct_transfer_applied": (
                    float(np.mean(gains_applied))
                    if len(gains_applied)
                    else float("nan")
                ),
                "macro_latency_gain_ci95_low_transfer_applied": applied_lo,
                "macro_latency_gain_ci95_high_transfer_applied": applied_hi,
                "target_positive_gain_rate_transfer_applied": (
                    float(np.mean(gains_applied > 0))
                    if len(gains_applied)
                    else float("nan")
                ),
                "macro_mean_optimal_reward_arm_selection_rate": float(
                    np.mean(optimal_rate)
                ),
                "macro_first_arm_optimal_probability": float(
                    np.mean(
                        [
                            float(r["first_arm_optimal_probability"])
                            for r in group
                        ]
                    )
                ),
                "macro_mean_reward_pseudo_regret": float(
                    np.mean(
                        [
                            float(r["mean_reward_pseudo_regret"])
                            for r in group
                        ]
                    )
                ),
                "macro_mean_latency_pseudo_regret_ms": float(
                    np.mean(
                        [
                            float(r["mean_latency_pseudo_regret_ms"])
                            for r in group
                        ]
                    )
                ),
            }
        )

    return output


def plot_summary_metric(
    summary: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for strategy in STRATEGY_ORDER:
        for weight in sorted(
            {
                float(r["equivalent_observation_weight"])
                for r in summary
                if r["strategy"] == strategy
            }
        ):
            group = [
                r
                for r in summary
                if r["strategy"] == strategy
                and float(r["equivalent_observation_weight"]) == weight
            ]
            group.sort(key=lambda r: int(r["horizon"]))

            ax.plot(
                [int(r["horizon"]) for r in group],
                [float(r[metric]) for r in group],
                marker="o",
                label=f"{strategy} / w={weight:g}",
            )

    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Requests")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        out_path.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        out_path.with_suffix(".svg"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--donor-selection", required=True, type=Path)
    p.add_argument("--raw-x86", required=True, type=Path)
    p.add_argument("--raw-arm64", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--weight", action="append", type=float)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--replicates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    weights = args.weight or DEFAULT_WEIGHTS
    horizons = [h for h in DEFAULT_HORIZONS if h <= args.horizon]
    if args.horizon not in horizons:
        horizons.append(args.horizon)
        horizons.sort()

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    donor_rows = read_csv(args.donor_selection.resolve())
    donor_map = donor_rows_by_target(donor_rows)

    raw_x86 = load_raw_durations(args.raw_x86.resolve())
    raw_arm64 = load_raw_durations(args.raw_arm64.resolve())
    performance = empirical_stats(raw_x86, raw_arm64)

    expected = set(performance)
    if set(donor_map) != expected:
        raise RuntimeError(
            "Donor-selection/performance mismatch: "
            f"missing_donor_rows={sorted(expected - set(donor_map))}, "
            f"extra_donor_rows={sorted(set(donor_map) - expected)}"
        )

    for target, algorithms in donor_map.items():
        if set(algorithms) != {"kmeans", "dbscan"}:
            raise RuntimeError(
                f"Expected K-Means and DBSCAN rows for {target}, "
                f"found {sorted(algorithms)}"
            )

    rows: list[dict[str, Any]] = []

    for target in sorted(performance):
        rows.extend(
            simulate_target(
                target=target,
                target_stats=performance[target],
                donor_rows=donor_map[target],
                performance=performance,
                strategies=list(STRATEGY_ORDER),
                weights=weights,
                horizons=horizons,
                max_horizon=args.horizon,
                replicates=args.replicates,
                c=args.c,
                seed=args.seed,
            )
        )

    rows.sort(
        key=lambda r: (
            STRATEGY_ORDER.index(r["strategy"]),
            float(r["equivalent_observation_weight"]),
            int(r["horizon"]),
            r["target_function"],
        )
    )
    summary = summarize(rows, seed=args.seed)

    write_csv(out / "ucb1-prior-diagnostics-per-target.csv", rows)
    write_csv(out / "ucb1-prior-diagnostics-summary.csv", summary)

    plot_summary_metric(
        summary,
        "macro_mean_latency_gain_pct_all_targets",
        "Latency gain vs no-transfer (%)",
        "Current UCB1 transfer formula: prior-quality diagnostic",
        figures / "ucb1-prior-diagnostic-latency-gain",
    )
    plot_summary_metric(
        summary,
        "macro_mean_optimal_reward_arm_selection_rate",
        "Optimal-arm selection rate",
        "Current UCB1 transfer formula: optimal-arm selection",
        figures / "ucb1-prior-diagnostic-optimal-arm-rate",
    )
    plot_summary_metric(
        summary,
        "macro_mean_reward_pseudo_regret",
        "Cumulative reward pseudo-regret",
        "Current UCB1 transfer formula: reward pseudo-regret",
        figures / "ucb1-prior-diagnostic-reward-regret",
    )

    manifest = {
        "schema_version": 1,
        "experiment": "UCB1 prior-quality diagnostic",
        "purpose": (
            "Separate donor-selection limitations from limitations of the "
            "current transfer formula."
        ),
        "reward": "-ln(duration_ms)",
        "c": args.c,
        "weights": weights,
        "horizons": horizons,
        "replicates_per_target": args.replicates,
        "seed": args.seed,
        "strategies": {
            "selected-kmeans": (
                "LOFO K-Means donor; no oracle information used for selection"
            ),
            "selected-dbscan": (
                "LOFO DBSCAN donor; no-transfer when DBSCAN abstains"
            ),
            "oracle": (
                "diagnostic upper bound: target's own empirical mean rewards "
                "used as prior"
            ),
            "wrong": (
                "diagnostic adversarial bound: target x86/ARM empirical mean "
                "rewards swapped"
            ),
        },
        "formula": (
            "same current UCB1 transfer formula for every strategy; no "
            "dynamic c_t and no prior reset"
        ),
        "sampling": (
            "paired bootstrap with replacement from the measured eligible "
            "warm target durations; identical empirical arm sequences across "
            "baseline and all diagnostic strategies"
        ),
        "decision_rule": {
            "oracle_good_selected_weak": (
                "formula can exploit good knowledge; donor quality is the "
                "main bottleneck"
            ),
            "oracle_weak": (
                "evidence that the current transfer formula/exploration "
                "interaction should be revised"
            ),
            "wrong_degrades_with_weight": (
                "confirms risk of over-trusting an incorrect donor and "
                "supports weak equivalent-observation weights"
            ),
        },
        "inputs": {
            "donor_selection": str(args.donor_selection.resolve()),
            "raw_x86": str(args.raw_x86.resolve()),
            "raw_arm64": str(args.raw_arm64.resolve()),
        },
        "outputs": [
            "ucb1-prior-diagnostics-per-target.csv",
            "ucb1-prior-diagnostics-summary.csv",
            "figures/ucb1-prior-diagnostic-latency-gain.{png,svg}",
            "figures/ucb1-prior-diagnostic-optimal-arm-rate.{png,svg}",
            "figures/ucb1-prior-diagnostic-reward-regret.{png,svg}",
        ],
    }
    (out / "ucb1-prior-diagnostics-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"targets={len(performance)} "
        f"strategies={','.join(STRATEGY_ORDER)} "
        f"weights={','.join(f'{w:g}' for w in weights)} "
        f"horizon={args.horizon} "
        f"replicates={args.replicates} "
        f"output={out}"
    )

    primary_horizon = min(10, args.horizon)
    print(f"\nDiagnostic summary @ H={primary_horizon}:")
    for row in summary:
        if int(row["horizon"]) != primary_horizon:
            continue
        print(
            f"{row['strategy']:<17} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"coverage={float(row['transfer_coverage']):.4f} "
            f"prior-best-arm-match="
            f"{float(row['prior_best_arm_agreement_rate']):.4f} "
            f"latency-gain="
            f"{float(row['macro_mean_latency_gain_pct_all_targets']):+.4f}% "
            f"CI95=["
            f"{float(row['macro_latency_gain_ci95_low_all_targets']):+.4f},"
            f"{float(row['macro_latency_gain_ci95_high_all_targets']):+.4f}"
            f"] "
            f"optimal-rate="
            f"{float(row['macro_mean_optimal_reward_arm_selection_rate']):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
