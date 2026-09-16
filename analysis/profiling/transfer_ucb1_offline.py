#!/usr/bin/env python3
"""
Offline replay/bootstrap evaluation of Serverledge UCB1 transfer learning.

Uses donor selections produced by transfer_donor_lofo.py and the measured warm
duration samples of each target function on x86 and ARM.

The UCB1 transfer formula mirrors the documented Serverledge implementation:
  effective_mean_j = (live_reward_sum_j + w * donor_mean_reward_j)
                     / (live_count_j + w)

  exploration_j = 0                               if effective_total <= 1
                  c * sqrt(log(effective_total) /
                           (live_count_j + w))     otherwise

The no-transfer baseline uses classic UCB1 cold start: unseen arms are pulled
before score-based selection.

No dynamic c_t and no prior reset are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_WEIGHTS = [0.25, 0.50, 1.00]
DEFAULT_HORIZONS = [1, 2, 5, 10, 20, 50]
ARMS = ("x86", "arm64")


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


def sample_is_eligible(sample: dict[str, Any]) -> bool:
    profile = sample.get("profile") or {}
    eligibility = sample.get("eligibility") or {}
    config = sample.get("function_configuration") or {}

    return (
        sample.get("warm_start") is True
        and sample.get("execution_succeeded") is True
        and profile.get("valid") is True
        and profile.get("exclusive_container") is True
        and eligibility.get("resource_clustering") is True
        and eligibility.get("performance_analysis") is True
        and abs(float(config.get("configured_cpus", 0.0)) - 1.0) <= 1e-9
    )


def load_raw_durations(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not sample_is_eligible(sample):
                continue

            duration_ms = float((sample.get("timing") or {})["duration_ms"])
            if not math.isfinite(duration_ms) or duration_ms <= 0:
                raise ValueError(
                    f"Invalid duration at {path}:{line_no}: {duration_ms}"
                )
            out[sample["function_name"]].append(duration_ms)

    return dict(out)


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    suffix = int.from_bytes(digest[:8], "big")
    return (base_seed + suffix) % (2**63 - 1)


def empirical_stats(
    x86: dict[str, list[float]],
    arm: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = sorted(set(x86) & set(arm))

    for name in names:
        x = np.asarray(x86[name], dtype=float)
        a = np.asarray(arm[name], dtype=float)
        x_rewards = -np.log(x)
        a_rewards = -np.log(a)

        mean_rewards = {
            "x86": float(np.mean(x_rewards)),
            "arm64": float(np.mean(a_rewards)),
        }
        mean_durations = {
            "x86": float(np.mean(x)),
            "arm64": float(np.mean(a)),
        }

        result[name] = {
            "durations": {
                "x86": x,
                "arm64": a,
            },
            "mean_reward": mean_rewards,
            "mean_duration": mean_durations,
            "best_reward_arm": max(
                ARMS,
                key=lambda arm_name: (
                    mean_rewards[arm_name],
                    -ARMS.index(arm_name),
                ),
            ),
            "best_latency_arm": min(
                ARMS,
                key=lambda arm_name: (
                    mean_durations[arm_name],
                    ARMS.index(arm_name),
                ),
            ),
        }

    return result


class UCB1Policy:
    def __init__(
        self,
        c: float,
        prior_weight: float = 0.0,
        prior_mean_reward: dict[str, float] | None = None,
    ):
        self.c = float(c)
        self.counts = {arm: 0 for arm in ARMS}
        self.reward_sums = {arm: 0.0 for arm in ARMS}
        self.prior_weights = {
            arm: (
                float(prior_weight)
                if prior_mean_reward is not None
                else 0.0
            )
            for arm in ARMS
        }
        self.prior_reward_sums = {
            arm: (
                float(prior_weight) * float(prior_mean_reward[arm])
                if prior_mean_reward is not None
                else 0.0
            )
            for arm in ARMS
        }

    def effective_count(self, arm: str) -> float:
        return self.counts[arm] + self.prior_weights[arm]

    def effective_total(self) -> float:
        return sum(self.effective_count(arm) for arm in ARMS)

    def score(self, arm: str) -> float:
        effective_count = self.effective_count(arm)
        if effective_count <= 0:
            return float("inf")

        mean = (
            self.reward_sums[arm] + self.prior_reward_sums[arm]
        ) / effective_count

        total = self.effective_total()
        if total <= 1.0:
            exploration = 0.0
        else:
            exploration = self.c * math.sqrt(
                math.log(total) / effective_count
            )
        return mean + exploration

    def select_arm(self) -> str:
        unseen = [
            arm
            for arm in ARMS
            if self.effective_count(arm) <= 0
        ]
        if unseen:
            return unseen[0]

        scores = {arm: self.score(arm) for arm in ARMS}
        return max(
            ARMS,
            key=lambda arm: (
                scores[arm],
                -ARMS.index(arm),
            ),
        )

    def update(self, arm: str, duration_ms: float) -> float:
        reward = -math.log(duration_ms)
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        return reward


def simulate_policy(
    target_stats: dict[str, Any],
    arm_sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    prior_weight: float,
    donor_stats: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    prior_mean = None
    if donor_stats is not None:
        prior_mean = donor_stats["mean_reward"]

    policy = UCB1Policy(
        c=c,
        prior_weight=prior_weight,
        prior_mean_reward=prior_mean,
    )

    arm_positions = {arm: 0 for arm in ARMS}
    cumulative_latency = np.zeros(horizon, dtype=float)
    cumulative_reward = np.zeros(horizon, dtype=float)
    cumulative_reward_regret = np.zeros(horizon, dtype=float)
    cumulative_latency_regret = np.zeros(horizon, dtype=float)
    optimal_reward_pulls = np.zeros(horizon, dtype=float)

    best_reward_arm = target_stats["best_reward_arm"]
    best_reward = target_stats["mean_reward"][best_reward_arm]
    best_latency = min(target_stats["mean_duration"].values())

    total_latency = 0.0
    total_reward = 0.0
    total_reward_regret = 0.0
    total_latency_regret = 0.0
    optimal_count = 0

    chosen_arms: list[str] = []

    for t in range(horizon):
        arm = policy.select_arm()
        position = arm_positions[arm]
        duration = float(arm_sequences[arm][position])
        arm_positions[arm] += 1

        reward = policy.update(arm, duration)
        chosen_arms.append(arm)

        total_latency += duration
        total_reward += reward
        total_reward_regret += (
            best_reward - target_stats["mean_reward"][arm]
        )
        total_latency_regret += (
            target_stats["mean_duration"][arm] - best_latency
        )
        if arm == best_reward_arm:
            optimal_count += 1

        cumulative_latency[t] = total_latency
        cumulative_reward[t] = total_reward
        cumulative_reward_regret[t] = total_reward_regret
        cumulative_latency_regret[t] = total_latency_regret
        optimal_reward_pulls[t] = optimal_count

    return {
        "cumulative_latency": cumulative_latency,
        "cumulative_reward": cumulative_reward,
        "cumulative_reward_regret": cumulative_reward_regret,
        "cumulative_latency_regret": cumulative_latency_regret,
        "optimal_reward_pulls": optimal_reward_pulls,
        "chosen_arms": np.asarray(chosen_arms, dtype=object),
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def simulate_target(
    target: str,
    target_stats: dict[str, Any],
    donor_rows: list[dict[str, str]],
    performance: dict[str, dict[str, Any]],
    weights: list[float],
    horizons: list[int],
    horizon: int,
    replicates: int,
    c: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(stable_seed(seed, target))

    sequences_by_rep: list[dict[str, np.ndarray]] = []
    for _ in range(replicates):
        sequences_by_rep.append(
            {
                arm: rng.choice(
                    target_stats["durations"][arm],
                    size=horizon,
                    replace=True,
                )
                for arm in ARMS
            }
        )

    baseline_results = [
        simulate_policy(
            target_stats,
            sequences,
            horizon=horizon,
            c=c,
            prior_weight=0.0,
            donor_stats=None,
        )
        for sequences in sequences_by_rep
    ]

    output: list[dict[str, Any]] = []

    for donor_row in donor_rows:
        algorithm = donor_row["algorithm"]
        donor = donor_row.get("donor_function", "")
        selected = donor_row["selection_status"] == "selected" and bool(donor)

        for weight in weights:
            donor_stats = performance[donor] if selected else None
            effective_weight = weight if selected else 0.0

            strategy_results = [
                simulate_policy(
                    target_stats,
                    sequences_by_rep[i],
                    horizon=horizon,
                    c=c,
                    prior_weight=effective_weight,
                    donor_stats=donor_stats,
                )
                for i in range(replicates)
            ]

            for h in horizons:
                idx = h - 1

                baseline_latency = np.asarray(
                    [r["cumulative_latency"][idx] for r in baseline_results],
                    dtype=float,
                )
                strategy_latency = np.asarray(
                    [r["cumulative_latency"][idx] for r in strategy_results],
                    dtype=float,
                )
                latency_gain_pct = 100.0 * (
                    baseline_latency - strategy_latency
                ) / baseline_latency

                strategy_reward = np.asarray(
                    [r["cumulative_reward"][idx] for r in strategy_results],
                    dtype=float,
                )
                baseline_reward = np.asarray(
                    [r["cumulative_reward"][idx] for r in baseline_results],
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
                        r["optimal_reward_pulls"][idx] / h
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

                gain_lo, gain_hi = percentile_ci(latency_gain_pct)
                opt_lo, opt_hi = percentile_ci(optimal_rate)

                output.append(
                    {
                        "target_function": target,
                        "algorithm": algorithm,
                        "donor_function": donor,
                        "transfer_applied": selected,
                        "equivalent_observation_weight": weight,
                        "c": c,
                        "horizon": h,
                        "replicates": replicates,
                        "target_best_reward_arm": target_stats[
                            "best_reward_arm"
                        ],
                        "target_best_latency_arm": target_stats[
                            "best_latency_arm"
                        ],
                        "mean_cumulative_latency_ms": float(
                            np.mean(strategy_latency)
                        ),
                        "mean_baseline_cumulative_latency_ms": float(
                            np.mean(baseline_latency)
                        ),
                        "mean_latency_gain_pct_vs_no_transfer": float(
                            np.mean(latency_gain_pct)
                        ),
                        "median_latency_gain_pct_vs_no_transfer": float(
                            np.median(latency_gain_pct)
                        ),
                        "latency_gain_pct_ci95_low": gain_lo,
                        "latency_gain_pct_ci95_high": gain_hi,
                        "positive_latency_gain_probability": float(
                            np.mean(latency_gain_pct > 0)
                        ),
                        "mean_cumulative_reward": float(
                            np.mean(strategy_reward)
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
                        "optimal_rate_ci95_low": opt_lo,
                        "optimal_rate_ci95_high": opt_hi,
                        "mean_suboptimal_pull_count": float(
                            h - np.mean(optimal_rate) * h
                        ),
                        "first_arm_optimal_probability": float(
                            np.mean(first_optimal)
                        ),
                    }
                )

    return output


def bootstrap_ci_across_targets(
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


def summarize(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    keys = sorted(
        {
            (
                r["algorithm"],
                float(r["equivalent_observation_weight"]),
                int(r["horizon"]),
            )
            for r in rows
        }
    )

    for algorithm, weight, horizon in keys:
        group = [
            r
            for r in rows
            if r["algorithm"] == algorithm
            and float(r["equivalent_observation_weight"]) == weight
            and int(r["horizon"]) == horizon
        ]
        transferred = [r for r in group if r["transfer_applied"]]

        gains_all = np.asarray(
            [r["mean_latency_gain_pct_vs_no_transfer"] for r in group],
            dtype=float,
        )
        gains_selected = np.asarray(
            [
                r["mean_latency_gain_pct_vs_no_transfer"]
                for r in transferred
            ],
            dtype=float,
        )
        optimal_all = np.asarray(
            [
                r["mean_optimal_reward_arm_selection_rate"]
                for r in group
            ],
            dtype=float,
        )
        reward_regret = np.asarray(
            [r["mean_reward_pseudo_regret"] for r in group],
            dtype=float,
        )
        latency_regret = np.asarray(
            [r["mean_latency_pseudo_regret_ms"] for r in group],
            dtype=float,
        )

        ci_low, ci_high = bootstrap_ci_across_targets(
            gains_all,
            stable_seed(
                seed,
                f"{algorithm}-{weight}-{horizon}-all",
            ),
        )

        selected_ci_low, selected_ci_high = bootstrap_ci_across_targets(
            gains_selected,
            stable_seed(
                seed,
                f"{algorithm}-{weight}-{horizon}-selected",
            ),
        )

        output.append(
            {
                "algorithm": algorithm,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(group),
                "transferred_target_count": len(transferred),
                "transfer_coverage": (
                    len(transferred) / len(group)
                    if group
                    else float("nan")
                ),
                "macro_mean_latency_gain_pct_all_targets": float(
                    np.mean(gains_all)
                ),
                "macro_median_latency_gain_pct_all_targets": float(
                    np.median(gains_all)
                ),
                "macro_latency_gain_ci95_low_all_targets": ci_low,
                "macro_latency_gain_ci95_high_all_targets": ci_high,
                "target_positive_gain_rate_all_targets": float(
                    np.mean(gains_all > 0)
                ),
                "macro_mean_latency_gain_pct_transferred_only": (
                    float(np.mean(gains_selected))
                    if len(gains_selected)
                    else float("nan")
                ),
                "macro_latency_gain_ci95_low_transferred_only": selected_ci_low,
                "macro_latency_gain_ci95_high_transferred_only": selected_ci_high,
                "target_positive_gain_rate_transferred_only": (
                    float(np.mean(gains_selected > 0))
                    if len(gains_selected)
                    else float("nan")
                ),
                "macro_mean_optimal_reward_arm_selection_rate": float(
                    np.mean(optimal_all)
                ),
                "macro_mean_reward_pseudo_regret": float(
                    np.mean(reward_regret)
                ),
                "macro_mean_latency_pseudo_regret_ms": float(
                    np.mean(latency_regret)
                ),
                "macro_first_arm_optimal_probability": float(
                    np.mean(
                        [
                            r["first_arm_optimal_probability"]
                            for r in group
                        ]
                    )
                ),
            }
        )

    return output


def plot_metric(
    summary: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))

    algorithms = sorted({r["algorithm"] for r in summary})
    weights = sorted(
        {float(r["equivalent_observation_weight"]) for r in summary}
    )

    for algorithm in algorithms:
        for weight in weights:
            rows = [
                r
                for r in summary
                if r["algorithm"] == algorithm
                and float(r["equivalent_observation_weight"]) == weight
            ]
            rows.sort(key=lambda r: int(r["horizon"]))
            ax.plot(
                [int(r["horizon"]) for r in rows],
                [float(r[metric]) for r in rows],
                marker="o",
                label=f"{algorithm} / w={weight:g}",
            )

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
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
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
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in donor_rows:
        by_target[row["target_function"]].append(row)

    raw_x86 = load_raw_durations(args.raw_x86.resolve())
    raw_arm = load_raw_durations(args.raw_arm64.resolve())
    performance = empirical_stats(raw_x86, raw_arm)

    expected_targets = set(by_target)
    if expected_targets != set(performance):
        raise RuntimeError(
            "Donor-selection/raw-performance target mismatch: "
            f"missing_performance="
            f"{sorted(expected_targets - set(performance))}, "
            f"extra_performance="
            f"{sorted(set(performance) - expected_targets)}"
        )

    rows: list[dict[str, Any]] = []
    for target in sorted(by_target):
        rows.extend(
            simulate_target(
                target=target,
                target_stats=performance[target],
                donor_rows=by_target[target],
                performance=performance,
                weights=weights,
                horizons=horizons,
                horizon=args.horizon,
                replicates=args.replicates,
                c=args.c,
                seed=args.seed,
            )
        )

    rows.sort(
        key=lambda r: (
            r["algorithm"],
            float(r["equivalent_observation_weight"]),
            int(r["horizon"]),
            r["target_function"],
        )
    )
    summary = summarize(rows, seed=args.seed)

    write_csv(out / "ucb1-transfer-per-target.csv", rows)
    write_csv(out / "ucb1-transfer-summary.csv", summary)

    plot_metric(
        summary,
        "macro_mean_latency_gain_pct_all_targets",
        "Latency gain vs no transfer (%)",
        "UCB1 transfer: cumulative latency gain",
        figures / "ucb1-transfer-latency-gain",
    )
    plot_metric(
        summary,
        "macro_mean_optimal_reward_arm_selection_rate",
        "Optimal-arm selection rate",
        "UCB1 transfer: optimal-arm selection",
        figures / "ucb1-transfer-optimal-arm-rate",
    )
    plot_metric(
        summary,
        "macro_mean_reward_pseudo_regret",
        "Cumulative reward pseudo-regret",
        "UCB1 transfer: reward pseudo-regret",
        figures / "ucb1-transfer-reward-regret",
    )

    manifest = {
        "schema_version": 1,
        "experiment": "offline empirical-bootstrap UCB1 transfer replay",
        "reward": "-ln(duration_ms)",
        "c": args.c,
        "weights": weights,
        "horizons": horizons,
        "maximum_horizon": args.horizon,
        "replicates_per_target": args.replicates,
        "seed": args.seed,
        "sampling": (
            "bootstrap with replacement from the 12 eligible measured warm "
            "durations per target/architecture; paired common random numbers "
            "across baseline and transfer strategies"
        ),
        "baseline": (
            "no-transfer classic UCB1; unseen arms are selected before UCB "
            "scoring"
        ),
        "transfer": (
            "donor means are weak pseudo-observations; prior weight enters "
            "both exploitation mean and effective exploration counts"
        ),
        "exploration_edge_case": (
            "exploration bonus is zero while effective_total_count <= 1, "
            "matching observed Serverledge prior behavior"
        ),
        "dynamic_c": False,
        "prior_reset": False,
        "metrics": [
            "paired cumulative latency gain vs no transfer",
            "cumulative reward gain",
            "reward pseudo-regret",
            "latency pseudo-regret",
            "optimal reward arm selection rate",
            "suboptimal pull count",
            "first-arm optimal probability",
            "transfer coverage",
        ],
        "inputs": {
            "donor_selection": str(args.donor_selection.resolve()),
            "raw_x86": str(args.raw_x86.resolve()),
            "raw_arm64": str(args.raw_arm64.resolve()),
        },
        "outputs": [
            "ucb1-transfer-per-target.csv",
            "ucb1-transfer-summary.csv",
            "figures/ucb1-transfer-latency-gain.{png,svg}",
            "figures/ucb1-transfer-optimal-arm-rate.{png,svg}",
            "figures/ucb1-transfer-reward-regret.{png,svg}",
        ],
    }
    (out / "ucb1-transfer-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"targets={len(by_target)} "
        f"weights={','.join(f'{w:g}' for w in weights)} "
        f"horizon={args.horizon} "
        f"replicates={args.replicates} "
        f"output={out}"
    )

    primary_horizon = min(10, args.horizon)
    print(f"\nSummary @ H={primary_horizon}:")
    for row in summary:
        if int(row["horizon"]) != primary_horizon:
            continue
        print(
            f"{row['algorithm']} w="
            f"{float(row['equivalent_observation_weight']):g}: "
            f"coverage={float(row['transfer_coverage']):.4f} "
            f"latency_gain="
            f"{float(row['macro_mean_latency_gain_pct_all_targets']):+.4f}% "
            f"optimal_rate="
            f"{float(row['macro_mean_optimal_reward_arm_selection_rate']):.4f} "
            f"positive_target_rate="
            f"{float(row['target_positive_gain_rate_all_targets']):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
