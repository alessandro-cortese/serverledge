#!/usr/bin/env python3
"""
Decoupled weak-prior UCB1 study for Serverledge.

Purpose
-------
The current transfer formulation uses one equivalent-observation weight w for
two conceptually different roles:

  1. exploitation trust:
       how strongly the donor prior contributes to the effective reward mean;

  2. exploration pseudo-count:
       how strongly the prior makes an arm look already explored.

This experiment separates them:

  w_R = reward / exploitation prior weight
  w_E = exploration pseudo-count weight

Effective exploitation mean:
    mu_eff_j =
        (S_j + w_R * mu_prior_j)
        / (n_j + w_R)

Exploration bonus:
    B_j = 0
          if N + sum_k(w_E,k) <= 1

          c * sqrt(
              ln(N + sum_k(w_E,k))
              / (n_j + w_E)
          )
          otherwise

Score:
    UCB_j = mu_eff_j + B_j

For w_R == w_E the implementation is exactly the current Serverledge transfer
formula used by the existing offline simulator. The unit tests explicitly
check this diagonal equivalence.

Experimental isolation
----------------------
This study freezes the best-motivated upstream choices from the previous local
experiments:

  clustering  : K-Means PAPER-5
  donor       : Manhattan-ranked donor INSIDE the assigned cluster
  prior       : reference-anchored architecture-effect prior
  reward      : -ln(duration_ms)
  c           : 0.8

Only the relationship between w_R and w_E changes.

The target's x86 performance is available because a new function is profiled
on the reference architecture before transfer. Target ARM observations are
used only for post-hoc offline evaluation.

Grid
----
By default:
    w_R in {0.25, 0.50, 1.00}
    w_E in {0.25, 0.50, 1.00}

The diagonal:
    (0.25,0.25), (0.50,0.50), (1.00,1.00)
is the CURRENT formula.

Off-diagonal combinations are the DECOUPLED variants.

Convergence
-----------
UCB1 continues deliberate exploration, therefore convergence is measured on
the exploitation estimate.

A run converges at the completed request that closes the first run of N
consecutive updates for which the exploitation-only effective mean ranks the
empirical best architecture correctly.

The script reports:
- cumulative latency gain vs no-transfer;
- wrong selections saved;
- wrong selections before convergence saved;
- convergence probability;
- censored requests-to-convergence (H+1 for non-converged runs);
- requests-to-convergence saved;
- first-arm optimal probability;
- reward pseudo-regret.

These metrics map directly to the future GCP validation, where the same
requests-to-convergence criterion should additionally be measured in wall-clock
seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    DEFAULT_HORIZONS,
    UCB1Policy,
    empirical_stats,
    load_raw_durations,
    stable_seed,
)

DEFAULT_REWARD_WEIGHTS = [0.25, 0.50, 1.00]
DEFAULT_EXPLORATION_WEIGHTS = [0.25, 0.50, 1.00]


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


def build_reference_anchored_prior(
    target_stats: dict[str, Any],
    donor_stats: dict[str, Any],
) -> dict[str, float]:
    target_x86 = float(target_stats["mean_reward"]["x86"])
    donor_x86 = float(donor_stats["mean_reward"]["x86"])
    donor_arm = float(donor_stats["mean_reward"]["arm64"])

    donor_arch_effect = donor_arm - donor_x86

    return {
        "x86": target_x86,
        "arm64": target_x86 + donor_arch_effect,
    }


class DecoupledUCB1Policy:
    """
    Weak-prior UCB1 with independent exploitation and exploration weights.

    When reward_prior_weight == exploration_prior_weight, score(), select_arm()
    and update() are intentionally equivalent to the existing UCB1Policy.
    """

    def __init__(
        self,
        c: float,
        reward_prior_weight: float,
        exploration_prior_weight: float,
        prior_mean_reward: dict[str, float],
    ):
        if reward_prior_weight <= 0:
            raise ValueError(
                "reward_prior_weight must be > 0 for transfer"
            )
        if exploration_prior_weight <= 0:
            raise ValueError(
                "exploration_prior_weight must be > 0 for transfer"
            )

        self.c = float(c)
        self.reward_prior_weight = float(reward_prior_weight)
        self.exploration_prior_weight = float(
            exploration_prior_weight
        )

        self.counts = {arm: 0 for arm in ARMS}
        self.reward_sums = {arm: 0.0 for arm in ARMS}

        self.prior_reward_sums = {
            arm: (
                self.reward_prior_weight
                * float(prior_mean_reward[arm])
            )
            for arm in ARMS
        }

    def exploitation_count(self, arm: str) -> float:
        return (
            self.counts[arm]
            + self.reward_prior_weight
        )

    def exploration_count(self, arm: str) -> float:
        return (
            self.counts[arm]
            + self.exploration_prior_weight
        )

    def exploration_total(self) -> float:
        return sum(
            self.exploration_count(arm)
            for arm in ARMS
        )

    def effective_mean(self, arm: str) -> float:
        count = self.exploitation_count(arm)
        return float(
            (
                self.reward_sums[arm]
                + self.prior_reward_sums[arm]
            )
            / count
        )

    def exploration_bonus(self, arm: str) -> float:
        total = self.exploration_total()
        if total <= 1.0:
            return 0.0

        return float(
            self.c
            * math.sqrt(
                math.log(total)
                / self.exploration_count(arm)
            )
        )

    def score(self, arm: str) -> float:
        return (
            self.effective_mean(arm)
            + self.exploration_bonus(arm)
        )

    def select_arm(self) -> str:
        scores = {
            arm: self.score(arm)
            for arm in ARMS
        }
        return max(
            ARMS,
            key=lambda arm: (
                scores[arm],
                -ARMS.index(arm),
            ),
        )

    def update(
        self,
        arm: str,
        duration_ms: float,
    ) -> float:
        reward = -math.log(duration_ms)
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        return reward


def first_completed_true_run(
    values: list[bool],
    run_length: int,
) -> int | None:
    if run_length <= 0:
        raise ValueError(
            "run_length must be positive"
        )

    streak = 0
    for idx, value in enumerate(
        values,
        start=1,
    ):
        if value:
            streak += 1
            if streak >= run_length:
                return idx
        else:
            streak = 0

    return None


def make_sequences(
    target_name: str,
    target_stats: dict[str, Any],
    horizon: int,
    replicates: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(
        stable_seed(seed, target_name)
    )

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


def exploitation_arm_current(
    policy: UCB1Policy,
) -> str | None:
    means: dict[str, float] = {}

    for arm in ARMS:
        count = policy.effective_count(arm)
        if count <= 0:
            return None

        means[arm] = float(
            (
                policy.reward_sums[arm]
                + policy.prior_reward_sums[arm]
            )
            / count
        )

    return max(
        ARMS,
        key=lambda arm: (
            means[arm],
            -ARMS.index(arm),
        ),
    )


def exploitation_arm_decoupled(
    policy: DecoupledUCB1Policy,
) -> str:
    means = {
        arm: policy.effective_mean(arm)
        for arm in ARMS
    }

    return max(
        ARMS,
        key=lambda arm: (
            means[arm],
            -ARMS.index(arm),
        ),
    )


def simulate_baseline_trace(
    target_stats: dict[str, Any],
    sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    convergence_run: int,
) -> dict[str, Any]:
    policy = UCB1Policy(
        c=c,
        prior_weight=0.0,
        prior_mean_reward=None,
    )

    positions = {arm: 0 for arm in ARMS}
    chosen: list[str] = []
    cumulative_latency = np.zeros(
        horizon,
        dtype=float,
    )
    cumulative_reward_regret = np.zeros(
        horizon,
        dtype=float,
    )
    exploitation_correct: list[bool] = []

    best_arm = target_stats["best_reward_arm"]
    best_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    latency_sum = 0.0
    regret_sum = 0.0

    for t in range(horizon):
        arm = policy.select_arm()
        duration = float(
            sequences[arm][positions[arm]]
        )
        positions[arm] += 1

        policy.update(
            arm,
            duration,
        )

        chosen.append(arm)

        latency_sum += duration
        cumulative_latency[t] = latency_sum

        regret_sum += (
            best_reward
            - float(
                target_stats[
                    "mean_reward"
                ][arm]
            )
        )
        cumulative_reward_regret[t] = regret_sum

        exploit = exploitation_arm_current(
            policy
        )
        exploitation_correct.append(
            exploit == best_arm
            if exploit is not None
            else False
        )

    return {
        "chosen_arms": chosen,
        "cumulative_latency": cumulative_latency,
        "cumulative_reward_regret": (
            cumulative_reward_regret
        ),
        "exploitation_correct": (
            exploitation_correct
        ),
    }


def simulate_transfer_trace(
    target_stats: dict[str, Any],
    prior_mean_reward: dict[str, float],
    sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    reward_prior_weight: float,
    exploration_prior_weight: float,
    convergence_run: int,
) -> dict[str, Any]:
    policy = DecoupledUCB1Policy(
        c=c,
        reward_prior_weight=reward_prior_weight,
        exploration_prior_weight=exploration_prior_weight,
        prior_mean_reward=prior_mean_reward,
    )

    positions = {arm: 0 for arm in ARMS}
    chosen: list[str] = []
    cumulative_latency = np.zeros(
        horizon,
        dtype=float,
    )
    cumulative_reward_regret = np.zeros(
        horizon,
        dtype=float,
    )
    exploitation_correct: list[bool] = []

    best_arm = target_stats["best_reward_arm"]
    best_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    latency_sum = 0.0
    regret_sum = 0.0

    for t in range(horizon):
        arm = policy.select_arm()

        duration = float(
            sequences[arm][positions[arm]]
        )
        positions[arm] += 1

        policy.update(
            arm,
            duration,
        )

        chosen.append(arm)

        latency_sum += duration
        cumulative_latency[t] = latency_sum

        regret_sum += (
            best_reward
            - float(
                target_stats[
                    "mean_reward"
                ][arm]
            )
        )
        cumulative_reward_regret[t] = regret_sum

        exploit = exploitation_arm_decoupled(
            policy
        )
        exploitation_correct.append(
            exploit == best_arm
        )

    return {
        "chosen_arms": chosen,
        "cumulative_latency": cumulative_latency,
        "cumulative_reward_regret": (
            cumulative_reward_regret
        ),
        "exploitation_correct": (
            exploitation_correct
        ),
    }


def trace_metrics_at_horizon(
    trace: dict[str, Any],
    target_stats: dict[str, Any],
    horizon: int,
    convergence_run: int,
) -> dict[str, float]:
    best_arm = target_stats["best_reward_arm"]

    chosen = trace["chosen_arms"][:horizon]
    exploitation = trace[
        "exploitation_correct"
    ][:horizon]

    wrong_count = sum(
        1
        for arm in chosen
        if arm != best_arm
    )

    convergence_request = (
        first_completed_true_run(
            exploitation,
            convergence_run,
        )
    )

    censored_convergence = (
        convergence_request
        if convergence_request is not None
        else horizon + 1
    )

    if convergence_request is None:
        wrong_before = wrong_count
    else:
        wrong_before = sum(
            1
            for arm in chosen[
                :convergence_request
            ]
            if arm != best_arm
        )

    return {
        "cumulative_latency": float(
            trace["cumulative_latency"][
                horizon - 1
            ]
        ),
        "reward_pseudo_regret": float(
            trace[
                "cumulative_reward_regret"
            ][horizon - 1]
        ),
        "wrong_count": float(wrong_count),
        "wrong_before_convergence": float(
            wrong_before
        ),
        "converged": (
            1.0
            if convergence_request is not None
            else 0.0
        ),
        "censored_convergence_request": float(
            censored_convergence
        ),
        "convergence_request_if_converged": (
            float(convergence_request)
            if convergence_request
            is not None
            else float("nan")
        ),
        "first_arm_optimal": (
            1.0
            if chosen[0] == best_arm
            else 0.0
        ),
    }


def load_manhattan_donors(
    path: Path,
    algorithm: str,
    ranker: str,
) -> dict[str, str]:
    rows = read_csv(path)
    result: dict[str, str] = {}

    for row in rows:
        if row["algorithm"] != algorithm:
            continue
        if row["ranker"] != ranker:
            continue

        if row["selection_status"] != "selected":
            raise RuntimeError(
                f"{algorithm}/{ranker}/"
                f"{row['target_function']} "
                "has no selected donor"
            )

        target = row["target_function"]
        if target in result:
            raise RuntimeError(
                f"Duplicate target donor: {target}"
            )

        result[target] = row[
            "donor_function"
        ]

    return result


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = 5000,
) -> tuple[float, float]:
    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    means = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):
        means[i] = np.mean(
            rng.choice(
                values,
                size=len(values),
                replace=True,
            )
        )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def summarize(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    keys = sorted(
        {
            (
                float(
                    row[
                        "reward_prior_weight"
                    ]
                ),
                float(
                    row[
                        "exploration_prior_weight"
                    ]
                ),
                int(row["horizon"]),
            )
            for row in rows
        }
    )

    for w_r, w_e, horizon in keys:
        group = [
            row
            for row in rows
            if float(
                row["reward_prior_weight"]
            )
            == w_r
            and float(
                row[
                    "exploration_prior_weight"
                ]
            )
            == w_e
            and int(row["horizon"])
            == horizon
        ]

        gains = np.asarray(
            [
                float(
                    row[
                        "mean_latency_gain_pct"
                    ]
                )
                for row in group
            ],
            dtype=float,
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                gains,
                stable_seed(
                    seed,
                    f"decoupled-{w_r}-{w_e}-{horizon}",
                ),
            )
        )

        output.append(
            {
                "formula_mode": (
                    "current-diagonal"
                    if abs(w_r - w_e)
                    <= 1e-12
                    else "decoupled"
                ),
                "reward_prior_weight": w_r,
                "exploration_prior_weight": w_e,
                "horizon": horizon,
                "target_count": len(group),
                "macro_mean_latency_gain_pct": float(
                    np.mean(gains)
                ),
                "macro_latency_gain_ci95_low": (
                    ci_low
                ),
                "macro_latency_gain_ci95_high": (
                    ci_high
                ),
                "macro_mean_wrong_choices_saved": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_wrong_choices_saved"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_wrong_before_convergence_saved": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_wrong_before_convergence_saved"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_convergence_probability": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "convergence_probability"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_baseline_convergence_probability": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "baseline_convergence_probability"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_censored_convergence_request": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_censored_convergence_request"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_baseline_censored_convergence_request": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_baseline_censored_convergence_request"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_convergence_requests_saved": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_convergence_requests_saved"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_first_arm_optimal_probability": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "first_arm_optimal_probability"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
                "macro_mean_reward_pseudo_regret": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_reward_pseudo_regret"
                                ]
                            )
                            for row in group
                        ]
                    )
                ),
            }
        )

    return output


def compare_to_current_diagonal(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Compare each (w_R, w_E) against the CURRENT formula with the same w_R,
    i.e. the diagonal configuration (w_R, w_R).

    This isolates the effect of changing only the exploration pseudo-count.
    """
    index = {
        (
            row["target_function"],
            float(
                row["reward_prior_weight"]
            ),
            float(
                row[
                    "exploration_prior_weight"
                ]
            ),
            int(row["horizon"]),
        ): row
        for row in rows
    }

    targets = sorted(
        {
            row["target_function"]
            for row in rows
        }
    )

    keys = sorted(
        {
            (
                float(
                    row[
                        "reward_prior_weight"
                    ]
                ),
                float(
                    row[
                        "exploration_prior_weight"
                    ]
                ),
                int(row["horizon"]),
            )
            for row in rows
        }
    )

    output: list[dict[str, Any]] = []

    for w_r, w_e, horizon in keys:
        gain_delta = []
        wrong_delta = []
        wrong_before_delta = []
        convergence_delta = []

        for target in targets:
            candidate = index[
                (
                    target,
                    w_r,
                    w_e,
                    horizon,
                )
            ]
            current = index[
                (
                    target,
                    w_r,
                    w_r,
                    horizon,
                )
            ]

            gain_delta.append(
                float(
                    candidate[
                        "mean_latency_gain_pct"
                    ]
                )
                - float(
                    current[
                        "mean_latency_gain_pct"
                    ]
                )
            )

            wrong_delta.append(
                float(
                    candidate[
                        "mean_wrong_choices_saved"
                    ]
                )
                - float(
                    current[
                        "mean_wrong_choices_saved"
                    ]
                )
            )

            wrong_before_delta.append(
                float(
                    candidate[
                        "mean_wrong_before_convergence_saved"
                    ]
                )
                - float(
                    current[
                        "mean_wrong_before_convergence_saved"
                    ]
                )
            )

            convergence_delta.append(
                float(
                    candidate[
                        "mean_convergence_requests_saved"
                    ]
                )
                - float(
                    current[
                        "mean_convergence_requests_saved"
                    ]
                )
            )

        gain_array = np.asarray(
            gain_delta,
            dtype=float,
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                gain_array,
                stable_seed(
                    seed,
                    (
                        "vs-current-"
                        f"{w_r}-{w_e}-{horizon}"
                    ),
                ),
            )
        )

        output.append(
            {
                "reward_prior_weight": w_r,
                "exploration_prior_weight": w_e,
                "horizon": horizon,
                "is_current_diagonal": (
                    abs(w_r - w_e)
                    <= 1e-12
                ),
                "delta_latency_gain_pct_vs_current_same_wR": float(
                    np.mean(gain_array)
                ),
                "delta_latency_gain_ci95_low": (
                    ci_low
                ),
                "delta_latency_gain_ci95_high": (
                    ci_high
                ),
                "delta_wrong_choices_saved_vs_current_same_wR": float(
                    np.mean(
                        np.asarray(
                            wrong_delta,
                            dtype=float,
                        )
                    )
                ),
                "delta_wrong_before_convergence_saved_vs_current_same_wR": float(
                    np.mean(
                        np.asarray(
                            wrong_before_delta,
                            dtype=float,
                        )
                    )
                ),
                "delta_convergence_requests_saved_vs_current_same_wR": float(
                    np.mean(
                        np.asarray(
                            convergence_delta,
                            dtype=float,
                        )
                    )
                ),
            }
        )

    return output


def plot_grid(
    summary: list[dict[str, Any]],
    figures: Path,
    primary_horizon: int,
) -> None:
    rows = [
        row
        for row in summary
        if int(row["horizon"])
        == primary_horizon
    ]

    reward_weights = sorted(
        {
            float(
                row["reward_prior_weight"]
            )
            for row in rows
        }
    )
    exploration_weights = sorted(
        {
            float(
                row[
                    "exploration_prior_weight"
                ]
            )
            for row in rows
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    for w_r in reward_weights:
        subset = [
            row
            for row in rows
            if float(
                row[
                    "reward_prior_weight"
                ]
            )
            == w_r
        ]
        subset.sort(
            key=lambda row: float(
                row[
                    "exploration_prior_weight"
                ]
            )
        )

        ax.plot(
            [
                float(
                    row[
                        "exploration_prior_weight"
                    ]
                )
                for row in subset
            ],
            [
                float(
                    row[
                        "macro_mean_latency_gain_pct"
                    ]
                )
                for row in subset
            ],
            marker="o",
            label=f"w_R={w_r:g}",
        )

    ax.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
    )
    ax.set_xticks(
        exploration_weights
    )
    ax.set_xlabel(
        "Exploration pseudo-count weight w_E"
    )
    ax.set_ylabel(
        "Mean latency gain vs no-transfer (%)"
    )
    ax.set_title(
        "Decoupled UCB1 latency gain "
        f"at H={primary_horizon}"
    )
    ax.grid(
        True,
        alpha=0.2,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figures
        / (
            "decoupled-ucb1-latency-"
            f"h{primary_horizon}.png"
        ),
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        figures
        / (
            "decoupled-ucb1-latency-"
            f"h{primary_horizon}.svg"
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--donor-metric-results",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--raw-x86",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--raw-arm64",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--algorithm",
        default="kmeans",
    )
    p.add_argument(
        "--ranker",
        default="manhattan",
    )
    p.add_argument(
        "--c",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--reward-weight",
        action="append",
        type=float,
    )
    p.add_argument(
        "--exploration-weight",
        action="append",
        type=float,
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=50,
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=500,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--convergence-run",
        type=int,
        default=5,
    )
    p.add_argument(
        "--primary-horizon",
        type=int,
        default=10,
    )

    args = p.parse_args()

    reward_weights = (
        args.reward_weight
        or DEFAULT_REWARD_WEIGHTS
    )
    exploration_weights = (
        args.exploration_weight
        or DEFAULT_EXPLORATION_WEIGHTS
    )

    for value in (
        reward_weights
        + exploration_weights
    ):
        if value <= 0:
            raise ValueError(
                "All prior weights must be > 0"
            )

    horizons = [
        h
        for h in DEFAULT_HORIZONS
        if h <= args.horizon
    ]
    if args.horizon not in horizons:
        horizons.append(args.horizon)
        horizons.sort()

    donors = load_manhattan_donors(
        args.donor_metric_results.resolve(),
        algorithm=args.algorithm,
        ranker=args.ranker,
    )

    raw_x86 = load_raw_durations(
        args.raw_x86.resolve()
    )
    raw_arm64 = load_raw_durations(
        args.raw_arm64.resolve()
    )
    performance = empirical_stats(
        raw_x86,
        raw_arm64,
    )

    if set(donors) != set(performance):
        raise RuntimeError(
            "Donor/performance mismatch: "
            f"missing={sorted(set(performance)-set(donors))}, "
            f"extra={sorted(set(donors)-set(performance))}"
        )

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_target: list[
        dict[str, Any]
    ] = []

    targets = sorted(performance)

    for target_idx, target_name in enumerate(
        targets,
        start=1,
    ):
        print(
            f"[{target_idx:02d}/{len(targets)}] "
            f"{target_name}"
        )

        target_stats = performance[
            target_name
        ]
        donor_name = donors[target_name]
        donor_stats = performance[
            donor_name
        ]

        prior_mean = (
            build_reference_anchored_prior(
                target_stats,
                donor_stats,
            )
        )

        sequences = make_sequences(
            target_name=target_name,
            target_stats=target_stats,
            horizon=args.horizon,
            replicates=args.replicates,
            seed=args.seed,
        )

        baseline_traces = [
            simulate_baseline_trace(
                target_stats=target_stats,
                sequences=sequences[i],
                horizon=args.horizon,
                c=args.c,
                convergence_run=(
                    args.convergence_run
                ),
            )
            for i in range(
                args.replicates
            )
        ]

        baseline_metrics = {
            horizon: [
                trace_metrics_at_horizon(
                    trace=trace,
                    target_stats=target_stats,
                    horizon=horizon,
                    convergence_run=(
                        args.convergence_run
                    ),
                )
                for trace in baseline_traces
            ]
            for horizon in horizons
        }

        for w_r in reward_weights:
            for w_e in exploration_weights:
                transfer_traces = [
                    simulate_transfer_trace(
                        target_stats=target_stats,
                        prior_mean_reward=prior_mean,
                        sequences=sequences[i],
                        horizon=args.horizon,
                        c=args.c,
                        reward_prior_weight=w_r,
                        exploration_prior_weight=w_e,
                        convergence_run=(
                            args.convergence_run
                        ),
                    )
                    for i in range(
                        args.replicates
                    )
                ]

                for horizon in horizons:
                    transfer_metrics = [
                        trace_metrics_at_horizon(
                            trace=trace,
                            target_stats=target_stats,
                            horizon=horizon,
                            convergence_run=(
                                args.convergence_run
                            ),
                        )
                        for trace in transfer_traces
                    ]

                    baseline_h = (
                        baseline_metrics[
                            horizon
                        ]
                    )

                    baseline_latency = (
                        np.asarray(
                            [
                                row[
                                    "cumulative_latency"
                                ]
                                for row
                                in baseline_h
                            ],
                            dtype=float,
                        )
                    )
                    transfer_latency = (
                        np.asarray(
                            [
                                row[
                                    "cumulative_latency"
                                ]
                                for row
                                in transfer_metrics
                            ],
                            dtype=float,
                        )
                    )

                    latency_gain = (
                        100.0
                        * (
                            baseline_latency
                            - transfer_latency
                        )
                        / baseline_latency
                    )

                    baseline_wrong = (
                        np.asarray(
                            [
                                row[
                                    "wrong_count"
                                ]
                                for row
                                in baseline_h
                            ],
                            dtype=float,
                        )
                    )
                    transfer_wrong = (
                        np.asarray(
                            [
                                row[
                                    "wrong_count"
                                ]
                                for row
                                in transfer_metrics
                            ],
                            dtype=float,
                        )
                    )

                    baseline_wrong_before = (
                        np.asarray(
                            [
                                row[
                                    "wrong_before_convergence"
                                ]
                                for row
                                in baseline_h
                            ],
                            dtype=float,
                        )
                    )
                    transfer_wrong_before = (
                        np.asarray(
                            [
                                row[
                                    "wrong_before_convergence"
                                ]
                                for row
                                in transfer_metrics
                            ],
                            dtype=float,
                        )
                    )

                    baseline_conv = (
                        np.asarray(
                            [
                                row[
                                    "censored_convergence_request"
                                ]
                                for row
                                in baseline_h
                            ],
                            dtype=float,
                        )
                    )
                    transfer_conv = (
                        np.asarray(
                            [
                                row[
                                    "censored_convergence_request"
                                ]
                                for row
                                in transfer_metrics
                            ],
                            dtype=float,
                        )
                    )

                    per_target.append(
                        {
                            "target_function": (
                                target_name
                            ),
                            "donor_function": (
                                donor_name
                            ),
                            "algorithm": (
                                args.algorithm
                            ),
                            "ranker": (
                                args.ranker
                            ),
                            "prior_mode": (
                                "reference-anchored"
                            ),
                            "formula_mode": (
                                "current-diagonal"
                                if abs(
                                    w_r - w_e
                                )
                                <= 1e-12
                                else "decoupled"
                            ),
                            "reward_prior_weight": (
                                w_r
                            ),
                            "exploration_prior_weight": (
                                w_e
                            ),
                            "horizon": horizon,
                            "mean_latency_gain_pct": float(
                                np.mean(
                                    latency_gain
                                )
                            ),
                            "mean_wrong_choices_saved": float(
                                np.mean(
                                    baseline_wrong
                                    - transfer_wrong
                                )
                            ),
                            "mean_wrong_before_convergence_saved": float(
                                np.mean(
                                    baseline_wrong_before
                                    - transfer_wrong_before
                                )
                            ),
                            "convergence_probability": float(
                                np.mean(
                                    [
                                        row[
                                            "converged"
                                        ]
                                        for row
                                        in transfer_metrics
                                    ]
                                )
                            ),
                            "baseline_convergence_probability": float(
                                np.mean(
                                    [
                                        row[
                                            "converged"
                                        ]
                                        for row
                                        in baseline_h
                                    ]
                                )
                            ),
                            "mean_censored_convergence_request": float(
                                np.mean(
                                    transfer_conv
                                )
                            ),
                            "mean_baseline_censored_convergence_request": float(
                                np.mean(
                                    baseline_conv
                                )
                            ),
                            "mean_convergence_requests_saved": float(
                                np.mean(
                                    baseline_conv
                                    - transfer_conv
                                )
                            ),
                            "first_arm_optimal_probability": float(
                                np.mean(
                                    [
                                        row[
                                            "first_arm_optimal"
                                        ]
                                        for row
                                        in transfer_metrics
                                    ]
                                )
                            ),
                            "mean_reward_pseudo_regret": float(
                                np.mean(
                                    [
                                        row[
                                            "reward_pseudo_regret"
                                        ]
                                        for row
                                        in transfer_metrics
                                    ]
                                )
                            ),
                        }
                    )

    per_target.sort(
        key=lambda row: (
            float(
                row[
                    "reward_prior_weight"
                ]
            ),
            float(
                row[
                    "exploration_prior_weight"
                ]
            ),
            int(row["horizon"]),
            row["target_function"],
        )
    )

    summary = summarize(
        per_target,
        seed=args.seed,
    )

    comparison = (
        compare_to_current_diagonal(
            per_target,
            seed=args.seed,
        )
    )

    write_csv(
        out
        / "decoupled-ucb1-per-target.csv",
        per_target,
    )
    write_csv(
        out
        / "decoupled-ucb1-summary.csv",
        summary,
    )
    write_csv(
        out
        / "decoupled-vs-current.csv",
        comparison,
    )

    plot_grid(
        summary=summary,
        figures=figures,
        primary_horizon=(
            args.primary_horizon
        ),
    )

    manifest = {
        "schema_version": 1,
        "experiment": (
            "decoupled UCB1 weak prior "
            "reward weight vs exploration pseudo-count"
        ),
        "frozen_upstream_choices": {
            "algorithm": args.algorithm,
            "ranker": args.ranker,
            "prior": (
                "reference-anchored "
                "architecture-effect"
            ),
            "reward": "-ln(duration_ms)",
            "c": args.c,
        },
        "formula": {
            "exploitation": (
                "(S_j + w_R * mu_prior_j) "
                "/ (n_j + w_R)"
            ),
            "exploration": (
                "c * sqrt("
                "ln(N + sum_k w_E,k) "
                "/ (n_j + w_E))"
            ),
            "zero_bonus_rule": (
                "exploration bonus = 0 "
                "when exploration total <= 1"
            ),
            "current_formula_equivalence": (
                "w_R == w_E"
            ),
        },
        "grid": {
            "reward_prior_weights": (
                reward_weights
            ),
            "exploration_prior_weights": (
                exploration_weights
            ),
        },
        "convergence": {
            "run_length": (
                args.convergence_run
            ),
            "criterion": (
                "completed request closing "
                "the first run of N consecutive "
                "updates where exploitation-only "
                "means rank the empirical best "
                "architecture correctly"
            ),
            "censoring": (
                "H+1 when not converged "
                "within horizon H"
            ),
        },
        "evaluation": {
            "horizons": horizons,
            "replicates": (
                args.replicates
            ),
            "seed": args.seed,
            "paired_common_random_numbers": (
                True
            ),
        },
        "future_gcp_validation": {
            "required": True,
            "same_metrics": [
                "requests_to_convergence",
                "seconds_to_convergence",
                "wrong_choices_before_convergence",
                "wrong_choices_by_horizon",
                "cumulative_latency",
                "reward_regret",
            ],
            "note": (
                "Local results are for hypothesis "
                "selection; final conclusions must "
                "be checked end-to-end on GCP."
            ),
        },
        "outputs": [
            "decoupled-ucb1-per-target.csv",
            "decoupled-ucb1-summary.csv",
            "decoupled-vs-current.csv",
            "figures/",
        ],
    }

    (
        out
        / "decoupled-ucb1-manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\noutput={out}"
    )

    primary = [
        row
        for row in summary
        if int(row["horizon"])
        == args.primary_horizon
    ]
    primary.sort(
        key=lambda row: (
            float(
                row[
                    "reward_prior_weight"
                ]
            ),
            float(
                row[
                    "exploration_prior_weight"
                ]
            ),
        )
    )

    print(
        f"\nDECOUPLED GRID "
        f"@ H={args.primary_horizon}"
    )

    for row in primary:
        print(
            f"{row['formula_mode']:<16} "
            f"wR={float(row['reward_prior_weight']):<4g} "
            f"wE={float(row['exploration_prior_weight']):<4g} "
            f"gain="
            f"{float(row['macro_mean_latency_gain_pct']):+.4f}% "
            f"CI95=["
            f"{float(row['macro_latency_gain_ci95_low']):+.4f},"
            f"{float(row['macro_latency_gain_ci95_high']):+.4f}] "
            f"wrong-saved="
            f"{float(row['macro_mean_wrong_choices_saved']):+.4f} "
            f"wrong-before-saved="
            f"{float(row['macro_mean_wrong_before_convergence_saved']):+.4f} "
            f"conv-saved="
            f"{float(row['macro_mean_convergence_requests_saved']):+.4f} "
            f"conv-prob="
            f"{float(row['macro_mean_convergence_probability']):.4f}"
        )

    print(
        f"\nDECOUPLED - CURRENT SAME wR "
        f"@ H={args.primary_horizon}"
    )

    direct = [
        row
        for row in comparison
        if int(row["horizon"])
        == args.primary_horizon
    ]
    direct.sort(
        key=lambda row: (
            float(
                row[
                    "reward_prior_weight"
                ]
            ),
            float(
                row[
                    "exploration_prior_weight"
                ]
            ),
        )
    )

    for row in direct:
        print(
            f"wR={float(row['reward_prior_weight']):<4g} "
            f"wE={float(row['exploration_prior_weight']):<4g} "
            f"Δgain="
            f"{float(row['delta_latency_gain_pct_vs_current_same_wR']):+.4f}pp "
            f"CI95=["
            f"{float(row['delta_latency_gain_ci95_low']):+.4f},"
            f"{float(row['delta_latency_gain_ci95_high']):+.4f}] "
            f"Δwrong="
            f"{float(row['delta_wrong_choices_saved_vs_current_same_wR']):+.4f} "
            f"Δwrong-before="
            f"{float(row['delta_wrong_before_convergence_saved_vs_current_same_wR']):+.4f} "
            f"Δconv="
            f"{float(row['delta_convergence_requests_saved_vs_current_same_wR']):+.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
