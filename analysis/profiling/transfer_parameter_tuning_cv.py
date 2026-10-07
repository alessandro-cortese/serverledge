#!/usr/bin/env python3
"""
Serverledge — controlled parameter tuning for BOTH TL prior formulas.

This experiment is intentionally downstream of the frozen clustering /
donor-selection pipeline.

Frozen upstream pipeline
------------------------
- representation / clustering / donor-selection: already frozen
- donor variant: weighted_vote_p2
- donor ranking: Manhattan, as encoded in the frozen donor artifact
- target set: the 59 static-eligible functions represented in that artifact

Transfer formulas tuned SEPARATELY
----------------------------------
difference:
    Delta_D = mu_D,ARM - mu_D,x86
    mu_prior_T,x86 = mu_T,x86
    mu_prior_T,ARM = mu_T,x86 + Delta_D

ratio:
    R_D = mu_D,ARM / mu_D,x86
    mu_prior_T,x86 = mu_T,x86
    mu_prior_T,ARM = mu_T,x86 * R_D

Reward:
    r = -ln(duration_ms)

Parameters
----------
w_R varies for both formulas.

Two exploration regimes are predeclared:

A) positive-w_E regime
   w_E in a broad grid including very large values.
   c varies jointly.

B) zero-w_E / high-c regime
   w_E = 0 exactly.
   c uses a deliberately higher grid.

IMPORTANT semantics for w_E = 0
-------------------------------
For an unobserved arm n_j = 0, the textbook exploration denominator is zero.
This script therefore uses the classic-UCB convention:
    B_j = +infinity for every untried arm.

Consequently both architectures are forced to be tried once before the
finite UCB scores take over. The reward prior (w_R) is still used in the
effective mean. This is the clean operational interpretation of:
    "w_E = 0, compensate with a larger c".

Anti-overfitting design
-----------------------
1) Hyperparameter tuning uses ONLY H=10 (predeclared primary horizon).
2) Five target-level outer folds are fixed deterministically.
3) For each outer fold and each formula, parameters are selected using ONLY
   the other four folds.
4) Outer-fold evaluation uses an INDEPENDENT Monte Carlo seed and 500 replay
   replicates at H={5,10,20,50}.
5) The exact same CRN streams are shared across configurations/formulas.
6) The target x86 anchor uses a distinct RNG stream from the online replay.
7) No outer-held-out target contributes to parameter selection.

Selection rule on the training portion of each outer fold
---------------------------------------------------------
Lexicographic, no arbitrary weighted score:
  1. maximize mean latency gain vs no-transfer at H=10;
  2. maximize reward pseudo-regret saved;
  3. maximize optimal-arm-rate gain;
  4. maximize wrong selections saved;
  5. deterministic parameter tie-break.

The script also reports a full-data recommended configuration for each formula
AFTER the cross-validated performance estimate is produced. That recommendation
is for subsequent experiments/deployment; its training-set score is NOT an
unbiased performance estimate.

Default predeclared grid
------------------------
w_R:
    0.05, 0.10, 0.25, 0.50, 1.00, 2.00

positive w_E:
    0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 32.00

c when w_E > 0:
    0.40, 0.80, 1.20, 1.60, 2.40

c when w_E = 0:
    0.80, 1.20, 1.60, 2.40, 3.20, 4.80

=> 276 configurations per formula, 552 formula/config combinations.

Outputs
-------
tl-tuning-grid.csv
tl-tuning-per-target.csv
tl-tuning-all-target-ranking.csv
tl-tuning-fold-assignments.csv
tl-tuning-fold-selections.csv
tl-tuning-selection-frequency.csv
tl-tuning-full-data-recommendation.csv
tl-tuning-outer-evaluation-per-target.csv
tl-tuning-outer-evaluation-summary.csv
tl-tuning-outer-paired-bootstrap.csv
tl-tuning-report.md
tl-tuning-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.profiling.transfer_decoupled_ucb1 import (
    DecoupledUCB1Policy,
)
from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    UCB1Policy,
)
from analysis.profiling.transfer_prior_formula_comparison import (
    build_performance,
    build_prior,
    load_eligible_durations,
    load_frozen_donor_map,
    make_crn_samples,
    stable_seed,
)


FORMULAS = ("difference", "ratio")
DONOR_VARIANT = "weighted_vote_p2"
EPS = 1e-12


def parse_float_list(value: str) -> list[float]:
    values = [
        float(x.strip())
        for x in value.split(",")
        if x.strip()
    ]
    if not values:
        raise ValueError("empty float list")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    ]
    if not values:
        raise ValueError("empty integer list")
    return values


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Config:
    config_id: str
    regime: str
    w_r: float
    w_e: float
    c: float


def make_grid(
    reward_weights: list[float],
    positive_exploration_weights: list[float],
    positive_c_values: list[float],
    zero_c_values: list[float],
) -> list[Config]:
    configs: list[Config] = []
    idx = 0

    for w_r in reward_weights:
        if w_r <= 0:
            raise ValueError("w_R must be > 0")

        for w_e in positive_exploration_weights:
            if w_e <= 0:
                raise ValueError(
                    "positive-w_E grid must contain only values > 0"
                )

            for c in positive_c_values:
                if c <= 0:
                    raise ValueError("c must be > 0")

                idx += 1
                configs.append(
                    Config(
                        config_id=f"cfg-{idx:03d}",
                        regime=(
                            "large_positive_we"
                            if w_e >= 8.0
                            else "positive_we"
                        ),
                        w_r=float(w_r),
                        w_e=float(w_e),
                        c=float(c),
                    )
                )

        for c in zero_c_values:
            if c <= 0:
                raise ValueError("c must be > 0")

            idx += 1
            configs.append(
                Config(
                    config_id=f"cfg-{idx:03d}",
                    regime="zero_we_high_c",
                    w_r=float(w_r),
                    w_e=0.0,
                    c=float(c),
                )
            )

    return configs


class NumericTLPolicy:
    """
    Lightweight exact implementation of the tuned UCB score.

    For w_E > 0 this is intended to be numerically equivalent to the existing
    DecoupledUCB1Policy.

    For w_E == 0, untried arms receive +inf exploration bonus, matching the
    classic UCB convention and avoiding division by zero.
    """

    def __init__(
        self,
        *,
        c: float,
        w_r: float,
        w_e: float,
        prior_mean: np.ndarray,
    ):
        if c <= 0:
            raise ValueError("c must be > 0")
        if w_r <= 0:
            raise ValueError("w_R must be > 0")
        if w_e < 0:
            raise ValueError("w_E must be >= 0")

        self.c = float(c)
        self.w_r = float(w_r)
        self.w_e = float(w_e)
        self.prior_mean = np.asarray(
            prior_mean,
            dtype=float,
        )
        self.counts = np.zeros(2, dtype=np.int64)
        self.reward_sums = np.zeros(2, dtype=float)

    def effective_mean(self, idx: int) -> float:
        return float(
            (
                self.reward_sums[idx]
                + self.w_r * self.prior_mean[idx]
            )
            / (
                float(self.counts[idx])
                + self.w_r
            )
        )

    def exploration_bonus(self, idx: int) -> float:
        if self.w_e == 0.0:
            if self.counts[idx] == 0:
                return math.inf

            total = int(self.counts.sum())

            if total <= 1:
                return 0.0

            return float(
                self.c
                * math.sqrt(
                    math.log(float(total))
                    / float(self.counts[idx])
                )
            )

        effective_counts = (
            self.counts.astype(float)
            + self.w_e
        )
        total = float(effective_counts.sum())

        if total <= 1.0:
            return 0.0

        return float(
            self.c
            * math.sqrt(
                math.log(total)
                / float(effective_counts[idx])
            )
        )

    def score(self, idx: int) -> float:
        return (
            self.effective_mean(idx)
            + self.exploration_bonus(idx)
        )

    def select_idx(self) -> int:
        s0 = self.score(0)
        s1 = self.score(1)

        # Same tie-breaking as max(ARMS, key=(score, -ARMS.index())):
        # x86 (index 0) wins exact ties.
        return 0 if s0 >= s1 else 1

    def update(self, idx: int, duration_ms: float) -> float:
        reward = -math.log(float(duration_ms))
        self.counts[idx] += 1
        self.reward_sums[idx] += reward
        return reward


def implementation_self_check() -> None:
    """
    Verify the positive-w_E numeric policy against the repository's current
    DecoupledUCB1Policy before starting the expensive grid.
    """
    prior_dict = {
        "x86": -7.2,
        "arm64": -6.9,
    }
    prior_arr = np.asarray(
        [
            prior_dict["x86"],
            prior_dict["arm64"],
        ],
        dtype=float,
    )

    durations = [
        1200.0,
        900.0,
        1000.0,
        850.0,
        1100.0,
        920.0,
    ]

    arm_to_idx = {
        "x86": 0,
        "arm64": 1,
    }

    for (
        c,
        w_r,
        w_e,
    ) in [
        (0.8, 0.25, 1.0),
        (1.6, 1.0, 8.0),
        (0.4, 2.0, 32.0),
    ]:
        reference = DecoupledUCB1Policy(
            c=c,
            reward_prior_weight=w_r,
            exploration_prior_weight=w_e,
            prior_mean_reward=prior_dict,
        )

        numeric = NumericTLPolicy(
            c=c,
            w_r=w_r,
            w_e=w_e,
            prior_mean=prior_arr,
        )

        for duration in durations:
            ref_arm = reference.select_arm()
            num_idx = numeric.select_idx()
            num_arm = ARMS[num_idx]

            if ref_arm != num_arm:
                raise SystemExit(
                    "SELF-CHECK FAILED: arm mismatch "
                    f"c={c} w_R={w_r} w_E={w_e} "
                    f"reference={ref_arm} numeric={num_arm}"
                )

            for arm in ARMS:
                idx = arm_to_idx[arm]
                ref_mean = float(
                    reference.effective_mean(arm)
                )
                num_mean = float(
                    numeric.effective_mean(idx)
                )

                if not math.isclose(
                    ref_mean,
                    num_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise SystemExit(
                        "SELF-CHECK FAILED: effective mean mismatch"
                    )

                ref_bonus = float(
                    reference.exploration_bonus(arm)
                )
                num_bonus = float(
                    numeric.exploration_bonus(idx)
                )

                if not math.isclose(
                    ref_bonus,
                    num_bonus,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise SystemExit(
                        "SELF-CHECK FAILED: exploration bonus mismatch"
                    )

            idx = arm_to_idx[ref_arm]

            reference.update(
                ref_arm,
                duration,
            )
            numeric.update(
                idx,
                duration,
            )

    # w_E = 0 invariant: each arm must be tried before finite exploration.
    zero = NumericTLPolicy(
        c=2.4,
        w_r=0.25,
        w_e=0.0,
        prior_mean=prior_arr,
    )

    first = zero.select_idx()
    zero.update(first, 1000.0)
    second = zero.select_idx()

    if first == second:
        raise SystemExit(
            "SELF-CHECK FAILED: w_E=0 did not force the untried arm"
        )

    print("implementation_self_check=PASS")


def baseline_effective_mean(
    policy: UCB1Policy,
    arm: str,
) -> float | None:
    count = float(
        policy.effective_count(arm)
    )
    if count <= 0:
        return None

    return float(
        (
            policy.reward_sums[arm]
            + policy.prior_reward_sums[arm]
        )
        / count
    )


def first_completed_true_run(
    values: list[bool],
    run_length: int,
) -> int | None:
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


def simulate_baseline_trace(
    *,
    target_stats: dict[str, Any],
    sequences: dict[str, np.ndarray],
    c: float,
    convergence_run: int,
) -> dict[str, Any]:
    policy = UCB1Policy(
        c=c,
        prior_weight=0.0,
        prior_mean_reward=None,
    )

    positions = {
        arm: 0
        for arm in ARMS
    }

    best_arm = str(
        target_stats[
            "best_reward_arm"
        ]
    )

    best_mean_reward = float(
        target_stats[
            "mean_reward"
        ][best_arm]
    )

    chosen: list[str] = []
    durations: list[float] = []
    pseudo_regret_steps: list[float] = []
    exploitation_correct: list[bool] = []

    max_horizon = len(
        sequences[ARMS[0]]
    )

    for _ in range(max_horizon):
        arm = str(
            policy.select_arm()
        )

        duration = float(
            sequences[arm][
                positions[arm]
            ]
        )

        positions[arm] += 1

        policy.update(
            arm,
            duration,
        )

        chosen.append(arm)
        durations.append(duration)

        pseudo_regret_steps.append(
            best_mean_reward
            - float(
                target_stats[
                    "mean_reward"
                ][arm]
            )
        )

        means = {
            candidate: baseline_effective_mean(
                policy,
                candidate,
            )
            for candidate in ARMS
        }

        if all(
            means[candidate] is not None
            for candidate in ARMS
        ):
            exploitation_arm = max(
                ARMS,
                key=lambda candidate: (
                    float(means[candidate]),
                    -ARMS.index(candidate),
                ),
            )
            exploitation_correct.append(
                exploitation_arm == best_arm
            )
        else:
            exploitation_correct.append(False)

    return {
        "chosen": chosen,
        "durations": np.asarray(
            durations,
            dtype=float,
        ),
        "pseudo_regret_steps": np.asarray(
            pseudo_regret_steps,
            dtype=float,
        ),
        "exploitation_correct": exploitation_correct,
        "best_arm": best_arm,
        "convergence_run": convergence_run,
    }


def simulate_tl_trace(
    *,
    target_stats: dict[str, Any],
    sequences: dict[str, np.ndarray],
    prior_mean_reward: dict[str, float],
    config: Config,
    convergence_run: int,
) -> dict[str, Any]:
    prior = np.asarray(
        [
            float(
                prior_mean_reward[
                    "x86"
                ]
            ),
            float(
                prior_mean_reward[
                    "arm64"
                ]
            ),
        ],
        dtype=float,
    )

    policy = NumericTLPolicy(
        c=config.c,
        w_r=config.w_r,
        w_e=config.w_e,
        prior_mean=prior,
    )

    positions = np.zeros(
        2,
        dtype=np.int64,
    )

    best_arm = str(
        target_stats[
            "best_reward_arm"
        ]
    )
    best_idx = ARMS.index(
        best_arm
    )

    best_mean_reward = float(
        target_stats[
            "mean_reward"
        ][best_arm]
    )

    chosen_idx: list[int] = []
    durations: list[float] = []
    pseudo_regret_steps: list[float] = []
    exploitation_correct: list[bool] = []

    max_horizon = len(
        sequences[ARMS[0]]
    )

    arm_arrays = [
        sequences["x86"],
        sequences["arm64"],
    ]

    mean_rewards = np.asarray(
        [
            float(
                target_stats[
                    "mean_reward"
                ]["x86"]
            ),
            float(
                target_stats[
                    "mean_reward"
                ]["arm64"]
            ),
        ],
        dtype=float,
    )

    for _ in range(max_horizon):
        idx = policy.select_idx()

        pos = int(
            positions[idx]
        )

        duration = float(
            arm_arrays[idx][pos]
        )

        positions[idx] += 1

        policy.update(
            idx,
            duration,
        )

        chosen_idx.append(idx)
        durations.append(duration)

        pseudo_regret_steps.append(
            best_mean_reward
            - float(
                mean_rewards[idx]
            )
        )

        m0 = policy.effective_mean(0)
        m1 = policy.effective_mean(1)

        exploitation_idx = (
            0
            if m0 >= m1
            else 1
        )

        exploitation_correct.append(
            exploitation_idx
            == best_idx
        )

    return {
        "chosen_idx": chosen_idx,
        "durations": np.asarray(
            durations,
            dtype=float,
        ),
        "pseudo_regret_steps": np.asarray(
            pseudo_regret_steps,
            dtype=float,
        ),
        "exploitation_correct": exploitation_correct,
        "best_idx": best_idx,
        "convergence_run": convergence_run,
    }


def prefix_metrics_baseline(
    trace: dict[str, Any],
    horizon: int,
) -> dict[str, float]:
    chosen = trace[
        "chosen"
    ][:horizon]

    best_arm = str(
        trace["best_arm"]
    )

    wrong = sum(
        1
        for arm in chosen
        if arm != best_arm
    )

    convergence = first_completed_true_run(
        trace[
            "exploitation_correct"
        ][:horizon],
        int(
            trace[
                "convergence_run"
            ]
        ),
    )

    return {
        "cumulative_latency_ms": float(
            np.sum(
                trace[
                    "durations"
                ][:horizon]
            )
        ),
        "optimal_arm_rate": float(
            (
                horizon
                - wrong
            )
            / horizon
        ),
        "wrong_selections": float(
            wrong
        ),
        "cumulative_reward_pseudo_regret": float(
            np.sum(
                trace[
                    "pseudo_regret_steps"
                ][:horizon]
            )
        ),
        "first_arm_optimal": float(
            bool(chosen)
            and chosen[0]
            == best_arm
        ),
        "convergence_probability": float(
            convergence is not None
        ),
        "censored_convergence_request": float(
            convergence
            if convergence is not None
            else horizon + 1
        ),
    }


def prefix_metrics_tl(
    trace: dict[str, Any],
    horizon: int,
) -> dict[str, float]:
    chosen = trace[
        "chosen_idx"
    ][:horizon]

    best_idx = int(
        trace["best_idx"]
    )

    wrong = sum(
        1
        for idx in chosen
        if idx != best_idx
    )

    convergence = first_completed_true_run(
        trace[
            "exploitation_correct"
        ][:horizon],
        int(
            trace[
                "convergence_run"
            ]
        ),
    )

    return {
        "cumulative_latency_ms": float(
            np.sum(
                trace[
                    "durations"
                ][:horizon]
            )
        ),
        "optimal_arm_rate": float(
            (
                horizon
                - wrong
            )
            / horizon
        ),
        "wrong_selections": float(
            wrong
        ),
        "cumulative_reward_pseudo_regret": float(
            np.sum(
                trace[
                    "pseudo_regret_steps"
                ][:horizon]
            )
        ),
        "first_arm_optimal": float(
            bool(chosen)
            and chosen[0]
            == best_idx
        ),
        "convergence_probability": float(
            convergence is not None
        ),
        "censored_convergence_request": float(
            convergence
            if convergence is not None
            else horizon + 1
        ),
    }


def mean_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    df = pd.DataFrame(
        rows
    )

    return {
        column: float(
            df[column].mean()
        )
        for column in df.columns
    }


def weighted_mean_metric_dicts(
    weighted_rows: list[
        tuple[
            int,
            dict[str, float],
        ]
    ],
) -> dict[str, float]:
    total_weight = sum(
        weight
        for weight, _ in weighted_rows
    )

    if total_weight <= 0:
        raise ValueError(
            "weighted metric aggregation has zero weight"
        )

    columns = list(
        weighted_rows[0][1].keys()
    )

    return {
        column: float(
            sum(
                weight
                * metrics[column]
                for weight, metrics
                in weighted_rows
            )
            / total_weight
        )
        for column in columns
    }


def compare_to_baseline(
    strategy: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    base_latency = float(
        baseline[
            "cumulative_latency_ms"
        ]
    )

    return {
        "latency_gain_percent_vs_no_transfer": float(
            100.0
            * (
                base_latency
                - float(
                    strategy[
                        "cumulative_latency_ms"
                    ]
                )
            )
            / base_latency
        ),
        "reward_pseudo_regret_saved": float(
            baseline[
                "cumulative_reward_pseudo_regret"
            ]
            - strategy[
                "cumulative_reward_pseudo_regret"
            ]
        ),
        "optimal_arm_rate_gain": float(
            strategy[
                "optimal_arm_rate"
            ]
            - baseline[
                "optimal_arm_rate"
            ]
        ),
        "wrong_selections_saved": float(
            baseline[
                "wrong_selections"
            ]
            - strategy[
                "wrong_selections"
            ]
        ),
        "first_arm_optimal_gain": float(
            strategy[
                "first_arm_optimal"
            ]
            - baseline[
                "first_arm_optimal"
            ]
        ),
        "convergence_probability_gain": float(
            strategy[
                "convergence_probability"
            ]
            - baseline[
                "convergence_probability"
            ]
        ),
        "convergence_requests_saved": float(
            baseline[
                "censored_convergence_request"
            ]
            - strategy[
                "censored_convergence_request"
            ]
        ),
    }


def donor_counter_for_target(
    donor_maps: dict[
        int,
        dict[
            str,
            str | None,
        ],
    ],
    cluster_seeds: list[int],
    target: str,
) -> Counter:
    return Counter(
        donor_maps[seed][target]
        for seed in cluster_seeds
    )


def evaluate_target_config(
    *,
    target: str,
    target_stats: dict[str, Any],
    performance: dict[str, dict[str, Any]],
    donor_counts: Counter,
    anchor_rewards: np.ndarray,
    online: dict[str, np.ndarray],
    baseline_metrics: dict[str, float],
    formula: str,
    config: Config,
    horizon: int,
    convergence_run: int,
) -> dict[str, float]:
    """
    Average exactly over clustering seeds by weighting each unique donor with
    its multiplicity among the frozen seed-specific LOFO mappings.
    """
    weighted_strategy: list[
        tuple[
            int,
            dict[str, float],
        ]
    ] = []

    replicates = int(
        online["x86"].shape[0]
    )

    for donor, donor_weight in (
        donor_counts.items()
    ):
        if donor is None:
            weighted_strategy.append(
                (
                    int(donor_weight),
                    dict(
                        baseline_metrics
                    ),
                )
            )
            continue

        donor_stats = performance[
            str(donor)
        ]

        run_rows: list[
            dict[str, float]
        ] = []

        for rep in range(
            replicates
        ):
            prior, _ = build_prior(
                mode=formula,
                anchor_mean_reward_x86=float(
                    anchor_rewards[rep]
                ),
                donor_stats=donor_stats,
            )

            trace = simulate_tl_trace(
                target_stats=target_stats,
                sequences={
                    "x86": online[
                        "x86"
                    ][rep],
                    "arm64": online[
                        "arm64"
                    ][rep],
                },
                prior_mean_reward=prior,
                config=config,
                convergence_run=convergence_run,
            )

            run_rows.append(
                prefix_metrics_tl(
                    trace,
                    horizon,
                )
            )

        weighted_strategy.append(
            (
                int(donor_weight),
                mean_metrics(
                    run_rows
                ),
            )
        )

    strategy_metrics = (
        weighted_mean_metric_dicts(
            weighted_strategy
        )
    )

    return {
        **strategy_metrics,
        **compare_to_baseline(
            strategy_metrics,
            baseline_metrics,
        ),
    }


def make_fold_assignments(
    *,
    targets: list[str],
    performance: dict[
        str,
        dict[str, Any],
    ],
    folds: int,
    seed: int,
) -> pd.DataFrame:
    """
    Deterministic best-arm-stratified target folds.
    """
    rng = np.random.default_rng(
        seed
    )

    by_arm: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for target in targets:
        by_arm[
            str(
                performance[
                    target
                ][
                    "best_reward_arm"
                ]
            )
        ].append(target)

    rows = []

    for arm in sorted(
        by_arm
    ):
        names = sorted(
            by_arm[arm]
        )

        permutation = rng.permutation(
            len(names)
        )

        shuffled = [
            names[int(i)]
            for i in permutation
        ]

        for idx, target in enumerate(
            shuffled
        ):
            rows.append(
                {
                    "target_function": target,
                    "best_reward_arm": arm,
                    "outer_fold": int(
                        idx % folds
                    ),
                }
            )

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "target_function"
        )
        .reset_index(
            drop=True
        )
    )

    return result


def ranking_table(
    tuning_per_target: pd.DataFrame,
    target_subset: set[str] | None,
) -> pd.DataFrame:
    data = tuning_per_target

    if target_subset is not None:
        data = data[
            data[
                "target_function"
            ].isin(
                target_subset
            )
        ]

    grouped = (
        data.groupby(
            [
                "formula",
                "config_id",
                "regime",
                "w_r",
                "w_e",
                "c",
            ],
            as_index=False,
        )
        .agg(
            target_count=(
                "target_function",
                "nunique",
            ),
            latency_gain=(
                "latency_gain_percent_vs_no_transfer",
                "mean",
            ),
            reward_pseudo_regret_saved=(
                "reward_pseudo_regret_saved",
                "mean",
            ),
            optimal_arm_rate_gain=(
                "optimal_arm_rate_gain",
                "mean",
            ),
            wrong_selections_saved=(
                "wrong_selections_saved",
                "mean",
            ),
            first_arm_optimal_gain=(
                "first_arm_optimal_gain",
                "mean",
            ),
        )
    )

    grouped = grouped.sort_values(
        [
            "formula",
            "latency_gain",
            "reward_pseudo_regret_saved",
            "optimal_arm_rate_gain",
            "wrong_selections_saved",
            "w_r",
            "w_e",
            "c",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
    )

    grouped[
        "rank_within_formula"
    ] = (
        grouped.groupby(
            "formula"
        )
        .cumcount()
        + 1
    )

    return grouped


def select_best_config(
    ranking: pd.DataFrame,
    formula: str,
) -> pd.Series:
    subset = (
        ranking[
            ranking[
                "formula"
            ]
            == formula
        ]
        .sort_values(
            "rank_within_formula"
        )
    )

    if subset.empty:
        raise ValueError(
            f"No configs available for formula={formula}"
        )

    return subset.iloc[0]


def bootstrap_pair(
    *,
    eval_per_target: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    B - A. Resampling unit: outer-held-out target.
    """
    a = (
        eval_per_target[
            (
                eval_per_target[
                    "strategy"
                ]
                == strategy_a
            )
            & (
                eval_per_target[
                    "horizon"
                ]
                == horizon
            )
        ]
        .set_index(
            "target_function"
        )
    )

    b = (
        eval_per_target[
            (
                eval_per_target[
                    "strategy"
                ]
                == strategy_b
            )
            & (
                eval_per_target[
                    "horizon"
                ]
                == horizon
            )
        ]
        .set_index(
            "target_function"
        )
    )

    targets = sorted(
        set(a.index)
        & set(b.index)
    )

    metrics = [
        "latency_gain_percent_vs_no_transfer",
        "optimal_arm_rate",
        "wrong_selections",
        "cumulative_reward_pseudo_regret",
        "first_arm_optimal",
        "convergence_probability",
        "censored_convergence_request",
    ]

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for metric in metrics:
        deltas = np.asarray(
            [
                float(
                    b.loc[
                        target,
                        metric,
                    ]
                )
                - float(
                    a.loc[
                        target,
                        metric,
                    ]
                )
                for target in targets
            ],
            dtype=float,
        )

        boot_means = np.empty(
            replicates,
            dtype=float,
        )

        for i in range(
            replicates
        ):
            idx = rng.integers(
                0,
                len(deltas),
                size=len(deltas),
            )

            boot_means[i] = float(
                np.mean(
                    deltas[idx]
                )
            )

        rows.append(
            {
                "comparison": (
                    f"{strategy_b}"
                    f"_minus_"
                    f"{strategy_a}"
                ),
                "horizon": int(
                    horizon
                ),
                "metric": metric,
                "point_delta": float(
                    np.mean(
                        deltas
                    )
                ),
                "ci95_low": float(
                    np.quantile(
                        boot_means,
                        0.025,
                    )
                ),
                "ci95_high": float(
                    np.quantile(
                        boot_means,
                        0.975,
                    )
                ),
                "target_count": int(
                    len(deltas)
                ),
                "bootstrap_replicates": int(
                    replicates
                ),
            }
        )

    return rows


def main(args: argparse.Namespace) -> None:
    implementation_self_check()

    horizons = sorted(
        set(
            parse_int_list(
                args.eval_horizons
            )
        )
    )

    if args.tuning_horizon not in horizons:
        # Not mathematically required, but helps keep the final output coherent.
        print(
            "note=tuning horizon is not in eval horizons; this is allowed"
        )

    reward_weights = parse_float_list(
        args.reward_weights
    )
    positive_we = parse_float_list(
        args.positive_exploration_weights
    )
    positive_c = parse_float_list(
        args.positive_c_values
    )
    zero_c = parse_float_list(
        args.zero_c_values
    )

    configs = make_grid(
        reward_weights,
        positive_we,
        positive_c,
        zero_c,
    )

    print(
        "GRID "
        f"configs_per_formula={len(configs)} "
        f"formula_configurations={len(configs) * len(FORMULAS)}"
    )

    raw_x86_path = (
        args.raw_x86.resolve()
    )
    raw_arm64_path = (
        args.raw_arm64.resolve()
    )
    donor_detail_path = (
        args.donor_detail.resolve()
    )

    x86 = load_eligible_durations(
        raw_x86_path
    )
    arm64 = load_eligible_durations(
        raw_arm64_path
    )

    performance = build_performance(
        x86,
        arm64,
    )

    (
        donor_maps,
        cluster_seeds,
        targets,
    ) = load_frozen_donor_map(
        donor_detail_path
    )

    missing_targets = [
        target
        for target in targets
        if target not in performance
    ]

    if missing_targets:
        raise SystemExit(
            "Targets missing performance data: "
            + ", ".join(
                missing_targets
            )
        )

    donor_counts_by_target = {
        target: donor_counter_for_target(
            donor_maps,
            cluster_seeds,
            target,
        )
        for target in targets
    }

    all_donors = sorted(
        {
            donor
            for counter in (
                donor_counts_by_target.values()
            )
            for donor in counter
            if donor is not None
        }
    )

    missing_donors = [
        donor
        for donor in all_donors
        if donor not in performance
    ]

    if missing_donors:
        raise SystemExit(
            "Donors missing performance data: "
            + ", ".join(
                missing_donors
            )
        )

    folds = make_fold_assignments(
        targets=targets,
        performance=performance,
        folds=args.outer_folds,
        seed=args.fold_seed,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_path = (
        args.output_dir
        / "tl-tuning-fold-assignments.csv"
    )
    folds.to_csv(
        fold_path,
        index=False,
    )

    grid_rows = [
        {
            "config_id": config.config_id,
            "regime": config.regime,
            "w_r": config.w_r,
            "w_e": config.w_e,
            "c": config.c,
        }
        for config in configs
    ]

    grid_df = pd.DataFrame(
        grid_rows
    )

    grid_path = (
        args.output_dir
        / "tl-tuning-grid.csv"
    )
    grid_df.to_csv(
        grid_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Stage 1 — tuning-only simulation at H=10.
    # ------------------------------------------------------------
    tuning_cache: dict[
        str,
        dict[str, Any],
    ] = {}

    for target_idx, target in enumerate(
        targets,
        start=1,
    ):
        target_stats = performance[
            target
        ]

        (
            anchor_rewards,
            online,
        ) = make_crn_samples(
            target=target,
            target_stats=target_stats,
            replicates=args.tuning_replicates,
            max_horizon=args.tuning_horizon,
            anchor_n=args.anchor_samples,
            base_seed=args.tuning_mc_seed,
        )

        baseline_rows = []

        for rep in range(
            args.tuning_replicates
        ):
            trace = simulate_baseline_trace(
                target_stats=target_stats,
                sequences={
                    "x86": online[
                        "x86"
                    ][rep],
                    "arm64": online[
                        "arm64"
                    ][rep],
                },
                c=args.baseline_c,
                convergence_run=args.convergence_run,
            )

            baseline_rows.append(
                prefix_metrics_baseline(
                    trace,
                    args.tuning_horizon,
                )
            )

        tuning_cache[target] = {
            "anchor_rewards": anchor_rewards,
            "online": online,
            "baseline_metrics": mean_metrics(
                baseline_rows
            ),
        }

        if (
            target_idx % 10 == 0
            or target_idx == len(targets)
        ):
            print(
                "tuning_input_progress="
                f"{target_idx}/{len(targets)}"
            )

    tuning_rows: list[
        dict[str, Any]
    ] = []

    total_formula_configs = (
        len(FORMULAS)
        * len(configs)
    )
    formula_config_idx = 0

    for formula in FORMULAS:
        for config in configs:
            formula_config_idx += 1

            for target in targets:
                cache = tuning_cache[
                    target
                ]

                metrics = evaluate_target_config(
                    target=target,
                    target_stats=performance[
                        target
                    ],
                    performance=performance,
                    donor_counts=donor_counts_by_target[
                        target
                    ],
                    anchor_rewards=cache[
                        "anchor_rewards"
                    ],
                    online=cache[
                        "online"
                    ],
                    baseline_metrics=cache[
                        "baseline_metrics"
                    ],
                    formula=formula,
                    config=config,
                    horizon=args.tuning_horizon,
                    convergence_run=args.convergence_run,
                )

                tuning_rows.append(
                    {
                        "target_function": target,
                        "formula": formula,
                        "config_id": config.config_id,
                        "regime": config.regime,
                        "w_r": config.w_r,
                        "w_e": config.w_e,
                        "c": config.c,
                        "horizon": int(
                            args.tuning_horizon
                        ),
                        **metrics,
                    }
                )

            if (
                formula_config_idx % 10 == 0
                or formula_config_idx
                == total_formula_configs
            ):
                print(
                    "tuning_grid_progress="
                    f"{formula_config_idx}/"
                    f"{total_formula_configs} "
                    f"formula={formula} "
                    f"config={config.config_id}"
                )

    tuning_per_target = pd.DataFrame(
        tuning_rows
    )

    tuning_per_target_path = (
        args.output_dir
        / "tl-tuning-per-target.csv"
    )
    tuning_per_target.to_csv(
        tuning_per_target_path,
        index=False,
    )

    full_ranking = ranking_table(
        tuning_per_target,
        target_subset=None,
    )

    full_ranking_path = (
        args.output_dir
        / "tl-tuning-all-target-ranking.csv"
    )
    full_ranking.to_csv(
        full_ranking_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Stage 2 — outer-fold training-only config selection.
    # ------------------------------------------------------------
    fold_selection_rows = []

    target_to_fold = dict(
        zip(
            folds[
                "target_function"
            ],
            folds[
                "outer_fold"
            ],
        )
    )

    for outer_fold in range(
        args.outer_folds
    ):
        train_targets = {
            target
            for target in targets
            if int(
                target_to_fold[target]
            )
            != outer_fold
        }

        fold_ranking = ranking_table(
            tuning_per_target,
            target_subset=train_targets,
        )

        for formula in FORMULAS:
            best = select_best_config(
                fold_ranking,
                formula,
            )

            fold_selection_rows.append(
                {
                    "outer_fold": int(
                        outer_fold
                    ),
                    "formula": formula,
                    "config_id": str(
                        best[
                            "config_id"
                        ]
                    ),
                    "regime": str(
                        best[
                            "regime"
                        ]
                    ),
                    "w_r": float(
                        best[
                            "w_r"
                        ]
                    ),
                    "w_e": float(
                        best[
                            "w_e"
                        ]
                    ),
                    "c": float(
                        best[
                            "c"
                        ]
                    ),
                    "training_target_count": int(
                        len(
                            train_targets
                        )
                    ),
                    "training_latency_gain": float(
                        best[
                            "latency_gain"
                        ]
                    ),
                    "training_reward_pseudo_regret_saved": float(
                        best[
                            "reward_pseudo_regret_saved"
                        ]
                    ),
                    "training_optimal_arm_rate_gain": float(
                        best[
                            "optimal_arm_rate_gain"
                        ]
                    ),
                    "training_wrong_selections_saved": float(
                        best[
                            "wrong_selections_saved"
                        ]
                    ),
                }
            )

    fold_selections = pd.DataFrame(
        fold_selection_rows
    )

    fold_selections_path = (
        args.output_dir
        / "tl-tuning-fold-selections.csv"
    )
    fold_selections.to_csv(
        fold_selections_path,
        index=False,
    )

    selection_frequency = (
        fold_selections.groupby(
            [
                "formula",
                "config_id",
                "regime",
                "w_r",
                "w_e",
                "c",
            ],
            as_index=False,
        )
        .agg(
            selected_fold_count=(
                "outer_fold",
                "count",
            ),
        )
        .sort_values(
            [
                "formula",
                "selected_fold_count",
                "config_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

    selection_frequency_path = (
        args.output_dir
        / "tl-tuning-selection-frequency.csv"
    )
    selection_frequency.to_csv(
        selection_frequency_path,
        index=False,
    )

    full_recommendation_rows = []

    for formula in FORMULAS:
        best = select_best_config(
            full_ranking,
            formula,
        )

        full_recommendation_rows.append(
            {
                "formula": formula,
                "config_id": str(
                    best[
                        "config_id"
                    ]
                ),
                "regime": str(
                    best[
                        "regime"
                    ]
                ),
                "w_r": float(
                    best[
                        "w_r"
                    ]
                ),
                "w_e": float(
                    best[
                        "w_e"
                    ]
                ),
                "c": float(
                    best[
                        "c"
                    ]
                ),
                "training_all_targets_latency_gain": float(
                    best[
                        "latency_gain"
                    ]
                ),
                "training_all_targets_reward_pseudo_regret_saved": float(
                    best[
                        "reward_pseudo_regret_saved"
                    ]
                ),
                "training_all_targets_optimal_arm_rate_gain": float(
                    best[
                        "optimal_arm_rate_gain"
                    ]
                ),
                "training_all_targets_wrong_selections_saved": float(
                    best[
                        "wrong_selections_saved"
                    ]
                ),
                "note": (
                    "for subsequent experiments only; "
                    "outer-CV is the unbiased performance estimate"
                ),
            }
        )

    full_recommendation = pd.DataFrame(
        full_recommendation_rows
    )

    full_recommendation_path = (
        args.output_dir
        / "tl-tuning-full-data-recommendation.csv"
    )
    full_recommendation.to_csv(
        full_recommendation_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Stage 3 — independent outer-held-out evaluation.
    # ------------------------------------------------------------
    config_lookup = {
        config.config_id: config
        for config in configs
    }

    outer_rows: list[
        dict[str, Any]
    ] = []

    max_eval_horizon = max(
        horizons
    )

    for target_idx, target in enumerate(
        targets,
        start=1,
    ):
        outer_fold = int(
            target_to_fold[
                target
            ]
        )

        target_stats = performance[
            target
        ]

        (
            anchor_rewards,
            online,
        ) = make_crn_samples(
            target=target,
            target_stats=target_stats,
            replicates=args.eval_replicates,
            max_horizon=max_eval_horizon,
            anchor_n=args.anchor_samples,
            base_seed=args.eval_mc_seed,
        )

        baseline_runs_by_horizon = {
            horizon: []
            for horizon in horizons
        }

        for rep in range(
            args.eval_replicates
        ):
            trace = simulate_baseline_trace(
                target_stats=target_stats,
                sequences={
                    "x86": online[
                        "x86"
                    ][rep],
                    "arm64": online[
                        "arm64"
                    ][rep],
                },
                c=args.baseline_c,
                convergence_run=args.convergence_run,
            )

            for horizon in horizons:
                baseline_runs_by_horizon[
                    horizon
                ].append(
                    prefix_metrics_baseline(
                        trace,
                        horizon,
                    )
                )

        baseline_by_horizon = {
            horizon: mean_metrics(
                rows
            )
            for horizon, rows
            in baseline_runs_by_horizon.items()
        }

        for horizon in horizons:
            base = baseline_by_horizon[
                horizon
            ]

            outer_rows.append(
                {
                    "target_function": target,
                    "outer_fold": outer_fold,
                    "strategy": "no_transfer",
                    "formula": "none",
                    "selected_config_id": "",
                    "selected_regime": "",
                    "w_r": 0.0,
                    "w_e": 0.0,
                    "c": float(
                        args.baseline_c
                    ),
                    "horizon": int(
                        horizon
                    ),
                    **base,
                    "latency_gain_percent_vs_no_transfer": 0.0,
                    "reward_pseudo_regret_saved": 0.0,
                    "optimal_arm_rate_gain": 0.0,
                    "wrong_selections_saved": 0.0,
                    "first_arm_optimal_gain": 0.0,
                    "convergence_probability_gain": 0.0,
                    "convergence_requests_saved": 0.0,
                }
            )

        for formula in FORMULAS:
            selection = fold_selections[
                (
                    fold_selections[
                        "outer_fold"
                    ]
                    == outer_fold
                )
                & (
                    fold_selections[
                        "formula"
                    ]
                    == formula
                )
            ]

            if len(selection) != 1:
                raise SystemExit(
                    "Expected exactly one selected config "
                    f"for fold={outer_fold} formula={formula}"
                )

            selected_id = str(
                selection.iloc[
                    0
                ][
                    "config_id"
                ]
            )

            config = config_lookup[
                selected_id
            ]

            donor_counts = (
                donor_counts_by_target[
                    target
                ]
            )

            for horizon in horizons:
                metrics = evaluate_target_config(
                    target=target,
                    target_stats=target_stats,
                    performance=performance,
                    donor_counts=donor_counts,
                    anchor_rewards=anchor_rewards,
                    online=online,
                    baseline_metrics=baseline_by_horizon[
                        horizon
                    ],
                    formula=formula,
                    config=config,
                    horizon=horizon,
                    convergence_run=args.convergence_run,
                )

                outer_rows.append(
                    {
                        "target_function": target,
                        "outer_fold": outer_fold,
                        "strategy": (
                            f"tuned_{formula}"
                        ),
                        "formula": formula,
                        "selected_config_id": (
                            selected_id
                        ),
                        "selected_regime": (
                            config.regime
                        ),
                        "w_r": config.w_r,
                        "w_e": config.w_e,
                        "c": config.c,
                        "horizon": int(
                            horizon
                        ),
                        **metrics,
                    }
                )

        if (
            target_idx % 5 == 0
            or target_idx
            == len(targets)
        ):
            print(
                "outer_eval_progress="
                f"{target_idx}/{len(targets)}"
            )

    outer_eval = pd.DataFrame(
        outer_rows
    )

    outer_eval_path = (
        args.output_dir
        / "tl-tuning-outer-evaluation-per-target.csv"
    )
    outer_eval.to_csv(
        outer_eval_path,
        index=False,
    )

    summary_rows = []

    for (
        strategy,
        formula,
        horizon,
    ), group in outer_eval.groupby(
        [
            "strategy",
            "formula",
            "horizon",
        ]
    ):
        summary_rows.append(
            {
                "strategy": strategy,
                "formula": formula,
                "horizon": int(
                    horizon
                ),
                "target_count": int(
                    group[
                        "target_function"
                    ].nunique()
                ),
                "latency_gain_percent_vs_no_transfer": float(
                    group[
                        "latency_gain_percent_vs_no_transfer"
                    ].mean()
                ),
                "optimal_arm_rate": float(
                    group[
                        "optimal_arm_rate"
                    ].mean()
                ),
                "wrong_selections": float(
                    group[
                        "wrong_selections"
                    ].mean()
                ),
                "cumulative_reward_pseudo_regret": float(
                    group[
                        "cumulative_reward_pseudo_regret"
                    ].mean()
                ),
                "first_arm_optimal": float(
                    group[
                        "first_arm_optimal"
                    ].mean()
                ),
                "convergence_probability": float(
                    group[
                        "convergence_probability"
                    ].mean()
                ),
                "censored_convergence_request": float(
                    group[
                        "censored_convergence_request"
                    ].mean()
                ),
            }
        )

    outer_summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "horizon",
            "strategy",
        ]
    )

    outer_summary_path = (
        args.output_dir
        / "tl-tuning-outer-evaluation-summary.csv"
    )
    outer_summary.to_csv(
        outer_summary_path,
        index=False,
    )

    paired_rows = []

    comparisons = [
        (
            "no_transfer",
            "tuned_difference",
        ),
        (
            "no_transfer",
            "tuned_ratio",
        ),
        (
            "tuned_difference",
            "tuned_ratio",
        ),
    ]

    for (
        strategy_a,
        strategy_b,
    ) in comparisons:
        for horizon in horizons:
            paired_rows.extend(
                bootstrap_pair(
                    eval_per_target=outer_eval,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    horizon=horizon,
                    replicates=args.bootstrap_replicates,
                    seed=stable_seed(
                        args.bootstrap_seed,
                        strategy_a,
                        strategy_b,
                        horizon,
                    ),
                )
            )

    paired = pd.DataFrame(
        paired_rows
    )

    paired_path = (
        args.output_dir
        / "tl-tuning-outer-paired-bootstrap.csv"
    )
    paired.to_csv(
        paired_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------
    report_lines = [
        "# Transfer Learning parameter tuning",
        "",
        "Clustering and donor selection are frozen.",
        "",
        "## Anti-overfitting protocol",
        "",
        f"- outer folds: {args.outer_folds}",
        f"- tuning horizon: H={args.tuning_horizon}",
        f"- tuning Monte Carlo replicates: {args.tuning_replicates}",
        f"- tuning MC seed: {args.tuning_mc_seed}",
        f"- independent evaluation MC replicates: {args.eval_replicates}",
        f"- independent evaluation MC seed: {args.eval_mc_seed}",
        "- selection uses training targets only",
        "- evaluation uses only outer-held-out targets",
        "- CRN shared across configurations/formulas",
        "",
        "## Grid",
        "",
        f"- w_R = {reward_weights}",
        f"- positive w_E = {positive_we}",
        f"- c for positive w_E = {positive_c}",
        "- w_E = 0 classic-UCB regime",
        f"- c for w_E = 0 = {zero_c}",
        f"- configurations per formula = {len(configs)}",
        "",
        "## Fold selections",
        "",
    ]

    for _, row in fold_selections.sort_values(
        [
            "outer_fold",
            "formula",
        ]
    ).iterrows():
        report_lines.append(
            f"- fold={int(row['outer_fold'])}, "
            f"formula={row['formula']}, "
            f"config={row['config_id']}, "
            f"regime={row['regime']}, "
            f"w_R={row['w_r']}, "
            f"w_E={row['w_e']}, "
            f"c={row['c']}, "
            f"train-H10-latency-gain="
            f"{row['training_latency_gain']:.4f}%"
        )

    report_lines += [
        "",
        "## Outer-held-out performance",
        "",
    ]

    for _, row in outer_summary.iterrows():
        report_lines.append(
            f"- strategy={row['strategy']}, "
            f"H={int(row['horizon'])}, "
            f"latency_gain="
            f"{row['latency_gain_percent_vs_no_transfer']:.4f}%, "
            f"optimal_arm_rate="
            f"{row['optimal_arm_rate']:.4f}, "
            f"wrong="
            f"{row['wrong_selections']:.4f}, "
            f"pseudo_regret="
            f"{row['cumulative_reward_pseudo_regret']:.4f}, "
            f"first_arm_optimal="
            f"{row['first_arm_optimal']:.4f}"
        )

    report_lines += [
        "",
        "## Full-data recommendation",
        "",
        (
            "These configurations are intended for the NEXT experiment after "
            "outer-CV has estimated generalization. Their full-data training "
            "score is not an unbiased performance estimate."
        ),
        "",
    ]

    for _, row in full_recommendation.iterrows():
        report_lines.append(
            f"- formula={row['formula']}, "
            f"config={row['config_id']}, "
            f"regime={row['regime']}, "
            f"w_R={row['w_r']}, "
            f"w_E={row['w_e']}, "
            f"c={row['c']}"
        )

    report_path = (
        args.output_dir
        / "tl-tuning-report.md"
    )
    report_path.write_text(
        "\n".join(
            report_lines
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment": (
            "controlled TL parameter tuning for difference and ratio priors"
        ),
        "frozen_upstream": {
            "donor_variant": DONOR_VARIANT,
            "clustering_refit": False,
            "donor_reselection": False,
        },
        "formulas": list(
            FORMULAS
        ),
        "parameter_grid": {
            "reward_weights_wR": (
                reward_weights
            ),
            "positive_exploration_weights_wE": (
                positive_we
            ),
            "positive_c_values": (
                positive_c
            ),
            "zero_wE_c_values": (
                zero_c
            ),
            "configs_per_formula": int(
                len(configs)
            ),
        },
        "zero_wE_semantics": (
            "untried arms receive +infinity bonus; "
            "classic-UCB forced initial exploration"
        ),
        "selection": {
            "primary_horizon": int(
                args.tuning_horizon
            ),
            "rule": [
                "maximize latency gain",
                "maximize reward pseudo-regret saved",
                "maximize optimal-arm-rate gain",
                "maximize wrong selections saved",
                "deterministic parameter tie-break",
            ],
        },
        "validation": {
            "outer_folds": int(
                args.outer_folds
            ),
            "fold_seed": int(
                args.fold_seed
            ),
            "fold_stratification": (
                "empirical best reward arm"
            ),
            "tuning_replicates": int(
                args.tuning_replicates
            ),
            "tuning_mc_seed": int(
                args.tuning_mc_seed
            ),
            "evaluation_replicates": int(
                args.eval_replicates
            ),
            "evaluation_mc_seed": int(
                args.eval_mc_seed
            ),
            "evaluation_horizons": horizons,
            "anchor_samples": int(
                args.anchor_samples
            ),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
        },
        "bootstrap": {
            "unit": "outer-held-out target_function",
            "replicates": int(
                args.bootstrap_replicates
            ),
            "seed": int(
                args.bootstrap_seed
            ),
            "ci": "percentile 95%",
        },
        "inputs": {
            "raw_x86": {
                "path": str(
                    raw_x86_path
                ),
                "sha256": sha256_file(
                    raw_x86_path
                ),
            },
            "raw_arm64": {
                "path": str(
                    raw_arm64_path
                ),
                "sha256": sha256_file(
                    raw_arm64_path
                ),
            },
            "donor_detail": {
                "path": str(
                    donor_detail_path
                ),
                "sha256": sha256_file(
                    donor_detail_path
                ),
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "tl-tuning-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("FOLD SELECTIONS")
    print(
        fold_selections.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("FULL-DATA RECOMMENDATION")
    print(
        full_recommendation.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("OUTER-HELD-OUT SUMMARY")
    print(
        outer_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("TUNED RATIO MINUS TUNED DIFFERENCE")
    primary = paired[
        paired["comparison"]
        == "tuned_ratio_minus_tuned_difference"
    ]
    print(
        primary.sort_values(
            [
                "horizon",
                "metric",
            ]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"grid={grid_path}")
    print(f"folds={fold_path}")
    print(
        f"tuning_per_target="
        f"{tuning_per_target_path}"
    )
    print(
        f"all_target_ranking="
        f"{full_ranking_path}"
    )
    print(
        f"fold_selections="
        f"{fold_selections_path}"
    )
    print(
        f"selection_frequency="
        f"{selection_frequency_path}"
    )
    print(
        f"full_data_recommendation="
        f"{full_recommendation_path}"
    )
    print(
        f"outer_evaluation="
        f"{outer_eval_path}"
    )
    print(
        f"outer_summary="
        f"{outer_summary_path}"
    )
    print(
        f"paired_bootstrap="
        f"{paired_path}"
    )
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Target-level outer-CV tuning of w_R, w_E and c for both "
            "Serverledge TL prior formulas, including large-w_E and "
            "w_E=0/high-c regimes."
        )
    )

    parser.add_argument(
        "--raw-x86",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raw-arm64",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--donor-detail",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reward-weights",
        default="0.05,0.10,0.25,0.50,1.00,2.00",
    )

    parser.add_argument(
        "--positive-exploration-weights",
        default="0.25,0.50,1.00,2.00,4.00,8.00,16.00,32.00",
    )

    parser.add_argument(
        "--positive-c-values",
        default="0.40,0.80,1.20,1.60,2.40",
    )

    parser.add_argument(
        "--zero-c-values",
        default="0.80,1.20,1.60,2.40,3.20,4.80",
    )

    parser.add_argument(
        "--tuning-horizon",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tuning-replicates",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--tuning-mc-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fold-seed",
        type=int,
        default=20261007,
    )

    parser.add_argument(
        "--eval-horizons",
        default="5,10,20,50",
    )

    parser.add_argument(
        "--eval-replicates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval-mc-seed",
        type=int,
        default=424242,
    )

    parser.add_argument(
        "--anchor-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--baseline-c",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--convergence-run",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    return parser


if __name__ == "__main__":
    main(
        build_parser().parse_args()
    )
