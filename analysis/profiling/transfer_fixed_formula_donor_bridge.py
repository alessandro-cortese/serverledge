#!/usr/bin/env python3
"""
Fixed-formula Transfer Learning bridge experiment for Serverledge.

Purpose
-------
Close the clustering/donor-selection phase and measure whether the improved
donor rules translate into better UCB1 behavior WITHOUT retuning Transfer
Learning.

Frozen Transfer Learning configuration
--------------------------------------
policy                 : UCB1Decoupled
reward                  : -ln(duration_ms)
reference architecture  : x86
prior                   : reference-anchored donor architecture effect
w_R                     : 0.25
w_E                     : 1.00
c                       : 0.80
convergence run         : 5
horizons                : 5, 10, 20, 50

Compared strategies
-------------------
1. no_transfer
2. professor_majority
      donor mapping from variant=cluster_majority
3. weighted_vote_p2
      donor mapping from variant=weighted_vote_p2
4. local_k5_majority
      sensitivity only

If a donor rule abstains, that target/run falls back to EXACT no-transfer
behavior. The target is never dropped.

Experimental controls
---------------------
- Donor mappings are READ from the already-frozen vote-refinement artifact.
  This script does not refit clustering and does not reselect donors.
- Common Random Numbers (CRN): the exact same online x86/ARM bootstrap streams
  are reused by all strategies, all donor-selection K-Means seeds, and all
  horizons for a target/replicate.
- Anchor separation: the target x86 reference anchor is drawn from a distinct
  RNG stream from the online replay. By default it uses 10 bootstrap x86
  observations, matching the intended "profile target on x86 before transfer"
  workflow.
- Donor architecture effect uses the donor's empirical warm means over the
  frozen 12-sample corpus.
- Target ARM observations are never used to build the prior; they are used only
  as online replay/evaluation data.

Important
---------
This experiment is an OFFLINE empirical replay/bootstrap study. It evaluates
the currently frozen formula. It does NOT tune w_R, w_E, c, K, features, or
vote parameters.

Outputs
-------
tl-fixed-formula-per-target-seed.csv
tl-fixed-formula-per-target.csv
tl-fixed-formula-summary.csv
tl-fixed-formula-paired-bootstrap.csv
tl-fixed-formula-report.md
tl-fixed-formula-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
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


STRATEGY_TO_VARIANT = {
    "professor_majority": "cluster_majority",
    "weighted_vote_p2": "weighted_vote_p2",
    "local_k5_majority": "local_k5_majority",
}

STRATEGY_ORDER = [
    "no_transfer",
    "professor_majority",
    "weighted_vote_p2",
    "local_k5_majority",
]

DEFAULT_HORIZONS = [5, 10, 20, 50]
EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def parse_int_list(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    return values


def load_eligible_durations(
        path: Path,
) -> dict[str, np.ndarray]:
    """
    Read InvocationSample JSONL and keep only successful warm samples eligible
    for performance analysis.
    """
    grouped: dict[str, list[float]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

            if row.get("warm_start") is not True:
                continue
            if row.get("execution_succeeded") is not True:
                continue

            eligibility = row.get("eligibility") or {}
            if eligibility.get("performance_analysis") is not True:
                continue

            timing = row.get("timing") or {}
            duration = timing.get("duration_ms")
            if duration is None:
                continue

            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise SystemExit(
                    f"{path}:{line_no}: invalid duration_ms={duration}"
                )

            name = str(row.get("function_name") or "")
            if not name:
                raise SystemExit(
                    f"{path}:{line_no}: missing function_name"
                )

            grouped[name].append(duration)

    return {
        name: np.asarray(values, dtype=float)
        for name, values in grouped.items()
    }


def build_performance(
        x86: dict[str, np.ndarray],
        arm64: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    names = sorted(set(x86) & set(arm64))
    result: dict[str, dict[str, Any]] = {}

    for name in names:
        durations = {
            "x86": x86[name],
            "arm64": arm64[name],
        }

        mean_reward = {
            arm: float(
                np.mean(
                    -np.log(durations[arm])
                )
            )
            for arm in ARMS
        }

        best_arm = max(
            ARMS,
            key=lambda arm: (
                mean_reward[arm],
                -ARMS.index(arm),
            ),
        )

        result[name] = {
            "durations": durations,
            "mean_reward": mean_reward,
            "mean_duration": {
                arm: float(
                    np.mean(durations[arm])
                )
                for arm in ARMS
            },
            "best_reward_arm": best_arm,
        }

    return result


def load_donor_detail(
        path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "variant",
        "seed",
        "target_function",
        "selection_status",
        "donor_function",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Donor detail missing columns: {sorted(missing)}"
        )

    wanted = set(STRATEGY_TO_VARIANT.values())

    df = df[
        df["variant"].isin(wanted)
    ].copy()

    if df.empty:
        raise SystemExit(
            "No requested donor variants found in donor detail"
        )

    key_counts = (
        df.groupby(
            ["variant", "seed", "target_function"]
        )
        .size()
    )

    if (key_counts != 1).any():
        bad = key_counts[
            key_counts != 1
            ].head(20)
        raise SystemExit(
            "Expected one donor row per variant/seed/target:\n"
            + bad.to_string()
        )

    return df


def donor_maps(
        detail: pd.DataFrame,
) -> tuple[
    dict[str, dict[int, dict[str, str | None]]],
    list[int],
    list[str],
]:
    maps: dict[
        str,
        dict[int, dict[str, str | None]],
    ] = {}

    seeds = sorted(
        int(x)
        for x in detail["seed"].unique()
    )
    targets = sorted(
        str(x)
        for x in detail["target_function"].unique()
    )

    for strategy, variant in STRATEGY_TO_VARIANT.items():
        v = detail[
            detail["variant"] == variant
            ]

        maps[strategy] = {}

        for seed in seeds:
            s = v[
                v["seed"].astype(int) == seed
                ]

            mapping: dict[str, str | None] = {}

            for target in targets:
                row = s[
                    s["target_function"].astype(str)
                    == target
                    ]

                if len(row) != 1:
                    raise SystemExit(
                        f"Missing/duplicate donor row: "
                        f"strategy={strategy} seed={seed} target={target}"
                    )

                r = row.iloc[0]

                selected = (
                        str(r["selection_status"])
                        == "selected"
                )

                donor = (
                    str(r["donor_function"])
                    if selected
                       and pd.notna(r["donor_function"])
                       and str(r["donor_function"])
                    else None
                )

                mapping[target] = donor

            maps[strategy][seed] = mapping

    return maps, seeds, targets


def build_prior(
        anchor_mean_reward_x86: float,
        donor_stats: dict[str, Any],
) -> dict[str, float]:
    donor_delta = (
            float(
                donor_stats[
                    "mean_reward"
                ]["arm64"]
            )
            - float(
        donor_stats[
            "mean_reward"
        ]["x86"]
    )
    )

    return {
        "x86": float(
            anchor_mean_reward_x86
        ),
        "arm64": float(
            anchor_mean_reward_x86
            + donor_delta
        ),
    }


def make_crn_samples(
        *,
        target: str,
        target_stats: dict[str, Any],
        replicates: int,
        max_horizon: int,
        anchor_n: int,
        base_seed: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    Separate RNG streams:
      anchor stream != online x86 stream != online ARM stream.
    """
    anchor_rng = np.random.default_rng(
        stable_seed(
            base_seed,
            target,
            "anchor-x86",
        )
    )

    anchor_durations = anchor_rng.choice(
        target_stats["durations"]["x86"],
        size=(
            replicates,
            anchor_n,
        ),
        replace=True,
    )

    anchor_mean_reward = np.mean(
        -np.log(
            anchor_durations
        ),
        axis=1,
    )

    online: dict[str, np.ndarray] = {}

    for arm in ARMS:
        rng = np.random.default_rng(
            stable_seed(
                base_seed,
                target,
                "online",
                arm,
            )
        )

        online[arm] = rng.choice(
            target_stats["durations"][arm],
            size=(
                replicates,
                max_horizon,
            ),
            replace=True,
        )

    return (
        anchor_mean_reward.astype(float),
        online,
    )


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


def policy_effective_mean(
        policy: Any,
        arm: str,
) -> float | None:
    if isinstance(
            policy,
            DecoupledUCB1Policy,
    ):
        return float(
            policy.effective_mean(arm)
        )

    return baseline_effective_mean(
        policy,
        arm,
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


def simulate_trace(
        *,
        target_stats: dict[str, Any],
        sequences: dict[str, np.ndarray],
        c: float,
        convergence_run: int,
        prior_mean_reward: dict[str, float] | None,
        reward_weight: float,
        exploration_weight: float,
) -> dict[str, Any]:
    if prior_mean_reward is None:
        policy: Any = UCB1Policy(
            c=c,
            prior_weight=0.0,
            prior_mean_reward=None,
        )
    else:
        policy = DecoupledUCB1Policy(
            c=c,
            reward_prior_weight=reward_weight,
            exploration_prior_weight=exploration_weight,
            prior_mean_reward=prior_mean_reward,
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

    chosen_arms: list[str] = []
    durations: list[float] = []
    rewards: list[float] = []
    pseudo_regret_steps: list[float] = []
    exploitation_correct: list[bool] = []

    horizon = len(
        sequences[ARMS[0]]
    )

    for _ in range(horizon):
        arm = str(
            policy.select_arm()
        )

        position = positions[arm]

        duration = float(
            sequences[arm][position]
        )

        positions[arm] += 1

        reward = float(
            policy.update(
                arm,
                duration,
            )
        )

        chosen_arms.append(
            arm
        )
        durations.append(
            duration
        )
        rewards.append(
            reward
        )

        pseudo_regret_steps.append(
            best_mean_reward
            - float(
                target_stats[
                    "mean_reward"
                ][arm]
            )
        )

        means = {
            candidate: policy_effective_mean(
                policy,
                candidate,
            )
            for candidate in ARMS
        }

        if all(
                means[arm_name]
                is not None
                for arm_name in ARMS
        ):
            exploitation_arm = max(
                ARMS,
                key=lambda candidate: (
                    float(
                        means[candidate]
                    ),
                    -ARMS.index(
                        candidate
                    ),
                ),
            )

            exploitation_correct.append(
                exploitation_arm
                == best_arm
            )
        else:
            exploitation_correct.append(
                False
            )

    return {
        "chosen_arms": chosen_arms,
        "durations": np.asarray(
            durations,
            dtype=float,
        ),
        "rewards": np.asarray(
            rewards,
            dtype=float,
        ),
        "pseudo_regret_steps": np.asarray(
            pseudo_regret_steps,
            dtype=float,
        ),
        "exploitation_correct": (
            exploitation_correct
        ),
        "best_arm": best_arm,
        "convergence_run": (
            convergence_run
        ),
    }


def prefix_metrics(
        trace: dict[str, Any],
        horizon: int,
) -> dict[str, float]:
    chosen = trace[
        "chosen_arms"
    ][:horizon]

    durations = trace[
        "durations"
    ][:horizon]

    pseudo_regret_steps = trace[
        "pseudo_regret_steps"
    ][:horizon]

    exploitation_correct = trace[
        "exploitation_correct"
    ][:horizon]

    best_arm = str(
        trace["best_arm"]
    )

    wrong = sum(
        1
        for arm in chosen
        if arm != best_arm
    )

    convergence = (
        first_completed_true_run(
            exploitation_correct,
            int(
                trace[
                    "convergence_run"
                ]
            ),
        )
    )

    if convergence is None:
        wrong_before_convergence = (
            wrong
        )
        censored_convergence = (
                horizon + 1
        )
        convergence_probability = (
            0.0
        )
    else:
        wrong_before_convergence = sum(
            1
            for arm in chosen[
                :convergence
            ]
            if arm != best_arm
        )
        censored_convergence = float(
            convergence
        )
        convergence_probability = (
            1.0
        )

    optimal_count = (
            len(chosen) - wrong
    )

    return {
        "cumulative_latency_ms": float(
            np.sum(
                durations
            )
        ),
        "optimal_arm_rate": float(
            optimal_count
            / horizon
        ),
        "wrong_selections": float(
            wrong
        ),
        "cumulative_reward_pseudo_regret": float(
            np.sum(
                pseudo_regret_steps
            )
        ),
        "first_arm_optimal": float(
            bool(chosen)
            and chosen[0] == best_arm
        ),
        "convergence_probability": (
            convergence_probability
        ),
        "censored_convergence_request": float(
            censored_convergence
        ),
        "wrong_before_convergence": float(
            wrong_before_convergence
        ),
    }


def aggregate_runs(
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


def compare_to_baseline(
        strategy: dict[str, float],
        baseline: dict[str, float],
) -> dict[str, float]:
    base_latency = float(
        baseline[
            "cumulative_latency_ms"
        ]
    )

    latency_gain = (
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
        if base_latency > 0
        else np.nan
    )

    return {
        "latency_gain_percent_vs_no_transfer": (
            latency_gain
        ),
        "optimal_arm_rate_gain": (
                float(
                    strategy[
                        "optimal_arm_rate"
                    ]
                )
                - float(
            baseline[
                "optimal_arm_rate"
            ]
        )
        ),
        "wrong_selections_saved": (
                float(
                    baseline[
                        "wrong_selections"
                    ]
                )
                - float(
            strategy[
                "wrong_selections"
            ]
        )
        ),
        "reward_pseudo_regret_saved": (
                float(
                    baseline[
                        "cumulative_reward_pseudo_regret"
                    ]
                )
                - float(
            strategy[
                "cumulative_reward_pseudo_regret"
            ]
        )
        ),
        "first_arm_optimal_gain": (
                float(
                    strategy[
                        "first_arm_optimal"
                    ]
                )
                - float(
            baseline[
                "first_arm_optimal"
            ]
        )
        ),
        "convergence_probability_gain": (
                float(
                    strategy[
                        "convergence_probability"
                    ]
                )
                - float(
            baseline[
                "convergence_probability"
            ]
        )
        ),
        "convergence_requests_saved": (
                float(
                    baseline[
                        "censored_convergence_request"
                    ]
                )
                - float(
            strategy[
                "censored_convergence_request"
            ]
        )
        ),
        "wrong_before_convergence_saved": (
                float(
                    baseline[
                        "wrong_before_convergence"
                    ]
                )
                - float(
            strategy[
                "wrong_before_convergence"
            ]
        )
        ),
    }


def target_level_summary(
        per_target_seed: pd.DataFrame,
) -> pd.DataFrame:
    numeric_cols = [
        column
        for column in per_target_seed.columns
        if column
           not in {
               "target_function",
               "strategy",
               "cluster_seed",
               "donor_function",
               "horizon",
           }
           and pd.api.types.is_numeric_dtype(
            per_target_seed[column]
        )
    ]

    return (
        per_target_seed.groupby(
            [
                "target_function",
                "strategy",
                "horizon",
            ],
            as_index=False,
        )[numeric_cols]
        .mean()
    )


def macro_summary(
        per_target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
            strategy,
            horizon,
    ), group in per_target.groupby(
        [
            "strategy",
            "horizon",
        ]
    ):
        row = {
            "strategy": strategy,
            "horizon": int(
                horizon
            ),
            "target_count": int(
                group[
                    "target_function"
                ].nunique()
            ),
        }

        for column in group.columns:
            if column in {
                "target_function",
                "strategy",
                "horizon",
            }:
                continue

            if pd.api.types.is_numeric_dtype(
                    group[column]
            ):
                row[
                    f"{column}_macro_mean"
                ] = float(
                    group[
                        column
                    ].mean()
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def target_bootstrap_pair(
        per_target: pd.DataFrame,
        strategy_a: str,
        strategy_b: str,
        horizon: int,
        replicates: int,
        seed: int,
) -> list[dict[str, Any]]:
    a = (
        per_target[
            (
                    per_target[
                        "strategy"
                    ]
                    == strategy_a
            )
            & (
                    per_target[
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
        per_target[
            (
                    per_target[
                        "strategy"
                    ]
                    == strategy_b
            )
            & (
                    per_target[
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

    delta_by_metric = {
        metric: np.asarray(
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
        for metric in metrics
    }

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for metric, deltas in delta_by_metric.items():
        means = np.empty(
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

            means[i] = float(
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
                        means,
                        0.025,
                    )
                ),
                "ci95_high": float(
                    np.quantile(
                        means,
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


def write_report(
        path: Path,
        summary: pd.DataFrame,
        paired: pd.DataFrame,
        *,
        c: float,
        reward_weight: float,
        exploration_weight: float,
        anchor_n: int,
        mc_replicates: int,
        mc_seed: int,
) -> None:
    lines = [
        "# Fixed-formula Transfer Learning bridge",
        "",
        "Clustering and donor rules are frozen; no tuning is performed here.",
        "",
        "## Frozen TL configuration",
        "",
        f"- c = {c}",
        f"- w_R = {reward_weight}",
        f"- w_E = {exploration_weight}",
        "- reward = -ln(duration_ms)",
        "- prior = reference-anchored donor architecture effect",
        f"- target x86 anchor bootstrap size = {anchor_n}",
        f"- Monte Carlo replicates = {mc_replicates}",
        f"- Monte Carlo seed = {mc_seed}",
        "- CRN shared across all strategies",
        "",
        "## Macro results",
        "",
    ]

    focus_cols = [
        "strategy",
        "horizon",
        "transfer_applied_macro_mean",
        "latency_gain_percent_vs_no_transfer_macro_mean",
        "optimal_arm_rate_macro_mean",
        "wrong_selections_macro_mean",
        "cumulative_reward_pseudo_regret_macro_mean",
        "first_arm_optimal_macro_mean",
        "convergence_probability_macro_mean",
        "censored_convergence_request_macro_mean",
    ]

    existing = [
        col
        for col in focus_cols
        if col in summary.columns
    ]

    for _, row in summary[
        existing
    ].sort_values(
        [
            "horizon",
            "strategy",
        ]
    ).iterrows():
        lines.append(
            "- "
            + ", ".join(
                (
                    f"{column}={row[column]:.4f}"
                    if isinstance(
                        row[column],
                        (float, np.floating),
                    )
                    else f"{column}={row[column]}"
                )
                for column in existing
            )
        )

    lines += [
        "",
        "## Primary paired comparison",
        "",
        (
            "`weighted_vote_p2 - professor_majority` uses target-level paired "
            "bootstrap confidence intervals. A positive latency-gain delta "
            "favors weighted-p2; a negative pseudo-regret delta favors "
            "weighted-p2."
        ),
        "",
    ]

    primary = paired[
        paired["comparison"]
        == "weighted_vote_p2_minus_professor_majority"
        ]

    for _, row in primary.sort_values(
            [
                "horizon",
                "metric",
            ]
    ).iterrows():
        lines.append(
            f"- H={int(row['horizon'])}, "
            f"{row['metric']}: "
            f"delta={row['point_delta']:.4f}, "
            f"95% CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        (
            "Do not retune TL parameters from this result. First determine "
            "whether the upstream donor improvement survives downstream under "
            "the already-frozen formula. Parameter tuning, if time permits, "
            "must be a separate experiment with predeclared grids and held-out "
            "or nested validation."
        ),
        "",
    ]

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def main(args) -> None:
    horizons = sorted(
        set(
            parse_int_list(
                args.horizons
            )
        )
    )

    max_horizon = max(
        horizons
    )

    raw_x86 = (
        args.raw_x86.resolve()
    )
    raw_arm64 = (
        args.raw_arm64.resolve()
    )
    donor_detail_path = (
        args.donor_detail.resolve()
    )

    x86 = load_eligible_durations(
        raw_x86
    )
    arm64 = load_eligible_durations(
        raw_arm64
    )

    performance = build_performance(
        x86,
        arm64,
    )

    detail = load_donor_detail(
        donor_detail_path
    )

    (
        maps,
        cluster_seeds,
        targets,
    ) = donor_maps(
        detail
    )

    missing_targets = [
        target
        for target in targets
        if target not in performance
    ]

    if missing_targets:
        raise SystemExit(
            "Targets missing raw performance data: "
            + ", ".join(
                missing_targets
            )
        )

    all_donors = sorted(
        {
            donor
            for strategy_maps in maps.values()
            for target_map in strategy_maps.values()
            for donor in target_map.values()
            if donor
        }
    )

    missing_donors = [
        donor
        for donor in all_donors
        if donor not in performance
    ]

    if missing_donors:
        raise SystemExit(
            "Donors missing raw performance data: "
            + ", ".join(
                missing_donors
            )
        )

    sample_check_rows = []

    for target in sorted(
            set(targets)
            | set(all_donors)
    ):
        sample_check_rows.append(
            {
                "function_name": target,
                "x86_samples": int(
                    len(
                        performance[
                            target
                        ][
                            "durations"
                        ][
                            "x86"
                        ]
                    )
                ),
                "arm64_samples": int(
                    len(
                        performance[
                            target
                        ][
                            "durations"
                        ][
                            "arm64"
                        ]
                    )
                ),
            }
        )

    bad_samples = [
        row
        for row in sample_check_rows
        if row["x86_samples"] < 2
           or row["arm64_samples"] < 2
    ]

    if bad_samples:
        raise SystemExit(
            "Insufficient empirical samples: "
            + json.dumps(
                bad_samples[:10]
            )
        )

    per_target_seed_rows: list[
        dict[str, Any]
    ] = []

    total_targets = len(
        targets
    )

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
            replicates=args.replicates,
            max_horizon=max_horizon,
            anchor_n=args.anchor_samples,
            base_seed=args.mc_seed,
        )

        baseline_runs_by_horizon: dict[
            int,
            list[dict[str, float]],
        ] = {
            horizon: []
            for horizon in horizons
        }

        baseline_traces = []

        for rep in range(
                args.replicates
        ):
            sequences = {
                arm: online[
                    arm
                ][rep]
                for arm in ARMS
            }

            trace = simulate_trace(
                target_stats=target_stats,
                sequences=sequences,
                c=args.c,
                convergence_run=args.convergence_run,
                prior_mean_reward=None,
                reward_weight=args.reward_weight,
                exploration_weight=args.exploration_weight,
            )

            baseline_traces.append(
                trace
            )

            for horizon in horizons:
                baseline_runs_by_horizon[
                    horizon
                ].append(
                    prefix_metrics(
                        trace,
                        horizon,
                    )
                )

        baseline_agg = {
            horizon: aggregate_runs(
                rows
            )
            for horizon, rows in (
                baseline_runs_by_horizon.items()
            )
        }

        # One baseline row per clustering seed keeps paired tables rectangular.
        for cluster_seed in cluster_seeds:
            for horizon in horizons:
                base = baseline_agg[
                    horizon
                ]

                per_target_seed_rows.append(
                    {
                        "target_function": target,
                        "strategy": (
                            "no_transfer"
                        ),
                        "cluster_seed": int(
                            cluster_seed
                        ),
                        "donor_function": "",
                        "transfer_applied": 0.0,
                        "horizon": int(
                            horizon
                        ),
                        **base,
                        "latency_gain_percent_vs_no_transfer": 0.0,
                        "optimal_arm_rate_gain": 0.0,
                        "wrong_selections_saved": 0.0,
                        "reward_pseudo_regret_saved": 0.0,
                        "first_arm_optimal_gain": 0.0,
                        "convergence_probability_gain": 0.0,
                        "convergence_requests_saved": 0.0,
                        "wrong_before_convergence_saved": 0.0,
                    }
                )

        for strategy in (
                "professor_majority",
                "weighted_vote_p2",
                "local_k5_majority",
        ):
            for cluster_seed in cluster_seeds:
                donor = maps[
                    strategy
                ][
                    cluster_seed
                ][
                    target
                ]

                # Abstention => exact no-transfer fallback.
                if donor is None:
                    for horizon in horizons:
                        base = baseline_agg[
                            horizon
                        ]

                        per_target_seed_rows.append(
                            {
                                "target_function": target,
                                "strategy": strategy,
                                "cluster_seed": int(
                                    cluster_seed
                                ),
                                "donor_function": "",
                                "transfer_applied": 0.0,
                                "horizon": int(
                                    horizon
                                ),
                                **base,
                                "latency_gain_percent_vs_no_transfer": 0.0,
                                "optimal_arm_rate_gain": 0.0,
                                "wrong_selections_saved": 0.0,
                                "reward_pseudo_regret_saved": 0.0,
                                "first_arm_optimal_gain": 0.0,
                                "convergence_probability_gain": 0.0,
                                "convergence_requests_saved": 0.0,
                                "wrong_before_convergence_saved": 0.0,
                            }
                        )
                    continue

                donor_stats = performance[
                    donor
                ]

                strategy_runs_by_horizon: dict[
                    int,
                    list[dict[str, float]],
                ] = {
                    horizon: []
                    for horizon in horizons
                }

                for rep in range(
                        args.replicates
                ):
                    sequences = {
                        arm: online[
                            arm
                        ][rep]
                        for arm in ARMS
                    }

                    prior = build_prior(
                        anchor_mean_reward_x86=float(
                            anchor_rewards[
                                rep
                            ]
                        ),
                        donor_stats=donor_stats,
                    )

                    trace = simulate_trace(
                        target_stats=target_stats,
                        sequences=sequences,
                        c=args.c,
                        convergence_run=args.convergence_run,
                        prior_mean_reward=prior,
                        reward_weight=args.reward_weight,
                        exploration_weight=args.exploration_weight,
                    )

                    for horizon in horizons:
                        strategy_runs_by_horizon[
                            horizon
                        ].append(
                            prefix_metrics(
                                trace,
                                horizon,
                            )
                        )

                for horizon in horizons:
                    strategy_agg = (
                        aggregate_runs(
                            strategy_runs_by_horizon[
                                horizon
                            ]
                        )
                    )

                    baseline = baseline_agg[
                        horizon
                    ]

                    comparisons = (
                        compare_to_baseline(
                            strategy_agg,
                            baseline,
                        )
                    )

                    per_target_seed_rows.append(
                        {
                            "target_function": target,
                            "strategy": strategy,
                            "cluster_seed": int(
                                cluster_seed
                            ),
                            "donor_function": donor,
                            "transfer_applied": 1.0,
                            "horizon": int(
                                horizon
                            ),
                            **strategy_agg,
                            **comparisons,
                        }
                    )

        if (
                target_idx % 5 == 0
                or target_idx
                == total_targets
        ):
            print(
                f"target_progress="
                f"{target_idx}/"
                f"{total_targets}"
            )

    per_target_seed = pd.DataFrame(
        per_target_seed_rows
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_target_seed_path = (
            args.output_dir
            / "tl-fixed-formula-per-target-seed.csv"
    )

    per_target_seed.to_csv(
        per_target_seed_path,
        index=False,
    )

    per_target = (
        target_level_summary(
            per_target_seed
        )
    )

    per_target_path = (
            args.output_dir
            / "tl-fixed-formula-per-target.csv"
    )

    per_target.to_csv(
        per_target_path,
        index=False,
    )

    summary = macro_summary(
        per_target
    )

    order = {
        strategy: idx
        for idx, strategy in enumerate(
            STRATEGY_ORDER
        )
    }

    summary[
        "_strategy_order"
    ] = summary[
        "strategy"
    ].map(order)

    summary = (
        summary.sort_values(
            [
                "horizon",
                "_strategy_order",
            ]
        )
        .drop(
            columns=[
                "_strategy_order"
            ]
        )
    )

    summary_path = (
            args.output_dir
            / "tl-fixed-formula-summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    paired_rows: list[
        dict[str, Any]
    ] = []

    comparisons = [
        (
            "professor_majority",
            "weighted_vote_p2",
        ),
        (
            "professor_majority",
            "local_k5_majority",
        ),
        (
            "no_transfer",
            "professor_majority",
        ),
        (
            "no_transfer",
            "weighted_vote_p2",
        ),
    ]

    for (
            strategy_a,
            strategy_b,
    ) in comparisons:
        for horizon in horizons:
            paired_rows.extend(
                target_bootstrap_pair(
                    per_target=per_target,
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
            / "tl-fixed-formula-paired-bootstrap.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    report_path = (
            args.output_dir
            / "tl-fixed-formula-report.md"
    )

    write_report(
        report_path,
        summary,
        paired,
        c=args.c,
        reward_weight=args.reward_weight,
        exploration_weight=args.exploration_weight,
        anchor_n=args.anchor_samples,
        mc_replicates=args.replicates,
        mc_seed=args.mc_seed,
    )

    manifest = {
        "experiment": (
            "fixed-formula donor-selection downstream bridge"
        ),
        "clustering_status": (
            "frozen; no fit or tuning in this script"
        ),
        "strategies": (
            STRATEGY_ORDER
        ),
        "strategy_variant_mapping": (
            STRATEGY_TO_VARIANT
        ),
        "transfer": {
            "policy": (
                "UCB1Decoupled"
            ),
            "reward": (
                "-ln(duration_ms)"
            ),
            "reference_arm": (
                "x86"
            ),
            "prior": (
                "reference-anchored donor architecture effect"
            ),
            "reward_weight_wR": (
                args.reward_weight
            ),
            "exploration_weight_wE": (
                args.exploration_weight
            ),
            "c": args.c,
            "convergence_run": (
                args.convergence_run
            ),
        },
        "simulation": {
            "horizons": horizons,
            "replicates": (
                args.replicates
            ),
            "mc_seed": (
                args.mc_seed
            ),
            "anchor_samples": (
                args.anchor_samples
            ),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "abstention_fallback": (
                "exact no-transfer"
            ),
            "cluster_seeds": (
                cluster_seeds
            ),
        },
        "bootstrap": {
            "unit": (
                "target_function"
            ),
            "replicates": (
                args.bootstrap_replicates
            ),
            "seed": (
                args.bootstrap_seed
            ),
            "ci": (
                "percentile 95%"
            ),
        },
        "sample_counts": (
            sample_check_rows
        ),
        "inputs": {
            "raw_x86": {
                "path": str(
                    raw_x86
                ),
                "sha256": (
                    sha256_file(
                        raw_x86
                    )
                ),
            },
            "raw_arm64": {
                "path": str(
                    raw_arm64
                ),
                "sha256": (
                    sha256_file(
                        raw_arm64
                    )
                ),
            },
            "donor_detail": {
                "path": str(
                    donor_detail_path
                ),
                "sha256": (
                    sha256_file(
                        donor_detail_path
                    )
                ),
            },
        },
    }

    manifest_path = (
            args.output_dir
            / "tl-fixed-formula-manifest.json"
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

    focus_columns = [
        "strategy",
        "horizon",
        "transfer_applied_macro_mean",
        "latency_gain_percent_vs_no_transfer_macro_mean",
        "optimal_arm_rate_macro_mean",
        "wrong_selections_macro_mean",
        "cumulative_reward_pseudo_regret_macro_mean",
        "first_arm_optimal_macro_mean",
        "convergence_probability_macro_mean",
        "censored_convergence_request_macro_mean",
    ]

    print()
    print(
        "FIXED-FORMULA TL SUMMARY"
    )

    print(
        summary[
            focus_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "WEIGHTED-P2 MINUS PROFESSOR-MAJORITY"
    )

    primary = paired[
        paired["comparison"]
        == "weighted_vote_p2_minus_professor_majority"
        ]

    print(
        primary[
            primary[
                "metric"
            ].isin(
                {
                    "latency_gain_percent_vs_no_transfer",
                    "optimal_arm_rate",
                    "wrong_selections",
                    "cumulative_reward_pseudo_regret",
                    "first_arm_optimal",
                    "convergence_probability",
                    "censored_convergence_request",
                }
            )
        ]
        .sort_values(
            [
                "horizon",
                "metric",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"per_target_seed="
        f"{per_target_seed_path}"
    )
    print(
        f"per_target="
        f"{per_target_path}"
    )
    print(
        f"summary="
        f"{summary_path}"
    )
    print(
        f"paired="
        f"{paired_path}"
    )
    print(
        f"report="
        f"{report_path}"
    )
    print(
        f"manifest="
        f"{manifest_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen donor-selection rules downstream using the "
            "existing Serverledge UCB1Decoupled formula without parameter tuning."
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
        "--horizons",
        default="5,10,20,50",
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--mc-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--anchor-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--c",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--reward-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=1.0,
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
