#!/usr/bin/env python3
"""
Raw-donor prior vs reference-anchored prior for Serverledge UCB1 transfer.

This experiment intentionally keeps:
- the clustering result,
- the donor chosen inside the cluster,
- the UCB1 transfer formula,
- c = 0.8,
- equivalent-observation weights w = {0.25, 0.5, 1.0}

UNCHANGED.

Only the knowledge transferred from the selected donor is changed.

Prior modes
-----------
raw
    Current behavior:
        prior_x86 = donor_mean_reward_x86
        prior_arm = donor_mean_reward_arm

reference-anchored
    Use the real x86 profiling result of the new target as the reference scale,
    and transfer only the donor's x86->ARM architectural effect:
        donor_delta = donor_mean_reward_arm - donor_mean_reward_x86

        prior_x86 = target_mean_reward_x86
        prior_arm = target_mean_reward_x86 + donor_delta

Because reward = -ln(duration_ms), donor_delta is the log performance ratio
between architectures. Thus this prior transfers the donor's architectural
effect rather than the donor's absolute execution-time scale.

The target ARM observations are NEVER used to construct the prior. They are
used only after selection for offline evaluation.

Donor rankers
-------------
By default we test the three K-Means donor strategies already identified by
the previous experiment:
    native
    manhattan
    profile-ref

The donor mapping is read from donor-metric-per-target.csv, so this experiment
does not re-fit or re-select donors. This isolates PRIOR CONSTRUCTION from
DONOR SELECTION.

Convergence
-----------
UCB1 intentionally keeps exploring, therefore convergence is defined on the
exploitation estimate, not on permanent arm selection.

A run converges when the exploitation-only effective mean ranks the empirical
best architecture correctly for N consecutive completed updates.

We report:
- convergence probability by horizon;
- mean convergence request among converged runs;
- censored convergence request, assigning H+1 to runs not converged by H;
- convergence requests saved vs no-transfer;
- wrong selections over the horizon;
- wrong selections before convergence;
- cumulative latency gain and reward pseudo-regret.

For the final GCP experiment, requests-to-convergence should be accompanied by
wall-clock seconds-to-convergence measured with the same predeclared rule.
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
    DEFAULT_WEIGHTS,
    UCB1Policy,
    empirical_stats,
    load_raw_durations,
    stable_seed,
)

DEFAULT_RANKERS = ("native", "manhattan", "profile-ref")
PRIOR_MODES = ("raw", "reference-anchored")


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


def build_prior_mean_reward(
    prior_mode: str,
    target_stats: dict[str, Any],
    donor_stats: dict[str, Any],
) -> dict[str, float]:
    donor_x86 = float(donor_stats["mean_reward"]["x86"])
    donor_arm = float(donor_stats["mean_reward"]["arm64"])

    if prior_mode == "raw":
        return {
            "x86": donor_x86,
            "arm64": donor_arm,
        }

    if prior_mode == "reference-anchored":
        target_x86 = float(target_stats["mean_reward"]["x86"])
        donor_delta = donor_arm - donor_x86

        return {
            "x86": target_x86,
            "arm64": target_x86 + donor_delta,
        }

    raise ValueError(f"Unsupported prior mode: {prior_mode}")


def first_completed_true_run(
    values: list[bool],
    run_length: int,
) -> int | None:
    """Return the 1-based update completing the first all-True run."""
    if run_length <= 0:
        raise ValueError("run_length must be > 0")

    streak = 0
    for idx, value in enumerate(values, start=1):
        if value:
            streak += 1
            if streak >= run_length:
                return idx
        else:
            streak = 0

    return None


def effective_mean(
    policy: UCB1Policy,
    arm: str,
) -> float | None:
    count = policy.effective_count(arm)
    if count <= 0:
        return None

    return float(
        (
            policy.reward_sums[arm]
            + policy.prior_reward_sums[arm]
        )
        / count
    )


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


def simulate_trace(
    target_stats: dict[str, Any],
    prior_mean_reward: dict[str, float] | None,
    sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    prior_weight: float,
    convergence_run: int,
) -> dict[str, Any]:
    policy = UCB1Policy(
        c=c,
        prior_weight=(
            prior_weight
            if prior_mean_reward is not None
            else 0.0
        ),
        prior_mean_reward=prior_mean_reward,
    )

    positions = {arm: 0 for arm in ARMS}
    chosen_arms: list[str] = []
    exploitation_correct: list[bool] = []

    cumulative_latency = 0.0
    cumulative_reward = 0.0
    cumulative_reward_regret = 0.0

    best_arm = target_stats["best_reward_arm"]
    best_mean_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    for _ in range(horizon):
        arm = policy.select_arm()
        duration = float(
            sequences[arm][positions[arm]]
        )
        positions[arm] += 1

        reward = policy.update(arm, duration)

        chosen_arms.append(arm)
        cumulative_latency += duration
        cumulative_reward += reward
        cumulative_reward_regret += (
            best_mean_reward
            - float(target_stats["mean_reward"][arm])
        )

        means = {
            candidate: effective_mean(
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
                    means[candidate],
                    -ARMS.index(candidate),
                ),
            )
            exploitation_correct.append(
                exploitation_arm == best_arm
            )
        else:
            exploitation_correct.append(False)

    convergence_request = first_completed_true_run(
        exploitation_correct,
        convergence_run,
    )

    if convergence_request is None:
        wrong_before_convergence = sum(
            1
            for arm in chosen_arms
            if arm != best_arm
        )
    else:
        wrong_before_convergence = sum(
            1
            for arm in chosen_arms[:convergence_request]
            if arm != best_arm
        )

    return {
        "chosen_arms": chosen_arms,
        "cumulative_latency": cumulative_latency,
        "cumulative_reward": cumulative_reward,
        "cumulative_reward_regret": cumulative_reward_regret,
        "convergence_request": convergence_request,
        "wrong_before_convergence": wrong_before_convergence,
        "first_arm_optimal": (
            chosen_arms[0] == best_arm
            if chosen_arms
            else False
        ),
    }


def donor_mapping(
    donor_rows: list[dict[str, str]],
    algorithm: str,
    rankers: list[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)

    for row in donor_rows:
        if row["algorithm"] != algorithm:
            continue
        if row["ranker"] not in rankers:
            continue

        target = row["target_function"]
        ranker = row["ranker"]

        if ranker in result[target]:
            raise RuntimeError(
                f"Duplicate donor row for {target}/{ranker}"
            )

        result[target][ranker] = row

    return dict(result)


def evaluate_one(
    target_name: str,
    donor_name: str,
    ranker: str,
    prior_mode: str,
    target_stats: dict[str, Any],
    donor_stats: dict[str, Any],
    sequences_by_rep: list[dict[str, np.ndarray]],
    baseline_cache: dict[int, list[dict[str, Any]]],
    weights: list[float],
    horizons: list[int],
    c: float,
    convergence_run: int,
) -> list[dict[str, Any]]:
    prior_mean = build_prior_mean_reward(
        prior_mode=prior_mode,
        target_stats=target_stats,
        donor_stats=donor_stats,
    )

    donor_delta = float(
        donor_stats["mean_reward"]["arm64"]
        - donor_stats["mean_reward"]["x86"]
    )
    target_delta = float(
        target_stats["mean_reward"]["arm64"]
        - target_stats["mean_reward"]["x86"]
    )

    prior_best_arm = max(
        ARMS,
        key=lambda arm: (
            prior_mean[arm],
            -ARMS.index(arm),
        ),
    )

    output: list[dict[str, Any]] = []

    for weight in weights:
        for horizon in horizons:
            baseline = baseline_cache[horizon]

            transfer = [
                simulate_trace(
                    target_stats=target_stats,
                    prior_mean_reward=prior_mean,
                    sequences={
                        arm: sequences_by_rep[i][arm][:horizon]
                        for arm in ARMS
                    },
                    horizon=horizon,
                    c=c,
                    prior_weight=weight,
                    convergence_run=convergence_run,
                )
                for i in range(len(sequences_by_rep))
            ]

            baseline_latency = np.asarray(
                [
                    row["cumulative_latency"]
                    for row in baseline
                ],
                dtype=float,
            )
            transfer_latency = np.asarray(
                [
                    row["cumulative_latency"]
                    for row in transfer
                ],
                dtype=float,
            )

            latency_gain = 100.0 * (
                baseline_latency - transfer_latency
            ) / baseline_latency

            best_arm = target_stats["best_reward_arm"]

            baseline_wrong = np.asarray(
                [
                    sum(
                        1
                        for arm in row["chosen_arms"]
                        if arm != best_arm
                    )
                    for row in baseline
                ],
                dtype=float,
            )
            transfer_wrong = np.asarray(
                [
                    sum(
                        1
                        for arm in row["chosen_arms"]
                        if arm != best_arm
                    )
                    for row in transfer
                ],
                dtype=float,
            )

            baseline_wrong_before = np.asarray(
                [
                    row["wrong_before_convergence"]
                    for row in baseline
                ],
                dtype=float,
            )
            transfer_wrong_before = np.asarray(
                [
                    row["wrong_before_convergence"]
                    for row in transfer
                ],
                dtype=float,
            )

            baseline_conv = [
                row["convergence_request"]
                for row in baseline
            ]
            transfer_conv = [
                row["convergence_request"]
                for row in transfer
            ]

            baseline_censored = np.asarray(
                [
                    value
                    if value is not None
                    else horizon + 1
                    for value in baseline_conv
                ],
                dtype=float,
            )
            transfer_censored = np.asarray(
                [
                    value
                    if value is not None
                    else horizon + 1
                    for value in transfer_conv
                ],
                dtype=float,
            )

            transfer_converged = [
                value
                for value in transfer_conv
                if value is not None
            ]
            baseline_converged = [
                value
                for value in baseline_conv
                if value is not None
            ]

            output.append(
                {
                    "target_function": target_name,
                    "donor_function": donor_name,
                    "ranker": ranker,
                    "prior_mode": prior_mode,
                    "equivalent_observation_weight": weight,
                    "horizon": horizon,
                    "target_best_reward_arm": best_arm,
                    "donor_best_reward_arm": donor_stats[
                        "best_reward_arm"
                    ],
                    "donor_best_arm_match": (
                        donor_stats["best_reward_arm"]
                        == best_arm
                    ),
                    "prior_best_reward_arm": prior_best_arm,
                    "prior_best_arm_match": (
                        prior_best_arm == best_arm
                    ),
                    "target_x86_mean_reward": float(
                        target_stats["mean_reward"]["x86"]
                    ),
                    "donor_x86_mean_reward": float(
                        donor_stats["mean_reward"]["x86"]
                    ),
                    "donor_arch_effect_arm_minus_x86": donor_delta,
                    "target_arch_effect_arm_minus_x86": target_delta,
                    "prior_x86_mean_reward": float(
                        prior_mean["x86"]
                    ),
                    "prior_arm64_mean_reward": float(
                        prior_mean["arm64"]
                    ),
                    "mean_latency_gain_pct_vs_no_transfer": float(
                        np.mean(latency_gain)
                    ),
                    "mean_wrong_choice_count": float(
                        np.mean(transfer_wrong)
                    ),
                    "mean_baseline_wrong_choice_count": float(
                        np.mean(baseline_wrong)
                    ),
                    "mean_wrong_choices_saved": float(
                        np.mean(
                            baseline_wrong - transfer_wrong
                        )
                    ),
                    "mean_wrong_before_convergence": float(
                        np.mean(transfer_wrong_before)
                    ),
                    "mean_baseline_wrong_before_convergence": float(
                        np.mean(baseline_wrong_before)
                    ),
                    "mean_wrong_before_convergence_saved": float(
                        np.mean(
                            baseline_wrong_before
                            - transfer_wrong_before
                        )
                    ),
                    "convergence_probability": float(
                        sum(
                            value is not None
                            for value in transfer_conv
                        )
                        / len(transfer_conv)
                    ),
                    "baseline_convergence_probability": float(
                        sum(
                            value is not None
                            for value in baseline_conv
                        )
                        / len(baseline_conv)
                    ),
                    "mean_convergence_request_if_converged": (
                        float(np.mean(transfer_converged))
                        if transfer_converged
                        else float("nan")
                    ),
                    "mean_baseline_convergence_request_if_converged": (
                        float(np.mean(baseline_converged))
                        if baseline_converged
                        else float("nan")
                    ),
                    "mean_censored_convergence_request": float(
                        np.mean(transfer_censored)
                    ),
                    "mean_baseline_censored_convergence_request": float(
                        np.mean(baseline_censored)
                    ),
                    "mean_convergence_requests_saved": float(
                        np.mean(
                            baseline_censored
                            - transfer_censored
                        )
                    ),
                    "first_arm_optimal_probability": float(
                        np.mean(
                            [
                                1.0
                                if row["first_arm_optimal"]
                                else 0.0
                                for row in transfer
                            ]
                        )
                    ),
                    "mean_reward_pseudo_regret": float(
                        np.mean(
                            [
                                row["cumulative_reward_regret"]
                                for row in transfer
                            ]
                        )
                    ),
                }
            )

    return output


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
        means[i] = np.mean(
            rng.choice(
                values,
                size=len(values),
                replace=True,
            )
        )

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
                row["ranker"],
                row["prior_mode"],
                float(
                    row[
                        "equivalent_observation_weight"
                    ]
                ),
                int(row["horizon"]),
            )
            for row in rows
        }
    )

    for ranker, prior_mode, weight, horizon in keys:
        group = [
            row
            for row in rows
            if row["ranker"] == ranker
            and row["prior_mode"] == prior_mode
            and float(
                row[
                    "equivalent_observation_weight"
                ]
            )
            == weight
            and int(row["horizon"]) == horizon
        ]

        gains = np.asarray(
            [
                float(
                    row[
                        "mean_latency_gain_pct_vs_no_transfer"
                    ]
                )
                for row in group
            ],
            dtype=float,
        )

        low, high = bootstrap_mean_ci(
            gains,
            stable_seed(
                seed,
                f"{ranker}-{prior_mode}-{weight}-{horizon}",
            ),
        )

        output.append(
            {
                "ranker": ranker,
                "prior_mode": prior_mode,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(group),
                "donor_best_arm_agreement_rate": float(
                    np.mean(
                        [
                            1.0
                            if row["donor_best_arm_match"]
                            else 0.0
                            for row in group
                        ]
                    )
                ),
                "prior_best_arm_agreement_rate": float(
                    np.mean(
                        [
                            1.0
                            if row["prior_best_arm_match"]
                            else 0.0
                            for row in group
                        ]
                    )
                ),
                "macro_mean_latency_gain_pct": float(
                    np.mean(gains)
                ),
                "macro_latency_gain_ci95_low": low,
                "macro_latency_gain_ci95_high": high,
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


def compare_prior_modes(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    index = {
        (
            row["target_function"],
            row["ranker"],
            row["prior_mode"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
            int(row["horizon"]),
        ): row
        for row in rows
    }

    keys = sorted(
        {
            (
                row["ranker"],
                float(
                    row[
                        "equivalent_observation_weight"
                    ]
                ),
                int(row["horizon"]),
            )
            for row in rows
        }
    )

    targets = sorted(
        {row["target_function"] for row in rows}
    )

    for ranker, weight, horizon in keys:
        gain_delta = []
        wrong_delta = []
        wrong_before_delta = []
        convergence_saved_delta = []

        for target in targets:
            raw = index[
                (
                    target,
                    ranker,
                    "raw",
                    weight,
                    horizon,
                )
            ]
            anchored = index[
                (
                    target,
                    ranker,
                    "reference-anchored",
                    weight,
                    horizon,
                )
            ]

            gain_delta.append(
                float(
                    anchored[
                        "mean_latency_gain_pct_vs_no_transfer"
                    ]
                )
                - float(
                    raw[
                        "mean_latency_gain_pct_vs_no_transfer"
                    ]
                )
            )
            wrong_delta.append(
                float(
                    anchored[
                        "mean_wrong_choices_saved"
                    ]
                )
                - float(
                    raw[
                        "mean_wrong_choices_saved"
                    ]
                )
            )
            wrong_before_delta.append(
                float(
                    anchored[
                        "mean_wrong_before_convergence_saved"
                    ]
                )
                - float(
                    raw[
                        "mean_wrong_before_convergence_saved"
                    ]
                )
            )
            convergence_saved_delta.append(
                float(
                    anchored[
                        "mean_convergence_requests_saved"
                    ]
                )
                - float(
                    raw[
                        "mean_convergence_requests_saved"
                    ]
                )
            )

        gain_delta_arr = np.asarray(
            gain_delta,
            dtype=float,
        )
        low, high = bootstrap_mean_ci(
            gain_delta_arr,
            stable_seed(
                seed,
                f"anchored-minus-raw-{ranker}-{weight}-{horizon}",
            ),
        )

        output.append(
            {
                "ranker": ranker,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(targets),
                "anchored_minus_raw_latency_gain_pct": float(
                    np.mean(gain_delta_arr)
                ),
                "anchored_minus_raw_latency_gain_ci95_low": low,
                "anchored_minus_raw_latency_gain_ci95_high": high,
                "anchored_minus_raw_wrong_choices_saved": float(
                    np.mean(
                        np.asarray(
                            wrong_delta,
                            dtype=float,
                        )
                    )
                ),
                "anchored_minus_raw_wrong_before_convergence_saved": float(
                    np.mean(
                        np.asarray(
                            wrong_before_delta,
                            dtype=float,
                        )
                    )
                ),
                "anchored_minus_raw_convergence_requests_saved": float(
                    np.mean(
                        np.asarray(
                            convergence_saved_delta,
                            dtype=float,
                        )
                    )
                ),
            }
        )

    return output


def plot_primary(
    summary: list[dict[str, Any]],
    figures: Path,
    primary_horizon: int,
) -> None:
    for ranker in DEFAULT_RANKERS:
        rows = [
            row
            for row in summary
            if row["ranker"] == ranker
            and int(row["horizon"]) == primary_horizon
        ]

        fig, ax = plt.subplots(figsize=(8.5, 5.8))

        labels = []
        values = []

        for prior_mode in PRIOR_MODES:
            for row in sorted(
                [
                    candidate
                    for candidate in rows
                    if candidate["prior_mode"]
                    == prior_mode
                ],
                key=lambda candidate: float(
                    candidate[
                        "equivalent_observation_weight"
                    ]
                ),
            ):
                labels.append(
                    f"{prior_mode}\nw={float(row['equivalent_observation_weight']):g}"
                )
                values.append(
                    float(
                        row[
                            "macro_mean_latency_gain_pct"
                        ]
                    )
                )

        ax.bar(labels, values)
        ax.axhline(
            0.0,
            linewidth=1,
            linestyle="--",
        )
        ax.set_title(
            f"{ranker}: raw vs reference-anchored prior @ H={primary_horizon}"
        )
        ax.set_ylabel(
            "Mean latency gain vs no-transfer (%)"
        )
        ax.tick_params(
            axis="x",
            rotation=25,
        )
        ax.grid(
            True,
            axis="y",
            alpha=0.2,
        )
        fig.tight_layout()
        fig.savefig(
            figures
            / f"{ranker}-raw-vs-anchored-latency-h{primary_horizon}.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures
            / f"{ranker}-raw-vs-anchored-latency-h{primary_horizon}.svg",
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
        action="append",
    )

    p.add_argument(
        "--c",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--weight",
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

    rankers = args.ranker or list(DEFAULT_RANKERS)
    weights = args.weight or DEFAULT_WEIGHTS

    horizons = [
        h
        for h in DEFAULT_HORIZONS
        if h <= args.horizon
    ]
    if args.horizon not in horizons:
        horizons.append(args.horizon)
        horizons.sort()

    donor_rows = read_csv(
        args.donor_metric_results.resolve()
    )
    mapping = donor_mapping(
        donor_rows,
        algorithm=args.algorithm,
        rankers=rankers,
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

    expected_targets = set(performance)
    if set(mapping) != expected_targets:
        raise RuntimeError(
            "Donor mapping/performance mismatch: "
            f"missing={sorted(expected_targets - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected_targets)}"
        )

    for target in sorted(mapping):
        missing_rankers = set(rankers) - set(
            mapping[target]
        )
        if missing_rankers:
            raise RuntimeError(
                f"{target} missing rankers "
                f"{sorted(missing_rankers)}"
            )

        for ranker in rankers:
            row = mapping[target][ranker]
            if row["selection_status"] != "selected":
                raise RuntimeError(
                    f"{args.algorithm}/{ranker}/{target} "
                    "has no donor. This experiment expects "
                    "the K-Means full-coverage donor study."
                )

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict[str, Any]] = []

    for target_idx, target_name in enumerate(
        sorted(performance),
        start=1,
    ):
        print(
            f"[{target_idx:02d}/{len(performance)}] "
            f"{target_name}"
        )

        target_stats = performance[target_name]

        sequences = make_sequences(
            target_name=target_name,
            target_stats=target_stats,
            horizon=args.horizon,
            replicates=args.replicates,
            seed=args.seed,
        )

        baseline_cache: dict[
            int,
            list[dict[str, Any]],
        ] = {}

        for horizon in horizons:
            baseline_cache[horizon] = [
                simulate_trace(
                    target_stats=target_stats,
                    prior_mean_reward=None,
                    sequences={
                        arm: sequences[i][arm][:horizon]
                        for arm in ARMS
                    },
                    horizon=horizon,
                    c=args.c,
                    prior_weight=0.0,
                    convergence_run=args.convergence_run,
                )
                for i in range(args.replicates)
            ]

        for ranker in rankers:
            donor_row = mapping[target_name][ranker]
            donor_name = donor_row["donor_function"]
            donor_stats = performance[donor_name]

            for prior_mode in PRIOR_MODES:
                results.extend(
                    evaluate_one(
                        target_name=target_name,
                        donor_name=donor_name,
                        ranker=ranker,
                        prior_mode=prior_mode,
                        target_stats=target_stats,
                        donor_stats=donor_stats,
                        sequences_by_rep=sequences,
                        baseline_cache=baseline_cache,
                        weights=weights,
                        horizons=horizons,
                        c=args.c,
                        convergence_run=args.convergence_run,
                    )
                )

    results.sort(
        key=lambda row: (
            row["ranker"],
            row["prior_mode"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
            int(row["horizon"]),
            row["target_function"],
        )
    )

    summary = summarize(
        results,
        seed=args.seed,
    )
    comparison = compare_prior_modes(
        results,
        seed=args.seed,
    )

    write_csv(
        out / "reference-anchored-prior-per-target.csv",
        results,
    )
    write_csv(
        out / "reference-anchored-prior-summary.csv",
        summary,
    )
    write_csv(
        out / "reference-anchored-vs-raw.csv",
        comparison,
    )

    plot_primary(
        summary=summary,
        figures=figures,
        primary_horizon=args.primary_horizon,
    )

    manifest = {
        "schema_version": 1,
        "experiment": (
            "raw donor prior vs reference-anchored "
            "architecture-effect prior"
        ),
        "algorithm": args.algorithm,
        "rankers": rankers,
        "prior_modes": {
            "raw": (
                "prior arm means are the donor's absolute "
                "mean rewards"
            ),
            "reference-anchored": (
                "prior_x86 = target measured x86 mean reward; "
                "prior_arm = target_x86 + "
                "(donor_arm - donor_x86)"
            ),
        },
        "important_methodological_note": (
            "Target x86 performance is available because the new "
            "function is profiled on the reference architecture "
            "before transfer. Target ARM observations are used "
            "only for post-hoc evaluation."
        ),
        "ucb1": {
            "formula": "current formula unchanged",
            "c": args.c,
            "weights": weights,
            "dynamic_c": False,
            "prior_reset": False,
        },
        "convergence": {
            "run_length": args.convergence_run,
            "definition": (
                "completed request closing the first run of N "
                "consecutive updates where exploitation-only "
                "effective means rank the empirical best "
                "architecture correctly"
            ),
            "censoring": (
                "runs not converged by horizon H are assigned "
                "H+1 for the censored convergence metric"
            ),
        },
        "evaluation": {
            "horizons": horizons,
            "replicates_per_target": args.replicates,
            "seed": args.seed,
            "paired_empirical_sequences": True,
            "reward": "-ln(duration_ms)",
        },
        "outputs": [
            "reference-anchored-prior-per-target.csv",
            "reference-anchored-prior-summary.csv",
            "reference-anchored-vs-raw.csv",
            "figures/",
        ],
    }

    (
        out
        / "reference-anchored-prior-manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\noutput={out}")

    primary = [
        row
        for row in summary
        if int(row["horizon"])
        == args.primary_horizon
    ]
    primary.sort(
        key=lambda row: (
            row["ranker"],
            row["prior_mode"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
        )
    )

    print(
        f"\nRAW VS REFERENCE-ANCHORED "
        f"@ H={args.primary_horizon}"
    )
    for row in primary:
        print(
            f"{row['ranker']:<12} "
            f"{row['prior_mode']:<18} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"prior-match="
            f"{float(row['prior_best_arm_agreement_rate']):.4f} "
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
        f"\nANCHORED - RAW DIRECT COMPARISON "
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
            row["ranker"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
        )
    )

    for row in direct:
        print(
            f"{row['ranker']:<12} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"Δgain="
            f"{float(row['anchored_minus_raw_latency_gain_pct']):+.4f}pp "
            f"CI95=["
            f"{float(row['anchored_minus_raw_latency_gain_ci95_low']):+.4f},"
            f"{float(row['anchored_minus_raw_latency_gain_ci95_high']):+.4f}] "
            f"Δwrong-saved="
            f"{float(row['anchored_minus_raw_wrong_choices_saved']):+.4f} "
            f"Δwrong-before-saved="
            f"{float(row['anchored_minus_raw_wrong_before_convergence_saved']):+.4f} "
            f"Δconv-saved="
            f"{float(row['anchored_minus_raw_convergence_requests_saved']):+.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
