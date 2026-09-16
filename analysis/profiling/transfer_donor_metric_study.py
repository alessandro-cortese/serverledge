#!/usr/bin/env python3
"""
Alternative donor-ranking study for Serverledge transfer learning.

This experiment keeps BOTH the clustering configuration and the current UCB1
transfer formula unchanged. Only the way the donor is ranked inside the
assigned cluster changes.

For every outer target function:
  1. remove the target from the known-function catalogue;
  2. fit the frozen clustering configuration on the remaining functions;
  3. assign the target using PAPER-5 x86 features only;
  4. rank candidate donors INSIDE the assigned cluster using several metrics;
  5. evaluate donor quality using ARM data only after donor selection;
  6. replay the current UCB1 transfer formula with w={0.25,0.5,1.0}.

The target's measured x86 mean reward may be used by the "profile-ref" metric,
because a genuinely new function is already profiled on the reference x86
architecture before transfer. No ARM observation of the target is used for
clustering or donor selection.

Rankers
-------
native
    Frozen clustering-native distance:
      K-Means -> Euclidean in MinMax-scaled PAPER-5
      DBSCAN  -> cosine in Robust-scaled PAPER-5

euclidean
    Euclidean distance in the clustering algorithm's scaled PAPER-5 space.

manhattan
    Manhattan distance in the clustering algorithm's scaled PAPER-5 space.

cosine
    Cosine distance in the clustering algorithm's scaled PAPER-5 space.

mahalanobis
    Shrinkage Mahalanobis distance (Ledoit-Wolf) inside the assigned cluster.
    Falls back to native distance when covariance estimation is not meaningful.

profile-ref
    Transfer-aware but deployable distance. Combines:
      - the native PAPER-5 profile distance;
      - |target_x86_mean_reward - donor_x86_mean_reward|.

    Both components are normalized by the median candidate distance/gap inside
    the assigned cluster, then combined as Euclidean norm:
        sqrt(profile_norm^2 + ref_reward_norm^2)

    This tests whether matching the target's measured reference-architecture
    performance improves donor quality without using ARM information.

Operational convergence metric
------------------------------
UCB1 intentionally continues exploring, so "convergence" is not defined as
"never choose the other arm again". Instead we record:

  exploitation_run5_update:
      the completed target update that closes the first run of 5 consecutive
      updates for which the exploitation estimate ranks the empirically
      optimal architecture correctly.

We also report suboptimal selections in the first 5/10/20 requests.

This is a local diagnostic. The final GCP experiment should report both
requests-to-convergence and wall-clock seconds-to-convergence using the same
predeclared operational criterion.
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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    DEFAULT_HORIZONS,
    DEFAULT_WEIGHTS,
    UCB1Policy,
    empirical_stats,
    load_raw_durations,
    stable_seed,
)

PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

RANKERS = (
    "native",
    "euclidean",
    "manhattan",
    "cosine",
    "mahalanobis",
    "profile-ref",
)

ALGORITHMS = ("kmeans", "dbscan")
EPS = 1e-12


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


def load_profiles(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    required = {"function_name", *PAPER5_FEATURES}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing profile columns: {sorted(missing)}")

    rows.sort(key=lambda row: row["function_name"])
    names = [row["function_name"] for row in rows]
    x = np.asarray(
        [
            [float(row[f]) for f in PAPER5_FEATURES]
            for row in rows
        ],
        dtype=float,
    )
    return names, x


def dbscan_eps(
    x_train: np.ndarray,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> float:
    nn = NearestNeighbors(
        n_neighbors=min_samples,
        metric=metric,
    )
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_train)
    return float(np.quantile(distances[:, -1], eps_quantile))


def assign_cluster(
    algorithm: str,
    names: list[str],
    x: np.ndarray,
    target_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask = np.arange(len(names)) != target_index
    train_global_indices = np.flatnonzero(mask)
    train_names = np.asarray(names, dtype=object)[mask]
    x_train_raw = x[mask]
    x_target_raw = x[target_index : target_index + 1]

    if algorithm == "kmeans":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_target = scaler.transform(x_target_raw)

        model = KMeans(
            n_clusters=args.k,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        labels = model.fit_predict(x_train)
        target_cluster = int(model.predict(x_target)[0])
        candidate_local = np.flatnonzero(labels == target_cluster)

        return {
            "status": "selected",
            "reason": "",
            "scaler": "minmax",
            "native_metric": "euclidean",
            "target_cluster": target_cluster,
            "x_train": x_train,
            "x_target": x_target[0],
            "train_names": train_names,
            "train_global_indices": train_global_indices,
            "candidate_local_indices": candidate_local,
            "labels": labels,
            "eps": "",
            "nearest_core_distance": "",
        }

    if algorithm == "dbscan":
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_target = scaler.transform(x_target_raw)

        eps = dbscan_eps(
            x_train,
            min_samples=args.dbscan_min_samples,
            eps_quantile=args.dbscan_eps_quantile,
            metric=args.dbscan_metric,
        )
        model = DBSCAN(
            eps=eps,
            min_samples=args.dbscan_min_samples,
            metric=args.dbscan_metric,
        )
        labels = model.fit_predict(x_train)
        core_local = np.asarray(model.core_sample_indices_, dtype=int)

        if len(core_local) == 0:
            return {
                "status": "no-transfer",
                "reason": "no_core_points",
                "eps": eps,
            }

        core_distances = pairwise_distances(
            x_target,
            x_train[core_local],
            metric=args.dbscan_metric,
        )[0]
        within = np.flatnonzero(core_distances <= eps + EPS)

        if len(within) == 0:
            return {
                "status": "no-transfer",
                "reason": "no_core_within_eps",
                "eps": eps,
                "nearest_core_distance": float(np.min(core_distances)),
            }

        candidate_core = core_local[within]
        candidate_core_distances = core_distances[within]
        candidate_core_names = train_names[candidate_core]
        order = np.lexsort(
            (candidate_core_names, candidate_core_distances)
        )
        nearest_core_local = int(candidate_core[int(order[0])])
        target_cluster = int(labels[nearest_core_local])
        candidate_local = np.flatnonzero(labels == target_cluster)

        return {
            "status": "selected",
            "reason": "",
            "scaler": "robust",
            "native_metric": args.dbscan_metric,
            "target_cluster": target_cluster,
            "x_train": x_train,
            "x_target": x_target[0],
            "train_names": train_names,
            "train_global_indices": train_global_indices,
            "candidate_local_indices": candidate_local,
            "labels": labels,
            "eps": eps,
            "nearest_core_distance": float(
                candidate_core_distances[int(order[0])]
            ),
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def safe_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    finite = finite[finite > EPS]
    if len(finite) == 0:
        return 1.0
    return float(np.median(finite))


def metric_distances(
    ranker: str,
    assignment: dict[str, Any],
    target_name: str,
    performance: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_local = assignment["candidate_local_indices"]
    candidates = assignment["x_train"][candidate_local]
    target = assignment["x_target"].reshape(1, -1)
    candidate_names = assignment["train_names"][candidate_local]

    native_metric = assignment["native_metric"]

    if ranker == "native":
        distances = pairwise_distances(
            target,
            candidates,
            metric=native_metric,
        )[0]
        return distances, {"metric_detail": native_metric}

    if ranker == "euclidean":
        distances = pairwise_distances(
            target,
            candidates,
            metric="euclidean",
        )[0]
        return distances, {"metric_detail": "euclidean"}

    if ranker == "manhattan":
        distances = pairwise_distances(
            target,
            candidates,
            metric="manhattan",
        )[0]
        return distances, {"metric_detail": "manhattan"}

    if ranker == "cosine":
        distances = pairwise_distances(
            target,
            candidates,
            metric="cosine",
        )[0]
        return distances, {"metric_detail": "cosine"}

    if ranker == "mahalanobis":
        if len(candidates) < 2:
            distances = pairwise_distances(
                target,
                candidates,
                metric=native_metric,
            )[0]
            return distances, {
                "metric_detail": f"fallback-{native_metric}",
            }

        try:
            lw = LedoitWolf().fit(candidates)
            precision = lw.precision_
            diff = candidates - target[0]
            squared = np.einsum(
                "ij,jk,ik->i",
                diff,
                precision,
                diff,
            )
            squared = np.maximum(squared, 0.0)
            distances = np.sqrt(squared)
            return distances, {
                "metric_detail": "ledoitwolf-mahalanobis",
            }
        except Exception:
            distances = pairwise_distances(
                target,
                candidates,
                metric=native_metric,
            )[0]
            return distances, {
                "metric_detail": f"fallback-{native_metric}",
            }

    if ranker == "profile-ref":
        profile = pairwise_distances(
            target,
            candidates,
            metric=native_metric,
        )[0]

        target_ref_reward = float(
            performance[target_name]["mean_reward"]["x86"]
        )
        candidate_ref_rewards = np.asarray(
            [
                performance[str(name)]["mean_reward"]["x86"]
                for name in candidate_names
            ],
            dtype=float,
        )
        ref_gap = np.abs(
            candidate_ref_rewards - target_ref_reward
        )

        profile_scale = safe_scale(profile)
        ref_scale = safe_scale(ref_gap)

        profile_norm = profile / profile_scale
        ref_norm = ref_gap / ref_scale

        distances = np.sqrt(
            profile_norm**2 + ref_norm**2
        )

        return distances, {
            "metric_detail": "native-profile-plus-x86-reward",
            "profile_scale": profile_scale,
            "reference_reward_scale": ref_scale,
        }

    raise ValueError(f"Unsupported ranker: {ranker}")


def choose_donor(
    ranker: str,
    assignment: dict[str, Any],
    target_name: str,
    performance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    distances, detail = metric_distances(
        ranker,
        assignment,
        target_name,
        performance,
    )

    candidate_local = assignment["candidate_local_indices"]
    candidate_names = assignment["train_names"][candidate_local]

    order = np.lexsort((candidate_names, distances))
    position = int(order[0])
    donor_local = int(candidate_local[position])
    donor_name = str(assignment["train_names"][donor_local])

    return {
        "donor_function": donor_name,
        "donor_distance": float(distances[position]),
        "candidate_count": len(candidate_local),
        **detail,
    }


def reward_gap(stats: dict[str, Any]) -> float:
    return float(
        stats["mean_reward"]["x86"]
        - stats["mean_reward"]["arm64"]
    )


def first_true_run(values: list[bool], run_length: int) -> int | None:
    """Return the 1-based update that COMPLETES the first correct run."""
    if run_length <= 0:
        raise ValueError("run_length must be positive")
    streak = 0
    for idx, value in enumerate(values, start=1):
        if value:
            streak += 1
            if streak >= run_length:
                return idx
        else:
            streak = 0
    return None


def effective_mean(policy: UCB1Policy, arm: str) -> float | None:
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


def simulate_trace(
    target_stats: dict[str, Any],
    donor_stats: dict[str, Any] | None,
    sequences: dict[str, np.ndarray],
    horizon: int,
    c: float,
    weight: float,
    convergence_run: int,
) -> dict[str, Any]:
    prior_mean = (
        donor_stats["mean_reward"]
        if donor_stats is not None
        else None
    )

    policy = UCB1Policy(
        c=c,
        prior_weight=(weight if prior_mean is not None else 0.0),
        prior_mean_reward=prior_mean,
    )

    positions = {arm: 0 for arm in ARMS}
    chosen: list[str] = []
    exploitation_correct: list[bool] = []

    cumulative_latency = 0.0
    cumulative_reward = 0.0

    best_arm = target_stats["best_reward_arm"]
    best_reward = target_stats["mean_reward"][best_arm]

    cumulative_reward_regret = 0.0

    for _ in range(horizon):
        arm = policy.select_arm()
        duration = float(
            sequences[arm][positions[arm]]
        )
        positions[arm] += 1
        reward = policy.update(arm, duration)

        chosen.append(arm)
        cumulative_latency += duration
        cumulative_reward += reward
        cumulative_reward_regret += (
            best_reward
            - target_stats["mean_reward"][arm]
        )

        means = {
            candidate: effective_mean(policy, candidate)
            for candidate in ARMS
        }
        known = all(
            means[arm_name] is not None
            for arm_name in ARMS
        )
        if known:
            exploit_arm = max(
                ARMS,
                key=lambda arm_name: (
                    means[arm_name],
                    -ARMS.index(arm_name),
                ),
            )
            exploitation_correct.append(
                exploit_arm == best_arm
            )
        else:
            exploitation_correct.append(False)

    convergence_start = first_true_run(
        exploitation_correct,
        convergence_run,
    )

    return {
        "chosen_arms": chosen,
        "cumulative_latency": cumulative_latency,
        "cumulative_reward": cumulative_reward,
        "cumulative_reward_regret": cumulative_reward_regret,
        "exploitation_correct": exploitation_correct,
        "convergence_start": convergence_start,
    }


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


def evaluate_transfer(
    target_name: str,
    donor_name: str | None,
    performance: dict[str, dict[str, Any]],
    weights: list[float],
    horizons: list[int],
    max_horizon: int,
    replicates: int,
    c: float,
    seed: int,
    convergence_run: int,
) -> list[dict[str, Any]]:
    target_stats = performance[target_name]
    donor_stats = (
        performance[donor_name]
        if donor_name
        else None
    )

    sequences = make_sequences(
        target_name,
        target_stats,
        max_horizon,
        replicates,
        seed,
    )

    baseline = [
        simulate_trace(
            target_stats,
            None,
            sequences[i],
            max_horizon,
            c,
            0.0,
            convergence_run,
        )
        for i in range(replicates)
    ]

    output = []

    for weight in weights:
        transfer = [
            simulate_trace(
                target_stats,
                donor_stats,
                sequences[i],
                max_horizon,
                c,
                weight,
                convergence_run,
            )
            for i in range(replicates)
        ]

        for horizon in horizons:
            # Re-run only the prefix for convergence/choice metrics to avoid
            # accidentally using future updates when H < max_horizon.
            if horizon == max_horizon:
                base_h = baseline
                transfer_h = transfer
            else:
                base_h = [
                    simulate_trace(
                        target_stats,
                        None,
                        {
                            arm: sequences[i][arm][:horizon]
                            for arm in ARMS
                        },
                        horizon,
                        c,
                        0.0,
                        convergence_run,
                    )
                    for i in range(replicates)
                ]
                transfer_h = [
                    simulate_trace(
                        target_stats,
                        donor_stats,
                        {
                            arm: sequences[i][arm][:horizon]
                            for arm in ARMS
                        },
                        horizon,
                        c,
                        weight,
                        convergence_run,
                    )
                    for i in range(replicates)
                ]

            base_latency = np.asarray(
                [r["cumulative_latency"] for r in base_h],
                dtype=float,
            )
            transfer_latency = np.asarray(
                [r["cumulative_latency"] for r in transfer_h],
                dtype=float,
            )
            latency_gain = 100.0 * (
                base_latency - transfer_latency
            ) / base_latency

            best_arm = target_stats["best_reward_arm"]
            wrong_counts = np.asarray(
                [
                    sum(
                        1
                        for arm in r["chosen_arms"]
                        if arm != best_arm
                    )
                    for r in transfer_h
                ],
                dtype=float,
            )
            base_wrong_counts = np.asarray(
                [
                    sum(
                        1
                        for arm in r["chosen_arms"]
                        if arm != best_arm
                    )
                    for r in base_h
                ],
                dtype=float,
            )

            convergence = [
                r["convergence_start"]
                for r in transfer_h
                if r["convergence_start"] is not None
            ]
            baseline_convergence = [
                r["convergence_start"]
                for r in base_h
                if r["convergence_start"] is not None
            ]

            output.append(
                {
                    "equivalent_observation_weight": weight,
                    "horizon": horizon,
                    "mean_latency_gain_pct_vs_no_transfer": float(
                        np.mean(latency_gain)
                    ),
                    "mean_wrong_choice_count": float(
                        np.mean(wrong_counts)
                    ),
                    "mean_baseline_wrong_choice_count": float(
                        np.mean(base_wrong_counts)
                    ),
                    "mean_wrong_choices_saved": float(
                        np.mean(
                            base_wrong_counts - wrong_counts
                        )
                    ),
                    "convergence_probability": float(
                        len(convergence) / replicates
                    ),
                    "mean_exploitation_convergence_request": (
                        float(np.mean(convergence))
                        if convergence
                        else float("nan")
                    ),
                    "median_exploitation_convergence_request": (
                        float(np.median(convergence))
                        if convergence
                        else float("nan")
                    ),
                    "baseline_convergence_probability": float(
                        len(baseline_convergence) / replicates
                    ),
                    "mean_baseline_exploitation_convergence_request": (
                        float(np.mean(baseline_convergence))
                        if baseline_convergence
                        else float("nan")
                    ),
                    "mean_reward_pseudo_regret": float(
                        np.mean(
                            [
                                r["cumulative_reward_regret"]
                                for r in transfer_h
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


def summarize_donors(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for algorithm in ALGORITHMS:
        for ranker in RANKERS:
            group = [
                r
                for r in rows
                if r["algorithm"] == algorithm
                and r["ranker"] == ranker
            ]
            selected = [
                r for r in group if r["selection_status"] == "selected"
            ]
            matches = [
                1.0 if r["best_reward_arm_agreement"] else 0.0
                for r in selected
            ]
            gap_errors = [
                float(r["abs_reward_gap_error"])
                for r in selected
            ]
            out.append(
                {
                    "algorithm": algorithm,
                    "ranker": ranker,
                    "target_count": len(group),
                    "selected_count": len(selected),
                    "coverage": (
                        len(selected) / len(group)
                        if group else float("nan")
                    ),
                    "best_reward_arm_agreement_rate": (
                        float(np.mean(matches))
                        if matches else float("nan")
                    ),
                    "mean_abs_reward_gap_error": (
                        float(np.mean(gap_errors))
                        if gap_errors else float("nan")
                    ),
                    "median_abs_reward_gap_error": (
                        float(np.median(gap_errors))
                        if gap_errors else float("nan")
                    ),
                }
            )
    return out


def summarize_transfer(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    keys = sorted(
        {
            (
                r["algorithm"],
                r["ranker"],
                float(r["equivalent_observation_weight"]),
                int(r["horizon"]),
            )
            for r in rows
        }
    )

    for algorithm, ranker, weight, horizon in keys:
        group = [
            r
            for r in rows
            if r["algorithm"] == algorithm
            and r["ranker"] == ranker
            and float(r["equivalent_observation_weight"]) == weight
            and int(r["horizon"]) == horizon
        ]
        selected = [
            r for r in group if r["transfer_applied"]
        ]

        gains = np.asarray(
            [
                float(r["mean_latency_gain_pct_vs_no_transfer"])
                if r["transfer_applied"]
                else 0.0
                for r in group
            ],
            dtype=float,
        )
        wrong_saved = np.asarray(
            [
                float(r["mean_wrong_choices_saved"])
                if r["transfer_applied"]
                else 0.0
                for r in group
            ],
            dtype=float,
        )

        ci_low, ci_high = bootstrap_mean_ci(
            gains,
            stable_seed(
                seed,
                f"{algorithm}-{ranker}-{weight}-{horizon}",
            ),
        )

        convergence_values = [
            float(r["mean_exploitation_convergence_request"])
            for r in selected
            if math.isfinite(
                float(r["mean_exploitation_convergence_request"])
            )
        ]
        baseline_convergence_values = [
            float(r["mean_baseline_exploitation_convergence_request"])
            for r in selected
            if math.isfinite(
                float(r["mean_baseline_exploitation_convergence_request"])
            )
        ]

        out.append(
            {
                "algorithm": algorithm,
                "ranker": ranker,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(group),
                "transfer_applied_count": len(selected),
                "coverage": (
                    len(selected) / len(group)
                    if group else float("nan")
                ),
                "macro_mean_latency_gain_pct": float(
                    np.mean(gains)
                ),
                "macro_latency_gain_ci95_low": ci_low,
                "macro_latency_gain_ci95_high": ci_high,
                "macro_mean_wrong_choices_saved": float(
                    np.mean(wrong_saved)
                ),
                "macro_mean_exploitation_convergence_request": (
                    float(np.mean(convergence_values))
                    if convergence_values else float("nan")
                ),
                "macro_mean_baseline_exploitation_convergence_request": (
                    float(np.mean(baseline_convergence_values))
                    if baseline_convergence_values else float("nan")
                ),
                "macro_mean_convergence_probability": float(
                    np.mean(
                        [
                            float(r["convergence_probability"])
                            if r["transfer_applied"] else
                            float(r["baseline_convergence_probability"])
                            for r in group
                        ]
                    )
                ),
                "macro_mean_baseline_convergence_probability": float(
                    np.mean(
                        [
                            float(r["baseline_convergence_probability"])
                            for r in group
                        ]
                    )
                ),
            }
        )

    return out


def plot_donor_quality(
    summary: list[dict[str, Any]],
    figures: Path,
) -> None:
    for algorithm in ALGORITHMS:
        rows = [
            r for r in summary
            if r["algorithm"] == algorithm
        ]
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.bar(
            [r["ranker"] for r in rows],
            [
                float(r["best_reward_arm_agreement_rate"])
                for r in rows
            ],
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Best-arm agreement rate")
        ax.set_title(
            f"{algorithm.upper()} donor ranking quality"
        )
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figures / f"{algorithm}-donor-ranker-agreement.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures / f"{algorithm}-donor-ranker-agreement.svg",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--raw-x86", required=True, type=Path)
    p.add_argument("--raw-arm64", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--c", type=float, default=0.8)
    p.add_argument("--weight", action="append", type=float)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--replicates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convergence-run", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dbscan-min-samples", type=int, default=4)
    p.add_argument("--dbscan-eps-quantile", type=float, default=0.80)
    p.add_argument("--dbscan-metric", default="cosine")
    args = p.parse_args()

    weights = args.weight or DEFAULT_WEIGHTS
    horizons = [
        h for h in DEFAULT_HORIZONS
        if h <= args.horizon
    ]
    if args.horizon not in horizons:
        horizons.append(args.horizon)
        horizons.sort()

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    names, x = load_profiles(args.profiles.resolve())
    raw_x86 = load_raw_durations(args.raw_x86.resolve())
    raw_arm = load_raw_durations(args.raw_arm64.resolve())
    performance = empirical_stats(raw_x86, raw_arm)

    if set(names) != set(performance):
        raise RuntimeError(
            "Profile/performance mismatch: "
            f"missing={sorted(set(names) - set(performance))}, "
            f"extra={sorted(set(performance) - set(names))}"
        )

    donor_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []

    for target_index, target_name in enumerate(names):
        print(f"[{target_index + 1:02d}/{len(names)}] {target_name}")

        for algorithm in ALGORITHMS:
            assignment = assign_cluster(
                algorithm,
                names,
                x,
                target_index,
                args,
            )

            if assignment["status"] != "selected":
                for ranker in RANKERS:
                    donor_rows.append(
                        {
                            "algorithm": algorithm,
                            "ranker": ranker,
                            "target_function": target_name,
                            "selection_status": "no-transfer",
                            "selection_reason": assignment["reason"],
                            "target_cluster": -1,
                            "candidate_count": 0,
                            "donor_function": "",
                            "donor_distance": "",
                            "metric_detail": "",
                            "best_reward_arm_agreement": "",
                            "abs_reward_gap_error": "",
                        }
                    )
                    evaluations = evaluate_transfer(
                        target_name,
                        None,
                        performance,
                        weights,
                        horizons,
                        args.horizon,
                        args.replicates,
                        args.c,
                        args.seed,
                        args.convergence_run,
                    )
                    for result in evaluations:
                        transfer_rows.append(
                            {
                                "algorithm": algorithm,
                                "ranker": ranker,
                                "target_function": target_name,
                                "donor_function": "",
                                "transfer_applied": False,
                                **result,
                            }
                        )
                continue

            for ranker in RANKERS:
                donor = choose_donor(
                    ranker,
                    assignment,
                    target_name,
                    performance,
                )
                donor_name = donor["donor_function"]

                target_stats = performance[target_name]
                donor_stats = performance[donor_name]

                arm_match = (
                    target_stats["best_reward_arm"]
                    == donor_stats["best_reward_arm"]
                )
                gap_error = abs(
                    reward_gap(target_stats)
                    - reward_gap(donor_stats)
                )

                donor_rows.append(
                    {
                        "algorithm": algorithm,
                        "ranker": ranker,
                        "target_function": target_name,
                        "selection_status": "selected",
                        "selection_reason": "",
                        "target_cluster": assignment["target_cluster"],
                        "candidate_count": donor["candidate_count"],
                        "donor_function": donor_name,
                        "donor_distance": donor["donor_distance"],
                        "metric_detail": donor["metric_detail"],
                        "best_reward_arm_agreement": arm_match,
                        "abs_reward_gap_error": gap_error,
                        "target_x86_mean_reward": target_stats[
                            "mean_reward"
                        ]["x86"],
                        "donor_x86_mean_reward": donor_stats[
                            "mean_reward"
                        ]["x86"],
                        "target_reward_gap_x86_minus_arm": reward_gap(
                            target_stats
                        ),
                        "donor_reward_gap_x86_minus_arm": reward_gap(
                            donor_stats
                        ),
                    }
                )

                evaluations = evaluate_transfer(
                    target_name,
                    donor_name,
                    performance,
                    weights,
                    horizons,
                    args.horizon,
                    args.replicates,
                    args.c,
                    args.seed,
                    args.convergence_run,
                )
                for result in evaluations:
                    transfer_rows.append(
                        {
                            "algorithm": algorithm,
                            "ranker": ranker,
                            "target_function": target_name,
                            "donor_function": donor_name,
                            "transfer_applied": True,
                            **result,
                        }
                    )

    donor_rows.sort(
        key=lambda r: (
            r["algorithm"],
            r["ranker"],
            r["target_function"],
        )
    )
    transfer_rows.sort(
        key=lambda r: (
            r["algorithm"],
            r["ranker"],
            float(r["equivalent_observation_weight"]),
            int(r["horizon"]),
            r["target_function"],
        )
    )

    donor_summary = summarize_donors(donor_rows)
    transfer_summary = summarize_transfer(
        transfer_rows,
        seed=args.seed,
    )

    write_csv(
        out / "donor-metric-per-target.csv",
        donor_rows,
    )
    write_csv(
        out / "donor-metric-summary.csv",
        donor_summary,
    )
    write_csv(
        out / "donor-metric-transfer-per-target.csv",
        transfer_rows,
    )
    write_csv(
        out / "donor-metric-transfer-summary.csv",
        transfer_summary,
    )

    plot_donor_quality(
        donor_summary,
        figures,
    )

    manifest = {
        "schema_version": 1,
        "experiment": "alternative donor ranking inside frozen clusters",
        "features": PAPER5_FEATURES,
        "rankers": list(RANKERS),
        "selection_constraints": (
            "target excluded before scaler/clustering fit; donor always chosen "
            "inside target's assigned cluster; no target ARM information used"
        ),
        "profile_ref_metric": (
            "combines clustering-native PAPER-5 distance with measured x86 "
            "mean-reward similarity; both normalized within candidate cluster"
        ),
        "kmeans": {
            "scaler": "minmax",
            "k": args.k,
            "cluster_metric": "euclidean",
        },
        "dbscan": {
            "scaler": "robust",
            "cluster_metric": args.dbscan_metric,
            "min_samples": args.dbscan_min_samples,
            "eps_quantile": args.dbscan_eps_quantile,
            "native_abstention": True,
        },
        "ucb1": {
            "formula": "current implementation unchanged",
            "c": args.c,
            "weights": weights,
            "dynamic_c": False,
            "prior_reset": False,
            "prior_values": "raw donor mean rewards, unchanged",
        },
        "convergence": {
            "criterion": (
                "first update starting a run of N consecutive updates where "
                "the exploitation-only effective mean ranks the empirical "
                "best architecture correctly"
            ),
            "run_length": args.convergence_run,
            "note": (
                "UCB1 continues exploration after learning; therefore "
                "convergence is defined on exploitation estimates, not as "
                "permanent arm selection"
            ),
        },
        "evaluation": {
            "horizons": horizons,
            "replicates": args.replicates,
            "paired_empirical_bootstrap": True,
        },
        "outputs": [
            "donor-metric-per-target.csv",
            "donor-metric-summary.csv",
            "donor-metric-transfer-per-target.csv",
            "donor-metric-transfer-summary.csv",
            "figures/",
        ],
    }

    (out / "donor-metric-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\noutput={out}")
    print("\nDONOR QUALITY")
    for row in donor_summary:
        print(
            f"{row['algorithm']:<7} "
            f"{row['ranker']:<12} "
            f"coverage={float(row['coverage']):.4f} "
            f"arm-match="
            f"{float(row['best_reward_arm_agreement_rate']):.4f} "
            f"median-gap-error="
            f"{float(row['median_abs_reward_gap_error']):.4f}"
        )

    print("\nTRANSFER @ H=10")
    primary = [
        row
        for row in transfer_summary
        if int(row["horizon"]) == 10
    ]
    primary.sort(
        key=lambda row: (
            row["algorithm"],
            row["ranker"],
            float(row["equivalent_observation_weight"]),
        )
    )

    for row in primary:
        print(
            f"{row['algorithm']:<7} "
            f"{row['ranker']:<12} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"gain={float(row['macro_mean_latency_gain_pct']):+.4f}% "
            f"CI95=["
            f"{float(row['macro_latency_gain_ci95_low']):+.4f},"
            f"{float(row['macro_latency_gain_ci95_high']):+.4f}] "
            f"wrong-saved="
            f"{float(row['macro_mean_wrong_choices_saved']):+.4f} "
            f"conv="
            f"{float(row['macro_mean_exploitation_convergence_request']):.3f} "
            f"baseline-conv="
            f"{float(row['macro_mean_baseline_exploitation_convergence_request']):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
