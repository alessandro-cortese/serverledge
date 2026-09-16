#!/usr/bin/env python3
"""
Nested leave-one-function-out confidence-gated transfer experiment.

Goal
----
Evaluate whether cluster-relative confidence can make clustering-derived
transfer safer WITHOUT modifying the current Serverledge UCB1 formula.

Outer fold:
    one function is treated as genuinely new.

Inner calibration:
    using only the remaining known functions, repeat leave-one-function-out
    donor selection and calibrate a confidence gate from historical donor
    correctness. The outer target never participates in threshold selection.

Clustering and donor choice remain x86/PAPER-5 only. ARM observations are used
only:
    1. on INNER known functions to calibrate the gate from historical outcomes;
    2. after the OUTER decision to evaluate the held-out target.

Frozen clustering configurations
--------------------------------
K-Means:
    PAPER-5 + MinMaxScaler + K=5 + Euclidean
    donor = nearest known member inside predicted cluster

DBSCAN:
    PAPER-5 + RobustScaler + cosine
    min_samples=4
    eps = q0.80 of training-only k-distances
    target cluster = nearest core point within eps
    donor = nearest known member inside assigned cluster
    otherwise native abstention

Cluster-relative confidence
---------------------------
K-Means:
    centroid_ratio =
        target->centroid distance /
        q90(training member->centroid distances in that cluster)

    donor_ratio =
        target->donor distance /
        q90(training within-cluster 1-NN distances)

    confidence_score = max(centroid_ratio, donor_ratio)

    Lower is better. This prevents raw Euclidean distances from being treated
    as comparable across clusters/folds.

DBSCAN:
    confidence_score =
        target->nearest-core cosine distance / eps

    Lower is better. The number of supporting core points within eps is also
    exported as a diagnostic but is not folded into an arbitrary scalar.

Nested gate calibration
-----------------------
For every OUTER fold and algorithm, INNER donor selections are evaluated at
predeclared score quantiles {25, 50, 75, 90, 100}%.

For a candidate gate:
    utility = (correct_transfers - incorrect_transfers) / inner_target_count
            = coverage * (2 * best_arm_accuracy - 1)

This rewards both correctness and useful coverage. "Abstain from all transfer"
with utility 0 is always a candidate, so a gate is selected only when the
training functions support positive historical evidence.

The gate is selected without using the outer target's ARM behavior.

The experiment reports both:
    - ungated: current clustering-derived donor policy;
    - nested-gated: same donor, accepted only by the nested confidence gate.

The UCB1 formula, c=0.8, and equivalent-observation weights are unchanged.
No dynamic c_t and no prior reset are introduced.
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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    DEFAULT_HORIZONS,
    DEFAULT_WEIGHTS,
    empirical_stats,
    load_raw_durations,
    simulate_policy,
    stable_seed,
)

PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

GATE_QUANTILES = [0.25, 0.50, 0.75, 0.90, 1.00]
MODES = ("ungated", "nested-gated")
EPSILON = 1e-12


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
        raise RuntimeError(f"No profiles in {path}")

    required = {"function_name", *PAPER5_FEATURES}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing profile columns: {sorted(missing)}")

    rows.sort(key=lambda row: row["function_name"])
    names = [row["function_name"] for row in rows]
    x = np.asarray(
        [
            [float(row[feature]) for feature in PAPER5_FEATURES]
            for row in rows
        ],
        dtype=float,
    )
    return names, x


def safe_ratio(value: float, reference: float) -> float:
    if reference > EPSILON:
        return float(value / reference)
    if value <= EPSILON:
        return 0.0
    return float("inf")


def q90(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, 0.90))


def kmeans_confident_select(
    names: list[str],
    x: np.ndarray,
    train_indices: np.ndarray,
    target_index: int,
    k: int,
    n_init: int,
    random_state: int,
) -> dict[str, Any]:
    train_names = np.asarray(names, dtype=object)[train_indices]
    x_train_raw = x[train_indices]
    x_target_raw = x[target_index : target_index + 1]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_target = scaler.transform(x_target_raw)

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(x_train)
    target_cluster = int(model.predict(x_target)[0])
    centroid = model.cluster_centers_[target_cluster]

    member_indices = np.flatnonzero(labels == target_cluster)
    member_matrix = x_train[member_indices]
    member_names = train_names[member_indices]

    donor_distances = np.linalg.norm(
        member_matrix - x_target[0],
        axis=1,
    )
    donor_order = np.lexsort((member_names, donor_distances))
    donor_pos = int(donor_order[0])
    donor_local_index = int(member_indices[donor_pos])
    donor_name = str(train_names[donor_local_index])
    donor_distance = float(donor_distances[donor_pos])

    target_centroid_distance = float(
        np.linalg.norm(x_target[0] - centroid)
    )
    member_centroid_distances = np.linalg.norm(
        member_matrix - centroid,
        axis=1,
    )
    centroid_reference_q90 = q90(member_centroid_distances)
    centroid_ratio = safe_ratio(
        target_centroid_distance,
        centroid_reference_q90,
    )

    if len(member_matrix) >= 2:
        pairwise = pairwise_distances(
            member_matrix,
            member_matrix,
            metric="euclidean",
        )
        np.fill_diagonal(pairwise, np.inf)
        member_nn_distances = np.min(pairwise, axis=1)
        donor_reference_q90 = q90(member_nn_distances)
        donor_ratio = safe_ratio(
            donor_distance,
            donor_reference_q90,
        )
    else:
        donor_reference_q90 = 0.0
        donor_ratio = 0.0 if donor_distance <= EPSILON else float("inf")

    confidence_score = max(centroid_ratio, donor_ratio)

    return {
        "selection_status": "selected",
        "selection_reason": "",
        "target_cluster": target_cluster,
        "cluster_size": int(len(member_indices)),
        "donor_function": donor_name,
        "donor_distance": donor_distance,
        "confidence_score": confidence_score,
        "centroid_distance": target_centroid_distance,
        "centroid_reference_q90": centroid_reference_q90,
        "centroid_ratio": centroid_ratio,
        "donor_reference_q90": donor_reference_q90,
        "donor_ratio": donor_ratio,
        "eps": "",
        "nearest_core_distance": "",
        "nearest_core_ratio": "",
        "core_support_count": "",
        "core_point_count": "",
    }


def dbscan_training_eps(
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


def dbscan_confident_select(
    names: list[str],
    x: np.ndarray,
    train_indices: np.ndarray,
    target_index: int,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> dict[str, Any]:
    train_names = np.asarray(names, dtype=object)[train_indices]
    x_train_raw = x[train_indices]
    x_target_raw = x[target_index : target_index + 1]

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_target = scaler.transform(x_target_raw)

    eps = dbscan_training_eps(
        x_train=x_train,
        min_samples=min_samples,
        eps_quantile=eps_quantile,
        metric=metric,
    )

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )
    labels = model.fit_predict(x_train)
    core_indices = np.asarray(
        model.core_sample_indices_,
        dtype=int,
    )

    if len(core_indices) == 0 or eps <= EPSILON:
        return {
            "selection_status": "no-transfer",
            "selection_reason": "no_usable_core_region",
            "target_cluster": -1,
            "cluster_size": 0,
            "donor_function": "",
            "donor_distance": "",
            "confidence_score": float("inf"),
            "centroid_distance": "",
            "centroid_reference_q90": "",
            "centroid_ratio": "",
            "donor_reference_q90": "",
            "donor_ratio": "",
            "eps": eps,
            "nearest_core_distance": "",
            "nearest_core_ratio": "",
            "core_support_count": 0,
            "core_point_count": len(core_indices),
        }

    core_matrix = x_train[core_indices]
    core_distances = pairwise_distances(
        x_target,
        core_matrix,
        metric=metric,
    )[0]

    within = np.flatnonzero(
        core_distances <= eps + EPSILON
    )
    if len(within) == 0:
        return {
            "selection_status": "no-transfer",
            "selection_reason": "no_core_within_eps",
            "target_cluster": -1,
            "cluster_size": 0,
            "donor_function": "",
            "donor_distance": "",
            "confidence_score": float("inf"),
            "centroid_distance": "",
            "centroid_reference_q90": "",
            "centroid_ratio": "",
            "donor_reference_q90": "",
            "donor_ratio": "",
            "eps": eps,
            "nearest_core_distance": float(np.min(core_distances)),
            "nearest_core_ratio": float(
                np.min(core_distances) / eps
            ),
            "core_support_count": 0,
            "core_point_count": len(core_indices),
        }

    candidate_core_global = core_indices[within]
    candidate_core_distances = core_distances[within]
    candidate_core_names = train_names[
        candidate_core_global
    ]

    core_order = np.lexsort(
        (
            candidate_core_names,
            candidate_core_distances,
        )
    )
    nearest_core_global = int(
        candidate_core_global[int(core_order[0])]
    )
    nearest_core_distance = float(
        candidate_core_distances[int(core_order[0])]
    )
    target_cluster = int(labels[nearest_core_global])

    cluster_member_indices = np.flatnonzero(
        labels == target_cluster
    )
    cluster_member_matrix = x_train[
        cluster_member_indices
    ]
    cluster_member_names = train_names[
        cluster_member_indices
    ]

    donor_distances = pairwise_distances(
        x_target,
        cluster_member_matrix,
        metric=metric,
    )[0]
    donor_order = np.lexsort(
        (
            cluster_member_names,
            donor_distances,
        )
    )
    donor_pos = int(donor_order[0])
    donor_global = int(
        cluster_member_indices[donor_pos]
    )

    cluster_core_mask = (
        labels[candidate_core_global] == target_cluster
    )
    support_count = int(
        np.sum(cluster_core_mask)
    )

    nearest_core_ratio = float(
        nearest_core_distance / eps
    )

    return {
        "selection_status": "selected",
        "selection_reason": "",
        "target_cluster": target_cluster,
        "cluster_size": int(
            len(cluster_member_indices)
        ),
        "donor_function": str(
            train_names[donor_global]
        ),
        "donor_distance": float(
            donor_distances[donor_pos]
        ),
        "confidence_score": nearest_core_ratio,
        "centroid_distance": "",
        "centroid_reference_q90": "",
        "centroid_ratio": "",
        "donor_reference_q90": "",
        "donor_ratio": "",
        "eps": eps,
        "nearest_core_distance": nearest_core_distance,
        "nearest_core_ratio": nearest_core_ratio,
        "core_support_count": support_count,
        "core_point_count": len(core_indices),
    }


def select_for_algorithm(
    algorithm: str,
    names: list[str],
    x: np.ndarray,
    train_indices: np.ndarray,
    target_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if algorithm == "kmeans":
        return kmeans_confident_select(
            names=names,
            x=x,
            train_indices=train_indices,
            target_index=target_index,
            k=args.k,
            n_init=args.n_init,
            random_state=args.random_state,
        )

    if algorithm == "dbscan":
        return dbscan_confident_select(
            names=names,
            x=x,
            train_indices=train_indices,
            target_index=target_index,
            min_samples=args.dbscan_min_samples,
            eps_quantile=args.dbscan_eps_quantile,
            metric=args.dbscan_metric,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def donor_best_arm_match(
    target_name: str,
    donor_name: str,
    performance: dict[str, dict[str, Any]],
) -> bool:
    return (
        performance[target_name]["best_reward_arm"]
        == performance[donor_name]["best_reward_arm"]
    )


def inner_records_for_outer_fold(
    algorithm: str,
    outer_index: int,
    names: list[str],
    x: np.ndarray,
    performance: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    all_indices = np.arange(len(names), dtype=int)
    known_indices = all_indices[
        all_indices != outer_index
    ]

    records: list[dict[str, Any]] = []

    for inner_target_index in known_indices:
        inner_train_indices = known_indices[
            known_indices != inner_target_index
        ]

        selection = select_for_algorithm(
            algorithm=algorithm,
            names=names,
            x=x,
            train_indices=inner_train_indices,
            target_index=int(inner_target_index),
            args=args,
        )

        inner_target_name = names[
            int(inner_target_index)
        ]
        donor_name = selection["donor_function"]

        selected = (
            selection["selection_status"] == "selected"
            and bool(donor_name)
            and math.isfinite(
                float(selection["confidence_score"])
            )
        )

        if selected:
            correct = donor_best_arm_match(
                target_name=inner_target_name,
                donor_name=donor_name,
                performance=performance,
            )
        else:
            correct = None

        records.append(
            {
                "inner_target_function": inner_target_name,
                "selection_status": selection[
                    "selection_status"
                ],
                "donor_function": donor_name,
                "confidence_score": float(
                    selection["confidence_score"]
                ),
                "best_arm_correct": correct,
            }
        )

    return records


def quantile_label(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def calibrate_gate(
    records: list[dict[str, Any]],
    quantiles: list[float] = GATE_QUANTILES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Select a confidence threshold from inner historical functions only.

    Candidate utility:
        (correct - incorrect) / total_inner_targets

    A no-transfer candidate with utility 0 is always available.
    """
    total = len(records)
    selected = [
        row
        for row in records
        if row["selection_status"] == "selected"
        and math.isfinite(
            float(row["confidence_score"])
        )
        and row["best_arm_correct"] is not None
    ]

    candidate_rows: list[dict[str, Any]] = [
        {
            "gate_label": "abstain-all",
            "gate_quantile": "",
            "threshold": float("-inf"),
            "retained_count": 0,
            "coverage": 0.0,
            "best_arm_accuracy": "",
            "utility": 0.0,
            "correct_count": 0,
            "incorrect_count": 0,
        }
    ]

    if selected:
        scores = np.asarray(
            [
                float(row["confidence_score"])
                for row in selected
            ],
            dtype=float,
        )

        for q in quantiles:
            threshold = float(
                np.quantile(scores, q)
            )
            retained = [
                row
                for row in selected
                if float(row["confidence_score"])
                <= threshold + EPSILON
            ]
            correct = sum(
                1
                for row in retained
                if row["best_arm_correct"] is True
            )
            incorrect = sum(
                1
                for row in retained
                if row["best_arm_correct"] is False
            )
            retained_count = len(retained)
            coverage = (
                retained_count / total
                if total
                else float("nan")
            )
            accuracy = (
                correct / retained_count
                if retained_count
                else float("nan")
            )
            utility = (
                (correct - incorrect) / total
                if total
                else float("nan")
            )

            candidate_rows.append(
                {
                    "gate_label": quantile_label(q),
                    "gate_quantile": q,
                    "threshold": threshold,
                    "retained_count": retained_count,
                    "coverage": coverage,
                    "best_arm_accuracy": accuracy,
                    "utility": utility,
                    "correct_count": correct,
                    "incorrect_count": incorrect,
                }
            )

    best = candidate_rows[0]
    for candidate in candidate_rows[1:]:
        utility = float(candidate["utility"])
        best_utility = float(best["utility"])

        if utility > best_utility + EPSILON:
            best = candidate
            continue

        if (
            abs(utility - best_utility) <= EPSILON
            and utility > 0
        ):
            candidate_accuracy = float(
                candidate["best_arm_accuracy"]
            )
            best_accuracy = float(
                best["best_arm_accuracy"]
            )
            if (
                candidate_accuracy
                > best_accuracy + EPSILON
            ):
                best = candidate
            elif (
                abs(
                    candidate_accuracy
                    - best_accuracy
                )
                <= EPSILON
                and float(candidate["coverage"])
                > float(best["coverage"])
                + EPSILON
            ):
                best = candidate

    for row in candidate_rows:
        row["selected_gate"] = (
            row["gate_label"] == best["gate_label"]
        )

    return dict(best), candidate_rows


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
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        means[i] = np.mean(sample)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def make_sequences(
    target_name: str,
    target_stats: dict[str, Any],
    max_horizon: int,
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
                size=max_horizon,
                replace=True,
            )
            for arm in ARMS
        }
        for _ in range(replicates)
    ]


def simulate_prior(
    target_stats: dict[str, Any],
    donor_stats: dict[str, Any] | None,
    sequences: dict[str, np.ndarray],
    max_horizon: int,
    c: float,
    weight: float,
) -> dict[str, np.ndarray]:
    return simulate_policy(
        target_stats=target_stats,
        arm_sequences=sequences,
        horizon=max_horizon,
        c=c,
        prior_weight=(
            weight if donor_stats is not None else 0.0
        ),
        donor_stats=donor_stats,
    )


def evaluate_outer_fold(
    outer_index: int,
    algorithm: str,
    selection: dict[str, Any],
    gate: dict[str, Any],
    names: list[str],
    performance: dict[str, dict[str, Any]],
    weights: list[float],
    horizons: list[int],
    max_horizon: int,
    replicates: int,
    c: float,
    seed: int,
) -> list[dict[str, Any]]:
    target_name = names[outer_index]
    target_stats = performance[target_name]
    donor_name = selection["donor_function"]

    native_selected = (
        selection["selection_status"] == "selected"
        and bool(donor_name)
        and math.isfinite(
            float(selection["confidence_score"])
        )
    )

    gate_threshold = float(gate["threshold"])
    gate_accept = (
        native_selected
        and float(selection["confidence_score"])
        <= gate_threshold + EPSILON
    )

    donor_stats = (
        performance[donor_name]
        if native_selected
        else None
    )

    sequences = make_sequences(
        target_name=target_name,
        target_stats=target_stats,
        max_horizon=max_horizon,
        replicates=replicates,
        seed=seed,
    )

    baseline = [
        simulate_prior(
            target_stats=target_stats,
            donor_stats=None,
            sequences=seq,
            max_horizon=max_horizon,
            c=c,
            weight=0.0,
        )
        for seq in sequences
    ]

    output: list[dict[str, Any]] = []

    for mode in MODES:
        transfer_applied = (
            native_selected
            if mode == "ungated"
            else gate_accept
        )

        active_donor_stats = (
            donor_stats if transfer_applied else None
        )

        for weight in weights:
            if transfer_applied:
                results = [
                    simulate_prior(
                        target_stats=target_stats,
                        donor_stats=active_donor_stats,
                        sequences=sequences[i],
                        max_horizon=max_horizon,
                        c=c,
                        weight=weight,
                    )
                    for i in range(replicates)
                ]
            else:
                # Exact no-transfer behavior.
                results = baseline

            for horizon in horizons:
                idx = horizon - 1

                baseline_latency = np.asarray(
                    [
                        row["cumulative_latency"][idx]
                        for row in baseline
                    ],
                    dtype=float,
                )
                strategy_latency = np.asarray(
                    [
                        row["cumulative_latency"][idx]
                        for row in results
                    ],
                    dtype=float,
                )
                latency_gain = 100.0 * (
                    baseline_latency
                    - strategy_latency
                ) / baseline_latency

                optimal_rate = np.asarray(
                    [
                        row["optimal_reward_pulls"][idx]
                        / horizon
                        for row in results
                    ],
                    dtype=float,
                )
                first_optimal = np.asarray(
                    [
                        1.0
                        if row["chosen_arms"][0]
                        == target_stats["best_reward_arm"]
                        else 0.0
                        for row in results
                    ],
                    dtype=float,
                )
                reward_regret = np.asarray(
                    [
                        row["cumulative_reward_regret"][idx]
                        for row in results
                    ],
                    dtype=float,
                )

                output.append(
                    {
                        "algorithm": algorithm,
                        "mode": mode,
                        "target_function": target_name,
                        "donor_function": donor_name,
                        "native_selection_status": selection[
                            "selection_status"
                        ],
                        "confidence_score": selection[
                            "confidence_score"
                        ],
                        "centroid_ratio": selection[
                            "centroid_ratio"
                        ],
                        "donor_ratio": selection[
                            "donor_ratio"
                        ],
                        "nearest_core_ratio": selection[
                            "nearest_core_ratio"
                        ],
                        "core_support_count": selection[
                            "core_support_count"
                        ],
                        "gate_label": gate[
                            "gate_label"
                        ],
                        "gate_threshold": gate_threshold,
                        "inner_gate_utility": gate[
                            "utility"
                        ],
                        "inner_gate_coverage": gate[
                            "coverage"
                        ],
                        "inner_gate_best_arm_accuracy": gate[
                            "best_arm_accuracy"
                        ],
                        "transfer_applied": transfer_applied,
                        "equivalent_observation_weight": weight,
                        "horizon": horizon,
                        "replicates": replicates,
                        "target_best_reward_arm": target_stats[
                            "best_reward_arm"
                        ],
                        "donor_best_reward_arm": (
                            donor_stats["best_reward_arm"]
                            if donor_stats is not None
                            else ""
                        ),
                        "donor_best_arm_match": (
                            donor_stats["best_reward_arm"]
                            == target_stats["best_reward_arm"]
                            if donor_stats is not None
                            else ""
                        ),
                        "mean_latency_gain_pct_vs_no_transfer": float(
                            np.mean(latency_gain)
                        ),
                        "median_latency_gain_pct_vs_no_transfer": float(
                            np.median(latency_gain)
                        ),
                        "positive_latency_gain_probability": float(
                            np.mean(latency_gain > 0)
                        ),
                        "mean_optimal_reward_arm_selection_rate": float(
                            np.mean(optimal_rate)
                        ),
                        "first_arm_optimal_probability": float(
                            np.mean(first_optimal)
                        ),
                        "mean_reward_pseudo_regret": float(
                            np.mean(reward_regret)
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
                row["algorithm"],
                row["mode"],
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

    for algorithm, mode, weight, horizon in keys:
        group = [
            row
            for row in rows
            if row["algorithm"] == algorithm
            and row["mode"] == mode
            and float(
                row["equivalent_observation_weight"]
            )
            == weight
            and int(row["horizon"]) == horizon
        ]

        applied = [
            row
            for row in group
            if row["transfer_applied"]
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
        gains_applied = np.asarray(
            [
                float(
                    row[
                        "mean_latency_gain_pct_vs_no_transfer"
                    ]
                )
                for row in applied
            ],
            dtype=float,
        )

        ci_low, ci_high = bootstrap_mean_ci(
            gains,
            stable_seed(
                seed,
                f"{algorithm}-{mode}-{weight}-{horizon}",
            ),
        )

        applied_ci_low, applied_ci_high = (
            bootstrap_mean_ci(
                gains_applied,
                stable_seed(
                    seed,
                    f"{algorithm}-{mode}-{weight}-{horizon}-applied",
                ),
            )
        )

        donor_matches = [
            1.0
            if row["donor_best_arm_match"] is True
            else 0.0
            for row in applied
        ]

        output.append(
            {
                "algorithm": algorithm,
                "mode": mode,
                "equivalent_observation_weight": weight,
                "horizon": horizon,
                "target_count": len(group),
                "transfer_applied_count": len(
                    applied
                ),
                "transfer_coverage": (
                    len(applied) / len(group)
                    if group
                    else float("nan")
                ),
                "accepted_donor_best_arm_agreement_rate": (
                    float(np.mean(donor_matches))
                    if donor_matches
                    else float("nan")
                ),
                "macro_mean_latency_gain_pct_all_targets": float(
                    np.mean(gains)
                ),
                "macro_median_latency_gain_pct_all_targets": float(
                    np.median(gains)
                ),
                "macro_latency_gain_ci95_low_all_targets": ci_low,
                "macro_latency_gain_ci95_high_all_targets": ci_high,
                "target_positive_gain_rate_all_targets": float(
                    np.mean(gains > 0)
                ),
                "macro_mean_latency_gain_pct_transfer_applied": (
                    float(np.mean(gains_applied))
                    if len(gains_applied)
                    else float("nan")
                ),
                "macro_latency_gain_ci95_low_transfer_applied": (
                    applied_ci_low
                ),
                "macro_latency_gain_ci95_high_transfer_applied": (
                    applied_ci_high
                ),
                "target_positive_gain_rate_transfer_applied": (
                    float(np.mean(gains_applied > 0))
                    if len(gains_applied)
                    else float("nan")
                ),
                "macro_mean_optimal_reward_arm_selection_rate": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_optimal_reward_arm_selection_rate"
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


def plot_summary(
    summary: list[dict[str, Any]],
    figures: Path,
    primary_horizon: int,
) -> None:
    for algorithm in sorted(
        {row["algorithm"] for row in summary}
    ):
        fig, ax = plt.subplots(figsize=(9.5, 6))

        for mode in MODES:
            for weight in sorted(
                {
                    float(
                        row[
                            "equivalent_observation_weight"
                        ]
                    )
                    for row in summary
                    if row["algorithm"] == algorithm
                }
            ):
                subset = [
                    row
                    for row in summary
                    if row["algorithm"] == algorithm
                    and row["mode"] == mode
                    and float(
                        row[
                            "equivalent_observation_weight"
                        ]
                    )
                    == weight
                ]
                subset.sort(
                    key=lambda row: int(
                        row["horizon"]
                    )
                )

                ax.plot(
                    [
                        int(row["horizon"])
                        for row in subset
                    ],
                    [
                        float(
                            row[
                                "macro_mean_latency_gain_pct_all_targets"
                            ]
                        )
                        for row in subset
                    ],
                    marker="o",
                    label=f"{mode} / w={weight:g}",
                )

        ax.axhline(
            0.0,
            linewidth=1,
            linestyle="--",
        )
        ax.set_title(
            f"{algorithm.upper()} nested confidence gate"
        )
        ax.set_xlabel("Requests")
        ax.set_ylabel(
            "Mean latency gain vs no-transfer (%)"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(
            figures
            / f"{algorithm}-nested-confidence-latency-gain.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures
            / f"{algorithm}-nested-confidence-latency-gain.svg",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Coverage vs gain at primary horizon.
        fig, ax = plt.subplots(figsize=(8.5, 6))
        subset = [
            row
            for row in summary
            if row["algorithm"] == algorithm
            and row["mode"] == "nested-gated"
            and int(row["horizon"])
            == primary_horizon
        ]
        subset.sort(
            key=lambda row: float(
                row[
                    "equivalent_observation_weight"
                ]
            )
        )

        ax.scatter(
            [
                float(row["transfer_coverage"])
                for row in subset
            ],
            [
                float(
                    row[
                        "macro_mean_latency_gain_pct_all_targets"
                    ]
                )
                for row in subset
            ],
        )
        for row in subset:
            ax.annotate(
                f'w={float(row["equivalent_observation_weight"]):g}',
                (
                    float(row["transfer_coverage"]),
                    float(
                        row[
                            "macro_mean_latency_gain_pct_all_targets"
                        ]
                    ),
                ),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.axhline(
            0.0,
            linewidth=1,
            linestyle="--",
        )
        ax.set_title(
            f"{algorithm.upper()} nested gate @ H={primary_horizon}"
        )
        ax.set_xlabel("Held-out transfer coverage")
        ax.set_ylabel(
            "Mean latency gain vs no-transfer (%)"
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figures
            / f"{algorithm}-nested-confidence-coverage-gain-h{primary_horizon}.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures
            / f"{algorithm}-nested-confidence-coverage-gain-h{primary_horizon}.svg",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--profiles",
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
        "--primary-horizon",
        type=int,
        default=10,
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
    )
    p.add_argument(
        "--n-init",
        type=int,
        default=50,
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    p.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=4,
    )
    p.add_argument(
        "--dbscan-eps-quantile",
        type=float,
        default=0.80,
    )
    p.add_argument(
        "--dbscan-metric",
        default="cosine",
    )
    args = p.parse_args()

    weights = args.weight or DEFAULT_WEIGHTS
    horizons = [
        h
        for h in DEFAULT_HORIZONS
        if h <= args.horizon
    ]
    if args.horizon not in horizons:
        horizons.append(args.horizon)
        horizons.sort()

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    names, x = load_profiles(
        args.profiles.resolve()
    )
    raw_x86 = load_raw_durations(
        args.raw_x86.resolve()
    )
    raw_arm = load_raw_durations(
        args.raw_arm64.resolve()
    )
    performance = empirical_stats(
        raw_x86,
        raw_arm,
    )

    if set(names) != set(performance):
        raise RuntimeError(
            "Profile/performance function mismatch: "
            f"missing={sorted(set(names) - set(performance))}, "
            f"extra={sorted(set(performance) - set(names))}"
        )

    all_indices = np.arange(
        len(names),
        dtype=int,
    )

    fold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    print(
        f"Starting nested LOFO: "
        f"functions={len(names)}"
    )

    for outer_index, target_name in enumerate(
        names
    ):
        outer_train_indices = all_indices[
            all_indices != outer_index
        ]

        print(
            f"[{outer_index + 1:02d}/{len(names)}] "
            f"{target_name}"
        )

        for algorithm in (
            "kmeans",
            "dbscan",
        ):
            inner_records = (
                inner_records_for_outer_fold(
                    algorithm=algorithm,
                    outer_index=outer_index,
                    names=names,
                    x=x,
                    performance=performance,
                    args=args,
                )
            )

            gate, candidates = calibrate_gate(
                inner_records
            )

            for candidate in candidates:
                calibration_rows.append(
                    {
                        "outer_target_function": target_name,
                        "algorithm": algorithm,
                        **candidate,
                    }
                )

            outer_selection = select_for_algorithm(
                algorithm=algorithm,
                names=names,
                x=x,
                train_indices=outer_train_indices,
                target_index=outer_index,
                args=args,
            )

            fold_rows.extend(
                evaluate_outer_fold(
                    outer_index=outer_index,
                    algorithm=algorithm,
                    selection=outer_selection,
                    gate=gate,
                    names=names,
                    performance=performance,
                    weights=weights,
                    horizons=horizons,
                    max_horizon=args.horizon,
                    replicates=args.replicates,
                    c=args.c,
                    seed=args.seed,
                )
            )

    fold_rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["mode"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
            int(row["horizon"]),
            row["target_function"],
        )
    )
    calibration_rows.sort(
        key=lambda row: (
            row["algorithm"],
            row["outer_target_function"],
            str(row["gate_label"]),
        )
    )

    summary = summarize(
        fold_rows,
        seed=args.seed,
    )

    write_csv(
        out
        / "nested-confidence-per-target.csv",
        fold_rows,
    )
    write_csv(
        out
        / "nested-confidence-calibration.csv",
        calibration_rows,
    )
    write_csv(
        out
        / "nested-confidence-summary.csv",
        summary,
    )

    plot_summary(
        summary,
        figures,
        primary_horizon=args.primary_horizon,
    )

    manifest = {
        "schema_version": 1,
        "experiment": (
            "nested leave-one-function-out "
            "cluster-relative confidence gate"
        ),
        "paper5_features": PAPER5_FEATURES,
        "outer_protocol": (
            "outer target excluded from scaler, clustering, "
            "donor selection, and confidence-gate calibration"
        ),
        "inner_protocol": (
            "within the remaining known functions, each inner "
            "target is held out once; historical best-reward-arm "
            "agreement calibrates the gate"
        ),
        "gate_quantiles": GATE_QUANTILES,
        "gate_utility": (
            "(correct_transfers - incorrect_transfers) / "
            "inner_target_count; abstain-all utility=0"
        ),
        "kmeans_confidence": {
            "scaler": "MinMaxScaler",
            "k": args.k,
            "metric": "euclidean",
            "centroid_ratio": (
                "target-centroid distance divided by q90 "
                "training member-centroid distance"
            ),
            "donor_ratio": (
                "target-donor distance divided by q90 "
                "within-cluster member 1-NN distance"
            ),
            "score": "max(centroid_ratio, donor_ratio)",
        },
        "dbscan_confidence": {
            "scaler": "RobustScaler",
            "metric": args.dbscan_metric,
            "min_samples": args.dbscan_min_samples,
            "eps_quantile": args.dbscan_eps_quantile,
            "score": "nearest_core_distance / eps",
            "secondary_diagnostic": (
                "core_support_count within eps in assigned cluster"
            ),
        },
        "ucb1": {
            "formula": "unchanged current Serverledge transfer UCB1",
            "c": args.c,
            "weights": weights,
            "dynamic_c": False,
            "prior_reset": False,
        },
        "evaluation": {
            "horizons": horizons,
            "replicates": args.replicates,
            "seed": args.seed,
            "reward": "-ln(duration_ms)",
            "modes": list(MODES),
            "paired_empirical_bootstrap": True,
        },
        "inputs": {
            "profiles": str(
                args.profiles.resolve()
            ),
            "raw_x86": str(
                args.raw_x86.resolve()
            ),
            "raw_arm64": str(
                args.raw_arm64.resolve()
            ),
        },
        "outputs": [
            "nested-confidence-per-target.csv",
            "nested-confidence-calibration.csv",
            "nested-confidence-summary.csv",
            "figures/",
        ],
    }

    (
        out
        / "nested-confidence-manifest.json"
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
    print(
        f"\nSummary @ H={args.primary_horizon}:"
    )

    primary = [
        row
        for row in summary
        if int(row["horizon"])
        == args.primary_horizon
    ]
    primary.sort(
        key=lambda row: (
            row["algorithm"],
            row["mode"],
            float(
                row[
                    "equivalent_observation_weight"
                ]
            ),
        )
    )

    for row in primary:
        print(
            f"{row['algorithm']:<7} "
            f"{row['mode']:<12} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"coverage="
            f"{float(row['transfer_coverage']):.4f} "
            f"donor-match="
            f"{float(row['accepted_donor_best_arm_agreement_rate']):.4f} "
            f"gain="
            f"{float(row['macro_mean_latency_gain_pct_all_targets']):+.4f}% "
            f"CI95=["
            f"{float(row['macro_latency_gain_ci95_low_all_targets']):+.4f},"
            f"{float(row['macro_latency_gain_ci95_high_all_targets']):+.4f}"
            f"] "
            f"optimal="
            f"{float(row['macro_mean_optimal_reward_arm_selection_rate']):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
