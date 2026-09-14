#!/usr/bin/env python3
"""
DBSCAN feature-set robustness study for the Serverledge thesis.

This script complements clustering_study.py without modifying the production
profiling/transfer feature contract.  It evaluates the feature configurations
that emerged from the PAPER-6 K-Means analysis using DBSCAN on the x86
reference architecture only.

Primary feature configurations:
- paper6: the six paper-inspired profiling features;
- paper6_no_framework_runtime_ms: PAPER-5 candidate for Serverledge;
- paper4_no_framework_runtime_ms_no_page_faults_delta: ablation control;
- paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms: ablation control.

Methodological rules:
- DBSCAN is fitted only on aggregated x86 FunctionProfile vectors;
- architecture-preference labels are joined only after clustering;
- eps candidates are derived from quantiles of the k-distance curve, separately
  for every feature-set / metric / min_samples combination, so the sweep adapts
  to the scale and geometry instead of using an arbitrary common eps grid;
- noise is NOT treated as an ordinary cluster for external metrics;
- therefore every external result reports coverage together with purity,
  homogeneity, ARI and NMI computed on clustered points only;
- no configuration is automatically selected using ground truth.

Outputs are CSV/JSON plus diagnostic PNG/SVG figures.  Markdown reporting is
intentionally left to the later documentation step, after K-Means and DBSCAN
results have been interpreted together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics import pairwise_distances

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import clustering_study, preference, preprocess


FEATURE_SETS = {
    "paper6": list(clustering_study.PAPER6_FEATURES),
    "paper6_no_framework_runtime_ms": [
        feature
        for feature in clustering_study.PAPER6_FEATURES
        if feature != "framework_runtime_ms"
    ],
    "paper4_no_framework_runtime_ms_no_page_faults_delta": [
        feature
        for feature in clustering_study.PAPER6_FEATURES
        if feature not in {"framework_runtime_ms", "page_faults_delta"}
    ],
    "paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms": [
        feature
        for feature in clustering_study.PAPER6_FEATURES
        if feature not in {"framework_runtime_ms", "cpu_kernel_delta_ms"}
    ],
}

DBSCAN_METRICS = ("euclidean", "manhattan", "cosine")
DEFAULT_MIN_SAMPLES = (3, 4, 5, 6)
DEFAULT_EPS_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

INTERNAL_HEADER = [
    "experiment_id",
    "configuration_id",
    "feature_set",
    "feature_count",
    "features",
    "scaler",
    "metric",
    "min_samples",
    "eps_quantile",
    "eps",
    "sample_count",
    "cluster_count",
    "noise_count",
    "coverage",
    "min_cluster_size",
    "max_cluster_size",
    "singleton_count",
    "cluster_size_distribution",
    "silhouette_clustered",
    "silhouette_defined",
    "run_dir",
]

EXTERNAL_HEADER = [
    *INTERNAL_HEADER,
    "preference_run_id",
    "threshold_percent",
    "clustered_count",
    "clustered_x86_preferred_count",
    "clustered_arm_preferred_count",
    "clustered_architecture_independent_count",
    "noise_x86_preferred_count",
    "noise_arm_preferred_count",
    "noise_architecture_independent_count",
    "clustered_majority_ground_truth_class",
    "clustered_majority_ground_truth_share",
    "overall_purity_clustered",
    "purity_gain_over_clustered_majority_baseline",
    "homogeneity_clustered",
    "completeness_clustered",
    "v_measure_clustered",
    "adjusted_rand_index_clustered",
    "normalized_mutual_information_clustered",
    "external_metrics_defined",
]

COMPOSITION_HEADER = [
    "experiment_id",
    "configuration_id",
    "feature_set",
    "scaler",
    "metric",
    "min_samples",
    "eps_quantile",
    "eps",
    "threshold_percent",
    "cluster_label",
    "cluster_size",
    "x86_preferred_count",
    "arm_preferred_count",
    "architecture_independent_count",
    "majority_preference",
    "cluster_purity",
]

NOISE_HEADER = [
    "experiment_id",
    "configuration_id",
    "feature_set",
    "scaler",
    "metric",
    "min_samples",
    "eps_quantile",
    "eps",
    "function_name",
]

EPS_GRID_HEADER = [
    "feature_set",
    "scaler",
    "metric",
    "min_samples",
    "eps_quantile",
    "eps",
    "k_distance_min",
    "k_distance_median",
    "k_distance_max",
]


class DBSCANStudyError(ValueError):
    pass


def _finite_positive_float(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise DBSCANStudyError(f"{label} must be finite and positive")
    return value


def _unique_ints(values: list[int], minimum: int, label: str) -> list[int]:
    result = []
    seen = set()
    for value in values:
        value = int(value)
        if value < minimum:
            raise DBSCANStudyError(f"{label} must be >= {minimum}")
        if value not in seen:
            seen.add(value)
            result.append(value)
    if not result:
        raise DBSCANStudyError(f"at least one {label} value is required")
    return result


def _unique_quantiles(values: list[float]) -> list[float]:
    result = []
    seen = set()
    for value in values:
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise DBSCANStudyError("eps quantiles must be strictly between 0 and 1")
        if value not in seen:
            seen.add(value)
            result.append(value)
    if not result:
        raise DBSCANStudyError("at least one eps quantile is required")
    return result


def _selected_feature_sets(requested: list[str]) -> list[str]:
    if not requested:
        return list(FEATURE_SETS)
    result = []
    seen = set()
    for name in requested:
        if name not in FEATURE_SETS:
            raise DBSCANStudyError(
                f"unknown feature set {name!r}; expected one of {sorted(FEATURE_SETS)}"
            )
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _selected_metrics(requested: list[str]) -> list[str]:
    if not requested:
        return list(DBSCAN_METRICS)
    result = []
    seen = set()
    for metric in requested:
        if metric not in DBSCAN_METRICS:
            raise DBSCANStudyError(
                f"unsupported metric {metric!r}; expected one of {DBSCAN_METRICS}"
            )
        if metric not in seen:
            seen.add(metric)
            result.append(metric)
    return result


def configuration_id(
    feature_set: str,
    scaler: str,
    metric: str,
    min_samples: int,
    quantile: float,
    eps: float,
) -> str:
    q = clustering_study.threshold_slug(100.0 * quantile)
    e = clustering_study.threshold_slug(eps)
    return (
        f"{feature_set}__{scaler}__dbscan-{metric}"
        f"__ms{min_samples}__q{q}__eps{e}"
    )


def k_distance_curve(
    matrix: np.ndarray,
    metric: str,
    min_samples: int,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise DBSCANStudyError("matrix must contain at least two samples")
    if min_samples > matrix.shape[0]:
        raise DBSCANStudyError(
            f"min_samples={min_samples} exceeds sample count {matrix.shape[0]}"
        )

    distances = pairwise_distances(matrix, metric=metric)
    # DBSCAN min_samples includes the sample itself.  Therefore the element at
    # index min_samples-1 in the sorted row is the radius required to contain
    # min_samples points including self (distance 0).
    sorted_distances = np.sort(distances, axis=1)
    return np.sort(sorted_distances[:, min_samples - 1].astype(float))


def derive_eps_candidates(
    curve: np.ndarray,
    quantiles: list[float],
) -> list[tuple[float, float]]:
    if curve.ndim != 1 or curve.size == 0:
        raise DBSCANStudyError("k-distance curve cannot be empty")

    result = []
    seen_eps = set()
    for quantile in quantiles:
        eps = float(np.quantile(curve, quantile))
        if not math.isfinite(eps) or eps <= 0:
            continue
        # Floating-point distance values are preserved with enough precision
        # to reproduce the configuration while avoiding duplicate candidates.
        key = round(eps, 12)
        if key in seen_eps:
            continue
        seen_eps.add(key)
        result.append((quantile, eps))

    if not result:
        raise DBSCANStudyError("no positive eps candidate derived from k-distance curve")
    return result


def dbscan_internal_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    noise_mask = labels == -1
    clustered_mask = ~noise_mask
    clustered_labels = labels[clustered_mask]
    cluster_ids = sorted(set(int(v) for v in clustered_labels))
    sizes = [int(np.sum(clustered_labels == label)) for label in cluster_ids]

    silhouette_defined = False
    silhouette = ""
    if len(cluster_ids) >= 2 and int(np.sum(clustered_mask)) > len(cluster_ids):
        silhouette = float(
            silhouette_score(
                matrix[clustered_mask],
                clustered_labels,
                metric=metric,
            )
        )
        silhouette_defined = True

    return {
        "sample_count": int(matrix.shape[0]),
        "cluster_count": len(cluster_ids),
        "noise_count": int(np.sum(noise_mask)),
        "coverage": float(np.mean(clustered_mask)),
        "min_cluster_size": min(sizes) if sizes else 0,
        "max_cluster_size": max(sizes) if sizes else 0,
        "singleton_count": sum(size == 1 for size in sizes),
        "cluster_size_distribution": "|".join(
            str(size) for size in sorted(sizes, reverse=True)
        ),
        "silhouette_clustered": silhouette,
        "silhouette_defined": silhouette_defined,
    }


def evaluate_external_clustered_only(
    labels: np.ndarray,
    functions: list[str],
    preferences: dict[str, dict],
) -> tuple[dict, list[dict]]:
    labels = np.asarray(labels, dtype=int)
    clustered_indexes = [index for index, label in enumerate(labels) if label != -1]
    noise_indexes = [index for index, label in enumerate(labels) if label == -1]

    noise_counts = Counter(
        preferences[functions[index]]["architecture_preference"]
        for index in noise_indexes
    )

    if not clustered_indexes:
        return (
            {
                "clustered_count": 0,
                "clustered_x86_preferred_count": 0,
                "clustered_arm_preferred_count": 0,
                "clustered_architecture_independent_count": 0,
                "noise_x86_preferred_count": noise_counts[preference.PREFERENCE_X86],
                "noise_arm_preferred_count": noise_counts[preference.PREFERENCE_ARM],
                "noise_architecture_independent_count": noise_counts[
                    preference.PREFERENCE_INDEPENDENT
                ],
                "clustered_majority_ground_truth_class": "",
                "clustered_majority_ground_truth_share": "",
                "overall_purity_clustered": "",
                "purity_gain_over_clustered_majority_baseline": "",
                "homogeneity_clustered": "",
                "completeness_clustered": "",
                "v_measure_clustered": "",
                "adjusted_rand_index_clustered": "",
                "normalized_mutual_information_clustered": "",
                "external_metrics_defined": False,
            },
            [],
        )

    true_labels = [
        preferences[functions[index]]["architecture_preference"]
        for index in clustered_indexes
    ]
    predicted = np.asarray([labels[index] for index in clustered_indexes], dtype=int)
    true_counts = Counter(true_labels)

    grouped: dict[int, list[str]] = defaultdict(list)
    for predicted_label, true_label in zip(predicted, true_labels):
        grouped[int(predicted_label)].append(true_label)

    majority_correct = 0
    composition = []
    for cluster_label in sorted(grouped):
        members = grouped[cluster_label]
        counts = Counter(members)
        maximum = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == maximum)
        majority = (
            winners[0]
            if len(winners) == 1
            else clustering_study.AMBIGUOUS_PREFERENCE
        )
        majority_correct += maximum
        composition.append(
            {
                "cluster_label": cluster_label,
                "cluster_size": len(members),
                "x86_preferred_count": counts[preference.PREFERENCE_X86],
                "arm_preferred_count": counts[preference.PREFERENCE_ARM],
                "architecture_independent_count": counts[
                    preference.PREFERENCE_INDEPENDENT
                ],
                "majority_preference": majority,
                "cluster_purity": maximum / len(members),
            }
        )

    majority_count = max(true_counts.values())
    majority_winners = sorted(
        label for label, count in true_counts.items() if count == majority_count
    )
    majority_class = (
        majority_winners[0]
        if len(majority_winners) == 1
        else clustering_study.AMBIGUOUS_PREFERENCE
    )
    majority_share = majority_count / len(clustered_indexes)
    purity_clustered = majority_correct / len(clustered_indexes)

    encoded_true = [clustering_study.PREFERENCE_LABELS.index(label) for label in true_labels]

    metrics_defined = len(set(predicted.tolist())) >= 1
    return (
        {
            "clustered_count": len(clustered_indexes),
            "clustered_x86_preferred_count": true_counts[preference.PREFERENCE_X86],
            "clustered_arm_preferred_count": true_counts[preference.PREFERENCE_ARM],
            "clustered_architecture_independent_count": true_counts[
                preference.PREFERENCE_INDEPENDENT
            ],
            "noise_x86_preferred_count": noise_counts[preference.PREFERENCE_X86],
            "noise_arm_preferred_count": noise_counts[preference.PREFERENCE_ARM],
            "noise_architecture_independent_count": noise_counts[
                preference.PREFERENCE_INDEPENDENT
            ],
            "clustered_majority_ground_truth_class": majority_class,
            "clustered_majority_ground_truth_share": majority_share,
            "overall_purity_clustered": purity_clustered,
            "purity_gain_over_clustered_majority_baseline": (
                purity_clustered - majority_share
            ),
            "homogeneity_clustered": float(homogeneity_score(encoded_true, predicted)),
            "completeness_clustered": float(completeness_score(encoded_true, predicted)),
            "v_measure_clustered": float(v_measure_score(encoded_true, predicted)),
            "adjusted_rand_index_clustered": float(
                adjusted_rand_score(encoded_true, predicted)
            ),
            "normalized_mutual_information_clustered": float(
                normalized_mutual_info_score(encoded_true, predicted)
            ),
            "external_metrics_defined": metrics_defined,
        },
        composition,
    )


def save_assignments(
    path: Path,
    profile_rows: list[dict[str, str]],
    transformed: np.ndarray,
    labels: np.ndarray,
    feature_set: str,
    features: list[str],
    scaler: str,
    metric: str,
    min_samples: int,
    eps_quantile: float,
    eps: float,
) -> None:
    header = [
        "feature_set",
        "scaler",
        "algorithm",
        "metric",
        "min_samples",
        "eps_quantile",
        "eps",
        "function_name",
        "machine_tag",
        "configured_cpus",
        "configured_memory_mb",
        "sample_count",
        "cluster_label",
        "is_noise",
        *features,
    ]
    rows = []
    for index, source in enumerate(profile_rows):
        row = {
            "feature_set": feature_set,
            "scaler": scaler,
            "algorithm": "dbscan",
            "metric": metric,
            "min_samples": min_samples,
            "eps_quantile": eps_quantile,
            "eps": eps,
            "function_name": source["function_name"],
            "machine_tag": source["machine_tag"],
            "configured_cpus": source["configured_cpus"],
            "configured_memory_mb": source["configured_memory_mb"],
            "sample_count": source["sample_count"],
            "cluster_label": int(labels[index]),
            "is_noise": bool(labels[index] == -1),
        }
        for column, feature in enumerate(features):
            row[feature] = format(float(transformed[index, column]), ".17g")
        rows.append(row)
    clustering_study.write_csv(path, header, rows)


def plot_silhouette_coverage(
    output_dir: Path,
    internal_rows: list[dict],
    metric: str,
) -> None:
    rows = [
        row
        for row in internal_rows
        if row["metric"] == metric and row["silhouette_defined"]
    ]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for feature_set in FEATURE_SETS:
        subset = [row for row in rows if row["feature_set"] == feature_set]
        if not subset:
            continue
        ax.scatter(
            [100.0 * float(row["coverage"]) for row in subset],
            [float(row["silhouette_clustered"]) for row in subset],
            label=feature_set,
            alpha=0.75,
        )
    ax.set_xlabel("Coverage of non-noise functions (%)")
    ax.set_ylabel(f"Silhouette on clustered points ({metric})")
    ax.set_title(f"DBSCAN internal trade-off — {metric}")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"dbscan-silhouette-vs-coverage-{metric}.png", dpi=220)
    fig.savefig(output_dir / f"dbscan-silhouette-vs-coverage-{metric}.svg")
    plt.close(fig)


def plot_cluster_count_noise(
    output_dir: Path,
    internal_rows: list[dict],
    metric: str,
) -> None:
    rows = [row for row in internal_rows if row["metric"] == metric]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for feature_set in FEATURE_SETS:
        subset = [row for row in rows if row["feature_set"] == feature_set]
        if not subset:
            continue
        ax.scatter(
            [int(row["noise_count"]) for row in subset],
            [int(row["cluster_count"]) for row in subset],
            label=feature_set,
            alpha=0.75,
        )
    ax.set_xlabel("Noise functions")
    ax.set_ylabel("DBSCAN clusters (excluding noise)")
    ax.set_title(f"DBSCAN cluster/noise trade-off — {metric}")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"dbscan-clusters-vs-noise-{metric}.png", dpi=220)
    fig.savefig(output_dir / f"dbscan-clusters-vs-noise-{metric}.svg")
    plt.close(fig)


def run_study(
    profiles_path: Path,
    preference_paths: list[Path],
    output_dir: Path,
    experiment_id: str,
    scaler_name: str,
    feature_sets: list[str],
    metrics: list[str],
    min_samples_values: list[int],
    eps_quantiles: list[float],
) -> dict:
    experiment_id = experiment_id.strip()
    if not experiment_id:
        raise DBSCANStudyError("experiment ID cannot be empty")
    if scaler_name not in preprocess.SCALERS:
        raise DBSCANStudyError(f"unsupported scaler {scaler_name!r}")

    feature_sets = _selected_feature_sets(feature_sets)
    metrics = _selected_metrics(metrics)
    min_samples_values = _unique_ints(min_samples_values, 2, "min_samples")
    eps_quantiles = _unique_quantiles(eps_quantiles)

    profile_rows, raw_matrix, profile_meta = clustering_study.load_reference_profiles(
        profiles_path
    )
    functions = [row["function_name"].strip() for row in profile_rows]
    preference_datasets = clustering_study.load_preference_sweep(
        preference_paths,
        functions,
        profile_meta["aggregation"],
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    column_index = {
        name: index for index, name in enumerate(clustering_study.PAPER6_FEATURES)
    }

    internal_rows: list[dict] = []
    external_rows: list[dict] = []
    composition_rows: list[dict] = []
    noise_rows: list[dict] = []
    eps_grid_rows: list[dict] = []

    for feature_set in feature_sets:
        features = FEATURE_SETS[feature_set]
        indexes = [column_index[feature] for feature in features]
        subset_raw = raw_matrix[:, indexes]
        transformed, scaler_state = clustering_study.fit_scaler(
            scaler_name, subset_raw
        )

        for metric in metrics:
            for min_samples in min_samples_values:
                curve = k_distance_curve(transformed, metric, min_samples)
                candidates = derive_eps_candidates(curve, eps_quantiles)

                for quantile, eps in candidates:
                    eps_grid_rows.append(
                        {
                            "feature_set": feature_set,
                            "scaler": scaler_name,
                            "metric": metric,
                            "min_samples": min_samples,
                            "eps_quantile": quantile,
                            "eps": eps,
                            "k_distance_min": float(np.min(curve)),
                            "k_distance_median": float(np.median(curve)),
                            "k_distance_max": float(np.max(curve)),
                        }
                    )

                    model = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric=metric,
                    )
                    labels = model.fit_predict(transformed).astype(np.int64)
                    metrics_internal = dbscan_internal_metrics(
                        transformed, labels, metric
                    )
                    config_id = configuration_id(
                        feature_set,
                        scaler_name,
                        metric,
                        min_samples,
                        quantile,
                        eps,
                    )
                    run_dir = runs_dir / config_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    save_assignments(
                        run_dir / "assignments.csv",
                        profile_rows,
                        transformed,
                        labels,
                        feature_set,
                        features,
                        scaler_name,
                        metric,
                        min_samples,
                        quantile,
                        eps,
                    )

                    base = {
                        "experiment_id": experiment_id,
                        "configuration_id": config_id,
                        "feature_set": feature_set,
                        "feature_count": len(features),
                        "features": "|".join(features),
                        "scaler": scaler_name,
                        "metric": metric,
                        "min_samples": min_samples,
                        "eps_quantile": quantile,
                        "eps": eps,
                        **metrics_internal,
                        "run_dir": str(run_dir),
                    }
                    internal_rows.append(base)

                    for index, label in enumerate(labels):
                        if label == -1:
                            noise_rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "configuration_id": config_id,
                                    "feature_set": feature_set,
                                    "scaler": scaler_name,
                                    "metric": metric,
                                    "min_samples": min_samples,
                                    "eps_quantile": quantile,
                                    "eps": eps,
                                    "function_name": functions[index],
                                }
                            )

                    threshold_documents = []
                    for preferences, pref_meta in preference_datasets:
                        external, composition = evaluate_external_clustered_only(
                            labels, functions, preferences
                        )
                        external_rows.append(
                            {
                                **base,
                                "preference_run_id": pref_meta["preference_run_id"],
                                "threshold_percent": pref_meta["threshold_percent"],
                                **external,
                            }
                        )

                        for row in composition:
                            composition_rows.append(
                                {
                                    "experiment_id": experiment_id,
                                    "configuration_id": config_id,
                                    "feature_set": feature_set,
                                    "scaler": scaler_name,
                                    "metric": metric,
                                    "min_samples": min_samples,
                                    "eps_quantile": quantile,
                                    "eps": eps,
                                    "threshold_percent": pref_meta[
                                        "threshold_percent"
                                    ],
                                    **row,
                                }
                            )

                        threshold_documents.append(
                            {
                                "threshold_percent": pref_meta[
                                    "threshold_percent"
                                ],
                                "preference_run_id": pref_meta[
                                    "preference_run_id"
                                ],
                                "preferences_sha256": pref_meta["sha256"],
                                "external_metrics_clustered_only": external,
                                "cluster_composition": composition,
                            }
                        )

                    clustering_study.write_json(
                        run_dir / "evaluation-all-thresholds.json",
                        {
                            "experiment_id": experiment_id,
                            "configuration_id": config_id,
                            "feature_set": feature_set,
                            "feature_names": features,
                            "scaler": scaler_name,
                            "scaler_state": scaler_state,
                            "algorithm": "dbscan",
                            "metric": metric,
                            "min_samples": min_samples,
                            "eps_quantile": quantile,
                            "eps": eps,
                            "eps_derivation": (
                                "quantile of the distance to the min_samples-th "
                                "point including self"
                            ),
                            "internal_metrics": metrics_internal,
                            "ground_truth_evaluations": threshold_documents,
                        },
                    )

    internal_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["metric"],
            row["min_samples"],
            row["eps_quantile"],
        )
    )
    external_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["metric"],
            row["min_samples"],
            row["eps_quantile"],
            row["threshold_percent"],
        )
    )
    composition_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["metric"],
            row["min_samples"],
            row["eps_quantile"],
            row["threshold_percent"],
            row["cluster_label"],
        )
    )
    noise_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["metric"],
            row["min_samples"],
            row["eps_quantile"],
            row["function_name"],
        )
    )

    clustering_study.write_csv(
        output_dir / "dbscan-internal-summary.csv", INTERNAL_HEADER, internal_rows
    )
    clustering_study.write_csv(
        output_dir / "dbscan-ground-truth-summary.csv", EXTERNAL_HEADER, external_rows
    )
    clustering_study.write_csv(
        output_dir / "dbscan-cluster-composition.csv",
        COMPOSITION_HEADER,
        composition_rows,
    )
    clustering_study.write_csv(
        output_dir / "dbscan-noise-functions.csv", NOISE_HEADER, noise_rows
    )
    clustering_study.write_csv(
        output_dir / "dbscan-eps-grid.csv", EPS_GRID_HEADER, eps_grid_rows
    )

    for metric in metrics:
        plot_silhouette_coverage(figures_dir, internal_rows, metric)
        plot_cluster_count_noise(figures_dir, internal_rows, metric)

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "reference_profiles": profile_meta,
        "feature_sets": {name: FEATURE_SETS[name] for name in feature_sets},
        "feature_set_roles": {
            "paper6": "paper-inspired baseline",
            "paper6_no_framework_runtime_ms": (
                "primary Serverledge PAPER-5 candidate; framework_runtime_ms "
                "is profiling/framework overhead in the current implementation"
            ),
            "paper4_no_framework_runtime_ms_no_page_faults_delta": (
                "ablation control for the page-fault/kernel-time correlation"
            ),
            "paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms": (
                "ablation control for the page-fault/kernel-time correlation"
            ),
        },
        "scaler": scaler_name,
        "algorithm": "dbscan",
        "metrics": metrics,
        "min_samples_values": min_samples_values,
        "eps_quantiles": eps_quantiles,
        "eps_selection_rule": (
            "per feature-set/metric/min_samples, eps candidates are quantiles "
            "of the k-distance curve; no architecture ground truth is used"
        ),
        "external_metric_rule": (
            "noise is excluded from external clustering metrics and coverage is "
            "reported explicitly"
        ),
        "ground_truth_thresholds_percent": [
            metadata["threshold_percent"] for _, metadata in preference_datasets
        ],
        "function_count": len(functions),
        "preference_inputs": [metadata for _, metadata in preference_datasets],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cli_invocation": sys.argv,
        "configuration_count": len(internal_rows),
        "methodological_notes": [
            "DBSCAN is fitted only on x86 reference-architecture features.",
            "Ground-truth labels are joined after clustering and never tune eps/min_samples.",
            "Euclidean, Manhattan and cosine neighborhoods are evaluated separately.",
            "Noise points are preserved as noise and are never manually removed from the corpus.",
            "External metrics are computed on clustered points only and must always be read together with coverage/noise count.",
            "No best configuration is automatically selected.",
        ],
    }
    clustering_study.write_json(output_dir / "dbscan-study-manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run the Serverledge DBSCAN feature-set robustness study using "
            "data-driven k-distance eps candidates."
        )
    )
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--preferences", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument(
        "--scaler",
        choices=preprocess.SCALERS,
        default="standard",
    )
    ap.add_argument(
        "--feature-set",
        action="append",
        default=[],
        choices=sorted(FEATURE_SETS),
        help="repeat to select feature sets; default: all four study sets",
    )
    ap.add_argument(
        "--metric",
        action="append",
        default=[],
        choices=DBSCAN_METRICS,
        help="repeat for DBSCAN distance metrics; default: euclidean, manhattan, cosine",
    )
    ap.add_argument(
        "--min-samples",
        action="append",
        type=int,
        default=[],
        help="repeat; default: 3,4,5,6",
    )
    ap.add_argument(
        "--eps-quantile",
        action="append",
        type=float,
        default=[],
        help="repeat as fraction in (0,1); default: 0.50,0.60,0.70,0.80,0.90,0.95",
    )
    return ap


def main() -> None:
    ap = parser()
    args = ap.parse_args()

    try:
        manifest = run_study(
            Path(args.profiles),
            [Path(path) for path in args.preferences],
            Path(args.output_dir),
            args.experiment_id,
            args.scaler,
            args.feature_set,
            args.metric,
            args.min_samples or list(DEFAULT_MIN_SAMPLES),
            args.eps_quantile or list(DEFAULT_EPS_QUANTILES),
        )
    except (DBSCANStudyError, clustering_study.StudyError, ValueError, OSError) as exc:
        ap.error(str(exc))

    print(
        f"functions={manifest['function_count']} "
        f"feature_sets={len(manifest['feature_sets'])} "
        f"metrics={len(manifest['metrics'])} "
        f"min_samples_values={len(manifest['min_samples_values'])} "
        f"eps_quantiles={len(manifest['eps_quantiles'])} "
        f"configurations={manifest['configuration_count']} "
        f"scaler={manifest['scaler']}"
    )
    print(f"output={Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
