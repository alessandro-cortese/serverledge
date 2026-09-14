#!/usr/bin/env python3
"""
Reproducible PAPER-6 clustering study for the Serverledge thesis.

The script is deliberately experimental and does NOT change the production
profiling/transfer feature contract in preprocess.FEATURE_NAMES.

Methodology implemented here:
- one aggregated FunctionProfile row per known function on the x86 reference
  architecture (median recommended; mean can be run as a sensitivity study);
- PAPER-6 baseline using the six features highlighted by the reference paper;
- systematic leave-one-feature-out (LOFO) ablation;
- K-Means fitted ONLY on x86 feature vectors;
- the same cluster assignments are evaluated against multiple architecture
  ground-truth thresholds without re-fitting the clustering;
- internal clustering metrics are kept separate from external ground-truth
  metrics;
- PCA is used ONLY for 2D visualization, never as K-Means input;
- correlation and assignment-stability diagnostics are post-hoc analyses and
  are never used to fit the clustering.

Inputs:
- --profiles: canonical FunctionProfile CSV produced by serverledge-profiling
  export-csv for the reference architecture;
- --preferences: one or more preference CSVs produced by preference.py,
  normally for thresholds 2.5, 5, 10, 15, 20 and 25 percent.

Outputs include CSV/JSON results and PNG/SVG figures suitable for inspection
and thesis reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import preference, preprocess


PAPER6_FEATURES = list(preprocess.FEATURE_NAMES)

LOFO_FEATURE_SETS = {
    f"paper6_no_{feature}": [
        candidate for candidate in PAPER6_FEATURES if candidate != feature
    ]
    for feature in PAPER6_FEATURES
}

# Data-driven candidates evaluated only after the systematic PAPER-6 baseline/LOFO
# study.  The PAPER-5 candidate is already represented by the corresponding LOFO
# set.  These PAPER-4 candidates test the two alternatives suggested by the
# observed PAPER-6 diagnostics: remove framework overhead and then remove one of
# the strongly related page-fault/kernel dimensions.
COMBINED_REDUCED_FEATURE_SETS = {
    "paper4_no_framework_runtime_ms_no_page_faults_delta": [
        feature
        for feature in PAPER6_FEATURES
        if feature not in {"framework_runtime_ms", "page_faults_delta"}
    ],
    "paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms": [
        feature
        for feature in PAPER6_FEATURES
        if feature not in {"framework_runtime_ms", "cpu_kernel_delta_ms"}
    ],
}

FEATURE_SETS = {
    "paper6": list(PAPER6_FEATURES),
    **LOFO_FEATURE_SETS,
    **COMBINED_REDUCED_FEATURE_SETS,
}

PREFERENCE_LABELS = (
    preference.PREFERENCE_X86,
    preference.PREFERENCE_ARM,
    preference.PREFERENCE_INDEPENDENT,
)

AMBIGUOUS_PREFERENCE = "ambiguous"

INTERNAL_HEADER = [
    "experiment_id",
    "feature_set",
    "feature_count",
    "features",
    "scaler",
    "k",
    "n_init",
    "random_state",
    "sample_count",
    "cluster_count",
    "min_cluster_size",
    "max_cluster_size",
    "singleton_count",
    "cluster_size_distribution",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "inertia",
    "assignment_ari_vs_paper6",
    "assignment_nmi_vs_paper6",
]

EXTERNAL_HEADER = [
    *INTERNAL_HEADER,
    "preference_run_id",
    "threshold_percent",
    "x86_preferred_count",
    "arm_preferred_count",
    "architecture_independent_count",
    "majority_ground_truth_class",
    "majority_ground_truth_count",
    "majority_ground_truth_share",
    "overall_purity",
    "purity_gain_over_majority_baseline",
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "normalized_mutual_information",
]

COMPOSITION_HEADER = [
    "experiment_id",
    "feature_set",
    "scaler",
    "k",
    "threshold_percent",
    "cluster_label",
    "cluster_size",
    "x86_preferred_count",
    "arm_preferred_count",
    "architecture_independent_count",
    "majority_preference",
    "cluster_purity",
]

ABLATION_HEADER = [
    "experiment_id",
    "feature_set",
    "removed_feature",
    "scaler",
    "k",
    "threshold_percent",
    "delta_silhouette",
    "delta_davies_bouldin",
    "delta_calinski_harabasz",
    "delta_inertia",
    "delta_overall_purity",
    "delta_homogeneity",
    "delta_adjusted_rand_index",
    "delta_normalized_mutual_information",
    "assignment_ari_vs_paper6",
    "assignment_nmi_vs_paper6",
]


class StudyError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def finite_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"row {row_number}: {field} is not numeric") from exc

    if not math.isfinite(parsed):
        raise StudyError(f"row {row_number}: {field} is not finite")

    return parsed


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reference_profiles(path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    path = path.expanduser().resolve()
    rows, matrix, experiment_id, aggregation = preprocess.load_source(path)

    machine_tags = {row["machine_tag"].strip() for row in rows}
    if len(machine_tags) != 1:
        raise StudyError(
            "reference FunctionProfile CSV must contain exactly one machine_tag; "
            f"found {sorted(machine_tags)}"
        )

    functions = [row["function_name"].strip() for row in rows]
    if len(functions) != len(set(functions)):
        raise StudyError("reference FunctionProfile CSV contains duplicate function names")

    if len(rows) < 3:
        raise StudyError("at least three reference functions are required")

    return rows, matrix, {
        "path": str(path),
        "sha256": sha256_file(path),
        "experiment_id": experiment_id,
        "aggregation": aggregation,
        "machine_tag": next(iter(machine_tags)),
        "sample_count": len(rows),
    }


def load_preference_dataset(path: Path) -> tuple[dict[str, dict], dict]:
    path = path.expanduser().resolve()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != preference.ARCHITECTURE_PREFERENCE_HEADER:
            raise StudyError(f"{path}: unexpected architecture preference CSV header")
        rows = list(reader)

    if not rows:
        raise StudyError(f"{path}: empty architecture preference dataset")

    by_function: dict[str, dict] = {}
    thresholds: set[float] = set()
    aggregations: set[str] = set()
    run_ids: set[str] = set()
    x86_tags: set[str] = set()
    arm_tags: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        label = row["architecture_preference"].strip()
        if label not in PREFERENCE_LABELS:
            raise StudyError(f"{path}: row {row_number}: invalid preference {label!r}")

        function_name = row["function_name"].strip()
        if not function_name:
            raise StudyError(f"{path}: row {row_number}: empty function name")
        if function_name in by_function:
            raise StudyError(f"{path}: duplicate function {function_name!r}")

        threshold = finite_float(row["threshold_percent"], "threshold_percent", row_number)
        delta = finite_float(
            row["arm_vs_x86_delta_percent"],
            "arm_vs_x86_delta_percent",
            row_number,
        )
        x86_duration = finite_float(row["x86_duration_ms"], "x86_duration_ms", row_number)
        arm_duration = finite_float(row["arm_duration_ms"], "arm_duration_ms", row_number)

        if threshold < 0 or x86_duration <= 0 or arm_duration <= 0:
            raise StudyError(f"{path}: row {row_number}: invalid threshold/duration")

        if row["performance_metric"].strip() != "duration_ms":
            raise StudyError(f"{path}: row {row_number}: performance metric is not duration_ms")

        by_function[function_name] = {
            "architecture_preference": label,
            "arm_vs_x86_delta_percent": delta,
            "threshold_percent": threshold,
            "x86_duration_ms": x86_duration,
            "arm_duration_ms": arm_duration,
        }

        thresholds.add(threshold)
        aggregations.add(row["aggregation"].strip())
        run_ids.add(row["preference_run_id"].strip())
        x86_tags.add(row["x86_machine_tag"].strip())
        arm_tags.add(row["arm_machine_tag"].strip())

    def single(values: set, label: str):
        if len(values) != 1:
            raise StudyError(f"{path}: dataset mixes {label}")
        return next(iter(values))

    return by_function, {
        "path": str(path),
        "sha256": sha256_file(path),
        "threshold_percent": single(thresholds, "thresholds"),
        "aggregation": single(aggregations, "aggregations"),
        "preference_run_id": single(run_ids, "preference run IDs"),
        "x86_machine_tag": single(x86_tags, "x86 machine tags"),
        "arm_machine_tag": single(arm_tags, "ARM machine tags"),
        "function_count": len(by_function),
    }


def load_preference_sweep(
    paths: list[Path],
    expected_functions: list[str],
    aggregation: str,
) -> list[tuple[dict[str, dict], dict]]:
    if not paths:
        raise StudyError("at least one --preferences dataset is required")

    expected = set(expected_functions)
    datasets = []
    seen_thresholds: set[float] = set()

    for path in paths:
        mapping, metadata = load_preference_dataset(path)

        if metadata["aggregation"] != aggregation:
            raise StudyError(
                f"{path}: preference aggregation {metadata['aggregation']!r} does not "
                f"match reference aggregation {aggregation!r}"
            )

        actual = set(mapping)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise StudyError(
                f"{path}: function set mismatch; missing={missing}, extra={extra}"
            )

        threshold = metadata["threshold_percent"]
        if threshold in seen_thresholds:
            raise StudyError(f"duplicate ground-truth threshold {threshold}")
        seen_thresholds.add(threshold)
        datasets.append((mapping, metadata))

    datasets.sort(key=lambda item: item[1]["threshold_percent"])
    return datasets


def fit_scaler(name: str, matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    if name == "none":
        return matrix.copy(), {"name": "none"}

    if name == "standard":
        scaler = StandardScaler()
    elif name == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    elif name == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
    else:
        raise StudyError(f"unsupported scaler {name!r}")

    transformed = scaler.fit_transform(matrix)
    state = {"name": name}

    if name == "standard":
        state.update({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    elif name == "robust":
        state.update(
            {
                "center": scaler.center_.tolist(),
                "scale": scaler.scale_.tolist(),
                "quantile_range": [25.0, 75.0],
            }
        )
    else:
        state.update(
            {
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                "scale": scaler.scale_.tolist(),
                "min": scaler.min_.tolist(),
            }
        )

    return transformed, state


def cluster_sizes(labels: np.ndarray) -> list[int]:
    return sorted(
        (int(np.sum(labels == label)) for label in sorted(set(int(v) for v in labels))),
        reverse=True,
    )


def internal_metrics(matrix: np.ndarray, labels: np.ndarray, inertia: float) -> dict:
    sizes = cluster_sizes(labels)
    return {
        "sample_count": int(matrix.shape[0]),
        "cluster_count": len(sizes),
        "min_cluster_size": min(sizes),
        "max_cluster_size": max(sizes),
        "singleton_count": sum(size == 1 for size in sizes),
        "cluster_size_distribution": "|".join(str(size) for size in sizes),
        "silhouette": float(silhouette_score(matrix, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "inertia": float(inertia),
    }


def evaluate_external(
    labels: np.ndarray,
    functions: list[str],
    preferences: dict[str, dict],
) -> tuple[dict, list[dict]]:
    true_labels = [preferences[name]["architecture_preference"] for name in functions]
    encoded_true = [PREFERENCE_LABELS.index(label) for label in true_labels]
    counts = Counter(true_labels)

    grouped: dict[int, list[str]] = defaultdict(list)
    for index, cluster_label in enumerate(labels):
        grouped[int(cluster_label)].append(true_labels[index])

    majority_correct = 0
    composition = []

    for cluster_label in sorted(grouped):
        members = grouped[cluster_label]
        member_counts = Counter(members)
        maximum = max(member_counts.values())
        winners = sorted(label for label, count in member_counts.items() if count == maximum)
        majority = winners[0] if len(winners) == 1 else AMBIGUOUS_PREFERENCE
        majority_correct += maximum

        composition.append(
            {
                "cluster_label": cluster_label,
                "cluster_size": len(members),
                "x86_preferred_count": member_counts[preference.PREFERENCE_X86],
                "arm_preferred_count": member_counts[preference.PREFERENCE_ARM],
                "architecture_independent_count": member_counts[
                    preference.PREFERENCE_INDEPENDENT
                ],
                "majority_preference": majority,
                "cluster_purity": maximum / len(members),
            }
        )

    majority_ground_truth_count = max(counts.values())
    majority_ground_truth_winners = sorted(
        label for label, count in counts.items() if count == majority_ground_truth_count
    )
    majority_ground_truth_class = (
        majority_ground_truth_winners[0]
        if len(majority_ground_truth_winners) == 1
        else AMBIGUOUS_PREFERENCE
    )
    majority_ground_truth_share = majority_ground_truth_count / len(true_labels)
    overall_purity = majority_correct / len(true_labels)

    return (
        {
            "x86_preferred_count": counts[preference.PREFERENCE_X86],
            "arm_preferred_count": counts[preference.PREFERENCE_ARM],
            "architecture_independent_count": counts[preference.PREFERENCE_INDEPENDENT],
            "majority_ground_truth_class": majority_ground_truth_class,
            "majority_ground_truth_count": majority_ground_truth_count,
            "majority_ground_truth_share": majority_ground_truth_share,
            "overall_purity": overall_purity,
            "purity_gain_over_majority_baseline": overall_purity - majority_ground_truth_share,
            "homogeneity": float(homogeneity_score(encoded_true, labels)),
            "completeness": float(completeness_score(encoded_true, labels)),
            "v_measure": float(v_measure_score(encoded_true, labels)),
            "adjusted_rand_index": float(adjusted_rand_score(encoded_true, labels)),
            "normalized_mutual_information": float(
                normalized_mutual_info_score(encoded_true, labels)
            ),
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
    k: int,
) -> None:
    header = [
        "feature_set",
        "scaler",
        "k",
        "function_name",
        "machine_tag",
        "configured_cpus",
        "configured_memory_mb",
        "sample_count",
        "cluster_label",
        *features,
    ]
    rows = []

    for index, source in enumerate(profile_rows):
        row = {
            "feature_set": feature_set,
            "scaler": scaler,
            "k": k,
            "function_name": source["function_name"],
            "machine_tag": source["machine_tag"],
            "configured_cpus": source["configured_cpus"],
            "configured_memory_mb": source["configured_memory_mb"],
            "sample_count": source["sample_count"],
            "cluster_label": int(labels[index]),
        }
        for column, feature in enumerate(features):
            row[feature] = format(float(transformed[index, column]), ".17g")
        rows.append(row)

    write_csv(path, header, rows)


def correlation_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    n_features = matrix.shape[1]
    result = np.full((n_features, n_features), np.nan, dtype=float)

    for i in range(n_features):
        for j in range(n_features):
            x = matrix[:, i]
            y = matrix[:, j]
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                if i == j:
                    result[i, j] = 1.0
                continue

            if method == "pearson":
                result[i, j] = float(pearsonr(x, y).statistic)
            elif method == "spearman":
                result[i, j] = float(spearmanr(x, y).statistic)
            else:
                raise StudyError(f"unknown correlation method {method!r}")

    return result


def save_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    header = ["feature", *names]
    rows = []
    for i, name in enumerate(names):
        row = {"feature": name}
        for j, other in enumerate(names):
            value = matrix[i, j]
            row[other] = "" if not math.isfinite(float(value)) else format(float(value), ".17g")
        rows.append(row)
    write_csv(path, header, rows)


def feature_distribution_diagnostics(
    output_dir: Path,
    raw_matrix: np.ndarray,
    functions: list[str],
) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    outlier_rows = []

    for index, feature in enumerate(PAPER6_FEATURES):
        values = raw_matrix[:, index].astype(float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        median = float(np.median(values))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))

        summary_rows.append(
            {
                "feature": feature,
                "mean": mean,
                "std": std,
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

        z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
        ranking = sorted(
            zip(functions, values, z_scores),
            key=lambda item: abs(float(item[2])),
            reverse=True,
        )
        for rank, (function_name, value, z_score) in enumerate(ranking, start=1):
            outlier_rows.append(
                {
                    "feature": feature,
                    "rank_by_abs_z": rank,
                    "function_name": function_name,
                    "raw_value": float(value),
                    "z_score": float(z_score),
                    "abs_z_score": abs(float(z_score)),
                    "is_abs_z_ge_3": abs(float(z_score)) >= 3.0,
                }
            )

    write_csv(
        output_dir / "feature-summary.csv",
        ["feature", "mean", "std", "median", "q1", "q3", "iqr", "min", "max"],
        summary_rows,
    )
    write_csv(
        output_dir / "feature-outlier-zscores.csv",
        [
            "feature",
            "rank_by_abs_z",
            "function_name",
            "raw_value",
            "z_score",
            "abs_z_score",
            "is_abs_z_ge_3",
        ],
        outlier_rows,
    )

    max_by_function = []
    for function_name in functions:
        rows = [row for row in outlier_rows if row["function_name"] == function_name]
        strongest = max(rows, key=lambda row: row["abs_z_score"])
        max_by_function.append(strongest)

    max_by_function.sort(key=lambda row: row["abs_z_score"], reverse=True)
    top = max_by_function[: min(15, len(max_by_function))]

    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(top))
    ax.barh(positions, [row["abs_z_score"] for row in top])
    ax.set_yticks(positions, labels=[row["function_name"] for row in top])
    ax.invert_yaxis()
    ax.axvline(3.0, linewidth=1)
    ax.set_xlabel("Maximum absolute z-score across PAPER-6 features")
    ax.set_title("Most extreme x86 function profiles")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "feature-outlier-max-abs-z.png", dpi=220)
    fig.savefig(output_dir / "figures" / "feature-outlier-max-abs-z.svg")
    plt.close(fig)

    return summary_rows, outlier_rows


def plot_purity_baseline_sensitivity(path_base: Path, external_rows: list[dict]) -> None:
    paper6 = [row for row in external_rows if row["feature_set"] == "paper6"]
    if not paper6:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in paper6:
        by_k[int(row["k"])].append(row)

    for k, rows in sorted(by_k.items()):
        rows = sorted(rows, key=lambda row: row["threshold_percent"])
        ax.plot(
            [row["threshold_percent"] for row in rows],
            [row["overall_purity"] for row in rows],
            marker="o",
            label=f"K={k}",
        )

    baseline_by_threshold = {}
    for row in paper6:
        baseline_by_threshold[row["threshold_percent"]] = row["majority_ground_truth_share"]
    thresholds = sorted(baseline_by_threshold)
    ax.plot(
        thresholds,
        [baseline_by_threshold[value] for value in thresholds],
        marker="x",
        linestyle="--",
        linewidth=2,
        label="majority-class baseline",
    )
    ax.set_xlabel("Ground-truth threshold (%)")
    ax.set_ylabel("Purity / majority-class share")
    ax.set_title("PAPER-6 purity sensitivity and majority-class baseline")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_external_metric_heatmap(
    path_base: Path,
    external_rows: list[dict],
    metric: str,
    title: str,
) -> None:
    rows = [row for row in external_rows if row["feature_set"] == "paper6"]
    if not rows:
        return

    ks = sorted({int(row["k"]) for row in rows})
    thresholds = sorted({float(row["threshold_percent"]) for row in rows})
    lookup = {(int(row["k"]), float(row["threshold_percent"])): float(row[metric]) for row in rows}
    matrix = np.asarray([[lookup[(k, threshold)] for threshold in thresholds] for k in ks])

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(thresholds)), labels=[f"{value:g}" for value in thresholds])
    ax.set_yticks(range(len(ks)), labels=[str(value) for value in ks])
    ax.set_xlabel("Ground-truth threshold (%)")
    ax.set_ylabel("K")
    ax.set_title(title)
    for i in range(len(ks)):
        for j in range(len(thresholds)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_correlation_heatmap(path_base: Path, names: list[str], matrix: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_ground_truth_distribution(path_base: Path, rows: list[dict]) -> None:
    thresholds = [row["threshold_percent"] for row in rows]
    x86 = [row["x86_preferred_count"] for row in rows]
    arm = [row["arm_preferred_count"] for row in rows]
    independent = [row["architecture_independent_count"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, x86, marker="o", label=preference.PREFERENCE_X86)
    ax.plot(thresholds, arm, marker="o", label=preference.PREFERENCE_ARM)
    ax.plot(thresholds, independent, marker="o", label=preference.PREFERENCE_INDEPENDENT)
    ax.set_xlabel("Ground-truth threshold (%)")
    ax.set_ylabel("Number of functions")
    ax.set_title("Ground-truth class distribution vs threshold")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_k_sweep(path_base: Path, internal_rows: list[dict]) -> None:
    by_feature_set: dict[str, list[dict]] = defaultdict(list)
    for row in internal_rows:
        by_feature_set[row["feature_set"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_set, rows in sorted(by_feature_set.items()):
        rows = sorted(rows, key=lambda item: item["k"])
        ax.plot(
            [row["k"] for row in rows],
            [row["silhouette"] for row in rows],
            marker="o",
            label=feature_set,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.set_title("K-Means silhouette across K and feature ablations")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_set, rows in sorted(by_feature_set.items()):
        rows = sorted(rows, key=lambda item: item["k"])
        ax.plot(
            [row["k"] for row in rows],
            [row["inertia"] for row in rows],
            marker="o",
            label=feature_set,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means elbow curves across feature ablations")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_name(path_base.name + "-elbow").with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_name(path_base.name + "-elbow").with_suffix(".svg"))
    plt.close(fig)


def plot_pca_pair(
    path_base: Path,
    transformed: np.ndarray,
    labels: np.ndarray,
    functions: list[str],
    preferences: dict[str, dict],
    title_suffix: str,
    annotate: bool,
) -> dict:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(transformed)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster_label in sorted(set(int(v) for v in labels)):
        mask = labels == cluster_label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"cluster {cluster_label}")

    if annotate:
        for index, name in enumerate(functions):
            ax.annotate(name, (coords[index, 0], coords[index, 1]), fontsize=6, alpha=0.75)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA visualization — K-Means clusters — {title_suffix}")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_name(path_base.name + "-clusters").with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_name(path_base.name + "-clusters").with_suffix(".svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for pref_label in PREFERENCE_LABELS:
        mask = np.asarray(
            [preferences[name]["architecture_preference"] == pref_label for name in functions]
        )
        ax.scatter(coords[mask, 0], coords[mask, 1], label=pref_label)

    if annotate:
        for index, name in enumerate(functions):
            ax.annotate(name, (coords[index, 0], coords[index, 1]), fontsize=6, alpha=0.75)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA visualization — ground truth — {title_suffix}")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_name(path_base.name + "-ground-truth").with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_name(path_base.name + "-ground-truth").with_suffix(".svg"))
    plt.close(fig)

    return {
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_ratio_pc2": float(pca.explained_variance_ratio_[1]),
        "explained_variance_ratio_pc1_pc2": float(np.sum(pca.explained_variance_ratio_)),
        "coordinates": coords,
    }


def plot_composition(path_base: Path, composition: list[dict], title_suffix: str) -> None:
    labels = [str(row["cluster_label"]) for row in composition]
    sizes = np.asarray([row["cluster_size"] for row in composition], dtype=float)
    bottom = np.zeros(len(composition), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    for pref_label, key in (
        (preference.PREFERENCE_X86, "x86_preferred_count"),
        (preference.PREFERENCE_ARM, "arm_preferred_count"),
        (preference.PREFERENCE_INDEPENDENT, "architecture_independent_count"),
    ):
        counts = np.asarray([row[key] for row in composition], dtype=float)
        pct = np.divide(counts * 100.0, sizes, out=np.zeros_like(counts), where=sizes != 0)
        ax.bar(labels, pct, bottom=bottom, label=pref_label)
        bottom += pct

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Composition (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Cluster ground-truth composition — {title_suffix}")
    ax.grid(axis="y", alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_ablation_deltas(path_base: Path, rows: list[dict], primary_k: int, primary_threshold: float) -> None:
    selected = [
        row
        for row in rows
        if row["k"] == primary_k
        and math.isclose(row["threshold_percent"], primary_threshold, abs_tol=1e-12)
    ]
    if not selected:
        return

    labels = [row["removed_feature"] for row in selected]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, [row["delta_silhouette"] for row in selected], width, label="Δ silhouette")
    ax.bar(x, [row["delta_overall_purity"] for row in selected], width, label="Δ purity")
    ax.bar(x + width, [row["delta_homogeneity"] for row in selected], width, label="Δ homogeneity")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Difference from PAPER-6 baseline")
    ax.set_title(
        f"Leave-one-feature-out effect — K={primary_k}, threshold={primary_threshold:g}%"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def feature_vs_delta_diagnostics(
    output_dir: Path,
    raw_matrix: np.ndarray,
    functions: list[str],
    primary_preferences: dict[str, dict],
) -> list[dict]:
    delta = np.asarray(
        [primary_preferences[name]["arm_vs_x86_delta_percent"] for name in functions],
        dtype=float,
    )
    rows = []

    for index, feature in enumerate(PAPER6_FEATURES):
        values = raw_matrix[:, index]
        if np.allclose(values, values[0]):
            pearson = math.nan
            spearman = math.nan
        else:
            pearson = float(pearsonr(values, delta).statistic)
            spearman = float(spearmanr(values, delta).statistic)

        rows.append(
            {
                "feature": feature,
                "pearson_vs_arm_x86_delta": pearson,
                "spearman_vs_arm_x86_delta": spearman,
            }
        )

    write_csv(
        output_dir / "feature-vs-architecture-delta.csv",
        ["feature", "pearson_vs_arm_x86_delta", "spearman_vs_arm_x86_delta"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(
        x - width / 2,
        [row["pearson_vs_arm_x86_delta"] for row in rows],
        width,
        label="Pearson",
    )
    ax.bar(
        x + width / 2,
        [row["spearman_vs_arm_x86_delta"] for row in rows],
        width,
        label="Spearman",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, labels=[row["feature"] for row in rows], rotation=35, ha="right")
    ax.set_ylabel("Correlation with ARM-vs-x86 performance delta")
    ax.set_title("Post-hoc feature correlation with architecture performance delta")
    ax.grid(axis="y", alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature-vs-architecture-delta.png", dpi=220)
    fig.savefig(output_dir / "feature-vs-architecture-delta.svg")
    plt.close(fig)

    return rows


def build_ablation_rows(external_rows: list[dict]) -> list[dict]:
    baseline = {}
    for row in external_rows:
        if row["feature_set"] == "paper6":
            baseline[(row["scaler"], row["k"], row["threshold_percent"])] = row

    output = []
    for row in external_rows:
        if row["feature_set"] not in LOFO_FEATURE_SETS:
            continue

        key = (row["scaler"], row["k"], row["threshold_percent"])
        base = baseline.get(key)
        if base is None:
            raise StudyError(f"missing PAPER-6 baseline for {key}")

        prefix = "paper6_no_"
        removed = row["feature_set"][len(prefix):] if row["feature_set"].startswith(prefix) else ""

        output.append(
            {
                "experiment_id": row["experiment_id"],
                "feature_set": row["feature_set"],
                "removed_feature": removed,
                "scaler": row["scaler"],
                "k": row["k"],
                "threshold_percent": row["threshold_percent"],
                "delta_silhouette": row["silhouette"] - base["silhouette"],
                "delta_davies_bouldin": row["davies_bouldin"] - base["davies_bouldin"],
                "delta_calinski_harabasz": row["calinski_harabasz"] - base["calinski_harabasz"],
                "delta_inertia": row["inertia"] - base["inertia"],
                "delta_overall_purity": row["overall_purity"] - base["overall_purity"],
                "delta_homogeneity": row["homogeneity"] - base["homogeneity"],
                "delta_adjusted_rand_index": (
                    row["adjusted_rand_index"] - base["adjusted_rand_index"]
                ),
                "delta_normalized_mutual_information": (
                    row["normalized_mutual_information"]
                    - base["normalized_mutual_information"]
                ),
                "assignment_ari_vs_paper6": row["assignment_ari_vs_paper6"],
                "assignment_nmi_vs_paper6": row["assignment_nmi_vs_paper6"],
            }
        )

    return output


def write_markdown_report(
    output_dir: Path,
    experiment_id: str,
    profile_meta: dict,
    ground_truth_rows: list[dict],
    pearson_matrix: np.ndarray,
    spearman_matrix: np.ndarray,
    outlier_rows: list[dict],
    internal_rows: list[dict],
    external_rows: list[dict],
    ablation_rows: list[dict],
    primary_threshold: float,
    primary_k: int,
    scaler_name: str,
    k_values: list[int],
) -> None:
    def f4(value) -> str:
        return f"{float(value):.4f}"

    paper6_internal = sorted(
        [row for row in internal_rows if row["feature_set"] == "paper6"],
        key=lambda row: row["k"],
    )
    paper6_primary_external = sorted(
        [
            row
            for row in external_rows
            if row["feature_set"] == "paper6"
            and math.isclose(row["threshold_percent"], primary_threshold, abs_tol=1e-12)
        ],
        key=lambda row: row["k"],
    )

    strong_correlations = []
    for i, left in enumerate(PAPER6_FEATURES):
        for j in range(i + 1, len(PAPER6_FEATURES)):
            right = PAPER6_FEATURES[j]
            pearson = float(pearson_matrix[i, j])
            spearman = float(spearman_matrix[i, j])
            if max(abs(pearson), abs(spearman)) >= 0.5:
                strong_correlations.append((left, right, pearson, spearman))

    extreme = sorted(
        [row for row in outlier_rows if row["is_abs_z_ge_3"]],
        key=lambda row: row["abs_z_score"],
        reverse=True,
    )

    candidate_sets = [
        "paper6",
        "paper6_no_framework_runtime_ms",
        "paper4_no_framework_runtime_ms_no_page_faults_delta",
        "paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms",
    ]
    candidate_primary = {
        row["feature_set"]: row
        for row in external_rows
        if row["feature_set"] in candidate_sets
        and row["k"] == primary_k
        and math.isclose(row["threshold_percent"], primary_threshold, abs_tol=1e-12)
    }

    lines = [
        "# Analisi del clustering delle funzioni Serverledge",
        "",
        f"Esperimento: `{experiment_id}`  ",
        f"Generato: `{datetime.now(timezone.utc).isoformat()}`  ",
        f"Architettura di riferimento: `{profile_meta['machine_tag']}`  ",
        f"Aggregazione: `{profile_meta['aggregation']}`  ",
        f"Funzioni: `{profile_meta['sample_count']}`  ",
        f"Scaler: `{scaler_name}`  ",
        f"K valutati: `{', '.join(str(v) for v in k_values)}`",
        "",
        "## Regole metodologiche",
        "",
        "Il clustering viene adattato esclusivamente sui profili della reference architecture x86. "
        "Le etichette x86-preferred, ARM-preferred e architecture-independent vengono aggiunte solo dopo "
        "il clustering e non partecipano al fit, alla standardizzazione o alla costruzione delle distanze.",
        "",
        "La configurazione PAPER-6 usa `page_faults_delta`, `utilized_cpus`, `free_memory_mb`, "
        "`cpu_user_delta_ms`, `cpu_kernel_delta_ms` e `framework_runtime_ms`. La PCA è usata solo per "
        "visualizzare i punti in due dimensioni e non viene usata come input di K-Means.",
        "",
        "La soglia del 15% è il riferimento principale. Le soglie 2.5%, 5%, 10%, 20% e 25% sono "
        "analisi di sensibilità: cambiare la soglia cambia soltanto la ground truth, non i cluster.",
        "",
        "## Distribuzione della ground truth",
        "",
        "| Soglia | x86-preferred | ARM-preferred | Independent | Quota classe maggioritaria |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ground_truth_rows, key=lambda item: item["threshold_percent"]):
        counts = [
            row["x86_preferred_count"],
            row["arm_preferred_count"],
            row["architecture_independent_count"],
        ]
        majority = max(counts) / row["function_count"]
        lines.append(
            f"| {row['threshold_percent']:g}% | {row['x86_preferred_count']} | "
            f"{row['arm_preferred_count']} | {row['architecture_independent_count']} | {majority:.4f} |"
        )

    lines += [
        "",
        "La quota della classe maggioritaria è riportata perché la purity può risultare elevata anche "
        "senza un reale allineamento tra clustering e ground truth quando `architecture-independent` domina.",
        "",
        "## PAPER-6: struttura interna K-Means",
        "",
        "| K | Distribuzione cluster | Singleton | Silhouette | Davies-Bouldin | Calinski-Harabasz | Inertia |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in paper6_internal:
        lines.append(
            f"| {row['k']} | {row['cluster_size_distribution']} | {row['singleton_count']} | "
            f"{f4(row['silhouette'])} | {f4(row['davies_bouldin'])} | "
            f"{float(row['calinski_harabasz']):.2f} | {float(row['inertia']):.2f} |"
        )

    lines += [
        "",
        f"## PAPER-6: confronto con ground truth a {primary_threshold:g}%",
        "",
        "| K | Purity | Baseline maggioritaria | Gain purity | Homogeneity | ARI | NMI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper6_primary_external:
        lines.append(
            f"| {row['k']} | {f4(row['overall_purity'])} | {f4(row['majority_ground_truth_share'])} | "
            f"{f4(row['purity_gain_over_majority_baseline'])} | {f4(row['homogeneity'])} | "
            f"{f4(row['adjusted_rand_index'])} | {f4(row['normalized_mutual_information'])} |"
        )

    lines += [
        "",
        "La purity va quindi interpretata insieme ad homogeneity, ARI e NMI. Non viene selezionato un K "
        "massimizzando le metriche esterne, perché ciò userebbe la ground truth per scegliere un clustering "
        "che deve restare unsupervised.",
        "",
        "## Correlazione fra feature PAPER-6",
        "",
    ]
    if strong_correlations:
        lines += [
            "| Feature A | Feature B | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
        for left, right, pearson, spearman in strong_correlations:
            lines.append(f"| {left} | {right} | {pearson:.4f} | {spearman:.4f} |")
    else:
        lines.append("Nessuna coppia supera |r| o |ρ| = 0.5.")

    lines += [
        "",
        "## Profili x86 estremi",
        "",
        "Sono riportati i casi con `|z| >= 3` calcolati sulle feature aggregate x86. Questi punti non vengono "
        "rimossi automaticamente: possono rappresentare workload realmente distinti e saranno verificati anche "
        "con algoritmi density-based.",
        "",
        "| Funzione | Feature | Valore | z-score |",
        "|---|---|---:|---:|",
    ]
    for row in extreme:
        lines.append(
            f"| {row['function_name']} | {row['feature']} | {float(row['raw_value']):.4f} | "
            f"{float(row['z_score']):+.3f} |"
        )

    lines += [
        "",
        "## Interpretazione di `framework_runtime_ms`",
        "",
        "Nel profiler Serverledge questa feature è la somma di `ProfilingStartOverheadMs` e "
        "`SnapshotStartOverheadMs`: rappresenta quindi overhead del framework/profiling, non la durata "
        "dell'esecuzione della funzione. Per questo viene mantenuta nel PAPER-6 baseline, ma la sua esclusione "
        "è valutata esplicitamente come candidato PAPER-5.",
        "",
        "## Feature set ridotti candidati",
        "",
        f"La tabella seguente mostra le configurazioni candidate alla soglia {primary_threshold:g}% e K={primary_k}. "
        "Questa tabella non costituisce una selezione automatica del modello.",
        "",
        "| Feature set | # feature | Silhouette | Purity | Gain purity | Homogeneity | GT-ARI | Assignment ARI vs PAPER-6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature_set in candidate_sets:
        row = candidate_primary.get(feature_set)
        if row is None:
            continue
        lines.append(
            f"| `{feature_set}` | {row['feature_count']} | {f4(row['silhouette'])} | "
            f"{f4(row['overall_purity'])} | {f4(row['purity_gain_over_majority_baseline'])} | "
            f"{f4(row['homogeneity'])} | {f4(row['adjusted_rand_index'])} | "
            f"{f4(row['assignment_ari_vs_paper6'])} |"
        )

    framework_ablation = sorted(
        [row for row in ablation_rows if row["removed_feature"] == "framework_runtime_ms"
         and math.isclose(row["threshold_percent"], primary_threshold, abs_tol=1e-12)],
        key=lambda row: row["k"],
    )
    if framework_ablation:
        lines += [
            "",
            "### Effetto della rimozione di `framework_runtime_ms`",
            "",
            "| K | Δ silhouette | Δ purity | Δ homogeneity | Δ GT-ARI | Assignment ARI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in framework_ablation:
            lines.append(
                f"| {row['k']} | {float(row['delta_silhouette']):+.4f} | "
                f"{float(row['delta_overall_purity']):+.4f} | "
                f"{float(row['delta_homogeneity']):+.4f} | "
                f"{float(row['delta_adjusted_rand_index']):+.4f} | "
                f"{float(row['assignment_ari_vs_paper6']):.4f} |"
            )

    lines += [
        "",
        "## Evidenze e decisioni correnti",
        "",
        "1. Il PAPER-6 completo resta il baseline principale perché riproduce il sottoinsieme di feature "
        "ritenuto importante nel lavoro di riferimento.",
        "2. `framework_runtime_ms` viene trattata come candidata all'esclusione nel reduced set perché in "
        "Serverledge misura overhead di profiling/framework e non runtime applicativo; la sua rimozione è inoltre "
        "valutata quantitativamente, senza usare la ground truth nel fit.",
        "3. `page_faults_delta` e `cpu_kernel_delta_ms` vengono testate alternativamente nel reduced set perché "
        "mostrano una relazione forte e possono pesare due volte una componente simile della geometria. Nessuna "
        "delle due viene eliminata a priori.",
        "4. I punti estremi non vengono cancellati dal corpus. La fase successiva usa DBSCAN con distanze "
        "Euclidean, Manhattan e Cosine per verificare se emergono naturalmente come noise o piccoli gruppi.",
        "5. Nessuna configurazione viene scelta massimizzando purity/homogeneity/ARI/NMI. Le metriche esterne "
        "servono a valutare a posteriori la corrispondenza con la ground truth.",
        "",
        "## File di evidenza generati",
        "",
        "- `ground-truth-distribution.csv` e `figures/ground-truth-distribution.{png,svg}`",
        "- `feature-summary.csv`",
        "- `feature-outlier-zscores.csv` e `figures/feature-outlier-max-abs-z.{png,svg}`",
        "- `feature-correlation-pearson.csv` e `feature-correlation-spearman.csv`",
        "- `figures/feature-correlation-pearson.{png,svg}` e `figures/feature-correlation-spearman.{png,svg}`",
        "- `feature-vs-architecture-delta.csv` e relativa figura",
        "- `kmeans-internal-summary.csv`",
        "- `kmeans-ground-truth-summary.csv`",
        "- `cluster-composition.csv`",
        "- `ablation-deltas.csv`",
        "- `pca-coordinates-primary.csv` e figure PCA",
        "- `figures/paper6-purity-vs-majority-baseline.{png,svg}`",
        "- `figures/paper6-homogeneity-sensitivity.{png,svg}`",
        "- `figures/paper6-ari-sensitivity.{png,svg}`",
        "- `figures/paper6-nmi-sensitivity.{png,svg}`",
        "- `clustering-study-manifest.json`",
        "",
        "## Riproducibilità",
        "",
        f"Input FunctionProfile: `{profile_meta['path']}`  ",
        f"SHA-256: `{profile_meta['sha256']}`  ",
        f"scikit-learn: `{sklearn.__version__}`  ",
        f"Python: `{platform.python_version()}`",
        "",
        "Il manifest conserva inoltre le soglie di ground truth, gli hash dei file preference, il seed, `n_init`, "
        "la lista delle feature e l'invocazione della CLI. I risultati DBSCAN verranno aggiunti a questa "
        "documentazione nella fase di robustness analysis.",
        "",
    ]

    (output_dir / "CLUSTERING_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def run_study(
    profiles_path: Path,
    preference_paths: list[Path],
    output_dir: Path,
    experiment_id: str,
    scaler_name: str,
    k_values: list[int],
    primary_k: int,
    primary_threshold: float,
    n_init: int,
    random_state: int,
    annotate_pca: bool,
) -> dict:
    experiment_id = experiment_id.strip()
    if not experiment_id:
        raise StudyError("experiment ID cannot be empty")
    if scaler_name not in preprocess.SCALERS:
        raise StudyError(f"unsupported scaler {scaler_name!r}")
    if n_init <= 0:
        raise StudyError("n_init must be positive")

    k_values = list(dict.fromkeys(int(value) for value in k_values))
    if not k_values:
        raise StudyError("at least one K value is required")

    profile_rows, raw_matrix, profile_meta = load_reference_profiles(profiles_path)
    functions = [row["function_name"].strip() for row in profile_rows]

    for k in k_values:
        if not 2 <= k < len(profile_rows):
            raise StudyError(f"K={k} is invalid for {len(profile_rows)} functions")
    if primary_k not in k_values:
        raise StudyError("primary K must be included in the K sweep")

    preference_datasets = load_preference_sweep(
        preference_paths,
        functions,
        profile_meta["aggregation"],
    )
    thresholds = [metadata["threshold_percent"] for _, metadata in preference_datasets]
    if not any(math.isclose(primary_threshold, value, abs_tol=1e-12) for value in thresholds):
        raise StudyError(
            f"primary threshold {primary_threshold:g}% is not among loaded thresholds {thresholds}"
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Feature correlation is computed on the unscaled, aggregated x86 profiles.
    pearson_matrix = correlation_matrix(raw_matrix, "pearson")
    spearman_matrix = correlation_matrix(raw_matrix, "spearman")
    save_matrix_csv(output_dir / "feature-correlation-pearson.csv", PAPER6_FEATURES, pearson_matrix)
    save_matrix_csv(output_dir / "feature-correlation-spearman.csv", PAPER6_FEATURES, spearman_matrix)
    plot_correlation_heatmap(
        figures_dir / "feature-correlation-pearson",
        PAPER6_FEATURES,
        pearson_matrix,
        "PAPER-6 feature correlation — Pearson",
    )
    plot_correlation_heatmap(
        figures_dir / "feature-correlation-spearman",
        PAPER6_FEATURES,
        spearman_matrix,
        "PAPER-6 feature correlation — Spearman",
    )
    feature_summary_rows, outlier_rows = feature_distribution_diagnostics(
        output_dir, raw_matrix, functions
    )

    ground_truth_rows = []
    for mapping, metadata in preference_datasets:
        counts = Counter(mapping[name]["architecture_preference"] for name in functions)
        ground_truth_rows.append(
            {
                "threshold_percent": metadata["threshold_percent"],
                "x86_preferred_count": counts[preference.PREFERENCE_X86],
                "arm_preferred_count": counts[preference.PREFERENCE_ARM],
                "architecture_independent_count": counts[preference.PREFERENCE_INDEPENDENT],
                "function_count": len(functions),
                "preference_run_id": metadata["preference_run_id"],
                "preferences_sha256": metadata["sha256"],
            }
        )

    write_csv(
        output_dir / "ground-truth-distribution.csv",
        [
            "threshold_percent",
            "x86_preferred_count",
            "arm_preferred_count",
            "architecture_independent_count",
            "function_count",
            "preference_run_id",
            "preferences_sha256",
        ],
        ground_truth_rows,
    )
    plot_ground_truth_distribution(figures_dir / "ground-truth-distribution", ground_truth_rows)

    primary_preferences, primary_pref_meta = min(
        preference_datasets,
        key=lambda item: abs(item[1]["threshold_percent"] - primary_threshold),
    )
    feature_vs_delta_diagnostics(output_dir, raw_matrix, functions, primary_preferences)

    column_index = {name: index for index, name in enumerate(PAPER6_FEATURES)}
    internal_rows = []
    external_rows = []
    composition_rows = []
    pca_rows = []
    fitted: dict[tuple[str, int], dict] = {}

    # Fit clustering once per feature-set/K. Ground-truth thresholds are applied later.
    for feature_set, features in FEATURE_SETS.items():
        indexes = [column_index[feature] for feature in features]
        subset_raw = raw_matrix[:, indexes]
        transformed, scaler_state = fit_scaler(scaler_name, subset_raw)

        for k in k_values:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=n_init,
                random_state=random_state,
                algorithm="lloyd",
            )
            labels = model.fit_predict(transformed).astype(np.int64)
            metrics = internal_metrics(transformed, labels, float(model.inertia_))
            configuration_id = f"{feature_set}__{scaler_name}__k{k}"
            run_dir = runs_dir / configuration_id
            run_dir.mkdir(parents=True, exist_ok=True)

            save_assignments(
                run_dir / "assignments.csv",
                profile_rows,
                transformed,
                labels,
                feature_set,
                features,
                scaler_name,
                k,
            )

            fitted[(feature_set, k)] = {
                "labels": labels,
                "transformed": transformed,
                "features": features,
                "scaler_state": scaler_state,
                "run_dir": run_dir,
                "metrics": metrics,
            }

    # Cluster stability compares each LOFO assignment to the PAPER-6 assignment.
    for (feature_set, k), result in fitted.items():
        baseline_labels = fitted[("paper6", k)]["labels"]
        if feature_set == "paper6":
            assignment_ari = 1.0
            assignment_nmi = 1.0
        else:
            assignment_ari = float(adjusted_rand_score(baseline_labels, result["labels"]))
            assignment_nmi = float(
                normalized_mutual_info_score(baseline_labels, result["labels"])
            )

        base = {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "feature_count": len(result["features"]),
            "features": "|".join(result["features"]),
            "scaler": scaler_name,
            "k": k,
            "n_init": n_init,
            "random_state": random_state,
            **result["metrics"],
            "assignment_ari_vs_paper6": assignment_ari,
            "assignment_nmi_vs_paper6": assignment_nmi,
        }
        internal_rows.append(base)

        threshold_documents = []
        for preferences, pref_meta in preference_datasets:
            ext, composition = evaluate_external(result["labels"], functions, preferences)
            external_row = {
                **base,
                "preference_run_id": pref_meta["preference_run_id"],
                "threshold_percent": pref_meta["threshold_percent"],
                **ext,
            }
            external_rows.append(external_row)

            for cluster_row in composition:
                composition_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "feature_set": feature_set,
                        "scaler": scaler_name,
                        "k": k,
                        "threshold_percent": pref_meta["threshold_percent"],
                        **cluster_row,
                    }
                )

            threshold_documents.append(
                {
                    "threshold_percent": pref_meta["threshold_percent"],
                    "preference_run_id": pref_meta["preference_run_id"],
                    "preferences_sha256": pref_meta["sha256"],
                    "external_metrics": ext,
                    "cluster_composition": composition,
                }
            )

            if k == primary_k and math.isclose(
                pref_meta["threshold_percent"], primary_threshold, abs_tol=1e-12
            ):
                plot_composition(
                    figures_dir / f"composition-{feature_set}-k{k}-tau{threshold_slug(primary_threshold)}",
                    composition,
                    f"{feature_set}, K={k}, threshold={primary_threshold:g}%",
                )

        write_json(
            result["run_dir"] / "evaluation-all-thresholds.json",
            {
                "experiment_id": experiment_id,
                "configuration_id": f"{feature_set}__{scaler_name}__k{k}",
                "feature_set": feature_set,
                "feature_names": result["features"],
                "scaler": scaler_name,
                "scaler_state": result["scaler_state"],
                "algorithm": "kmeans",
                "metric": "euclidean",
                "k": k,
                "n_init": n_init,
                "random_state": random_state,
                "internal_metrics": result["metrics"],
                "assignment_ari_vs_paper6": assignment_ari,
                "assignment_nmi_vs_paper6": assignment_nmi,
                "ground_truth_evaluations": threshold_documents,
            },
        )

        if k == primary_k:
            pca_meta = plot_pca_pair(
                figures_dir / f"pca-{feature_set}-k{k}-tau{threshold_slug(primary_threshold)}",
                result["transformed"],
                result["labels"],
                functions,
                primary_preferences,
                f"{feature_set}, K={k}, threshold={primary_threshold:g}%",
                annotate_pca,
            )
            coords = pca_meta.pop("coordinates")
            for index, name in enumerate(functions):
                pca_rows.append(
                    {
                        "feature_set": feature_set,
                        "k": k,
                        "function_name": name,
                        "cluster_label": int(result["labels"][index]),
                        "architecture_preference": primary_preferences[name][
                            "architecture_preference"
                        ],
                        "pc1": float(coords[index, 0]),
                        "pc2": float(coords[index, 1]),
                        **pca_meta,
                    }
                )

    internal_rows.sort(key=lambda row: (row["feature_set"], row["k"]))
    external_rows.sort(
        key=lambda row: (row["feature_set"], row["k"], row["threshold_percent"])
    )
    composition_rows.sort(
        key=lambda row: (
            row["feature_set"],
            row["k"],
            row["threshold_percent"],
            row["cluster_label"],
        )
    )

    write_csv(output_dir / "kmeans-internal-summary.csv", INTERNAL_HEADER, internal_rows)
    write_csv(output_dir / "kmeans-ground-truth-summary.csv", EXTERNAL_HEADER, external_rows)
    write_csv(output_dir / "cluster-composition.csv", COMPOSITION_HEADER, composition_rows)
    write_csv(
        output_dir / "pca-coordinates-primary.csv",
        [
            "feature_set",
            "k",
            "function_name",
            "cluster_label",
            "architecture_preference",
            "pc1",
            "pc2",
            "explained_variance_ratio_pc1",
            "explained_variance_ratio_pc2",
            "explained_variance_ratio_pc1_pc2",
        ],
        pca_rows,
    )

    ablation_rows = build_ablation_rows(external_rows)
    write_csv(output_dir / "ablation-deltas.csv", ABLATION_HEADER, ablation_rows)

    plot_k_sweep(figures_dir / "kmeans-silhouette", internal_rows)
    plot_ablation_deltas(
        figures_dir / "ablation-deltas-primary",
        ablation_rows,
        primary_k,
        primary_threshold,
    )
    plot_purity_baseline_sensitivity(
        figures_dir / "paper6-purity-vs-majority-baseline", external_rows
    )
    plot_external_metric_heatmap(
        figures_dir / "paper6-homogeneity-sensitivity",
        external_rows,
        "homogeneity",
        "PAPER-6 homogeneity across K and ground-truth threshold",
    )
    plot_external_metric_heatmap(
        figures_dir / "paper6-ari-sensitivity",
        external_rows,
        "adjusted_rand_index",
        "PAPER-6 adjusted Rand index across K and ground-truth threshold",
    )
    plot_external_metric_heatmap(
        figures_dir / "paper6-nmi-sensitivity",
        external_rows,
        "normalized_mutual_information",
        "PAPER-6 NMI across K and ground-truth threshold",
    )

    manifest = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "reference_profiles": profile_meta,
        "paper6_features": PAPER6_FEATURES,
        "feature_sets": FEATURE_SETS,
        "lofo_feature_sets": LOFO_FEATURE_SETS,
        "combined_reduced_feature_sets": COMBINED_REDUCED_FEATURE_SETS,
        "feature_semantics": {
            "framework_runtime_ms": (
                "ProfilingStartOverheadMs + SnapshotStartOverheadMs; "
                "profiling/framework overhead, not function duration"
            ),
        },
        "scaler": scaler_name,
        "algorithm": "kmeans",
        "distance_metric": "euclidean",
        "k_values": k_values,
        "primary_k": primary_k,
        "ground_truth_thresholds_percent": thresholds,
        "primary_ground_truth_threshold_percent": primary_threshold,
        "n_init": n_init,
        "random_state": random_state,
        "function_count": len(profile_rows),
        "preference_inputs": [metadata for _, metadata in preference_datasets],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cli_invocation": sys.argv,
        "methodological_notes": [
            "K-Means is fitted only on x86 reference-architecture FunctionProfile features.",
            "Ground-truth labels are joined only after clustering; changing the threshold does not refit K-Means.",
            "PCA is used only for visualization and does not alter cluster assignments.",
            "Feature-vs-architecture-delta correlations are post-hoc diagnostics and are not used for feature selection or clustering fit.",
            "LOFO cluster-assignment ARI/NMI measure structural sensitivity to removing one feature independently of ground truth.",
            "Combined reduced sets are evaluated after the systematic LOFO study and are not treated as paper-defined baselines.",
            "Purity is reported together with the majority-class baseline to expose class-imbalance effects.",
            "No automatic best configuration is selected.",
        ],
    }
    write_json(output_dir / "clustering-study-manifest.json", manifest)
    write_markdown_report(
        output_dir,
        experiment_id,
        profile_meta,
        ground_truth_rows,
        pearson_matrix,
        spearman_matrix,
        outlier_rows,
        internal_rows,
        external_rows,
        ablation_rows,
        primary_threshold,
        primary_k,
        scaler_name,
        k_values,
    )

    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run the Serverledge PAPER-6 K-Means baseline, systematic LOFO "
            "feature-ablation study, ground-truth threshold sensitivity, "
            "correlation diagnostics and PCA visualizations."
        )
    )
    ap.add_argument("--profiles", required=True)
    ap.add_argument(
        "--preferences",
        action="append",
        required=True,
        help=(
            "architecture-preference CSV; repeat for thresholds such as "
            "2.5, 5, 10, 15, 20 and 25 percent"
        ),
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument(
        "--scaler",
        choices=preprocess.SCALERS,
        default="standard",
        help="main study scaler; default: standard",
    )
    ap.add_argument(
        "--k",
        action="append",
        type=int,
        default=[],
        help="K-Means K; repeat for a sweep. Default: 2,3,4,5,6,7,8",
    )
    ap.add_argument("--primary-k", type=int, default=3)
    ap.add_argument("--primary-threshold", type=float, default=15.0)
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--annotate-pca", action="store_true")
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
            args.k or [2, 3, 4, 5, 6, 7, 8],
            args.primary_k,
            args.primary_threshold,
            args.n_init,
            args.random_state,
            args.annotate_pca,
        )
    except (StudyError, ValueError, OSError) as exc:
        ap.error(str(exc))

    print(
        f"functions={manifest['function_count']} "
        f"feature_sets={len(manifest['feature_sets'])} "
        f"k_values={len(manifest['k_values'])} "
        f"thresholds={len(manifest['ground_truth_thresholds_percent'])} "
        f"scaler={manifest['scaler']}"
    )
    print(f"output={Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
