#!/usr/bin/env python3
"""
LOFO clustering-as-classification study for the Serverledge thesis.

This experiment is deliberately separate from the historical clustering and
transfer-learning scripts so that old results remain reproducible.

For each target function and each feature-set variant:
  1. remove the target before every fit (scaler, clustering, DBSCAN eps);
  2. fit K-Means or DBSCAN using only the remaining x86 reference profiles;
  3. assign the held-out target to a cluster using only its x86/static features;
  4. predict the target architecture-preference label as the majority label of
     the assigned training cluster;
  5. if the majority is unique, restrict the donor pool to members of that
     majority class and choose the nearest donor;
  6. use ground truth only for post-hoc evaluation.

Primary labels are the existing three-class preference labels:
  - x86-preferred
  - arm-preferred
  - architecture-independent

Feature sets evaluated by default:
  - paper5
  - paper5_no_free_memory
  - paper5_plus_language            (when --language-map is supplied)
  - paper5_no_free_memory_plus_language

Language is categorical: numeric profiling features are scaled, while language
is appended as a one-hot vector.  The one-hot vocabulary is fitted inside each
LOFO training fold; an unseen held-out language is therefore encoded as all
zeros rather than leaking target information into the fit.

Ambiguous majority clusters are currently treated as abstentions.  This is
kept explicit in the outputs because the policy is awaiting supervisor input.

Outputs:
  lofo-per-target.csv
  lofo-classification-summary.csv
  lofo-per-class-metrics.csv
  lofo-confusion-matrix.csv
  full-fit-cluster-summary.csv
  full-fit-clustering-summary.csv
  clustering-lofo-classification-manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import preference


PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

PAPER4_NO_FREE_MEMORY_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

PREFERENCE_LABELS = (
    preference.PREFERENCE_X86,
    preference.PREFERENCE_INDEPENDENT,
    preference.PREFERENCE_ARM,
)

AMBIGUOUS_LABEL = "ambiguous"
ABSTAIN_LABEL = "abstain"

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "paper5": {
        "numeric_features": PAPER5_FEATURES,
        "include_language": False,
    },
    "paper5_no_free_memory": {
        "numeric_features": PAPER4_NO_FREE_MEMORY_FEATURES,
        "include_language": False,
    },
    "paper5_plus_language": {
        "numeric_features": PAPER5_FEATURES,
        "include_language": True,
    },
    "paper5_no_free_memory_plus_language": {
        "numeric_features": PAPER4_NO_FREE_MEMORY_FEATURES,
        "include_language": True,
    },
}

LANGUAGE_ALIASES = {
    "go": "go",
    "golang": "go",
    "python": "python",
    "py": "python",
    "node": "nodejs",
    "nodejs": "nodejs",
    "node.js": "nodejs",
    "javascript": "nodejs",
    "js": "nodejs",
}


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    numeric_features: tuple[str, ...]
    include_language: bool


@dataclass
class PreparedFold:
    train_names: np.ndarray
    train_labels: np.ndarray
    x_train: np.ndarray
    x_target: np.ndarray
    language_vocabulary: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def threshold_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def canonical_language(raw: str) -> str:
    key = raw.strip().lower()
    if not key:
        raise AnalysisError("empty language value")
    return LANGUAGE_ALIASES.get(key, key)


def load_language_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_csv(path)
    if not rows:
        raise AnalysisError(f"language map is empty: {path}")
    required = {"function_name", "language"}
    if not required.issubset(rows[0]):
        raise AnalysisError(
            f"language map must contain columns {sorted(required)}: {path}"
        )
    result: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        name = row["function_name"].strip()
        raw_language = row["language"].strip()
        if not name:
            raise AnalysisError(f"{path}:{row_number}: empty function_name")
        if name in result:
            raise AnalysisError(f"{path}:{row_number}: duplicate function_name {name!r}")
        if raw_language:
            result[name] = canonical_language(raw_language)
    return result


def load_profiles(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    rows = read_csv(path)
    if not rows:
        raise AnalysisError(f"no profiles in {path}")
    required = {"function_name", *PAPER5_FEATURES}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise AnalysisError(f"{path}: missing columns {missing}")

    profiles: dict[str, dict[str, float]] = {}
    for row_number, row in enumerate(rows, start=2):
        name = row["function_name"].strip()
        if not name:
            raise AnalysisError(f"{path}:{row_number}: empty function_name")
        if name in profiles:
            raise AnalysisError(f"{path}:{row_number}: duplicate function {name!r}")
        values: dict[str, float] = {}
        for feature in PAPER5_FEATURES:
            try:
                value = float(row[feature])
            except ValueError as exc:
                raise AnalysisError(
                    f"{path}:{row_number}: {feature} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise AnalysisError(
                    f"{path}:{row_number}: {feature} is not finite"
                )
            values[feature] = value
        profiles[name] = values

    names = sorted(profiles)
    if len(names) < 4:
        raise AnalysisError("at least four functions are required")
    return names, profiles


def load_preferences(preferences_dir: Path, threshold: float) -> dict[str, dict[str, Any]]:
    path = preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
    rows = read_csv(path)
    if not rows:
        raise AnalysisError(f"no preference rows in {path}")
    required = {
        "function_name",
        "architecture_preference",
        "arm_vs_x86_delta_percent",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise AnalysisError(f"{path}: missing columns {missing}")

    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        name = row["function_name"].strip()
        label = row["architecture_preference"].strip()
        if label not in PREFERENCE_LABELS:
            raise AnalysisError(
                f"{path}:{row_number}: unsupported architecture_preference {label!r}"
            )
        if name in result:
            raise AnalysisError(f"{path}:{row_number}: duplicate function {name!r}")
        result[name] = {
            "label": label,
            "delta_percent": float(row["arm_vs_x86_delta_percent"]),
        }
    return result


def selected_feature_specs(language_map: dict[str, str]) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for name, raw in FEATURE_SPECS.items():
        if raw["include_language"] and not language_map:
            continue
        specs.append(
            FeatureSpec(
                name=name,
                numeric_features=tuple(raw["numeric_features"]),
                include_language=bool(raw["include_language"]),
            )
        )
    return specs


def validate_inputs(
    names: list[str],
    preferences: dict[str, dict[str, Any]],
    language_map: dict[str, str],
    specs: Iterable[FeatureSpec],
) -> None:
    expected = set(names)
    if set(preferences) != expected:
        raise AnalysisError(
            "preference/profile function mismatch: "
            f"missing={sorted(expected - set(preferences))}, "
            f"extra={sorted(set(preferences) - expected)}"
        )

    if any(spec.include_language for spec in specs):
        missing = sorted(expected - set(language_map))
        extra = sorted(set(language_map) - expected)
        if missing or extra:
            raise AnalysisError(
                "language/profile function mismatch: "
                f"missing={missing}, extra={extra}. "
                "Fill every language entry before running language feature sets."
            )


def numeric_matrix(
    names: Iterable[str],
    profiles: dict[str, dict[str, float]],
    features: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [[profiles[name][feature] for feature in features] for name in names],
        dtype=float,
    )


def fit_one_hot(train_languages: list[str], target_language: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocabulary = sorted(set(train_languages))
    index = {language: position for position, language in enumerate(vocabulary)}
    train = np.zeros((len(train_languages), len(vocabulary)), dtype=float)
    for row, language in enumerate(train_languages):
        train[row, index[language]] = 1.0
    target = np.zeros((1, len(vocabulary)), dtype=float)
    if target_language in index:
        target[0, index[target_language]] = 1.0
    return train, target, vocabulary


def prepare_fold(
    algorithm: str,
    names: list[str],
    profiles: dict[str, dict[str, float]],
    preferences: dict[str, dict[str, Any]],
    language_map: dict[str, str],
    spec: FeatureSpec,
    target_index: int,
) -> PreparedFold:
    mask = np.arange(len(names)) != target_index
    names_array = np.asarray(names, dtype=object)
    train_names = names_array[mask]
    target_name = names[target_index]

    full_numeric = numeric_matrix(names, profiles, spec.numeric_features)
    x_train_raw = full_numeric[mask]
    x_target_raw = full_numeric[target_index : target_index + 1]

    if algorithm == "kmeans":
        scaler = MinMaxScaler()
    elif algorithm == "dbscan":
        scaler = RobustScaler()
    else:
        raise AnalysisError(f"unsupported algorithm {algorithm!r}")

    x_train = scaler.fit_transform(x_train_raw)
    x_target = scaler.transform(x_target_raw)
    vocabulary: list[str] = []

    if spec.include_language:
        train_languages = [language_map[str(name)] for name in train_names]
        target_language = language_map[target_name]
        train_one_hot, target_one_hot, vocabulary = fit_one_hot(
            train_languages, target_language
        )
        x_train = np.hstack([x_train, train_one_hot])
        x_target = np.hstack([x_target, target_one_hot])

    train_labels = np.asarray(
        [preferences[str(name)]["label"] for name in train_names],
        dtype=object,
    )

    return PreparedFold(
        train_names=train_names,
        train_labels=train_labels,
        x_train=x_train,
        x_target=x_target,
        language_vocabulary=vocabulary,
    )


def prepare_full(
    algorithm: str,
    names: list[str],
    profiles: dict[str, dict[str, float]],
    language_map: dict[str, str],
    spec: FeatureSpec,
) -> tuple[np.ndarray, list[str]]:
    raw = numeric_matrix(names, profiles, spec.numeric_features)
    scaler = MinMaxScaler() if algorithm == "kmeans" else RobustScaler()
    transformed = scaler.fit_transform(raw)
    vocabulary: list[str] = []
    if spec.include_language:
        languages = [language_map[name] for name in names]
        vocabulary = sorted(set(languages))
        index = {language: position for position, language in enumerate(vocabulary)}
        one_hot = np.zeros((len(names), len(vocabulary)), dtype=float)
        for row, language in enumerate(languages):
            one_hot[row, index[language]] = 1.0
        transformed = np.hstack([transformed, one_hot])
    return transformed, vocabulary


def majority_label(labels: Iterable[str]) -> tuple[str, dict[str, int], float]:
    values = list(labels)
    if not values:
        return AMBIGUOUS_LABEL, {label: 0 for label in PREFERENCE_LABELS}, float("nan")
    counts = Counter(values)
    expanded = {label: int(counts.get(label, 0)) for label in PREFERENCE_LABELS}
    maximum = max(expanded.values())
    winners = [label for label in PREFERENCE_LABELS if expanded[label] == maximum]
    majority = winners[0] if len(winners) == 1 else AMBIGUOUS_LABEL
    purity = maximum / len(values)
    return majority, expanded, purity


def rank_distances(target: np.ndarray, candidates: np.ndarray, metric: str) -> np.ndarray:
    if metric == "manhattan":
        return pairwise_distances(target, candidates, metric="manhattan")[0]
    if metric == "euclidean":
        return pairwise_distances(target, candidates, metric="euclidean")[0]
    if metric == "cosine":
        return pairwise_distances(target, candidates, metric="cosine")[0]
    raise AnalysisError(f"unsupported donor ranker {metric!r}")


def choose_class_filtered_donor(
    fold: PreparedFold,
    cluster_members: np.ndarray,
    predicted_label: str,
    donor_ranker: str,
) -> tuple[str, float, int]:
    eligible = np.asarray(
        [index for index in cluster_members if fold.train_labels[index] == predicted_label],
        dtype=int,
    )
    if len(eligible) == 0:
        raise RuntimeError(
            "majority label has no members in its own cluster; this should be impossible"
        )
    distances = rank_distances(fold.x_target, fold.x_train[eligible], donor_ranker)
    candidate_names = fold.train_names[eligible]
    order = np.lexsort((candidate_names, distances))
    chosen_position = int(order[0])
    chosen_index = int(eligible[chosen_position])
    return (
        str(fold.train_names[chosen_index]),
        float(distances[chosen_position]),
        int(len(eligible)),
    )


def dbscan_eps(x_train: np.ndarray, min_samples: int, eps_quantile: float, metric: str) -> float:
    if len(x_train) < min_samples:
        raise AnalysisError(
            f"DBSCAN min_samples={min_samples} exceeds training size {len(x_train)}"
        )
    nn = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_train)
    return float(np.quantile(distances[:, -1], eps_quantile))


def evaluate_target_row(
    algorithm: str,
    spec: FeatureSpec,
    names: list[str],
    profiles: dict[str, dict[str, float]],
    preferences: dict[str, dict[str, Any]],
    language_map: dict[str, str],
    target_index: int,
    donor_ranker: str,
    *,
    k: int,
    n_init: int,
    random_state: int,
    dbscan_min_samples: int,
    dbscan_eps_quantile: float,
    dbscan_metric: str,
) -> dict[str, Any]:
    target = names[target_index]
    target_truth = preferences[target]["label"]
    fold = prepare_fold(
        algorithm,
        names,
        profiles,
        preferences,
        language_map,
        spec,
        target_index,
    )

    base: dict[str, Any] = {
        "algorithm": algorithm,
        "feature_set": spec.name,
        "numeric_features": ",".join(spec.numeric_features),
        "uses_language": spec.include_language,
        "target_function": target,
        "target_ground_truth_label": target_truth,
        "target_architecture_delta_percent": preferences[target]["delta_percent"],
        "target_language": language_map.get(target, ""),
        "training_count": len(fold.train_names),
        "language_vocabulary": ",".join(fold.language_vocabulary),
        "donor_ranker": donor_ranker,
        "eps": "",
        "core_point_count": "",
    }

    if algorithm == "kmeans":
        if k < 2 or k >= len(fold.train_names):
            raise AnalysisError(f"invalid K={k} for training size {len(fold.train_names)}")
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        train_cluster_labels = model.fit_predict(fold.x_train)
        target_cluster = int(model.predict(fold.x_target)[0])
        assignment_distance = float(
            np.linalg.norm(fold.x_target[0] - model.cluster_centers_[target_cluster])
        )
        cluster_members = np.flatnonzero(train_cluster_labels == target_cluster)

    elif algorithm == "dbscan":
        eps = dbscan_eps(
            fold.x_train,
            min_samples=dbscan_min_samples,
            eps_quantile=dbscan_eps_quantile,
            metric=dbscan_metric,
        )
        model = DBSCAN(
            eps=eps,
            min_samples=dbscan_min_samples,
            metric=dbscan_metric,
        )
        train_cluster_labels = model.fit_predict(fold.x_train)
        core_indices = np.asarray(model.core_sample_indices_, dtype=int)
        base["eps"] = eps
        base["core_point_count"] = len(core_indices)

        if len(core_indices) == 0:
            return {
                **base,
                "prediction_status": "abstain",
                "abstention_reason": "no_core_points",
                "target_cluster": -1,
                "target_assignment_distance": "",
                "cluster_size": 0,
                "cluster_x86_count": 0,
                "cluster_independent_count": 0,
                "cluster_arm_count": 0,
                "cluster_majority_label": "",
                "cluster_purity": "",
                "prediction_correct": "",
                "donor_function": "",
                "donor_language": "",
                "donor_distance": "",
                "donor_pool_size": 0,
                "donor_ground_truth_label": "",
                "donor_target_label_agreement": "",
                "abs_architecture_delta_error_percent": "",
            }

        core_matrix = fold.x_train[core_indices]
        core_distances = pairwise_distances(
            fold.x_target, core_matrix, metric=dbscan_metric
        )[0]
        within = np.flatnonzero(core_distances <= eps + 1e-12)
        if len(within) == 0:
            return {
                **base,
                "prediction_status": "abstain",
                "abstention_reason": "no_core_within_eps",
                "target_cluster": -1,
                "target_assignment_distance": float(np.min(core_distances)),
                "cluster_size": 0,
                "cluster_x86_count": 0,
                "cluster_independent_count": 0,
                "cluster_arm_count": 0,
                "cluster_majority_label": "",
                "cluster_purity": "",
                "prediction_correct": "",
                "donor_function": "",
                "donor_language": "",
                "donor_distance": "",
                "donor_pool_size": 0,
                "donor_ground_truth_label": "",
                "donor_target_label_agreement": "",
                "abs_architecture_delta_error_percent": "",
            }

        candidate_core_global = core_indices[within]
        candidate_core_distances = core_distances[within]
        candidate_core_names = fold.train_names[candidate_core_global]
        order = np.lexsort((candidate_core_names, candidate_core_distances))
        nearest_core_global = int(candidate_core_global[int(order[0])])
        target_cluster = int(train_cluster_labels[nearest_core_global])
        assignment_distance = float(candidate_core_distances[int(order[0])])
        cluster_members = np.flatnonzero(train_cluster_labels == target_cluster)

    else:
        raise AnalysisError(f"unsupported algorithm {algorithm!r}")

    member_truth = fold.train_labels[cluster_members]
    majority, counts, purity = majority_label(member_truth)

    common = {
        **base,
        "target_cluster": target_cluster,
        "target_assignment_distance": assignment_distance,
        "cluster_size": len(cluster_members),
        "cluster_x86_count": counts[preference.PREFERENCE_X86],
        "cluster_independent_count": counts[preference.PREFERENCE_INDEPENDENT],
        "cluster_arm_count": counts[preference.PREFERENCE_ARM],
        "cluster_majority_label": majority,
        "cluster_purity": purity,
    }

    if majority == AMBIGUOUS_LABEL:
        return {
            **common,
            "prediction_status": "abstain",
            "abstention_reason": "ambiguous_cluster_majority",
            "prediction_correct": "",
            "donor_function": "",
            "donor_language": "",
            "donor_distance": "",
            "donor_pool_size": 0,
            "donor_ground_truth_label": "",
            "donor_target_label_agreement": "",
            "abs_architecture_delta_error_percent": "",
        }

    donor, donor_distance, donor_pool_size = choose_class_filtered_donor(
        fold,
        cluster_members,
        majority,
        donor_ranker,
    )
    donor_truth = preferences[donor]["label"]

    return {
        **common,
        "prediction_status": "predicted",
        "abstention_reason": "",
        "prediction_correct": majority == target_truth,
        "donor_function": donor,
        "donor_language": language_map.get(donor, ""),
        "donor_distance": donor_distance,
        "donor_pool_size": donor_pool_size,
        "donor_ground_truth_label": donor_truth,
        "donor_target_label_agreement": donor_truth == target_truth,
        "abs_architecture_delta_error_percent": abs(
            float(preferences[target]["delta_percent"])
            - float(preferences[donor]["delta_percent"])
        ),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def class_metrics(
    rows: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    if scope not in {"predicted_only", "all_targets"}:
        raise AnalysisError(f"unsupported metric scope {scope!r}")
    output: list[dict[str, Any]] = []

    for label in PREFERENCE_LABELS:
        considered = (
            [row for row in rows if row["prediction_status"] == "predicted"]
            if scope == "predicted_only"
            else rows
        )
        tp = sum(
            row["target_ground_truth_label"] == label
            and row["cluster_majority_label"] == label
            and row["prediction_status"] == "predicted"
            for row in considered
        )
        fp = sum(
            row["target_ground_truth_label"] != label
            and row["cluster_majority_label"] == label
            and row["prediction_status"] == "predicted"
            for row in considered
        )
        fn = sum(
            row["target_ground_truth_label"] == label
            and not (
                row["cluster_majority_label"] == label
                and row["prediction_status"] == "predicted"
            )
            for row in considered
        )
        support = sum(row["target_ground_truth_label"] == label for row in considered)
        precision_value = safe_div(tp, tp + fp)
        recall_value = safe_div(tp, tp + fn)
        f1_value = (
            safe_div(2.0 * precision_value * recall_value, precision_value + recall_value)
            if math.isfinite(precision_value)
            and math.isfinite(recall_value)
            and precision_value + recall_value > 0
            else 0.0
        )
        output.append(
            {
                "scope": scope,
                "class_label": label,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
            }
        )
    return output


def macro(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_lofo(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    groups = sorted({(row["algorithm"], row["feature_set"]) for row in rows})
    for algorithm, feature_set in groups:
        subset = [
            row
            for row in rows
            if row["algorithm"] == algorithm and row["feature_set"] == feature_set
        ]
        predicted = [row for row in subset if row["prediction_status"] == "predicted"]
        correct = [row for row in predicted if row["prediction_correct"] is True]

        scopes: dict[str, list[dict[str, Any]]] = {}
        for scope in ("predicted_only", "all_targets"):
            metrics = class_metrics(subset, scope)
            scopes[scope] = metrics
            for metric_row in metrics:
                per_class_rows.append(
                    {
                        "algorithm": algorithm,
                        "feature_set": feature_set,
                        **metric_row,
                    }
                )

        predicted_accuracy = safe_div(len(correct), len(predicted))
        strict_accuracy = safe_div(len(correct), len(subset))
        predicted_balanced = macro(row["recall"] for row in scopes["predicted_only"])
        strict_balanced = macro(row["recall"] for row in scopes["all_targets"])
        predicted_macro_f1 = macro(row["f1"] for row in scopes["predicted_only"])
        strict_macro_f1 = macro(row["f1"] for row in scopes["all_targets"])

        reasons = Counter(
            row["abstention_reason"]
            for row in subset
            if row["prediction_status"] != "predicted"
        )

        donor_delta_errors = [
            float(row["abs_architecture_delta_error_percent"])
            for row in predicted
            if row["abs_architecture_delta_error_percent"] not in ("", None)
        ]

        summary_rows.append(
            {
                "algorithm": algorithm,
                "feature_set": feature_set,
                "target_count": len(subset),
                "prediction_count": len(predicted),
                "abstention_count": len(subset) - len(predicted),
                "prediction_coverage": safe_div(len(predicted), len(subset)),
                "accuracy_on_predictions": predicted_accuracy,
                "strict_accuracy_all_targets": strict_accuracy,
                "balanced_accuracy_on_predictions": predicted_balanced,
                "balanced_accuracy_strict": strict_balanced,
                "macro_f1_on_predictions": predicted_macro_f1,
                "macro_f1_strict": strict_macro_f1,
                "ambiguous_majority_count": reasons.get("ambiguous_cluster_majority", 0),
                "no_core_points_count": reasons.get("no_core_points", 0),
                "no_core_within_eps_count": reasons.get("no_core_within_eps", 0),
                "mean_abs_architecture_delta_error_percent": (
                    float(np.mean(donor_delta_errors)) if donor_delta_errors else float("nan")
                ),
                "median_abs_architecture_delta_error_percent": (
                    float(np.median(donor_delta_errors)) if donor_delta_errors else float("nan")
                ),
            }
        )

        predicted_columns = [*PREFERENCE_LABELS, ABSTAIN_LABEL]
        for true_label in PREFERENCE_LABELS:
            row_out: dict[str, Any] = {
                "algorithm": algorithm,
                "feature_set": feature_set,
                "true_label": true_label,
            }
            for predicted_label in predicted_columns:
                row_out[f"pred_{predicted_label}"] = 0
            for row in subset:
                if row["target_ground_truth_label"] != true_label:
                    continue
                predicted_label = (
                    row["cluster_majority_label"]
                    if row["prediction_status"] == "predicted"
                    else ABSTAIN_LABEL
                )
                row_out[f"pred_{predicted_label}"] += 1
            confusion_rows.append(row_out)

    return summary_rows, per_class_rows, confusion_rows


def full_fit_cluster_analysis(
    algorithm: str,
    spec: FeatureSpec,
    names: list[str],
    profiles: dict[str, dict[str, float]],
    preferences: dict[str, dict[str, Any]],
    language_map: dict[str, str],
    *,
    k: int,
    n_init: int,
    random_state: int,
    dbscan_min_samples: int,
    dbscan_eps_quantile: float,
    dbscan_metric: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix, vocabulary = prepare_full(
        algorithm, names, profiles, language_map, spec
    )

    eps: float | str = ""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        cluster_labels = model.fit_predict(matrix)
        clustered_mask = np.ones(len(names), dtype=bool)
    elif algorithm == "dbscan":
        eps = dbscan_eps(
            matrix,
            min_samples=dbscan_min_samples,
            eps_quantile=dbscan_eps_quantile,
            metric=dbscan_metric,
        )
        model = DBSCAN(
            eps=eps,
            min_samples=dbscan_min_samples,
            metric=dbscan_metric,
        )
        cluster_labels = model.fit_predict(matrix)
        clustered_mask = cluster_labels >= 0
    else:
        raise AnalysisError(f"unsupported algorithm {algorithm!r}")

    cluster_rows: list[dict[str, Any]] = []
    majority_correct = 0
    ambiguous_clusters = 0
    valid_cluster_labels = sorted(int(value) for value in set(cluster_labels) if int(value) >= 0)

    for cluster_label in valid_cluster_labels:
        members = np.flatnonzero(cluster_labels == cluster_label)
        labels = [preferences[names[index]]["label"] for index in members]
        majority, counts, purity = majority_label(labels)
        if majority == AMBIGUOUS_LABEL:
            ambiguous_clusters += 1
        majority_correct += max(counts.values()) if counts else 0
        cluster_rows.append(
            {
                "algorithm": algorithm,
                "feature_set": spec.name,
                "cluster_label": cluster_label,
                "cluster_size": len(members),
                "x86_preferred_count": counts[preference.PREFERENCE_X86],
                "architecture_independent_count": counts[preference.PREFERENCE_INDEPENDENT],
                "arm_preferred_count": counts[preference.PREFERENCE_ARM],
                "majority_label": majority,
                "cluster_purity": purity,
                "members": ",".join(names[index] for index in members),
            }
        )

    ground_truth_counts = Counter(preferences[name]["label"] for name in names)
    majority_baseline = max(ground_truth_counts.values()) / len(names)
    clustered_count = int(np.sum(clustered_mask))
    noise_count = len(names) - clustered_count
    clustered_purity = safe_div(majority_correct, clustered_count)
    strict_purity = safe_div(majority_correct, len(names))

    internal_matrix = matrix[clustered_mask]
    internal_labels = cluster_labels[clustered_mask]
    unique_clusters = sorted(set(int(value) for value in internal_labels))
    if len(unique_clusters) >= 2 and len(internal_matrix) > len(unique_clusters):
        silhouette_metric = dbscan_metric if algorithm == "dbscan" else "euclidean"
        silhouette = float(
            silhouette_score(internal_matrix, internal_labels, metric=silhouette_metric)
        )
        # Davies-Bouldin and Calinski-Harabasz are Euclidean-space diagnostics;
        # for DBSCAN they are reported as secondary diagnostics on the transformed
        # coordinates, exactly as descriptive metrics rather than fit objectives.
        davies_bouldin = float(davies_bouldin_score(internal_matrix, internal_labels))
        calinski_harabasz = float(calinski_harabasz_score(internal_matrix, internal_labels))
    else:
        silhouette = float("nan")
        davies_bouldin = float("nan")
        calinski_harabasz = float("nan")

    summary = {
        "algorithm": algorithm,
        "feature_set": spec.name,
        "target_count": len(names),
        "cluster_count": len(valid_cluster_labels),
        "clustered_count": clustered_count,
        "noise_count": noise_count,
        "cluster_coverage": safe_div(clustered_count, len(names)),
        "ambiguous_cluster_count": ambiguous_clusters,
        "majority_ground_truth_share": majority_baseline,
        "purity_clustered_only": clustered_purity,
        "purity_strict_all_targets": strict_purity,
        "purity_gain_over_majority_baseline_clustered": clustered_purity - majority_baseline,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "language_vocabulary": ",".join(vocabulary),
        "eps": eps,
    }
    return cluster_rows, summary


def run_analysis(
    profiles_path: Path,
    preferences_dir: Path,
    output_dir: Path,
    *,
    threshold: float = 15.0,
    language_map_path: Path | None = None,
    donor_ranker: str = "manhattan",
    k: int = 5,
    n_init: int = 50,
    random_state: int = 42,
    dbscan_min_samples: int = 4,
    dbscan_eps_quantile: float = 0.80,
    dbscan_metric: str = "cosine",
) -> dict[str, Any]:
    profiles_path = profiles_path.expanduser().resolve()
    preferences_dir = preferences_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    language_map_path = (
        language_map_path.expanduser().resolve() if language_map_path is not None else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    names, profiles = load_profiles(profiles_path)
    preferences = load_preferences(preferences_dir, threshold)
    language_map = load_language_map(language_map_path)
    specs = selected_feature_specs(language_map)
    validate_inputs(names, preferences, language_map, specs)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for algorithm in ("kmeans", "dbscan"):
            for target_index in range(len(names)):
                rows.append(
                    evaluate_target_row(
                        algorithm,
                        spec,
                        names,
                        profiles,
                        preferences,
                        language_map,
                        target_index,
                        donor_ranker,
                        k=k,
                        n_init=n_init,
                        random_state=random_state,
                        dbscan_min_samples=dbscan_min_samples,
                        dbscan_eps_quantile=dbscan_eps_quantile,
                        dbscan_metric=dbscan_metric,
                    )
                )

    rows.sort(key=lambda row: (row["feature_set"], row["algorithm"], row["target_function"]))
    summary_rows, per_class_rows, confusion_rows = summarize_lofo(rows)

    full_cluster_rows: list[dict[str, Any]] = []
    full_summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        for algorithm in ("kmeans", "dbscan"):
            cluster_rows, summary = full_fit_cluster_analysis(
                algorithm,
                spec,
                names,
                profiles,
                preferences,
                language_map,
                k=k,
                n_init=n_init,
                random_state=random_state,
                dbscan_min_samples=dbscan_min_samples,
                dbscan_eps_quantile=dbscan_eps_quantile,
                dbscan_metric=dbscan_metric,
            )
            full_cluster_rows.extend(cluster_rows)
            full_summary_rows.append(summary)

    write_csv(output_dir / "lofo-per-target.csv", rows)
    write_csv(output_dir / "lofo-classification-summary.csv", summary_rows)
    write_csv(output_dir / "lofo-per-class-metrics.csv", per_class_rows)
    write_csv(output_dir / "lofo-confusion-matrix.csv", confusion_rows)
    write_csv(output_dir / "full-fit-cluster-summary.csv", full_cluster_rows)
    write_csv(output_dir / "full-fit-clustering-summary.csv", full_summary_rows)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "LOFO clustering as three-class architecture-preference classification",
        "target_count": len(names),
        "primary_ground_truth_threshold_percent": threshold,
        "preference_labels": list(PREFERENCE_LABELS),
        "ambiguous_majority_policy": "abstain",
        "donor_rule": (
            "predict target label as unique majority class of assigned training cluster; "
            "restrict donor pool to that class; rank remaining donors by configured distance"
        ),
        "donor_ranker": donor_ranker,
        "feature_sets": {
            spec.name: {
                "numeric_features": list(spec.numeric_features),
                "include_language": spec.include_language,
            }
            for spec in specs
        },
        "language_encoding": (
            "numeric features scaled per algorithm; language one-hot fitted training-only in each LOFO fold; "
            "one-hot columns not rescaled"
            if language_map
            else "language feature sets skipped because no --language-map was supplied"
        ),
        "kmeans": {
            "scaler": "MinMaxScaler",
            "k": k,
            "n_init": n_init,
            "random_state": random_state,
            "target_assignment": "KMeans.predict / Euclidean centroid geometry",
        },
        "dbscan": {
            "scaler": "RobustScaler",
            "metric": dbscan_metric,
            "min_samples": dbscan_min_samples,
            "eps_quantile": dbscan_eps_quantile,
            "eps_policy": "recomputed training-only in every LOFO fold from k-distance quantile",
            "target_assignment": "nearest core point within eps; otherwise abstain",
        },
        "leakage_control": (
            "held-out target excluded before numeric scaler fit, language vocabulary fit, clustering fit, "
            "DBSCAN eps calibration, majority-label computation and donor selection; target ground truth "
            "used only after prediction"
        ),
        "inputs": {
            "profiles": str(profiles_path),
            "profiles_sha256": sha256_file(profiles_path),
            "preferences_dir": str(preferences_dir),
            "preference_csv": str(
                preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
            ),
            "preference_csv_sha256": sha256_file(
                preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
            ),
            "language_map": str(language_map_path) if language_map_path else "",
            "language_map_sha256": (
                sha256_file(language_map_path) if language_map_path else ""
            ),
        },
        "outputs": [
            "lofo-per-target.csv",
            "lofo-classification-summary.csv",
            "lofo-per-class-metrics.csv",
            "lofo-confusion-matrix.csv",
            "full-fit-cluster-summary.csv",
            "full-fit-clustering-summary.csv",
        ],
    }
    write_json(output_dir / "clustering-lofo-classification-manifest.json", manifest)

    print(
        f"targets={len(names)} feature_sets={len(specs)} rows={len(rows)} output={output_dir}"
    )
    for row in summary_rows:
        print(
            f"{row['feature_set']:<42} {row['algorithm']:<7} "
            f"coverage={float(row['prediction_coverage']):.4f} "
            f"acc={float(row['accuracy_on_predictions']):.4f} "
            f"strict_acc={float(row['strict_accuracy_all_targets']):.4f} "
            f"bal_acc={float(row['balanced_accuracy_strict']):.4f} "
            f"macro_f1={float(row['macro_f1_strict']):.4f}"
        )

    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument(
        "--language-map",
        type=Path,
        help=(
            "CSV with function_name,language. If omitted, the two language feature sets are skipped."
        ),
    )
    p.add_argument(
        "--donor-ranker",
        choices=("manhattan", "euclidean", "cosine"),
        default="manhattan",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dbscan-min-samples", type=int, default=4)
    p.add_argument("--dbscan-eps-quantile", type=float, default=0.80)
    p.add_argument("--dbscan-metric", default="cosine")
    return p


def main() -> int:
    args = parser().parse_args()
    run_analysis(
        args.profiles,
        args.preferences_dir,
        args.output_dir,
        threshold=args.threshold,
        language_map_path=args.language_map,
        donor_ranker=args.donor_ranker,
        k=args.k,
        n_init=args.n_init,
        random_state=args.random_state,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_eps_quantile=args.dbscan_eps_quantile,
        dbscan_metric=args.dbscan_metric,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
