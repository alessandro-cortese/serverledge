"""Supervised classification of the architecture preference.

The clustering pipeline is unsupervised and groups FunctionProfiles by resource
profile. It answers "which functions look alike", not "which architecture suits
this function". This module answers the second question directly, replicating
the approach of the reference paper:

    - one training example per invocation, not per aggregated profile;
    - a Random Forest over the six features of Table V;
    - prediction per function resolved by majority vote over its samples;
    - leave-one-function-out validation, because the question is how the model
      behaves on a function it has never seen.

The labels are the three classes already produced by preference.py:

    x86-preferred
    arm-preferred
    architecture-independent

Input:

    --samples      per-invocation CSV produced by
                   `serverledge-profiling export-samples-csv`
    --preferences  architecture preference CSV produced by
                   `preference.py preferences`
    --model        preprocessing model JSON produced by
                   `preprocess.py fit-transform`

Output: a JSON report with per-sample and per-function accuracy, the confusion
matrix over the three classes, and permutation feature importance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from analysis.profiling import preprocess, preference

CLASSIFICATION_REPORT_SCHEMA_VERSION = 1


SAMPLE_METADATA = [
    "sample_csv_schema_version",
    "experiment_id",
    "feature_vector_schema_version",
    "request_id",
    "function_name",
    "machine_tag",
    "configured_cpus",
    "configured_memory_mb",
]


SAMPLE_HEADER = SAMPLE_METADATA + preprocess.FEATURE_NAMES


PREFERENCE_CLASSES = [
    preference.PREFERENCE_X86,
    preference.PREFERENCE_ARM,
    preference.PREFERENCE_INDEPENDENT,
]


def parse_finite(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: " f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: " f"{field} is not finite")

    return result


def load_samples(path: Path, reference_machine_tag: str) -> list[dict]:
    """Load the per-invocation CSV, keeping only the reference architecture.

    The six features are absolute container-scoped measurements: mixing rows
    measured on different architectures would make the model learn where the
    profiling ran instead of how the function behaves.
    """
    rows = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != SAMPLE_HEADER:
            raise ValueError("unexpected sample CSV header")

        for row_number, row in enumerate(reader, start=2):
            if row["machine_tag"].strip() != reference_machine_tag:
                continue

            features = [parse_finite(row[name], name, row_number) for name in preprocess.FEATURE_NAMES]

            rows.append(
                {
                    "request_id": row["request_id"].strip(),
                    "function_name": row["function_name"].strip(),
                    "configured_cpus": parse_finite(
                        row["configured_cpus"], "configured_cpus", row_number
                    ),
                    "configured_memory_mb": int(row["configured_memory_mb"]),
                    "features": features,
                }
            )

    if not rows:
        raise ValueError(
            "no sample matches the " f"reference machine tag " f"{reference_machine_tag!r}"
        )

    return rows


def load_labels(path: Path) -> dict[tuple, str]:
    """Load the three-class label per (function, cpus, memory)."""
    labels = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != preference.ARCHITECTURE_PREFERENCE_HEADER:
            raise ValueError("unexpected architecture " "preference CSV header")

        for row_number, row in enumerate(reader, start=2):
            label = row["architecture_preference"].strip()

            if label not in PREFERENCE_CLASSES:
                raise ValueError(f"row {row_number}: " f"unknown preference {label!r}")

            key = (
                row["function_name"].strip(),
                parse_finite(row["configured_cpus"], "configured_cpus", row_number),
                int(row["configured_memory_mb"]),
            )

            if key in labels:
                raise ValueError(f"row {row_number}: " "duplicate preference " f"for {key}")

            labels[key] = label

    if not labels:
        raise ValueError("architecture preference " "dataset is empty")

    return labels


def join(samples: list[dict], labels: dict[tuple, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair every sample with the label of its function-configuration.

    Samples whose function-configuration has no measured preference are
    dropped: without observations on both architectures there is no ground
    truth to learn from.
    """
    matrix = []
    targets = []
    groups = []

    for sample in samples:
        key = (sample["function_name"], sample["configured_cpus"], sample["configured_memory_mb"])

        if key not in labels:
            continue

        matrix.append(sample["features"])
        targets.append(labels[key])
        groups.append(key)

    if not matrix:
        raise ValueError("no sample has a matching " "architecture preference label")

    return np.asarray(matrix, dtype=np.float64),np.asarray(targets, dtype=object),np.asarray(groups, dtype=object)


def scale(matrix: np.ndarray, model: dict, aggregation: str) -> np.ndarray:
    """Apply the same affine transform used by clustering and donor selection.

    Reusing the fitted model keeps the classifier in the same feature space as
    the rest of the pipeline, so that a single preprocessing decision governs
    every downstream consumer.
    """
    return preprocess.apply_model(matrix, model, aggregation)


def build_classifier(trees: int, random_state: int) -> RandomForestClassifier:
    """Random Forest, as in the reference paper.

    The paper compares seven classifiers and reports Random Forest as the most
    accurate, ahead of Decision Tree and Gaussian Process; there is no reason to
    depart from that choice.

    class_weight is balanced because the three preference classes are rarely
    equally populated: with an unbalanced benchmark the model would otherwise
    learn to predict the majority class.
    """
    return RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=random_state)


def leave_one_function_out(matrix: np.ndarray, targets: np.ndarray, groups: np.ndarray, trees: int, random_state: int) -> dict:
    """Validate by holding out one whole function-configuration at a time.

    Splitting by sample would place invocations of the same function in both
    the training and the test set, and the reported accuracy would be
    meaningless: the question is how the model behaves on a function it has
    never seen.
    """
    unique_groups = sorted({tuple(group) for group in groups})

    if len(unique_groups) < 3:
        raise ValueError(
            "at least three distinct " "function-configurations are " "required to validate"
        )

    group_tuples = [tuple(group) for group in groups]

    sample_correct = 0
    sample_total = 0

    function_results = []

    confusion = {
        actual: {predicted: 0 for predicted in PREFERENCE_CLASSES} for actual in PREFERENCE_CLASSES
    }

    for held_out in unique_groups:
        test_index = [index for index, group in enumerate(group_tuples) if group == held_out]
        train_index = [index for index, group in enumerate(group_tuples) if group != held_out]
        training_labels = set(targets[train_index].tolist())

        if len(training_labels) < 2:
            # Every remaining function shares one label: a classifier trained
            # here would be constant, and scoring it would be misleading.
            function_results.append(
                {
                    "function_name": held_out[0],
                    "configured_cpus": held_out[1],
                    "configured_memory_mb": held_out[2],
                    "skipped": "single_class_training_set",
                }
            )

            continue

        model = build_classifier(trees, random_state)
        model.fit(matrix[train_index], targets[train_index])
        predictions = model.predict(matrix[test_index])
        actual = targets[test_index][0]
        correct = int((predictions == actual).sum())
        sample_correct += correct
        sample_total += len(test_index)

        # Majority vote, as the paper does over ten runs.
        vote = Counter(predictions.tolist()).most_common(1)[0][0]
        confusion[actual][vote] += 1
        function_results.append(
            {
                "function_name": held_out[0],
                "configured_cpus": held_out[1],
                "configured_memory_mb": held_out[2],
                "actual": actual,
                "majority_vote": vote,
                "correct": vote == actual,
                "sample_count": len(test_index),
                "sample_accuracy": correct / len(test_index),
            }
        )

    evaluated = [result for result in function_results if "skipped" not in result]

    if not evaluated:
        raise ValueError("no function-configuration " "could be evaluated")

    return {
        "sample_accuracy": sample_correct / sample_total,
        "sample_count": sample_total,
        "function_accuracy": sum(1 for result in evaluated if result["correct"]) / len(evaluated),
        "function_count": len(evaluated),
        "skipped_count": len(function_results) - len(evaluated),
        "confusion_matrix": confusion,
        "per_function": function_results,
    }


def feature_importance(matrix: np.ndarray, targets: np.ndarray, trees: int, random_state: int) -> list[dict]:
    """Permutation importance, the same measure reported in the paper."""
    model = build_classifier(trees, random_state)
    model.fit(matrix, targets)
    result = permutation_importance(model, matrix, targets, n_repeats=10, random_state=random_state)

    return [
        {
            "feature": name,
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index, name in enumerate(preprocess.FEATURE_NAMES)
    ]


def run(samples_path: Path,preferences_path: Path,model_path: Path,reference_machine_tag: str,aggregation: str,trees: int,random_state: int,output_path: Path) -> None:
    model = preprocess.load_model(model_path)

    if list(model["feature_names"]) != list(preprocess.FEATURE_NAMES):
        raise ValueError("preprocessing model feature " "names or order are invalid")

    samples = load_samples(samples_path, reference_machine_tag)
    labels = load_labels(preferences_path)
    matrix, targets, groups = join(samples, labels)
    scaled = scale(matrix, model, aggregation)

    report = {
        "classification_report_schema_version": CLASSIFICATION_REPORT_SCHEMA_VERSION,
        "reference_machine_tag": reference_machine_tag,
        "aggregation": aggregation,
        "scaler": model["scaler"],
        "feature_names": list(preprocess.FEATURE_NAMES),
        "class_distribution": dict(Counter(targets.tolist())),
        "validation": leave_one_function_out(scaled, targets, groups, trees, random_state),
        "feature_importance": feature_importance(scaled, targets, trees, random_state),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = report["validation"]

    print(
        "samples="
        f"{validation['sample_count']} "
        "functions="
        f"{validation['function_count']} "
        "skipped="
        f"{validation['skipped_count']}"
    )

    print(
        "sample_accuracy="
        f"{validation['sample_accuracy']:.4f} "
        "function_accuracy="
        f"{validation['function_accuracy']:.4f}"
    )

    print(f"output={output_path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Train and validate the "
            "supervised classification of "
            "the architecture preference "
            "from per-invocation profiling "
            "features."
        )
    )

    root.add_argument("--samples", required=True)
    root.add_argument("--preferences", required=True)
    root.add_argument("--model", required=True)
    root.add_argument("--reference-machine-tag", required=True)
    root.add_argument("--aggregation", required=True, choices=["mean", "median"])
    root.add_argument("--trees", type=int, default=100)
    root.add_argument("--random-state", type=int, default=0)
    root.add_argument("--output", required=True)

    return root


def main() -> None:
    arguments = parser().parse_args()

    run(
        Path(arguments.samples),
        Path(arguments.preferences),
        Path(arguments.model),
        arguments.reference_machine_tag.strip(),
        arguments.aggregation,
        arguments.trees,
        arguments.random_state,
        Path(arguments.output),
    )


if __name__ == "__main__":
    main()
