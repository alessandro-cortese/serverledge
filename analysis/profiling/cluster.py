#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# Allow both:
#
# python analysis/profiling/cluster.py ...
#
# and:
#
# python -m analysis.profiling.cluster ...
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import preprocess

CLUSTERING_MODEL_SCHEMA_VERSION = 1

CLUSTERING_CSV_SCHEMA_VERSION = 1

FEATURE_NAMES = preprocess.FEATURE_NAMES

INPUT_HEADER = preprocess.OUTPUT_HEADER

ASSIGNMENT_METADATA = [
    "clustering_csv_schema_version",
    "clustering_model_schema_version",
    "clustering_run_id",
    "preprocessing_schema_version",
    "preprocessing_model_schema_version",
    "source_csv_schema_version",
    "experiment_id",
    "fit_experiment_id",
    "aggregation",
    "scaler",
    "algorithm",
    "function_profile_schema_version",
    "function_name",
    "machine_tag",
    "configured_cpus",
    "configured_memory_mb",
    "sample_count",
    "cluster_label",
    "is_noise",
]


ASSIGNMENT_HEADER = ASSIGNMENT_METADATA + FEATURE_NAMES


def parse_finite(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)

    except ValueError as exc:
        raise ValueError(f"row {row_number}: " f"{field} is not numeric") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: " f"{field} is not finite")

    return parsed


def load_preprocessed_dataset(path: Path) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != INPUT_HEADER:
            raise ValueError("unexpected preprocessed " "CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("preprocessed CSV " "contains no data rows")

    experiment_ids: set[str] = set()
    fit_experiment_ids: set[str] = set()
    aggregations: set[str] = set()
    scalers: set[str] = set()
    identities: set[tuple[str, str, float, int]] = set()
    matrix: list[list[float]] = []

    for row_number, row in enumerate(rows, start=2):
        if row["preprocessing_schema_version"] != str(preprocess.PREPROCESSED_CSV_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported preprocessing " "CSV schema")

        if row["preprocessing_model_schema_version"] != str(preprocess.MODEL_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported preprocessing " "model schema")

        if row["source_csv_schema_version"] != str(preprocess.SOURCE_CSV_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported source CSV schema")

        if row["function_profile_schema_version"] != str(
            preprocess.FUNCTION_PROFILE_SCHEMA_VERSION
        ):
            raise ValueError(f"row {row_number}: " "unsupported " "FunctionProfile schema")

        experiment_id = row["experiment_id"].strip()
        fit_experiment_id = row["fit_experiment_id"].strip()
        aggregation = row["aggregation"].strip()
        scaler = row["scaler"].strip()
        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()

        if not experiment_id or not fit_experiment_id or not function_name or not machine_tag:
            raise ValueError(f"row {row_number}: " "empty metadata field")

        if aggregation not in ("mean", "median"):
            raise ValueError(f"row {row_number}: " "invalid aggregation " f"{aggregation!r}")

        if scaler not in (preprocess.SCALERS):
            raise ValueError(f"row {row_number}: " "invalid scaler " f"{scaler!r}")

        cpus = parse_finite(row["configured_cpus"], "configured_cpus", row_number)

        try:
            memory_mb = int(row["configured_memory_mb"])
            sample_count = int(row["sample_count"])

        except ValueError as exc:
            raise ValueError(f"row {row_number}: " "invalid integer metadata") from exc

        if cpus <= 0 or memory_mb <= 0 or sample_count <= 0:
            raise ValueError(f"row {row_number}: " "invalid resource metadata")

        identity = (function_name, machine_tag, cpus, memory_mb)
        if identity in identities:
            raise ValueError(
                f"row {row_number}: " "duplicate FunctionProfile " f"identity {identity}"
            )

        identities.add(identity)
        values = [parse_finite(row[feature], feature, row_number) for feature in FEATURE_NAMES]

        # Scaled values may legitimately be negative.
        matrix.append(values)
        experiment_ids.add(experiment_id)
        fit_experiment_ids.add(fit_experiment_id)
        aggregations.add(aggregation)
        scalers.add(scaler)

    if len(experiment_ids) != 1:
        raise ValueError("input mixes experiment IDs")

    if len(fit_experiment_ids) != 1:
        raise ValueError("input mixes preprocessing " "fit experiment IDs")

    if len(aggregations) != 1:
        raise ValueError("input mixes mean and median " "datasets")

    if len(scalers) != 1:
        raise ValueError("input mixes preprocessing " "scalers")

    metadata = {
        "experiment_id": experiment_ids.pop(),
        "fit_experiment_id": fit_experiment_ids.pop(),
        "aggregation": aggregations.pop(),
        "scaler": scalers.pop(),
    }

    return rows, np.asarray(matrix, dtype=np.float64), metadata


def cluster_sizes(labels: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}

    for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        result[str(label)] = int(np.sum(labels == label))

    return result


def silhouette_if_defined(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = set(int(value) for value in labels)
    number_of_labels = len(unique_labels)
    number_of_samples = matrix.shape[0]

    if not (2 <= number_of_labels <= number_of_samples - 1):
        return None

    return float(silhouette_score(matrix, labels, metric="euclidean"))


def dbscan_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    # Noise is deliberately excluded.
    #
    # Treating DBSCAN noise (-1) as if it were a normal cluster
    # would give a misleading comparison with K-Means.
    clustered_mask = labels != -1
    clustered_matrix = matrix[clustered_mask]
    clustered_labels = labels[clustered_mask]

    if clustered_matrix.shape[0] == 0:
        return None

    return silhouette_if_defined(clustered_matrix, clustered_labels)


def fit_kmeans(matrix: np.ndarray, clusters: int, n_init: int, random_state: int) -> tuple[np.ndarray, dict, dict]:
    if clusters <= 0:
        raise ValueError("K-Means clusters must " "be positive")

    if clusters > (matrix.shape[0]):
        raise ValueError("K-Means clusters cannot " "exceed the number of samples")

    if n_init <= 0:
        raise ValueError("K-Means n_init must " "be positive")

    model = KMeans(
        n_clusters=clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd",
    )

    labels = model.fit_predict(matrix)
    silhouette = silhouette_if_defined(matrix, labels)

    parameters = {
        "clusters": clusters,
        "init": "k-means++",
        "n_init": n_init,
        "random_state": random_state,
        "algorithm": "lloyd",
        "metric": "euclidean",
    }

    result = {
        "cluster_count": clusters,
        "noise_count": 0,
        "clustered_sample_count": int(matrix.shape[0]),
        "coverage": 1.0,
        "cluster_sizes": cluster_sizes(labels),
        "silhouette_score": silhouette,
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
        "centroids": [[float(value) for value in center] for center in model.cluster_centers_],
    }

    return labels.astype(np.int64), parameters, result


def fit_dbscan(matrix: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, dict, dict]:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("DBSCAN eps must be " "finite and positive")

    if min_samples <= 0:
        raise ValueError("DBSCAN min_samples " "must be positive")

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = (model.fit_predict(matrix)).astype(np.int64)
    cluster_labels = sorted(set(int(value) for value in labels if int(value) >= 0))
    noise_count = int(np.sum(labels == -1))
    clustered_sample_count = int(matrix.shape[0]) - noise_count
    coverage = clustered_sample_count / matrix.shape[0]
    silhouette = dbscan_silhouette(matrix, labels)
    core_samples = []

    for index in model.core_sample_indices_:
        index = int(index)
        core_samples.append(
            {
                "sample_index": index,
                "cluster_label": int(labels[index]),
                "coordinates": [float(value) for value in matrix[index]],
            }
        )

    parameters = {"eps": float(eps), "min_samples": int(min_samples), "metric": "euclidean"}

    result = {
        "cluster_count": len(cluster_labels),
        "noise_count": noise_count,
        "clustered_sample_count": clustered_sample_count,
        "coverage": float(coverage),
        "cluster_sizes": cluster_sizes(labels),
        # Deliberately computed only on non-noise samples.
        "silhouette_clustered_only": silhouette,
        # DBSCAN has no K-Means-like inertia.
        "core_sample_count": len(core_samples),
        "core_samples": core_samples,
    }

    return labels, parameters, result


def predict_kmeans_from_artifact(matrix: np.ndarray, model: dict) -> np.ndarray:
    if model.get("algorithm") != "kmeans":
        raise ValueError("artifact is not " "a K-Means model")

    centroids = np.asarray(model["result"]["centroids"], dtype=np.float64)

    if centroids.ndim != 2 or centroids.shape[1] != len(FEATURE_NAMES):
        raise ValueError("invalid K-Means centroids")

    distances = np.linalg.norm(matrix[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

    return np.argmin(distances, axis=1).astype(np.int64)


def build_artifact(
    input_path: Path,
    run_id: str,
    metadata: dict,
    algorithm: str,
    sample_count: int,
    parameters: dict,
    result: dict,
) -> dict:
    run_id = run_id.strip()

    if not run_id:
        raise ValueError("clustering run ID " "cannot be empty")

    return {
        "schema_version": CLUSTERING_MODEL_SCHEMA_VERSION,
        "clustering_csv_schema_version": CLUSTERING_CSV_SCHEMA_VERSION,
        "clustering_run_id": run_id,
        "algorithm": algorithm,
        "sklearn_version": sklearn.__version__,
        "feature_names": FEATURE_NAMES,
        "fit_sample_count": int(sample_count),
        "input_sha256": preprocess.sha256_file(input_path),
        "preprocessing": {
            "preprocessing_schema_version": preprocess.PREPROCESSED_CSV_SCHEMA_VERSION,
            "preprocessing_model_schema_version": preprocess.MODEL_SCHEMA_VERSION,
            "source_csv_schema_version": preprocess.SOURCE_CSV_SCHEMA_VERSION,
            "experiment_id": metadata["experiment_id"],
            "fit_experiment_id": metadata["fit_experiment_id"],
            "aggregation": metadata["aggregation"],
            "scaler": metadata["scaler"],
        },
        "parameters": parameters,
        "result": result,
    }


def atomic_json(path: Path, data: dict) -> None:
    preprocess.atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_assignments(path: Path, rows: list[dict[str, str]], matrix: np.ndarray, labels: np.ndarray, artifact: dict) -> None:
    if len(rows) != matrix.shape[0] or len(rows) != labels.shape[0]:
        raise ValueError("assignment dimensions " "do not match")

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=(f".{path.name}."), suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:

            writer = csv.writer(handle)
            writer.writerow(ASSIGNMENT_HEADER)

            for row, values, label in zip(rows, matrix, labels, strict=True):
                label = int(label)

                writer.writerow(
                    [
                        CLUSTERING_CSV_SCHEMA_VERSION,
                        CLUSTERING_MODEL_SCHEMA_VERSION,
                        artifact["clustering_run_id"],
                        row["preprocessing_schema_version"],
                        row["preprocessing_model_schema_version"],
                        row["source_csv_schema_version"],
                        row["experiment_id"],
                        row["fit_experiment_id"],
                        row["aggregation"],
                        row["scaler"],
                        artifact["algorithm"],
                        row["function_profile_schema_version"],
                        row["function_name"],
                        row["machine_tag"],
                        row["configured_cpus"],
                        row["configured_memory_mb"],
                        row["sample_count"],
                        label,
                        str(label == -1).lower(),
                        *[format(float(value), ".17g") for value in values],
                    ]
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_kmeans(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    rows, matrix, metadata = load_preprocessed_dataset(input_path)
    labels, parameters, result = fit_kmeans(matrix, args.clusters, args.n_init, args.random_state)
    artifact = build_artifact(input_path, args.run_id, metadata, "kmeans", len(rows), parameters, result)

    # Verify that the portable centroid representation reproduces
    # sklearn's assignments on the fitted reference dataset.
    replay_labels = predict_kmeans_from_artifact(matrix, artifact)

    if not np.array_equal(labels, replay_labels):
        raise RuntimeError(
            "serialized K-Means " "centroids do not reproduce " "reference assignments"
        )

    atomic_json(Path(args.model), artifact)
    write_assignments(Path(args.output), rows, matrix, labels, artifact)
    silhouette = result["silhouette_score"]

    print(
        f"rows={len(rows)} "
        f"features={len(FEATURE_NAMES)} "
        f"algorithm=kmeans "
        f"clusters={result['cluster_count']} "
        f"aggregation={metadata['aggregation']} "
        f"scaler={metadata['scaler']}"
    )

    print(f"inertia={result['inertia']}")
    print("silhouette=" + ("undefined" if silhouette is None else str(silhouette)))
    print("input_sha256=" f"{artifact['input_sha256']}")
    print(f"model={args.model}")
    print(f"output={args.output}")

def run_dbscan(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    rows, matrix, metadata = load_preprocessed_dataset(input_path)
    labels, parameters, result = fit_dbscan(matrix, args.eps, args.min_samples)
    artifact = build_artifact(input_path, args.run_id, metadata, "dbscan", len(rows), parameters, result)
    atomic_json(Path(args.model), artifact)
    write_assignments(Path(args.output), rows, matrix, labels, artifact)
    silhouette = result["silhouette_clustered_only"]

    print(
        f"rows={len(rows)} "
        f"features={len(FEATURE_NAMES)} "
        f"algorithm=dbscan "
        f"clusters={result['cluster_count']} "
        f"noise={result['noise_count']} "
        f"coverage={result['coverage']} "
        f"aggregation={metadata['aggregation']} "
        f"scaler={metadata['scaler']}"
    )

    print("silhouette_clustered_only=" + ("undefined" if silhouette is None else str(silhouette)))
    print("input_sha256=" f"{artifact['input_sha256']}")
    print(f"model={args.model}")
    print(f"output={args.output}")

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=("Cluster preprocessed " "Serverledge FunctionProfile " "datasets.")
    )

    commands = root.add_subparsers(dest="algorithm", required=True)
    kmeans = commands.add_parser("kmeans", help=("fit K-Means to one " "preprocessed dataset"))
    kmeans.add_argument("--input", required=True)
    kmeans.add_argument("--run-id", required=True)
    kmeans.add_argument("--clusters", type=int, required=True)
    kmeans.add_argument("--n-init", type=int, default=10)
    kmeans.add_argument("--random-state", type=int, default=42)
    kmeans.add_argument("--model", required=True)
    kmeans.add_argument("--output", required=True)
    kmeans.set_defaults(func=run_kmeans)
    dbscan = commands.add_parser("dbscan", help=("fit DBSCAN to one " "preprocessed dataset"))
    dbscan.add_argument("--input", required=True)
    dbscan.add_argument("--run-id", required=True)
    dbscan.add_argument("--eps", type=float, required=True)
    dbscan.add_argument("--min-samples", type=int, required=True)
    dbscan.add_argument("--model", required=True)
    dbscan.add_argument("--output", required=True)
    dbscan.set_defaults(func=run_dbscan)

    return root


def main() -> None:
    root = parser()

    args = root.parse_args()

    try:
        args.func(args)

    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
