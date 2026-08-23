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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from analysis.profiling import (
    cluster,
    evaluate_clustering,
    preference,
    prepare_reference,
    preprocess,
)

SWEEP_MANIFEST_SCHEMA_VERSION = 1
SWEEP_CSV_SCHEMA_VERSION = 1

SWEEP_SUMMARY_HEADER = [
    "sweep_csv_schema_version",
    "sweep_run_id",
    "reference_run_id",
    "preference_run_id",
    "reference_machine_tag",
    "aggregation",
    "threshold_percent",
    "scaler",
    "algorithm",
    "configuration_id",
    "status",
    "reason",
    "k",
    "eps",
    "min_samples",
    "n_init",
    "random_state",
    "sample_count",
    "cluster_count",
    "noise_count",
    "coverage",
    "silhouette",
    "inertia",
    "matched_count",
    "assignment_match_coverage",
    "preference_match_coverage",
    "overall_purity",
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "external_metrics_defined",
    "run_dir",
]

def unique_ints(values: list[int], name: str, minimum: int) -> list[int]:
    result = []
    seen = set()

    for value in values:
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")

        if value in seen:
            raise ValueError(f"duplicate {name} value " f"{value}")

        seen.add(value)
        result.append(value)

    return result


def unique_floats(values: list[float], name: str) -> list[float]:
    result = []
    seen = set()

    for value in values:
        value = float(value)

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite " "and positive")

        if value in seen:
            raise ValueError(f"duplicate {name} value " f"{value}")

        seen.add(value)
        result.append(value)

    return result


def selected_scalers(requested: list[str], available: list[str]) -> list[str]:
    if not requested:
        return list(available)

    result = []
    seen = set()

    for scaler in requested:
        if scaler not in available:
            raise ValueError(f"scaler {scaler!r} " "is not available in " "the reference manifest")

        if scaler in seen:
            raise ValueError(f"duplicate scaler " f"{scaler!r}")

        seen.add(scaler)
        result.append(scaler)

    return result


def selected_algorithms(requested: list[str]) -> list[str]:
    if not requested:
        return ["kmeans", "dbscan"]

    result = []
    seen = set()

    for algorithm in requested:
        if algorithm not in ("kmeans", "dbscan"):
            raise ValueError("unsupported algorithm " f"{algorithm!r}")

        if algorithm in seen:
            raise ValueError("duplicate algorithm " f"{algorithm!r}")

        seen.add(algorithm)
        result.append(algorithm)

    return result


def slug_float(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def load_reference_manifest(path: Path) -> dict:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("reference manifest does " f"not exist: {path}")

    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != prepare_reference.REFERENCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported reference " "manifest schema")

    reference = manifest.get("reference") or {}
    machine_tag = str(reference.get("machine_tag", "")).strip()
    aggregation = str(reference.get("aggregation", "")).strip()

    if not machine_tag:
        raise ValueError("reference manifest contains " "an empty machine tag")

    if aggregation not in ("mean", "median"):
        raise ValueError("reference manifest contains " "an invalid aggregation")

    source_path = Path(reference.get("source_path", ""))

    if not source_path.is_file():
        raise ValueError("reference source CSV " f"does not exist: {source_path}")

    actual_source_hash = preprocess.sha256_file(source_path)
    expected_source_hash = reference.get("source_sha256")

    if actual_source_hash != expected_source_hash:
        raise ValueError("reference source SHA-256 " "does not match manifest")

    scalers = manifest.get("scalers")

    if not isinstance(scalers, list) or not scalers:
        raise ValueError("reference manifest contains " "no scalers")

    if len(scalers) != len(set(scalers)):
        raise ValueError("reference manifest contains " "duplicate scalers")

    preprocessing = manifest.get("preprocessing") or {}
    reference_rows = int(reference.get("row_count", 0))

    if reference_rows <= 0:
        raise ValueError("reference manifest contains " "an invalid row count")

    for scaler in scalers:
        if scaler not in (preprocess.SCALERS):
            raise ValueError("reference manifest " "contains unsupported " f"scaler {scaler!r}")

        entry = preprocessing.get(scaler) or {}
        csv_path = Path(entry.get("preprocessed_csv", ""))
        model_path = Path(entry.get("model", ""))

        if not csv_path.is_file():
            raise ValueError("preprocessed CSV does " f"not exist for {scaler}: " f"{csv_path}")

        if not model_path.is_file():
            raise ValueError(
                "preprocessing model does " f"not exist for {scaler}: " f"{model_path}"
            )

        if int(entry.get("fit_sample_count", 0)) != reference_rows:
            raise ValueError("preprocessing fit sample " f"count mismatch for {scaler}")

        if entry.get("model_source_sha256") != expected_source_hash:
            raise ValueError("preprocessing provenance " f"mismatch for {scaler}")

        model = preprocess.load_model(model_path)

        if model.get("source_sha256") != expected_source_hash:
            raise ValueError(
                "serialized preprocessing " "model points to another " f"source for {scaler}"
            )

        rows, _, metadata = cluster.load_preprocessed_dataset(csv_path)

        if len(rows) != reference_rows:
            raise ValueError("preprocessed row count " f"mismatch for {scaler}")

        if metadata["aggregation"] != aggregation:
            raise ValueError("preprocessed aggregation " f"mismatch for {scaler}")

        if metadata["scaler"] != scaler:
            raise ValueError("preprocessed scaler " f"mismatch for {scaler}")

        if {row["machine_tag"] for row in rows} != {machine_tag}:
            raise ValueError(
                "preprocessed reference " "contains another " f"machine tag for {scaler}"
            )

    manifest["_manifest_path"] = str(path)

    return manifest


def validate_reference_preferences(manifest: dict, preference_meta: dict) -> None:
    reference = manifest["reference"]

    if reference["aggregation"] != preference_meta["aggregation"]:
        raise ValueError(
            "reference aggregation " "does not match architecture " "preference aggregation"
        )

    if reference["machine_tag"] not in (
        preference_meta["x86_machine_tag"],
        preference_meta["arm_machine_tag"],
    ):
        raise ValueError(
            "reference machine tag must "
            "match either the x86 or ARM "
            "machine tag used by the "
            "architecture preference "
            "dataset"
        )


def atomic_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEEP_SUMMARY_HEADER)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field: ("" if row.get(field) is None else row.get(field, ""))
                        for field in SWEEP_SUMMARY_HEADER
                    }
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def base_summary_row(
    sweep_run_id: str,
    reference_manifest: dict,
    preference_meta: dict,
    scaler: str,
    algorithm: str,
    configuration_id: str,
    run_dir: Path,
) -> dict:
    return {
        "sweep_csv_schema_version": SWEEP_CSV_SCHEMA_VERSION,
        "sweep_run_id": sweep_run_id,
        "reference_run_id": reference_manifest["reference_run_id"],
        "preference_run_id": preference_meta["preference_run_id"],
        "reference_machine_tag": reference_manifest["reference"]["machine_tag"],
        "aggregation": reference_manifest["reference"]["aggregation"],
        "threshold_percent": preference_meta["threshold_percent"],
        "scaler": scaler,
        "algorithm": algorithm,
        "configuration_id": configuration_id,
        "status": None,
        "reason": None,
        "k": None,
        "eps": None,
        "min_samples": None,
        "n_init": None,
        "random_state": None,
        "sample_count": None,
        "cluster_count": None,
        "noise_count": None,
        "coverage": None,
        "silhouette": None,
        "inertia": None,
        "matched_count": None,
        "assignment_match_coverage": None,
        "preference_match_coverage": None,
        "overall_purity": None,
        "homogeneity": None,
        "completeness": None,
        "v_measure": None,
        "adjusted_rand_index": None,
        "normalized_mutual_information": None,
        "external_metrics_defined": None,
        "run_dir": str(run_dir),
    }


def evaluate_configuration(
    sweep_run_id: str,
    configuration_id: str,
    run_dir: Path,
    input_path: Path,
    rows: list[dict],
    matrix: np.ndarray,
    metadata: dict,
    labels: np.ndarray,
    parameters: dict,
    result: dict,
    algorithm: str,
    machine_tag: str,
    preferences_path: Path,
    preferences: list[dict],
    preference_meta: dict,
) -> dict:

    run_dir.mkdir(parents=True, exist_ok=True)
    clustering_run_id = f"{sweep_run_id}--" f"{configuration_id}"
    artifact = cluster.build_artifact(input_path, clustering_run_id, metadata, algorithm, len(rows), parameters, result)

    if algorithm == "kmeans":
        replay = cluster.predict_kmeans_from_artifact(matrix, artifact)

        if not np.array_equal(labels, replay):
            raise RuntimeError(
                "serialized K-Means " "centroids do not reproduce " "reference assignments"
            )

    model_path = run_dir / "clustering-model.json"
    assignments_path = run_dir / "assignments.csv"
    evaluation_path = run_dir / "evaluation.json"
    matched_path = run_dir / "matched.csv"
    cluster_summary_path = run_dir / "cluster-summary.csv"
    cluster.atomic_json(model_path, artifact)
    cluster.write_assignments(assignments_path, rows, matrix, labels, artifact)
    assignment_rows, assignment_meta = evaluate_clustering.load_assignments(assignments_path, machine_tag)
    matched, join_summary = evaluate_clustering.join_assignments_preferences(assignment_rows, preferences)

    evaluation, summaries = evaluate_clustering.build_evaluation(
        clustering_run_id,
        assignments_path,
        preferences_path,
        assignment_meta,
        preference_meta,
        matched,
        join_summary,
    )

    cluster.atomic_json(evaluation_path, evaluation)
    evaluate_clustering.write_matched(matched_path, clustering_run_id, assignment_meta, preference_meta, matched)
    evaluate_clustering.write_cluster_summary(cluster_summary_path, clustering_run_id, assignment_meta, preference_meta, summaries    )
    actual_labels = {int(label) for label in labels if int(label) >= 0}
    external = evaluation["external_metrics_clustered_only"]
    cluster_summary = evaluation["cluster_summary"]
    geometric_silhouette = (
        result.get("silhouette_score")
        if algorithm == "kmeans"
        else result.get("silhouette_clustered_only")
    )

    return {
        "sample_count": len(rows),
        "cluster_count": len(actual_labels),
        "noise_count": int(np.sum(labels == -1)),
        "coverage": cluster_summary["coverage"],
        "silhouette": geometric_silhouette,
        "inertia": (result.get("inertia") if algorithm == "kmeans" else None),
        "matched_count": join_summary["matched_count"],
        "assignment_match_coverage": join_summary["assignment_match_coverage"],
        "preference_match_coverage": join_summary["preference_match_coverage"],
        "overall_purity": cluster_summary["overall_purity"],
        "homogeneity": external["homogeneity"],
        "completeness": external["completeness"],
        "v_measure": external["v_measure"],
        "adjusted_rand_index": external["adjusted_rand_index"],
        "normalized_mutual_information": external["normalized_mutual_information"],
        "external_metrics_defined": external["defined"],
    }


def run_sweep(
    reference_manifest_path: Path,
    preferences_path: Path,
    run_id: str,
    output_dir: Path,
    requested_scalers: list[str],
    requested_algorithms: list[str],
    k_values: list[int],
    eps_values: list[float],
    min_samples_values: list[int],
    n_init: int,
    random_state: int,
) -> tuple[dict, list[dict]]:
    run_id = run_id.strip()

    if not run_id:
        raise ValueError("sweep run ID cannot " "be empty")

    if n_init <= 0:
        raise ValueError("K-Means n_init must " "be positive")

    algorithms = selected_algorithms(requested_algorithms)
    k_values = unique_ints(k_values, "K-Means k", 2)
    eps_values = unique_floats(eps_values, "DBSCAN eps")
    min_samples_values = unique_ints(min_samples_values, "DBSCAN min_samples", 1)

    if "kmeans" in algorithms and not k_values:
        raise ValueError("at least one --k is " "required when K-Means " "is selected")

    if "dbscan" in algorithms and (not eps_values or not min_samples_values):
        raise ValueError(
            "at least one --eps and " "one --min-samples are " "required when DBSCAN " "is selected"
        )

    manifest = load_reference_manifest(reference_manifest_path)
    scalers = selected_scalers(requested_scalers, manifest["scalers"])
    preferences_path = preferences_path.expanduser().resolve()
    preferences, preference_meta = evaluate_clustering.load_preferences(preferences_path)
    preference_meta = dict(preference_meta)
    preference_meta["_preferences_sha256"] = preference.sha256_file(preferences_path)
    validate_reference_preferences(manifest, preference_meta)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    summary_rows = []
    configuration_number = 0

    for scaler in scalers:
        preprocessed_path = Path(manifest["preprocessing"][scaler]["preprocessed_csv"])

        rows, matrix, metadata = cluster.load_preprocessed_dataset(preprocessed_path)

        if "kmeans" in algorithms:
            for k in k_values:
                configuration_number += 1

                configuration_id = f"{configuration_number:04d}_" f"{scaler}_kmeans_k{k}"

                run_dir = runs_dir / configuration_id

                summary = base_summary_row(
                    run_id, manifest, preference_meta, scaler, "kmeans", configuration_id, run_dir
                )

                summary["k"] = k
                summary["n_init"] = n_init
                summary["random_state"] = random_state
                summary["sample_count"] = len(rows)

                if k > len(rows):
                    summary["status"] = "skipped"
                    summary["reason"] = "k exceeds " "reference sample count"
                    summary_rows.append(summary)

                    continue

                labels, parameters, result = cluster.fit_kmeans(matrix, k, n_init, random_state)
                metrics = evaluate_configuration(
                    run_id,
                    configuration_id,
                    run_dir,
                    preprocessed_path,
                    rows,
                    matrix,
                    metadata,
                    labels,
                    parameters,
                    result,
                    "kmeans",
                    manifest["reference"]["machine_tag"],
                    preferences_path,
                    preferences,
                    preference_meta,
                )

                summary.update(metrics)
                summary["status"] = "ok"
                summary_rows.append(summary)

        if "dbscan" in algorithms:
            for eps in eps_values:
                for min_samples in min_samples_values:
                    configuration_number += 1

                    configuration_id = (
                        f"{configuration_number:04d}_"
                        f"{scaler}_dbscan_"
                        f"eps{slug_float(eps)}_"
                        f"min{min_samples}"
                    )

                    run_dir = runs_dir / configuration_id
                    summary = base_summary_row(
                        run_id,
                        manifest,
                        preference_meta,
                        scaler,
                        "dbscan",
                        configuration_id,
                        run_dir,
                    )

                    summary["eps"] = eps
                    summary["min_samples"] = min_samples
                    summary["sample_count"] = len(rows)
                    labels, parameters, result = cluster.fit_dbscan(matrix, eps, min_samples)
                    metrics = evaluate_configuration(
                        run_id,
                        configuration_id,
                        run_dir,
                        preprocessed_path,
                        rows,
                        matrix,
                        metadata,
                        labels,
                        parameters,
                        result,
                        "dbscan",
                        manifest["reference"]["machine_tag"],
                        preferences_path,
                        preferences,
                        preference_meta,
                    )

                    summary.update(metrics)
                    summary["status"] = "ok"
                    summary_rows.append(summary)

    summary_path = output_dir / "sweep-summary.csv"
    atomic_summary_csv(summary_path, summary_rows)
    successful = sum(row["status"] == "ok" for row in summary_rows)
    skipped = sum(row["status"] == "skipped" for row in summary_rows)

    sweep_manifest = {
        "schema_version": SWEEP_MANIFEST_SCHEMA_VERSION,
        "sweep_csv_schema_version": SWEEP_CSV_SCHEMA_VERSION,
        "sweep_run_id": run_id,
        "reference_manifest": str(Path(reference_manifest_path).expanduser().resolve()),
        "reference_manifest_sha256": preprocess.sha256_file(
            Path(reference_manifest_path).expanduser().resolve()
        ),
        "preferences": str(preferences_path),
        "preferences_sha256": preference_meta["_preferences_sha256"],
        "reference_machine_tag": manifest["reference"]["machine_tag"],
        "aggregation": manifest["reference"]["aggregation"],
        "threshold_percent": preference_meta["threshold_percent"],
        "scalers": scalers,
        "algorithms": algorithms,
        "k_values": k_values,
        "eps_values": eps_values,
        "min_samples_values": min_samples_values,
        "n_init": n_init,
        "random_state": random_state,
        "configuration_count": len(summary_rows),
        "successful_configuration_count": successful,
        "skipped_configuration_count": skipped,
        "summary_csv": str(summary_path),
        "automatic_winner_selection": False,
        "metric_policy": {
            "dbscan_external_metrics": "noise excluded; coverage " "reported separately",
            "dbscan_silhouette": "computed on clustered " "samples only",
            "kmeans_inertia": "do not compare directly " "across different scalers",
            "selection": "no composite score and " "no automatic winner",
        },
    }

    sweep_manifest_path = output_dir / "sweep-manifest.json"
    cluster.atomic_json(sweep_manifest_path, sweep_manifest)
    sweep_manifest["manifest_path"] = str(sweep_manifest_path)
    return sweep_manifest, summary_rows


def run(args: argparse.Namespace) -> None:
    manifest, rows = run_sweep(
        Path(args.reference_manifest),
        Path(args.preferences),
        args.run_id,
        Path(args.output_dir),
        args.scaler,
        args.algorithm,
        args.k,
        args.eps,
        args.min_samples,
        args.n_init,
        args.random_state,
    )

    print(
        "reference_machine_tag="
        f"{manifest['reference_machine_tag']} "
        "aggregation="
        f"{manifest['aggregation']} "
        "scalers="
        f"{len(manifest['scalers'])} "
        "configurations="
        f"{manifest['configuration_count']} "
        "successful="
        f"{manifest['successful_configuration_count']} "
        "skipped="
        f"{manifest['skipped_configuration_count']}"
    )

    for row in rows:
        if row["status"] == "skipped":
            print("[skip] " f"id={row['configuration_id']} " f"reason={row['reason']}")

            continue

        print(
            "[ok] "
            f"id={row['configuration_id']} "
            f"clusters="
            f"{row['cluster_count']} "
            f"noise={row['noise_count']} "
            f"coverage={row['coverage']} "
            f"silhouette={row['silhouette']} "
            f"purity={row['overall_purity']} "
            f"ari={row['adjusted_rand_index']} "
            f"nmi="
            f"{row['normalized_mutual_information']}"
        )

    print("summary=" f"{manifest['summary_csv']}")
    print("manifest=" f"{manifest['manifest_path']}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Run a reproducible sweep of "
            "K-Means and DBSCAN over an "
            "architecture-specific "
            "Serverledge reference dataset "
            "and evaluate each configuration "
            "against architecture preference "
            "ground truth."
        )
    )

    root.add_argument("--reference-manifest", required=True)
    root.add_argument("--preferences", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--output-dir", required=True)
    root.add_argument(
        "--scaler",
        action="append",
        default=[],
        choices=(preprocess.SCALERS),
        help=(
            "reference scaler to evaluate; "
            "repeat for multiple scalers. "
            "If omitted, all scalers in the "
            "reference manifest are used."
        ),
    )

    root.add_argument(
        "--algorithm",
        action="append",
        default=[],
        choices=("kmeans", "dbscan"),
        help=(
            "algorithm to evaluate; repeat "
            "for both. If omitted, both "
            "K-Means and DBSCAN are used."
        ),
    )

    root.add_argument(
        "--k",
        action="append",
        type=int,
        default=[],
        help=("K-Means cluster count; repeat " "for multiple values."),
    )

    root.add_argument(
        "--eps",
        action="append",
        type=float,
        default=[],
        help=("DBSCAN eps; repeat for " "multiple values."),
    )

    root.add_argument(
        "--min-samples",
        action="append",
        type=int,
        default=[],
        help=("DBSCAN min_samples; repeat " "for multiple values."),
    )

    root.add_argument("--n-init", type=int, default=10)
    root.add_argument("--random-state", type=int, default=42)
    return root


def main() -> None:
    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
