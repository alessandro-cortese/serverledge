#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from analysis.profiling import preprocess, similarity_selection, transfer_catalog

TRANSFER_QUERY_BUILD_SCHEMA_VERSION = 1


def load_catalog_feature_space(path: Path) -> dict:

    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("transfer donor catalog " f"does not exist: {path}")

    document = json.loads(path.read_text(encoding="utf-8"))

    if document.get("schema_version") != transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported transfer " "catalog schema")

    if document.get("feature_names") != list(preprocess.FEATURE_NAMES):
        raise ValueError("transfer catalog feature " "names or order are invalid")

    feature_space = document.get("feature_space") or {}

    if feature_space.get("representation") != "preprocessed":
        raise ValueError("transfer catalog does not " "use preprocessed features")

    scaler = str(feature_space.get("scaler", "")).strip()

    if scaler not in preprocess.SCALERS:
        raise ValueError("transfer catalog scaler " "is invalid")

    clustering = document.get("clustering") or {}

    profile_machine_tag = str(clustering.get("profile_machine_tag", "")).strip()

    aggregation = str(clustering.get("aggregation", "")).strip()

    if not profile_machine_tag:
        raise ValueError("transfer catalog profile " "machine tag is missing")

    if aggregation not in ("mean", "median"):
        raise ValueError("transfer catalog " "aggregation is invalid")

    return {
        "path": str(path),
        "sha256": preprocess.sha256_file(path),
        "catalog_run_id": str(document.get("catalog_run_id", "")).strip(),
        "profile_machine_tag": profile_machine_tag,
        "aggregation": aggregation,
        "scaler": scaler,
    }


def parse_optional_positive_float(value, field: str) -> float | None:

    if value is None:
        return None

    result = float(value)

    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be " "finite and positive")

    return result


def parse_optional_positive_int(value, field: str) -> int | None:

    if value is None:
        return None

    result = int(value)

    if result <= 0:
        raise ValueError(f"{field} must be positive")

    return result


def build_transfer_query(
    input_path: Path,
    catalog_path: Path,
    model_path: Path,
    function_name: str,
    query_id: str,
    configured_cpus: float | None = None,
    configured_memory_mb: int | None = None,
    cluster_label: int | None = None,
) -> dict:

    function_name = function_name.strip()

    query_id = query_id.strip()

    if not function_name:
        raise ValueError("function name cannot be empty")

    if not query_id:
        raise ValueError("query ID cannot be empty")

    configured_cpus = parse_optional_positive_float(configured_cpus, "configured_cpus")

    configured_memory_mb = parse_optional_positive_int(configured_memory_mb, "configured_memory_mb")

    if cluster_label is not None:
        cluster_label = int(cluster_label)

        if cluster_label < 0:
            raise ValueError("cluster_label " "cannot be negative")

    catalog_meta = load_catalog_feature_space(catalog_path)

    input_path = input_path.expanduser().resolve()

    if not input_path.is_file():
        raise ValueError("FunctionProfile CSV " f"does not exist: {input_path}")

    rows, matrix, experiment_id, aggregation = preprocess.load_source(input_path)

    if aggregation != catalog_meta["aggregation"]:
        raise ValueError(
            "FunctionProfile aggregation "
            "does not match donor catalog: "
            f"input={aggregation}, "
            "catalog="
            f"{catalog_meta['aggregation']}"
        )

    matching_indexes: list[int] = []

    for index, row in enumerate(rows):

        if row["function_name"].strip() != function_name:
            continue

        if row["machine_tag"].strip() != catalog_meta["profile_machine_tag"]:
            continue

        row_cpus = float(row["configured_cpus"])

        row_memory = int(row["configured_memory_mb"])

        if configured_cpus is not None and not math.isclose(
            row_cpus, configured_cpus, rel_tol=0.0, abs_tol=1e-12
        ):
            continue

        if configured_memory_mb is not None and row_memory != configured_memory_mb:
            continue

        matching_indexes.append(index)

    if not matching_indexes:
        raise ValueError(
            "no FunctionProfile row matches "
            "the requested function, "
            "reference machine tag "
            f"{catalog_meta['profile_machine_tag']!r} "
            "and configuration"
        )

    if len(matching_indexes) != 1:

        raise ValueError(
            "multiple FunctionProfile rows "
            "match the requested function "
            "and reference machine tag; "
            "specify --configured-cpus and "
            "--configured-memory-mb"
        )

    index = matching_indexes[0]

    row = rows[index]

    model_path = model_path.expanduser().resolve()

    if not model_path.is_file():
        raise ValueError("preprocessing model " f"does not exist: {model_path}")

    model = preprocess.load_model(model_path)

    if model["aggregation"] != catalog_meta["aggregation"]:
        raise ValueError("preprocessing model " "aggregation does not " "match donor catalog")

    if model["scaler"] != catalog_meta["scaler"]:
        raise ValueError("preprocessing model scaler " "does not match donor catalog")

    transformed = preprocess.apply_model(
        np.asarray([matrix[index]], dtype=np.float64), model, aggregation
    )[0]

    feature_vector = [float(value) for value in transformed.tolist()]

    if not all(math.isfinite(value) for value in feature_vector):
        raise ValueError("preprocessed query contains " "non-finite features")

    return {
        "schema_version": similarity_selection.TRANSFER_QUERY_SCHEMA_VERSION,
        "query_id": query_id,
        "function_name": function_name,
        "configured_cpus": float(row["configured_cpus"]),
        "configured_memory_mb": int(row["configured_memory_mb"]),
        "sample_count": int(row["sample_count"]),
        "profile_machine_tag": catalog_meta["profile_machine_tag"],
        "aggregation": aggregation,
        "scaler": catalog_meta["scaler"],
        "feature_names": list(preprocess.FEATURE_NAMES),
        "feature_vector": feature_vector,
        "cluster_label": cluster_label,
        "sources": {
            "function_profile_csv": {
                "path": str(input_path),
                "sha256": preprocess.sha256_file(input_path),
                "experiment_id": experiment_id,
            },
            "preprocessing_model": {
                "path": str(model_path),
                "sha256": preprocess.sha256_file(model_path),
                "fit_experiment_id": model["fit_experiment_id"],
            },
            "donor_catalog": {
                "path": catalog_meta["path"],
                "sha256": catalog_meta["sha256"],
                "catalog_run_id": catalog_meta["catalog_run_id"],
            },
        },
        "builder": {"schema_version": TRANSFER_QUERY_BUILD_SCHEMA_VERSION},
    }


def atomic_json(path: Path, document: dict) -> None:

    preprocess.atomic_text(
        (path.expanduser().resolve()), (json.dumps(document, indent=2, sort_keys=True) + "\n")
    )


def run(args: argparse.Namespace) -> None:

    document = build_transfer_query(
        Path(args.input),
        Path(args.catalog),
        Path(args.model),
        args.function,
        args.query_id,
        args.configured_cpus,
        args.configured_memory_mb,
        args.cluster_label,
    )

    output_path = Path(args.output)

    atomic_json(output_path, document)

    print(
        "query_id="
        f"{document['query_id']} "
        "function="
        f"{document['function_name']} "
        "machine_tag="
        f"{document['profile_machine_tag']} "
        "samples="
        f"{document['sample_count']} "
        "aggregation="
        f"{document['aggregation']} "
        "scaler="
        f"{document['scaler']}"
    )

    print(f"output={output_path}")


def parser() -> argparse.ArgumentParser:

    root = argparse.ArgumentParser(
        description=(
            "Build a Serverledge "
            "transfer-learning query for "
            "one new function by applying "
            "the donor catalog preprocessing "
            "model to its aggregated "
            "FunctionProfile."
        )
    )

    root.add_argument("--input", required=True)

    root.add_argument("--catalog", required=True)

    root.add_argument("--model", required=True)

    root.add_argument("--function", required=True)

    root.add_argument("--query-id", required=True)

    root.add_argument("--output", required=True)

    root.add_argument("--configured-cpus", type=float)

    root.add_argument("--configured-memory-mb", type=int)

    root.add_argument("--cluster-label", type=int)

    return root


def main() -> None:

    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
