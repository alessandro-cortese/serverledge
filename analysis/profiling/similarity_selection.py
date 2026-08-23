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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from analysis.profiling import preprocess, transfer_catalog

TRANSFER_QUERY_SCHEMA_VERSION = 1
DONOR_SELECTION_SCHEMA_VERSION = 1
DONOR_RANKING_CSV_SCHEMA_VERSION = 1


RANKING_HEADER = [
    "donor_ranking_csv_schema_version",
    "selection_run_id",
    "query_id",
    "rank",
    "function_name",
    "configured_cpus",
    "configured_memory_mb",
    "cluster_label",
    "architecture_preference",
    "distance",
    "within_threshold",
    "selected",
]


def parse_finite(value, field: str) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")

    return result


def parse_positive_float(value, field: str) -> float:
    result = parse_finite(value, field)

    if result <= 0:
        raise ValueError(f"{field} must be positive")

    return result


def parse_positive_int(value, field: str) -> int:
    try:
        result = int(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer") from exc

    if result <= 0:
        raise ValueError(f"{field} must be positive")

    return result


def load_catalog(path: Path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("transfer donor catalog " f"does not exist: {path}")

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if document.get("schema_version") != transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported transfer " "catalog schema")

    feature_names = document.get("feature_names") or []

    if feature_names != list(preprocess.FEATURE_NAMES):
        raise ValueError("transfer catalog feature " "names or order are invalid")

    feature_space = document.get("feature_space") or {}

    if feature_space.get("representation") != "preprocessed":
        raise ValueError("transfer catalog does not " "use preprocessed features")

    scaler = str(feature_space.get("scaler", "")).strip()

    if not scaler:
        raise ValueError("transfer catalog scaler " "is missing")

    clustering = document.get("clustering") or {}
    profile_machine_tag = str(clustering.get("profile_machine_tag", "")).strip()
    aggregation = str(clustering.get("aggregation", "")).strip()

    if not profile_machine_tag or not aggregation:
        raise ValueError("transfer catalog clustering " "metadata are incomplete")

    donors = document.get("donors") or []

    if not donors:
        raise ValueError("transfer donor catalog " "contains no donors")

    normalized = []

    for donor in donors:
        vector = donor.get("feature_vector")

        if not isinstance(vector, list) or len(vector) != len(preprocess.FEATURE_NAMES):
            raise ValueError("donor feature vector " "has invalid dimension")

        features = [parse_finite(value, "donor feature") for value in vector]
        configured_cpus = parse_positive_float(donor["configured_cpus"], "configured_cpus")
        configured_memory_mb = parse_positive_int(donor["configured_memory_mb"], "configured_memory_mb")
        function_name = str(donor["function_name"]).strip()

        if not function_name:
            raise ValueError("donor function name " "cannot be empty")

        donor_eligible = donor.get("donor_eligible")

        if not isinstance(donor_eligible, bool):
            raise ValueError("donor_eligible must " "be boolean")

        is_noise = donor.get("is_noise")

        if not isinstance(is_noise, bool):
            raise ValueError("is_noise must " "be boolean")

        cluster_label = int(donor["cluster_label"])

        if is_noise and cluster_label != -1:
            raise ValueError("noise donor must use " "cluster_label=-1")

        if donor.get("bandit_prior") is not None:
            raise ValueError(
                "10A similarity selection " "requires an unmaterialized " "bandit prior"
            )

        normalized.append(
            {
                **donor,
                "function_name": function_name,
                "configured_cpus": configured_cpus,
                "configured_memory_mb": configured_memory_mb,
                "cluster_label": cluster_label,
                "feature_vector": features,
            }
        )

    summary = document.get("summary") or {}

    if summary.get("donor_count") != len(normalized):
        raise ValueError("transfer catalog donor " "count does not match summary")

    return (
        document,
        normalized,
        {
            "path": str(path),
            "sha256": preprocess.sha256_file(path),
            "catalog_run_id": document["catalog_run_id"],
            "profile_machine_tag": profile_machine_tag,
            "aggregation": aggregation,
            "scaler": scaler,
        },
    )


def load_query(path: Path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("transfer query does " f"not exist: {path}")

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if document.get("schema_version") != TRANSFER_QUERY_SCHEMA_VERSION:
        raise ValueError("unsupported transfer " "query schema")

    query_id = str(document.get("query_id", "")).strip()
    function_name = str(document.get("function_name", "")).strip()

    if not query_id or not function_name:
        raise ValueError("query ID and function " "name are required")

    configured_cpus = parse_positive_float(document.get("configured_cpus"), "configured_cpus")
    configured_memory_mb = parse_positive_int(document.get("configured_memory_mb"), "configured_memory_mb")
    sample_count = parse_positive_int(document.get("sample_count"), "sample_count")
    profile_machine_tag = str(document.get("profile_machine_tag", "")).strip()
    aggregation = str(document.get("aggregation", "")).strip()
    scaler = str(document.get("scaler", "")).strip()

    if not profile_machine_tag or not aggregation or not scaler:
        raise ValueError("query feature-space " "metadata are incomplete")

    feature_names = document.get("feature_names") or []

    if feature_names != list(preprocess.FEATURE_NAMES):
        raise ValueError("query feature names " "or order are invalid")

    vector = document.get("feature_vector")

    if not isinstance(vector, list) or len(vector) != len(preprocess.FEATURE_NAMES):
        raise ValueError("query feature vector " "has invalid dimension")

    features = [parse_finite(value, "query feature") for value in vector]
    cluster_label = document.get("cluster_label")

    if cluster_label is not None:
        cluster_label = int(cluster_label)

        if cluster_label < 0:
            raise ValueError("query cluster label " "cannot be negative")

    return (
        {
            "schema_version": TRANSFER_QUERY_SCHEMA_VERSION,
            "query_id": query_id,
            "function_name": function_name,
            "configured_cpus": configured_cpus,
            "configured_memory_mb": configured_memory_mb,
            "sample_count": sample_count,
            "profile_machine_tag": profile_machine_tag,
            "aggregation": aggregation,
            "scaler": scaler,
            "feature_names": list(feature_names),
            "feature_vector": features,
            "cluster_label": cluster_label,
        },
        {"path": str(path), "sha256": preprocess.sha256_file(path)},
    )


def euclidean_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("feature vectors have " "different dimensions")

    return math.sqrt(
        sum((left_value - right_value) ** 2 for (left_value, right_value) in zip(left, right))
    )


def distance_within_threshold(distance: float, max_distance: float) -> bool:
    return distance < max_distance or math.isclose(
        distance, max_distance, rel_tol=1e-12, abs_tol=1e-12
    )


def same_configuration(query: dict, donor: dict) -> bool:
    return (
        math.isclose(query["configured_cpus"], donor["configured_cpus"], rel_tol=0.0, abs_tol=1e-12)
        and query["configured_memory_mb"] == donor["configured_memory_mb"]
    )


def validate_feature_space(query: dict, catalog_meta: dict):
    if query["profile_machine_tag"] != catalog_meta["profile_machine_tag"]:
        raise ValueError("query profile machine tag " "does not match donor catalog")

    if query["aggregation"] != catalog_meta["aggregation"]:
        raise ValueError("query aggregation does " "not match donor catalog")

    if query["scaler"] != catalog_meta["scaler"]:
        raise ValueError("query scaler does not " "match donor catalog")


def select_donor(catalog_path: Path, query_path: Path, selection_run_id: str, max_distance: float, require_same_cluster: bool):
    selection_run_id = selection_run_id.strip()

    if not selection_run_id:
        raise ValueError("selection run ID " "cannot be empty")

    max_distance = parse_finite(max_distance, "max_distance")

    if max_distance < 0:
        raise ValueError("max_distance cannot " "be negative")

    _, donors, catalog_meta = load_catalog(catalog_path)
    query, query_meta = load_query(query_path)
    validate_feature_space(query, catalog_meta)
    eligible = [donor for donor in donors if donor["donor_eligible"]]

    if not eligible:
        return build_no_transfer(
            selection_run_id,
            catalog_meta,
            query,
            query_meta,
            max_distance,
            require_same_cluster,
            "no_eligible_donors",
            [],
        )

    same_config = [donor for donor in eligible if same_configuration(query, donor)]

    if not same_config:
        return build_no_transfer(
            selection_run_id,
            catalog_meta,
            query,
            query_meta,
            max_distance,
            require_same_cluster,
            "no_matching_configuration",
            [],
        )

    candidates = same_config

    if require_same_cluster:
        if query["cluster_label"] is None:
            raise ValueError("same-cluster selection " "requires query cluster_label")

        candidates = [
            donor for donor in candidates if (donor["cluster_label"] == query["cluster_label"])
        ]

        if not candidates:
            return build_no_transfer(
                selection_run_id,
                catalog_meta,
                query,
                query_meta,
                max_distance,
                require_same_cluster,
                "no_same_cluster_candidates",
                [],
            )

    ranking = []

    for donor in candidates:
        distance = euclidean_distance(query["feature_vector"], donor["feature_vector"])

        ranking.append(
            {
                "function_name": donor["function_name"],
                "configured_cpus": donor["configured_cpus"],
                "configured_memory_mb": donor["configured_memory_mb"],
                "cluster_label": donor["cluster_label"],
                "architecture_preference": donor["architecture_preference"],
                "arm_vs_x86_delta_percent": donor["arm_vs_x86_delta_percent"],
                "x86_duration_ms": donor["x86_duration_ms"],
                "arm_duration_ms": donor["arm_duration_ms"],
                "distance": distance,
            }
        )

    ranking.sort(
        key=lambda item: (
            item["distance"],
            item["function_name"],
            item["configured_cpus"],
            item["configured_memory_mb"],
        )
    )

    for index, candidate in enumerate(ranking, start=1):
        candidate["rank"] = index
        candidate["within_threshold"] = distance_within_threshold(candidate["distance"], max_distance)
        candidate["selected"] = False

    nearest = ranking[0]

    if not distance_within_threshold(nearest["distance"], max_distance):
        return build_no_transfer(
            selection_run_id,
            catalog_meta,
            query,
            query_meta,
            max_distance,
            require_same_cluster,
            "distance_threshold_exceeded",
            ranking,
        )

    nearest["selected"] = True

    return {
        "schema_version": DONOR_SELECTION_SCHEMA_VERSION,
        "selection_run_id": selection_run_id,
        "status": "selected",
        "reason": "",
        "query": query,
        "selection_policy": {
            "distance": "euclidean",
            "max_distance": max_distance,
            "configuration_match_required": True,
            "require_same_cluster": require_same_cluster,
            "bandit_prior_materialized": False,
        },
        "sources": {
            "catalog": {
                "path": catalog_meta["path"],
                "sha256": catalog_meta["sha256"],
                "catalog_run_id": catalog_meta["catalog_run_id"],
            },
            "query": {"path": query_meta["path"], "sha256": query_meta["sha256"]},
        },
        "selected_donor": {
            key: nearest[key]
            for key in (
                "function_name",
                "configured_cpus",
                "configured_memory_mb",
                "cluster_label",
                "architecture_preference",
                "arm_vs_x86_delta_percent",
                "x86_duration_ms",
                "arm_duration_ms",
                "distance",
            )
        },
        "candidate_count": len(ranking),
        "ranking": ranking,
        "bandit_prior": None,
    }


def build_no_transfer(
    selection_run_id: str,
    catalog_meta: dict,
    query: dict,
    query_meta: dict,
    max_distance: float,
    require_same_cluster: bool,
    reason: str,
    ranking: list[dict],
):
    return {
        "schema_version": DONOR_SELECTION_SCHEMA_VERSION,
        "selection_run_id": selection_run_id,
        "status": "no-transfer",
        "reason": reason,
        "query": query,
        "selection_policy": {
            "distance": "euclidean",
            "max_distance": max_distance,
            "configuration_match_required": True,
            "require_same_cluster": require_same_cluster,
            "bandit_prior_materialized": False,
        },
        "sources": {
            "catalog": {
                "path": catalog_meta["path"],
                "sha256": catalog_meta["sha256"],
                "catalog_run_id": catalog_meta["catalog_run_id"],
            },
            "query": {"path": query_meta["path"], "sha256": query_meta["sha256"]},
        },
        "selected_donor": None,
        "candidate_count": len(ranking),
        "ranking": ranking,
        "bandit_prior": None,
    }


def atomic_json(path: Path, document: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:

            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_ranking_csv(path: Path, document: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RANKING_HEADER)
            writer.writeheader()

            for candidate in document["ranking"]:
                writer.writerow(
                    {
                        "donor_ranking_csv_schema_version": DONOR_RANKING_CSV_SCHEMA_VERSION,
                        "selection_run_id": document["selection_run_id"],
                        "query_id": document["query"]["query_id"],
                        "rank": candidate["rank"],
                        "function_name": candidate["function_name"],
                        "configured_cpus": format(candidate["configured_cpus"], ".17g"),
                        "configured_memory_mb": candidate["configured_memory_mb"],
                        "cluster_label": candidate["cluster_label"],
                        "architecture_preference": candidate["architecture_preference"],
                        "distance": format(candidate["distance"], ".17g"),
                        "within_threshold": str(candidate["within_threshold"]).lower(),
                        "selected": str(candidate["selected"]).lower(),
                    }
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def run(args: argparse.Namespace):
    document = select_donor(
        Path(args.catalog),
        Path(args.query),
        args.run_id,
        args.max_distance,
        args.require_same_cluster,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    atomic_json(output_json, document)
    write_ranking_csv(output_csv, document)

    print(
        "status="
        f"{document['status']} "
        "reason="
        f"{document['reason'] or 'none'} "
        "candidates="
        f"{document['candidate_count']}"
    )

    if document["selected_donor"] is not None:
        selected = document["selected_donor"]

        print(
            "selected_donor=" f"{selected['function_name']} " "distance=" f"{selected['distance']}"
        )

    print("bandit_prior_materialized=false")
    print(f"output_json={output_json}")
    print(f"output_csv={output_csv}")


def parser():
    root = argparse.ArgumentParser(
        description=(
            "Rank eligible Serverledge transfer donors "
            "for a preprocessed query function and "
            "select a donor only when the nearest "
            "distance satisfies an explicit threshold."
        )
    )

    root.add_argument("--catalog", required=True)
    root.add_argument("--query", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--max-distance", required=True, type=float)
    root.add_argument("--require-same-cluster", action="store_true")
    root.add_argument("--output-json", required=True)
    root.add_argument("--output-csv", required=True)

    return root


def main():
    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
