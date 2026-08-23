#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from analysis.profiling import similarity_selection

SIMILARITY_CALIBRATION_SCHEMA_VERSION = 1
LOO_CSV_SCHEMA_VERSION = 1
THRESHOLD_SWEEP_CSV_SCHEMA_VERSION = 1


LOO_HEADER = [
    "loo_csv_schema_version",
    "calibration_run_id",
    "query_function_name",
    "configured_cpus",
    "configured_memory_mb",
    "query_cluster_label",
    "query_architecture_preference",
    "status",
    "candidate_count",
    "nearest_donor_function_name",
    "nearest_donor_cluster_label",
    "nearest_donor_architecture_preference",
    "distance",
    "preference_match",
]


SWEEP_HEADER = [
    "threshold_sweep_csv_schema_version",
    "calibration_run_id",
    "threshold",
    "accepted_count",
    "rejected_count",
    "coverage",
    "preference_match_count",
    "preference_mismatch_count",
    "preference_agreement_rate",
    "selected_threshold",
]


def donor_identity(donor: dict):
    return (donor["function_name"], donor["configured_cpus"], donor["configured_memory_mb"])


def percentile(values: list[float], q: float):
    if not values:
        return None

    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def candidate_donors(query: dict, donors: list[dict], require_same_cluster: bool):

    query_identity = donor_identity(query)
    candidates = []

    for donor in donors:
        if not donor["donor_eligible"]:
            continue

        if donor_identity(donor) == query_identity:
            continue

        if not similarity_selection.same_configuration(query, donor):
            continue

        if require_same_cluster and donor["cluster_label"] != query["cluster_label"]:
            continue

        candidates.append(donor)

    return candidates


def nearest_for_query(query: dict, donors: list[dict], require_same_cluster: bool):

    candidates = candidate_donors(query, donors, require_same_cluster)

    if not candidates:
        return {
            "query_function_name": query["function_name"],
            "configured_cpus": query["configured_cpus"],
            "configured_memory_mb": query["configured_memory_mb"],
            "query_cluster_label": query["cluster_label"],
            "query_architecture_preference": query["architecture_preference"],
            "status": "no-candidate",
            "candidate_count": 0,
            "nearest_donor_function_name": None,
            "nearest_donor_cluster_label": None,
            "nearest_donor_architecture_preference": None,
            "distance": None,
            "preference_match": None,
        }

    ranked = []

    for donor in candidates:
        distance = similarity_selection.euclidean_distance(
            query["feature_vector"], donor["feature_vector"]
        )

        ranked.append((distance, donor))

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1]["function_name"],
            item[1]["configured_cpus"],
            item[1]["configured_memory_mb"],
        )
    )

    distance, nearest = ranked[0]

    return {
        "query_function_name": query["function_name"],
        "configured_cpus": query["configured_cpus"],
        "configured_memory_mb": query["configured_memory_mb"],
        "query_cluster_label": query["cluster_label"],
        "query_architecture_preference": query["architecture_preference"],
        "status": "neighbor-found",
        "candidate_count": len(candidates),
        "nearest_donor_function_name": nearest["function_name"],
        "nearest_donor_cluster_label": nearest["cluster_label"],
        "nearest_donor_architecture_preference": nearest["architecture_preference"],
        "distance": distance,
        "preference_match": (query["architecture_preference"] == nearest["architecture_preference"]),
    }


def threshold_candidates(rows: list[dict]) -> list[float]:
    distances = sorted({row["distance"] for row in rows if row["status"] == "neighbor-found"})

    if not distances:
        return []

    if distances[0] == 0.0:
        return distances

    return [0.0, *distances]


def evaluate_threshold(rows: list[dict], threshold: float):
    comparable = [row for row in rows if row["status"] == "neighbor-found"]

    accepted = [
        row
        for row in comparable
        if (similarity_selection.distance_within_threshold(row["distance"], threshold))
    ]

    accepted_count = len(accepted)

    rejected_count = len(comparable) - accepted_count

    matches = sum(row["preference_match"] is True for row in accepted)

    mismatches = sum(row["preference_match"] is False for row in accepted)

    return {
        "threshold": threshold,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "coverage": (accepted_count / len(comparable) if comparable else None),
        "preference_match_count": matches,
        "preference_mismatch_count": mismatches,
        "preference_agreement_rate": (matches / accepted_count if accepted_count else None),
        "selected_threshold": False,
    }


def calibrate(catalog_path: Path, calibration_run_id: str, require_same_cluster: bool):
    calibration_run_id = calibration_run_id.strip()

    if not calibration_run_id:
        raise ValueError("calibration run ID " "cannot be empty")

    _, donors, catalog_meta = similarity_selection.load_catalog(catalog_path)

    queries = [donor for donor in donors if donor["donor_eligible"]]

    if not queries:
        raise ValueError("transfer catalog contains " "no eligible donors for " "calibration")

    rows = [nearest_for_query(query, donors, require_same_cluster) for query in queries]
    distances = [row["distance"] for row in rows if row["status"] == "neighbor-found"]
    sweep = [evaluate_threshold(rows, threshold) for threshold in threshold_candidates(rows)]

    distance_summary = {
        "count": len(distances),
        "min": (min(distances) if distances else None),
        "q25": percentile(distances, 0.25),
        "median": (statistics.median(distances) if distances else None),
        "q75": percentile(distances, 0.75),
        "q90": percentile(distances, 0.90),
        "q95": percentile(distances, 0.95),
        "max": (max(distances) if distances else None),
    }

    preference_matches = sum(
        row["preference_match"] is True for row in rows if row["status"] == "neighbor-found"
    )

    preference_mismatches = sum(
        row["preference_match"] is False for row in rows if row["status"] == "neighbor-found"
    )

    return {
        "schema_version": SIMILARITY_CALIBRATION_SCHEMA_VERSION,
        "calibration_run_id": calibration_run_id,
        "method": "leave-one-out-nearest-neighbor",
        "distance": "euclidean",
        "require_same_configuration": True,
        "require_same_cluster": require_same_cluster,
        "threshold_selected": False,
        "selected_threshold": None,
        "bandit_prior_materialized": False,
        "source": {
            "catalog": {
                "path": catalog_meta["path"],
                "sha256": catalog_meta["sha256"],
                "catalog_run_id": catalog_meta["catalog_run_id"],
                "profile_machine_tag": catalog_meta["profile_machine_tag"],
                "aggregation": catalog_meta["aggregation"],
                "scaler": catalog_meta["scaler"],
            }
        },
        "summary": {
            "eligible_query_count": len(queries),
            "neighbor_found_count": len(distances),
            "no_candidate_count": (len(queries) - len(distances)),
            "preference_match_count": preference_matches,
            "preference_mismatch_count": preference_mismatches,
            "nearest_neighbor_preference_agreement_rate": (
                preference_matches / len(distances) if distances else None
            ),
            "threshold_candidate_count": len(sweep),
        },
        "nearest_distance_summary": distance_summary,
        "leave_one_out": rows,
        "threshold_sweep": sweep,
    }


def atomic_json(path: Path, document: dict):
    similarity_selection.atomic_json(path, document)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def loo_csv_rows(artifact: dict):
    rows = []

    for row in artifact["leave_one_out"]:
        rows.append(
            {
                "loo_csv_schema_version": LOO_CSV_SCHEMA_VERSION,
                "calibration_run_id": artifact["calibration_run_id"],
                "query_function_name": row["query_function_name"],
                "configured_cpus": format(row["configured_cpus"], ".17g"),
                "configured_memory_mb": row["configured_memory_mb"],
                "query_cluster_label": row["query_cluster_label"],
                "query_architecture_preference": row["query_architecture_preference"],
                "status": row["status"],
                "candidate_count": row["candidate_count"],
                "nearest_donor_function_name": (row["nearest_donor_function_name"] or ""),
                "nearest_donor_cluster_label": (
                    row["nearest_donor_cluster_label"]
                    if row["nearest_donor_cluster_label"] is not None
                    else ""
                ),
                "nearest_donor_architecture_preference": (
                    row["nearest_donor_architecture_preference"] or ""
                ),
                "distance": (
                    format(row["distance"], ".17g") if row["distance"] is not None else ""
                ),
                "preference_match": (
                    str(row["preference_match"]).lower()
                    if row["preference_match"] is not None
                    else ""
                ),
            }
        )

    return rows


def sweep_csv_rows(artifact: dict):
    rows = []

    for row in artifact["threshold_sweep"]:
        rows.append(
            {
                "threshold_sweep_csv_schema_version": THRESHOLD_SWEEP_CSV_SCHEMA_VERSION,
                "calibration_run_id": artifact["calibration_run_id"],
                "threshold": format(row["threshold"], ".17g"),
                "accepted_count": row["accepted_count"],
                "rejected_count": row["rejected_count"],
                "coverage": (
                    format(row["coverage"], ".17g") if row["coverage"] is not None else ""
                ),
                "preference_match_count": row["preference_match_count"],
                "preference_mismatch_count": row["preference_mismatch_count"],
                "preference_agreement_rate": (
                    format(row["preference_agreement_rate"], ".17g")
                    if row["preference_agreement_rate"] is not None
                    else ""
                ),
                "selected_threshold": "false",
            }
        )

    return rows


def run(args: argparse.Namespace):
    artifact = calibrate(Path(args.catalog), args.run_id, args.require_same_cluster)
    output_json = Path(args.output_json)
    output_loo_csv = Path(args.output_loo_csv)
    output_sweep_csv = Path(args.output_sweep_csv)
    atomic_json(output_json, artifact)
    atomic_csv(output_loo_csv, LOO_HEADER, loo_csv_rows(artifact))
    atomic_csv(output_sweep_csv, SWEEP_HEADER, sweep_csv_rows(artifact))
    summary = artifact["summary"]
    distances = artifact["nearest_distance_summary"]
    print(
        "queries="
        f"{summary['eligible_query_count']} "
        "neighbor_found="
        f"{summary['neighbor_found_count']} "
        "no_candidate="
        f"{summary['no_candidate_count']}"
    )

    print(
        "nearest_distance_min="
        f"{distances['min']} "
        "median="
        f"{distances['median']} "
        "max="
        f"{distances['max']}"
    )

    print(
        "preference_agreement="
        f"{summary['preference_match_count']}/"
        f"{summary['neighbor_found_count']} "
        "threshold_candidates="
        f"{summary['threshold_candidate_count']}"
    )

    print("selected_threshold=none")
    print("bandit_prior_materialized=false")
    print(f"output_json={output_json}")
    print(f"output_loo_csv={output_loo_csv}")
    print(f"output_sweep_csv={output_sweep_csv}")


def parser():
    root = argparse.ArgumentParser(
        description=(
            "Calibrate candidate similarity thresholds "
            "for Serverledge transfer learning with a "
            "leave-one-out nearest-neighbor analysis. "
            "No scientific threshold is selected "
            "automatically."
        )
    )

    root.add_argument("--catalog", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--require-same-cluster", action="store_true")
    root.add_argument("--output-json", required=True)
    root.add_argument("--output-loo-csv", required=True)
    root.add_argument("--output-sweep-csv", required=True)

    return root


def main():
    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
