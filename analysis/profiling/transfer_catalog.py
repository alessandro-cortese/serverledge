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


from analysis.profiling import benchmark_readiness, cluster, evaluate_clustering, preprocess

TRANSFER_CATALOG_SCHEMA_VERSION = 1
TRANSFER_CATALOG_CSV_SCHEMA_VERSION = 1


CATALOG_HEADER = [
    "transfer_catalog_csv_schema_version",
    "catalog_run_id",
    "clustering_run_id",
    "preference_run_id",
    "function_name",
    "configured_cpus",
    "configured_memory_mb",
    "profile_machine_tag",
    "aggregation",
    "scaler",
    "algorithm",
    "cluster_label",
    "is_noise",
    "donor_eligible",
    "donor_ineligibility_reason",
    "architecture_preference",
    "arm_vs_x86_delta_percent",
    "threshold_percent",
    "x86_duration_ms",
    "arm_duration_ms",
    *preprocess.FEATURE_NAMES,
    "bandit_prior_materialized",
]

PREFERENCE_TRANSFER_REQUIRED_COLUMNS = [
    "function_name",
    "configured_cpus",
    "configured_memory_mb",
    "aggregation",
    "performance_metric",
    "threshold_percent",
    "x86_machine_tag",
    "arm_machine_tag",
    "x86_sample_count",
    "arm_sample_count",
    "x86_duration_ms",
    "arm_duration_ms",
    "arm_vs_x86_delta_percent",
    "architecture_preference",
]


def parse_float(value, field: str) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")

    return result


def parse_positive_float(value, field: str) -> float:
    result = parse_float(value, field)

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


def parse_bool(value, field: str) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    raise ValueError(f"{field} must be true or false")


def identity(function_name: str, configured_cpus, configured_memory_mb):
    function_name = str(function_name).strip()

    if not function_name:
        raise ValueError("function_name cannot be empty")

    return (
        function_name,
        parse_positive_float(configured_cpus, "configured_cpus"),
        parse_positive_int(configured_memory_mb, "configured_memory_mb"),
    )


def load_readiness(path: Path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("benchmark readiness JSON " f"does not exist: {path}")

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)

    if document.get("schema_version") != benchmark_readiness.BENCHMARK_READINESS_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark " "readiness schema")

    summary = document.get("summary") or {}

    if summary.get("structural_ready") is not True:
        raise ValueError("benchmark is not " "structurally ready")

    architectures = document.get("architectures") or {}

    x86_tag = str(architectures.get("x86_machine_tag", "")).strip()

    arm_tag = str(architectures.get("arm_machine_tag", "")).strip()

    if not x86_tag or not arm_tag or x86_tag == arm_tag:
        raise ValueError("invalid benchmark " "architecture tags")

    functions = document.get("functions") or []

    if not functions:
        raise ValueError("benchmark readiness " "contains no functions")

    ready = {}

    for row in functions:
        if row.get("status") != "ready":
            raise ValueError("structurally ready benchmark " "contains a non-ready function")

        key = identity(row["function_name"], row["configured_cpus"], row["configured_memory_mb"])

        if key in ready:
            raise ValueError("duplicate function identity " "in benchmark readiness")

        ready[key] = row

    return (
        document,
        ready,
        {
            "path": str(path),
            "sha256": preprocess.sha256_file(path),
            "x86_machine_tag": x86_tag,
            "arm_machine_tag": arm_tag,
            "function_count": len(ready),
        },
    )


def validate_assignments_file_machine_tag(path: Path, profile_machine_tag: str):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("clustering assignments CSV " f"does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames or "machine_tag" not in reader.fieldnames:
            raise ValueError("assignments CSV has no " "machine_tag column")

        tags = {row["machine_tag"].strip() for row in reader}

    if tags != {profile_machine_tag}:
        raise ValueError(
            "assignments CSV must contain "
            "exactly the requested profile "
            f"machine tag {profile_machine_tag!r}; "
            f"found {sorted(tags)}"
        )


def load_assignments_with_features(path: Path, profile_machine_tag: str):
    path = path.expanduser().resolve()

    # Manteniamo le validazioni già introdotte
    # nella pipeline di evaluation.
    validate_assignments_file_machine_tag(path, profile_machine_tag)

    _, metadata = evaluate_clustering.load_assignments(path, profile_machine_tag)

    # La 09C.3A non aveva bisogno delle feature
    # e restituisce una rappresentazione ridotta
    # delle assignment.
    #
    # Il transfer catalog deve invece conservare
    # l'intero vettore preprocessato, quindi
    # rileggiamo il CSV originale dopo averlo
    # validato.
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != cluster.ASSIGNMENT_HEADER:
            raise ValueError("unexpected clustering " "assignments CSV header")

        rows = [row for row in reader if (row["machine_tag"].strip() == profile_machine_tag)]

    if not rows:
        raise ValueError(
            "assignments CSV contains "
            "no rows for profile machine "
            f"tag {profile_machine_tag!r}"
        )

    return (rows, metadata)


def normalize_assignment(row: dict):
    key = identity(row["function_name"], row["configured_cpus"], row["configured_memory_mb"])

    cluster_label = int(row["cluster_label"])

    is_noise = parse_bool(row["is_noise"], "is_noise")

    if is_noise and cluster_label != -1:
        raise ValueError("noise assignment must " "use cluster_label=-1")

    if not is_noise and cluster_label < 0:
        raise ValueError("non-noise assignment cannot " "use a negative cluster label")

    features = []

    for feature_name in preprocess.FEATURE_NAMES:
        features.append(parse_float(row[feature_name], feature_name))

    return (
        key,
        {
            "function_name": key[0],
            "configured_cpus": key[1],
            "configured_memory_mb": key[2],
            "machine_tag": str(row["machine_tag"]).strip(),
            "cluster_label": cluster_label,
            "is_noise": is_noise,
            "features": features,
        },
    )


def normalize_preference(row: dict):
    key = identity(row["function_name"], row["configured_cpus"], row["configured_memory_mb"])

    return (
        key,
        {
            "function_name": key[0],
            "configured_cpus": key[1],
            "configured_memory_mb": key[2],
            "architecture_preference": str(row["architecture_preference"]).strip(),
            "arm_vs_x86_delta_percent": parse_float(
                row["arm_vs_x86_delta_percent"], "arm_vs_x86_delta_percent"
            ),
            "threshold_percent": parse_float(row["threshold_percent"], "threshold_percent"),
            "x86_duration_ms": parse_float(row["x86_duration_ms"], "x86_duration_ms"),
            "arm_duration_ms": parse_float(row["arm_duration_ms"], "arm_duration_ms"),
        },
    )


def exact_key_match(name: str, expected: set, actual: set):
    missing = sorted(expected - actual)

    extra = sorted(actual - expected)

    if missing or extra:
        raise ValueError(
            f"{name} identities do not match "
            "benchmark readiness; "
            f"missing={missing} extra={extra}"
        )


def load_preferences_with_performance(path: Path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError("architecture preference CSV " f"does not exist: {path}")

    # Riutilizziamo il loader della 09C.3A
    # per validare schema, metadata,
    # aggregation, machine tag, threshold,
    # preference labels e provenance.
    validated_rows, metadata = evaluate_clustering.load_preferences(path)

    # Il loader della evaluation restituisce
    # soltanto i campi necessari alla
    # cluster↔preference evaluation.
    #
    # Il Transfer Catalog deve conservare
    # anche le performance x86/ARM complete,
    # quindi rileggiamo il CSV originale.
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        fieldnames = reader.fieldnames or []

        missing_columns = [
            column for column in PREFERENCE_TRANSFER_REQUIRED_COLUMNS if column not in fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "architecture preference CSV "
                "is missing columns required "
                "by the transfer catalog: " + ", ".join(missing_columns)
            )

        rows = list(reader)

    if not rows:
        raise ValueError("architecture preference CSV " "contains no data rows")

    # Il loader validato e la rilettura raw
    # devono rappresentare lo stesso dataset.
    if len(rows) != len(validated_rows):
        raise RuntimeError("validated preference row count " "does not match raw CSV row count")

    return (rows, metadata)


def build_catalog(
    readiness_path: Path,
    assignments_path: Path,
    preferences_path: Path,
    profile_machine_tag: str,
    catalog_run_id: str,
):
    catalog_run_id = catalog_run_id.strip()

    profile_machine_tag = profile_machine_tag.strip()

    if not catalog_run_id:
        raise ValueError("catalog run ID " "cannot be empty")

    if not profile_machine_tag:
        raise ValueError("profile machine tag " "cannot be empty")

    _, ready_functions, readiness_meta = load_readiness(readiness_path)

    if profile_machine_tag not in {
        readiness_meta["x86_machine_tag"],
        readiness_meta["arm_machine_tag"],
    }:
        raise ValueError("profile machine tag must " "match one of the benchmark " "architectures")

    assignments_path = assignments_path.expanduser().resolve()

    assignment_rows, assignment_meta = load_assignments_with_features(
        assignments_path, profile_machine_tag
    )

    preferences_path = preferences_path.expanduser().resolve()

    preference_rows, preference_meta = load_preferences_with_performance(preferences_path)

    if assignment_meta["aggregation"] != preference_meta["aggregation"]:
        raise ValueError("assignment aggregation " "does not match preference " "aggregation")

    if (
        preference_meta["x86_machine_tag"] != readiness_meta["x86_machine_tag"]
        or preference_meta["arm_machine_tag"] != readiness_meta["arm_machine_tag"]
    ):
        raise ValueError("preference architecture tags " "do not match benchmark readiness")

    assignments = {}

    for row in assignment_rows:
        key, normalized = normalize_assignment(row)

        if key in assignments:
            raise ValueError("duplicate assignment " f"identity {key}")

        assignments[key] = normalized

    preferences = {}

    for row in preference_rows:
        key, normalized = normalize_preference(row)

        if key in preferences:
            raise ValueError("duplicate preference " f"identity {key}")

        preferences[key] = normalized

    expected_keys = set(ready_functions)

    exact_key_match("assignment", expected_keys, set(assignments))

    exact_key_match("preference", expected_keys, set(preferences))

    donors = []

    for key in sorted(expected_keys):
        assignment = assignments[key]

        pref = preferences[key]

        is_noise = assignment["is_noise"]

        donor_eligible = not is_noise

        donor = {
            "function_name": key[0],
            "configured_cpus": key[1],
            "configured_memory_mb": key[2],
            "profile_machine_tag": profile_machine_tag,
            "aggregation": assignment_meta["aggregation"],
            "scaler": assignment_meta["scaler"],
            "algorithm": assignment_meta["algorithm"],
            "cluster_label": assignment["cluster_label"],
            "is_noise": is_noise,
            "donor_eligible": donor_eligible,
            "donor_ineligibility_reason": ("" if donor_eligible else "clustering_noise"),
            "architecture_preference": pref["architecture_preference"],
            "arm_vs_x86_delta_percent": pref["arm_vs_x86_delta_percent"],
            "threshold_percent": pref["threshold_percent"],
            "x86_duration_ms": pref["x86_duration_ms"],
            "arm_duration_ms": pref["arm_duration_ms"],
            "feature_vector": assignment["features"],
            # Deliberatamente non derivato
            # dalla architecture preference.
            "bandit_prior": None,
        }

        donors.append(donor)

    eligible_count = sum(donor["donor_eligible"] for donor in donors)

    noise_count = sum(donor["is_noise"] for donor in donors)

    artifact = {
        "schema_version": TRANSFER_CATALOG_SCHEMA_VERSION,
        "catalog_run_id": catalog_run_id,
        "feature_names": list(preprocess.FEATURE_NAMES),
        "feature_space": {
            "representation": "preprocessed",
            "scaler": assignment_meta["scaler"],
            "distance_candidate": "euclidean",
            "distance_policy_selected": False,
        },
        "clustering": {
            "clustering_run_id": assignment_meta["clustering_run_id"],
            "algorithm": assignment_meta["algorithm"],
            "aggregation": assignment_meta["aggregation"],
            "profile_machine_tag": profile_machine_tag,
        },
        "architecture_ground_truth": {
            "preference_run_id": preference_meta["preference_run_id"],
            "x86_machine_tag": preference_meta["x86_machine_tag"],
            "arm_machine_tag": preference_meta["arm_machine_tag"],
            "threshold_percent": preference_meta["threshold_percent"],
        },
        "sources": {
            "benchmark_readiness": {
                "path": readiness_meta["path"],
                "sha256": readiness_meta["sha256"],
            },
            "assignments": {
                "path": str(assignments_path),
                "sha256": preprocess.sha256_file(assignments_path),
            },
            "preferences": {
                "path": str(preferences_path),
                "sha256": preprocess.sha256_file(preferences_path),
            },
        },
        "donor_policy": {
            "readiness_required": True,
            "noise_is_eligible": False,
            "bandit_prior_materialized": False,
            "architecture_preference_is_bandit_reward": False,
        },
        "summary": {
            "donor_count": len(donors),
            "eligible_donor_count": eligible_count,
            "ineligible_donor_count": (len(donors) - eligible_count),
            "noise_donor_count": noise_count,
        },
        "donors": donors,
    }

    return artifact


def atomic_json(path: Path, artifact: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )

    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:

            json.dump(artifact, handle, indent=2, sort_keys=True)

            handle.write("\n")

            handle.flush()

            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_catalog_csv(path: Path, artifact: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )

    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:

            writer = csv.DictWriter(handle, fieldnames=CATALOG_HEADER)

            writer.writeheader()

            for donor in artifact["donors"]:
                row = {
                    "transfer_catalog_csv_schema_version": TRANSFER_CATALOG_CSV_SCHEMA_VERSION,
                    "catalog_run_id": artifact["catalog_run_id"],
                    "clustering_run_id": artifact["clustering"]["clustering_run_id"],
                    "preference_run_id": artifact["architecture_ground_truth"]["preference_run_id"],
                    "function_name": donor["function_name"],
                    "configured_cpus": format(donor["configured_cpus"], ".17g"),
                    "configured_memory_mb": donor["configured_memory_mb"],
                    "profile_machine_tag": donor["profile_machine_tag"],
                    "aggregation": donor["aggregation"],
                    "scaler": donor["scaler"],
                    "algorithm": donor["algorithm"],
                    "cluster_label": donor["cluster_label"],
                    "is_noise": str(donor["is_noise"]).lower(),
                    "donor_eligible": str(donor["donor_eligible"]).lower(),
                    "donor_ineligibility_reason": donor["donor_ineligibility_reason"],
                    "architecture_preference": donor["architecture_preference"],
                    "arm_vs_x86_delta_percent": format(donor["arm_vs_x86_delta_percent"], ".17g"),
                    "threshold_percent": format(donor["threshold_percent"], ".17g"),
                    "x86_duration_ms": format(donor["x86_duration_ms"], ".17g"),
                    "arm_duration_ms": format(donor["arm_duration_ms"], ".17g"),
                    "bandit_prior_materialized": "false",
                }

                for feature_name, value in zip(artifact["feature_names"], donor["feature_vector"]):
                    row[feature_name] = format(value, ".17g")

                writer.writerow(row)

            handle.flush()

            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def run(args: argparse.Namespace):
    artifact = build_catalog(
        Path(args.readiness),
        Path(args.assignments),
        Path(args.preferences),
        args.profile_machine_tag,
        args.run_id,
    )

    output_json = Path(args.output_json)

    output_csv = Path(args.output_csv)

    atomic_json(output_json, artifact)

    write_catalog_csv(output_csv, artifact)

    summary = artifact["summary"]

    print(
        "donors="
        f"{summary['donor_count']} "
        "eligible="
        f"{summary['eligible_donor_count']} "
        "ineligible="
        f"{summary['ineligible_donor_count']} "
        "noise="
        f"{summary['noise_donor_count']}"
    )

    print(
        "profile_machine_tag="
        f"{artifact['clustering']['profile_machine_tag']} "
        "aggregation="
        f"{artifact['clustering']['aggregation']} "
        "scaler="
        f"{artifact['feature_space']['scaler']} "
        "algorithm="
        f"{artifact['clustering']['algorithm']}"
    )

    print("bandit_prior_materialized=false")

    print(f"output_json={output_json}")

    print(f"output_csv={output_csv}")


def parser():
    root = argparse.ArgumentParser(
        description=(
            "Build a Serverledge transfer-learning "
            "candidate donor catalog from a structurally "
            "ready benchmark, clustering assignments "
            "and architecture-preference ground truth."
        )
    )

    root.add_argument("--readiness", required=True)

    root.add_argument("--assignments", required=True)

    root.add_argument("--preferences", required=True)

    root.add_argument("--profile-machine-tag", required=True)

    root.add_argument("--run-id", required=True)

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
