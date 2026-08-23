#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import cluster, preference

EVALUATION_SCHEMA_VERSION = 1
MATCHED_CSV_SCHEMA_VERSION = 1
CLUSTER_SUMMARY_CSV_SCHEMA_VERSION = 1

AMBIGUOUS_PREFERENCE = "ambiguous"

PREFERENCE_LABELS = {
    preference.PREFERENCE_X86,
    preference.PREFERENCE_ARM,
    preference.PREFERENCE_INDEPENDENT,
}


MATCHED_HEADER = [
    "matched_csv_schema_version",
    "evaluation_run_id",
    "clustering_run_id",
    "preference_run_id",
    "aggregation",
    "scaler",
    "algorithm",
    "profile_machine_tag",
    "function_name",
    "configured_cpus",
    "configured_memory_mb",
    "cluster_label",
    "is_noise",
    "architecture_preference",
    "arm_vs_x86_delta_percent",
    "threshold_percent",
]


CLUSTER_SUMMARY_HEADER = [
    "cluster_summary_csv_schema_version",
    "evaluation_run_id",
    "clustering_run_id",
    "preference_run_id",
    "aggregation",
    "scaler",
    "algorithm",
    "profile_machine_tag",
    "cluster_label",
    "cluster_size",
    "x86_preferred_count",
    "arm_preferred_count",
    "architecture_independent_count",
    "majority_preference",
    "cluster_purity",
]


def _finite(value, field: str, row_number: int) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: " f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: " f"{field} is not finite")

    return result


def _boolean(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"row {row_number}: " f"{field} must be true or false")


def _single(values: set, label: str):
    if len(values) != 1:
        raise ValueError(f"dataset mixes {label}")

    return next(iter(values))


def load_assignments(path: Path, profile_machine_tag: str) -> tuple[list[dict], dict]:
    profile_machine_tag = profile_machine_tag.strip()

    if not profile_machine_tag:
        raise ValueError("profile machine tag " "cannot be empty")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != cluster.ASSIGNMENT_HEADER:
            raise ValueError("unexpected clustering " "assignment CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("clustering assignment CSV " "contains no data rows")

    run_ids = set()
    aggregations = set()
    scalers = set()
    algorithms = set()
    experiment_ids = set()
    fit_experiment_ids = set()
    machine_tags = set()
    selected = []
    identities = set()

    for row_number, row in enumerate(rows, start=2):
        if row["clustering_csv_schema_version"] != str(cluster.CLUSTERING_CSV_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported clustering " "CSV schema")

        if row["clustering_model_schema_version"] != str(cluster.CLUSTERING_MODEL_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported clustering " "model schema")

        run_id = row["clustering_run_id"].strip()
        aggregation = row["aggregation"].strip()
        scaler = row["scaler"].strip()
        algorithm = row["algorithm"].strip()
        experiment_id = row["experiment_id"].strip()
        fit_experiment_id = row["fit_experiment_id"].strip()
        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()

        if not all(
            (
                run_id,
                aggregation,
                scaler,
                algorithm,
                experiment_id,
                fit_experiment_id,
                function_name,
                machine_tag,
            )
        ):
            raise ValueError(f"row {row_number}: " "empty clustering metadata")

        if aggregation not in ("mean", "median"):
            raise ValueError(f"row {row_number}: " "invalid aggregation " f"{aggregation!r}")

        if algorithm not in ("kmeans", "dbscan"):
            raise ValueError(f"row {row_number}: " "invalid algorithm " f"{algorithm!r}")

        cpus = _finite(row["configured_cpus"], "configured_cpus", row_number)

        try:
            memory_mb = int(row["configured_memory_mb"])
            label = int(row["cluster_label"])

        except ValueError as exc:
            raise ValueError(f"row {row_number}: " "invalid integer metadata") from exc

        is_noise = _boolean(row["is_noise"], "is_noise", row_number)

        if cpus <= 0 or memory_mb <= 0:
            raise ValueError(f"row {row_number}: " "invalid resource " "configuration")

        if label < -1 or is_noise != (label == -1):
            raise ValueError(f"row {row_number}: " "invalid " "cluster_label/is_noise pair")

        run_ids.add(run_id)
        aggregations.add(aggregation)
        scalers.add(scaler)
        algorithms.add(algorithm)
        experiment_ids.add(experiment_id)
        fit_experiment_ids.add(fit_experiment_id)
        machine_tags.add(machine_tag)

        if machine_tag != profile_machine_tag:
            continue

        identity = (function_name, cpus, memory_mb)

        if identity in identities:
            raise ValueError("duplicate selected " "clustering identity " f"{identity}")

        identities.add(identity)

        selected.append(
            {
                "function_name": function_name,
                "machine_tag": machine_tag,
                "configured_cpus": cpus,
                "configured_memory_mb": memory_mb,
                "cluster_label": label,
                "is_noise": is_noise,
            }
        )

    if not selected:
        raise ValueError(
            "no assignments for "
            f"machine tag "
            f"{profile_machine_tag!r}; "
            f"available tags: "
            f"{sorted(machine_tags)}"
        )

    return (
        selected,
        {
            "clustering_run_id": _single(run_ids, "clustering run IDs"),
            "aggregation": _single(aggregations, "aggregations"),
            "scaler": _single(scalers, "scalers"),
            "algorithm": _single(algorithms, "algorithms"),
            "experiment_id": _single(experiment_ids, "experiment IDs"),
            "fit_experiment_id": _single(fit_experiment_ids, "fit experiment IDs"),
            "available_machine_tags": sorted(machine_tags),
            "profile_machine_tag": profile_machine_tag,
            "source_assignment_count": len(rows),
            "selected_assignment_count": len(selected),
        },
    )


def load_preferences(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != preference.ARCHITECTURE_PREFERENCE_HEADER:
            raise ValueError("unexpected architecture " "preference CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("architecture preference CSV " "contains no data rows")

    run_ids = set()
    performance_runs = set()
    hashes = set()
    aggregations = set()
    metrics = set()
    thresholds = set()
    x86_tags = set()
    arm_tags = set()
    parsed = []
    identities = set()

    for row_number, row in enumerate(rows, start=2):
        if row["architecture_preference_csv_schema_version"] != str(
            preference.ARCHITECTURE_PREFERENCE_CSV_SCHEMA_VERSION
        ):
            raise ValueError(f"row {row_number}: " "unsupported architecture " "preference schema")

        if row["performance_profile_csv_schema_version"] != str(
            preference.PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION
        ):
            raise ValueError(f"row {row_number}: " "unsupported performance " "profile schema")

        function_name = row["function_name"].strip()
        pref_label = row["architecture_preference"].strip()
        aggregation = row["aggregation"].strip()
        metric = row["performance_metric"].strip()
        run_id = row["preference_run_id"].strip()
        perf_run = row["performance_run_id"].strip()
        source_hash = row["performance_profiles_sha256"].strip()
        x86_tag = row["x86_machine_tag"].strip()
        arm_tag = row["arm_machine_tag"].strip()

        if pref_label not in PREFERENCE_LABELS:
            raise ValueError(
                f"row {row_number}: " "invalid architecture " "preference " f"{pref_label!r}"
            )

        if aggregation not in ("mean", "median") or metric != "duration_ms":
            raise ValueError(f"row {row_number}: " "invalid preference " "aggregation/metric")

        if not all((function_name, run_id, perf_run, source_hash, x86_tag, arm_tag)):
            raise ValueError(f"row {row_number}: " "empty preference metadata")

        cpus = _finite(row["configured_cpus"], "configured_cpus", row_number)
        threshold = _finite(row["threshold_percent"], "threshold_percent", row_number)
        delta = _finite(row["arm_vs_x86_delta_percent"], "arm_vs_x86_delta_percent", row_number)

        try:
            memory_mb = int(row["configured_memory_mb"])

        except ValueError as exc:
            raise ValueError(f"row {row_number}: " "invalid " "configured_memory_mb") from exc

        if cpus <= 0 or memory_mb <= 0 or threshold < 0:
            raise ValueError(f"row {row_number}: " "invalid preference " "configuration")

        identity = (function_name, cpus, memory_mb)

        if identity in identities:
            raise ValueError(f"row {row_number}: " "duplicate preference " f"identity {identity}")

        identities.add(identity)

        parsed.append(
            {
                "function_name": function_name,
                "configured_cpus": cpus,
                "configured_memory_mb": memory_mb,
                "aggregation": aggregation,
                "threshold_percent": threshold,
                "arm_vs_x86_delta_percent": delta,
                "architecture_preference": pref_label,
            }
        )

        run_ids.add(run_id)
        performance_runs.add(perf_run)
        hashes.add(source_hash)
        aggregations.add(aggregation)
        metrics.add(metric)
        thresholds.add(threshold)
        x86_tags.add(x86_tag)
        arm_tags.add(arm_tag)

    return (
        parsed,
        {
            "preference_run_id": _single(run_ids, "preference run IDs"),
            "performance_run_id": _single(performance_runs, "performance run IDs"),
            "performance_profiles_sha256": _single(hashes, "performance profile hashes"),
            "aggregation": _single(aggregations, "aggregations"),
            "performance_metric": _single(metrics, "performance metrics"),
            "threshold_percent": _single(thresholds, "thresholds"),
            "x86_machine_tag": _single(x86_tags, "x86 machine tags"),
            "arm_machine_tag": _single(arm_tags, "ARM machine tags"),
            "preference_count": len(parsed),
        },
    )


def join_assignments_preferences(assignments: list[dict], preferences: list[dict]):
    by_identity = {
        (
            item["function_name"],
            float(item["configured_cpus"]),
            int(item["configured_memory_mb"]),
        ): item
        for item in preferences
    }

    matched = []
    unmatched_assignments = []
    used = set()

    for assignment in assignments:
        key = (
            assignment["function_name"],
            float(assignment["configured_cpus"]),
            int(assignment["configured_memory_mb"]),
        )

        pref = by_identity.get(key)

        if pref is None:
            unmatched_assignments.append(
                {"function_name": key[0], "configured_cpus": key[1], "configured_memory_mb": key[2]}
            )

            continue

        used.add(key)
        matched.append(
            {
                **assignment,
                "architecture_preference": pref["architecture_preference"],
                "arm_vs_x86_delta_percent": pref["arm_vs_x86_delta_percent"],
                "threshold_percent": pref["threshold_percent"],
            }
        )

    if not matched:
        raise ValueError("no clustering assignments " "match architecture preferences")

    unmatched_preferences = [
        {"function_name": key[0], "configured_cpus": key[1], "configured_memory_mb": key[2]}
        for key in sorted(by_identity)
        if key not in used
    ]

    return (
        matched,
        {
            "matched_count": len(matched),
            "assignment_match_coverage": len(matched) / len(assignments),
            "preference_match_coverage": len(matched) / len(preferences),
            "unmatched_assignment_count": len(unmatched_assignments),
            "unmatched_preference_count": len(unmatched_preferences),
            "unmatched_assignments": unmatched_assignments,
            "unmatched_preferences": unmatched_preferences,
        },
    )


def cluster_preference_summaries(matched: list[dict]):
    clustered = [item for item in matched if not item["is_noise"]]
    noise = [item for item in matched if item["is_noise"]]
    groups = defaultdict(list)

    for item in clustered:
        groups[int(item["cluster_label"])].append(item)

    summaries = []
    majority_correct = 0

    for label in sorted(groups):
        items = groups[label]
        counts = Counter(item["architecture_preference"] for item in items)
        maximum = max(counts.values())
        winners = sorted(key for (key, value) in counts.items() if value == maximum)
        majority = winners[0] if len(winners) == 1 else AMBIGUOUS_PREFERENCE
        majority_correct += maximum

        summaries.append(
            {
                "cluster_label": label,
                "cluster_size": len(items),
                "x86_preferred_count": counts.get(preference.PREFERENCE_X86, 0),
                "arm_preferred_count": counts.get(preference.PREFERENCE_ARM, 0),
                "architecture_independent_count": counts.get(preference.PREFERENCE_INDEPENDENT, 0),
                "majority_preference": majority,
                "cluster_purity": maximum / len(items),
            }
        )

    noise_counts = Counter(item["architecture_preference"] for item in noise)

    return (
        summaries,
        {
            "clustered_count": len(clustered),
            "noise_count": len(noise),
            "coverage": len(clustered) / len(matched),
            "overall_purity": (majority_correct / len(clustered) if clustered else None),
            "noise_preference_counts": {
                preference.PREFERENCE_X86: noise_counts.get(preference.PREFERENCE_X86, 0),
                preference.PREFERENCE_ARM: noise_counts.get(preference.PREFERENCE_ARM, 0),
                preference.PREFERENCE_INDEPENDENT: noise_counts.get(
                    preference.PREFERENCE_INDEPENDENT, 0
                ),
            },
        },
    )


def external_metrics_if_informative(matched: list[dict]) -> dict:
    clustered = [item for item in matched if not item["is_noise"]]
    true_labels = [item["architecture_preference"] for item in clustered]
    predicted_labels = [int(item["cluster_label"]) for item in clustered]
    true_count = len(set(true_labels))
    cluster_count = len(set(predicted_labels))

    if len(clustered) < 2 or true_count < 2 or cluster_count < 2:
        return {
            "defined": False,
            "reason": (
                "need at least two "
                "clustered samples, "
                "two preference classes, "
                "and two cluster labels"
            ),
            "architecture_preference_class_count": true_count,
            "cluster_label_count": cluster_count,
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "adjusted_rand_index": None,
            "normalized_mutual_information": None,
        }

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, predicted_labels
    )

    return {
        "defined": True,
        "reason": None,
        "architecture_preference_class_count": true_count,
        "cluster_label_count": cluster_count,
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, predicted_labels)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(true_labels, predicted_labels)
        ),
    }


def build_evaluation(run_id,assignments_path,preferences_path,assignment_meta,preference_meta,matched,join_summary):
    run_id = run_id.strip()

    if not run_id:
        raise ValueError("evaluation run ID " "cannot be empty")

    if assignment_meta["aggregation"] != preference_meta["aggregation"]:
        raise ValueError(
            "clustering aggregation " "does not match architecture " "preference aggregation"
        )

    if assignment_meta["profile_machine_tag"] not in (
        preference_meta["x86_machine_tag"],
        preference_meta["arm_machine_tag"],
    ):
        raise ValueError(
            "profile machine tag must "
            "match either the x86 or ARM "
            "machine tag used by the "
            "architecture preference dataset"
        )

    summaries, summary = cluster_preference_summaries(matched)

    return (
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_run_id": run_id,
            "assignments_sha256": preference.sha256_file(assignments_path),
            "preferences_sha256": preference.sha256_file(preferences_path),
            "clustering": assignment_meta,
            "architecture_preferences": preference_meta,
            "join": join_summary,
            "cluster_summary": summary,
            "external_metrics_clustered_only": external_metrics_if_informative(matched),
            "clusters": summaries,
        },
        summaries,
    )


def _atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:

            json.dump(data, handle, indent=2, sort_keys=True)

            handle.write("\n")

            handle.flush()

            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_matched(path, run_id, assignment_meta, preference_meta, matched):
    rows = []

    for item in sorted(
        matched,
        key=lambda row: (row["function_name"], row["configured_cpus"], row["configured_memory_mb"]),
    ):
        rows.append(
            [
                MATCHED_CSV_SCHEMA_VERSION,
                run_id,
                assignment_meta["clustering_run_id"],
                preference_meta["preference_run_id"],
                assignment_meta["aggregation"],
                assignment_meta["scaler"],
                assignment_meta["algorithm"],
                assignment_meta["profile_machine_tag"],
                item["function_name"],
                format(float(item["configured_cpus"]), ".17g"),
                int(item["configured_memory_mb"]),
                int(item["cluster_label"]),
                str(bool(item["is_noise"])).lower(),
                item["architecture_preference"],
                format(float(item["arm_vs_x86_delta_percent"]), ".17g"),
                format(float(item["threshold_percent"]), ".17g"),
            ]
        )

    _atomic_csv(path, MATCHED_HEADER, rows)


def write_cluster_summary(path, run_id, assignment_meta, preference_meta, summaries):
    rows = [
        [
            CLUSTER_SUMMARY_CSV_SCHEMA_VERSION,
            run_id,
            assignment_meta["clustering_run_id"],
            preference_meta["preference_run_id"],
            assignment_meta["aggregation"],
            assignment_meta["scaler"],
            assignment_meta["algorithm"],
            assignment_meta["profile_machine_tag"],
            int(item["cluster_label"]),
            int(item["cluster_size"]),
            int(item["x86_preferred_count"]),
            int(item["arm_preferred_count"]),
            int(item["architecture_independent_count"]),
            item["majority_preference"],
            format(float(item["cluster_purity"]), ".17g"),
        ]
        for item in summaries
    ]

    _atomic_csv(path, CLUSTER_SUMMARY_HEADER, rows)


def run(args):
    assignments_path = Path(args.assignments)
    preferences_path = Path(args.preferences)
    assignments, assignment_meta = load_assignments(assignments_path, args.profile_machine_tag)
    preferences, preference_meta = load_preferences(preferences_path)
    matched, join_summary = join_assignments_preferences(assignments, preferences)

    evaluation, summaries = build_evaluation(
        args.run_id,
        assignments_path,
        preferences_path,
        assignment_meta,
        preference_meta,
        matched,
        join_summary,
    )

    _atomic_json(Path(args.output_json), evaluation)
    write_matched(Path(args.matched_output), args.run_id, assignment_meta, preference_meta, matched)
    write_cluster_summary(Path(args.cluster_summary_output), args.run_id, assignment_meta, preference_meta, summaries)

    summary = evaluation["cluster_summary"]
    metrics = evaluation["external_metrics_clustered_only"]

    print(
        "selected_assignments="
        f"{assignment_meta['selected_assignment_count']} "
        "preferences="
        f"{preference_meta['preference_count']} "
        "matched="
        f"{join_summary['matched_count']} "
        "clustered="
        f"{summary['clustered_count']} "
        "noise="
        f"{summary['noise_count']} "
        "coverage="
        f"{summary['coverage']}"
    )

    print(
        "aggregation="
        f"{assignment_meta['aggregation']} "
        "scaler="
        f"{assignment_meta['scaler']} "
        "algorithm="
        f"{assignment_meta['algorithm']} "
        "profile_machine_tag="
        f"{assignment_meta['profile_machine_tag']} "
        "threshold_percent="
        f"{preference_meta['threshold_percent']}"
    )

    print(
        "overall_purity="
        + ("undefined" if summary["overall_purity"] is None else str(summary["overall_purity"]))
    )

    if metrics["defined"]:
        print(
            "homogeneity="
            f"{metrics['homogeneity']} "
            "completeness="
            f"{metrics['completeness']} "
            "v_measure="
            f"{metrics['v_measure']} "
            "ari="
            f"{metrics['adjusted_rand_index']} "
            "nmi="
            f"{metrics['normalized_mutual_information']}"
        )

    else:
        print("external_metrics=undefined " "reason=" f"{metrics['reason']}")

    for item in summaries:
        print(
            "[cluster] "
            f"label={item['cluster_label']} "
            f"size={item['cluster_size']} "
            f"x86={item['x86_preferred_count']} "
            f"arm={item['arm_preferred_count']} "
            "independent="
            f"{item['architecture_independent_count']} "
            "majority="
            f"{item['majority_preference']} "
            "purity="
            f"{item['cluster_purity']}"
        )

    print(f"output_json=" f"{args.output_json}")
    print(f"matched_output=" f"{args.matched_output}")
    print("cluster_summary_output=" f"{args.cluster_summary_output}")


def parser():
    root = argparse.ArgumentParser(
        description=(
            "Evaluate clustering "
            "assignments against x86/ARM "
            "architecture preference "
            "ground truth."
        )
    )

    root.add_argument("--assignments", required=True)
    root.add_argument("--preferences", required=True)
    root.add_argument("--profile-machine-tag", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--output-json", required=True)
    root.add_argument("--matched-output", required=True)
    root.add_argument("--cluster-summary-output", required=True)

    return root


def main():
    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
