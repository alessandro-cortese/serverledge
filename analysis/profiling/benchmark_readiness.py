#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.profiling import evaluate_clustering, preference, preprocess

BENCHMARK_READINESS_SCHEMA_VERSION = 1
BENCHMARK_FUNCTIONS_CSV_SCHEMA_VERSION = 1


PERFORMANCE_PROFILE_HEADER = [
    "performance_profile_csv_schema_version",
    "performance_run_id",
    "input_sha256",
    "function_name",
    "machine_tag",
    "configured_cpus",
    "configured_memory_mb",
    "sample_count",
    "source_request_ids_json",
    "duration_mean_ms",
    "duration_median_ms",
    "response_time_mean_ms",
    "response_time_median_ms",
]


FUNCTION_REPORT_HEADER = [
    "benchmark_functions_csv_schema_version",
    "readiness_run_id",
    "function_name",
    "configured_cpus",
    "configured_memory_mb",
    "mean_x86_samples",
    "mean_arm_samples",
    "median_x86_samples",
    "median_arm_samples",
    "performance_x86_samples",
    "performance_arm_samples",
    "preference_mean",
    "delta_mean_percent",
    "preference_median",
    "delta_median_percent",
    "preference_agreement",
    "status",
    "issues",
]


PREFERENCE_LABELS = {
    preference.PREFERENCE_X86,
    preference.PREFERENCE_ARM,
    preference.PREFERENCE_INDEPENDENT,
}


def parse_finite(value, field: str) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")

    return result


def parse_positive_int(value, field: str) -> int:
    try:
        result = int(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer") from exc

    if result <= 0:
        raise ValueError(f"{field} must be positive")

    return result


def function_key(function_name: str, cpus: float, memory_mb: int):
    return function_name, float(cpus), int(memory_mb)


def load_function_profiles(path: Path, expected_aggregation: str):
    path = path.expanduser().resolve()
    rows, _, experiment_id, aggregation = preprocess.load_source(path)

    if aggregation != expected_aggregation:
        raise ValueError(f"FunctionProfile aggregation must be {expected_aggregation!r} got {aggregation!r}")

    parsed = {}
    machine_tags = set()

    for row in rows:
        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()

        if not function_name:
            raise ValueError("empty FunctionProfile function name")

        if not machine_tag:
            raise ValueError("empty FunctionProfile machine tag")

        cpus = parse_finite(row["configured_cpus"], "configured_cpus")
        memory_mb = parse_positive_int(row["configured_memory_mb"], "configured_memory_mb")
        sample_count = parse_positive_int(row["sample_count"], "sample_count")

        if cpus <= 0:
            raise ValueError("configured_cpus must be positive")

        key = (function_name, machine_tag, cpus, memory_mb)

        if key in parsed:
            raise ValueError(f"duplicate FunctionProfile identity {key}")

        parsed[key] = {
            "function_name": function_name,
            "machine_tag": machine_tag,
            "configured_cpus": cpus,
            "configured_memory_mb": memory_mb,
            "sample_count": sample_count,
        }

        machine_tags.add(machine_tag)

    return (
        parsed,
        {
            "path": str(path),
            "sha256": preprocess.sha256_file(path),
            "experiment_id": experiment_id,
            "aggregation": aggregation,
            "row_count": len(parsed),
            "machine_tags": sorted(machine_tags),
        },
    )


def load_performance_profiles(path: Path):
    path = path.expanduser().resolve()

    if not path.is_file():
        raise ValueError(f"PerformanceProfile CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PERFORMANCE_PROFILE_HEADER:
            raise ValueError("unexpected PerformanceProfile CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("PerformanceProfile CSV contains no data rows")

    parsed = {}
    run_ids = set()
    input_hashes = set()
    machine_tags = set()
    expected_schema = getattr(preference, "PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION", 1)

    for row_number, row in enumerate(rows, start=2):
        if row["performance_profile_csv_schema_version"] != strexpected_schema:
            raise ValueError(f"row {row_number}: unsupported PerformanceProfile schema")

        run_id = row["performance_run_id"].strip()
        input_sha256 = row["input_sha256"].strip()
        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()

        if not all(run_id, input_sha256, function_name, machine_tag):
            raise ValueError(f"row {row_number}: empty PerformanceProfile metadata")

        cpus = parse_finite(row["configured_cpus"], "configured_cpus")
        memory_mb = parse_positive_int(row["configured_memory_mb"], "configured_memory_mb")
        sample_count = parse_positive_int(row["sample_count"], "sample_count")

        if cpus <= 0:
            raise ValueError(f"row {row_number}: configured_cpus must be positive")

        for field in (
            "duration_mean_ms",
            "duration_median_ms",
            "response_time_mean_ms",
            "response_time_median_ms",
        ):
            value = parse_finite(row[field], field)

            if value < 0:
                raise ValueError(f"row {row_number}: {field} cannot be negative")

        try:
            request_ids = json.loads(row["source_request_ids_json"])

        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_number}: invalid source_request_ids_json") from exc

        if not isinstance(request_ids, list) or len(request_ids) != sample_count:
            raise ValueError(
                f"row {row_number}: source request ID count does not match sample_count"
            )

        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"row {row_number}: duplicate source request ID")

        key = (function_name, machine_tag, cpus, memory_mb)

        if key in parsed:
            raise ValueError(f"duplicate PerformanceProfile identity {key}")

        parsed[key] = {
            "function_name": function_name,
            "machine_tag": machine_tag,
            "configured_cpus": cpus,
            "configured_memory_mb": memory_mb,
            "sample_count": sample_count,
            "duration_mean_ms": float(row["duration_mean_ms"]),
            "duration_median_ms": float(row["duration_median_ms"]),
        }

        run_ids.add(run_id)
        input_hashes.add(input_sha256)
        machine_tags.add(machine_tag)

    if len(run_ids) != 1:
        raise ValueError("PerformanceProfile CSV mixes performance run IDs")

    if len(input_hashes) != 1:
        raise ValueError("PerformanceProfile CSV mixes input SHA-256 values")

    return (
        parsed,
        {
            "path": str(path),
            "sha256": preprocess.sha256_file(path),
            "performance_run_id": next(iter(run_ids)),
            "input_sha256": next(iter(input_hashes)),
            "row_count": len(parsed),
            "machine_tags": sorted(machine_tags),
        },
    )


def load_preferences(path: Path, expected_aggregation: str):
    path = path.expanduser().resolve()

    rows, metadata = evaluate_clustering.load_preferences(path)

    if metadata["aggregation"] != expected_aggregation:
        raise ValueError(f"ArchitecturePreference aggregation must be {expected_aggregation!r}")

    parsed = {}

    for row in rows:
        key = function_key(
            row["function_name"], row["configured_cpus"], row["configured_memory_mb"]
        )
        if key in parsed:
            raise ValueError(f"duplicate ArchitecturePreference identity {key}")

        parsed[key] = row

    return (parsed, {**metadata, "path": str(path), "sha256": preprocess.sha256_file(path)})


def architecture_profile(dataset: dict, key, machine_tag: str):
    return dataset.get((key[0], machine_tag, key[1], key[2]))


def source_function_keys(dataset: dict, accepted_machine_tags: set[str]):
    return {
        function_key(item["function_name"], item["configured_cpus"], item["configured_memory_mb"])
        for item in dataset.values()
        if item["machine_tag"] in accepted_machine_tags
    }

def append_sample_issue(issues: list[str], name: str, item, minimum: int, maximum: int):
    if item is None:
        return

    count = item["sample_count"]

    if not (minimum <= count <= maximum):
        issues.append(f"{name}_sample_count_out_of_range")


def preference_distribution(rows: list[dict], field: str):
    counts = Counter(row[field] for row in rows if row["status"] == "ready" and row[field])

    return {label: counts.get(label, 0) for label in sorted(PREFERENCE_LABELS)}


def build_readiness(
    function_profiles_mean_path: Path,
    function_profiles_median_path: Path,
    performance_profiles_path: Path,
    preferences_mean_path: Path,
    preferences_median_path: Path,
    x86_tag: str,
    arm_tag: str,
    run_id: str,
    min_samples: int,
    max_samples: int,
):
    run_id = run_id.strip()
    x86_tag = x86_tag.strip()
    arm_tag = arm_tag.strip()

    if not run_id:
        raise ValueError("readiness run ID cannot be empty")

    if not x86_tag or not arm_tag or x86_tag == arm_tag:
        raise ValueError("x86 and ARM tags must be non-empty and distinct")

    if min_samples <= 0 or max_samples < min_samples:
        raise ValueError("invalid sample-count range")

    mean_profiles, mean_meta = load_function_profiles(function_profiles_mean_path, "mean")
    median_profiles, median_meta = load_function_profiles(function_profiles_median_path, "median")

    if mean_meta["experiment_id"] != median_meta["experiment_id"]:
        raise ValueError("Mean and Median FunctionProfile datasets have different experiment IDs")

    performance_profiles, performance_meta = load_performance_profiles(performance_profiles_path)
    preferences_mean, preferences_mean_meta = load_preferences(preferences_mean_path, "mean")
    preferences_median, preferences_median_meta = load_preferences(
        preferences_median_path, "median"
    )

    for metadata in (preferences_mean_meta, preferences_median_meta):
        if metadata["x86_machine_tag"] != x86_tag or metadata["arm_machine_tag"] != arm_tag:
            raise ValueError("ArchitecturePreference machine tags do not match the requested benchmark tags")

    if preferences_mean_meta["threshold_percent"] != preferences_median_meta["threshold_percent"]:
        raise ValueError("Mean and Median preferences use different thresholds")

    accepted_tags = {x86_tag, arm_tag}

    all_keys = (
        source_function_keys(mean_profiles, accepted_tags)
        | source_function_keys(median_profiles, accepted_tags)
        | source_function_keys(performance_profiles, accepted_tags)
        | set(preferences_mean)
        | set(preferences_median)
    )

    report_rows = []
    for key in sorted(all_keys):
        issues = []
        mean_x86 = architecture_profile(mean_profiles, key, x86_tag)
        mean_arm = architecture_profile(mean_profiles, key, arm_tag)
        median_x86 = architecture_profile(median_profiles, key, x86_tag)
        median_arm = architecture_profile(median_profiles, key, arm_tag)
        performance_x86 = architecture_profile(performance_profiles, key, x86_tag)
        performance_arm = architecture_profile(performance_profiles, key, arm_tag)
        pref_mean = preferences_mean.get(key)
        pref_median = preferences_median.get(key)
        required = [
            ("missing_mean_x86", mean_x86),
            ("missing_mean_arm", mean_arm),
            ("missing_median_x86", median_x86),
            ("missing_median_arm", median_arm),
            ("missing_performance_x86", performance_x86),
            ("missing_performance_arm", performance_arm),
            ("missing_preference_mean", pref_mean),
            ("missing_preference_median", pref_median),
        ]

        for issue, value in required:
            if value is None:
                issues.append(issue)

        for name, item in (
            ("mean_x86", mean_x86),
            ("mean_arm", mean_arm),
            ("median_x86", median_x86),
            ("median_arm", median_arm),
            ("performance_x86", performance_x86),
            ("performance_arm", performance_arm),
        ):
            append_sample_issue(issues, name, item, min_samples, max_samples)

        if mean_x86 and median_x86 and mean_x86["sample_count"] != median_x86["sample_count"]:
            issues.append("x86_mean_median_sample_mismatch")

        if mean_arm and median_arm and mean_arm["sample_count"] != median_arm["sample_count"]:
            issues.append("arm_mean_median_sample_mismatch")

        if mean_x86 and mean_arm and mean_x86["sample_count"] != mean_arm["sample_count"]:
            issues.append("mean_cross_arch_sample_mismatch")

        if median_x86 and median_arm and median_x86["sample_count"] != median_arm["sample_count"]:
            issues.append("median_cross_arch_sample_mismatch")

        if (
            performance_x86
            and performance_arm
            and performance_x86["sample_count"] != performance_arm["sample_count"]
        ):
            issues.append("performance_cross_arch_sample_mismatch")

        preference_mean_label = "" if pref_mean is None else pref_mean["architecture_preference"]

        preference_median_label = (
            "" if pref_median is None else pref_median["architecture_preference"]
        )

        preference_agreement = (
            ""
            if not preference_mean_label or not preference_median_label
            else str(preference_mean_label == preference_median_label).lower()
        )

        report_rows.append(
            {
                "function_name": key[0],
                "configured_cpus": key[1],
                "configured_memory_mb": key[2],
                "mean_x86_samples": (None if mean_x86 is None else mean_x86["sample_count"]),
                "mean_arm_samples": (None if mean_arm is None else mean_arm["sample_count"]),
                "median_x86_samples": (None if median_x86 is None else median_x86["sample_count"]),
                "median_arm_samples": (None if median_arm is None else median_arm["sample_count"]),
                "performance_x86_samples": (
                    None if performance_x86 is None else performance_x86["sample_count"]
                ),
                "performance_arm_samples": (
                    None if performance_arm is None else performance_arm["sample_count"]
                ),
                "preference_mean": preference_mean_label,
                "delta_mean_percent": (
                    None if pref_mean is None else pref_mean["arm_vs_x86_delta_percent"]
                ),
                "preference_median": preference_median_label,
                "delta_median_percent": (
                    None if pref_median is None else pref_median["arm_vs_x86_delta_percent"]
                ),
                "preference_agreement": preference_agreement,
                "status": ("ready" if not issues else "incomplete"),
                "issues": issues,
            }
        )

    ready_rows = [row for row in report_rows if row["status"] == "ready"]

    mean_distribution = preference_distribution(ready_rows, "preference_mean")
    median_distribution = preference_distribution(ready_rows, "preference_median")
    all_mean_classes = all(mean_distribution[label] > 0 for label in PREFERENCE_LABELS)

    all_median_classes = all(median_distribution[label] > 0 for label in PREFERENCE_LABELS)

    transitions = Counter()
    agreement_count = 0
    disagreement_count = 0
    for row in ready_rows:

        source = row["preference_mean"]
        target = row["preference_median"]
        transitions[f"{source} -> {target}"] += 1

        if source == target:
            agreement_count += 1

        else:
            disagreement_count += 1

    other_tags = sorted(
        (
            set(mean_meta["machine_tags"])
            | set(median_meta["machine_tags"])
            | set(performance_meta["machine_tags"])
        )
        - accepted_tags
    )

    structural_ready = bool(report_rows) and len(ready_rows) == len(report_rows)

    result = {
        "schema_version": BENCHMARK_READINESS_SCHEMA_VERSION,
        "readiness_run_id": run_id,
        "sample_count_policy": {"minimum": min_samples, "maximum": max_samples},
        "architectures": {
            "x86_machine_tag": x86_tag,
            "arm_machine_tag": arm_tag,
            "other_machine_tags": other_tags,
        },
        "sources": {
            "function_profiles_mean": mean_meta,
            "function_profiles_median": median_meta,
            "performance_profiles": performance_meta,
            "preferences_mean": preferences_mean_meta,
            "preferences_median": preferences_median_meta,
        },
        "summary": {
            "function_configuration_count": len(report_rows),
            "ready_function_configuration_count": len(ready_rows),
            "incomplete_function_configuration_count": (len(report_rows) - len(ready_rows)),
            "structural_ready": structural_ready,
            "mean_preference_distribution": mean_distribution,
            "median_preference_distribution": median_distribution,
            "all_three_preference_classes_present_mean": all_mean_classes,
            "all_three_preference_classes_present_median": all_median_classes,
            "ready_for_three_class_evaluation": (
                structural_ready and all_mean_classes and all_median_classes
            ),
            "mean_median_preference_agreement_count": agreement_count,
            "mean_median_preference_disagreement_count": disagreement_count,
            "mean_median_preference_transitions": dict(sorted(transitions.items())),
        },
        "functions": report_rows,
    }

    return result


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)

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


def write_function_report(path: Path, run_id: str, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FUNCTION_REPORT_HEADER)
            for row in rows:
                writer.writerow(
                    [
                        BENCHMARK_FUNCTIONS_CSV_SCHEMA_VERSION,
                        run_id,
                        row["function_name"],
                        format(float(row["configured_cpus"]), ".17g"),
                        row["configured_memory_mb"],
                        row["mean_x86_samples"] or "",
                        row["mean_arm_samples"] or "",
                        row["median_x86_samples"] or "",
                        row["median_arm_samples"] or "",
                        row["performance_x86_samples"] or "",
                        row["performance_arm_samples"] or "",
                        row["preference_mean"],
                        (
                            ""
                            if row["delta_mean_percent"] is None
                            else format(float(row["delta_mean_percent"]), ".17g")
                        ),
                        row["preference_median"],
                        (
                            ""
                            if row["delta_median_percent"] is None
                            else format(float(row["delta_median_percent"]), ".17g")
                        ),
                        row["preference_agreement"],
                        row["status"],
                        ";".join(row["issues"]),
                    ]
                )

            handle.flush()

            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def run(args: argparse.Namespace):
    result = build_readiness(
        Path(args.function_profiles_mean),
        Path(args.function_profiles_median),
        Path(args.performance_profiles),
        Path(args.preferences_mean),
        Path(args.preferences_median),
        args.x86_tag,
        args.arm_tag,
        args.run_id,
        args.min_samples,
        args.max_samples,
    )

    atomic_json(Path(args.output_json), result)
    write_function_report(Path(args.output_csv), args.run_id, result["functions"])
    summary = result["summary"]
    print(
        "function_configurations="
        f"{summary['function_configuration_count']} "
        "ready="
        f"{summary['ready_function_configuration_count']} "
        "incomplete="
        f"{summary['incomplete_function_configuration_count']} "
        "structural_ready="
        f"{str(summary['structural_ready']).lower()}"
    )

    print(
        "all_three_classes_mean="
        f"{str(summary['all_three_preference_classes_present_mean']).lower()} "
        "all_three_classes_median="
        f"{str(summary['all_three_preference_classes_present_median']).lower()} "
        "ready_for_three_class_evaluation="
        f"{str(summary['ready_for_three_class_evaluation']).lower()}"
    )

    print(
        "preference_agreement="
        f"{summary['mean_median_preference_agreement_count']} "
        "preference_disagreement="
        f"{summary['mean_median_preference_disagreement_count']}"
    )

    for row in result["functions"]:
        print(
            "[function] "
            f"name={row['function_name']} "
            f"status={row['status']} "
            f"mean={row['preference_mean']} "
            f"median={row['preference_median']} "
            "issues=" + ("none" if not row["issues"] else ",".join(row["issues"]))
        )

    print(f"output_json={args.output_json}")
    print(f"output_csv={args.output_csv}")


def parser():
    root = argparse.ArgumentParser(
        description="Validate structural readiness and architecture coverage of a Serverledge x86/ARM benchmark dataset before scientific clustering evaluation."
    )
    root.add_argument("--function-profiles-mean", required=True)
    root.add_argument("--function-profiles-median", required=True)
    root.add_argument("--performance-profiles", required=True)
    root.add_argument("--preferences-mean", required=True)
    root.add_argument("--preferences-median", required=True)
    root.add_argument("--x86-tag", required=True)
    root.add_argument("--arm-tag", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--min-samples", type=int, default=10)
    root.add_argument("--max-samples", type=int, default=20)
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
