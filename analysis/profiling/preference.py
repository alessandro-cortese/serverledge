#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

INVOCATION_SAMPLE_SCHEMA_VERSION = 3

PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION = 1

ARCHITECTURE_PREFERENCE_CSV_SCHEMA_VERSION = 1

MIN_PERFORMANCE_PROFILE_SAMPLES = 10

MAX_PERFORMANCE_PROFILE_SAMPLES = 20

PREFERENCE_X86 = "x86-preferred"

PREFERENCE_ARM = "arm-preferred"

PREFERENCE_INDEPENDENT = "architecture-independent"


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


ARCHITECTURE_PREFERENCE_HEADER = [
    "architecture_preference_csv_schema_version",
    "performance_profile_csv_schema_version",
    "preference_run_id",
    "performance_run_id",
    "performance_profiles_sha256",
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


@dataclass(frozen=True)
class PerformanceGroupKey:
    function_name: str
    machine_tag: str
    configured_cpus: float
    configured_memory_mb: int


@dataclass(frozen=True)
class PreferencePairKey:
    function_name: str
    configured_cpus: float
    configured_memory_mb: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_finite(value, field: str) -> float:
    try:
        parsed = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"{field} is not finite")

    return parsed


def normalize_input_paths(paths: list[str]) -> list[Path]:
    if not paths:
        raise ValueError("at least one raw " "InvocationSample dataset " "is required")

    normalized: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()

        if path in seen:
            raise ValueError(f"duplicate input " f"dataset {path}")

        if not path.is_file():
            raise ValueError("input dataset does not " "exist or is not a file: " f"{path}")

        seen.add(path)
        normalized.append(path)

    return sorted(normalized)


def discover_raw_datasets(root: Path) -> list[Path]:
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError("input directory does not " "exist or is not a directory: " f"{root}")

    paths = sorted(path.resolve() for path in root.rglob("profiling-samples.jsonl") if path.is_file())

    if not paths:
        raise ValueError("no profiling-samples.jsonl " f"datasets found under {root}")

    return paths


def combined_input_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()

    for file_hash in sorted(sha256_file(path) for path in paths):
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")

    return digest.hexdigest()


def load_invocation_samples(paths: list[Path]) -> list[dict]:
    samples: list[dict] = []
    request_ids: set[str] = set()

    for path in paths:

        with path.open(encoding="utf-8") as handle:

            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} " f"at line {line_number}: " f"{exc}"
                    ) from exc

                if sample.get("schema_version") != INVOCATION_SAMPLE_SCHEMA_VERSION:
                    raise ValueError(
                        "unsupported "
                        "InvocationSample schema "
                        f"in {path} at line "
                        f"{line_number}"
                    )

                request_id = str(sample.get("request_id", "")).strip()

                if not request_id:
                    raise ValueError(f"empty request_id in " f"{path} at line " f"{line_number}")

                if request_id in request_ids:
                    raise ValueError(
                        "duplicate request_id " f"{request_id!r} " "across raw datasets"
                    )

                request_ids.add(request_id)

                samples.append(sample)

    return samples


def validate_performance_sample(sample: dict) -> None:
    request_id = str(sample.get("request_id", "")).strip()
    function_name = str(sample.get("function_name", "")).strip()
    machine_tag = str(sample.get("machine_tag", "")).strip()

    if not request_id or not function_name or not machine_tag:
        raise ValueError("eligible performance sample " "contains empty identity metadata")

    if not sample.get("warm_start", False):
        raise ValueError("eligible performance sample " f"{request_id!r} is not warm")

    if not sample.get("execution_succeeded", False):
        raise ValueError("eligible performance sample " f"{request_id!r} did not succeed")

    eligibility = sample.get("eligibility") or {}

    if not eligibility.get("performance_analysis", False):
        raise ValueError(f"sample {request_id!r} " "is not eligible for " "performance analysis")

    configuration = sample.get("function_configuration") or {}
    cpus = parse_finite(configuration.get("configured_cpus"), "configured_cpus")

    try:
        memory_mb = int(configuration.get("configured_memory_mb"))

    except (TypeError, ValueError) as exc:

        raise ValueError("configured_memory_mb " "is invalid") from exc

    if cpus <= 0 or memory_mb <= 0:
        raise ValueError(
            "eligible performance sample " f"{request_id!r} has " "invalid configuration"
        )

    timing = sample.get("timing") or {}
    duration_ms = parse_finite(timing.get("duration_ms"), "duration_ms")
    response_time_ms = parse_finite(timing.get("response_time_ms"), "response_time_ms")

    if duration_ms < 0 or response_time_ms < 0:
        raise ValueError("eligible performance sample " f"{request_id!r} has " "negative timing")

    if not isinstance(sample.get("timestamp_ms"), int):
        raise ValueError(
            "eligible performance sample " f"{request_id!r} has " "invalid timestamp_ms"
        )


def build_performance_profiles(samples: list[dict], samples_per_profile: int) -> tuple[list[dict], dict]:
    if not (
        MIN_PERFORMANCE_PROFILE_SAMPLES <= samples_per_profile <= MAX_PERFORMANCE_PROFILE_SAMPLES
    ):
        raise ValueError(
            "samples per performance "
            "profile must be between "
            f"{MIN_PERFORMANCE_PROFILE_SAMPLES} "
            "and "
            f"{MAX_PERFORMANCE_PROFILE_SAMPLES}"
        )

    groups: dict[PerformanceGroupKey, list[tuple[int, int, dict]]] = {}
    eligible_count = 0
    ignored_count = 0

    for input_order, sample in enumerate(samples):
        eligibility = sample.get("eligibility") or {}

        if not eligibility.get("performance_analysis", False):
            ignored_count += 1

            continue

        validate_performance_sample(sample)
        eligible_count += 1
        configuration = sample["function_configuration"]

        key = PerformanceGroupKey(
            function_name=str(sample["function_name"]).strip(),
            machine_tag=str(sample["machine_tag"]).strip(),
            configured_cpus=float(configuration["configured_cpus"]),
            configured_memory_mb=int(configuration["configured_memory_mb"]),
        )

        groups.setdefault(key, []).append((int(sample["timestamp_ms"]), input_order, sample))

    profiles: list[dict] = []
    statuses: list[dict] = []

    for key in sorted(
        groups,
        key=lambda item: (
            item.function_name,
            item.machine_tag,
            item.configured_cpus,
            item.configured_memory_mb,
        ),
    ):
        candidates = sorted(groups[key], key=lambda item: (item[0], item[1]))

        status = {
            "function_name": key.function_name,
            "machine_tag": key.machine_tag,
            "configured_cpus": key.configured_cpus,
            "configured_memory_mb": key.configured_memory_mb,
            "eligible_sample_count": len(candidates),
            "selected_sample_count": 0,
            "built": False,
        }

        if len(candidates) < samples_per_profile:
            statuses.append(status)

            continue

        selected_samples = [item[2] for item in candidates[-samples_per_profile:]]
        durations = [float(sample["timing"]["duration_ms"]) for sample in selected_samples]
        responses = [float(sample["timing"]["response_time_ms"]) for sample in selected_samples]
        request_ids = [str(sample["request_id"]) for sample in selected_samples]

        profiles.append(
            {
                "function_name": key.function_name,
                "machine_tag": key.machine_tag,
                "configured_cpus": key.configured_cpus,
                "configured_memory_mb": key.configured_memory_mb,
                "sample_count": samples_per_profile,
                "source_request_ids": request_ids,
                "duration_mean_ms": float(statistics.fmean(durations)),
                "duration_median_ms": float(statistics.median(durations)),
                "response_time_mean_ms": float(statistics.fmean(responses)),
                "response_time_median_ms": float(statistics.median(responses)),
            }
        )

        status["selected_sample_count"] = samples_per_profile
        status["built"] = True
        statuses.append(status)

    return (
        profiles,
        {
            "raw_sample_count": len(samples),
            "eligible_sample_count": eligible_count,
            "ignored_sample_count": ignored_count,
            "groups": statuses,
        },
    )


def classify_delta(delta_percent: float, threshold_percent: float) -> str:
    if not math.isfinite(delta_percent):
        raise ValueError("architecture delta " "must be finite")

    if not math.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("threshold percent must be " "finite and non-negative")

    if delta_percent > threshold_percent:
        return PREFERENCE_X86

    if delta_percent < -threshold_percent:
        return PREFERENCE_ARM

    return PREFERENCE_INDEPENDENT


def build_architecture_preferences(profiles: list[dict], x86_tag: str, arm_tag: str, aggregation: str, threshold_percent: float) -> tuple[list[dict], list[dict]]:
    x86_tag = x86_tag.strip()
    arm_tag = arm_tag.strip()

    if not x86_tag or not arm_tag:
        raise ValueError("x86 and ARM machine tags " "must be non-empty")

    if x86_tag == arm_tag:
        raise ValueError("x86 and ARM machine tags " "must be different")

    if aggregation not in ("mean", "median"):
        raise ValueError("unsupported performance " f"aggregation {aggregation!r}")

    if not math.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("threshold percent must be " "finite and non-negative")

    by_identity: dict[tuple[str, str, float, int], dict] = {}
    pair_keys: set[PreferencePairKey] = set()

    for profile in profiles:
        identity = (
            profile["function_name"],
            profile["machine_tag"],
            float(profile["configured_cpus"]),
            int(profile["configured_memory_mb"]),
        )

        if identity in by_identity:
            raise ValueError("duplicate performance " "profile identity " f"{identity}")

        by_identity[identity] = profile

        if profile["machine_tag"] in (x86_tag, arm_tag):
            pair_keys.add(
                PreferencePairKey(
                    function_name=profile["function_name"],
                    configured_cpus=float(profile["configured_cpus"]),
                    configured_memory_mb=int(profile["configured_memory_mb"]),
                )
            )

    preferences: list[dict] = []
    statuses: list[dict] = []
    metric_field = f"duration_{aggregation}_ms"

    for key in sorted(
        pair_keys,
        key=lambda item: (item.function_name, item.configured_cpus, item.configured_memory_mb),
    ):
        x86_profile = by_identity.get((key.function_name, x86_tag, key.configured_cpus, key.configured_memory_mb))
        arm_profile = by_identity.get((key.function_name, arm_tag, key.configured_cpus, key.configured_memory_mb))

        status = {
            "function_name": key.function_name,
            "configured_cpus": key.configured_cpus,
            "configured_memory_mb": key.configured_memory_mb,
            "x86_present": x86_profile is not None,
            "arm_present": arm_profile is not None,
            "built": False,
        }

        if x86_profile is None or arm_profile is None:
            statuses.append(status)

            continue

        if int(x86_profile["sample_count"]) != int(arm_profile["sample_count"]):
            raise ValueError(
                "unbalanced x86/ARM " "sample counts for " f"function " f"{key.function_name!r}"
            )

        x86_duration = float(x86_profile[metric_field])
        arm_duration = float(arm_profile[metric_field])

        if x86_duration <= 0:
            raise ValueError(
                "cannot compute "
                "architecture delta for "
                f"{key.function_name!r}: "
                "x86 duration must be "
                "positive"
            )

        delta_percent = ((arm_duration - x86_duration) / x86_duration) * 100.0

        preferences.append(
            {
                "function_name": key.function_name,
                "configured_cpus": key.configured_cpus,
                "configured_memory_mb": key.configured_memory_mb,
                "aggregation": aggregation,
                "performance_metric": "duration_ms",
                "threshold_percent": float(threshold_percent),
                "x86_machine_tag": x86_tag,
                "arm_machine_tag": arm_tag,
                "x86_sample_count": int(x86_profile["sample_count"]),
                "arm_sample_count": int(arm_profile["sample_count"]),
                "x86_duration_ms": x86_duration,
                "arm_duration_ms": arm_duration,
                "arm_vs_x86_delta_percent": float(delta_percent),
                "architecture_preference": classify_delta(delta_percent, threshold_percent),
            }
        )

        status["built"] = True
        statuses.append(status)

    return preferences, statuses


def atomic_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(prefix=(f".{path.name}."), suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

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


def write_performance_profiles(path: Path, run_id: str, input_sha256: str, profiles: list[dict]) -> None:
    run_id = run_id.strip()

    if not run_id:
        raise ValueError("performance run ID " "cannot be empty")

    if not profiles:
        raise ValueError("cannot export an empty " "performance profile dataset")

    rows = [
        [
            PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION,
            run_id,
            input_sha256,
            profile["function_name"],
            profile["machine_tag"],
            format(float(profile["configured_cpus"]), ".17g"),
            int(profile["configured_memory_mb"]),
            int(profile["sample_count"]),
            json.dumps(profile["source_request_ids"], separators=(",", ":")),
            format(float(profile["duration_mean_ms"]), ".17g"),
            format(float(profile["duration_median_ms"]), ".17g"),
            format(float(profile["response_time_mean_ms"]), ".17g"),
            format(float(profile["response_time_median_ms"]), ".17g"),
        ]
        for profile in profiles
    ]

    atomic_csv(path, PERFORMANCE_PROFILE_HEADER, rows)


def load_performance_profiles(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != PERFORMANCE_PROFILE_HEADER:
            raise ValueError("unexpected performance " "profile CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("performance profile CSV " "contains no data rows")

    run_ids: set[str] = set()
    input_hashes: set[str] = set()
    identities: set[tuple[str, str, float, int]] = set()
    profiles: list[dict] = []

    for row_number, row in enumerate(rows, start=2):
        if row["performance_profile_csv_schema_version"] != str(
            PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION
        ):
            raise ValueError(f"row {row_number}: " "unsupported performance " "profile CSV schema")

        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()
        run_id = row["performance_run_id"].strip()
        input_hash = row["input_sha256"].strip()
        cpus = parse_finite(row["configured_cpus"], "configured_cpus")

        try:
            memory_mb = int(row["configured_memory_mb"])
            sample_count = int(row["sample_count"])
            request_ids = json.loads(row["source_request_ids_json"])

        except (ValueError, json.JSONDecodeError) as exc:

            raise ValueError(f"row {row_number}: " "invalid performance " "profile metadata") from exc

        if not function_name or not machine_tag or not run_id or not input_hash:
            raise ValueError(f"row {row_number}: " "empty performance " "profile metadata")

        if cpus <= 0 or memory_mb <= 0 or sample_count <= 0:
            raise ValueError(f"row {row_number}: " "invalid performance " "profile configuration")

        if not isinstance(request_ids, list) or len(request_ids) != sample_count:
            raise ValueError(f"row {row_number}: " "source request IDs do " "not match sample count")

        if len(set(request_ids)) != len(request_ids) or any(not str(item).strip() for item in request_ids):
            raise ValueError(f"row {row_number}: " "invalid source " "request IDs")

        identity = (function_name, machine_tag, cpus, memory_mb)

        if identity in identities:
            raise ValueError(f"row {row_number}: " "duplicate performance " "profile identity " f"{identity}")

        identities.add(identity)

        profile = {
            "function_name": function_name,
            "machine_tag": machine_tag,
            "configured_cpus": cpus,
            "configured_memory_mb": memory_mb,
            "sample_count": sample_count,
            "source_request_ids": request_ids,
        }

        for field in (
            "duration_mean_ms",
            "duration_median_ms",
            "response_time_mean_ms",
            "response_time_median_ms",
        ):
            value = parse_finite(row[field], field)

            if value < 0:
                raise ValueError(f"row {row_number}: " f"{field} is negative")

            profile[field] = value

        profiles.append(profile)
        run_ids.add(run_id)
        input_hashes.add(input_hash)

    if len(run_ids) != 1 or len(input_hashes) != 1:
        raise ValueError("performance profile CSV " "mixes provenance metadata")

    return profiles, {"performance_run_id": run_ids.pop(), "input_sha256": input_hashes.pop()}


def write_architecture_preferences(
    path: Path,
    preference_run_id: str,
    performance_run_id: str,
    performance_profiles_sha256: str,
    preferences: list[dict],
) -> None:
    preference_run_id = preference_run_id.strip()

    if not preference_run_id:
        raise ValueError("preference run ID " "cannot be empty")

    if not preferences:
        raise ValueError("cannot export an empty " "architecture preference " "dataset")

    rows = [
        [
            ARCHITECTURE_PREFERENCE_CSV_SCHEMA_VERSION,
            PERFORMANCE_PROFILE_CSV_SCHEMA_VERSION,
            preference_run_id,
            performance_run_id,
            performance_profiles_sha256,
            item["function_name"],
            format(float(item["configured_cpus"]), ".17g"),
            int(item["configured_memory_mb"]),
            item["aggregation"],
            item["performance_metric"],
            format(float(item["threshold_percent"]), ".17g"),
            item["x86_machine_tag"],
            item["arm_machine_tag"],
            int(item["x86_sample_count"]),
            int(item["arm_sample_count"]),
            format(float(item["x86_duration_ms"]), ".17g"),
            format(float(item["arm_duration_ms"]), ".17g"),
            format(float(item["arm_vs_x86_delta_percent"]), ".17g"),
            item["architecture_preference"],
        ]
        for item in preferences
    ]

    atomic_csv(path, ARCHITECTURE_PREFERENCE_HEADER, rows)


def resolve_raw_inputs(input_paths: list[str], input_dir: str | None) -> list[Path]:
    if input_paths and input_dir:
        raise ValueError("use either --input " "or --input-dir, not both")

    if input_paths:
        return normalize_input_paths(input_paths)

    if input_dir:
        return discover_raw_datasets(Path(input_dir))

    raise ValueError("one of --input or " "--input-dir is required")


def run_profiles(args: argparse.Namespace) -> None:
    paths = resolve_raw_inputs(args.input, args.input_dir)
    samples = load_invocation_samples(paths)
    profiles, summary = build_performance_profiles(samples, args.samples)

    if not profiles:

        for group in summary["groups"]:
            print(
                f"[skip] "
                f"function="
                f"{group['function_name']} "
                f"machine_tag="
                f"{group['machine_tag']} "
                f"cpus="
                f"{group['configured_cpus']} "
                f"memory_mb="
                f"{group['configured_memory_mb']} "
                f"eligible="
                f"{group['eligible_sample_count']} "
                f"required={args.samples}"
            )

        raise ValueError(
            "no complete performance "
            "groups: need at least "
            f"{args.samples} eligible "
            "samples per group"
        )

    input_hash = combined_input_sha256(paths)
    write_performance_profiles(Path(args.output), args.run_id, input_hash, profiles)

    print(
        f"input_datasets={len(paths)} "
        f"raw_samples="
        f"{summary['raw_sample_count']} "
        f"eligible_performance_samples="
        f"{summary['eligible_sample_count']} "
        f"ignored_samples="
        f"{summary['ignored_sample_count']} "
        f"groups="
        f"{len(summary['groups'])} "
        f"profiles={len(profiles)} "
        f"samples_per_profile="
        f"{args.samples}"
    )

    for path in paths:
        print(f"[input] {path}")

    for group in summary["groups"]:
        status = "built" if group["built"] else "skipped"

        print(
            f"[{status}] "
            f"function="
            f"{group['function_name']} "
            f"machine_tag="
            f"{group['machine_tag']} "
            f"cpus="
            f"{group['configured_cpus']} "
            f"memory_mb="
            f"{group['configured_memory_mb']} "
            f"eligible="
            f"{group['eligible_sample_count']} "
            f"selected="
            f"{group['selected_sample_count']}"
        )

    print(f"input_sha256={input_hash}")
    print(f"output={args.output}")


def run_preferences(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    profiles, provenance = load_performance_profiles(input_path)
    preferences, statuses = build_architecture_preferences(profiles, args.x86_tag, args.arm_tag, args.aggregation, args.threshold_percent)

    if not preferences:

        for status in statuses:
            print(
                f"[skip] "
                f"function="
                f"{status['function_name']} "
                f"cpus="
                f"{status['configured_cpus']} "
                f"memory_mb="
                f"{status['configured_memory_mb']} "
                f"x86_present="
                f"{str(status['x86_present']).lower()} "
                f"arm_present="
                f"{str(status['arm_present']).lower()}"
            )

        raise ValueError("no complete x86/ARM " "performance pairs were found")

    profile_hash = sha256_file(input_path)
    write_architecture_preferences(Path(args.output), args.run_id, provenance["performance_run_id"], profile_hash, preferences)

    print(
        f"profiles={len(profiles)} "
        f"pairs={len(preferences)} "
        f"aggregation={args.aggregation} "
        f"metric=duration_ms "
        f"threshold_percent="
        f"{args.threshold_percent} "
        f"x86_tag={args.x86_tag} "
        f"arm_tag={args.arm_tag}"
    )

    for item in preferences:
        print(
            f"[preference] "
            f"function="
            f"{item['function_name']} "
            f"x86_ms="
            f"{item['x86_duration_ms']} "
            f"arm_ms="
            f"{item['arm_duration_ms']} "
            f"delta_percent="
            f"{item['arm_vs_x86_delta_percent']} "
            f"label="
            f"{item['architecture_preference']}"
        )

    print("performance_profiles_sha256=" f"{profile_hash}")
    print(f"output={args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Build Serverledge "
            "performance profiles and "
            "x86/ARM architecture "
            "preference ground truth."
        )
    )

    commands = root.add_subparsers(dest="command", required=True)

    profiles = commands.add_parser(
        "profiles",
        help=(
            "build per-function/"
            "per-machine performance "
            "profiles from raw "
            "InvocationSample JSONL "
            "datasets"
        ),
    )

    profiles.add_argument("--input", action="append", default=[])
    profiles.add_argument("--input-dir")
    profiles.add_argument("--run-id", required=True)
    profiles.add_argument("--samples", type=int, required=True)
    profiles.add_argument("--output", required=True)
    profiles.set_defaults(func=run_profiles)

    preferences = commands.add_parser(
        "preferences",
        help=(
            "pair x86 and ARM "
            "performance profiles "
            "and derive architecture "
            "preference labels"
        ),
    )

    preferences.add_argument("--input", required=True)
    preferences.add_argument("--run-id", required=True)
    preferences.add_argument("--x86-tag", required=True)
    preferences.add_argument("--arm-tag", required=True)
    preferences.add_argument("--aggregation", required=True, choices=("mean", "median"))
    preferences.add_argument("--threshold-percent", type=float, required=True)
    preferences.add_argument("--output", required=True)
    preferences.set_defaults(func=run_preferences)

    return root


def main() -> None:
    root = parser()
    args = root.parse_args()

    try:
        args.func(args)

    except (ValueError, OSError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
