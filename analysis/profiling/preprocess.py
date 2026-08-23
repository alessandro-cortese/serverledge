#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

MODEL_SCHEMA_VERSION = 1

PREPROCESSED_CSV_SCHEMA_VERSION = 1

SOURCE_CSV_SCHEMA_VERSION = 1

FUNCTION_PROFILE_SCHEMA_VERSION = 1

SCALERS = ("none", "standard", "robust", "minmax")

FEATURE_NAMES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]


SOURCE_METADATA = [
    "csv_schema_version",
    "experiment_id",
    "aggregation",
    "function_profile_schema_version",
    "function_name",
    "machine_tag",
    "configured_cpus",
    "configured_memory_mb",
    "sample_count",
]

SOURCE_HEADER = SOURCE_METADATA + FEATURE_NAMES

OUTPUT_METADATA = [
    "preprocessing_schema_version",
    "preprocessing_model_schema_version",
    "source_csv_schema_version",
    "experiment_id",
    "fit_experiment_id",
    "aggregation",
    "scaler",
    "function_profile_schema_version",
    "function_name",
    "machine_tag",
    "configured_cpus",
    "configured_memory_mb",
    "sample_count",
]

OUTPUT_HEADER = OUTPUT_METADATA + FEATURE_NAMES

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_finite(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)

    except ValueError as exc:
        raise ValueError(f"row {row_number}: " f"{field} is not numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: " f"{field} is not finite")

    return result


def load_source(path: Path) -> tuple[list[dict[str, str]], np.ndarray, str, str]:
    with path.open(newline="", encoding="utf-8") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != SOURCE_HEADER:
            raise ValueError("unexpected " "FunctionProfile CSV header")

        rows = list(reader)

    if not rows:
        raise ValueError("FunctionProfile CSV " "contains no data rows")

    experiment_ids: set[str] = set()
    aggregations: set[str] = set()
    identities: set[tuple[str, str, float, int]] = set()
    matrix: list[list[float]] = []

    for row_number, row in enumerate(rows, start=2):

        if row["csv_schema_version"] != str(SOURCE_CSV_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported " "csv_schema_version")

        if row["function_profile_schema_version"] != str(FUNCTION_PROFILE_SCHEMA_VERSION):
            raise ValueError(f"row {row_number}: " "unsupported " "function_profile_schema_version")

        experiment_id = row["experiment_id"].strip()
        aggregation = row["aggregation"].strip()
        function_name = row["function_name"].strip()
        machine_tag = row["machine_tag"].strip()

        if not experiment_id or not function_name or not machine_tag:
            raise ValueError(f"row {row_number}: " "empty metadata field")

        if aggregation not in ("mean", "median"):
            raise ValueError(f"row {row_number}: " "unsupported aggregation " f"{aggregation!r}")

        cpus = parse_finite(row["configured_cpus"], "configured_cpus", row_number)

        try:
            memory_mb = int(row["configured_memory_mb"])

            sample_count = int(row["sample_count"])

        except ValueError as exc:
            raise ValueError(f"row {row_number}: " "invalid integer metadata") from exc

        if cpus <= 0 or memory_mb <= 0 or sample_count <= 0:
            raise ValueError(f"row {row_number}: " "resource metadata " "must be positive")

        identity = (function_name, machine_tag, cpus, memory_mb)

        if identity in identities:
            raise ValueError(
                f"row {row_number}: " "duplicate " "FunctionProfile identity " f"{identity}"
            )

        identities.add(identity)

        values = [parse_finite(row[name], name, row_number) for name in FEATURE_NAMES]

        if any(value < 0 for value in values):
            raise ValueError(f"row {row_number}: " "raw profiling features " "must be non-negative")

        experiment_ids.add(experiment_id)
        aggregations.add(aggregation)
        matrix.append(values)

    if len(experiment_ids) != 1:

        raise ValueError("input CSV mixes " "experiment_id values")

    if len(aggregations) != 1:

        raise ValueError("input CSV mixes " "mean and median rows")

    return rows, np.asarray(matrix, dtype=np.float64), experiment_ids.pop(), aggregations.pop()


def fit_transform(name: str, matrix: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | str]], dict]:
    if name == "none":

        transformed = matrix.copy()
        multiplier = np.ones(matrix.shape[1])
        offset = np.zeros(matrix.shape[1])
        state = {}

    elif name == "standard":

        scaler = StandardScaler()
        transformed = scaler.fit_transform(matrix)
        multiplier = 1.0 / scaler.scale_
        offset = -scaler.mean_ / scaler.scale_

        state = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(),
        }

    elif name == "robust":

        scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
        transformed = scaler.fit_transform(matrix)
        multiplier = 1.0 / scaler.scale_
        offset = -scaler.center_ / scaler.scale_

        state = {
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
            "quantile_range": [25.0, 75.0],
            "unit_variance": False,
        }

    elif name == "minmax":

        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
        transformed = scaler.fit_transform(matrix)
        multiplier = scaler.scale_
        offset = scaler.min_
        state = {
            "feature_range": [0.0, 1.0],
            "clip": False,
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "data_range": scaler.data_range_.tolist(),
        }

    else:
        raise ValueError(f"unsupported scaler " f"{name!r}")

    # Every supported transformation is saved as:
    #
    # x_scaled = x_raw * multiplier + offset
    #
    # This representation is deliberately library-independent and can
    # therefore be reused later from Go during transfer learning.
    affine = matrix * multiplier + offset

    if not np.allclose(transformed, affine, rtol=1e-12, atol=1e-12):
        raise RuntimeError("serialized transform " "does not reproduce " "scikit-learn output")

    # From this point on, the serialized affine representation is the
    # canonical preprocessing contract. Using it also for the reference
    # dataset guarantees that a later replay produces byte-identical output.
    transformed = affine

    if not np.isfinite(transformed).all():
        raise ValueError("preprocessing produced " "non-finite values")

    parameters = [
        {"feature": feature, "multiplier": float(multiplier[index]), "offset": float(offset[index])}
        for (index, feature) in enumerate(FEATURE_NAMES)
    ]

    return transformed, parameters, state


def build_model(
    source_path: Path,
    experiment_id: str,
    aggregation: str,
    sample_count: int,
    scaler: str,
    parameters: list[dict[str, float | str]],
    state: dict,
) -> dict:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "preprocessed_csv_schema_version": PREPROCESSED_CSV_SCHEMA_VERSION,
        "source_csv_schema_version": SOURCE_CSV_SCHEMA_VERSION,
        "function_profile_schema_version": FUNCTION_PROFILE_SCHEMA_VERSION,
        "fit_experiment_id": experiment_id,
        "aggregation": aggregation,
        "scaler": scaler,
        "sklearn_version": sklearn.__version__,
        "feature_names": FEATURE_NAMES,
        "fit_sample_count": sample_count,
        "source_sha256": sha256_file(source_path),
        "affine_transform": ("x_scaled = " "x_raw * multiplier " "+ offset"),
        "affine_parameters": parameters,
        "scaler_state": state,
    }


def load_model(path: Path) -> dict:
    model = json.loads(path.read_text(encoding="utf-8"))

    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported preprocessing " "model schema")

    if model.get("preprocessed_csv_schema_version") != PREPROCESSED_CSV_SCHEMA_VERSION:
        raise ValueError("unsupported preprocessed " "CSV schema in model")

    if model.get("source_csv_schema_version") != SOURCE_CSV_SCHEMA_VERSION:
        raise ValueError("unsupported source " "CSV schema in model")

    if model.get("function_profile_schema_version") != FUNCTION_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported FunctionProfile " "schema in model")

    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("preprocessing model " "feature order mismatch")

    if model.get("aggregation") not in ("mean", "median"):
        raise ValueError("invalid aggregation " "in preprocessing model")

    if model.get("scaler") not in SCALERS:
        raise ValueError("invalid scaler " "in preprocessing model")

    parameters = model.get("affine_parameters")

    if not isinstance(parameters, list) or len(parameters) != len(FEATURE_NAMES):
        raise ValueError("invalid affine parameters " "in preprocessing model")

    for index, feature in enumerate(FEATURE_NAMES):
        parameter = parameters[index]

        if parameter.get("feature") != feature:
            raise ValueError("affine parameter " "feature order mismatch")

        for field in ("multiplier", "offset"):
            value = float(parameter[field])

            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} " f"for feature {feature}")

    return model


def apply_model(matrix: np.ndarray, model: dict, aggregation: str) -> np.ndarray:
    if aggregation != model["aggregation"]:
        raise ValueError("aggregation mismatch: " f"input={aggregation}, " f"model=" f"{model['aggregation']}")

    multiplier = np.asarray([parameter["multiplier"] for parameter in model["affine_parameters"]], dtype=np.float64)
    offset = np.asarray([parameter["offset"] for parameter in model["affine_parameters"]], dtype=np.float64)
    transformed = matrix * multiplier + offset

    if not np.isfinite(transformed).all():
        raise ValueError("preprocessing model " "produced non-finite values")

    return transformed


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_name = tempfile.mkstemp(prefix=(f".{path.name}."), suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:

            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_model(path: Path, model: dict) -> None:
    atomic_text(path, json.dumps(model, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], matrix: np.ndarray, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=(f".{path.name}."), suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:

            writer = csv.writer(handle)
            writer.writerow(OUTPUT_HEADER)

            for row, values in zip(rows, matrix, strict=True):
                writer.writerow(
                    [
                        PREPROCESSED_CSV_SCHEMA_VERSION,
                        MODEL_SCHEMA_VERSION,
                        SOURCE_CSV_SCHEMA_VERSION,
                        row["experiment_id"],
                        model["fit_experiment_id"],
                        row["aggregation"],
                        model["scaler"],
                        row["function_profile_schema_version"],
                        row["function_name"],
                        row["machine_tag"],
                        row["configured_cpus"],
                        row["configured_memory_mb"],
                        row["sample_count"],
                        *[format(float(value), ".17g") for value in values],
                    ]
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_fit(args: argparse.Namespace) -> None:
    source_path = Path(args.input)
    rows, matrix, experiment_id, aggregation = load_source(source_path)
    transformed, parameters, state = fit_transform(args.scaler, matrix)
    model = build_model(source_path, experiment_id, aggregation, len(rows), args.scaler, parameters, state)
    write_model(Path(args.model), model)
    write_csv(Path(args.output), rows, transformed, model)

    print(
        f"rows={len(rows)} "
        f"features={len(FEATURE_NAMES)} "
        f"aggregation={aggregation} "
        f"scaler={args.scaler}"
    )

    print("source_sha256=" f"{model['source_sha256']}")
    print(f"model={args.model}")
    print(f"output={args.output}")


def run_transform(args: argparse.Namespace) -> None:
    rows, matrix, _, aggregation = load_source(Path(args.input))
    model = load_model(Path(args.model))
    transformed = apply_model(matrix, model, aggregation)
    write_csv(Path(args.output), rows, transformed, model)

    print(
        f"rows={len(rows)} "
        f"features={len(FEATURE_NAMES)} "
        f"aggregation={aggregation} "
        f"scaler={model['scaler']} "
        "fit_experiment_id="
        f"{model['fit_experiment_id']}"
    )

    print(f"model={args.model}")
    print(f"output={args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=("Preprocess Serverledge " "FunctionProfile CSV datasets."))
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-transform",help=("fit a scaler, save its " "JSON model, and transform " "the reference dataset"))
    fit.add_argument("--input", required=True)
    fit.add_argument("--scaler", required=True, choices=SCALERS)
    fit.add_argument("--model", required=True)
    fit.add_argument("--output", required=True)
    fit.set_defaults(func=run_fit)
    transform = commands.add_parser("transform", help=("transform a new dataset " "with an already fitted " "JSON model"))
    transform.add_argument("--input", required=True)
    transform.add_argument("--model", required=True)
    transform.add_argument("--output", required=True)
    transform.set_defaults(func=run_transform)

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
