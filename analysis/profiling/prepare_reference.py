#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import sklearn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from analysis.profiling import preprocess

REFERENCE_MANIFEST_SCHEMA_VERSION = 1


def validate_scalers(requested: list[str]) -> list[str]:
    if not requested:
        return list(preprocess.SCALERS)

    result: list[str] = []
    seen: set[str] = set()

    for scaler in requested:
        scaler = scaler.strip()

        if scaler not in preprocess.SCALERS:
            raise ValueError("unsupported scaler " f"{scaler!r}")

        if scaler in seen:
            raise ValueError("duplicate scaler " f"{scaler!r}")

        seen.add(scaler)
        result.append(scaler)

    return result


def select_machine_rows(rows: list[dict[str, str]], machine_tag: str) -> list[dict[str, str]]:
    machine_tag = machine_tag.strip()

    if not machine_tag:
        raise ValueError("machine tag " "cannot be empty")

    available_tags = sorted({row["machine_tag"].strip() for row in rows})
    selected = [row for row in rows if (row["machine_tag"].strip() == machine_tag)]

    if not selected:
        raise ValueError(
            "no FunctionProfile rows "
            f"for machine tag "
            f"{machine_tag!r}; "
            "available tags: "
            f"{available_tags}"
        )

    return selected


def atomic_source_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty " "reference source dataset")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=(f".{path.name}."), suffix=".tmp", dir=path.parent)

    temp_path = Path(temp_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:

            writer = csv.writer(handle)
            writer.writerow(preprocess.SOURCE_HEADER)

            for row in rows:
                writer.writerow([row[field] for field in preprocess.SOURCE_HEADER])

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def prepare_reference(input_path: Path, machine_tag: str, run_id: str, output_dir: Path, requested_scalers: list[str]) -> dict:
    run_id = run_id.strip()
    machine_tag = machine_tag.strip()

    if not run_id:
        raise ValueError("reference run ID " "cannot be empty")

    if not machine_tag:
        raise ValueError("machine tag " "cannot be empty")

    scalers = validate_scalers(requested_scalers)
    input_path = input_path.expanduser().resolve()

    if not input_path.is_file():
        raise ValueError("input FunctionProfile CSV " "does not exist: " f"{input_path}")

    output_dir = output_dir.expanduser().resolve()
    source_rows, _, source_experiment_id, source_aggregation = preprocess.load_source(input_path)
    selected_rows = select_machine_rows(source_rows, machine_tag)
    source_dir = output_dir / "source"

    filtered_source_path = source_dir / "function-profiles-" f"{source_aggregation}-" f"{machine_tag}.csv"

    atomic_source_csv(filtered_source_path, selected_rows)

    # Reload the newly written architecture-specific
    # dataset. From this point onward it is the actual
    # reference source for preprocessing and clustering.
    filtered_rows, filtered_matrix, filtered_experiment_id, filtered_aggregation = (preprocess.load_source(filtered_source_path))

    if filtered_experiment_id != source_experiment_id:
        raise RuntimeError("filtered reference changed " "experiment_id")

    if filtered_aggregation != source_aggregation:
        raise RuntimeError("filtered reference changed " "aggregation")

    if any(row["machine_tag"].strip() != machine_tag for row in filtered_rows):
        raise RuntimeError("filtered reference contains " "another machine tag")

    preprocessing_outputs = {}

    for scaler in scalers:
        scaler_dir = output_dir / "preprocessing" / scaler
        model_path = scaler_dir / "preprocessing-model.json"
        csv_path = scaler_dir / "preprocessed.csv"
        transformed, parameters, state = preprocess.fit_transform(scaler, filtered_matrix)

        model = preprocess.build_model(
            filtered_source_path,
            filtered_experiment_id,
            filtered_aggregation,
            len(filtered_rows),
            scaler,
            parameters,
            state,
        )

        preprocess.write_model(model_path, model)
        preprocess.write_csv(csv_path, filtered_rows, transformed, model)

        # Independent replay from the serialized
        # representation, exactly as required by 09B.
        replay_model = preprocess.load_model(model_path)
        replay = preprocess.apply_model(filtered_matrix, replay_model, filtered_aggregation)

        if replay.shape != transformed.shape:
            raise RuntimeError("preprocessing replay " "shape mismatch")

        # The affine model is canonical, therefore
        # the numerical result must be identical.
        if not (replay == transformed).all():
            raise RuntimeError("serialized preprocessing " "model does not reproduce " "reference transform")

        preprocessing_outputs[scaler] = {
            "model": str(model_path),
            "preprocessed_csv": str(csv_path),
            "fit_sample_count": len(filtered_rows),
            "model_source_sha256": model["source_sha256"],
        }

    manifest = {
        "schema_version": REFERENCE_MANIFEST_SCHEMA_VERSION,
        "reference_run_id": run_id,
        "sklearn_version": sklearn.__version__,
        "source_input": {
            "path": str(input_path),
            "sha256": preprocess.sha256_file(input_path),
            "row_count": len(source_rows),
            "experiment_id": source_experiment_id,
            "aggregation": source_aggregation,
        },
        "reference": {
            "machine_tag": machine_tag,
            "source_path": str(filtered_source_path),
            "source_sha256": preprocess.sha256_file(filtered_source_path),
            "row_count": len(filtered_rows),
            "experiment_id": filtered_experiment_id,
            "aggregation": filtered_aggregation,
            "feature_names": preprocess.FEATURE_NAMES,
        },
        "scalers": scalers,
        "preprocessing": preprocessing_outputs,
    }

    manifest_path = output_dir / "reference-manifest.json"
    preprocess.atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)

    return manifest


def run(args: argparse.Namespace) -> None:
    manifest = prepare_reference(Path(args.input), args.machine_tag, args.run_id, Path(args.output_dir), args.scaler)

    reference = manifest["reference"]

    print(
        "source_rows="
        f"{manifest['source_input']['row_count']} "
        "selected_rows="
        f"{reference['row_count']} "
        "aggregation="
        f"{reference['aggregation']} "
        "machine_tag="
        f"{reference['machine_tag']} "
        "scalers="
        f"{len(manifest['scalers'])}"
    )

    print("source_sha256=" f"{reference['source_sha256']}")

    for scaler in manifest["scalers"]:
        item = manifest["preprocessing"][scaler]

        print(
            "[prepared] "
            f"scaler={scaler} "
            "rows="
            f"{item['fit_sample_count']} "
            "model="
            f"{item['model']} "
            "csv="
            f"{item['preprocessed_csv']}"
        )

    print("manifest=" f"{manifest['manifest_path']}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Prepare one architecture-"
            "specific Serverledge "
            "FunctionProfile reference "
            "dataset and fit its "
            "preprocessing models."
        )
    )

    root.add_argument("--input", required=True)
    root.add_argument("--machine-tag", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--output-dir", required=True)

    root.add_argument(
        "--scaler",
        action="append",
        default=[],
        choices=(preprocess.SCALERS),
        help=(
            "scaler to prepare; "
            "repeat the option for "
            "multiple scalers. "
            "If omitted, all supported "
            "scalers are prepared."
        ),
    )

    return root


def main() -> None:
    root = parser()

    try:
        run(root.parse_args())

    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:

        root.error(str(exc))


if __name__ == "__main__":
    main()
