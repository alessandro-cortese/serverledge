#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "coupled_vs_baseline_gain_pct",
    "decoupled_vs_baseline_gain_pct",
    "decoupled_vs_coupled_gain_pct",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"File non trovato: {path}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"Nessuna riga da scrivere: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("Lista valori vuota")

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "positive_replicas": sum(value > 0 for value in values),
        "negative_replicas": sum(value < 0 for value in values),
        "zero_replicas": sum(value == 0 for value in values),
    }


def analyze(args: argparse.Namespace) -> None:
    input_paths = [
        Path(path).expanduser().resolve()
        for path in args.replica_analysis
    ]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [load_json(path) for path in input_paths]

    targets = {document["target_function"] for document in documents}

    if len(targets) != 1:
        raise SystemExit(
            f"Le repliche appartengono a target diversi: {sorted(targets)}"
        )

    target = next(iter(targets))

    replica_rows = []

    for document in documents:
        replica_id = document["replica_id"]
        donor = document["selected_donor"]

        for horizon, values in document["horizons"].items():
            replica_rows.append(
                {
                    "replica_id": replica_id,
                    "target_function": target,
                    "selected_donor": donor,
                    "horizon": int(horizon),

                    "baseline_cumulative_ms":
                        float(values["baseline_cumulative_ms"]),

                    "coupled_cumulative_ms":
                        float(values["coupled_cumulative_ms"]),

                    "decoupled_cumulative_ms":
                        float(values["decoupled_cumulative_ms"]),

                    "coupled_vs_baseline_gain_pct":
                        float(values["coupled_vs_baseline_gain_pct"]),

                    "decoupled_vs_baseline_gain_pct":
                        float(values["decoupled_vs_baseline_gain_pct"]),

                    "decoupled_vs_coupled_gain_pct":
                        float(values["decoupled_vs_coupled_gain_pct"]),

                    "baseline_amd64":
                        int(values["baseline_amd64"]),

                    "baseline_arm64":
                        int(values["baseline_arm64"]),

                    "coupled_amd64":
                        int(values["coupled_amd64"]),

                    "coupled_arm64":
                        int(values["coupled_arm64"]),

                    "decoupled_amd64":
                        int(values["decoupled_amd64"]),

                    "decoupled_arm64":
                        int(values["decoupled_arm64"]),
                }
            )

    replica_rows.sort(
        key=lambda row: (row["horizon"], row["replica_id"])
    )

    horizons = sorted({row["horizon"] for row in replica_rows})
    summary_rows = []
    summary_json = {}

    for horizon in horizons:
        horizon_rows = [
            row
            for row in replica_rows
            if row["horizon"] == horizon
        ]

        summary_json[str(horizon)] = {}

        for metric in METRICS:
            values = [
                float(row[metric])
                for row in horizon_rows
            ]

            stats = summarize(values)

            summary_rows.append(
                {
                    "target_function": target,
                    "horizon": horizon,
                    "metric": metric,
                    "n": stats["n"],
                    "mean_pct": stats["mean"],
                    "median_pct": stats["median"],
                    "stdev_pct": stats["stdev"],
                    "min_pct": stats["min"],
                    "max_pct": stats["max"],
                    "positive_replicas": stats["positive_replicas"],
                    "negative_replicas": stats["negative_replicas"],
                    "zero_replicas": stats["zero_replicas"],
                }
            )

            summary_json[str(horizon)][metric] = stats

    donors = {
        document["replica_id"]: document["selected_donor"]
        for document in documents
    }

    donor_counts = {}

    for donor in donors.values():
        donor_counts[donor] = donor_counts.get(donor, 0) + 1

    all_rows_csv = output_dir / "all-replica-horizons.csv"
    summary_csv = output_dir / "multi-replica-summary.csv"
    summary_json_path = output_dir / "multi-replica-analysis.json"

    write_csv(all_rows_csv, replica_rows)
    write_csv(summary_csv, summary_rows)

    result = {
        "schema_version": 1,
        "target_function": target,
        "replica_count": len(documents),
        "replicas": [
            document["replica_id"]
            for document in documents
        ],
        "donor_by_replica": donors,
        "donor_counts": donor_counts,
        "horizons": summary_json,
        "artifacts": {
            "all_replica_horizons_csv": str(all_rows_csv),
            "multi_replica_summary_csv": str(summary_csv),
        },
    }

    write_json(summary_json_path, result)

    print(f"target={target}")
    print(f"replicas={len(documents)}")
    print(f"donors={donor_counts}")
    print()

    print(
        f"{'H':>4} "
        f"{'metric':<31} "
        f"{'mean':>9} "
        f"{'median':>9} "
        f"{'sd':>9} "
        f"{'+':>4} "
        f"{'-':>4}"
    )

    for row in summary_rows:
        short_metric = {
            "coupled_vs_baseline_gain_pct": "coupled vs baseline",
            "decoupled_vs_baseline_gain_pct": "decoupled vs baseline",
            "decoupled_vs_coupled_gain_pct": "decoupled vs coupled",
        }[row["metric"]]

        print(
            f"{row['horizon']:>4} "
            f"{short_metric:<31} "
            f"{row['mean_pct']:>+8.3f}% "
            f"{row['median_pct']:>+8.3f}% "
            f"{row['stdev_pct']:>8.3f}% "
            f"{row['positive_replicas']:>4} "
            f"{row['negative_replicas']:>4}"
        )

    print()
    print(f"PASS: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggrega più repliche GCP dello stesso target."
    )

    parser.add_argument(
        "--replica-analysis",
        action="append",
        required=True,
        help="Path a replica-analysis.json. Ripetere per ogni replica.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze(args)


if __name__ == "__main__":
    main()