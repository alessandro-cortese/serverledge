#!/usr/bin/env python3
"""
Build master CSVs for the Serverledge randomaccess transfer-learning experiments.

Outputs:
  1) transfer_learning_results_master.csv
     One row per (campaign, replica, variant, horizon).

  2) transfer_learning_results_aggregate.csv
     Mean/stdev across replicas, ready for plotting.

The script intentionally keeps two experiment families separate:
  - transfer_main:
      no-transfer vs coupled vs decoupled, R01/R02/R03, c=0.8
  - exploration_ablation:
      coupled vs decoupled, R02/R03, c in {0.0, 0.8}

This prevents mixing results that answer different research questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HORIZONS = (5, 10, 20, 50)


@dataclass(frozen=True)
class RunSpec:
    campaign: str
    replica: str
    variant: str
    mode: str
    policy_expected: str
    c: float
    transfer_enabled: bool
    donor: str
    reward_observation_weight: float | None
    exploration_observation_weight: float | None
    run_dir: str


# ---------------------------------------------------------------------------
# Frozen experiment catalogue.
#
# Main transfer campaign: same c=0.8 for baseline/coupled/decoupled.
# R01 donor was named compression-node; R02/R03 use compression.
# ---------------------------------------------------------------------------

RUNS: tuple[RunSpec, ...] = (
    # ===== Main transfer campaign — R01 =====
    RunSpec(
        "transfer_main", "R01", "no_transfer", "no-transfer",
        "UCB1", 0.8, False, "",
        None, None,
        "gcp-transfer-final-randomaccess-no-transfer-20260920_154452",
    ),
    RunSpec(
        "transfer_main", "R01", "coupled", "coupled",
        "UCB1", 0.8, True, "compression-node",
        0.25, 0.25,
        "gcp-transfer-final-randomaccess-coupled-20260920_154804",
    ),
    RunSpec(
        "transfer_main", "R01", "decoupled", "decoupled",
        "UCB1Decoupled", 0.8, True, "compression-node",
        0.25, 1.0,
        "gcp-transfer-final-randomaccess-decoupled-20260920_155555",
    ),

    # ===== Main transfer campaign — R02 =====
    RunSpec(
        "transfer_main", "R02", "no_transfer", "no-transfer",
        "UCB1", 0.8, False, "",
        None, None,
        "gcp-transfer-final-randomaccess-no-transfer-20260920_161356",
    ),
    RunSpec(
        "transfer_main", "R02", "coupled", "coupled",
        "UCB1", 0.8, True, "compression",
        0.25, 0.25,
        "gcp-transfer-final-randomaccess-coupled-20260920_161638",
    ),
    RunSpec(
        "transfer_main", "R02", "decoupled", "decoupled",
        "UCB1Decoupled", 0.8, True, "compression",
        0.25, 1.0,
        "gcp-transfer-final-randomaccess-decoupled-20260920_160825",
    ),

    # ===== Main transfer campaign — R03 =====
    RunSpec(
        "transfer_main", "R03", "no_transfer", "no-transfer",
        "UCB1", 0.8, False, "",
        None, None,
        "gcp-transfer-final-randomaccess-no-transfer-20260920_163626",
    ),
    RunSpec(
        "transfer_main", "R03", "coupled", "coupled",
        "UCB1", 0.8, True, "compression",
        0.25, 0.25,
        "gcp-transfer-final-randomaccess-coupled-20260920_162514",
    ),
    RunSpec(
        "transfer_main", "R03", "decoupled", "decoupled",
        "UCB1Decoupled", 0.8, True, "compression",
        0.25, 1.0,
        "gcp-transfer-final-randomaccess-decoupled-20260920_163051",
    ),

    # ===== Exploration ablation — R02 =====
    RunSpec(
        "exploration_ablation", "R02", "coupled", "coupled",
        "UCB1", 0.0, True, "compression",
        0.25, 0.25,
        "gcp-transfer-materialized-randomaccess-c00-20260922_093802",
    ),
    RunSpec(
        "exploration_ablation", "R02", "coupled", "coupled",
        "UCB1", 0.8, True, "compression",
        0.25, 0.25,
        "gcp-transfer-materialized-randomaccess-c08-20260922_094344",
    ),
    RunSpec(
        "exploration_ablation", "R02", "decoupled", "decoupled",
        "UCB1Decoupled", 0.0, True, "compression",
        0.25, 1.0,
        "gcp-transfer-materialized-randomaccess-decoupled-c00-20260922_174716",
    ),
    RunSpec(
        "exploration_ablation", "R02", "decoupled", "decoupled",
        "UCB1Decoupled", 0.8, True, "compression",
        0.25, 1.0,
        "gcp-transfer-materialized-randomaccess-decoupled-c08-20260922_174404",
    ),

    # ===== Exploration ablation — R03 =====
    RunSpec(
        "exploration_ablation", "R03", "coupled", "coupled",
        "UCB1", 0.0, True, "compression",
        0.25, 0.25,
        "gcp-transfer-materialized-randomaccess-c00-20260922_095808",
    ),
    RunSpec(
        "exploration_ablation", "R03", "coupled", "coupled",
        "UCB1", 0.8, True, "compression",
        0.25, 0.25,
        "gcp-transfer-materialized-randomaccess-c08-20260922_095502",
    ),
    RunSpec(
        "exploration_ablation", "R03", "decoupled", "decoupled",
        "UCB1Decoupled", 0.0, True, "compression",
        0.25, 1.0,
        "gcp-transfer-materialized-randomaccess-decoupled-c00-20260923_080510",
    ),
    RunSpec(
        "exploration_ablation", "R03", "decoupled", "decoupled",
        "UCB1Decoupled", 0.8, True, "compression",
        0.25, 1.0,
        "gcp-transfer-materialized-randomaccess-decoupled-c08-20260923_080146",
    ),
)


MASTER_FIELDS = (
    "campaign",
    "replica",
    "variant",
    "mode",
    "policy",
    "c",
    "transfer_enabled",
    "target_function",
    "donor_function",
    "reward_observation_weight",
    "exploration_observation_weight",
    "horizon_requests",
    "cumulative_duration_ms",
    "cumulative_duration_s",
    "mean_duration_ms",
    "median_duration_ms",
    "amd64_executions",
    "arm64_executions",
    "amd64_share_pct",
    "arm64_share_pct",
    "fallbacks",
    "all_warm",
    "request_count_total",
    "baseline_cumulative_duration_ms",
    "gain_vs_baseline_ms",
    "gain_vs_baseline_pct",
    "source_run",
    "summary_path",
)


AGG_FIELDS = (
    "campaign",
    "variant",
    "mode",
    "policy",
    "c",
    "transfer_enabled",
    "reward_observation_weight",
    "exploration_observation_weight",
    "horizon_requests",
    "n_replicas",
    "cumulative_duration_mean_ms",
    "cumulative_duration_mean_s",
    "cumulative_duration_stdev_ms",
    "mean_duration_mean_ms",
    "median_duration_mean_ms",
    "amd64_executions_mean",
    "arm64_executions_mean",
    "amd64_share_mean_pct",
    "arm64_share_mean_pct",
    "fallbacks_total",
    "gain_vs_baseline_mean_pct",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: atteso oggetto JSON")
    return obj


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    return float(value)


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in RUNS:
        run_root = root / spec.run_dir
        summary_path = run_root / "measure" / "summary.json"

        if not summary_path.is_file():
            missing.append(str(summary_path))
            continue

        summary = load_json(summary_path)

        target_function = str(summary.get("target_function", ""))
        policy = str(summary.get("policy", ""))

        if target_function and target_function != "randomaccess":
            raise ValueError(
                f"{summary_path}: target inatteso {target_function!r}"
            )

        if policy and policy != spec.policy_expected:
            raise ValueError(
                f"{summary_path}: policy={policy!r}, "
                f"attesa={spec.policy_expected!r}"
            )

        horizons = summary.get("horizons")
        if not isinstance(horizons, dict):
            raise ValueError(f"{summary_path}: horizons mancante/non valido")

        for horizon in HORIZONS:
            h = horizons.get(str(horizon))
            if not isinstance(h, dict):
                raise ValueError(
                    f"{summary_path}: manca horizon={horizon}"
                )

            cumulative_ms = as_float(h.get("cumulative_duration_ms"))
            mean_ms = as_float(h.get("mean_duration_ms"))
            median_ms = as_float(h.get("median_duration_ms"))

            amd64 = as_int(h.get("amd64_executions"))
            arm64 = as_int(h.get("arm64_executions"))
            fallbacks = as_int(h.get("fallbacks"))

            executed = amd64 + arm64
            if executed != horizon:
                raise ValueError(
                    f"{summary_path}: horizon={horizon}, "
                    f"amd64+arm64={executed}, atteso={horizon}"
                )

            row = {
                "campaign": spec.campaign,
                "replica": spec.replica,
                "variant": spec.variant,
                "mode": spec.mode,
                "policy": policy or spec.policy_expected,
                "c": spec.c,
                "transfer_enabled": str(spec.transfer_enabled).lower(),
                "target_function": target_function or "randomaccess",
                "donor_function": spec.donor,
                "reward_observation_weight": (
                    "" if spec.reward_observation_weight is None
                    else spec.reward_observation_weight
                ),
                "exploration_observation_weight": (
                    "" if spec.exploration_observation_weight is None
                    else spec.exploration_observation_weight
                ),
                "horizon_requests": horizon,
                "cumulative_duration_ms": cumulative_ms,
                "cumulative_duration_s": cumulative_ms / 1000.0,
                "mean_duration_ms": mean_ms,
                "median_duration_ms": median_ms,
                "amd64_executions": amd64,
                "arm64_executions": arm64,
                "amd64_share_pct": 100.0 * amd64 / executed,
                "arm64_share_pct": 100.0 * arm64 / executed,
                "fallbacks": fallbacks,
                "all_warm": str(bool(summary.get("all_warm", False))).lower(),
                "request_count_total": as_int(summary.get("request_count")),
                "baseline_cumulative_duration_ms": "",
                "gain_vs_baseline_ms": "",
                "gain_vs_baseline_pct": "",
                "source_run": spec.run_dir,
                "summary_path": str(summary_path),
            }
            rows.append(row)

    if missing:
        print("ERROR — summary mancanti:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(
            f"STOP — {len(missing)} run attesi non trovati. "
            "Non genero un CSV parziale."
        )

    # Match transfer_main rows against the no-transfer baseline
    # of the same replica and horizon.
    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if (
            row["campaign"] == "transfer_main"
            and row["variant"] == "no_transfer"
        ):
            baseline[
                (str(row["replica"]), int(row["horizon_requests"]))
            ] = float(row["cumulative_duration_ms"])

    for row in rows:
        if row["campaign"] != "transfer_main":
            continue

        key = (str(row["replica"]), int(row["horizon_requests"]))
        base = baseline.get(key)
        if base is None:
            raise ValueError(f"baseline mancante per {key}")

        current = float(row["cumulative_duration_ms"])
        gain_ms = base - current
        gain_pct = (gain_ms / base) * 100.0 if base else math.nan

        row["baseline_cumulative_duration_ms"] = base
        row["gain_vs_baseline_ms"] = gain_ms
        row["gain_vs_baseline_pct"] = gain_pct

    return rows


def write_master(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_FIELDS)
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda r: (
                str(r["campaign"]),
                str(r["replica"]),
                str(r["variant"]),
                float(r["c"]),
                int(r["horizon_requests"]),
            ),
        ):
            writer.writerow(row)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def build_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str, float, int], list[dict[str, Any]]
    ] = {}

    for row in rows:
        key = (
            str(row["campaign"]),
            str(row["variant"]),
            str(row["mode"]),
            str(row["policy"]),
            float(row["c"]),
            int(row["horizon_requests"]),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []

    for key, group in groups.items():
        campaign, variant, mode, policy, c, horizon = key

        cumulative = [
            float(r["cumulative_duration_ms"]) for r in group
        ]
        mean_duration = [
            float(r["mean_duration_ms"]) for r in group
        ]
        median_duration = [
            float(r["median_duration_ms"]) for r in group
        ]
        amd64 = [float(r["amd64_executions"]) for r in group]
        arm64 = [float(r["arm64_executions"]) for r in group]
        amd64_share = [float(r["amd64_share_pct"]) for r in group]
        arm64_share = [float(r["arm64_share_pct"]) for r in group]
        fallbacks = [int(r["fallbacks"]) for r in group]

        gains = [
            float(r["gain_vs_baseline_pct"])
            for r in group
            if r["gain_vs_baseline_pct"] != ""
        ]

        first = group[0]

        out.append({
            "campaign": campaign,
            "variant": variant,
            "mode": mode,
            "policy": policy,
            "c": c,
            "transfer_enabled": first["transfer_enabled"],
            "reward_observation_weight":
                first["reward_observation_weight"],
            "exploration_observation_weight":
                first["exploration_observation_weight"],
            "horizon_requests": horizon,
            "n_replicas": len(group),
            "cumulative_duration_mean_ms": mean(cumulative),
            "cumulative_duration_mean_s": mean(cumulative) / 1000.0,
            "cumulative_duration_stdev_ms": stdev(cumulative),
            "mean_duration_mean_ms": mean(mean_duration),
            "median_duration_mean_ms": mean(median_duration),
            "amd64_executions_mean": mean(amd64),
            "arm64_executions_mean": mean(arm64),
            "amd64_share_mean_pct": mean(amd64_share),
            "arm64_share_mean_pct": mean(arm64_share),
            "fallbacks_total": sum(fallbacks),
            "gain_vs_baseline_mean_pct":
                mean(gains) if gains else "",
        })

    return sorted(
        out,
        key=lambda r: (
            str(r["campaign"]),
            str(r["variant"]),
            float(r["c"]),
            int(r["horizon_requests"]),
        ),
    )


def write_aggregate(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    runs = {
        (
            r["campaign"],
            r["replica"],
            r["variant"],
            r["c"],
            r["source_run"],
        )
        for r in rows
    }

    print(f"PASS — run validi caricati: {len(runs)}")
    print(f"PASS — righe master: {len(rows)}")

    expected_rows = len(RUNS) * len(HORIZONS)
    if len(rows) != expected_rows:
        raise SystemExit(
            f"STOP — righe={len(rows)}, attese={expected_rows}"
        )

    print("\n===== TRANSFER MAIN — H50 =====")
    h50 = [
        r for r in rows
        if r["campaign"] == "transfer_main"
        and int(r["horizon_requests"]) == 50
    ]

    for r in sorted(
        h50,
        key=lambda x: (str(x["replica"]), str(x["variant"])),
    ):
        gain = float(r["gain_vs_baseline_pct"])
        print(
            f'{r["replica"]:>3} '
            f'{r["variant"]:<11} '
            f'H50={float(r["cumulative_duration_s"]):8.3f}s '
            f'ARM={int(r["arm64_executions"]):2d}/50 '
            f'gain_vs_baseline={gain:+7.2f}%'
        )

    print("\n===== EXPLORATION ABLATION — H50 =====")
    h50_ab = [
        r for r in rows
        if r["campaign"] == "exploration_ablation"
        and int(r["horizon_requests"]) == 50
    ]

    for r in sorted(
        h50_ab,
        key=lambda x: (
            str(x["replica"]),
            str(x["variant"]),
            float(x["c"]),
        ),
    ):
        print(
            f'{r["replica"]:>3} '
            f'{r["variant"]:<9} '
            f'c={float(r["c"]):.1f} '
            f'H50={float(r["cumulative_duration_s"]):8.3f}s '
            f'ARM={int(r["arm64_executions"]):2d}/50'
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final"
        ),
        help=(
            "Directory containing the GCP transfer run folders. "
            "Default: data/profiling/gcp-transfer-final"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/analysis"
        ),
        help=(
            "Output directory. "
            "Default: data/profiling/gcp-transfer-final/analysis"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    if not root.is_dir():
        raise SystemExit(f"STOP — root non trovata: {root}")

    rows = build_rows(root)
    aggregate = build_aggregate(rows)

    master_path = output_dir / "transfer_learning_results_master.csv"
    aggregate_path = (
        output_dir / "transfer_learning_results_aggregate.csv"
    )

    write_master(master_path, rows)
    write_aggregate(aggregate_path, aggregate)
    print_summary(rows)

    print("\n===== OUTPUT =====")
    print(master_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()
