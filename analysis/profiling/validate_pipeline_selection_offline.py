#!/usr/bin/env python3
"""
Offline validation of the FINAL Serverledge pipeline-selection choices.

Purpose
-------
This script closes the remaining offline checks before moving to the online GCP
case studies. It does NOT collect new cloud data and therefore costs no GCP
credit.

It evaluates the FINAL decoupled Transfer Learning policy:

    reward = -ln(duration_ms)
    c      = 0.8
    w_R    = 0.25
    w_E    = 1.0

with the reference-anchored prior and the same paired empirical Monte Carlo
mechanism already validated in validate_transfer_simulation.py.

The script checks two decisions:

1) clustering mechanism:
       K-Means + Manhattan
       vs
       DBSCAN  + Manhattan

2) donor ranker, after fixing K-Means:
       Manhattan
       vs
       Euclidean

For both decisions it repeats H=10 evaluation on five seeds:
    11, 23, 42, 77, 101

Important reproducibility rule
------------------------------
The empirical arm sequences are ALWAYS generated up to Hmax=50, exactly as in
the original study, even though the metrics are read at H=10. This preserves
the original RNG stream.

The simulation core is imported directly from:
    analysis/profiling/transfer_selective_dbscan.py
    analysis/profiling/transfer_ucb1_offline.py

Therefore this script is an audit/orchestration layer, not a new simulator.

Outputs
-------
00_validation_summary.csv
01_per_seed_h10.csv
02_multiseed_aggregate_h10.csv
03_clustering_pairwise_by_seed.csv
04_distance_pairwise_by_seed.csv
05_historical_reproduction_seed42.csv
06_donor_quality_euclidean_vs_manhattan.csv
validation_manifest.json

figures/
    clustering_multiseed_latency_gain_h10.png/.svg
    distance_multiseed_latency_gain_h10.png/.svg
    clustering_multiseed_regret_h10.png/.svg
    distance_multiseed_regret_h10.png/.svg
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) >= 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.profiling.transfer_ucb1_offline import (  # noqa: E402
    empirical_stats,
    load_raw_durations,
)
from analysis.profiling.transfer_selective_dbscan import (  # noqa: E402
    build_reference_anchored_prior,
    load_manhattan_donors,
    make_sequences,
    simulate_baseline_trace,
    simulate_transfer_trace,
    trace_metrics_at_horizon,
)

DEFAULT_SEEDS = [11, 23, 42, 77, 101]
DEFAULT_C = 0.8
DEFAULT_WR = 0.25
DEFAULT_WE = 1.0
DEFAULT_REPLICATES = 500
DEFAULT_HMAX = 50
DEFAULT_PRIMARY_H = 10
DEFAULT_CONVERGENCE_RUN = 5

CONFIGS = [
    ("kmeans", "manhattan"),
    ("dbscan", "manhattan"),
    ("kmeans", "euclidean"),
]

METRIC_FIELDS = [
    "transfer_coverage",
    "macro_mean_latency_gain_pct",
    "macro_mean_wrong_choices_saved",
    "macro_mean_convergence_requests_saved",
    "macro_first_arm_optimal_probability",
    "macro_mean_reward_pseudo_regret",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def first_value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean_std_min_max(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def find_historical_row(
    path: Path,
    horizon: int,
    w_r: float,
    w_e: float,
) -> dict[str, str]:
    for row in read_csv(path):
        if (
            int(float(row["horizon"])) == horizon
            and abs(float(row["reward_prior_weight"]) - w_r) <= 1e-12
            and abs(float(row["exploration_prior_weight"]) - w_e) <= 1e-12
        ):
            return row
    raise RuntimeError(
        f"No historical row H={horizon}, w_R={w_r}, w_E={w_e} in {path}"
    )


# ---------------------------------------------------------------------------
# Exact H=10 replay using the repository's already validated simulation core
# ---------------------------------------------------------------------------

def evaluate_all(
    performance: dict[str, dict[str, Any]],
    donors_by_config: dict[tuple[str, str], dict[str, str | None]],
    seeds: list[int],
    replicates: int,
    hmax: int,
    primary_h: int,
    c: float,
    w_r: float,
    w_e: float,
    convergence_run: int,
) -> list[dict[str, Any]]:
    if primary_h > hmax:
        raise ValueError("primary_h cannot exceed hmax")

    targets = sorted(performance)
    rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"\n=== seed {seed} ({seed_index}/{len(seeds)}) ===")

        # Per-config, per-target records; macro aggregation happens after all targets.
        target_records: dict[tuple[str, str], list[dict[str, Any]]] = {
            cfg: [] for cfg in CONFIGS
        }

        for target_index, target_name in enumerate(targets, start=1):
            if target_index == 1 or target_index % 10 == 0 or target_index == len(targets):
                print(f"  [{target_index:02d}/{len(targets)}] {target_name}")

            target_stats = performance[target_name]

            # Critical reproducibility detail:
            # generate Hmax=50 sequences, but evaluate policy only to H=10.
            sequences = make_sequences(
                target_name=target_name,
                target_stats=target_stats,
                horizon=hmax,
                replicates=replicates,
                seed=seed,
            )

            baseline_traces = [
                simulate_baseline_trace(
                    target_stats=target_stats,
                    sequences=sequences[i],
                    horizon=primary_h,
                    c=c,
                    convergence_run=convergence_run,
                )
                for i in range(replicates)
            ]
            baseline_metrics = [
                trace_metrics_at_horizon(
                    trace=trace,
                    target_stats=target_stats,
                    horizon=primary_h,
                    convergence_run=convergence_run,
                )
                for trace in baseline_traces
            ]

            baseline_latency = np.asarray(
                [m["cumulative_latency"] for m in baseline_metrics],
                dtype=float,
            )
            baseline_wrong = np.asarray(
                [m["wrong_count"] for m in baseline_metrics],
                dtype=float,
            )
            baseline_conv = np.asarray(
                [m["censored_convergence_request"] for m in baseline_metrics],
                dtype=float,
            )

            for cfg in CONFIGS:
                algorithm, ranker = cfg
                donor_name = donors_by_config[cfg][target_name]
                transfer_applied = donor_name is not None

                if transfer_applied:
                    prior_mean = build_reference_anchored_prior(
                        target_stats,
                        performance[donor_name],
                    )
                    transfer_traces = [
                        simulate_transfer_trace(
                            target_stats=target_stats,
                            prior_mean_reward=prior_mean,
                            sequences=sequences[i],
                            horizon=primary_h,
                            c=c,
                            reward_prior_weight=w_r,
                            exploration_prior_weight=w_e,
                            convergence_run=convergence_run,
                        )
                        for i in range(replicates)
                    ]
                    transfer_metrics = [
                        trace_metrics_at_horizon(
                            trace=trace,
                            target_stats=target_stats,
                            horizon=primary_h,
                            convergence_run=convergence_run,
                        )
                        for trace in transfer_traces
                    ]
                else:
                    # Same semantics as transfer_selective_dbscan.py:
                    # DBSCAN abstention -> no prior -> exact baseline.
                    transfer_metrics = baseline_metrics

                transfer_latency = np.asarray(
                    [m["cumulative_latency"] for m in transfer_metrics],
                    dtype=float,
                )
                transfer_wrong = np.asarray(
                    [m["wrong_count"] for m in transfer_metrics],
                    dtype=float,
                )
                transfer_conv = np.asarray(
                    [m["censored_convergence_request"] for m in transfer_metrics],
                    dtype=float,
                )

                target_records[cfg].append(
                    {
                        "transfer_applied": transfer_applied,
                        "mean_latency_gain_pct": float(
                            np.mean(
                                100.0
                                * (baseline_latency - transfer_latency)
                                / baseline_latency
                            )
                        ),
                        "mean_wrong_choices_saved": float(
                            np.mean(baseline_wrong - transfer_wrong)
                        ),
                        "mean_convergence_requests_saved": float(
                            np.mean(baseline_conv - transfer_conv)
                        ),
                        "first_arm_optimal_probability": float(
                            np.mean(
                                [m["first_arm_optimal"] for m in transfer_metrics]
                            )
                        ),
                        "mean_reward_pseudo_regret": float(
                            np.mean(
                                [m["reward_pseudo_regret"] for m in transfer_metrics]
                            )
                        ),
                    }
                )

        for algorithm, ranker in CONFIGS:
            recs = target_records[(algorithm, ranker)]
            rows.append(
                {
                    "algorithm": algorithm,
                    "ranker": ranker,
                    "seed": seed,
                    "horizon": primary_h,
                    "replicates": replicates,
                    "target_count": len(recs),
                    "transfer_applied_count": sum(
                        1 for r in recs if r["transfer_applied"]
                    ),
                    "transfer_coverage": float(
                        np.mean([float(r["transfer_applied"]) for r in recs])
                    ),
                    "macro_mean_latency_gain_pct": float(
                        np.mean([r["mean_latency_gain_pct"] for r in recs])
                    ),
                    "macro_mean_wrong_choices_saved": float(
                        np.mean([r["mean_wrong_choices_saved"] for r in recs])
                    ),
                    "macro_mean_convergence_requests_saved": float(
                        np.mean(
                            [r["mean_convergence_requests_saved"] for r in recs]
                        )
                    ),
                    "macro_first_arm_optimal_probability": float(
                        np.mean(
                            [r["first_arm_optimal_probability"] for r in recs]
                        )
                    ),
                    "macro_mean_reward_pseudo_regret": float(
                        np.mean([r["mean_reward_pseudo_regret"] for r in recs])
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Aggregation / comparisons
# ---------------------------------------------------------------------------

def aggregate_multiseed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for algorithm, ranker in CONFIGS:
        group = [
            r for r in rows
            if r["algorithm"] == algorithm and r["ranker"] == ranker
        ]
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "ranker": ranker,
            "seed_count": len(group),
            "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: x["seed"])),
        }

        for field in METRIC_FIELDS:
            s = mean_std_min_max([float(r[field]) for r in group])
            row[f"{field}_mean"] = s["mean"]
            row[f"{field}_std"] = s["std"]
            row[f"{field}_min"] = s["min"]
            row[f"{field}_max"] = s["max"]

        out.append(row)

    return out


def pairwise_rows(
    rows: list[dict[str, Any]],
    left_cfg: tuple[str, str],
    right_cfg: tuple[str, str],
    comparison_name: str,
) -> list[dict[str, Any]]:
    index = {
        (r["algorithm"], r["ranker"], int(r["seed"])): r
        for r in rows
    }
    seeds = sorted(
        {
            int(r["seed"])
            for r in rows
            if (r["algorithm"], r["ranker"]) == left_cfg
        }
    )

    out: list[dict[str, Any]] = []
    for seed in seeds:
        left = index[(left_cfg[0], left_cfg[1], seed)]
        right = index[(right_cfg[0], right_cfg[1], seed)]

        out.append(
            {
                "comparison": comparison_name,
                "seed": seed,
                "left": f"{left_cfg[0]}+{left_cfg[1]}",
                "right": f"{right_cfg[0]}+{right_cfg[1]}",
                # Positive = left better for all *_gain/saved/probability fields.
                "delta_transfer_coverage": (
                    float(left["transfer_coverage"])
                    - float(right["transfer_coverage"])
                ),
                "delta_latency_gain_pp": (
                    float(left["macro_mean_latency_gain_pct"])
                    - float(right["macro_mean_latency_gain_pct"])
                ),
                "delta_wrong_choices_saved": (
                    float(left["macro_mean_wrong_choices_saved"])
                    - float(right["macro_mean_wrong_choices_saved"])
                ),
                "delta_convergence_requests_saved": (
                    float(left["macro_mean_convergence_requests_saved"])
                    - float(right["macro_mean_convergence_requests_saved"])
                ),
                "delta_first_arm_optimal_probability": (
                    float(left["macro_first_arm_optimal_probability"])
                    - float(right["macro_first_arm_optimal_probability"])
                ),
                # Negative = left has LOWER regret, therefore better.
                "delta_reward_pseudo_regret": (
                    float(left["macro_mean_reward_pseudo_regret"])
                    - float(right["macro_mean_reward_pseudo_regret"])
                ),
            }
        )
    return out


def historical_reproduction(
    current_rows: list[dict[str, Any]],
    historical_kmeans: Path,
    historical_dbscan: Path,
    primary_h: int,
    w_r: float,
    w_e: float,
    tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    current = {
        (r["algorithm"], r["ranker"], int(r["seed"])): r
        for r in current_rows
    }

    checks = [
        ("kmeans", "manhattan", historical_kmeans),
        ("dbscan", "manhattan", historical_dbscan),
    ]

    out: list[dict[str, Any]] = []
    for algorithm, ranker, path in checks:
        old = find_historical_row(path, primary_h, w_r, w_e)
        new = current[(algorithm, ranker, 42)]

        for field in METRIC_FIELDS:
            old_value = float(old[field])
            new_value = float(new[field])
            delta = new_value - old_value
            out.append(
                {
                    "algorithm": algorithm,
                    "ranker": ranker,
                    "seed": 42,
                    "metric": field,
                    "historical": old_value,
                    "recomputed": new_value,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "status": "PASS" if abs(delta) <= tolerance else "FAIL",
                }
            )

    return out


def donor_quality_check(path: Path) -> list[dict[str, Any]]:
    wanted = {"euclidean", "manhattan"}
    rows = [
        r for r in read_csv(path)
        if r.get("algorithm") == "kmeans" and r.get("ranker") in wanted
    ]
    rows.sort(key=lambda r: r["ranker"])
    return [
        {
            "algorithm": r["algorithm"],
            "ranker": r["ranker"],
            "target_count": int(float(r["target_count"])),
            "selected_count": int(float(r["selected_count"])),
            "coverage": float(r["coverage"]),
            "best_reward_arm_agreement_rate": float(
                r["best_reward_arm_agreement_rate"]
            ),
            "mean_abs_reward_gap_error": float(
                r["mean_abs_reward_gap_error"]
            ),
            "median_abs_reward_gap_error": float(
                r["median_abs_reward_gap_error"]
            ),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_metric_by_seed(
    rows: list[dict[str, Any]],
    configs: list[tuple[str, str]],
    metric: str,
    ylabel: str,
    title: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(7.5, 4.5))

    for cfg in configs:
        group = sorted(
            [
                r for r in rows
                if (r["algorithm"], r["ranker"]) == cfg
            ],
            key=lambda r: int(r["seed"]),
        )
        plt.plot(
            [int(r["seed"]) for r in group],
            [float(r[metric]) for r in group],
            marker="o",
            label=f"{cfg[0]} + {cfg[1]}",
        )

    plt.xlabel("Seed")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(DEFAULT_SEEDS)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()

    default_data = REPO_ROOT / "data" / "profiling"

    p.add_argument(
        "--donor-results",
        type=Path,
        default=default_data
        / "analysis-transfer-donor-metrics-20260918"
        / "donor-metric-per-target.csv",
    )
    p.add_argument(
        "--donor-quality",
        type=Path,
        default=default_data
        / "analysis-transfer-donor-metrics-20260918"
        / "presentation_exports"
        / "kmeans_manhattan_vs_euclidean.csv",
    )
    p.add_argument(
        "--raw-x86",
        type=Path,
        default=default_data
        / "final-20260913-analysis-01"
        / "raw"
        / "x86"
        / "all_samples.jsonl",
    )
    p.add_argument(
        "--raw-arm64",
        type=Path,
        default=default_data
        / "final-20260913-analysis-01"
        / "raw"
        / "arm64"
        / "all_samples.jsonl",
    )
    p.add_argument(
        "--historical-kmeans",
        type=Path,
        default=default_data
        / "analysis-transfer-selective-kmeans-final"
        / "decoupled-ucb1-summary.csv",
    )
    p.add_argument(
        "--historical-dbscan",
        type=Path,
        default=default_data
        / "analysis-transfer-selective-dbscan-final"
        / "decoupled-ucb1-summary.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_data / "analysis-selection-multiseed-validation",
    )

    p.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    p.add_argument("--hmax", type=int, default=DEFAULT_HMAX)
    p.add_argument("--primary-h", type=int, default=DEFAULT_PRIMARY_H)
    p.add_argument("--c", type=float, default=DEFAULT_C)
    p.add_argument("--reward-weight", type=float, default=DEFAULT_WR)
    p.add_argument("--exploration-weight", type=float, default=DEFAULT_WE)
    p.add_argument("--convergence-run", type=int, default=DEFAULT_CONVERGENCE_RUN)

    args = p.parse_args()

    donor_results = require_file(args.donor_results, "donor results")
    donor_quality = require_file(args.donor_quality, "donor-quality summary")
    raw_x86_path = require_file(args.raw_x86, "x86 raw samples")
    raw_arm_path = require_file(args.raw_arm64, "ARM raw samples")
    historical_kmeans = require_file(args.historical_kmeans, "historical K-Means summary")
    historical_dbscan = require_file(args.historical_dbscan, "historical DBSCAN summary")

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("Loading measured GCP samples...")
    performance = empirical_stats(
        load_raw_durations(raw_x86_path),
        load_raw_durations(raw_arm_path),
    )
    print(f"Eligible targets: {len(performance)}")

    donors_by_config = {
        cfg: load_manhattan_donors(
            donor_results,
            algorithm=cfg[0],
            ranker=cfg[1],
        )
        for cfg in CONFIGS
    }

    for cfg, donors in donors_by_config.items():
        if set(donors) != set(performance):
            raise RuntimeError(
                f"Donor/performance mismatch for {cfg}: "
                f"missing={sorted(set(performance) - set(donors))}, "
                f"extra={sorted(set(donors) - set(performance))}"
            )

    print("\nRunning fast exact H=10 replay...")
    per_seed = evaluate_all(
        performance=performance,
        donors_by_config=donors_by_config,
        seeds=args.seeds,
        replicates=args.replicates,
        hmax=args.hmax,
        primary_h=args.primary_h,
        c=args.c,
        w_r=args.reward_weight,
        w_e=args.exploration_weight,
        convergence_run=args.convergence_run,
    )
    write_csv(out / "01_per_seed_h10.csv", per_seed)

    aggregate = aggregate_multiseed(per_seed)
    write_csv(out / "02_multiseed_aggregate_h10.csv", aggregate)

    clustering_pairwise = pairwise_rows(
        per_seed,
        ("kmeans", "manhattan"),
        ("dbscan", "manhattan"),
        "kmeans_vs_dbscan",
    )
    write_csv(out / "03_clustering_pairwise_by_seed.csv", clustering_pairwise)

    distance_pairwise = pairwise_rows(
        per_seed,
        ("kmeans", "manhattan"),
        ("kmeans", "euclidean"),
        "manhattan_vs_euclidean",
    )
    write_csv(out / "04_distance_pairwise_by_seed.csv", distance_pairwise)

    reproduction = historical_reproduction(
        current_rows=per_seed,
        historical_kmeans=historical_kmeans,
        historical_dbscan=historical_dbscan,
        primary_h=args.primary_h,
        w_r=args.reward_weight,
        w_e=args.exploration_weight,
    )
    write_csv(out / "05_historical_reproduction_seed42.csv", reproduction)

    donor_quality_rows = donor_quality_check(donor_quality)
    write_csv(out / "06_donor_quality_euclidean_vs_manhattan.csv", donor_quality_rows)

    # -----------------------------------------------------------------------
    # PASS/WARN summary
    # -----------------------------------------------------------------------

    hist_pass = all(r["status"] == "PASS" for r in reproduction)

    clustering_pass = all(
        r["delta_transfer_coverage"] > 0
        and r["delta_latency_gain_pp"] > 0
        and r["delta_wrong_choices_saved"] > 0
        and r["delta_convergence_requests_saved"] > 0
        and r["delta_first_arm_optimal_probability"] > 0
        and r["delta_reward_pseudo_regret"] < 0
        for r in clustering_pairwise
    )

    distance_pass = all(
        r["delta_latency_gain_pp"] > 0
        and r["delta_wrong_choices_saved"] > 0
        and r["delta_convergence_requests_saved"] > 0
        and r["delta_first_arm_optimal_probability"] > 0
        and r["delta_reward_pseudo_regret"] < 0
        for r in distance_pairwise
    )

    dq = {r["ranker"]: r for r in donor_quality_rows}
    donor_quality_pass = (
        "manhattan" in dq
        and "euclidean" in dq
        and dq["manhattan"]["coverage"] >= dq["euclidean"]["coverage"]
        and dq["manhattan"]["best_reward_arm_agreement_rate"]
        > dq["euclidean"]["best_reward_arm_agreement_rate"]
        and dq["manhattan"]["mean_abs_reward_gap_error"]
        <= dq["euclidean"]["mean_abs_reward_gap_error"]
    )

    summary_rows = [
        {
            "check": "historical_seed42_reproduction",
            "status": "PASS" if hist_pass else "FAIL",
            "details": (
                "Recomputed K-Means/DBSCAN H=10 seed-42 metrics match "
                "the original stored summaries."
            ),
        },
        {
            "check": "kmeans_vs_dbscan_5seed",
            "status": "PASS" if clustering_pass else "WARN",
            "details": (
                "K-Means+Manhattan is better than DBSCAN+Manhattan on all "
                "five seeds for coverage, latency gain, wrong choices saved, "
                "convergence requests saved and first-arm optimal probability, "
                "with lower reward pseudo-regret."
            ),
        },
        {
            "check": "manhattan_vs_euclidean_5seed",
            "status": "PASS" if distance_pass else "WARN",
            "details": (
                "With K-Means fixed, Manhattan is better than Euclidean on "
                "all five seeds for latency gain, wrong choices saved, "
                "convergence requests saved and first-arm optimal probability, "
                "with lower reward pseudo-regret."
            ),
        },
        {
            "check": "donor_quality_deterministic",
            "status": "PASS" if donor_quality_pass else "WARN",
            "details": (
                "LOFO donor-quality table independently supports Manhattan: "
                "same coverage, higher best-arm agreement and no worse mean "
                "architecture-gap error."
            ),
        },
    ]
    write_csv(out / "00_validation_summary.csv", summary_rows)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    plot_metric_by_seed(
        per_seed,
        [("kmeans", "manhattan"), ("dbscan", "manhattan")],
        "macro_mean_latency_gain_pct",
        "Macro mean latency gain at H=10 (%)",
        "K-Means vs DBSCAN — 5-seed stability",
        figures / "clustering_multiseed_latency_gain_h10",
    )
    plot_metric_by_seed(
        per_seed,
        [("kmeans", "manhattan"), ("kmeans", "euclidean")],
        "macro_mean_latency_gain_pct",
        "Macro mean latency gain at H=10 (%)",
        "Manhattan vs Euclidean — 5-seed stability",
        figures / "distance_multiseed_latency_gain_h10",
    )
    plot_metric_by_seed(
        per_seed,
        [("kmeans", "manhattan"), ("dbscan", "manhattan")],
        "macro_mean_reward_pseudo_regret",
        "Macro mean reward pseudo-regret at H=10",
        "K-Means vs DBSCAN — reward pseudo-regret",
        figures / "clustering_multiseed_regret_h10",
    )
    plot_metric_by_seed(
        per_seed,
        [("kmeans", "manhattan"), ("kmeans", "euclidean")],
        "macro_mean_reward_pseudo_regret",
        "Macro mean reward pseudo-regret at H=10",
        "Manhattan vs Euclidean — reward pseudo-regret",
        figures / "distance_multiseed_regret_h10",
    )

    manifest = {
        "purpose": "offline multiseed validation of final clustering/ranker selection",
        "not_a_new_simulator": True,
        "simulation_core": [
            "analysis.profiling.transfer_selective_dbscan",
            "analysis.profiling.transfer_ucb1_offline",
        ],
        "policy": {
            "reward": "-ln(duration_ms)",
            "prior": "reference-anchored architecture-effect",
            "c": args.c,
            "reward_prior_weight": args.reward_weight,
            "exploration_prior_weight": args.exploration_weight,
        },
        "evaluation": {
            "seeds": args.seeds,
            "replicates_per_seed": args.replicates,
            "sequence_hmax": args.hmax,
            "primary_horizon": args.primary_h,
            "convergence_run": args.convergence_run,
            "paired_common_random_numbers": True,
        },
        "configs": [
            {"algorithm": a, "ranker": r}
            for a, r in CONFIGS
        ],
        "inputs": {
            "donor_results": str(donor_results),
            "donor_results_sha256": sha256_file(donor_results),
            "raw_x86": str(raw_x86_path),
            "raw_x86_sha256": sha256_file(raw_x86_path),
            "raw_arm64": str(raw_arm_path),
            "raw_arm64_sha256": sha256_file(raw_arm_path),
            "historical_kmeans": str(historical_kmeans),
            "historical_dbscan": str(historical_dbscan),
            "donor_quality": str(donor_quality),
        },
    }
    with (out / "validation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL OFFLINE SELECTION VALIDATION")
    print("=" * 72)

    for row in summary_rows:
        print(f"{row['status']:>4}  {row['check']}")
        print(f"      {row['details']}")

    print("\n5-seed means at H=10:")
    for row in aggregate:
        print(
            f"  {row['algorithm']:7s} + {row['ranker']:9s} | "
            f"gain={row['macro_mean_latency_gain_pct_mean']:+.6f}% "
            f"(sd={row['macro_mean_latency_gain_pct_std']:.6f} pp) | "
            f"coverage={100*row['transfer_coverage_mean']:.2f}% | "
            f"first-arm={100*row['macro_first_arm_optimal_probability_mean']:.2f}% | "
            f"regret={row['macro_mean_reward_pseudo_regret_mean']:.6f}"
        )

    print(f"\nOutputs: {out}")
    print("=" * 72)

    return 0 if hist_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
