#!/usr/bin/env python3
"""
Confidence analysis for clustering-derived transfer donors.

This script does NOT modify the clustering or the UCB1 formula.

It asks a narrower question:

    Does donor distance contain useful confidence information?

For each clustering method separately, it studies whether closer donors have:

- higher best-arm agreement;
- lower architecture-delta error;
- lower reward-gap error;
- better UCB1 transfer gain.

It also evaluates deterministic distance gates at fixed donor-distance
quantiles:

    25%, 50%, 75%, 90%, 100%

A target receives transfer only when its donor distance is at or below the
gate threshold. Otherwise it falls back to the no-transfer baseline.

Important:
- K-Means and DBSCAN distances are never compared numerically with each other.
- DBSCAN's native abstention remains in force before the confidence gate.
- Gate quantiles are a sensitivity analysis, not a fitted "best" threshold.
- No ARM/ground-truth information is used to compute the distance itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

GATE_QUANTILES = [0.25, 0.50, 0.75, 0.90, 1.00]


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any, default: float = float("nan")) -> float:
    if value in ("", None):
        return default
    return float(value)


def quantile_label(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def donor_rows_by_key(
    donor_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    result = {}
    for row in donor_rows:
        key = (row["algorithm"], row["target_function"])
        if key in result:
            raise RuntimeError(f"Duplicate donor row: {key}")
        result[key] = row
    return result


def transfer_rows_by_key(
    transfer_rows: list[dict[str, str]],
) -> dict[tuple[str, str, float, int], dict[str, str]]:
    result = {}

    for row in transfer_rows:
        key = (
            row["algorithm"],
            row["target_function"],
            float(row["equivalent_observation_weight"]),
            int(row["horizon"]),
        )
        if key in result:
            raise RuntimeError(f"Duplicate transfer row: {key}")
        result[key] = row

    return result


def build_selected_donor_rows(
    donor_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in donor_rows:
        if row["selection_status"] != "selected":
            continue

        result[row["algorithm"]].append(
            {
                "algorithm": row["algorithm"],
                "target_function": row["target_function"],
                "donor_function": row["donor_function"],
                "donor_distance": float(row["donor_distance"]),
                "best_reward_arm_agreement": as_bool(
                    row["best_reward_arm_agreement"]
                ),
                "abs_architecture_delta_error_percent": float(
                    row["abs_architecture_delta_error_percent"]
                ),
                "abs_reward_gap_error": float(
                    row["abs_reward_gap_error"]
                ),
                "target_ground_truth_label": row[
                    "target_ground_truth_label"
                ],
                "donor_ground_truth_label": row[
                    "donor_ground_truth_label"
                ],
            }
        )

    for algorithm in result:
        result[algorithm].sort(
            key=lambda r: (
                r["donor_distance"],
                r["target_function"],
            )
        )

    return dict(result)


def compute_gate_thresholds(
    selected_rows: list[dict[str, Any]],
    quantiles: list[float] = GATE_QUANTILES,
) -> list[dict[str, Any]]:
    distances = np.asarray(
        [r["donor_distance"] for r in selected_rows],
        dtype=float,
    )

    if len(distances) == 0:
        return []

    rows = []
    for q in quantiles:
        threshold = float(np.quantile(distances, q))
        rows.append(
            {
                "gate_quantile": q,
                "gate_label": quantile_label(q),
                "distance_threshold": threshold,
                "native_selected_count": len(selected_rows),
                "retained_selected_count": int(
                    np.sum(distances <= threshold + 1e-12)
                ),
            }
        )

    return rows


def retained_targets(
    selected_rows: list[dict[str, Any]],
    threshold: float,
) -> set[str]:
    return {
        r["target_function"]
        for r in selected_rows
        if r["donor_distance"] <= threshold + 1e-12
    }


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def gate_quality_rows(
    donor_rows: list[dict[str, str]],
    selected_by_algorithm: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    total_targets_by_algorithm = defaultdict(set)
    native_selected_by_algorithm = defaultdict(set)

    for row in donor_rows:
        algorithm = row["algorithm"]
        target = row["target_function"]
        total_targets_by_algorithm[algorithm].add(target)
        if row["selection_status"] == "selected":
            native_selected_by_algorithm[algorithm].add(target)

    output = []

    for algorithm in sorted(selected_by_algorithm):
        selected = selected_by_algorithm[algorithm]
        thresholds = compute_gate_thresholds(selected)

        for gate in thresholds:
            threshold = float(gate["distance_threshold"])
            retained = [
                r
                for r in selected
                if r["donor_distance"] <= threshold + 1e-12
            ]

            arm_matches = [
                1.0 if r["best_reward_arm_agreement"] else 0.0
                for r in retained
            ]
            reward_gap_errors = [
                float(r["abs_reward_gap_error"])
                for r in retained
            ]
            delta_errors = [
                float(r["abs_architecture_delta_error_percent"])
                for r in retained
            ]
            distances = [
                float(r["donor_distance"])
                for r in retained
            ]

            total_targets = len(total_targets_by_algorithm[algorithm])
            native_selected = len(
                native_selected_by_algorithm[algorithm]
            )

            output.append(
                {
                    "algorithm": algorithm,
                    "gate_quantile": gate["gate_quantile"],
                    "gate_label": gate["gate_label"],
                    "distance_threshold": threshold,
                    "target_count": total_targets,
                    "native_selected_count": native_selected,
                    "native_coverage": (
                        native_selected / total_targets
                        if total_targets
                        else float("nan")
                    ),
                    "gate_selected_count": len(retained),
                    "gate_coverage": (
                        len(retained) / total_targets
                        if total_targets
                        else float("nan")
                    ),
                    "best_reward_arm_agreement_rate": mean_or_nan(
                        arm_matches
                    ),
                    "mean_abs_reward_gap_error": mean_or_nan(
                        reward_gap_errors
                    ),
                    "median_abs_reward_gap_error": median_or_nan(
                        reward_gap_errors
                    ),
                    "mean_abs_architecture_delta_error_percent": (
                        mean_or_nan(delta_errors)
                    ),
                    "median_abs_architecture_delta_error_percent": (
                        median_or_nan(delta_errors)
                    ),
                    "mean_donor_distance": mean_or_nan(distances),
                    "median_donor_distance": median_or_nan(distances),
                }
            )

    return output


def gate_transfer_rows(
    donor_rows: list[dict[str, str]],
    transfer_rows: list[dict[str, str]],
    selected_by_algorithm: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    donor_map = donor_rows_by_key(donor_rows)
    transfer_map = transfer_rows_by_key(transfer_rows)

    algorithms = sorted(
        {row["algorithm"] for row in donor_rows}
    )
    weights = sorted(
        {
            float(row["equivalent_observation_weight"])
            for row in transfer_rows
        }
    )
    horizons = sorted(
        {int(row["horizon"]) for row in transfer_rows}
    )

    targets_by_algorithm: dict[str, list[str]] = defaultdict(list)
    for row in donor_rows:
        targets_by_algorithm[row["algorithm"]].append(
            row["target_function"]
        )
    for algorithm in targets_by_algorithm:
        targets_by_algorithm[algorithm] = sorted(
            set(targets_by_algorithm[algorithm])
        )

    output = []

    for algorithm in algorithms:
        selected = selected_by_algorithm.get(algorithm, [])
        gates = compute_gate_thresholds(selected)

        for gate in gates:
            threshold = float(gate["distance_threshold"])
            accepted_targets = retained_targets(
                selected,
                threshold,
            )

            accepted_quality_rows = [
                r
                for r in selected
                if r["target_function"] in accepted_targets
            ]
            accepted_arm_match = mean_or_nan(
                [
                    1.0 if r["best_reward_arm_agreement"] else 0.0
                    for r in accepted_quality_rows
                ]
            )
            accepted_reward_gap_error = mean_or_nan(
                [
                    float(r["abs_reward_gap_error"])
                    for r in accepted_quality_rows
                ]
            )

            for weight in weights:
                for horizon in horizons:
                    gains_all = []
                    gains_transfer_only = []
                    optimal_rates = []
                    first_optimal = []

                    for target in targets_by_algorithm[algorithm]:
                        donor = donor_map[(algorithm, target)]
                        transfer = transfer_map[
                            (algorithm, target, weight, horizon)
                        ]

                        apply_transfer = (
                            donor["selection_status"] == "selected"
                            and target in accepted_targets
                        )

                        if apply_transfer:
                            gain = float(
                                transfer[
                                    "mean_latency_gain_pct_vs_no_transfer"
                                ]
                            )
                            optimal = float(
                                transfer[
                                    "mean_optimal_reward_arm_selection_rate"
                                ]
                            )
                            first = float(
                                transfer[
                                    "first_arm_optimal_probability"
                                ]
                            )
                            gains_transfer_only.append(gain)
                        else:
                            # Gate rejects the donor -> exact no-transfer
                            # baseline. Its relative gain is therefore zero.
                            gain = 0.0

                            # The transfer CSV does not store baseline-only
                            # optimal-arm rates as a separate row. These two
                            # metrics are therefore left as NaN at aggregate
                            # level when the gate abstains. The main gate
                            # decision metric is latency gain vs baseline.
                            optimal = float("nan")
                            first = float("nan")

                        gains_all.append(gain)
                        if math.isfinite(optimal):
                            optimal_rates.append(optimal)
                        if math.isfinite(first):
                            first_optimal.append(first)

                    total_targets = len(
                        targets_by_algorithm[algorithm]
                    )
                    coverage = (
                        len(accepted_targets) / total_targets
                        if total_targets
                        else float("nan")
                    )

                    output.append(
                        {
                            "algorithm": algorithm,
                            "gate_quantile": gate["gate_quantile"],
                            "gate_label": gate["gate_label"],
                            "distance_threshold": threshold,
                            "equivalent_observation_weight": weight,
                            "horizon": horizon,
                            "target_count": total_targets,
                            "gate_selected_count": len(
                                accepted_targets
                            ),
                            "gate_coverage": coverage,
                            "best_reward_arm_agreement_rate_retained": (
                                accepted_arm_match
                            ),
                            "mean_abs_reward_gap_error_retained": (
                                accepted_reward_gap_error
                            ),
                            "macro_mean_latency_gain_pct_all_targets": (
                                mean_or_nan(gains_all)
                            ),
                            "macro_median_latency_gain_pct_all_targets": (
                                median_or_nan(gains_all)
                            ),
                            "target_positive_gain_rate_all_targets": (
                                mean_or_nan(
                                    [
                                        1.0 if x > 0 else 0.0
                                        for x in gains_all
                                    ]
                                )
                            ),
                            "macro_mean_latency_gain_pct_transfer_only": (
                                mean_or_nan(gains_transfer_only)
                            ),
                            "target_positive_gain_rate_transfer_only": (
                                mean_or_nan(
                                    [
                                        1.0 if x > 0 else 0.0
                                        for x in gains_transfer_only
                                    ]
                                )
                            ),
                            "mean_optimal_reward_arm_selection_rate_transfer_only": (
                                mean_or_nan(optimal_rates)
                            ),
                            "mean_first_arm_optimal_probability_transfer_only": (
                                mean_or_nan(first_optimal)
                            ),
                        }
                    )

    return output


def spearman_safe(
    x: list[float],
    y: list[float],
) -> tuple[float, float]:
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")

    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def correlation_rows(
    donor_rows: list[dict[str, str]],
    transfer_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    transfer_map = transfer_rows_by_key(transfer_rows)

    weights = sorted(
        {
            float(row["equivalent_observation_weight"])
            for row in transfer_rows
        }
    )
    horizons = sorted(
        {int(row["horizon"]) for row in transfer_rows}
    )

    output = []

    for algorithm in sorted(
        {row["algorithm"] for row in donor_rows}
    ):
        selected = [
            row
            for row in donor_rows
            if row["algorithm"] == algorithm
            and row["selection_status"] == "selected"
        ]

        distance = [
            float(row["donor_distance"])
            for row in selected
        ]
        arm_correct = [
            1.0 if as_bool(row["best_reward_arm_agreement"]) else 0.0
            for row in selected
        ]
        reward_gap_error = [
            float(row["abs_reward_gap_error"])
            for row in selected
        ]
        delta_error = [
            float(row["abs_architecture_delta_error_percent"])
            for row in selected
        ]

        for metric, values in [
            ("best_reward_arm_agreement", arm_correct),
            ("abs_reward_gap_error", reward_gap_error),
            ("abs_architecture_delta_error_percent", delta_error),
        ]:
            rho, p = spearman_safe(distance, values)
            output.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "equivalent_observation_weight": "",
                    "horizon": "",
                    "selected_count": len(selected),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

        for weight in weights:
            for horizon in horizons:
                gains = [
                    float(
                        transfer_map[
                            (
                                algorithm,
                                row["target_function"],
                                weight,
                                horizon,
                            )
                        ][
                            "mean_latency_gain_pct_vs_no_transfer"
                        ]
                    )
                    for row in selected
                ]

                rho, p = spearman_safe(distance, gains)
                output.append(
                    {
                        "algorithm": algorithm,
                        "metric": "latency_gain_pct_vs_no_transfer",
                        "equivalent_observation_weight": weight,
                        "horizon": horizon,
                        "selected_count": len(selected),
                        "spearman_rho": rho,
                        "p_value": p,
                    }
                )

    return output


def scatter_distance_quality(
    selected_by_algorithm: dict[str, list[dict[str, Any]]],
    figures_dir: Path,
) -> None:
    for algorithm, rows in sorted(selected_by_algorithm.items()):
        x = np.asarray(
            [r["donor_distance"] for r in rows],
            dtype=float,
        )
        y = np.asarray(
            [r["abs_reward_gap_error"] for r in rows],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.scatter(x, y, alpha=0.8)

        # Donor distance is strictly non-negative and can span several orders
        # of magnitude, especially for leave-one-out K-Means outliers.
        # symlog keeps zero representable while making large-distance outliers
        # readable. Raw values remain in the CSV.
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_title(
            f"{algorithm.upper()} donor distance vs reward-gap error"
        )
        ax.set_xlabel("Donor distance")
        ax.set_ylabel("|target reward gap - donor reward gap|")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figures_dir
            / f"{algorithm}-donor-distance-vs-reward-gap-error.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir
            / f"{algorithm}-donor-distance-vs-reward-gap-error.svg",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_gate_quality(
    gate_quality: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    for algorithm in sorted(
        {row["algorithm"] for row in gate_quality}
    ):
        rows = [
            row
            for row in gate_quality
            if row["algorithm"] == algorithm
        ]
        rows.sort(key=lambda r: float(r["gate_coverage"]))

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.plot(
            [float(r["gate_coverage"]) for r in rows],
            [
                float(r["best_reward_arm_agreement_rate"])
                for r in rows
            ],
            marker="o",
        )
        for row in rows:
            ax.annotate(
                row["gate_label"],
                (
                    float(row["gate_coverage"]),
                    float(row["best_reward_arm_agreement_rate"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(
            f"{algorithm.upper()} confidence gate: coverage vs donor correctness"
        )
        ax.set_xlabel("Transfer coverage")
        ax.set_ylabel("Best-arm agreement rate")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figures_dir
            / f"{algorithm}-confidence-gate-donor-correctness.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir
            / f"{algorithm}-confidence-gate-donor-correctness.svg",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_gate_transfer(
    gate_transfer: list[dict[str, Any]],
    figures_dir: Path,
    primary_horizon: int,
) -> None:
    for algorithm in sorted(
        {row["algorithm"] for row in gate_transfer}
    ):
        rows_algorithm = [
            row
            for row in gate_transfer
            if row["algorithm"] == algorithm
            and int(row["horizon"]) == primary_horizon
        ]

        weights = sorted(
            {
                float(r["equivalent_observation_weight"])
                for r in rows_algorithm
            }
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        for weight in weights:
            rows = [
                r
                for r in rows_algorithm
                if float(r["equivalent_observation_weight"]) == weight
            ]
            rows.sort(key=lambda r: float(r["gate_coverage"]))

            ax.plot(
                [float(r["gate_coverage"]) for r in rows],
                [
                    float(
                        r[
                            "macro_mean_latency_gain_pct_all_targets"
                        ]
                    )
                    for r in rows
                ],
                marker="o",
                label=f"w={weight:g}",
            )

        ax.axhline(0.0, linewidth=1, linestyle="--")
        ax.set_title(
            f"{algorithm.upper()} confidence gate: "
            f"latency gain at H={primary_horizon}"
        )
        ax.set_xlabel("Transfer coverage")
        ax.set_ylabel("Mean latency gain vs no-transfer (%)")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir
            / f"{algorithm}-confidence-gate-latency-gain-h{primary_horizon}.png",
            dpi=180,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir
            / f"{algorithm}-confidence-gate-latency-gain-h{primary_horizon}.svg",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--donor-selection", required=True, type=Path)
    p.add_argument("--transfer-results", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--primary-horizon", type=int, default=10)
    args = p.parse_args()

    donor_rows = read_csv(args.donor_selection.resolve())
    transfer_rows = read_csv(args.transfer_results.resolve())

    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    selected_by_algorithm = build_selected_donor_rows(
        donor_rows
    )

    quality = gate_quality_rows(
        donor_rows,
        selected_by_algorithm,
    )
    transfer = gate_transfer_rows(
        donor_rows,
        transfer_rows,
        selected_by_algorithm,
    )
    correlations = correlation_rows(
        donor_rows,
        transfer_rows,
    )

    write_csv(
        out / "donor-confidence-gate-quality.csv",
        quality,
    )
    write_csv(
        out / "donor-confidence-gate-transfer.csv",
        transfer,
    )
    write_csv(
        out / "donor-distance-correlations.csv",
        correlations,
    )

    # Raw selected donor rows are exported as well so the thesis can reproduce
    # every scatter/threshold calculation directly from CSV.
    selected_flat = []
    for algorithm in sorted(selected_by_algorithm):
        selected_flat.extend(selected_by_algorithm[algorithm])
    write_csv(
        out / "selected-donor-distance-quality.csv",
        selected_flat,
    )

    scatter_distance_quality(
        selected_by_algorithm,
        figures,
    )
    plot_gate_quality(
        quality,
        figures,
    )
    plot_gate_transfer(
        transfer,
        figures,
        primary_horizon=args.primary_horizon,
    )

    manifest = {
        "schema_version": 1,
        "experiment": "distance-based donor confidence analysis",
        "purpose": (
            "Test whether donor distance is informative enough to support "
            "confidence-gated transfer without modifying the UCB1 formula."
        ),
        "gate_quantiles": GATE_QUANTILES,
        "gate_policy": (
            "transfer only when donor_distance <= the algorithm-specific "
            "distance quantile threshold; rejected targets use no-transfer"
        ),
        "important_notes": [
            "K-Means and DBSCAN distance magnitudes are not compared across algorithms.",
            "DBSCAN native abstention is preserved before the distance gate.",
            "Quantiles are evaluated as a predefined sensitivity grid and are not automatically optimized.",
            "Ground truth is used only post-hoc to measure donor quality.",
            "The UCB1 transfer formula is unchanged.",
        ],
        "primary_horizon": args.primary_horizon,
        "inputs": {
            "donor_selection": str(
                args.donor_selection.resolve()
            ),
            "transfer_results": str(
                args.transfer_results.resolve()
            ),
        },
        "outputs": [
            "selected-donor-distance-quality.csv",
            "donor-confidence-gate-quality.csv",
            "donor-confidence-gate-transfer.csv",
            "donor-distance-correlations.csv",
            "figures/",
        ],
    }

    (out / "donor-confidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"output={out}")
    print("\nGate quality:")
    for row in quality:
        print(
            f"{row['algorithm']:<7} "
            f"{row['gate_label']} "
            f"coverage={float(row['gate_coverage']):.4f} "
            f"best-arm-match="
            f"{float(row['best_reward_arm_agreement_rate']):.4f} "
            f"reward-gap-error="
            f"{float(row['mean_abs_reward_gap_error']):.4f}"
        )

    print(f"\nTransfer gain @ H={args.primary_horizon}:")
    primary = [
        row
        for row in transfer
        if int(row["horizon"]) == args.primary_horizon
    ]
    primary.sort(
        key=lambda r: (
            r["algorithm"],
            float(r["equivalent_observation_weight"]),
            float(r["gate_quantile"]),
        )
    )

    for row in primary:
        print(
            f"{row['algorithm']:<7} "
            f"w={float(row['equivalent_observation_weight']):<4g} "
            f"{row['gate_label']} "
            f"coverage={float(row['gate_coverage']):.4f} "
            f"gain="
            f"{float(row['macro_mean_latency_gain_pct_all_targets']):+.4f}% "
            f"retained-match="
            f"{float(row['best_reward_arm_agreement_rate_retained']):.4f}"
        )

    print("\nDistance correlations:")
    for row in correlations:
        if row["metric"] == "latency_gain_pct_vs_no_transfer":
            if int(row["horizon"]) != args.primary_horizon:
                continue
        print(
            f"{row['algorithm']:<7} "
            f"{row['metric']:<40} "
            f"w={row['equivalent_observation_weight'] or '-':<4} "
            f"H={row['horizon'] or '-':<3} "
            f"rho={float(row['spearman_rho']):+.4f} "
            f"p={float(row['p_value']):.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
