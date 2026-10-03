#!/usr/bin/env python3
"""
Post-process the outputs of clustering_lofo_classification.py.

The goal is to make explicit *why* a clustering configuration succeeds or
fails as a 3-class predictor (x86-preferred / architecture-independent /
arm-preferred), without changing the clustering experiment itself.

Inputs
------
An output directory produced by clustering_lofo_classification.py, containing
at least ``lofo-per-target.csv``.

Outputs
-------
- lofo-confidence-per-target.csv
- lofo-diagnostics-summary.csv

The diagnostics add:
- majority-class baseline and gain over that baseline;
- prediction distribution by class;
- strict accuracy on directional targets only;
- coverage split between directional and independent targets;
- majority margin inside the assigned cluster;
- normalized label entropy inside the assigned cluster.

No arbitrary confidence threshold is applied: margin/entropy are exported as
continuous diagnostics so a threshold can be discussed with the supervisors
later.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


X86 = "x86-preferred"
INDEPENDENT = "architecture-independent"
ARM = "arm-preferred"
LABELS = (X86, INDEPENDENT, ARM)


class DiagnosticsError(RuntimeError):
    pass


def _as_int(value: Any) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return int(float(value))


def _as_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _normalized_entropy(counts: list[int]) -> float:
    """Entropy in [0,1] over the three ground-truth labels."""
    total = sum(counts)
    if total <= 0:
        return float("nan")
    h = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        h -= p * math.log(p)
    return h / math.log(len(LABELS))


def enrich_row(row: dict[str, str]) -> dict[str, Any]:
    counts = [
        _as_int(row.get("cluster_x86_count")),
        _as_int(row.get("cluster_independent_count")),
        _as_int(row.get("cluster_arm_count")),
    ]
    size = sum(counts)
    ranked = sorted(counts, reverse=True)
    majority_margin = (
        (ranked[0] - ranked[1]) / size if size > 0 and len(ranked) >= 2 else float("nan")
    )
    majority_fraction = ranked[0] / size if size > 0 else float("nan")

    truth = row.get("target_ground_truth_label", "")
    status = row.get("prediction_status", "")
    predicted = row.get("cluster_majority_label", "") if status == "predicted" else ""
    prediction_correct = status == "predicted" and predicted == truth

    return {
        **row,
        "cluster_majority_fraction_recomputed": majority_fraction,
        "cluster_majority_margin": majority_margin,
        "cluster_label_entropy_normalized": _normalized_entropy(counts),
        "target_is_directional": truth in {X86, ARM},
        "target_is_independent": truth == INDEPENDENT,
        "prediction_correct_recomputed": prediction_correct,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise DiagnosticsError(f"missing input file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DiagnosticsError(f"empty input file: {path}")
    required = {
        "algorithm",
        "feature_set",
        "target_ground_truth_label",
        "prediction_status",
        "cluster_majority_label",
        "cluster_x86_count",
        "cluster_independent_count",
        "cluster_arm_count",
    }
    missing = required - set(rows[0])
    if missing:
        raise DiagnosticsError(f"missing columns in {path}: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise DiagnosticsError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({(r["algorithm"], r["feature_set"]) for r in rows})
    output: list[dict[str, Any]] = []

    for algorithm, feature_set in groups:
        subset = [
            r for r in rows
            if r["algorithm"] == algorithm and r["feature_set"] == feature_set
        ]
        target_count = len(subset)
        predicted = [r for r in subset if r["prediction_status"] == "predicted"]
        abstained = [r for r in subset if r["prediction_status"] != "predicted"]
        correct = [r for r in predicted if bool(r["prediction_correct_recomputed"])]

        true_counts = Counter(r["target_ground_truth_label"] for r in subset)
        pred_counts = Counter(r["cluster_majority_label"] for r in predicted)
        baseline_label, baseline_correct = sorted(
            true_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0]
        baseline_acc = baseline_correct / target_count
        strict_acc = len(correct) / target_count

        directional = [r for r in subset if bool(r["target_is_directional"])]
        directional_pred = [r for r in directional if r["prediction_status"] == "predicted"]
        directional_correct = [
            r for r in directional_pred if bool(r["prediction_correct_recomputed"])
        ]
        independent = [r for r in subset if bool(r["target_is_independent"])]
        independent_pred = [r for r in independent if r["prediction_status"] == "predicted"]
        independent_correct = [
            r for r in independent_pred if bool(r["prediction_correct_recomputed"])
        ]

        reason_counts = Counter(r.get("abstention_reason", "") for r in abstained)

        output.append(
            {
                "algorithm": algorithm,
                "feature_set": feature_set,
                "target_count": target_count,
                "predicted_count": len(predicted),
                "abstention_count": len(abstained),
                "prediction_coverage": _safe_div(len(predicted), target_count),
                "strict_accuracy_all_targets": strict_acc,
                "majority_baseline_label": baseline_label,
                "majority_baseline_accuracy": baseline_acc,
                "strict_accuracy_gain_over_majority_baseline": strict_acc - baseline_acc,
                "true_x86_count": true_counts[X86],
                "true_independent_count": true_counts[INDEPENDENT],
                "true_arm_count": true_counts[ARM],
                "pred_x86_count": pred_counts[X86],
                "pred_independent_count": pred_counts[INDEPENDENT],
                "pred_arm_count": pred_counts[ARM],
                "directional_target_count": len(directional),
                "directional_prediction_count": len(directional_pred),
                "directional_prediction_coverage": _safe_div(
                    len(directional_pred), len(directional)
                ),
                "directional_correct_count": len(directional_correct),
                "directional_accuracy_strict": _safe_div(
                    len(directional_correct), len(directional)
                ),
                "directional_accuracy_on_predictions": _safe_div(
                    len(directional_correct), len(directional_pred)
                ),
                "independent_target_count": len(independent),
                "independent_prediction_count": len(independent_pred),
                "independent_prediction_coverage": _safe_div(
                    len(independent_pred), len(independent)
                ),
                "independent_correct_count": len(independent_correct),
                "independent_recall_strict": _safe_div(
                    len(independent_correct), len(independent)
                ),
                "mean_assigned_cluster_purity": _mean(
                    _as_float(r.get("cluster_purity")) for r in predicted
                ),
                "median_assigned_cluster_purity": _median(
                    _as_float(r.get("cluster_purity")) for r in predicted
                ),
                "mean_majority_margin": _mean(
                    _as_float(r.get("cluster_majority_margin")) for r in predicted
                ),
                "median_majority_margin": _median(
                    _as_float(r.get("cluster_majority_margin")) for r in predicted
                ),
                "mean_cluster_label_entropy_normalized": _mean(
                    _as_float(r.get("cluster_label_entropy_normalized")) for r in predicted
                ),
                "median_cluster_label_entropy_normalized": _median(
                    _as_float(r.get("cluster_label_entropy_normalized")) for r in predicted
                ),
                "mean_donor_pool_size": _mean(
                    _as_float(r.get("donor_pool_size")) for r in predicted
                ),
                "ambiguous_majority_count": reason_counts["ambiguous_cluster_majority"],
                "no_core_points_count": reason_counts["no_core_points"],
                "no_core_within_eps_count": reason_counts["no_core_within_eps"],
            }
        )

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        required=True,
        type=Path,
        help="directory produced by clustering_lofo_classification.py",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.expanduser().resolve()
    source = analysis_dir / "lofo-per-target.csv"
    raw_rows = read_csv(source)
    enriched = [enrich_row(row) for row in raw_rows]
    summary = summarize(enriched)

    write_csv(analysis_dir / "lofo-confidence-per-target.csv", enriched)
    write_csv(analysis_dir / "lofo-diagnostics-summary.csv", summary)

    print(f"rows={len(enriched)} groups={len(summary)} output={analysis_dir}")
    for row in summary:
        print(
            f"{row['feature_set']:<42} {row['algorithm']:<7} "
            f"coverage={row['prediction_coverage']:.4f} "
            f"strict={row['strict_accuracy_all_targets']:.4f} "
            f"baseline={row['majority_baseline_accuracy']:.4f} "
            f"gain={row['strict_accuracy_gain_over_majority_baseline']:+.4f} "
            f"directional={row['directional_accuracy_strict']:.4f} "
            f"pred=[x86:{row['pred_x86_count']},ind:{row['pred_independent_count']},arm:{row['pred_arm_count']}] "
            f"margin={row['mean_majority_margin']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
