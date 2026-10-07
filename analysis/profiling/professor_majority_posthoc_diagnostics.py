#!/usr/bin/env python3
"""
Post-process professor-compliant LOFO donor-selection results.

This script DOES NOT rerun K-Means. It reads the already generated
`professor-donor-lofo-per-target.csv` and produces corrected, auditable summaries.

Why this post-processing exists
-------------------------------
The original V3 experiment stored an empty string for majority ties. In its
classification helper, coverage was computed as `prediction != "abstain"`,
therefore empty-string abstentions were incorrectly counted as covered.
Strict accuracy / balanced recall were still conservative because an empty
prediction never matched a real class, but coverage, covered accuracy and
abstain rate need to be recomputed correctly.

This post-processor also adds the missing class-stratified donor analysis.
That is necessary because large thresholds make `architecture-independent`
the dominant class, so aggregate accuracy/error can look good while directional
(x86/ARM-sensitive) targets are poorly served.

Outputs
-------
professor-majority-classification-summary-corrected.csv
professor-majority-prediction-distribution.csv
professor-donor-quality-by-target-class.csv
professor-donor-quality-directional.csv
professor-majority-vs-unfiltered-by-target-class.csv
professor-majority-posthoc-report.md
professor-majority-posthoc-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]

DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_prediction(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).isin(LABELS)


def classification_metrics(frame: pd.DataFrame) -> dict:
    y_true = frame["target_label"].astype(str).to_numpy()
    y_pred = (
        frame["predicted_majority_label"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    covered = np.isin(y_pred, LABELS)

    recalls = {}
    precisions = {}
    f1s = []

    for label in LABELS:
        truth = y_true == label
        pred = y_pred == label

        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        recalls[label] = recall
        precisions[label] = precision
        f1s.append(f1)

    directional_mask = np.isin(
        y_true,
        list(DIRECTIONAL),
    )

    directional_covered = directional_mask & covered

    return {
        "classification_coverage": float(np.mean(covered)),
        "abstain_rate": float(np.mean(~covered)),
        "strict_accuracy": float(np.mean(y_true == y_pred)),
        "covered_accuracy": (
            float(np.mean(y_true[covered] == y_pred[covered]))
            if np.any(covered)
            else np.nan
        ),
        "balanced_accuracy": float(
            np.mean(list(recalls.values()))
        ),
        "macro_f1": float(np.mean(f1s)),
        "directional_strict_accuracy": (
            float(
                np.mean(
                    y_true[directional_mask]
                    == y_pred[directional_mask]
                )
            )
            if np.any(directional_mask)
            else np.nan
        ),
        "directional_covered_accuracy": (
            float(
                np.mean(
                    y_true[directional_covered]
                    == y_pred[directional_covered]
                )
            )
            if np.any(directional_covered)
            else np.nan
        ),
        "directional_coverage": (
            float(
                np.mean(
                    covered[directional_mask]
                )
            )
            if np.any(directional_mask)
            else np.nan
        ),
        "x86_recall": recalls["x86-preferred"],
        "independent_recall": recalls[
            "architecture-independent"
        ],
        "arm_recall": recalls["arm-preferred"],
        "x86_precision": precisions["x86-preferred"],
        "independent_precision": precisions[
            "architecture-independent"
        ],
        "arm_precision": precisions["arm-preferred"],
        "pred_x86": int(
            np.sum(y_pred == "x86-preferred")
        ),
        "pred_independent": int(
            np.sum(
                y_pred == "architecture-independent"
            )
        ),
        "pred_arm": int(
            np.sum(y_pred == "arm-preferred")
        ),
        "pred_abstain": int(np.sum(~covered)),
    }


def build_corrected_classification(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    majority = detail[
        detail["selection_mode"] == "majority_filtered"
    ].copy()

    per_seed_rows = []

    for (
        config,
        threshold,
        seed,
    ), group in majority.groupby(
        [
            "configuration",
            "threshold_percent",
            "seed",
        ]
    ):
        per_seed_rows.append(
            {
                "configuration": config,
                "threshold_percent": float(threshold),
                "seed": int(seed),
                **classification_metrics(group),
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)

    metric_cols = [
        c
        for c in per_seed.columns
        if c
        not in {
            "configuration",
            "threshold_percent",
            "seed",
        }
    ]

    agg_spec = {}
    for col in metric_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")
        if col not in {
            "pred_x86",
            "pred_independent",
            "pred_arm",
            "pred_abstain",
        }:
            agg_spec[f"{col}_std"] = (col, "std")

    summary = (
        per_seed.groupby(
            [
                "configuration",
                "threshold_percent",
            ],
            as_index=False,
        )
        .agg(**agg_spec)
    )

    prediction_distribution = (
        majority.assign(
            valid_prediction=is_valid_prediction(
                majority["predicted_majority_label"]
            )
        )
        .groupby(
            [
                "configuration",
                "threshold_percent",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            target_count=("target_function", "count"),
            true_x86=(
                "target_label",
                lambda s: int(
                    np.sum(
                        s.astype(str).to_numpy()
                        == "x86-preferred"
                    )
                ),
            ),
            true_independent=(
                "target_label",
                lambda s: int(
                    np.sum(
                        s.astype(str).to_numpy()
                        == "architecture-independent"
                    )
                ),
            ),
            true_arm=(
                "target_label",
                lambda s: int(
                    np.sum(
                        s.astype(str).to_numpy()
                        == "arm-preferred"
                    )
                ),
            ),
            pred_x86=(
                "predicted_majority_label",
                lambda s: int(
                    np.sum(
                        s.fillna("")
                        .astype(str)
                        .to_numpy()
                        == "x86-preferred"
                    )
                ),
            ),
            pred_independent=(
                "predicted_majority_label",
                lambda s: int(
                    np.sum(
                        s.fillna("")
                        .astype(str)
                        .to_numpy()
                        == "architecture-independent"
                    )
                ),
            ),
            pred_arm=(
                "predicted_majority_label",
                lambda s: int(
                    np.sum(
                        s.fillna("")
                        .astype(str)
                        .to_numpy()
                        == "arm-preferred"
                    )
                ),
            ),
            pred_abstain=(
                "valid_prediction",
                lambda s: int(np.sum(~s.to_numpy())),
            ),
        )
    )

    return summary, prediction_distribution


def donor_metrics(group: pd.DataFrame) -> dict:
    selected = group[
        group["selection_status"] == "selected"
    ].copy()

    result = {
        "target_rows": int(len(group)),
        "selected_rows": int(len(selected)),
        "selection_coverage": float(
            len(selected) / len(group)
        )
        if len(group)
        else np.nan,
        "mean_same_cluster_size": float(
            group["same_cluster_size"].mean()
        )
        if len(group)
        else np.nan,
        "mean_majority_share": float(
            group["cluster_majority_share"].mean()
        )
        if len(group)
        else np.nan,
    }

    if selected.empty:
        result.update(
            {
                "best_arm_agreement_rate": np.nan,
                "mean_abs_delta_error_percent": np.nan,
                "median_abs_delta_error_percent": np.nan,
                "mean_abs_reward_gap_error": np.nan,
                "median_abs_reward_gap_error": np.nan,
                "mean_candidate_pool_size": np.nan,
                "cross_language_donor_rate": np.nan,
            }
        )
        return result

    result.update(
        {
            "best_arm_agreement_rate": float(
                selected["best_arm_agreement"]
                .astype(float)
                .mean()
            ),
            "mean_abs_delta_error_percent": float(
                selected["abs_delta_error_percent"].mean()
            ),
            "median_abs_delta_error_percent": float(
                selected[
                    "abs_delta_error_percent"
                ].median()
            ),
            "mean_abs_reward_gap_error": float(
                selected["abs_reward_gap_error"].mean()
            ),
            "median_abs_reward_gap_error": float(
                selected[
                    "abs_reward_gap_error"
                ].median()
            ),
            "mean_candidate_pool_size": float(
                selected["candidate_pool_size"].mean()
            ),
            "cross_language_donor_rate": float(
                selected["cross_language_donor"]
                .astype(float)
                .mean()
            ),
        }
    )

    return result


def build_by_class(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_class_rows = []
    directional_rows = []

    for (
        config,
        threshold,
        mode,
        target_label,
    ), group in detail.groupby(
        [
            "configuration",
            "threshold_percent",
            "selection_mode",
            "target_label",
        ]
    ):
        by_class_rows.append(
            {
                "configuration": config,
                "threshold_percent": float(threshold),
                "selection_mode": mode,
                "target_class": target_label,
                **donor_metrics(group),
            }
        )

    by_class = pd.DataFrame(by_class_rows)

    directional = detail[
        detail["target_label"].isin(DIRECTIONAL)
    ].copy()

    for (
        config,
        threshold,
        mode,
    ), group in directional.groupby(
        [
            "configuration",
            "threshold_percent",
            "selection_mode",
        ]
    ):
        directional_rows.append(
            {
                "configuration": config,
                "threshold_percent": float(threshold),
                "selection_mode": mode,
                "target_class": "directional-only",
                **donor_metrics(group),
            }
        )

    return (
        by_class,
        pd.DataFrame(directional_rows),
    )


def compare_modes_by_class(
    by_class: pd.DataFrame,
) -> pd.DataFrame:
    filtered = (
        by_class[
            by_class["selection_mode"]
            == "majority_filtered"
        ]
        .set_index(
            [
                "configuration",
                "threshold_percent",
                "target_class",
            ]
        )
    )

    unfiltered = (
        by_class[
            by_class["selection_mode"]
            == "unfiltered_same_cluster"
        ]
        .set_index(
            [
                "configuration",
                "threshold_percent",
                "target_class",
            ]
        )
    )

    metrics = [
        "selection_coverage",
        "best_arm_agreement_rate",
        "mean_abs_delta_error_percent",
        "median_abs_delta_error_percent",
        "mean_abs_reward_gap_error",
        "median_abs_reward_gap_error",
        "mean_candidate_pool_size",
        "cross_language_donor_rate",
    ]

    rows = []

    for key in sorted(
        set(filtered.index)
        & set(unfiltered.index)
    ):
        config, threshold, target_class = key

        row = {
            "configuration": config,
            "threshold_percent": threshold,
            "target_class": target_class,
        }

        for metric in metrics:
            f = float(filtered.loc[key, metric])
            u = float(unfiltered.loc[key, metric])

            row[f"majority_{metric}"] = f
            row[f"unfiltered_{metric}"] = u
            row[
                f"delta_majority_minus_unfiltered_{metric}"
            ] = f - u

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    classification: pd.DataFrame,
    by_class: pd.DataFrame,
    directional: pd.DataFrame,
):
    lines = [
        "# Professor majority-selection post-hoc diagnostics",
        "",
        "## Correction",
        "",
        (
            "The original V3 terminal table counted empty-string majority ties "
            "as covered classifications. This post-processor corrects coverage, "
            "covered accuracy and abstain rate by treating only the three valid "
            "architecture labels as covered predictions."
        ),
        "",
        (
            "Strict accuracy, per-class recall and macro-F1 from V3 were already "
            "conservative because an empty prediction did not match any real "
            "class. They are recomputed here for consistency."
        ),
        "",
        "## Why class-stratified donor quality is required",
        "",
        (
            "At larger architecture thresholds the independent class becomes "
            "dominant. Aggregate strict accuracy and aggregate donor error can "
            "therefore improve even if directional x86/ARM targets are poorly "
            "served. The files emitted here separate x86-preferred, independent, "
            "arm-preferred and directional-only targets."
        ),
        "",
        "## Best balanced majority prediction by threshold",
        "",
    ]

    for threshold, block in classification.groupby(
        "threshold_percent"
    ):
        best = block.sort_values(
            [
                "balanced_accuracy_mean",
                "macro_f1_mean",
            ],
            ascending=[False, False],
        ).iloc[0]

        lines.append(
            f"- t={threshold:.1f}%: `{best['configuration']}`; "
            f"coverage={best['classification_coverage_mean']:.4f}, "
            f"balanced-acc={best['balanced_accuracy_mean']:.4f}, "
            f"macro-F1={best['macro_f1_mean']:.4f}, "
            f"x86-recall={best['x86_recall_mean']:.4f}, "
            f"independent-recall={best['independent_recall_mean']:.4f}, "
            f"arm-recall={best['arm_recall_mean']:.4f}"
        )

    lines += [
        "",
        "## Directional-only donor quality",
        "",
    ]

    filtered_directional = directional[
        directional["selection_mode"]
        == "majority_filtered"
    ]

    for threshold, block in filtered_directional.groupby(
        "threshold_percent"
    ):
        ranked = block.sort_values(
            [
                "best_arm_agreement_rate",
                "mean_abs_reward_gap_error",
                "mean_abs_delta_error_percent",
            ],
            ascending=[False, True, True],
        )

        best = ranked.iloc[0]

        lines.append(
            f"- t={threshold:.1f}%: `{best['configuration']}`; "
            f"coverage={best['selection_coverage']:.4f}, "
            f"best-arm={best['best_arm_agreement_rate']:.4f}, "
            f"mean |delta err|="
            f"{best['mean_abs_delta_error_percent']:.3f} pp, "
            f"mean reward-gap err="
            f"{best['mean_abs_reward_gap_error']:.4f}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Threshold selection must not be based on aggregate strict accuracy "
            "alone. Inspect balanced accuracy, all three class recalls, abstention "
            "coverage and especially donor quality on directional-only targets. "
            "The architecture threshold changes the operational donor filter, "
            "so this is a sensitivity analysis rather than a cosmetic relabeling."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(args):
    detail_path = args.detail_csv.resolve()
    detail = pd.read_csv(detail_path)

    required = {
        "configuration",
        "threshold_percent",
        "seed",
        "selection_mode",
        "selection_status",
        "target_function",
        "target_label",
        "predicted_majority_label",
        "same_cluster_size",
        "cluster_majority_share",
        "candidate_pool_size",
        "best_arm_agreement",
        "abs_delta_error_percent",
        "abs_reward_gap_error",
        "cross_language_donor",
    }

    missing = sorted(required - set(detail.columns))
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        classification,
        prediction_distribution,
    ) = build_corrected_classification(detail)

    by_class, directional = build_by_class(detail)

    comparison = compare_modes_by_class(by_class)

    classification_path = (
        args.output_dir
        / "professor-majority-classification-summary-corrected.csv"
    )
    distribution_path = (
        args.output_dir
        / "professor-majority-prediction-distribution.csv"
    )
    by_class_path = (
        args.output_dir
        / "professor-donor-quality-by-target-class.csv"
    )
    directional_path = (
        args.output_dir
        / "professor-donor-quality-directional.csv"
    )
    comparison_path = (
        args.output_dir
        / "professor-majority-vs-unfiltered-by-target-class.csv"
    )
    report_path = (
        args.output_dir
        / "professor-majority-posthoc-report.md"
    )
    manifest_path = (
        args.output_dir
        / "professor-majority-posthoc-manifest.json"
    )

    classification.to_csv(
        classification_path,
        index=False,
    )
    prediction_distribution.to_csv(
        distribution_path,
        index=False,
    )
    by_class.to_csv(
        by_class_path,
        index=False,
    )
    directional.to_csv(
        directional_path,
        index=False,
    )
    comparison.to_csv(
        comparison_path,
        index=False,
    )

    write_report(
        path=report_path,
        classification=classification,
        by_class=by_class,
        directional=directional,
    )

    manifest = {
        "source_detail_csv": {
            "path": str(detail_path),
            "sha256": sha256_file(detail_path),
        },
        "correction": (
            "Only predictions equal to one of the three valid architecture "
            "labels count as covered; empty-string majority ties are abstentions."
        ),
        "additional_analysis": (
            "Donor quality stratified by target architecture class and "
            "directional-only targets."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("CORRECTED MAJORITY CLASSIFICATION")
    print(
        classification.sort_values(
            [
                "threshold_percent",
                "balanced_accuracy_mean",
            ],
            ascending=[True, False],
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("DIRECTIONAL-ONLY DONOR QUALITY")
    print(
        directional[
            directional["selection_mode"]
            == "majority_filtered"
        ]
        .sort_values(
            [
                "threshold_percent",
                "best_arm_agreement_rate",
                "mean_abs_reward_gap_error",
            ],
            ascending=[True, False, True],
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"classification={classification_path}")
    print(f"prediction_distribution={distribution_path}")
    print(f"by_class={by_class_path}")
    print(f"directional={directional_path}")
    print(f"comparison={comparison_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Correct and stratify professor-compliant majority donor results."
        )
    )

    parser.add_argument(
        "--detail-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
