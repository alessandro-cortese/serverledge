#!/usr/bin/env python3
"""
Robust validation of the final directional-vote candidates.

This script does NOT tune clustering or vote parameters. It validates three
already-predeclared candidates from `directional_vote_refinement_study.py`:

- cluster_majority       : professor baseline
- weighted_vote_p2       : best agreement-oriented refinement
- local_k5_majority      : best regret/tail-oriented refinement

The bootstrap resampling unit is the FUNCTION/TARGET, not individual seed rows,
so the five K-Means seed evaluations for one target remain together.

Primary questions:
1. Does weighted_vote_p2 really improve correctness over cluster_majority?
2. Is the gain obtained without sacrificing coverage, speedup, or regret?
3. Is local_k5 preferable because of lower regret / p90 regret?
4. Are the differences robust under target-level bootstrap uncertainty?

Outputs
-------
vote-validation-summary.csv
vote-validation-paired-bootstrap.csv
vote-validation-per-class.csv
vote-validation-stability.csv
vote-validation-stability-per-target.csv
vote-validation-report.md
vote-validation-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = [
    "cluster_majority",
    "weighted_vote_p2",
    "local_k5_majority",
]

DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
}

EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_best(x86_ms: float, arm_ms: float) -> str:
    return (
        "x86-preferred"
        if x86_ms <= arm_ms
        else "arm-preferred"
    )


def geometric_mean(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().to_numpy(float)

    x = x[x > 0]

    if not len(x):
        return np.nan

    return float(
        np.exp(
            np.mean(
                np.log(x)
            )
        )
    )


def q90(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().to_numpy(float)

    if not len(x):
        return np.nan

    return float(
        np.quantile(
            x,
            0.90,
        )
    )


def prepare(
        detail_path: Path,
        preferences_path: Path,
) -> pd.DataFrame:
    detail = pd.read_csv(detail_path)
    prefs = pd.read_csv(preferences_path)

    required_detail = {
        "variant",
        "seed",
        "target_function",
        "selection_status",
        "predicted_direction",
        "target_is_directional",
        "directional_preference_agreement",
        "raw_best_arm_agreement",
        "realized_speedup_factor",
        "regret_percent",
        "multiplicative_mismatch_factor",
    }

    missing = required_detail - set(detail.columns)
    if missing:
        raise SystemExit(
            f"Detail missing columns: {sorted(missing)}"
        )

    required_pref = {
        "function_name",
        "architecture_preference",
        "x86_duration_ms",
        "arm_duration_ms",
    }

    missing = required_pref - set(prefs.columns)
    if missing:
        raise SystemExit(
            f"Preferences missing columns: {sorted(missing)}"
        )

    truth = prefs[
        [
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
        ]
    ].copy()

    # IMPORTANT:
    # directional_vote_refinement_study.py already writes a column named
    # `target_raw_best`.  The previous version of this validation script
    # created another `target_raw_best` before merging, so pandas renamed the
    # two columns to target_raw_best_x / target_raw_best_y and the subsequent
    # lookup of `target_raw_best` raised KeyError.
    #
    # Keep the independently recomputed value under an explicit checked name.
    truth["target_raw_best_checked"] = [
        raw_best(
            float(x86_ms),
            float(arm_ms),
        )
        for x86_ms, arm_ms in zip(
            truth["x86_duration_ms"],
            truth["arm_duration_ms"],
        )
    ]

    truth = truth.rename(
        columns={
            "architecture_preference":
                "target_threshold_label_checked"
        }
    )

    detail = detail[
        detail["variant"].isin(VARIANTS)
    ].copy()

    detail = detail.merge(
        truth[
            [
                "function_name",
                "target_threshold_label_checked",
                "target_raw_best_checked",
            ]
        ],
        left_on="target_function",
        right_on="function_name",
        how="left",
        validate="many_to_one",
    )

    if detail[
        "target_raw_best_checked"
    ].isna().any():
        raise SystemExit(
            "Some targets could not be matched to preferences"
        )

    # If the upstream detail already contains target_raw_best, verify that it
    # agrees exactly with the value independently recomputed from preferences.
    # Then normalize to the checked value so all downstream code uses one
    # canonical column.
    if "target_raw_best" in detail.columns:
        # The upstream refinement study writes target_raw_best only on
        # successfully selected rows. Abstention rows legitimately contain NaN.
        # Therefore validate ONLY rows where the upstream value is present.
        present = detail["target_raw_best"].notna()

        mismatch = present & (
                detail["target_raw_best"]
                .astype(str)
                != detail[
                    "target_raw_best_checked"
                ]
                .astype(str)
        )

        if mismatch.any():
            bad = (
                detail.loc[
                    mismatch,
                    [
                        "target_function",
                        "target_raw_best",
                        "target_raw_best_checked",
                    ],
                ]
                .drop_duplicates()
                .head(20)
            )

            raise SystemExit(
                "Upstream target_raw_best disagrees with preferences "
                "on rows where it is defined:\n"
                + bad.to_string(index=False)
            )

    detail["target_raw_best"] = (
        detail["target_raw_best_checked"]
    )

    detail["selected"] = (
            detail["selection_status"]
            .astype(str)
            == "selected"
    )

    detail[
        "predicted_direction_clean"
    ] = (
        detail["predicted_direction"]
        .fillna("")
        .astype(str)
    )

    # Strict accuracy: abstention is an error.
    detail["strict_raw_correct"] = (
            detail[
                "predicted_direction_clean"
            ]
            == detail["target_raw_best"]
    )

    detail[
        "target_directional_checked"
    ] = (
        detail[
            "target_threshold_label_checked"
        ].isin(DIRECTIONAL)
    )

    # Sanity-check the upstream directional flag too.
    upstream_directional = (
        detail["target_is_directional"]
        .astype(bool)
    )

    if not (
            upstream_directional.to_numpy()
            == detail[
                "target_directional_checked"
            ].to_numpy()
    ).all():
        raise SystemExit(
            "Upstream target_is_directional disagrees "
            "with preferences-2p5 ground truth"
        )

    detail[
        "strict_directional_correct"
    ] = (
            detail[
                "target_directional_checked"
            ]
            & (
                    detail[
                        "predicted_direction_clean"
                    ]
                    == detail[
                        "target_threshold_label_checked"
                    ]
            )
    )

    return detail


def compute_metrics(
        df: pd.DataFrame,
) -> dict:
    selected = df[
        df["selected"]
    ].copy()

    directional = df[
        df["target_directional_checked"]
    ].copy()

    selected_directional = selected[
        selected["target_directional_checked"]
    ].copy()

    recalls = {}

    for label in [
        "x86-preferred",
        "arm-preferred",
    ]:
        truth = (
                df["target_raw_best"]
                .astype(str)
                == label
        )

        recalls[label] = (
            float(
                np.mean(
                    df.loc[
                        truth,
                        "predicted_direction_clean",
                    ].astype(str)
                    == label
                )
            )
            if truth.any()
            else np.nan
        )

    return {
        "selection_coverage": float(
            np.mean(
                df["selected"]
            )
        ),
        "strict_raw_accuracy": float(
            np.mean(
                df["strict_raw_correct"]
            )
        ),
        "covered_raw_accuracy": (
            float(
                np.mean(
                    selected[
                        "predicted_direction_clean"
                    ].astype(str)
                    == selected[
                        "target_raw_best"
                    ].astype(str)
                )
            )
            if len(selected)
            else np.nan
        ),
        "raw_x86_recall": recalls[
            "x86-preferred"
        ],
        "raw_arm_recall": recalls[
            "arm-preferred"
        ],
        "raw_balanced_accuracy": float(
            np.nanmean(
                [
                    recalls[
                        "x86-preferred"
                    ],
                    recalls[
                        "arm-preferred"
                    ],
                ]
            )
        ),
        "directional_coverage": (
            float(
                np.mean(
                    directional[
                        "selected"
                    ]
                )
            )
            if len(directional)
            else np.nan
        ),
        "strict_directional_accuracy": (
            float(
                np.mean(
                    directional[
                        "strict_directional_correct"
                    ]
                )
            )
            if len(directional)
            else np.nan
        ),
        "covered_directional_accuracy": (
            float(
                np.mean(
                    selected_directional[
                        "predicted_direction_clean"
                    ].astype(str)
                    == selected_directional[
                        "target_threshold_label_checked"
                    ].astype(str)
                )
            )
            if len(
                selected_directional
            )
            else np.nan
        ),
        "directional_geomean_speedup": (
            geometric_mean(
                selected_directional[
                    "realized_speedup_factor"
                ]
            )
            if len(
                selected_directional
            )
            else np.nan
        ),
        "directional_mean_regret_percent": (
            float(
                selected_directional[
                    "regret_percent"
                ].mean()
            )
            if len(
                selected_directional
            )
            else np.nan
        ),
        "directional_p90_regret_percent": (
            q90(
                selected_directional[
                    "regret_percent"
                ]
            )
            if len(
                selected_directional
            )
            else np.nan
        ),
        "directional_geomean_mismatch": (
            geometric_mean(
                selected_directional[
                    "multiplicative_mismatch_factor"
                ]
            )
            if len(
                selected_directional
            )
            else np.nan
        ),
    }


def bootstrap_metric_deltas(
        detail: pd.DataFrame,
        variant_a: str,
        variant_b: str,
        replicates: int,
        seed: int,
) -> pd.DataFrame:
    """
    Return B - A paired deltas.

    Bootstrap unit = target function.
    All five seed rows of each target remain grouped together.
    """
    targets = sorted(
        detail[
            "target_function"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    rng = np.random.default_rng(
        seed
    )

    metrics = list(
        compute_metrics(
            detail[
                detail["variant"]
                == variant_a
                ]
        ).keys()
    )

    samples = {
        metric: []
        for metric in metrics
    }

    by_variant_target = {}

    for variant in [
        variant_a,
        variant_b,
    ]:
        v = detail[
            detail["variant"]
            == variant
            ]

        by_variant_target[
            variant
        ] = {
            target: v[
                v["target_function"]
                == target
                ].copy()
            for target in targets
        }

    for _ in range(
            replicates
    ):
        drawn = rng.choice(
            targets,
            size=len(targets),
            replace=True,
        )

        a_parts = []
        b_parts = []

        for i, target in enumerate(
                drawn
        ):
            a = (
                by_variant_target[
                    variant_a
                ][target].copy()
            )

            b = (
                by_variant_target[
                    variant_b
                ][target].copy()
            )

            a["_bootstrap_id"] = i
            b["_bootstrap_id"] = i

            a_parts.append(a)
            b_parts.append(b)

        a_df = pd.concat(
            a_parts,
            ignore_index=True,
        )

        b_df = pd.concat(
            b_parts,
            ignore_index=True,
        )

        ma = compute_metrics(
            a_df
        )
        mb = compute_metrics(
            b_df
        )

        for metric in metrics:
            samples[
                metric
            ].append(
                mb[metric]
                - ma[metric]
            )

    rows = []

    for metric, values in (
            samples.items()
    ):
        arr = np.asarray(
            values,
            dtype=float,
        )

        arr = arr[
            np.isfinite(arr)
        ]

        rows.append(
            {
                "comparison": (
                    f"{variant_b}"
                    f"_minus_"
                    f"{variant_a}"
                ),
                "metric": metric,
                "delta_mean": float(
                    np.mean(arr)
                ),
                "ci95_low": float(
                    np.quantile(
                        arr,
                        0.025,
                    )
                ),
                "ci95_high": float(
                    np.quantile(
                        arr,
                        0.975,
                    )
                ),
                "bootstrap_replicates": (
                    len(arr)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def stability_table(
        detail: pd.DataFrame,
):
    rows = []

    for (
            variant,
            target,
    ), group in detail.groupby(
        [
            "variant",
            "target_function",
        ]
    ):
        preds = (
            group[
                "predicted_direction_clean"
            ]
            .replace(
                "",
                "abstain",
            )
            .astype(str)
        )

        counts = (
            preds.value_counts()
        )

        consensus = (
            float(
                counts.iloc[0]
                / len(preds)
            )
            if len(preds)
            else np.nan
        )

        rows.append(
            {
                "variant": variant,
                "target_function": target,
                "seed_count": len(group),
                "prediction_consensus": (
                    consensus
                ),
                "unanimous_across_seeds": (
                        consensus
                        >= 1.0 - EPS
                ),
                "selected_all_seeds": bool(
                    group[
                        "selected"
                    ].all()
                ),
            }
        )

    per_target = pd.DataFrame(
        rows
    )

    summary = (
        per_target.groupby(
            "variant",
            as_index=False,
        )
        .agg(
            mean_prediction_consensus=(
                "prediction_consensus",
                "mean",
            ),
            unanimous_target_rate=(
                "unanimous_across_seeds",
                "mean",
            ),
            selected_all_seeds_rate=(
                "selected_all_seeds",
                "mean",
            ),
        )
    )

    return (
        per_target,
        summary,
    )


def main(args):
    detail_path = (
        args.detail_csv.resolve()
    )

    preferences_path = (
        args.preferences.resolve()
    )

    detail = prepare(
        detail_path,
        preferences_path,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for variant in VARIANTS:
        metrics = compute_metrics(
            detail[
                detail["variant"]
                == variant
                ]
        )

        summary_rows.append(
            {
                "variant": variant,
                **metrics,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "covered_raw_accuracy",
            "selection_coverage",
            "directional_mean_regret_percent",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    summary_path = (
            args.output_dir
            / "vote-validation-summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    class_rows = []

    for variant in VARIANTS:
        v = detail[
            detail["variant"]
            == variant
            ]

        for label in [
            "x86-preferred",
            "arm-preferred",
        ]:
            g = v[
                v["target_raw_best"]
                == label
                ]

            class_rows.append(
                {
                    "variant": variant,
                    "raw_class": label,
                    "rows": len(g),
                    "coverage": float(
                        np.mean(
                            g[
                                "selected"
                            ]
                        )
                    ),
                    "strict_recall": float(
                        np.mean(
                            g[
                                "predicted_direction_clean"
                            ].astype(str)
                            == label
                        )
                    ),
                }
            )

    per_class = pd.DataFrame(
        class_rows
    )

    per_class_path = (
            args.output_dir
            / "vote-validation-per-class.csv"
    )

    per_class.to_csv(
        per_class_path,
        index=False,
    )

    boot = pd.concat(
        [
            bootstrap_metric_deltas(
                detail,
                "cluster_majority",
                "weighted_vote_p2",
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
            bootstrap_metric_deltas(
                detail,
                "cluster_majority",
                "local_k5_majority",
                args.bootstrap_replicates,
                args.bootstrap_seed
                + 1,
                ),
            bootstrap_metric_deltas(
                detail,
                "local_k5_majority",
                "weighted_vote_p2",
                args.bootstrap_replicates,
                args.bootstrap_seed
                + 2,
                ),
        ],
        ignore_index=True,
    )

    boot_path = (
            args.output_dir
            / "vote-validation-paired-bootstrap.csv"
    )

    boot.to_csv(
        boot_path,
        index=False,
    )

    (
        stability_per_target,
        stability_summary,
    ) = stability_table(
        detail
    )

    stability_path = (
            args.output_dir
            / "vote-validation-stability.csv"
    )

    stability_summary.to_csv(
        stability_path,
        index=False,
    )

    stability_target_path = (
            args.output_dir
            / "vote-validation-stability-per-target.csv"
    )

    stability_per_target.to_csv(
        stability_target_path,
        index=False,
    )

    report_lines = [
        "# Final vote-rule validation",
        "",
        (
            "This is validation only: the candidate rules were fixed before "
            "this bootstrap analysis."
        ),
        "",
        "## Point estimates",
        "",
    ]

    for _, r in (
            summary.iterrows()
    ):
        report_lines.append(
            f"- `{r['variant']}`: "
            f"coverage="
            f"{r['selection_coverage']:.4f}, "
            f"raw-covered="
            f"{r['covered_raw_accuracy']:.4f}, "
            f"raw-strict="
            f"{r['strict_raw_accuracy']:.4f}, "
            f"raw-balanced="
            f"{r['raw_balanced_accuracy']:.4f}, "
            f"directional-covered="
            f"{r['covered_directional_accuracy']:.4f}, "
            f"speedup="
            f"{r['directional_geomean_speedup']:.4f}x, "
            f"mean-regret="
            f"{r['directional_mean_regret_percent']:.3f}%, "
            f"p90-regret="
            f"{r['directional_p90_regret_percent']:.3f}%"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
        (
            "A vote refinement is convincing only if its correctness gain is "
            "not explained by reduced coverage or one architecture class, and "
            "if speedup/regret remain at least competitive. Bootstrap intervals "
            "are target-level and keep all K-Means seed runs of a function "
            "together."
        ),
        "",
    ]

    report_path = (
            args.output_dir
            / "vote-validation-report.md"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    manifest = {
        "validated_variants": (
            VARIANTS
        ),
        "bugfix": (
            "Avoid target_raw_best merge collision. "
            "The independently recomputed ground-truth value is stored as "
            "target_raw_best_checked. Upstream target_raw_best is validated "
            "only where it is present, because abstention rows from the "
            "refinement study legitimately contain NaN; all rows are then "
            "normalized to the independently recomputed checked value."
        ),
        "bootstrap": {
            "resampling_unit": (
                "target_function"
            ),
            "replicates": (
                args.bootstrap_replicates
            ),
            "seed": (
                args.bootstrap_seed
            ),
            "ci": (
                "percentile 95%"
            ),
        },
        "inputs": {
            "detail_csv": {
                "path": str(
                    detail_path
                ),
                "sha256": sha256_file(
                    detail_path
                ),
            },
            "preferences": {
                "path": str(
                    preferences_path
                ),
                "sha256": sha256_file(
                    preferences_path
                ),
            },
        },
    }

    manifest_path = (
            args.output_dir
            / "vote-validation-manifest.json"
    )

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
    print(
        "FINAL VOTE VALIDATION"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "PER-CLASS RAW RECALL"
    )

    print(
        per_class.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "TARGET-LEVEL BOOTSTRAP DELTAS"
    )

    focus_metrics = {
        "selection_coverage",
        "strict_raw_accuracy",
        "covered_raw_accuracy",
        "raw_balanced_accuracy",
        "covered_directional_accuracy",
        "directional_geomean_speedup",
        "directional_mean_regret_percent",
        "directional_p90_regret_percent",
    }

    print(
        boot[
            boot["metric"].isin(
                focus_metrics
            )
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "SEED STABILITY"
    )

    print(
        stability_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"summary={summary_path}"
    )
    print(
        f"per_class={per_class_path}"
    )
    print(
        f"bootstrap={boot_path}"
    )
    print(
        f"stability={stability_path}"
    )
    print(
        f"stability_per_target="
        f"{stability_target_path}"
    )
    print(
        f"report={report_path}"
    )
    print(
        f"manifest={manifest_path}"
    )


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Validate cluster-majority, weighted-p2 and local-k5 vote rules "
            "with per-class metrics, seed stability and target-level bootstrap."
        )
    )

    p.add_argument(
        "--detail-csv",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--preferences",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    return p


if __name__ == "__main__":
    main(
        build_parser().parse_args()
    )
