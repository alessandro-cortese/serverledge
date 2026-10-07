#!/usr/bin/env python3
"""
Downstream donor-outcome metrics for Serverledge.

This post-processes the already-generated V4 LOFO donor-selection detail.
It does NOT rerun clustering.

Metrics added
-------------
For every selected target/donor pair:

1. Absolute architecture-delta mismatch:
       E_delta = |Delta_target - Delta_donor|

2. Architecture ratio:
       R_f = T_ARM,f / T_x86,f

3. Symmetric log-ratio mismatch:
       E_log = |ln(R_target) - ln(R_donor)|

4. Multiplicative mismatch factor:
       M = exp(E_log)
   M >= 1 and M = 1 means identical ARM/x86 ratios.

5. Realized speedup factor from FOLLOWING THE DONOR PREFERENCE:
   if donor is ARM-preferred:
       S = T_x86,target / T_ARM,target
   if donor is x86-preferred:
       S = T_ARM,target / T_x86,target
   S > 1  => donor-guided architecture is faster
   S = 1  => equal
   S < 1  => donor-guided architecture is slower

6. Realized latency change percentage relative to the alternative:
       G% = (T_other - T_chosen) / T_other * 100
   positive => latency reduction
   negative => slowdown

7. Regret:
       regret_ms = T_chosen - min(T_x86, T_ARM)
       regret_%  = regret_ms / min(T_x86, T_ARM) * 100

Preference agreement is evaluated only when the target itself is directional.
For an architecture-independent target, directional agreement is N/A.

Important:
- `directional_majority` should always select a directional donor.
- `unfiltered_same_cluster` can select an architecture-independent donor.
  In that case donor-guided speedup/regret is N/A because no directional action
  is implied by the thresholded donor preference.

Outputs
-------
donor-outcome-metrics-per-target.csv
donor-outcome-metrics-per-seed.csv
donor-outcome-metrics-summary.csv
donor-outcome-metrics-report.md
donor-outcome-metrics-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
}

THRESHOLD_TO_FILE = {
    2.5: "preferences-2p5.csv",
    5.0: "preferences-5.csv",
    10.0: "preferences-10.csv",
    15.0: "preferences-15.csv",
    20.0: "preferences-20.csv",
    25.0: "preferences-25.csv",
}

EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_preferences(preferences_dir: Path):
    maps = {}
    paths = {}

    for threshold, filename in THRESHOLD_TO_FILE.items():
        path = preferences_dir / filename
        table = pd.read_csv(path)

        required = {
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
            "arm_vs_x86_delta_percent",
        }
        missing = required - set(table.columns)
        if missing:
            raise SystemExit(
                f"{path}: missing columns {sorted(missing)}"
            )

        maps[threshold] = (
            table.set_index("function_name")[
                [
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ].to_dict(orient="index")
        )
        paths[threshold] = path

    return maps, paths


def threshold_key(value: float) -> float:
    value = float(value)
    for known in THRESHOLD_TO_FILE:
        if abs(value - known) <= 1e-9:
            return known
    raise ValueError(f"Unsupported threshold: {value}")


def geometric_mean_positive(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    values = values[values > 0]
    if not len(values):
        return np.nan
    return float(np.exp(np.mean(np.log(values))))


def pct90(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return np.nan
    return float(np.quantile(values, 0.90))


def enrich_row(row: pd.Series, pref_maps) -> dict:
    out = row.to_dict()

    threshold = threshold_key(row["threshold_percent"])
    lookup = pref_maps[threshold]

    target_name = str(row["target_function"])
    target = lookup[target_name]

    target_label = str(target["architecture_preference"])
    target_x86 = float(target["x86_duration_ms"])
    target_arm = float(target["arm_duration_ms"])
    target_delta = float(target["arm_vs_x86_delta_percent"])
    target_ratio = target_arm / target_x86
    target_log_ratio = math.log(target_ratio)

    out.update(
        {
            "target_preference_label_checked": target_label,
            "target_x86_duration_ms": target_x86,
            "target_arm_duration_ms": target_arm,
            "target_ratio_arm_over_x86": target_ratio,
            "target_log_ratio_arm_over_x86": target_log_ratio,
            "target_is_directional_checked": target_label in DIRECTIONAL,
        }
    )

    selected = str(row["selection_status"]) == "selected"
    donor_name = str(row.get("donor_function", "") or "")

    if not selected or not donor_name:
        out.update(
            {
                "donor_preference_label_checked": np.nan,
                "donor_x86_duration_ms": np.nan,
                "donor_arm_duration_ms": np.nan,
                "donor_ratio_arm_over_x86": np.nan,
                "donor_log_ratio_arm_over_x86": np.nan,
                "abs_delta_error_percent_recomputed": np.nan,
                "log_ratio_mismatch": np.nan,
                "multiplicative_mismatch_factor": np.nan,
                "donor_actionable_directional": False,
                "directional_preference_agreement_recomputed": np.nan,
                "chosen_architecture": np.nan,
                "chosen_duration_ms": np.nan,
                "alternative_duration_ms": np.nan,
                "realized_speedup_factor": np.nan,
                "realized_latency_change_percent": np.nan,
                "optimal_architecture_raw": np.nan,
                "optimal_duration_ms": np.nan,
                "regret_ms": np.nan,
                "regret_percent": np.nan,
                "chosen_is_raw_optimal": np.nan,
            }
        )
        return out

    donor = lookup[donor_name]
    donor_label = str(donor["architecture_preference"])
    donor_x86 = float(donor["x86_duration_ms"])
    donor_arm = float(donor["arm_duration_ms"])
    donor_delta = float(donor["arm_vs_x86_delta_percent"])
    donor_ratio = donor_arm / donor_x86
    donor_log_ratio = math.log(donor_ratio)

    log_mismatch = abs(target_log_ratio - donor_log_ratio)
    multiplicative_mismatch = math.exp(log_mismatch)

    target_directional = target_label in DIRECTIONAL
    donor_directional = donor_label in DIRECTIONAL

    pref_agreement = (
        donor_label == target_label
        if target_directional and donor_directional
        else np.nan
    )

    if target_x86 <= target_arm:
        optimal_arch = "amd64"
        optimal_ms = target_x86
    else:
        optimal_arch = "arm64"
        optimal_ms = target_arm

    if donor_label == "x86-preferred":
        chosen_arch = "amd64"
        chosen_ms = target_x86
        alternative_ms = target_arm
    elif donor_label == "arm-preferred":
        chosen_arch = "arm64"
        chosen_ms = target_arm
        alternative_ms = target_x86
    else:
        chosen_arch = np.nan
        chosen_ms = np.nan
        alternative_ms = np.nan

    if donor_directional:
        realized_speedup = alternative_ms / chosen_ms
        realized_change = (
            (alternative_ms - chosen_ms)
            / alternative_ms
            * 100.0
        )
        regret_ms = chosen_ms - optimal_ms
        regret_percent = regret_ms / optimal_ms * 100.0
        chosen_optimal = regret_ms <= EPS
    else:
        realized_speedup = np.nan
        realized_change = np.nan
        regret_ms = np.nan
        regret_percent = np.nan
        chosen_optimal = np.nan

    out.update(
        {
            "donor_preference_label_checked": donor_label,
            "donor_x86_duration_ms": donor_x86,
            "donor_arm_duration_ms": donor_arm,
            "donor_ratio_arm_over_x86": donor_ratio,
            "donor_log_ratio_arm_over_x86": donor_log_ratio,
            "abs_delta_error_percent_recomputed": abs(
                target_delta - donor_delta
            ),
            "log_ratio_mismatch": log_mismatch,
            "multiplicative_mismatch_factor": multiplicative_mismatch,
            "donor_actionable_directional": donor_directional,
            "directional_preference_agreement_recomputed": pref_agreement,
            "chosen_architecture": chosen_arch,
            "chosen_duration_ms": chosen_ms,
            "alternative_duration_ms": alternative_ms,
            "realized_speedup_factor": realized_speedup,
            "realized_latency_change_percent": realized_change,
            "optimal_architecture_raw": optimal_arch,
            "optimal_duration_ms": optimal_ms,
            "regret_ms": regret_ms,
            "regret_percent": regret_percent,
            "chosen_is_raw_optimal": chosen_optimal,
        }
    )

    return out


def summarize_group(group: pd.DataFrame) -> dict:
    selected = group[
        group["selection_status"] == "selected"
    ].copy()

    actionable = selected[
        selected["donor_actionable_directional"] == True
    ].copy()

    directional_targets = group[
        group["target_is_directional_checked"] == True
    ].copy()

    selected_directional = selected[
        selected["target_is_directional_checked"] == True
    ].copy()

    actionable_directional = actionable[
        actionable["target_is_directional_checked"] == True
    ].copy()

    def mean_col(df, col):
        return (
            float(pd.to_numeric(df[col], errors="coerce").mean())
            if len(df)
            else np.nan
        )

    def median_col(df, col):
        return (
            float(pd.to_numeric(df[col], errors="coerce").median())
            if len(df)
            else np.nan
        )

    return {
        "target_rows": int(len(group)),
        "selected_rows": int(len(selected)),
        "selection_coverage": (
            len(selected) / len(group)
            if len(group)
            else np.nan
        ),
        "actionable_rows": int(len(actionable)),
        "actionable_coverage": (
            len(actionable) / len(group)
            if len(group)
            else np.nan
        ),
        "directional_target_rows": int(len(directional_targets)),
        "selected_directional_rows": int(len(selected_directional)),
        "directional_selection_coverage": (
            len(selected_directional) / len(directional_targets)
            if len(directional_targets)
            else np.nan
        ),
        "actionable_directional_rows": int(len(actionable_directional)),
        "directional_actionable_coverage": (
            len(actionable_directional) / len(directional_targets)
            if len(directional_targets)
            else np.nan
        ),
        "directional_preference_agreement": mean_col(
            actionable_directional,
            "directional_preference_agreement_recomputed",
        ),
        "mean_abs_delta_error_percent": mean_col(
            selected,
            "abs_delta_error_percent_recomputed",
        ),
        "median_abs_delta_error_percent": median_col(
            selected,
            "abs_delta_error_percent_recomputed",
        ),
        "mean_log_ratio_mismatch": mean_col(
            selected,
            "log_ratio_mismatch",
        ),
        "median_multiplicative_mismatch_factor": median_col(
            selected,
            "multiplicative_mismatch_factor",
        ),
        "geomean_multiplicative_mismatch_factor": geometric_mean_positive(
            selected["multiplicative_mismatch_factor"]
        ),
        "directional_mean_abs_delta_error_percent": mean_col(
            selected_directional,
            "abs_delta_error_percent_recomputed",
        ),
        "directional_median_multiplicative_mismatch_factor": median_col(
            selected_directional,
            "multiplicative_mismatch_factor",
        ),
        "directional_geomean_multiplicative_mismatch_factor": (
            geometric_mean_positive(
                selected_directional[
                    "multiplicative_mismatch_factor"
                ]
            )
        ),
        "geomean_realized_speedup_factor": geometric_mean_positive(
            actionable["realized_speedup_factor"]
        ),
        "median_realized_speedup_factor": median_col(
            actionable,
            "realized_speedup_factor",
        ),
        "mean_realized_latency_change_percent": mean_col(
            actionable,
            "realized_latency_change_percent",
        ),
        "positive_speedup_rate": (
            float(
                np.mean(
                    actionable["realized_speedup_factor"]
                    .astype(float)
                    .to_numpy()
                    > 1.0 + EPS
                )
            )
            if len(actionable)
            else np.nan
        ),
        "slowdown_rate": (
            float(
                np.mean(
                    actionable["realized_speedup_factor"]
                    .astype(float)
                    .to_numpy()
                    < 1.0 - EPS
                )
            )
            if len(actionable)
            else np.nan
        ),
        "zero_regret_rate": (
            float(
                np.mean(
                    actionable["regret_percent"]
                    .astype(float)
                    .to_numpy()
                    <= EPS
                )
            )
            if len(actionable)
            else np.nan
        ),
        "mean_regret_ms": mean_col(
            actionable,
            "regret_ms",
        ),
        "median_regret_ms": median_col(
            actionable,
            "regret_ms",
        ),
        "mean_regret_percent": mean_col(
            actionable,
            "regret_percent",
        ),
        "median_regret_percent": median_col(
            actionable,
            "regret_percent",
        ),
        "p90_regret_percent": pct90(
            actionable["regret_percent"]
        ),
        "directional_geomean_realized_speedup_factor": (
            geometric_mean_positive(
                actionable_directional[
                    "realized_speedup_factor"
                ]
            )
        ),
        "directional_mean_regret_percent": mean_col(
            actionable_directional,
            "regret_percent",
        ),
        "directional_median_regret_percent": median_col(
            actionable_directional,
            "regret_percent",
        ),
        "directional_p90_regret_percent": pct90(
            actionable_directional[
                "regret_percent"
            ]
        ),
        "directional_zero_regret_rate": (
            float(
                np.mean(
                    actionable_directional[
                        "regret_percent"
                    ]
                    .astype(float)
                    .to_numpy()
                    <= EPS
                )
            )
            if len(actionable_directional)
            else np.nan
        ),
    }


def build_per_seed(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (
        config,
        threshold,
        mode,
        seed,
    ), group in enriched.groupby(
        [
            "configuration",
            "threshold_percent",
            "selection_mode",
            "seed",
        ]
    ):
        rows.append(
            {
                "configuration": config,
                "threshold_percent": float(threshold),
                "selection_mode": mode,
                "seed": int(seed),
                **summarize_group(group),
            }
        )

    return pd.DataFrame(rows)


def build_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    id_cols = {
        "configuration",
        "threshold_percent",
        "selection_mode",
        "seed",
    }

    metric_cols = [
        col for col in per_seed.columns
        if col not in id_cols
    ]

    agg = {}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    return (
        per_seed.groupby(
            [
                "configuration",
                "threshold_percent",
                "selection_mode",
            ],
            as_index=False,
        )
        .agg(**agg)
    )


def write_report(
    path: Path,
    summary: pd.DataFrame,
):
    lines = [
        "# Donor outcome metrics",
        "",
        "## Definitions",
        "",
        (
            "- `multiplicative_mismatch_factor = exp(|ln(R_target) - "
            "ln(R_donor)|)`, with `R = T_ARM / T_x86`. Lower is better; "
            "1.0 is perfect ratio agreement."
        ),
        (
            "- `realized_speedup_factor` measures the target speedup obtained "
            "by following the donor's directional architecture preference. "
            "Greater than 1 is beneficial; less than 1 is a slowdown."
        ),
        (
            "- `regret_percent` measures latency lost relative to the target's "
            "raw optimal architecture. Lower is better; 0% is optimal."
        ),
        "",
        "## Focus thresholds",
        "",
        (
            "The report highlights 2.5%, 5% and 15%. The complete CSV retains "
            "all evaluated thresholds."
        ),
        "",
    ]

    focus = summary[
        summary["threshold_percent"].isin(
            [2.5, 5.0, 15.0]
        )
    ].copy()

    professor = focus[
        focus["selection_mode"] == "directional_majority"
    ].sort_values(
        [
            "threshold_percent",
            "directional_preference_agreement_mean",
            "directional_mean_regret_percent_mean",
            "directional_geomean_multiplicative_mismatch_factor_mean",
        ],
        ascending=[True, False, True, True],
    )

    lines += [
        "## Directional-majority shortlist",
        "",
    ]

    for _, r in professor.iterrows():
        lines.append(
            f"- t={r['threshold_percent']:.1f}% / "
            f"`{r['configuration']}`: "
            f"coverage={r['selection_coverage_mean']:.4f}, "
            f"dir-agreement="
            f"{r['directional_preference_agreement_mean']:.4f}, "
            f"dir-M="
            f"{r['directional_geomean_multiplicative_mismatch_factor_mean']:.4f}, "
            f"dir-realized-speedup="
            f"{r['directional_geomean_realized_speedup_factor_mean']:.4f}x, "
            f"dir-regret="
            f"{r['directional_mean_regret_percent_mean']:.3f}%"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        (
            "Do not rank configurations by one metric alone. Prefer high "
            "coverage and directional preference agreement, multiplicative "
            "mismatch close to 1, realized speedup above 1, and low mean/p90 "
            "regret. These are downstream evaluation metrics and are not used "
            "as clustering input features."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(args):
    detail_path = args.detail_csv.resolve()
    preferences_dir = args.preferences_dir.resolve()

    detail = pd.read_csv(detail_path)

    required = {
        "configuration",
        "seed",
        "threshold_percent",
        "selection_mode",
        "selection_status",
        "target_function",
        "donor_function",
    }
    missing = required - set(detail.columns)
    if missing:
        raise SystemExit(
            f"Missing detail columns: {sorted(missing)}"
        )

    pref_maps, pref_paths = load_preferences(
        preferences_dir
    )

    enriched_rows = [
        enrich_row(row, pref_maps)
        for _, row in detail.iterrows()
    ]
    enriched = pd.DataFrame(enriched_rows)

    # Consistency check: professor rule must never return independent donors.
    professor_selected = enriched[
        (enriched["selection_mode"] == "directional_majority")
        & (enriched["selection_status"] == "selected")
    ]

    if len(professor_selected):
        invariant = (
            professor_selected[
                "donor_actionable_directional"
            ]
            .astype(bool)
            .all()
        )
        if not invariant:
            raise SystemExit(
                "Invariant violation: directional_majority selected "
                "a non-directional donor"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_target_path = (
        args.output_dir
        / "donor-outcome-metrics-per-target.csv"
    )
    enriched.to_csv(
        per_target_path,
        index=False,
    )

    per_seed = build_per_seed(enriched)
    per_seed_path = (
        args.output_dir
        / "donor-outcome-metrics-per-seed.csv"
    )
    per_seed.to_csv(
        per_seed_path,
        index=False,
    )

    summary = build_summary(per_seed)
    summary_path = (
        args.output_dir
        / "donor-outcome-metrics-summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "donor-outcome-metrics-report.md"
    )
    write_report(
        report_path,
        summary,
    )

    manifest = {
        "source_detail": {
            "path": str(detail_path),
            "sha256": sha256_file(detail_path),
        },
        "preference_inputs": {
            str(threshold): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for threshold, path in pref_paths.items()
        },
        "metric_definitions": {
            "ratio": "R = T_ARM / T_x86",
            "log_ratio_mismatch": (
                "|ln(R_target) - ln(R_donor)|"
            ),
            "multiplicative_mismatch_factor": (
                "exp(log_ratio_mismatch)"
            ),
            "realized_speedup_factor": (
                "alternative target latency / donor-guided chosen latency"
            ),
            "realized_latency_change_percent": (
                "(alternative - chosen) / alternative * 100"
            ),
            "regret_ms": (
                "chosen target latency - min(target x86, target ARM)"
            ),
            "regret_percent": (
                "regret_ms / min(target x86, target ARM) * 100"
            ),
        },
        "aggregation_note": (
            "Speedup factors are summarized primarily with geometric means; "
            "regret and mismatch also include robust median/p90 summaries."
        ),
    }

    manifest_path = (
        args.output_dir
        / "donor-outcome-metrics-manifest.json"
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

    focus = summary[
        (
            summary["selection_mode"]
            == "directional_majority"
        )
        & summary["threshold_percent"].isin(
            [2.5, 5.0, 15.0]
        )
    ].sort_values(
        [
            "threshold_percent",
            "directional_preference_agreement_mean",
            "directional_mean_regret_percent_mean",
        ],
        ascending=[True, False, True],
    )

    cols = [
        "configuration",
        "threshold_percent",
        "selection_coverage_mean",
        "directional_selection_coverage_mean",
        "directional_preference_agreement_mean",
        "directional_geomean_multiplicative_mismatch_factor_mean",
        "directional_geomean_realized_speedup_factor_mean",
        "directional_mean_regret_percent_mean",
        "directional_p90_regret_percent_mean",
        "directional_zero_regret_rate_mean",
    ]

    print()
    print("FOCUS: DIRECTIONAL-MAJORITY @ 2.5 / 5 / 15")
    print(
        focus[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"per_target={per_target_path}")
    print(f"per_seed={per_seed_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process V4 donor-selection results with multiplicative "
            "mismatch, realized speedup/slowdown and regret metrics."
        )
    )

    parser.add_argument(
        "--detail-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--preferences-dir",
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
