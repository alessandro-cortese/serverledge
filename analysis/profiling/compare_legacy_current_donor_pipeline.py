#!/usr/bin/env python3
"""
Bridge comparison: historical slide metrics vs current donor-selection pipeline.

No clustering and no Transfer Learning are rerun.

The controlled comparison is performed on the SAME current 59-function
static-eligible corpus:

A) current_legacy_style
   PAPER5 MinMax K=5, same-cluster Manhattan nearest donor, no majority filter
   (from analysis-donor-quality-lofo-v2).

B) current_hybrid_unfiltered
   PAPER5 + {token_count_mean, function_count}, MinMax K=6,
   same-cluster Manhattan nearest donor, no majority filter
   (from V4 control rows).

C) current_final_majority
   same hybrid representation/K, professor-aligned directional-majority filter
   + Manhattan donor
   (from V4 final rows).

This decomposition lets us distinguish:
- representation/clustering effect: A -> B
- directional-majority donor-rule effect: B -> C
- total pipeline effect: A -> C

The report also includes the old slide K-Means+Manhattan values as a historical
reference only. Those values were obtained on the old 53-function corpus and
must NOT be treated as a controlled causal comparison.

Primary threshold for the new directional rule: 2.5%.

Outputs
-------
legacy-current-donor-bridge.csv
legacy-current-donor-bridge-deltas.csv
legacy-current-donor-bridge-report.md
legacy-current-donor-bridge-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLD = 2.5
DIRECTIONAL = {"x86-preferred", "arm-preferred"}
EPS = 1e-12

HISTORICAL_SLIDE = {
    "row": "historical_slide_kmeans_manhattan_53",
    "controlled_current_corpus": False,
    "selection_coverage": 1.0,
    "raw_best_arm_agreement": 0.5660,
    "mean_abs_reward_gap_error": 0.3341,
    "median_abs_reward_gap_error": 0.1944,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def geometric_mean(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(float)
    arr = arr[arr > 0]
    if not len(arr):
        return np.nan
    return float(np.exp(np.mean(np.log(arr))))


def q90(values) -> float:
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(float)
    if not len(arr):
        return np.nan
    return float(np.quantile(arr, 0.90))


def raw_best_label(x86_ms: float, arm_ms: float) -> str:
    return "x86-preferred" if x86_ms <= arm_ms else "arm-preferred"


def load_preferences(path: Path):
    df = pd.read_csv(path)
    required = {
        "function_name",
        "architecture_preference",
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path}: missing columns {sorted(missing)}"
        )
    return df.set_index("function_name").to_dict(orient="index")


def pair_metrics(
    target_info: dict,
    donor_info: dict,
    action_rule: str,
) -> dict:
    tx = float(target_info["x86_duration_ms"])
    ta = float(target_info["arm_duration_ms"])
    dx = float(donor_info["x86_duration_ms"])
    da = float(donor_info["arm_duration_ms"])

    target_threshold_label = str(
        target_info["architecture_preference"]
    )
    donor_threshold_label = str(
        donor_info["architecture_preference"]
    )

    target_raw = raw_best_label(tx, ta)
    donor_raw = raw_best_label(dx, da)

    target_delta = float(
        target_info["arm_vs_x86_delta_percent"]
    )
    donor_delta = float(
        donor_info["arm_vs_x86_delta_percent"]
    )

    # Same log-reward gap used in the current donor diagnostics.
    target_reward_gap = math.log(tx / ta)
    donor_reward_gap = math.log(dx / da)

    target_ratio = ta / tx
    donor_ratio = da / dx
    log_ratio_mismatch = abs(
        math.log(target_ratio) - math.log(donor_ratio)
    )

    if action_rule == "donor_raw_best":
        chosen_pref = donor_raw
    elif action_rule == "donor_threshold_preference":
        if donor_threshold_label not in DIRECTIONAL:
            return {
                "actionable": False,
                "target_directional": (
                    target_threshold_label in DIRECTIONAL
                ),
                "raw_best_arm_agreement": (
                    target_raw == donor_raw
                ),
                "directional_preference_agreement": np.nan,
                "abs_delta_error_percent": abs(
                    target_delta - donor_delta
                ),
                "abs_reward_gap_error": abs(
                    target_reward_gap - donor_reward_gap
                ),
                "multiplicative_mismatch_factor": math.exp(
                    log_ratio_mismatch
                ),
                "realized_speedup_factor": np.nan,
                "regret_percent": np.nan,
                "zero_regret": np.nan,
            }
        chosen_pref = donor_threshold_label
    else:
        raise ValueError(action_rule)

    if chosen_pref == "x86-preferred":
        chosen = tx
        alternative = ta
    else:
        chosen = ta
        alternative = tx

    optimal = min(tx, ta)
    regret = (chosen - optimal) / optimal * 100.0

    target_directional = target_threshold_label in DIRECTIONAL

    return {
        "actionable": True,
        "target_directional": target_directional,
        "raw_best_arm_agreement": (
            target_raw == donor_raw
        ),
        "directional_preference_agreement": (
            chosen_pref == target_threshold_label
            if target_directional
            else np.nan
        ),
        "abs_delta_error_percent": abs(
            target_delta - donor_delta
        ),
        "abs_reward_gap_error": abs(
            target_reward_gap - donor_reward_gap
        ),
        "multiplicative_mismatch_factor": math.exp(
            log_ratio_mismatch
        ),
        "realized_speedup_factor": alternative / chosen,
        "regret_percent": regret,
        "zero_regret": regret <= EPS,
    }


def normalize_v2(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "configuration",
        "ranking_scope",
        "seed",
        "target_function",
        "donor_function",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"V2 detail missing columns: {sorted(missing)}"
        )

    return df[
        (df["configuration"] == "paper5_minmax_k5")
        & (df["ranking_scope"] == "same_cluster")
    ].copy()


def normalize_v4(
    df: pd.DataFrame,
    selection_mode: str,
) -> pd.DataFrame:
    required = {
        "configuration",
        "selection_mode",
        "selection_status",
        "threshold_percent",
        "seed",
        "target_function",
        "donor_function",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"V4 detail missing columns: {sorted(missing)}"
        )

    return df[
        (
            df["configuration"]
            == "hybrid_tokens_functions_minmax_k6_a1"
        )
        & (df["selection_mode"] == selection_mode)
        & (
            np.isclose(
                df["threshold_percent"].astype(float),
                THRESHOLD,
            )
        )
    ].copy()


def summarize(
    name: str,
    df: pd.DataFrame,
    prefs: dict,
    action_rule: str,
    require_selected: bool,
) -> dict:
    metrics = []

    for _, row in df.iterrows():
        if require_selected:
            selected = str(
                row["selection_status"]
            ) == "selected"
        else:
            selected = bool(
                str(row.get("donor_function", "") or "")
            )

        if not selected:
            metrics.append(
                {
                    "selected": False,
                    "actionable": False,
                    "target_directional": (
                        prefs[str(row["target_function"])][
                            "architecture_preference"
                        ]
                        in DIRECTIONAL
                    ),
                }
            )
            continue

        target_name = str(row["target_function"])
        donor_name = str(row["donor_function"])

        result = pair_metrics(
            prefs[target_name],
            prefs[donor_name],
            action_rule,
        )
        result["selected"] = True
        metrics.append(result)

    m = pd.DataFrame(metrics)

    selected = m[m["selected"] == True].copy()
    actionable = selected[
        selected["actionable"] == True
    ].copy()
    directional = m[
        m["target_directional"] == True
    ].copy()
    selected_directional = selected[
        selected["target_directional"] == True
    ].copy()
    actionable_directional = actionable[
        actionable["target_directional"] == True
    ].copy()

    row = {
        "row": name,
        "controlled_current_corpus": True,
        "target_rows": len(m),
        "selection_coverage": (
            len(selected) / len(m)
            if len(m)
            else np.nan
        ),
        "actionable_coverage": (
            len(actionable) / len(m)
            if len(m)
            else np.nan
        ),
        "directional_target_rows": len(directional),
        "directional_selection_coverage": (
            len(selected_directional) / len(directional)
            if len(directional)
            else np.nan
        ),
        "raw_best_arm_agreement": (
            float(
                selected[
                    "raw_best_arm_agreement"
                ].astype(float).mean()
            )
            if len(selected)
            else np.nan
        ),
        "directional_preference_agreement": (
            float(
                actionable_directional[
                    "directional_preference_agreement"
                ]
                .astype(float)
                .mean()
            )
            if len(actionable_directional)
            else np.nan
        ),
        "mean_abs_delta_error_percent": (
            float(
                selected[
                    "abs_delta_error_percent"
                ].mean()
            )
            if len(selected)
            else np.nan
        ),
        "median_abs_delta_error_percent": (
            float(
                selected[
                    "abs_delta_error_percent"
                ].median()
            )
            if len(selected)
            else np.nan
        ),
        "mean_abs_reward_gap_error": (
            float(
                selected[
                    "abs_reward_gap_error"
                ].mean()
            )
            if len(selected)
            else np.nan
        ),
        "median_abs_reward_gap_error": (
            float(
                selected[
                    "abs_reward_gap_error"
                ].median()
            )
            if len(selected)
            else np.nan
        ),
        "geomean_multiplicative_mismatch_factor": (
            geometric_mean(
                selected[
                    "multiplicative_mismatch_factor"
                ]
            )
            if len(selected)
            else np.nan
        ),
        "directional_geomean_realized_speedup_factor": (
            geometric_mean(
                actionable_directional[
                    "realized_speedup_factor"
                ]
            )
            if len(actionable_directional)
            else np.nan
        ),
        "directional_mean_regret_percent": (
            float(
                actionable_directional[
                    "regret_percent"
                ].mean()
            )
            if len(actionable_directional)
            else np.nan
        ),
        "directional_p90_regret_percent": (
            q90(
                actionable_directional[
                    "regret_percent"
                ]
            )
            if len(actionable_directional)
            else np.nan
        ),
        "directional_zero_regret_rate": (
            float(
                actionable_directional[
                    "zero_regret"
                ]
                .astype(float)
                .mean()
            )
            if len(actionable_directional)
            else np.nan
        ),
    }

    return row


def delta_row(
    label: str,
    a: pd.Series,
    b: pd.Series,
) -> dict:
    """
    b - a: positive means increase from A to B.
    Error/regret metrics must be interpreted with the sign reversed
    (negative = improvement).
    """
    metrics = [
        "selection_coverage",
        "directional_selection_coverage",
        "raw_best_arm_agreement",
        "directional_preference_agreement",
        "mean_abs_delta_error_percent",
        "mean_abs_reward_gap_error",
        "geomean_multiplicative_mismatch_factor",
        "directional_geomean_realized_speedup_factor",
        "directional_mean_regret_percent",
        "directional_p90_regret_percent",
    ]

    out = {"comparison": label}

    for metric in metrics:
        av = float(a[metric])
        bv = float(b[metric])
        out[f"from_{metric}"] = av
        out[f"to_{metric}"] = bv
        out[f"delta_{metric}"] = bv - av

    return out


def main(args):
    v2_path = args.v2_detail.resolve()
    v4_path = args.v4_detail.resolve()
    pref_path = args.preferences.resolve()

    v2 = pd.read_csv(v2_path)
    v4 = pd.read_csv(v4_path)
    prefs = load_preferences(pref_path)

    current_legacy = summarize(
        "current_legacy_style_paper5_k5_unfiltered",
        normalize_v2(v2),
        prefs,
        action_rule="donor_raw_best",
        require_selected=False,
    )

    current_hybrid_unfiltered = summarize(
        "current_hybrid_k6_unfiltered",
        normalize_v4(
            v4,
            "unfiltered_same_cluster",
        ),
        prefs,
        action_rule="donor_raw_best",
        require_selected=True,
    )

    current_final = summarize(
        "current_final_hybrid_k6_directional_majority",
        normalize_v4(
            v4,
            "directional_majority",
        ),
        prefs,
        action_rule="donor_threshold_preference",
        require_selected=True,
    )

    rows = [
        HISTORICAL_SLIDE,
        current_legacy,
        current_hybrid_unfiltered,
        current_final,
    ]

    table = pd.DataFrame(rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        args.output_dir
        / "legacy-current-donor-bridge.csv"
    )
    table.to_csv(
        table_path,
        index=False,
    )

    current = table[
        table["controlled_current_corpus"] == True
    ].set_index("row")

    deltas = pd.DataFrame(
        [
            delta_row(
                "representation_effect_A_to_B",
                current.loc[
                    "current_legacy_style_paper5_k5_unfiltered"
                ],
                current.loc[
                    "current_hybrid_k6_unfiltered"
                ],
            ),
            delta_row(
                "majority_rule_effect_B_to_C",
                current.loc[
                    "current_hybrid_k6_unfiltered"
                ],
                current.loc[
                    "current_final_hybrid_k6_directional_majority"
                ],
            ),
            delta_row(
                "total_pipeline_effect_A_to_C",
                current.loc[
                    "current_legacy_style_paper5_k5_unfiltered"
                ],
                current.loc[
                    "current_final_hybrid_k6_directional_majority"
                ],
            ),
        ]
    )

    delta_path = (
        args.output_dir
        / "legacy-current-donor-bridge-deltas.csv"
    )
    deltas.to_csv(
        delta_path,
        index=False,
    )

    report = [
        "# Legacy vs current donor-selection bridge",
        "",
        "No Transfer Learning is executed here.",
        "",
        "## Historical slide reference",
        "",
        (
            "The historical slide row is shown only as context because it uses "
            "the old 53-function corpus. It is not a controlled estimate of the "
            "new method's gain."
        ),
        "",
        "## Controlled current-corpus decomposition",
        "",
        (
            "A -> B changes the representation/clustering candidate while "
            "keeping nearest same-cluster donor selection."
        ),
        (
            "B -> C keeps the hybrid representation fixed and adds the "
            "professor-aligned x86-vs-ARM directional-majority donor filter."
        ),
        (
            "A -> C is the total donor-pipeline change on the same current "
            "59-function evaluation corpus."
        ),
        "",
        "For agreement/coverage/speedup, positive deltas are favorable.",
        "For gap/mismatch/regret, negative deltas are favorable.",
        "",
    ]

    report_path = (
        args.output_dir
        / "legacy-current-donor-bridge-report.md"
    )
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    manifest = {
        "threshold_percent": THRESHOLD,
        "historical_slide_reference": HISTORICAL_SLIDE,
        "controlled_rows": {
            "A": "PAPER5 MinMax K=5 + unfiltered same-cluster Manhattan",
            "B": (
                "PAPER5 + tokens_mean + function_count, MinMax K=6 + "
                "unfiltered same-cluster Manhattan"
            ),
            "C": (
                "same hybrid representation + directional-majority filter + "
                "Manhattan"
            ),
        },
        "inputs": {
            "v2_detail": {
                "path": str(v2_path),
                "sha256": sha256_file(v2_path),
            },
            "v4_detail": {
                "path": str(v4_path),
                "sha256": sha256_file(v4_path),
            },
            "preferences": {
                "path": str(pref_path),
                "sha256": sha256_file(pref_path),
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "legacy-current-donor-bridge-manifest.json"
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

    display_cols = [
        "row",
        "selection_coverage",
        "directional_selection_coverage",
        "raw_best_arm_agreement",
        "directional_preference_agreement",
        "mean_abs_reward_gap_error",
        "median_abs_reward_gap_error",
        "geomean_multiplicative_mismatch_factor",
        "directional_geomean_realized_speedup_factor",
        "directional_mean_regret_percent",
        "directional_p90_regret_percent",
    ]

    print()
    print("LEGACY / CURRENT BRIDGE")
    print(
        table[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("CONTROLLED DELTAS ON CURRENT CORPUS")
    print(
        deltas.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"table={table_path}")
    print(f"deltas={delta_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Compare historical donor-selection slide metrics with a controlled "
            "current-corpus old-vs-new donor pipeline decomposition."
        )
    )

    p.add_argument(
        "--v2-detail",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--v4-detail",
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

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
