#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    return path


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    representation = pd.DataFrame([
        {"representation": "PAPER5 MinMax K5", "coverage": 1.0000, "directional_coverage": 1.0000,
         "directional_preference_agreement": 0.5840, "raw_best_arm_agreement": 0.5458,
         "role": "dynamic-only baseline"},
        {"representation": "PAPER5 + tokens + functions, MinMax K6", "coverage": 0.9831,
         "directional_coverage": 0.9800, "directional_preference_agreement": 0.6694,
         "raw_best_arm_agreement": 0.6345, "role": "FINAL"},
        {"representation": "PAPER5 + STATIC4, Standard K9", "coverage": 0.9661,
         "directional_coverage": 0.9600, "directional_preference_agreement": 0.6458,
         "raw_best_arm_agreement": 0.6246, "role": "control"},
    ])

    cluster_algorithms = pd.DataFrame([
        {"configuration": "KMeans hybrid tokens+functions K6", "algorithm": "KMeans",
         "coverage": 0.9831, "directional_preference_agreement": 0.6694,
         "multiplicative_mismatch_M": 1.4539, "realized_speedup_x": 1.1441,
         "mean_regret_percent": 7.0159, "role": "FINAL"},
        {"configuration": "DBSCAN hybrid STATIC4 q80", "algorithm": "DBSCAN",
         "coverage": 0.9661, "directional_preference_agreement": 0.6667,
         "multiplicative_mismatch_M": None, "realized_speedup_x": 1.0624,
         "mean_regret_percent": 18.4695, "role": "sensitivity"},
        {"configuration": "DBSCAN hybrid tokens+functions q80", "algorithm": "DBSCAN",
         "coverage": 0.9492, "directional_preference_agreement": 0.6596,
         "multiplicative_mismatch_M": None, "realized_speedup_x": 1.0629,
         "mean_regret_percent": 18.8625, "role": "sensitivity"},
        {"configuration": "KMeans PAPER5 K5", "algorithm": "KMeans",
         "coverage": 1.0000, "directional_preference_agreement": 0.5840,
         "multiplicative_mismatch_M": 1.3890, "realized_speedup_x": 1.0284,
         "mean_regret_percent": 19.8232, "role": "dynamic-only baseline"},
    ])

    donor_rules = pd.DataFrame([
        {"donor_rule": "Professor cluster majority", "selection_coverage": 0.9831,
         "raw_best_arm_agreement_covered": 0.6345, "directional_agreement_covered": 0.6694,
         "realized_speedup_x": 1.1440, "mean_regret_percent": 7.0159,
         "multiplicative_mismatch_M": 1.4539, "role": "professor baseline"},
        {"donor_rule": "Weighted vote 1/d^2", "selection_coverage": 0.9831,
         "raw_best_arm_agreement_covered": 0.6862, "directional_agreement_covered": 0.7306,
         "realized_speedup_x": 1.1902, "mean_regret_percent": 4.3970,
         "multiplicative_mismatch_M": 1.4506, "role": "FINAL"},
        {"donor_rule": "Local k=5 majority", "selection_coverage": 0.9831,
         "raw_best_arm_agreement_covered": 0.6517, "directional_agreement_covered": 0.6898,
         "realized_speedup_x": 1.1944, "mean_regret_percent": 4.1014,
         "multiplicative_mismatch_M": 1.4472, "role": "sensitivity"},
    ])

    formula_dir = root / "analysis-transfer-prior-formula-comparison"
    tuning_dir = root / "analysis-transfer-parameter-tuning-boundary-extension"
    post_dir = root / "analysis-transfer-post-tuning-validation"

    formula_summary_path = require(formula_dir / "tl-prior-formula-summary.csv")
    tuning_rec_path = require(tuning_dir / "tl-tuning-full-data-recommendation.csv")
    tuning_folds_path = require(tuning_dir / "tl-tuning-fold-selections.csv")
    post_summary_path = require(post_dir / "tl-post-tuning-summary.csv")
    post_effects_path = require(post_dir / "tl-post-tuning-effect-sizes.csv")
    post_paired_path = require(post_dir / "tl-post-tuning-paired-bootstrap.csv")

    formula_summary = pd.read_csv(formula_summary_path)
    tuning_rec = pd.read_csv(tuning_rec_path)
    tuning_folds = pd.read_csv(tuning_folds_path)
    post_summary = pd.read_csv(post_summary_path)
    post_effects = pd.read_csv(post_effects_path)
    post_paired = pd.read_csv(post_paired_path)

    formula_fixed = formula_summary[
        ["strategy", "horizon", "latency_gain_percent_vs_no_transfer_macro_mean",
         "optimal_arm_rate_macro_mean", "wrong_selections_macro_mean",
         "cumulative_reward_pseudo_regret_macro_mean", "first_arm_optimal_macro_mean"]
    ].copy()
    formula_fixed.columns = [
        "strategy", "H", "latency_gain_percent", "optimal_arm_rate",
        "wrong_selections", "pseudo_regret", "first_arm_optimal"
    ]

    tuned_params = tuning_rec[
        ["formula", "config_id", "regime", "w_r", "w_e", "c",
         "training_all_targets_latency_gain"]
    ].copy()
    tuned_params.columns = [
        "formula", "config_id", "regime", "w_R", "w_E", "c",
        "training_H10_latency_gain_percent"
    ]

    fold_param_stability = tuning_folds[
        ["outer_fold", "formula", "config_id", "regime", "w_r", "w_e", "c",
         "training_latency_gain"]
    ].copy()
    fold_param_stability.columns = [
        "outer_fold", "formula", "config_id", "regime", "w_R", "w_E", "c",
        "training_H10_latency_gain_percent"
    ]

    post_final = post_summary[
        ["strategy", "horizon", "latency_gain_percent_vs_no_transfer_macro_mean",
         "optimal_arm_rate_macro_mean", "wrong_selections_macro_mean",
         "cumulative_reward_pseudo_regret_macro_mean", "first_arm_optimal_macro_mean",
         "convergence_probability_macro_mean", "censored_convergence_request_macro_mean"]
    ].copy()
    post_final.columns = [
        "strategy", "H", "latency_gain_percent", "optimal_arm_rate",
        "wrong_selections", "pseudo_regret", "first_arm_optimal",
        "convergence_probability", "censored_convergence_request"
    ]

    uplift_names = {
        "difference_tuned_minus_difference_original",
        "ratio_tuned_minus_ratio_original",
    }
    tuning_uplift = post_effects[
        post_effects["comparison"].isin(uplift_names)
    ][
        ["comparison", "horizon", "latency_gain_delta_pp",
         "relative_lift_in_latency_gain_percent",
         "mean_latency_reduction_vs_strategy_a_percent",
         "target_latency_win_rate", "optimal_arm_rate_delta",
         "wrong_selections_delta", "pseudo_regret_delta"]
    ].copy()
    tuning_uplift.columns = [
        "comparison", "H", "latency_gain_delta_pp",
        "relative_lift_in_latency_gain_percent",
        "direct_latency_reduction_percent",
        "target_latency_win_rate", "optimal_arm_rate_delta",
        "wrong_selections_delta", "pseudo_regret_delta"
    ]

    direct = post_paired[
        post_paired["comparison"] == "ratio_tuned_minus_difference_tuned"
    ].copy()
    wanted_metrics = {
        "latency_gain_percent_vs_no_transfer",
        "optimal_arm_rate",
        "wrong_selections",
        "cumulative_reward_pseudo_regret",
    }
    tuned_formula_direct = direct[
        direct["metric"].isin(wanted_metrics)
    ][["horizon", "metric", "point_delta", "ci95_low", "ci95_high"]].copy()
    tuned_formula_direct.columns = [
        "H", "metric_ratio_minus_difference", "point_delta", "ci95_low", "ci95_high"
    ]

    professor_regimes = pd.DataFrame([
        {"regime": "Large positive w_E", "tested_range": "w_E = 8..256",
         "result": "SUPPORTED",
         "evidence": "All selected fold configurations use large positive w_E; robust region ~32-64, occasional 256."},
        {"regime": "w_E = 0 + high c", "tested_range": "c = 0.8..4.8",
         "result": "NOT SELECTED",
         "evidence": "No outer-fold winner and no full-data recommendation uses w_E = 0."},
    ])

    h10 = post_final[post_final["H"] == 10][
        ["strategy", "latency_gain_percent", "optimal_arm_rate",
         "wrong_selections", "pseudo_regret", "first_arm_optimal"]
    ].copy()

    tables = {
        "table_01_representation_ablation.csv": representation,
        "table_02_kmeans_vs_dbscan.csv": cluster_algorithms,
        "table_03_donor_rule_comparison.csv": donor_rules,
        "table_04_fixed_formula_comparison.csv": formula_fixed,
        "table_05_tuned_parameters.csv": tuned_params,
        "table_06_tuning_fold_stability.csv": fold_param_stability,
        "table_07_professor_exploration_regimes.csv": professor_regimes,
        "table_08_final_post_tuning_summary.csv": post_final,
        "table_09_tuning_uplift.csv": tuning_uplift,
        "table_10_tuned_ratio_vs_difference_ci.csv": tuned_formula_direct,
        "table_11_h10_executive_summary.csv": h10,
    }

    for filename, df in tables.items():
        df.to_csv(out / filename, index=False)

    source_paths = [
        formula_summary_path, tuning_rec_path, tuning_folds_path,
        post_summary_path, post_effects_path, post_paired_path,
    ]
    manifest = {
        "purpose": "presentation-ready frozen clustering + UCB1 TL tables",
        "no_new_experiments": True,
        "table_count": len(tables),
        "sources": [{"path": str(p), "sha256": sha256_file(p)} for p in source_paths],
        "notes": [
            "Cluster/donor tables reproduce finalized frozen values from completed studies.",
            "Formula/tuning/post-tuning tables are extracted directly from saved CSV artifacts.",
            "H=10 remains the primary tuning horizon.",
        ],
    }
    manifest_path = out / "presentation-tables-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Serverledge — Presentation Tables", "",
        "Frozen results package for clustering, donor selection and UCB1 Transfer Learning.", "",
        "## 1. Representation ablation", "", markdown_table(representation), "",
        "## 2. K-Means vs DBSCAN", "", markdown_table(cluster_algorithms), "",
        "## 3. Donor rule comparison", "", markdown_table(donor_rules), "",
        "## 4. Additive vs ratio — fixed pre-tuning parameters", "", markdown_table(formula_fixed), "",
        "## 5. Final tuned parameters", "", markdown_table(tuned_params), "",
        "## 6. Tuning stability across outer folds", "", markdown_table(fold_param_stability), "",
        "## 7. Professor exploration-regime tests", "", markdown_table(professor_regimes), "",
        "## 8. Final independent post-tuning replay", "", markdown_table(post_final), "",
        "## 9. Effective tuning uplift", "", markdown_table(tuning_uplift), "",
        "## 10. Tuned ratio vs tuned difference — paired 95% CI", "", markdown_table(tuned_formula_direct), "",
        "## 11. Executive H=10 table", "", markdown_table(h10), "",
        "## Frozen conclusions", "",
        "- Final clustering: hybrid PAPER5 + token count + function count, MinMax, K-Means K=6.",
        "- Final donor rule: weighted inverse-Manhattan vote 1/d^2, followed by nearest Manhattan donor inside the predicted directional class.",
        "- Large positive w_E is supported; w_E=0 with high c was tested but was not selected.",
        "- Final additive UCB1 TL parameters: w_R=0.05, w_E=64, c=0.2.",
        "- Final ratio UCB1 TL parameters: w_R=0.01, w_E=64, c=0.4.",
        "- Final post-tuning replay: strong gain over no-transfer, no systematic significant ratio advantage over additive.",
        "",
    ]
    report_path = out / "PRESENTATION_TABLES.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"output_dir={out}")
    for filename in tables:
        print(f"table={out / filename}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
