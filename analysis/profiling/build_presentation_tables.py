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


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    representation = pd.DataFrame([
        {"representation":"PAPER5 MinMax K5","coverage":1.0,"directional_coverage":1.0,
         "directional_preference_agreement":0.5840,"raw_best_arm_agreement":0.5458,
         "role":"dynamic-only baseline"},
        {"representation":"PAPER5 + tokens + functions, MinMax K6","coverage":0.9831,
         "directional_coverage":0.9800,"directional_preference_agreement":0.6694,
         "raw_best_arm_agreement":0.6345,"role":"FINAL"},
        {"representation":"PAPER5 + STATIC4, Standard K9","coverage":0.9661,
         "directional_coverage":0.9600,"directional_preference_agreement":0.6458,
         "raw_best_arm_agreement":0.6246,"role":"control"},
    ])

    cluster_algorithms = pd.DataFrame([
        {"configuration":"KMeans hybrid tokens+functions K6","algorithm":"KMeans","coverage":0.9831,
         "directional_preference_agreement":0.6694,"multiplicative_mismatch_M":1.4539,
         "realized_speedup_x":1.1441,"mean_regret_percent":7.0159,"role":"FINAL"},
        {"configuration":"DBSCAN hybrid STATIC4 q80","algorithm":"DBSCAN","coverage":0.9661,
         "directional_preference_agreement":0.6667,"multiplicative_mismatch_M":None,
         "realized_speedup_x":1.0624,"mean_regret_percent":18.4695,"role":"sensitivity"},
        {"configuration":"DBSCAN hybrid tokens+functions q80","algorithm":"DBSCAN","coverage":0.9492,
         "directional_preference_agreement":0.6596,"multiplicative_mismatch_M":None,
         "realized_speedup_x":1.0629,"mean_regret_percent":18.8625,"role":"sensitivity"},
        {"configuration":"KMeans PAPER5 K5","algorithm":"KMeans","coverage":1.0,
         "directional_preference_agreement":0.5840,"multiplicative_mismatch_M":1.3890,
         "realized_speedup_x":1.0284,"mean_regret_percent":19.8232,"role":"dynamic-only baseline"},
    ])

    donor_rules = pd.DataFrame([
        {"donor_rule":"Professor cluster majority","selection_coverage":0.9831,
         "raw_best_arm_agreement_covered":0.6345,"directional_agreement_covered":0.6694,
         "realized_speedup_x":1.1440,"mean_regret_percent":7.0159,
         "multiplicative_mismatch_M":1.4539,"role":"professor baseline"},
        {"donor_rule":"Weighted vote 1/d^2","selection_coverage":0.9831,
         "raw_best_arm_agreement_covered":0.6862,"directional_agreement_covered":0.7306,
         "realized_speedup_x":1.1902,"mean_regret_percent":4.3970,
         "multiplicative_mismatch_M":1.4506,"role":"FINAL"},
        {"donor_rule":"Local k=5 majority","selection_coverage":0.9831,
         "raw_best_arm_agreement_covered":0.6517,"directional_agreement_covered":0.6898,
         "realized_speedup_x":1.1944,"mean_regret_percent":4.1014,
         "multiplicative_mismatch_M":1.4472,"role":"sensitivity"},
    ])

    formula_dir = root / "analysis-transfer-prior-formula-comparison"
    tuning_dir = root / "analysis-transfer-parameter-tuning-boundary-extension"
    post_dir = root / "analysis-transfer-post-tuning-validation"
    ndonor_dir = root / "analysis-transfer-ndonor-study"

    formula_summary_path = require(formula_dir / "tl-prior-formula-summary.csv")
    tuning_rec_path = require(tuning_dir / "tl-tuning-full-data-recommendation.csv")
    tuning_folds_path = require(tuning_dir / "tl-tuning-fold-selections.csv")
    post_summary_path = require(post_dir / "tl-post-tuning-summary.csv")
    post_effects_path = require(post_dir / "tl-post-tuning-effect-sizes.csv")
    post_paired_path = require(post_dir / "tl-post-tuning-paired-bootstrap.csv")
    ndonor_summary_path = require(ndonor_dir / "tl-ndonor-summary.csv")
    ndonor_paired_path = require(ndonor_dir / "tl-ndonor-paired-bootstrap.csv")
    ndonor_effects_path = require(ndonor_dir / "tl-ndonor-primary-effect-sizes.csv")
    ndonor_diag_path = require(ndonor_dir / "tl-ndonor-evidence-diagnostics.csv")

    formula_summary = pd.read_csv(formula_summary_path)
    tuning_rec = pd.read_csv(tuning_rec_path)
    tuning_folds = pd.read_csv(tuning_folds_path)
    post_summary = pd.read_csv(post_summary_path)
    post_effects = pd.read_csv(post_effects_path)
    post_paired = pd.read_csv(post_paired_path)
    ndonor_summary = pd.read_csv(ndonor_summary_path)
    ndonor_paired = pd.read_csv(ndonor_paired_path)
    ndonor_effects = pd.read_csv(ndonor_effects_path)
    ndonor_diag = pd.read_csv(ndonor_diag_path)

    formula_fixed = formula_summary[
        ["strategy","horizon","latency_gain_percent_vs_no_transfer_macro_mean",
         "optimal_arm_rate_macro_mean","wrong_selections_macro_mean",
         "cumulative_reward_pseudo_regret_macro_mean","first_arm_optimal_macro_mean"]
    ].copy()
    formula_fixed.columns = ["strategy","H","latency_gain_percent","optimal_arm_rate",
                             "wrong_selections","pseudo_regret","first_arm_optimal"]

    tuned_params = tuning_rec[
        ["formula","config_id","regime","w_r","w_e","c","training_all_targets_latency_gain"]
    ].copy()
    tuned_params.columns = ["formula","config_id","regime","w_R","w_E","c",
                            "training_H10_latency_gain_percent"]

    fold_param_stability = tuning_folds[
        ["outer_fold","formula","config_id","regime","w_r","w_e","c","training_latency_gain"]
    ].copy()
    fold_param_stability.columns = ["outer_fold","formula","config_id","regime","w_R","w_E","c",
                                    "training_H10_latency_gain_percent"]

    professor_regimes = pd.DataFrame([
        {"regime":"Large positive w_E","tested_range":"w_E = 8..256","result":"SUPPORTED",
         "evidence":"All selected fold configurations use large positive w_E; robust region ~32-64, occasional 256."},
        {"regime":"w_E = 0 + high c","tested_range":"c = 0.8..4.8","result":"NOT SELECTED",
         "evidence":"No outer-fold winner and no full-data recommendation uses w_E = 0."},
    ])

    post_final = post_summary[
        ["strategy","horizon","latency_gain_percent_vs_no_transfer_macro_mean",
         "optimal_arm_rate_macro_mean","wrong_selections_macro_mean",
         "cumulative_reward_pseudo_regret_macro_mean","first_arm_optimal_macro_mean",
         "convergence_probability_macro_mean","censored_convergence_request_macro_mean"]
    ].copy()
    post_final.columns = ["strategy","H","latency_gain_percent","optimal_arm_rate","wrong_selections",
                          "pseudo_regret","first_arm_optimal","convergence_probability",
                          "censored_convergence_request"]

    uplift_names = {"difference_tuned_minus_difference_original","ratio_tuned_minus_ratio_original"}
    tuning_uplift = post_effects[post_effects["comparison"].isin(uplift_names)][
        ["comparison","horizon","latency_gain_delta_pp","relative_lift_in_latency_gain_percent",
         "mean_latency_reduction_vs_strategy_a_percent","target_latency_win_rate",
         "optimal_arm_rate_delta","wrong_selections_delta","pseudo_regret_delta"]
    ].copy()
    tuning_uplift.columns = ["comparison","H","latency_gain_delta_pp",
                             "relative_lift_in_latency_gain_percent","direct_latency_reduction_percent",
                             "target_latency_win_rate","optimal_arm_rate_delta","wrong_selections_delta",
                             "pseudo_regret_delta"]

    direct = post_paired[post_paired["comparison"]=="ratio_tuned_minus_difference_tuned"].copy()
    tuned_formula_direct = direct[direct["metric"].isin({
        "latency_gain_percent_vs_no_transfer","optimal_arm_rate","wrong_selections",
        "cumulative_reward_pseudo_regret"})][
        ["horizon","metric","point_delta","ci95_low","ci95_high"]
    ].copy()
    tuned_formula_direct.columns = ["H","metric_ratio_minus_difference","point_delta","ci95_low","ci95_high"]

    h10 = post_final[post_final["H"]==10][
        ["strategy","latency_gain_percent","optimal_arm_rate","wrong_selections","pseudo_regret","first_arm_optimal"]
    ].copy()

    ndonor_curve = ndonor_summary[ndonor_summary["formula"].isin(["difference","ratio"])][
        ["formula","strategy","variant_kind","n_donor","horizon",
         "latency_gain_percent_vs_no_transfer","optimal_arm_rate","wrong_selections",
         "cumulative_reward_pseudo_regret","convergence_probability"]
    ].copy()
    ndonor_curve.columns = ["formula","strategy","variant","n_donor","H","latency_gain_percent",
                            "optimal_arm_rate","wrong_selections","pseudo_regret","convergence_probability"]

    ndonor_primary_names = {
        "difference_ndonor_n12_minus_difference_standard",
        "ratio_ndonor_n12_minus_ratio_standard",
    }
    ndonor_primary_ci = ndonor_paired[
        ndonor_paired["comparison"].isin(ndonor_primary_names)
        & ndonor_paired["metric"].isin({
            "latency_gain_percent_vs_no_transfer","optimal_arm_rate","wrong_selections",
            "cumulative_reward_pseudo_regret","convergence_probability"})
        ][["comparison","horizon","metric","point_delta","ci95_low","ci95_high"]].copy()
    ndonor_primary_ci.columns = ["comparison","H","metric","point_delta","ci95_low","ci95_high"]

    ndonor_primary_effects = ndonor_effects.rename(columns={"horizon":"H"}).copy()
    ndonor_h10 = ndonor_curve[ndonor_curve["H"]==10].copy()

    ndonor_direction_diag = ndonor_diag.groupby(
        ["formula","n_donor"], as_index=False
    ).agg(
        mean_subset_direction_match_rate=("subset_direction_match_rate","mean"),
        donor_target_pairs=("target_function","count"),
    )

    final_ucb1 = pd.DataFrame([
        {"component":"Clustering representation",
         "final_choice":"PAPER5 + static_token_count_mean + static_function_count",
         "reason":"best overall donor/downstream trade-off"},
        {"component":"Scaler / clustering","final_choice":"MinMax + K-Means K=6",
         "reason":"preferred over DBSCAN in downstream donor utility"},
        {"component":"Donor decision","final_choice":"weighted inverse-Manhattan vote 1/d^2",
         "reason":"best validated donor refinement"},
        {"component":"Donor ranking","final_choice":"Manhattan",
         "reason":"frozen donor-ranking metric"},
        {"component":"Primary TL formula","final_choice":"Difference / additive",
         "reason":"simpler; Ratio has no systematic significant advantage"},
        {"component":"UCB1 TL parameters","final_choice":"w_R=0.05, w_E=64, c=0.2",
         "reason":"independently replayed tuned Difference configuration"},
        {"component":"n_donor scaling","final_choice":"NOT USED",
         "reason":"w_E*n_donor gives no meaningful latency benefit over standard tuned TL"},
        {"component":"Reference architecture","final_choice":"x86 / AMD64",
         "reason":"frozen thesis methodology"},
        {"component":"Primary evaluation horizon","final_choice":"H=10",
         "reason":"predeclared primary tuning horizon"},
    ])

    tables = {
        "table_01_representation_ablation.csv":representation,
        "table_02_kmeans_vs_dbscan.csv":cluster_algorithms,
        "table_03_donor_rule_comparison.csv":donor_rules,
        "table_04_fixed_formula_comparison.csv":formula_fixed,
        "table_05_tuned_parameters.csv":tuned_params,
        "table_06_tuning_fold_stability.csv":fold_param_stability,
        "table_07_professor_exploration_regimes.csv":professor_regimes,
        "table_08_final_post_tuning_summary.csv":post_final,
        "table_09_tuning_uplift.csv":tuning_uplift,
        "table_10_tuned_ratio_vs_difference_ci.csv":tuned_formula_direct,
        "table_11_h10_executive_summary.csv":h10,
        "table_12_ndonor_sensitivity_curve.csv":ndonor_curve,
        "table_13_ndonor_n12_vs_standard_ci.csv":ndonor_primary_ci,
        "table_14_ndonor_n12_effect_sizes.csv":ndonor_primary_effects,
        "table_15_ndonor_h10_sensitivity.csv":ndonor_h10,
        "table_16_ndonor_direction_reliability.csv":ndonor_direction_diag,
        "table_17_ucb1_final_recommendation.csv":final_ucb1,
    }

    for filename, df in tables.items():
        df.to_csv(out / filename, index=False)

    source_paths = [
        formula_summary_path,tuning_rec_path,tuning_folds_path,post_summary_path,
        post_effects_path,post_paired_path,ndonor_summary_path,ndonor_paired_path,
        ndonor_effects_path,ndonor_diag_path,
    ]

    manifest = {
        "purpose":"complete presentation-ready frozen clustering + donor + UCB1 TL evidence package",
        "no_new_experiments":True,
        "table_count":len(tables),
        "ucb1_status":"FROZEN / COMPLETE",
        "sources":[{"path":str(p),"sha256":sha256_file(p)} for p in source_paths],
        "notes":[
            "Cluster/donor tables reproduce finalized frozen values from completed studies.",
            "Formula/tuning/post-tuning/n_donor tables are extracted directly from saved CSV artifacts.",
            "H=10 remains the primary UCB1 tuning horizon.",
            "n_donor sensitivity is not interpreted as a new hyperparameter tuning stage.",
            "Primary n_donor conclusion uses n=12 vs standard tuned TL with paired target-level bootstrap.",
        ],
    }
    (out/"presentation-tables-manifest.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n", encoding="utf-8"
    )

    sections = [
        ("1. Representation ablation",representation),
        ("2. K-Means vs DBSCAN",cluster_algorithms),
        ("3. Donor rule comparison",donor_rules),
        ("4. Additive vs Ratio — fixed pre-tuning parameters",formula_fixed),
        ("5. Final tuned parameters",tuned_params),
        ("6. Tuning stability across outer folds",fold_param_stability),
        ("7. Professor exploration-regime tests",professor_regimes),
        ("8. Final independent post-tuning replay",post_final),
        ("9. Effective tuning uplift",tuning_uplift),
        ("10. Tuned Ratio vs tuned Difference — paired 95% CI",tuned_formula_direct),
        ("11. Executive H=10 table",h10),
        ("12. n_donor sensitivity curve",ndonor_curve),
        ("13. n_donor=12 vs standard tuned TL — paired 95% CI",ndonor_primary_ci),
        ("14. n_donor=12 effect sizes",ndonor_primary_effects),
        ("15. n_donor H=10 sensitivity",ndonor_h10),
        ("16. Donor-direction reliability vs n_donor",ndonor_direction_diag),
        ("17. Final frozen UCB1 recommendation",final_ucb1),
    ]

    report = [
        "# Serverledge — Complete Presentation Tables","",
        "Frozen results package for clustering, donor selection and UCB1 Transfer Learning.",""
    ]
    for title, df in sections:
        report += [f"## {title}","",md(df),""]

    report += [
        "## Frozen conclusions","",
        "- Final clustering: hybrid PAPER5 + token count + function count, MinMax, K-Means K=6.",
        "- Final donor rule: weighted inverse-Manhattan vote 1/d^2, followed by nearest Manhattan donor in the predicted directional class.",
        "- Large positive w_E is supported; w_E=0 with high c was tested but not selected.",
        "- Final primary UCB1 TL formula: Difference / additive.",
        "- Final additive UCB1 TL parameters: w_R=0.05, w_E=64, c=0.2.",
        "- Ratio is validated but has no systematic significant advantage over Difference.",
        "- n_donor scaling w_E*n_donor gives no meaningful additional latency benefit and is not retained.",
        "- UCB1 + Transfer Learning stage is frozen and complete.",""
    ]
    (out/"PRESENTATION_TABLES.md").write_text("\n".join(report),encoding="utf-8")

    print(f"output_dir={out}")
    print(f"tables_written={len(tables)}")
    for filename in tables:
        print(f"table={out/filename}")
    print(f"report={out/'PRESENTATION_TABLES.md'}")
    print(f"manifest={out/'presentation-tables-manifest.json'}")
    print("ucb1_status=FROZEN_COMPLETE")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
