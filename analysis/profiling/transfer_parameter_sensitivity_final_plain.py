#!/usr/bin/env python3
"""Exploratory Monte Carlo sensitivity of UCB1Decoupled TL parameters.

Purpose
-------
Explore a broad, predeclared Cartesian grid over (w_R, w_E, c) for BOTH
Difference and Ratio priors, while freezing the FINAL donor pipeline:

- donor mapping from the final K=6 / plain-Manhattan ablation output
- by default donor-selection seed 11, matching the final GCP methodology
- target ARM observations never construct the prior
- target x86 anchor sampled independently from online replay
- common random numbers across every parameter configuration
- no-transfer baseline MATCHED ON c for every configuration

This is an exploratory sensitivity/stress analysis. It does NOT replace or
retroactively retune the frozen GCP configurations.

Default grid is the union/extension of the values already explored historically:
    w_R = 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0
    w_E = 0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256
    c   = 0.20, 0.40, 0.80, 1.20, 1.60, 2.40, 3.20, 4.80

That is 768 configurations per formula, 1536 total.

Primary endpoint: H=10 latency gain vs a no-transfer UCB1 with the SAME c.
Secondary: optimal-arm rate gain, wrong selections saved, reward pseudo-regret
saved, first-arm optimal gain, and target-level negative-transfer rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.profiling.transfer_prior_formula_comparison import (
    build_performance,
    build_prior,
    load_eligible_durations,
    make_crn_samples,
)
from analysis.profiling.transfer_parameter_tuning_cv import (
    Config,
    implementation_self_check,
    prefix_metrics_baseline,
    prefix_metrics_tl,
    simulate_baseline_trace,
    simulate_tl_trace,
)

SYNTHETIC_TARGETS = {"amd_faster", "arm_faster"}
FINAL_VARIANT = "manhattan_plain"
FORMULAS = ("difference", "ratio")

DEFAULT_WR = "0.01,0.025,0.05,0.10,0.25,0.50,1.0,2.0"
DEFAULT_WE = "0,0.25,0.50,1,2,4,8,16,32,64,128,256"
DEFAULT_C = "0.20,0.40,0.80,1.20,1.60,2.40,3.20,4.80"

METRIC_NAMES = (
    "optimal_arm_rate",
    "wrong_selections",
    "cumulative_reward_pseudo_regret",
    "cumulative_latency_ms",
    "first_arm_optimal",
    "convergence_probability",
    "censored_convergence_request",
)


def parse_floats(value: str) -> list[float]:
    vals = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not vals:
        raise ValueError("empty float list")
    if any(not np.isfinite(v) or v < 0 for v in vals):
        raise ValueError("all parameter values must be finite and >= 0")
    return vals


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_metric(rows: list[dict[str, float]], name: str) -> float:
    return float(np.mean([float(r[name]) for r in rows]))


def load_final_plain_map(path: Path, donor_seed: int) -> tuple[list[str], dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).fillna({"donor_function": ""})
    required = {"variant", "seed", "target_function", "status", "donor_function"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"donor detail missing columns: {sorted(missing)}")

    df = df[(df["variant"] == FINAL_VARIANT) & (df["seed"].astype(int) == donor_seed)].copy()
    if len(df) != 59:
        raise ValueError(
            f"expected 59 rows for variant={FINAL_VARIANT}, seed={donor_seed}; got {len(df)}"
        )
    if df["target_function"].duplicated().any():
        raise ValueError("duplicate target_function in frozen donor map")

    targets = sorted(df["target_function"].astype(str).tolist())
    mapping: dict[str, str | None] = {}
    for r in df.itertuples(index=False):
        target = str(r.target_function)
        donor = str(r.donor_function).strip()
        mapping[target] = donor if str(r.status) == "selected" and donor else None

    return targets, mapping


def build_grid(wr: list[float], we: list[float], cs: list[float]) -> list[tuple[str, Config]]:
    out: list[tuple[str, Config]] = []
    idx = 0
    for formula in FORMULAS:
        for w_r in wr:
            for w_e in we:
                for c in cs:
                    idx += 1
                    cfg = Config(
                        config_id=f"sens-{idx:04d}",
                        regime="final_plain_sensitivity",
                        w_r=float(w_r),
                        w_e=float(w_e),
                        c=float(c),
                    )
                    out.append((formula, cfg))
    return out


def config_key(formula: str, cfg: Config) -> str:
    return f"{formula}|wr={cfg.w_r:g}|we={cfg.w_e:g}|c={cfg.c:g}"


def evaluate(args: argparse.Namespace) -> None:
    implementation_self_check()

    wr_values = parse_floats(args.reward_weights)
    we_values = parse_floats(args.exploration_weights)
    c_values = parse_floats(args.c_values)
    configs = build_grid(wr_values, we_values, c_values)

    root = args.root.resolve()
    donor_detail = args.donor_detail.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"
    if not raw_x.is_file() or not raw_a.is_file():
        raise SystemExit("missing raw/{x86,arm64}/all_samples.jsonl")

    targets, donor_map = load_final_plain_map(donor_detail, args.donor_seed)
    perf = build_performance(
        load_eligible_durations(raw_x),
        load_eligible_durations(raw_a),
    )
    missing_targets = [t for t in targets if t not in perf]
    if missing_targets:
        raise SystemExit("missing raw target data: " + ", ".join(missing_targets))
    missing_donors = sorted({d for d in donor_map.values() if d and d not in perf})
    if missing_donors:
        raise SystemExit("missing raw donor data: " + ", ".join(missing_donors))
    leaked = [(t, d) for t, d in donor_map.items() if d == t]
    if leaked:
        raise SystemExit(f"target leaked as donor: {leaked[:3]}")

    print(
        "FINAL-PLAIN PARAMETER SENSITIVITY "
        f"targets={len(targets)} donor_seed={args.donor_seed} "
        f"configs={len(configs)} replicates={args.replicates} H={args.horizon}"
    )
    print(f"w_R={wr_values}")
    print(f"w_E={we_values}")
    print(f"c={c_values}")
    print("baseline=matched-c no-transfer UCB1")

    per_target_path = out / "sensitivity_per_target_h10.csv"

    if args.summarize_only:
        if not per_target_path.is_file():
            raise SystemExit(
                "--summarize-only requested but missing existing file: "
                f"{per_target_path}"
            )
        per_target = pd.read_csv(per_target_path)
        expected_rows = len(targets) * len(configs)
        if len(per_target) != expected_rows:
            raise SystemExit(
                "existing per-target file has unexpected row count: "
                f"got={len(per_target)} expected={expected_rows}. "
                "Do not resume from a partial/incompatible run."
            )
        required_resume = {
            "target_function", "formula", "w_r", "w_e", "c",
            "target_is_synthetic", "transfer_applied",
            "latency_gain_percent_vs_matched_c_no_transfer",
            "optimal_arm_rate_gain_vs_matched_c_no_transfer",
            "wrong_selections_saved_vs_matched_c_no_transfer",
            "reward_pseudo_regret_saved_vs_matched_c_no_transfer",
            "first_arm_optimal_gain_vs_matched_c_no_transfer",
            "optimal_arm_rate", "wrong_selections",
            "cumulative_reward_pseudo_regret",
        }
        missing_resume = required_resume - set(per_target.columns)
        if missing_resume:
            raise SystemExit(
                "existing per-target file is incompatible; missing columns: "
                + ", ".join(sorted(missing_resume))
            )
        print(
            f"RESUME SUMMARY ONLY — reusing {per_target_path} "
            f"rows={len(per_target)}"
        )
    else:
        rows: list[dict[str, Any]] = []

        # Group configs by formula/c only for modest cache locality.
        by_formula: dict[str, list[Config]] = {f: [] for f in FORMULAS}
        for formula, cfg in configs:
            by_formula[formula].append(cfg)

        for ti, target in enumerate(targets, start=1):
            stats = perf[target]
            anchor, online = make_crn_samples(
                target=target,
                target_stats=stats,
                replicates=args.replicates,
                max_horizon=args.horizon,
                anchor_n=args.anchor_samples,
                base_seed=args.mc_seed,
            )

            # Baselines are matched by c and share the exact same online samples.
            baseline_by_c: dict[float, list[dict[str, float]]] = {}
            for c in c_values:
                vals: list[dict[str, float]] = []
                for rep in range(args.replicates):
                    trace = simulate_baseline_trace(
                        target_stats=stats,
                        sequences={
                            "x86": online["x86"][rep],
                            "arm64": online["arm64"][rep],
                        },
                        c=float(c),
                        convergence_run=args.convergence_run,
                    )
                    vals.append(prefix_metrics_baseline(trace, args.horizon))
                baseline_by_c[float(c)] = vals

            donor = donor_map[target]
            prior_cache: dict[str, list[dict[str, float]]] = {}
            if donor is not None:
                donor_stats = perf[donor]
                for formula in FORMULAS:
                    prior_cache[formula] = [
                        build_prior(
                            mode=formula,
                            anchor_mean_reward_x86=float(anchor[rep]),
                            donor_stats=donor_stats,
                        )[0]
                        for rep in range(args.replicates)
                    ]

            for formula in FORMULAS:
                for cfg in by_formula[formula]:
                    baseline = baseline_by_c[float(cfg.c)]

                    if donor is None:
                        tl_rows = baseline
                    else:
                        tl_rows = []
                        priors = prior_cache[formula]
                        for rep in range(args.replicates):
                            trace = simulate_tl_trace(
                                target_stats=stats,
                                sequences={
                                    "x86": online["x86"][rep],
                                    "arm64": online["arm64"][rep],
                                },
                                prior_mean_reward=priors[rep],
                                config=cfg,
                                convergence_run=args.convergence_run,
                            )
                            tl_rows.append(prefix_metrics_tl(trace, args.horizon))

                    gains = []
                    optimal_gain = []
                    wrong_saved = []
                    regret_saved = []
                    first_gain = []
                    for rep in range(args.replicates):
                        b = baseline[rep]
                        t = tl_rows[rep]
                        b_latency = float(b["cumulative_latency_ms"])
                        t_latency = float(t["cumulative_latency_ms"])
                        gains.append(100.0 * (b_latency - t_latency) / b_latency)
                        optimal_gain.append(float(t["optimal_arm_rate"]) - float(b["optimal_arm_rate"]))
                        wrong_saved.append(float(b["wrong_selections"]) - float(t["wrong_selections"]))
                        regret_saved.append(
                            float(b["cumulative_reward_pseudo_regret"])
                            - float(t["cumulative_reward_pseudo_regret"])
                        )
                        first_gain.append(float(t["first_arm_optimal"]) - float(b["first_arm_optimal"]))

                    row: dict[str, Any] = {
                        "target_function": target,
                        "target_is_synthetic": target in SYNTHETIC_TARGETS,
                        "donor_seed": int(args.donor_seed),
                        "donor_function": donor or "",
                        "transfer_applied": float(donor is not None),
                        "formula": formula,
                        "config_id": cfg.config_id,
                        "w_r": float(cfg.w_r),
                        "w_e": float(cfg.w_e),
                        "c": float(cfg.c),
                        "horizon": int(args.horizon),
                        "replicates": int(args.replicates),
                        "latency_gain_percent_vs_matched_c_no_transfer": float(np.mean(gains)),
                        "optimal_arm_rate_gain_vs_matched_c_no_transfer": float(np.mean(optimal_gain)),
                        "wrong_selections_saved_vs_matched_c_no_transfer": float(np.mean(wrong_saved)),
                        "reward_pseudo_regret_saved_vs_matched_c_no_transfer": float(np.mean(regret_saved)),
                        "first_arm_optimal_gain_vs_matched_c_no_transfer": float(np.mean(first_gain)),
                    }
                    for metric in METRIC_NAMES:
                        row[metric] = mean_metric(tl_rows, metric)
                        row[f"baseline_{metric}"] = mean_metric(baseline, metric)
                    rows.append(row)

            if ti % 5 == 0 or ti == len(targets):
                print(f"target_progress={ti}/{len(targets)}", flush=True)

        per_target = pd.DataFrame(rows)
        per_target.to_csv(per_target_path, index=False)

    def make_summary(scope_name: str, data: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["formula", "w_r", "w_e", "c"]
        grouped = data.groupby(group_cols, as_index=False)
        summary = grouped.agg(
            target_count=("target_function", "nunique"),
            transfer_coverage=("transfer_applied", "mean"),
            latency_gain_percent=("latency_gain_percent_vs_matched_c_no_transfer", "mean"),
            optimal_arm_rate_gain=("optimal_arm_rate_gain_vs_matched_c_no_transfer", "mean"),
            wrong_selections_saved=("wrong_selections_saved_vs_matched_c_no_transfer", "mean"),
            reward_pseudo_regret_saved=("reward_pseudo_regret_saved_vs_matched_c_no_transfer", "mean"),
            first_arm_optimal_gain=("first_arm_optimal_gain_vs_matched_c_no_transfer", "mean"),
            optimal_arm_rate=("optimal_arm_rate", "mean"),
            wrong_selections=("wrong_selections", "mean"),
            cumulative_reward_pseudo_regret=("cumulative_reward_pseudo_regret", "mean"),
        )

        gain_col = "latency_gain_percent_vs_matched_c_no_transfer"
        distribution = (
            data.groupby(group_cols, as_index=False)[gain_col]
            .agg(
                median_target_gain="median",
                min_target_gain="min",
                max_target_gain="max",
            )
        )
        neg = (
            data.assign(
                _negative_transfer=(data[gain_col].to_numpy(float) < 0.0).astype(float)
            )
            .groupby(group_cols, as_index=False)["_negative_transfer"]
            .mean()
            .rename(columns={"_negative_transfer": "negative_transfer_target_rate"})
        )
        p10 = (
            data.groupby(group_cols)[gain_col]
            .quantile(0.10)
            .rename("p10_target_gain")
            .reset_index()
        )

        summary = summary.merge(distribution, on=group_cols).merge(neg, on=group_cols).merge(p10, on=group_cols)
        summary.insert(0, "scope", scope_name)

        # Historical selection rule, but the latency gain here is now matched-c.
        summary = summary.sort_values(
            [
                "formula",
                "latency_gain_percent",
                "reward_pseudo_regret_saved",
                "optimal_arm_rate_gain",
                "wrong_selections_saved",
                "w_r",
                "w_e",
                "c",
            ],
            ascending=[True, False, False, False, False, True, True, True],
        ).reset_index(drop=True)
        summary["rank_within_formula"] = summary.groupby("formula").cumcount() + 1
        return summary

    summary_all = make_summary("all_59", per_target)
    primary = per_target[~per_target["target_is_synthetic"]].copy()
    summary_primary = make_summary("primary_57_non_synthetic", primary)
    summary = pd.concat([summary_primary, summary_all], ignore_index=True)
    summary_path = out / "sensitivity_summary_h10.csv"
    summary.to_csv(summary_path, index=False)

    primary_rank = summary_primary.copy()
    top = primary_rank[primary_rank["rank_within_formula"] <= args.top_n].copy()
    top_path = out / "sensitivity_top_configs_h10.csv"
    top.to_csv(top_path, index=False)

    # Keep the historically frozen and original configs visible regardless of rank.
    keys = [
        ("difference", 0.05, 64.0, 0.2, "frozen_difference"),
        ("ratio", 0.01, 64.0, 0.4, "frozen_ratio"),
        ("difference", 0.25, 1.0, 0.8, "historical_difference"),
        ("ratio", 0.25, 1.0, 0.8, "historical_ratio"),
    ]
    key_rows = []
    for formula, wr, we, c, role in keys:
        hit = primary_rank[
            (primary_rank.formula == formula)
            & np.isclose(primary_rank.w_r, wr)
            & np.isclose(primary_rank.w_e, we)
            & np.isclose(primary_rank.c, c)
            ].copy()
        if len(hit) == 1:
            hit.insert(0, "role", role)
            key_rows.append(hit)
    key_df = pd.concat(key_rows, ignore_index=True) if key_rows else pd.DataFrame()
    key_path = out / "sensitivity_frozen_and_historical_h10.csv"
    key_df.to_csv(key_path, index=False)

    # Best achievable result conditional on each value: useful for sensitivity plots/tables.
    for param in ("w_r", "w_e", "c"):
        best = (
            primary_rank.sort_values(
                ["formula", param, "latency_gain_percent"],
                ascending=[True, True, False],
            )
            .groupby(["formula", param], as_index=False)
            .first()
        )
        best.to_csv(out / f"sensitivity_best_by_{param}.csv", index=False)

    manifest = {
        "experiment": "final plain-Manhattan UCB1Decoupled parameter sensitivity",
        "exploratory": True,
        "not_used_to_retroactively_change_gcp": True,
        "donor_pipeline": {
            "variant": FINAL_VARIANT,
            "donor_seed": int(args.donor_seed),
            "reason": "seed 11 matches final GCP donor-selection methodology" if args.donor_seed == 11 else "user-selected donor seed",
            "donor_detail_path": str(donor_detail),
            "donor_detail_sha256": sha256(donor_detail),
        },
        "simulation": {
            "horizon": int(args.horizon),
            "replicates": int(args.replicates),
            "mc_seed": int(args.mc_seed),
            "anchor_samples": int(args.anchor_samples),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "baseline": "no-transfer UCB1 matched on c for every TL configuration",
            "convergence_run": int(args.convergence_run),
        },
        "parameter_grid": {
            "reward_weights_wR": wr_values,
            "exploration_weights_wE": we_values,
            "c_values": c_values,
            "configs_per_formula": len(wr_values) * len(we_values) * len(c_values),
            "formula_count": len(FORMULAS),
            "total_configs": len(configs),
        },
        "frozen_reference_configs": {
            "difference": {"w_R": 0.05, "w_E": 64.0, "c": 0.2},
            "ratio": {"w_R": 0.01, "w_E": 64.0, "c": 0.4},
            "historical": {"w_R": 0.25, "w_E": 1.0, "c": 0.8},
        },
        "inputs": {
            "raw_x86": {"path": str(raw_x), "sha256": sha256(raw_x)},
            "raw_arm64": {"path": str(raw_a), "sha256": sha256(raw_a)},
        },
        "scopes": ["primary_57_non_synthetic", "all_59"],
        "primary_endpoint": "macro mean H10 latency gain vs matched-c no-transfer",
    }
    (out / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    report: list[str] = [
        "# Final plain-Manhattan TL parameter sensitivity",
        "",
        "Exploratory local Monte Carlo only; no GCP rerun and no retroactive change to frozen GCP settings.",
        "",
        f"- donor mapping: `{FINAL_VARIANT}`, clustering seed `{args.donor_seed}`",
        f"- H={args.horizon}, replicates={args.replicates}, MC seed={args.mc_seed}",
        "- baseline is matched on `c` for every TL configuration",
        f"- total parameter configurations: {len(configs)} ({len(configs)//2} per formula)",
        "- primary scope: 57 non-synthetic targets; all-59 also saved",
        "",
        "## Top configurations — primary 57",
        "",
    ]
    cols = [
        "formula", "rank_within_formula", "w_r", "w_e", "c",
        "latency_gain_percent", "optimal_arm_rate_gain", "wrong_selections_saved",
        "reward_pseudo_regret_saved", "negative_transfer_target_rate", "p10_target_gain",
    ]
    report.append(top[cols].to_markdown(index=False, floatfmt=".6f"))
    report += ["", "## Frozen and historical reference configurations", ""]
    if len(key_df):
        rcols = ["role", *cols]
        report.append(key_df[rcols].to_markdown(index=False, floatfmt=".6f"))
    else:
        report.append("Reference rows not found in grid.")
    report += [
        "",
        "## Interpretation guardrail",
        "",
        "This sweep is a post-hoc sensitivity/stress analysis. The frozen GCP configurations remain Difference (.05,64,.2) and Ratio (.01,64,.4). Any new local optimum must be described as exploratory unless validated on a fresh Monte Carlo stream.",
        "",
    ]
    (out / "SENSITIVITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("PASS — final-plain parameter sensitivity completed")
    print("PRIMARY TOP CONFIGS")
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("FROZEN/HISTORICAL")
    if len(key_df):
        print(key_df[["role", *cols]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"per_target={per_target_path}")
    print(f"summary={summary_path}")
    print(f"top={top_path}")
    print(f"report={out / 'SENSITIVITY_REPORT.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--donor-detail", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--donor-seed", type=int, default=11)
    p.add_argument("--reward-weights", default=DEFAULT_WR)
    p.add_argument("--exploration-weights", default=DEFAULT_WE)
    p.add_argument("--c-values", default=DEFAULT_C)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--mc-seed", type=int, default=2026101201)
    p.add_argument("--anchor-samples", type=int, default=10)
    p.add_argument("--convergence-run", type=int, default=5)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Reuse an already completed sensitivity_per_target_h10.csv in "
            "--output-dir and regenerate summaries/reports without rerunning Monte Carlo."
        ),
    )
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
