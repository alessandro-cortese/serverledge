#!/usr/bin/env python3
"""
Serverledge — matched-parameter multi-seed Monte Carlo formula study.

Goal
----
Compare Difference vs Ratio with IDENTICAL hyperparameters so that the direct
Difference/Ratio delta isolates the prior formula rather than w_R, w_E or c.

Three fixed parameter families are evaluated:

A) posthoc_best:
   Difference = Ratio = (w_R=.01, w_E=2,  c=.2)

B) frozen_difference_params:
   Difference = Ratio = (w_R=.05, w_E=64, c=.2)

C) frozen_ratio_params:
   Difference = Ratio = (w_R=.01, w_E=64, c=.4)

The actually deployed/frozen pair is also retained descriptively:
   Difference frozen = (.05,64,.2)
   Ratio frozen      = (.01,64,.4)

This is NOT tuning. All configurations are fixed before the run.

Protocol
--------
- final donor variant: manhattan_plain
- donor-selection seed: 11 by default
- primary scope: 57 non-synthetic target functions
- empirical warm durations sampled with replacement
- multiple independent Monte Carlo seeds
- anchor x86 RNG independent from online replay
- common random numbers across ALL strategies within target/MC seed
- H={5,10,20,50}; H=10 primary
- MC seeds averaged within target before inference
- paired percentile bootstrap on target_function

Outputs
-------
matched_multiseed_per_target_seed.csv
matched_multiseed_per_target.csv
matched_multiseed_summary.csv
matched_multiseed_seed_summary.csv
matched_multiseed_paired_bootstrap.csv
matched_multiseed_chart_long.csv
matched_multiseed_formula_delta.csv
MATCHED_FORMULA_REPORT.md
matched_multiseed_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
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

FINAL_VARIANT = "manhattan_plain"
PRIMARY_HORIZON = 10
SYNTHETIC_TARGETS = {
    "amd_faster", "arm_faster", "amd-faster", "arm-faster"
}
RAW_METRICS = (
    "cumulative_latency_ms",
    "optimal_arm_rate",
    "wrong_selections",
    "cumulative_reward_pseudo_regret",
    "first_arm_optimal",
    "convergence_probability",
    "censored_convergence_request",
)


@dataclass(frozen=True)
class TLSpec:
    strategy: str
    family: str
    formula: str
    w_r: float
    w_e: float
    c: float


TL_SPECS = (
    TLSpec("difference_A", "A_posthoc_best", "difference", 0.01, 2.0, 0.2),
    TLSpec("ratio_A",      "A_posthoc_best", "ratio",      0.01, 2.0, 0.2),

    TLSpec("difference_B", "B_frozen_difference_params", "difference", 0.05, 64.0, 0.2),
    TLSpec("ratio_B",      "B_frozen_difference_params", "ratio",      0.05, 64.0, 0.2),

    TLSpec("difference_C", "C_frozen_ratio_params", "difference", 0.01, 64.0, 0.4),
    TLSpec("ratio_C",      "C_frozen_ratio_params", "ratio",      0.01, 64.0, 0.4),
)

BASELINES = {
    "no_transfer_c02": 0.2,
    "no_transfer_c04": 0.4,
    "no_transfer_c08": 0.8,  # historical/common reference only
}

FAMILY_BASELINE = {
    "A_posthoc_best": "no_transfer_c02",
    "B_frozen_difference_params": "no_transfer_c02",
    "C_frozen_ratio_params": "no_transfer_c04",
}


def parse_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    if any(v <= 0 for v in values):
        raise ValueError("integers must be positive")
    if len(values) != len(set(values)):
        raise ValueError("duplicate integer values are not allowed")
    return values


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def is_synthetic(name: str) -> bool:
    return str(name).strip().lower() in SYNTHETIC_TARGETS


def load_plain_map(
        path: Path,
        donor_seed: int,
) -> tuple[list[str], dict[str, str | None]]:
    df = pd.read_csv(path).fillna({"donor_function": ""})
    required = {
        "variant", "seed", "target_function", "status", "donor_function"
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"donor detail missing columns: {sorted(missing)}"
        )

    x = df[
        (df["variant"].astype(str) == FINAL_VARIANT)
        & (df["seed"].astype(int) == int(donor_seed))
        ].copy()

    if len(x) != 59:
        raise SystemExit(
            f"expected 59 rows for {FINAL_VARIANT}/seed={donor_seed}, "
            f"got {len(x)}"
        )
    if x["target_function"].duplicated().any():
        raise SystemExit("duplicate target_function in donor map")

    targets = sorted(x["target_function"].astype(str).tolist())
    mapping: dict[str, str | None] = {}
    for r in x.itertuples(index=False):
        target = str(r.target_function)
        donor = str(r.donor_function).strip()
        mapping[target] = (
            donor
            if str(r.status) == "selected" and donor
            else None
        )
    return targets, mapping


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in RAW_METRICS
    }


def run_one_seed(
        *,
        mc_seed: int,
        replicates: int,
        horizons: list[int],
        anchor_samples: int,
        convergence_run: int,
        targets: list[str],
        donor_map: dict[str, str | None],
        performance: dict[str, Any],
) -> pd.DataFrame:
    max_h = max(horizons)
    rows: list[dict[str, Any]] = []

    for ti, target in enumerate(targets, start=1):
        stats = performance[target]

        anchor, online = make_crn_samples(
            target=target,
            target_stats=stats,
            replicates=replicates,
            max_horizon=max_h,
            anchor_n=anchor_samples,
            base_seed=mc_seed,
        )

        # Baselines for c=.2, .4, .8; same CRN as every TL strategy.
        baseline_means: dict[str, dict[int, dict[str, float]]] = {}
        for baseline_name, c in BASELINES.items():
            by_h = {h: [] for h in horizons}
            for rep in range(replicates):
                trace = simulate_baseline_trace(
                    target_stats=stats,
                    sequences={
                        "x86": online["x86"][rep],
                        "arm64": online["arm64"][rep],
                    },
                    c=c,
                    convergence_run=convergence_run,
                )
                for h in horizons:
                    by_h[h].append(prefix_metrics_baseline(trace, h))
            baseline_means[baseline_name] = {
                h: mean_rows(by_h[h]) for h in horizons
            }

        donor = donor_map[target]

        for baseline_name, c in BASELINES.items():
            for h in horizons:
                met = baseline_means[baseline_name][h]
                rows.append({
                    "mc_seed": int(mc_seed),
                    "target_function": target,
                    "target_is_synthetic": bool(is_synthetic(target)),
                    "donor_function": donor or "",
                    "transfer_applied": 0.0,
                    "strategy": baseline_name,
                    "family": "baseline",
                    "formula": "none",
                    "w_r": 0.0,
                    "w_e": 0.0,
                    "c": float(c),
                    "horizon": int(h),
                    **met,
                    "matched_baseline": baseline_name,
                    "gain_percent_vs_matched_notl": 0.0,
                    "gain_percent_vs_common_c08": (
                        0.0 if baseline_name == "no_transfer_c08"
                        else 100.0 * (
                                baseline_means["no_transfer_c08"][h]["cumulative_latency_ms"]
                                - met["cumulative_latency_ms"]
                        ) / baseline_means["no_transfer_c08"][h]["cumulative_latency_ms"]
                    ),
                })

        if donor is None:
            # Exact fallback to the family-matched no-TL baseline.
            for spec in TL_SPECS:
                matched_name = FAMILY_BASELINE[spec.family]
                for h in horizons:
                    met = baseline_means[matched_name][h]
                    c08 = baseline_means["no_transfer_c08"][h]
                    rows.append({
                        "mc_seed": int(mc_seed),
                        "target_function": target,
                        "target_is_synthetic": bool(is_synthetic(target)),
                        "donor_function": "",
                        "transfer_applied": 0.0,
                        "strategy": spec.strategy,
                        "family": spec.family,
                        "formula": spec.formula,
                        "w_r": spec.w_r,
                        "w_e": spec.w_e,
                        "c": spec.c,
                        "horizon": int(h),
                        **met,
                        "matched_baseline": matched_name,
                        "gain_percent_vs_matched_notl": 0.0,
                        "gain_percent_vs_common_c08": 100.0 * (
                                c08["cumulative_latency_ms"]
                                - met["cumulative_latency_ms"]
                        ) / c08["cumulative_latency_ms"],
                    })
        else:
            donor_stats = performance[donor]

            # Prior depends only on formula + anchor, so cache once per formula/rep.
            priors: dict[str, list[dict[str, float]]] = {}
            for formula in ("difference", "ratio"):
                vals = []
                for rep in range(replicates):
                    prior, _ = build_prior(
                        mode=formula,
                        anchor_mean_reward_x86=float(anchor[rep]),
                        donor_stats=donor_stats,
                    )
                    vals.append(prior)
                priors[formula] = vals

            for spec in TL_SPECS:
                matched_name = FAMILY_BASELINE[spec.family]
                cfg = Config(
                    config_id=spec.strategy,
                    regime="fixed_formula_control",
                    w_r=spec.w_r,
                    w_e=spec.w_e,
                    c=spec.c,
                )
                by_h = {h: [] for h in horizons}

                for rep in range(replicates):
                    trace = simulate_tl_trace(
                        target_stats=stats,
                        sequences={
                            "x86": online["x86"][rep],
                            "arm64": online["arm64"][rep],
                        },
                        prior_mean_reward=priors[spec.formula][rep],
                        config=cfg,
                        convergence_run=convergence_run,
                    )
                    for h in horizons:
                        by_h[h].append(prefix_metrics_tl(trace, h))

                for h in horizons:
                    met = mean_rows(by_h[h])
                    matched = baseline_means[matched_name][h]
                    c08 = baseline_means["no_transfer_c08"][h]

                    rows.append({
                        "mc_seed": int(mc_seed),
                        "target_function": target,
                        "target_is_synthetic": bool(is_synthetic(target)),
                        "donor_function": donor,
                        "transfer_applied": 1.0,
                        "strategy": spec.strategy,
                        "family": spec.family,
                        "formula": spec.formula,
                        "w_r": spec.w_r,
                        "w_e": spec.w_e,
                        "c": spec.c,
                        "horizon": int(h),
                        **met,
                        "matched_baseline": matched_name,
                        "gain_percent_vs_matched_notl": 100.0 * (
                                matched["cumulative_latency_ms"]
                                - met["cumulative_latency_ms"]
                        ) / matched["cumulative_latency_ms"],
                        "gain_percent_vs_common_c08": 100.0 * (
                                c08["cumulative_latency_ms"]
                                - met["cumulative_latency_ms"]
                        ) / c08["cumulative_latency_ms"],
                    })

        if ti % 10 == 0 or ti == len(targets):
            print(
                f"mc_seed={mc_seed} target_progress={ti}/{len(targets)}",
                flush=True,
            )

    return pd.DataFrame(rows)


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "mc_seed",
        "target_function",
        "target_is_synthetic",
        "donor_function",
        "strategy",
        "family",
        "formula",
        "matched_baseline",
    }
    numeric = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    keys = [
        "target_function",
        "target_is_synthetic",
        "donor_function",
        "strategy",
        "family",
        "formula",
        "matched_baseline",
        "horizon",
    ]
    return (
        df.groupby(keys, as_index=False, dropna=False)[numeric]
        .mean()
    )


def build_summary(per_target: pd.DataFrame) -> pd.DataFrame:
    primary = per_target[~per_target["target_is_synthetic"]].copy()
    rows: list[dict[str, Any]] = []

    for (strategy, family, formula, horizon), g in primary.groupby(
            ["strategy", "family", "formula", "horizon"]
    ):
        rows.append({
            "strategy": strategy,
            "family": family,
            "formula": formula,
            "horizon": int(horizon),
            "target_count": int(g["target_function"].nunique()),
            "w_r": float(g["w_r"].iloc[0]),
            "w_e": float(g["w_e"].iloc[0]),
            "c": float(g["c"].iloc[0]),
            "transfer_coverage": float(g["transfer_applied"].mean()),
            "gain_percent_vs_matched_notl":
                float(g["gain_percent_vs_matched_notl"].mean()),
            "gain_percent_vs_common_c08":
                float(g["gain_percent_vs_common_c08"].mean()),
            "cumulative_latency_ms":
                float(g["cumulative_latency_ms"].mean()),
            "optimal_arm_rate":
                float(g["optimal_arm_rate"].mean()),
            "wrong_selections":
                float(g["wrong_selections"].mean()),
            "cumulative_reward_pseudo_regret":
                float(g["cumulative_reward_pseudo_regret"].mean()),
            "negative_transfer_rate_matched_c":
                float((g["gain_percent_vs_matched_notl"] < 0).mean()),
            "p10_gain_matched_c":
                float(g["gain_percent_vs_matched_notl"].quantile(0.10)),
        })

    return pd.DataFrame(rows).sort_values(
        ["horizon", "family", "strategy"]
    ).reset_index(drop=True)


def build_seed_summary(per_target_seed: pd.DataFrame) -> pd.DataFrame:
    primary = per_target_seed[
        ~per_target_seed["target_is_synthetic"]
    ].copy()

    rows: list[dict[str, Any]] = []
    for (mc_seed, strategy, family, horizon), g in primary.groupby(
            ["mc_seed", "strategy", "family", "horizon"]
    ):
        rows.append({
            "mc_seed": int(mc_seed),
            "strategy": strategy,
            "family": family,
            "horizon": int(horizon),
            "target_count": int(g["target_function"].nunique()),
            "gain_percent_vs_matched_notl":
                float(g["gain_percent_vs_matched_notl"].mean()),
            "gain_percent_vs_common_c08":
                float(g["gain_percent_vs_common_c08"].mean()),
            "optimal_arm_rate":
                float(g["optimal_arm_rate"].mean()),
            "wrong_selections":
                float(g["wrong_selections"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["horizon", "family", "strategy", "mc_seed"]
    )


def paired_bootstrap(
        per_target: pd.DataFrame,
        *,
        strategy_a: str,
        strategy_b: str,
        horizon: int,
        replicates: int,
        seed: int,
        comparison: str,
) -> list[dict[str, Any]]:
    primary = per_target[
        ~per_target["target_is_synthetic"]
    ].copy()

    a = primary[
        (primary["strategy"] == strategy_a)
        & (primary["horizon"] == horizon)
        ].set_index("target_function")
    b = primary[
        (primary["strategy"] == strategy_b)
        & (primary["horizon"] == horizon)
        ].set_index("target_function")

    targets = sorted(set(a.index) & set(b.index))
    if len(targets) != 57:
        raise ValueError(
            f"{comparison}: expected 57 paired targets, got {len(targets)}"
        )

    metrics = {
        "latency_reduction_percent": 100.0 * (
                a.loc[targets, "cumulative_latency_ms"].to_numpy(float)
                - b.loc[targets, "cumulative_latency_ms"].to_numpy(float)
        ) / a.loc[targets, "cumulative_latency_ms"].to_numpy(float),

        "gain_vs_matched_notl_delta_pp":
            b.loc[targets, "gain_percent_vs_matched_notl"].to_numpy(float)
            - a.loc[targets, "gain_percent_vs_matched_notl"].to_numpy(float),

        "gain_vs_common_c08_delta_pp":
            b.loc[targets, "gain_percent_vs_common_c08"].to_numpy(float)
            - a.loc[targets, "gain_percent_vs_common_c08"].to_numpy(float),

        "optimal_arm_rate_delta":
            b.loc[targets, "optimal_arm_rate"].to_numpy(float)
            - a.loc[targets, "optimal_arm_rate"].to_numpy(float),

        "wrong_selections_delta":
            b.loc[targets, "wrong_selections"].to_numpy(float)
            - a.loc[targets, "wrong_selections"].to_numpy(float),

        "pseudo_regret_delta":
            b.loc[
                targets, "cumulative_reward_pseudo_regret"
            ].to_numpy(float)
            - a.loc[
                targets, "cumulative_reward_pseudo_regret"
            ].to_numpy(float),
    }

    rng = np.random.default_rng(seed)
    inds = rng.integers(
        0, len(targets),
        size=(replicates, len(targets)),
    )

    rows = []
    for metric, delta in metrics.items():
        boot = delta[inds].mean(axis=1)
        rows.append({
            "comparison": comparison,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "horizon": int(horizon),
            "metric": metric,
            "point_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "paired_targets": int(len(targets)),
            "bootstrap_replicates": int(replicates),
        })
    return rows


def main(args: argparse.Namespace) -> None:
    implementation_self_check()

    mc_seeds = parse_ints(args.mc_seeds)
    horizons = parse_ints(args.horizons)
    if PRIMARY_HORIZON not in horizons:
        raise SystemExit("H=10 must be included")
    if args.replicates <= 0 or args.anchor_samples <= 0:
        raise SystemExit("replicates and anchor-samples must be positive")

    root = args.root.resolve()
    donor_detail = args.donor_detail.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"
    if not raw_x.is_file() or not raw_a.is_file():
        raise SystemExit("missing raw arm/x86 JSONL")
    if not donor_detail.is_file():
        raise SystemExit(f"missing donor detail: {donor_detail}")

    targets, donor_map = load_plain_map(
        donor_detail, args.donor_seed
    )

    performance = build_performance(
        load_eligible_durations(raw_x),
        load_eligible_durations(raw_a),
    )

    missing = [t for t in targets if t not in performance]
    if missing:
        raise SystemExit("missing target raw data: " + ", ".join(missing))

    donor_missing = sorted({
        d for d in donor_map.values()
        if d and d not in performance
    })
    if donor_missing:
        raise SystemExit("missing donor raw data: " + ", ".join(donor_missing))

    if any(t == d for t, d in donor_map.items() if d):
        raise SystemExit("target appears as its own donor")

    primary_count = sum(not is_synthetic(t) for t in targets)

    print(
        "MATCHED-PARAMETER MULTI-SEED FORMULA STUDY "
        f"targets={len(targets)} "
        f"primary_non_synthetic={primary_count}"
    )
    print(
        f"donor_variant={FINAL_VARIANT} donor_seed={args.donor_seed}"
    )
    print(
        f"mc_seeds={mc_seeds} replicates_per_seed={args.replicates} "
        f"total_replicates_per_target={len(mc_seeds) * args.replicates}"
    )
    print(f"horizons={horizons} CRN=ON across all strategies")
    print("A: both formulas (.01,2,.2)")
    print("B: both formulas (.05,64,.2)")
    print("C: both formulas (.01,64,.4)")
    print("Baselines: NoTL c=.2/.4; NoTL c=.8 saved as common reference")

    frames = []
    for i, mc_seed in enumerate(mc_seeds, start=1):
        print()
        print(f"===== MC SEED {i}/{len(mc_seeds)}: {mc_seed} =====")
        frames.append(
            run_one_seed(
                mc_seed=mc_seed,
                replicates=args.replicates,
                horizons=horizons,
                anchor_samples=args.anchor_samples,
                convergence_run=args.convergence_run,
                targets=targets,
                donor_map=donor_map,
                performance=performance,
            )
        )

    per_target_seed = pd.concat(frames, ignore_index=True)
    pts_path = out / "matched_multiseed_per_target_seed.csv"
    per_target_seed.to_csv(pts_path, index=False)

    per_target = aggregate_across_seeds(per_target_seed)
    pt_path = out / "matched_multiseed_per_target.csv"
    per_target.to_csv(pt_path, index=False)

    summary = build_summary(per_target)
    summary_path = out / "matched_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    seed_summary = build_seed_summary(per_target_seed)
    seed_summary_path = out / "matched_multiseed_seed_summary.csv"
    seed_summary.to_csv(seed_summary_path, index=False)

    # Paired comparisons:
    # - each TL strategy vs its matched NoTL baseline
    # - Difference vs Ratio within each identical-parameter family
    # - actually frozen deployed configs (B Difference vs C Ratio)
    pairs = [
        ("no_transfer_c02", "difference_A", "A_difference_vs_notl_c02"),
        ("no_transfer_c02", "ratio_A",      "A_ratio_vs_notl_c02"),
        ("difference_A",    "ratio_A",      "A_ratio_vs_difference"),

        ("no_transfer_c02", "difference_B", "B_difference_vs_notl_c02"),
        ("no_transfer_c02", "ratio_B",      "B_ratio_vs_notl_c02"),
        ("difference_B",    "ratio_B",      "B_ratio_vs_difference"),

        ("no_transfer_c04", "difference_C", "C_difference_vs_notl_c04"),
        ("no_transfer_c04", "ratio_C",      "C_ratio_vs_notl_c04"),
        ("difference_C",    "ratio_C",      "C_ratio_vs_difference"),

        ("difference_B",    "ratio_C",      "frozen_ratio_vs_frozen_difference"),
    ]

    paired_rows = []
    for a, b, name in pairs:
        for h in horizons:
            paired_rows.extend(
                paired_bootstrap(
                    per_target,
                    strategy_a=a,
                    strategy_b=b,
                    horizon=h,
                    replicates=args.bootstrap_replicates,
                    seed=stable_seed(
                        args.bootstrap_seed,
                        name,
                        h,
                    ),
                    comparison=name,
                )
            )

    paired = pd.DataFrame(paired_rows)
    paired_path = out / "matched_multiseed_paired_bootstrap.csv"
    paired.to_csv(paired_path, index=False)

    chart_long = summary.copy()
    chart_long_path = out / "matched_multiseed_chart_long.csv"
    chart_long.to_csv(chart_long_path, index=False)

    formula_delta = paired[
        paired["comparison"].isin([
            "A_ratio_vs_difference",
            "B_ratio_vs_difference",
            "C_ratio_vs_difference",
            "frozen_ratio_vs_frozen_difference",
        ])
        & (paired["metric"] == "latency_reduction_percent")
        ].copy()
    formula_delta_path = out / "matched_multiseed_formula_delta.csv"
    formula_delta.to_csv(formula_delta_path, index=False)

    manifest = {
        "experiment":
            "matched-parameter multi-seed Difference vs Ratio formula study",
        "is_tuning": False,
        "primary_scope": "57 non-synthetic targets",
        "primary_horizon": PRIMARY_HORIZON,
        "donor": {
            "variant": FINAL_VARIANT,
            "seed": int(args.donor_seed),
            "path": str(donor_detail),
            "sha256": sha256(donor_detail),
        },
        "families": {
            "A_posthoc_best": {
                "difference": {"w_R": .01, "w_E": 2, "c": .2},
                "ratio": {"w_R": .01, "w_E": 2, "c": .2},
                "matched_no_transfer_c": .2,
            },
            "B_frozen_difference_params": {
                "difference": {"w_R": .05, "w_E": 64, "c": .2},
                "ratio": {"w_R": .05, "w_E": 64, "c": .2},
                "matched_no_transfer_c": .2,
            },
            "C_frozen_ratio_params": {
                "difference": {"w_R": .01, "w_E": 64, "c": .4},
                "ratio": {"w_R": .01, "w_E": 64, "c": .4},
                "matched_no_transfer_c": .4,
            },
            "frozen_deployed_pair": {
                "difference": {"w_R": .05, "w_E": 64, "c": .2},
                "ratio": {"w_R": .01, "w_E": 64, "c": .4},
                "note": "configuration-vs-configuration comparison, not pure formula effect",
            },
        },
        "simulation": {
            "mc_seeds": mc_seeds,
            "replicates_per_seed": int(args.replicates),
            "total_replicates_per_target":
                int(len(mc_seeds) * args.replicates),
            "horizons": horizons,
            "anchor_samples": int(args.anchor_samples),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "empirical_resampling":
                "with replacement from warm eligible durations",
            "convergence_run": int(args.convergence_run),
        },
        "aggregation": {
            "mc_seeds":
                "averaged within target before inference",
            "inference_unit": "target_function",
        },
        "bootstrap": {
            "unit": "target_function",
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "ci": "percentile 95%",
        },
        "inputs": {
            "raw_x86": {"path": str(raw_x), "sha256": sha256(raw_x)},
            "raw_arm64": {"path": str(raw_a), "sha256": sha256(raw_a)},
        },
    }
    manifest_path = out / "matched_multiseed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    h10 = summary[summary["horizon"] == PRIMARY_HORIZON].copy()
    h10_formula = paired[
        (paired["horizon"] == PRIMARY_HORIZON)
        & (paired["metric"] == "latency_reduction_percent")
        & paired["comparison"].isin([
            "A_ratio_vs_difference",
            "B_ratio_vs_difference",
            "C_ratio_vs_difference",
            "frozen_ratio_vs_frozen_difference",
        ])
        ].copy()

    report_lines = [
        "# Matched-parameter multi-seed Difference vs Ratio",
        "",
        "Fixed post-hoc formula-control study; not tuning.",
        "",
        "## Parameter families",
        "",
        "- A: Difference = Ratio = (.01, 2, .2)",
        "- B: Difference = Ratio = (.05, 64, .2)",
        "- C: Difference = Ratio = (.01, 64, .4)",
        "- Frozen deployed pair: Difference (.05,64,.2) vs Ratio (.01,64,.4); descriptive configuration comparison only.",
        "",
        "## H=10 summary",
        "",
        h10[
            [
                "strategy", "family", "formula",
                "w_r", "w_e", "c",
                "gain_percent_vs_matched_notl",
                "gain_percent_vs_common_c08",
                "optimal_arm_rate",
                "wrong_selections",
                "cumulative_reward_pseudo_regret",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## H=10 direct formula comparisons",
        "",
        h10_formula[
            [
                "comparison",
                "point_delta",
                "ci95_low",
                "ci95_high",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    report_path = out / "MATCHED_FORMULA_REPORT.md"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("PASS — matched-parameter multi-seed study completed")
    print()
    print("===== H=10 SUMMARY =====")
    print(
        h10[
            [
                "strategy", "family", "formula",
                "w_r", "w_e", "c",
                "gain_percent_vs_matched_notl",
                "gain_percent_vs_common_c08",
                "optimal_arm_rate",
                "wrong_selections",
                "cumulative_reward_pseudo_regret",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("===== H=10 DIRECT FORMULA COMPARISONS =====")
    print(
        h10_formula[
            [
                "comparison",
                "point_delta",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("===== H=10 SEED STABILITY =====")
    h10_seed = seed_summary[
        seed_summary["horizon"] == PRIMARY_HORIZON
        ]
    print(
        h10_seed[
            [
                "mc_seed",
                "strategy",
                "family",
                "gain_percent_vs_matched_notl",
                "optimal_arm_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(f"per_target_seed={pts_path}")
    print(f"per_target={pt_path}")
    print(f"summary={summary_path}")
    print(f"seed_summary={seed_summary_path}")
    print(f"paired={paired_path}")
    print(f"chart_long={chart_long_path}")
    print(f"formula_delta={formula_delta_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--donor-detail", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--donor-seed", type=int, default=11)
    p.add_argument(
        "--mc-seeds",
        default="2026101210,2026101211,2026101212,2026101213,2026101214",
    )
    p.add_argument("--horizons", default="5,10,20,50")
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--anchor-samples", type=int, default=10)
    p.add_argument("--convergence-run", type=int, default=5)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=2026101215)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
