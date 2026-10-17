#!/usr/bin/env python3
"""Confirmatory Monte Carlo replay for final plain-Manhattan Serverledge TL.

Purpose
-------
Validate, on a fresh Monte Carlo stream, the post-hoc sensitivity winner and
compare it against the already-frozen GCP and historical configurations.
This script does NOT tune or select hyperparameters.

Frozen donor pipeline
---------------------
- final K=6 donor geometry output
- donor variant: manhattan_plain
- donor-selection seed: 11 by default (same methodological seed used for GCP)
- target excluded from donor fit/selection upstream
- target ARM data NEVER constructs the prior
- target x86 anchor: independent bootstrap samples
- common random numbers across every policy

Strategies
----------
Baselines:
- no_transfer_c02: c=.2
- no_transfer_c04: c=.4
- no_transfer_c08: c=.8 (common reference)

Post-hoc sensitivity candidates (NOT used on GCP):
- difference_new: Difference, w_R=.01, w_E=2, c=.2
- ratio_new:      Ratio,      w_R=.01, w_E=2, c=.2

Frozen GCP configurations:
- difference_frozen: Difference, w_R=.05, w_E=64, c=.2
- ratio_frozen:      Ratio,      w_R=.01, w_E=64, c=.4

Historical configurations:
- difference_historical: Difference, w_R=.25, w_E=1, c=.8
- ratio_historical:      Ratio,      w_R=.25, w_E=1, c=.8

Primary scope
-------------
57 non-synthetic targets. The all-59 scope is saved as a secondary view.

Primary horizons
----------------
H = 5, 10, 20, 50. H=10 remains the primary endpoint.

Comparisons
-----------
Two views are kept separate:
1. matched-c gain: TL compared with no-transfer using the SAME c;
2. common-baseline gain: every strategy compared with no_transfer_c08.

Paired percentile bootstrap resamples target_function, not Monte Carlo runs.
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
SYNTHETIC_TARGETS = {
    "amd_faster", "arm_faster", "amd-faster", "arm-faster"
}

HISTORICAL = {"w_r": 0.25, "w_e": 1.0, "c": 0.8}
FROZEN_DIFFERENCE = {"w_r": 0.05, "w_e": 64.0, "c": 0.2}
FROZEN_RATIO = {"w_r": 0.01, "w_e": 64.0, "c": 0.4}
NEW_CANDIDATE = {"w_r": 0.01, "w_e": 2.0, "c": 0.2}

BASELINE_CS = (0.2, 0.4, 0.8)
COMMON_BASELINE = "no_transfer_c08"
PRIMARY_HORIZON = 10

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
class Strategy:
    name: str
    kind: str  # baseline | tl
    c: float
    formula: str | None = None
    w_r: float = 0.0
    w_e: float = 0.0
    role: str = ""


STRATEGIES = (
    Strategy("no_transfer_c02", "baseline", 0.2, role="baseline"),
    Strategy("no_transfer_c04", "baseline", 0.4, role="baseline"),
    Strategy("no_transfer_c08", "baseline", 0.8, role="common_reference"),
    Strategy(
        "difference_new", "tl", 0.2, "difference", 0.01, 2.0,
        "posthoc_sensitivity_candidate",
    ),
    Strategy(
        "ratio_new", "tl", 0.2, "ratio", 0.01, 2.0,
        "posthoc_sensitivity_candidate",
    ),
    Strategy(
        "difference_frozen", "tl", 0.2, "difference", 0.05, 64.0,
        "frozen_gcp",
    ),
    Strategy(
        "ratio_frozen", "tl", 0.4, "ratio", 0.01, 64.0,
        "frozen_gcp",
    ),
    Strategy(
        "difference_historical", "tl", 0.8, "difference", 0.25, 1.0,
        "historical",
    ),
    Strategy(
        "ratio_historical", "tl", 0.8, "ratio", 0.25, 1.0,
        "historical",
    ),
)

STRATEGY_BY_NAME = {s.name: s for s in STRATEGIES}

# B relative to A. Positive latency_reduction_percent means B is faster/better.
PAIRS = (
    ("no_transfer_c08", "no_transfer_c02", "effect_of_c_08_to_02"),
    ("no_transfer_c08", "no_transfer_c04", "effect_of_c_08_to_04"),
    ("no_transfer_c02", "difference_new", "new_difference_vs_matched_notl"),
    ("no_transfer_c02", "ratio_new", "new_ratio_vs_matched_notl"),
    ("no_transfer_c02", "difference_frozen", "frozen_difference_vs_matched_notl"),
    ("no_transfer_c04", "ratio_frozen", "frozen_ratio_vs_matched_notl"),
    ("no_transfer_c08", "difference_historical", "historical_difference_vs_matched_notl"),
    ("no_transfer_c08", "ratio_historical", "historical_ratio_vs_matched_notl"),
    ("difference_frozen", "difference_new", "new_vs_frozen_difference"),
    ("ratio_frozen", "ratio_new", "new_vs_frozen_ratio"),
    ("difference_historical", "difference_new", "new_vs_historical_difference"),
    ("ratio_historical", "ratio_new", "new_vs_historical_ratio"),
    ("difference_new", "ratio_new", "ratio_vs_difference_same_parameters"),
    ("difference_frozen", "ratio_frozen", "ratio_vs_difference_frozen_configs"),
)


def parse_ints(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("horizons must be positive integers")
    return vals


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


def load_plain_map(path: Path, donor_seed: int) -> tuple[list[str], dict[str, str | None]]:
    df = pd.read_csv(path).fillna({"donor_function": ""})
    required = {"variant", "seed", "target_function", "status", "donor_function"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"donor detail missing columns: {sorted(missing)}")

    x = df[
        (df["variant"].astype(str) == FINAL_VARIANT)
        & (df["seed"].astype(int) == int(donor_seed))
        ].copy()

    if len(x) != 59:
        raise SystemExit(
            f"expected 59 donor rows for variant={FINAL_VARIANT}, seed={donor_seed}; got {len(x)}"
        )
    if x["target_function"].duplicated().any():
        raise SystemExit("duplicate target_function in donor map")

    targets = sorted(x["target_function"].astype(str).tolist())
    mapping: dict[str, str | None] = {}
    for r in x.itertuples(index=False):
        target = str(r.target_function)
        donor = str(r.donor_function).strip()
        mapping[target] = donor if str(r.status) == "selected" and donor else None
    return targets, mapping


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty metric rows")
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in RAW_METRICS
    }


def matched_baseline_name(c: float) -> str:
    if np.isclose(c, 0.2):
        return "no_transfer_c02"
    if np.isclose(c, 0.4):
        return "no_transfer_c04"
    if np.isclose(c, 0.8):
        return "no_transfer_c08"
    raise ValueError(f"no matched baseline declared for c={c}")


def run_replay(args: argparse.Namespace) -> tuple[pd.DataFrame, list[int], list[str], dict[str, str | None]]:
    horizons = parse_ints(args.horizons)
    max_h = max(horizons)

    root = args.root.resolve()
    donor_detail = args.donor_detail.resolve()
    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"

    if not raw_x.is_file() or not raw_a.is_file():
        raise SystemExit("missing raw/{x86,arm64}/all_samples.jsonl")
    if not donor_detail.is_file():
        raise SystemExit(f"missing donor detail: {donor_detail}")

    targets, donor_map = load_plain_map(donor_detail, args.donor_seed)
    performance = build_performance(
        load_eligible_durations(raw_x),
        load_eligible_durations(raw_a),
    )

    missing_targets = [t for t in targets if t not in performance]
    if missing_targets:
        raise SystemExit("missing target raw data: " + ", ".join(missing_targets))
    missing_donors = sorted({d for d in donor_map.values() if d and d not in performance})
    if missing_donors:
        raise SystemExit("missing donor raw data: " + ", ".join(missing_donors))
    leaks = [(t, d) for t, d in donor_map.items() if d == t]
    if leaks:
        raise SystemExit(f"target used as its own donor: {leaks[:5]}")

    print(
        "FINAL PLAIN CONFIRMATORY REPLAY "
        f"targets={len(targets)} primary_non_synthetic={sum(not is_synthetic(t) for t in targets)} "
        f"donor_seed={args.donor_seed} replicates={args.replicates} "
        f"horizons={horizons} mc_seed={args.mc_seed}"
    )
    print("Donor variant: manhattan_plain")
    print("Common baseline: no_transfer_c08; matched baselines: c=.2/.4/.8")

    rows: list[dict[str, Any]] = []
    baseline_strategies = [s for s in STRATEGIES if s.kind == "baseline"]
    tl_strategies = [s for s in STRATEGIES if s.kind == "tl"]

    for ti, target in enumerate(targets, start=1):
        stats = performance[target]
        anchor, online = make_crn_samples(
            target=target,
            target_stats=stats,
            replicates=args.replicates,
            max_horizon=max_h,
            anchor_n=args.anchor_samples,
            base_seed=args.mc_seed,
        )

        # Mean metrics per strategy/horizon for this target.
        target_metrics: dict[str, dict[int, dict[str, float]]] = {
            s.name: {h: {} for h in horizons}
            for s in STRATEGIES
        }

        # Baselines: one full trace per c/replicate, then prefixes for all H.
        for strat in baseline_strategies:
            by_h: dict[int, list[dict[str, float]]] = {h: [] for h in horizons}
            for rep in range(args.replicates):
                tr = simulate_baseline_trace(
                    target_stats=stats,
                    sequences={
                        "x86": online["x86"][rep],
                        "arm64": online["arm64"][rep],
                    },
                    c=float(strat.c),
                    convergence_run=args.convergence_run,
                )
                for h in horizons:
                    by_h[h].append(prefix_metrics_baseline(tr, h))
            for h in horizons:
                target_metrics[strat.name][h] = mean_rows(by_h[h])

        donor = donor_map[target]
        if donor is not None:
            donor_stats = performance[donor]

            # Priors depend on formula and anchor replicate, but NOT on w_R/w_E/c.
            prior_by_formula: dict[str, list[dict[str, float]]] = {}
            for formula in ("difference", "ratio"):
                priors: list[dict[str, float]] = []
                for rep in range(args.replicates):
                    prior, _ = build_prior(
                        mode=formula,
                        anchor_mean_reward_x86=float(anchor[rep]),
                        donor_stats=donor_stats,
                    )
                    priors.append(prior)
                prior_by_formula[formula] = priors

            for strat in tl_strategies:
                cfg = Config(
                    config_id=strat.name,
                    regime=strat.role,
                    w_r=float(strat.w_r),
                    w_e=float(strat.w_e),
                    c=float(strat.c),
                )
                by_h = {h: [] for h in horizons}
                for rep in range(args.replicates):
                    tr = simulate_tl_trace(
                        target_stats=stats,
                        sequences={
                            "x86": online["x86"][rep],
                            "arm64": online["arm64"][rep],
                        },
                        prior_mean_reward=prior_by_formula[strat.formula][rep],
                        config=cfg,
                        convergence_run=args.convergence_run,
                    )
                    for h in horizons:
                        by_h[h].append(prefix_metrics_tl(tr, h))
                for h in horizons:
                    target_metrics[strat.name][h] = mean_rows(by_h[h])
        else:
            # No donor => exact no-transfer behavior with the strategy's own c.
            for strat in tl_strategies:
                base_name = matched_baseline_name(strat.c)
                for h in horizons:
                    target_metrics[strat.name][h] = dict(target_metrics[base_name][h])

        for strat in STRATEGIES:
            for h in horizons:
                met = target_metrics[strat.name][h]
                matched_name = matched_baseline_name(strat.c)
                matched = target_metrics[matched_name][h]
                common = target_metrics[COMMON_BASELINE][h]

                matched_gain = 100.0 * (
                        float(matched["cumulative_latency_ms"])
                        - float(met["cumulative_latency_ms"])
                ) / float(matched["cumulative_latency_ms"])

                common_gain = 100.0 * (
                        float(common["cumulative_latency_ms"])
                        - float(met["cumulative_latency_ms"])
                ) / float(common["cumulative_latency_ms"])

                rows.append({
                    "target_function": target,
                    "target_is_synthetic": bool(is_synthetic(target)),
                    "donor_function": donor or "",
                    "transfer_applied": float(strat.kind == "tl" and donor is not None),
                    "strategy": strat.name,
                    "role": strat.role,
                    "kind": strat.kind,
                    "formula": strat.formula or "none",
                    "w_r": float(strat.w_r),
                    "w_e": float(strat.w_e),
                    "c": float(strat.c),
                    "horizon": int(h),
                    **met,
                    "matched_baseline_strategy": matched_name,
                    "matched_c_gain_percent": float(matched_gain),
                    "common_c08_gain_percent": float(common_gain),
                    "matched_optimal_arm_rate_gain": float(
                        met["optimal_arm_rate"] - matched["optimal_arm_rate"]
                    ),
                    "matched_wrong_selections_saved": float(
                        matched["wrong_selections"] - met["wrong_selections"]
                    ),
                    "matched_pseudo_regret_saved": float(
                        matched["cumulative_reward_pseudo_regret"]
                        - met["cumulative_reward_pseudo_regret"]
                    ),
                })

        if ti % 5 == 0 or ti == len(targets):
            print(f"target_progress={ti}/{len(targets)}")

    return pd.DataFrame(rows), horizons, targets, donor_map


def build_summary(per_target: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "primary_57_non_synthetic":
        data = per_target[~per_target["target_is_synthetic"]].copy()
    elif scope == "all_59":
        data = per_target.copy()
    else:
        raise ValueError(scope)

    rows: list[dict[str, Any]] = []
    for (strategy, horizon), g in data.groupby(["strategy", "horizon"]):
        s = STRATEGY_BY_NAME[strategy]
        rows.append({
            "scope": scope,
            "strategy": strategy,
            "role": s.role,
            "formula": s.formula or "none",
            "w_r": s.w_r,
            "w_e": s.w_e,
            "c": s.c,
            "horizon": int(horizon),
            "target_count": int(g["target_function"].nunique()),
            "transfer_coverage": float(g["transfer_applied"].mean()),
            "matched_c_gain_percent": float(g["matched_c_gain_percent"].mean()),
            "common_c08_gain_percent": float(g["common_c08_gain_percent"].mean()),
            "cumulative_latency_ms": float(g["cumulative_latency_ms"].mean()),
            "optimal_arm_rate": float(g["optimal_arm_rate"].mean()),
            "wrong_selections": float(g["wrong_selections"].mean()),
            "cumulative_reward_pseudo_regret": float(
                g["cumulative_reward_pseudo_regret"].mean()
            ),
            "first_arm_optimal": float(g["first_arm_optimal"].mean()),
            "matched_optimal_arm_rate_gain": float(
                g["matched_optimal_arm_rate_gain"].mean()
            ),
            "matched_wrong_selections_saved": float(
                g["matched_wrong_selections_saved"].mean()
            ),
            "matched_pseudo_regret_saved": float(
                g["matched_pseudo_regret_saved"].mean()
            ),
            "negative_transfer_rate_matched_c": float(
                (g["matched_c_gain_percent"] < 0).mean()
            ),
            "p10_matched_c_gain": float(g["matched_c_gain_percent"].quantile(0.10)),
        })
    order = {s.name: i for i, s in enumerate(STRATEGIES)}
    out = pd.DataFrame(rows)
    out["_order"] = out["strategy"].map(order)
    return (
        out.sort_values(["horizon", "_order"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def paired_bootstrap(
        data: pd.DataFrame,
        strategy_a: str,
        strategy_b: str,
        horizon: int,
        bootstrap_replicates: int,
        seed: int,
        comparison: str,
) -> list[dict[str, Any]]:
    a = data[
        (data["strategy"] == strategy_a) & (data["horizon"] == horizon)
        ].set_index("target_function")
    b = data[
        (data["strategy"] == strategy_b) & (data["horizon"] == horizon)
        ].set_index("target_function")
    targets = sorted(set(a.index) & set(b.index))
    if len(targets) != 57:
        raise ValueError(
            f"primary paired comparison expected 57 targets, got {len(targets)}: {comparison}"
        )

    metrics: dict[str, np.ndarray] = {}
    a_lat = a.loc[targets, "cumulative_latency_ms"].to_numpy(float)
    b_lat = b.loc[targets, "cumulative_latency_ms"].to_numpy(float)
    metrics["latency_reduction_percent"] = 100.0 * (a_lat - b_lat) / a_lat
    metrics["common_c08_gain_delta_pp"] = (
            b.loc[targets, "common_c08_gain_percent"].to_numpy(float)
            - a.loc[targets, "common_c08_gain_percent"].to_numpy(float)
    )
    metrics["optimal_arm_rate_delta"] = (
            b.loc[targets, "optimal_arm_rate"].to_numpy(float)
            - a.loc[targets, "optimal_arm_rate"].to_numpy(float)
    )
    metrics["wrong_selections_delta"] = (
            b.loc[targets, "wrong_selections"].to_numpy(float)
            - a.loc[targets, "wrong_selections"].to_numpy(float)
    )
    metrics["pseudo_regret_delta"] = (
            b.loc[targets, "cumulative_reward_pseudo_regret"].to_numpy(float)
            - a.loc[targets, "cumulative_reward_pseudo_regret"].to_numpy(float)
    )

    rng = np.random.default_rng(seed)
    inds = rng.integers(
        0, len(targets), size=(bootstrap_replicates, len(targets))
    )
    rows: list[dict[str, Any]] = []
    for metric, delta in metrics.items():
        means = delta[inds].mean(axis=1)
        rows.append({
            "comparison": comparison,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "horizon": int(horizon),
            "metric": metric,
            "point_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "paired_targets": int(len(targets)),
            "bootstrap_replicates": int(bootstrap_replicates),
        })
    return rows


def pair_effects(
        data: pd.DataFrame,
        strategy_a: str,
        strategy_b: str,
        horizon: int,
        comparison: str,
) -> dict[str, Any]:
    a = data[
        (data["strategy"] == strategy_a) & (data["horizon"] == horizon)
        ].set_index("target_function")
    b = data[
        (data["strategy"] == strategy_b) & (data["horizon"] == horizon)
        ].set_index("target_function")
    targets = sorted(set(a.index) & set(b.index))
    a_lat = a.loc[targets, "cumulative_latency_ms"].to_numpy(float)
    b_lat = b.loc[targets, "cumulative_latency_ms"].to_numpy(float)
    red = 100.0 * (a_lat - b_lat) / a_lat
    ties = np.isclose(a_lat, b_lat, rtol=1e-12, atol=1e-12)
    wins = b_lat < a_lat
    losses = b_lat > a_lat
    return {
        "comparison": comparison,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "horizon": int(horizon),
        "target_count": int(len(targets)),
        "mean_latency_reduction_percent": float(red.mean()),
        "median_latency_reduction_percent": float(np.median(red)),
        "target_win_rate_b": float(wins.mean()),
        "target_loss_rate_b": float(losses.mean()),
        "target_tie_rate": float(ties.mean()),
        "common_c08_gain_delta_pp": float(
            (
                    b.loc[targets, "common_c08_gain_percent"].to_numpy(float)
                    - a.loc[targets, "common_c08_gain_percent"].to_numpy(float)
            ).mean()
        ),
        "optimal_arm_rate_delta": float(
            (
                    b.loc[targets, "optimal_arm_rate"].to_numpy(float)
                    - a.loc[targets, "optimal_arm_rate"].to_numpy(float)
            ).mean()
        ),
        "wrong_selections_delta": float(
            (
                    b.loc[targets, "wrong_selections"].to_numpy(float)
                    - a.loc[targets, "wrong_selections"].to_numpy(float)
            ).mean()
        ),
        "pseudo_regret_delta": float(
            (
                    b.loc[targets, "cumulative_reward_pseudo_regret"].to_numpy(float)
                    - a.loc[targets, "cumulative_reward_pseudo_regret"].to_numpy(float)
            ).mean()
        ),
    }


def write_report(
        out: Path,
        summary_primary: pd.DataFrame,
        paired: pd.DataFrame,
        effects: pd.DataFrame,
        args: argparse.Namespace,
) -> None:
    h10 = summary_primary[summary_primary["horizon"] == PRIMARY_HORIZON].copy()
    cols = [
        "strategy", "formula", "w_r", "w_e", "c",
        "matched_c_gain_percent", "common_c08_gain_percent",
        "optimal_arm_rate", "wrong_selections",
        "cumulative_reward_pseudo_regret",
        "negative_transfer_rate_matched_c",
    ]

    key_comparisons = {
        "effect_of_c_08_to_02",
        "new_difference_vs_matched_notl",
        "new_ratio_vs_matched_notl",
        "new_vs_frozen_difference",
        "new_vs_frozen_ratio",
        "ratio_vs_difference_same_parameters",
    }
    p = paired[
        (paired["horizon"] == PRIMARY_HORIZON)
        & (paired["comparison"].isin(key_comparisons))
        & (paired["metric"] == "latency_reduction_percent")
        ].copy()

    lines = [
        "# Final plain-Manhattan confirmatory replay",
        "",
        "This is a fresh-seed confirmatory Monte Carlo replay. It does not retune the GCP configurations.",
        "",
        f"- donor variant: `{FINAL_VARIANT}`",
        f"- donor seed: `{args.donor_seed}`",
        f"- MC replicates: `{args.replicates}`",
        f"- MC seed: `{args.mc_seed}`",
        f"- horizons: `{args.horizons}`",
        f"- anchor samples: `{args.anchor_samples}`",
        f"- bootstrap replicates: `{args.bootstrap_replicates}`",
        "- primary scope: 57 non-synthetic target functions",
        "- common random numbers across all strategies",
        "- target ARM samples are replay/evaluation only and never construct the prior",
        "",
        "## H=10 primary summary",
        "",
        h10[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## H=10 key paired latency comparisons",
        "",
        p[[
            "comparison", "strategy_a", "strategy_b",
            "point_delta", "ci95_low", "ci95_high"
        ]].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation guardrail",
        "",
        "`difference_new` and `ratio_new` are post-hoc sensitivity candidates. The frozen GCP settings remain Difference (.05,64,.2) and Ratio (.01,64,.4).",
        "",
    ]
    (out / "CONFIRMATORY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    implementation_self_check()
    horizons = parse_ints(args.horizons)
    if PRIMARY_HORIZON not in horizons:
        raise SystemExit("H=10 must be included because it is the primary endpoint")
    if args.replicates <= 0 or args.anchor_samples <= 0:
        raise SystemExit("replicates and anchor-samples must be positive")
    if args.bootstrap_replicates <= 0:
        raise SystemExit("bootstrap-replicates must be positive")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    per_target, horizons, targets, donor_map = run_replay(args)
    per_target_path = out / "confirmatory_per_target.csv"
    per_target.to_csv(per_target_path, index=False)

    summary_primary = build_summary(per_target, "primary_57_non_synthetic")
    summary_all = build_summary(per_target, "all_59")
    summary = pd.concat([summary_primary, summary_all], ignore_index=True)
    summary_path = out / "confirmatory_summary.csv"
    summary.to_csv(summary_path, index=False)

    primary = per_target[~per_target["target_is_synthetic"]].copy()

    paired_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for strategy_a, strategy_b, comparison in PAIRS:
        for h in horizons:
            paired_rows.extend(
                paired_bootstrap(
                    primary,
                    strategy_a,
                    strategy_b,
                    h,
                    args.bootstrap_replicates,
                    stable_seed(args.bootstrap_seed, comparison, h),
                    comparison,
                )
            )
            effect_rows.append(
                pair_effects(primary, strategy_a, strategy_b, h, comparison)
            )

    paired = pd.DataFrame(paired_rows)
    paired_path = out / "confirmatory_paired_bootstrap.csv"
    paired.to_csv(paired_path, index=False)

    effects = pd.DataFrame(effect_rows)
    effects_path = out / "confirmatory_pair_effects.csv"
    effects.to_csv(effects_path, index=False)

    # Chart-ready long table: primary scope only.
    chart_cols = [
        "strategy", "role", "formula", "w_r", "w_e", "c", "horizon",
        "matched_c_gain_percent", "common_c08_gain_percent",
        "optimal_arm_rate", "wrong_selections",
        "cumulative_reward_pseudo_regret",
        "negative_transfer_rate_matched_c",
    ]
    chart_path = out / "confirmatory_chart_ready.csv"
    summary_primary[chart_cols].to_csv(chart_path, index=False)

    root = args.root.resolve()
    donor_detail = args.donor_detail.resolve()
    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"
    manifest = {
        "experiment": "final plain-Manhattan post-sensitivity confirmatory replay",
        "is_tuning": False,
        "posthoc_candidates": ["difference_new", "ratio_new"],
        "gcp_frozen_configs_unchanged": True,
        "primary_scope": "57 non-synthetic targets",
        "secondary_scope": "all 59 targets",
        "primary_horizon": PRIMARY_HORIZON,
        "donor_pipeline": {
            "variant": FINAL_VARIANT,
            "seed": int(args.donor_seed),
            "donor_detail": str(donor_detail),
            "sha256": sha256(donor_detail),
        },
        "simulation": {
            "horizons": horizons,
            "replicates": int(args.replicates),
            "mc_seed": int(args.mc_seed),
            "anchor_samples": int(args.anchor_samples),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "convergence_run": int(args.convergence_run),
        },
        "bootstrap": {
            "unit": "target_function",
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "ci": "percentile 95%",
        },
        "strategies": {
            s.name: {
                "kind": s.kind,
                "role": s.role,
                "formula": s.formula,
                "w_R": s.w_r,
                "w_E": s.w_e,
                "c": s.c,
                "matched_baseline": matched_baseline_name(s.c),
            }
            for s in STRATEGIES
        },
        "inputs": {
            "raw_x86": {"path": str(raw_x), "sha256": sha256(raw_x)},
            "raw_arm64": {"path": str(raw_a), "sha256": sha256(raw_a)},
        },
        "donor_coverage": {
            "selected": int(sum(d is not None for d in donor_map.values())),
            "total": int(len(targets)),
        },
    }
    (out / "confirmatory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    write_report(out, summary_primary, paired, effects, args)

    print("PASS — final plain confirmatory replay completed")
    print()
    print("===== PRIMARY H=10 SUMMARY =====")
    h10 = summary_primary[summary_primary["horizon"] == PRIMARY_HORIZON]
    print(
        h10[[
            "strategy", "formula", "w_r", "w_e", "c",
            "matched_c_gain_percent", "common_c08_gain_percent",
            "optimal_arm_rate", "wrong_selections",
            "cumulative_reward_pseudo_regret",
            "negative_transfer_rate_matched_c",
        ]].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print()
    print("===== PRIMARY H=10 PAIRED LATENCY =====")
    key = paired[
        (paired["horizon"] == PRIMARY_HORIZON)
        & (paired["metric"] == "latency_reduction_percent")
        ]
    print(
        key[[
            "comparison", "strategy_a", "strategy_b",
            "point_delta", "ci95_low", "ci95_high",
        ]].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print()
    print(f"per_target={per_target_path}")
    print(f"summary={summary_path}")
    print(f"paired={paired_path}")
    print(f"effects={effects_path}")
    print(f"chart_ready={chart_path}")
    print(f"report={out / 'CONFIRMATORY_REPORT.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--donor-detail", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--donor-seed", type=int, default=11)
    p.add_argument("--horizons", default="5,10,20,50")
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--mc-seed", type=int, default=2026101202)
    p.add_argument("--anchor-samples", type=int, default=10)
    p.add_argument("--convergence-run", type=int, default=5)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2026101203)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
