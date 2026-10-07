#!/usr/bin/env python3
"""
Serverledge — FINAL post-tuning validation replay.

This is NOT a tuning script.

Purpose
-------
Evaluate the already-frozen pre-tuning and post-tuning configurations on a
fresh Monte Carlo stream, with the frozen weighted_vote_p2 donor mapping and
identical CRN across all strategies.

Frozen strategies
-----------------
1. no_transfer
   classic UCB1 baseline, c = 0.8

2. difference_original
   formula = difference
   w_R = 0.25, w_E = 1.0, c = 0.8

3. ratio_original
   formula = ratio
   w_R = 0.25, w_E = 1.0, c = 0.8

4. difference_tuned
   formula = difference
   w_R = 0.05, w_E = 64.0, c = 0.2

5. ratio_tuned
   formula = ratio
   w_R = 0.01, w_E = 64.0, c = 0.4

The tuned configurations come from the previously completed boundary-extension
study. This script MUST NOT select or alter hyperparameters.

Primary paired comparisons
--------------------------
- difference_tuned - difference_original
- ratio_tuned      - ratio_original
- ratio_tuned      - difference_tuned

Secondary comparisons
---------------------
- every TL strategy vs no_transfer
- ratio_original - difference_original

Simulation
----------
- frozen donor variant: weighted_vote_p2
- target x86 anchor: 10 bootstrap samples
- anchor RNG independent from online replay
- common random numbers across all strategies
- fresh Monte Carlo seed (default 2026100709)
- 1000 replay replicates by default
- H = {5, 10, 20, 50}
- paired target-level percentile bootstrap, 10000 replicates by default

Outputs
-------
tl-post-tuning-per-target-seed.csv
tl-post-tuning-per-target.csv
tl-post-tuning-summary.csv
tl-post-tuning-paired-bootstrap.csv
tl-post-tuning-effect-sizes.csv
tl-post-tuning-report.md
tl-post-tuning-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.profiling.transfer_prior_formula_comparison import (
    build_performance,
    build_prior,
    load_eligible_durations,
    load_frozen_donor_map,
    make_crn_samples,
    stable_seed,
)
from analysis.profiling.transfer_parameter_tuning_cv import (
    Config,
    compare_to_baseline,
    implementation_self_check,
    mean_metrics,
    prefix_metrics_baseline,
    prefix_metrics_tl,
    simulate_baseline_trace,
    simulate_tl_trace,
)


DONOR_VARIANT = "weighted_vote_p2"

STRATEGY_SPECS = {
    "difference_original": {
        "formula": "difference",
        "config": Config(
            config_id="difference-original",
            regime="pre_tuning",
            w_r=0.25,
            w_e=1.0,
            c=0.8,
        ),
    },
    "ratio_original": {
        "formula": "ratio",
        "config": Config(
            config_id="ratio-original",
            regime="pre_tuning",
            w_r=0.25,
            w_e=1.0,
            c=0.8,
        ),
    },
    "difference_tuned": {
        "formula": "difference",
        "config": Config(
            config_id="difference-tuned",
            regime="post_tuning_frozen",
            w_r=0.05,
            w_e=64.0,
            c=0.2,
        ),
    },
    "ratio_tuned": {
        "formula": "ratio",
        "config": Config(
            config_id="ratio-tuned",
            regime="post_tuning_frozen",
            w_r=0.01,
            w_e=64.0,
            c=0.4,
        ),
    },
}

STRATEGY_ORDER = [
    "no_transfer",
    "difference_original",
    "ratio_original",
    "difference_tuned",
    "ratio_tuned",
]

PRIMARY_COMPARISONS = [
    ("difference_original", "difference_tuned"),
    ("ratio_original", "ratio_tuned"),
    ("difference_tuned", "ratio_tuned"),
]

ALL_COMPARISONS = [
    ("no_transfer", "difference_original"),
    ("no_transfer", "ratio_original"),
    ("no_transfer", "difference_tuned"),
    ("no_transfer", "ratio_tuned"),
    ("difference_original", "difference_tuned"),
    ("ratio_original", "ratio_tuned"),
    ("difference_original", "ratio_original"),
    ("difference_tuned", "ratio_tuned"),
]


def parse_int_list(value: str) -> list[int]:
    values = [
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    ]
    if not values:
        raise ValueError("empty integer list")
    return values


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_target_seed_to_target(
    per_target_seed: pd.DataFrame,
) -> pd.DataFrame:
    excluded = {
        "target_function",
        "strategy",
        "formula",
        "cluster_seed",
        "donor_function",
        "regime",
        "horizon",
    }

    numeric_cols = [
        column
        for column in per_target_seed.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            per_target_seed[column]
        )
    ]

    return (
        per_target_seed.groupby(
            [
                "target_function",
                "strategy",
                "formula",
                "regime",
                "horizon",
            ],
            as_index=False,
        )[numeric_cols]
        .mean()
    )


def build_macro_summary(
    per_target: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        strategy,
        formula,
        regime,
        horizon,
    ), group in per_target.groupby(
        [
            "strategy",
            "formula",
            "regime",
            "horizon",
        ]
    ):
        rows.append(
            {
                "strategy": strategy,
                "formula": formula,
                "regime": regime,
                "horizon": int(horizon),
                "target_count": int(
                    group["target_function"].nunique()
                ),
                "transfer_applied_macro_mean": float(
                    group["transfer_applied"].mean()
                ),
                "latency_gain_percent_vs_no_transfer_macro_mean": float(
                    group[
                        "latency_gain_percent_vs_no_transfer"
                    ].mean()
                ),
                "cumulative_latency_ms_macro_mean": float(
                    group["cumulative_latency_ms"].mean()
                ),
                "optimal_arm_rate_macro_mean": float(
                    group["optimal_arm_rate"].mean()
                ),
                "wrong_selections_macro_mean": float(
                    group["wrong_selections"].mean()
                ),
                "cumulative_reward_pseudo_regret_macro_mean": float(
                    group[
                        "cumulative_reward_pseudo_regret"
                    ].mean()
                ),
                "first_arm_optimal_macro_mean": float(
                    group["first_arm_optimal"].mean()
                ),
                "convergence_probability_macro_mean": float(
                    group["convergence_probability"].mean()
                ),
                "censored_convergence_request_macro_mean": float(
                    group[
                        "censored_convergence_request"
                    ].mean()
                ),
            }
        )

    result = pd.DataFrame(rows)
    order = {
        strategy: idx
        for idx, strategy in enumerate(
            STRATEGY_ORDER
        )
    }
    result["_order"] = result["strategy"].map(order)

    return (
        result.sort_values(
            ["horizon", "_order"]
        )
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def paired_bootstrap(
    *,
    per_target: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Report B - A. Bootstrap resampling unit is target_function.
    """
    a = (
        per_target[
            (per_target["strategy"] == strategy_a)
            & (per_target["horizon"] == horizon)
        ]
        .set_index("target_function")
    )

    b = (
        per_target[
            (per_target["strategy"] == strategy_b)
            & (per_target["horizon"] == horizon)
        ]
        .set_index("target_function")
    )

    targets = sorted(set(a.index) & set(b.index))

    metrics = [
        "latency_gain_percent_vs_no_transfer",
        "optimal_arm_rate",
        "wrong_selections",
        "cumulative_reward_pseudo_regret",
        "first_arm_optimal",
        "convergence_probability",
        "censored_convergence_request",
    ]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        deltas = np.asarray(
            [
                float(b.loc[target, metric])
                - float(a.loc[target, metric])
                for target in targets
            ],
            dtype=float,
        )

        boot = np.empty(replicates, dtype=float)

        for i in range(replicates):
            idx = rng.integers(
                0,
                len(deltas),
                size=len(deltas),
            )
            boot[i] = float(
                np.mean(deltas[idx])
            )

        rows.append(
            {
                "comparison": f"{strategy_b}_minus_{strategy_a}",
                "horizon": int(horizon),
                "metric": metric,
                "point_delta": float(np.mean(deltas)),
                "ci95_low": float(
                    np.quantile(boot, 0.025)
                ),
                "ci95_high": float(
                    np.quantile(boot, 0.975)
                ),
                "target_count": int(len(deltas)),
                "bootstrap_replicates": int(replicates),
            }
        )

    return rows


def pair_effect_size(
    *,
    per_target: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    horizon: int,
) -> dict[str, Any]:
    """
    B relative to A. This table is descriptive; inferential intervals are in
    tl-post-tuning-paired-bootstrap.csv.
    """
    a = (
        per_target[
            (per_target["strategy"] == strategy_a)
            & (per_target["horizon"] == horizon)
        ]
        .set_index("target_function")
    )
    b = (
        per_target[
            (per_target["strategy"] == strategy_b)
            & (per_target["horizon"] == horizon)
        ]
        .set_index("target_function")
    )

    targets = sorted(set(a.index) & set(b.index))

    a_latency = np.asarray(
        [
            float(a.loc[t, "cumulative_latency_ms"])
            for t in targets
        ],
        dtype=float,
    )
    b_latency = np.asarray(
        [
            float(b.loc[t, "cumulative_latency_ms"])
            for t in targets
        ],
        dtype=float,
    )

    a_gain = np.asarray(
        [
            float(
                a.loc[
                    t,
                    "latency_gain_percent_vs_no_transfer",
                ]
            )
            for t in targets
        ],
        dtype=float,
    )
    b_gain = np.asarray(
        [
            float(
                b.loc[
                    t,
                    "latency_gain_percent_vs_no_transfer",
                ]
            )
            for t in targets
        ],
        dtype=float,
    )

    latency_reduction_vs_a = (
        100.0
        * (a_latency - b_latency)
        / a_latency
    )

    wins = int(np.sum(b_latency < a_latency))
    ties = int(
        np.sum(
            np.isclose(
                b_latency,
                a_latency,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    )
    losses = int(len(targets) - wins - ties)

    a_gain_macro = float(np.mean(a_gain))
    b_gain_macro = float(np.mean(b_gain))
    gain_delta = b_gain_macro - a_gain_macro

    if abs(a_gain_macro) > 1e-12:
        relative_gain_lift = (
            100.0
            * gain_delta
            / abs(a_gain_macro)
        )
    else:
        relative_gain_lift = np.nan

    return {
        "comparison": f"{strategy_b}_minus_{strategy_a}",
        "horizon": int(horizon),
        "target_count": int(len(targets)),
        "strategy_a_latency_gain_percent": a_gain_macro,
        "strategy_b_latency_gain_percent": b_gain_macro,
        "latency_gain_delta_pp": float(gain_delta),
        "relative_lift_in_latency_gain_percent": float(
            relative_gain_lift
        )
        if np.isfinite(relative_gain_lift)
        else np.nan,
        "mean_latency_reduction_vs_strategy_a_percent": float(
            np.mean(latency_reduction_vs_a)
        ),
        "median_latency_reduction_vs_strategy_a_percent": float(
            np.median(latency_reduction_vs_a)
        ),
        "target_latency_win_rate": float(
            wins / len(targets)
        ),
        "target_latency_loss_rate": float(
            losses / len(targets)
        ),
        "target_latency_tie_rate": float(
            ties / len(targets)
        ),
        "optimal_arm_rate_delta": float(
            np.mean(
                [
                    float(b.loc[t, "optimal_arm_rate"])
                    - float(a.loc[t, "optimal_arm_rate"])
                    for t in targets
                ]
            )
        ),
        "wrong_selections_delta": float(
            np.mean(
                [
                    float(b.loc[t, "wrong_selections"])
                    - float(a.loc[t, "wrong_selections"])
                    for t in targets
                ]
            )
        ),
        "pseudo_regret_delta": float(
            np.mean(
                [
                    float(
                        b.loc[
                            t,
                            "cumulative_reward_pseudo_regret",
                        ]
                    )
                    - float(
                        a.loc[
                            t,
                            "cumulative_reward_pseudo_regret",
                        ]
                    )
                    for t in targets
                ]
            )
        ),
    }


def main(args: argparse.Namespace) -> None:
    implementation_self_check()

    horizons = sorted(
        set(
            parse_int_list(
                args.horizons
            )
        )
    )
    max_horizon = max(horizons)

    raw_x86_path = args.raw_x86.resolve()
    raw_arm64_path = args.raw_arm64.resolve()
    donor_detail_path = args.donor_detail.resolve()

    x86 = load_eligible_durations(raw_x86_path)
    arm64 = load_eligible_durations(raw_arm64_path)
    performance = build_performance(x86, arm64)

    (
        donor_maps,
        cluster_seeds,
        targets,
    ) = load_frozen_donor_map(donor_detail_path)

    missing_targets = [
        target
        for target in targets
        if target not in performance
    ]
    if missing_targets:
        raise SystemExit(
            "Missing target raw data: "
            + ", ".join(missing_targets)
        )

    all_donors = sorted(
        {
            donor
            for target_map in donor_maps.values()
            for donor in target_map.values()
            if donor is not None
        }
    )

    missing_donors = [
        donor
        for donor in all_donors
        if donor not in performance
    ]
    if missing_donors:
        raise SystemExit(
            "Missing donor raw data: "
            + ", ".join(missing_donors)
        )

    print(
        "FINAL POST-TUNING VALIDATION "
        f"targets={len(targets)} "
        f"cluster_seeds={len(cluster_seeds)} "
        f"mc_replicates={args.replicates} "
        f"mc_seed={args.mc_seed}"
    )

    print("FROZEN CONFIGURATIONS")
    for strategy, spec in STRATEGY_SPECS.items():
        cfg = spec["config"]
        print(
            f"{strategy}: "
            f"formula={spec['formula']} "
            f"w_R={cfg.w_r} "
            f"w_E={cfg.w_e} "
            f"c={cfg.c}"
        )

    per_target_seed_rows: list[dict[str, Any]] = []

    total_targets = len(targets)

    for target_idx, target in enumerate(
        targets,
        start=1,
    ):
        target_stats = performance[target]

        (
            anchor_rewards,
            online,
        ) = make_crn_samples(
            target=target,
            target_stats=target_stats,
            replicates=args.replicates,
            max_horizon=max_horizon,
            anchor_n=args.anchor_samples,
            base_seed=args.mc_seed,
        )

        baseline_runs_by_horizon = {
            horizon: []
            for horizon in horizons
        }

        for rep in range(args.replicates):
            trace = simulate_baseline_trace(
                target_stats=target_stats,
                sequences={
                    "x86": online["x86"][rep],
                    "arm64": online["arm64"][rep],
                },
                c=args.baseline_c,
                convergence_run=args.convergence_run,
            )

            for horizon in horizons:
                baseline_runs_by_horizon[
                    horizon
                ].append(
                    prefix_metrics_baseline(
                        trace,
                        horizon,
                    )
                )

        baseline_by_horizon = {
            horizon: mean_metrics(rows)
            for horizon, rows
            in baseline_runs_by_horizon.items()
        }

        # Keep the table rectangular over the frozen cluster seeds.
        for cluster_seed in cluster_seeds:
            for horizon in horizons:
                base = baseline_by_horizon[horizon]

                per_target_seed_rows.append(
                    {
                        "target_function": target,
                        "cluster_seed": int(cluster_seed),
                        "strategy": "no_transfer",
                        "formula": "none",
                        "regime": "baseline",
                        "donor_function": "",
                        "transfer_applied": 0.0,
                        "w_r": 0.0,
                        "w_e": 0.0,
                        "c": float(args.baseline_c),
                        "horizon": int(horizon),
                        **base,
                        "latency_gain_percent_vs_no_transfer": 0.0,
                        "reward_pseudo_regret_saved": 0.0,
                        "optimal_arm_rate_gain": 0.0,
                        "wrong_selections_saved": 0.0,
                        "first_arm_optimal_gain": 0.0,
                        "convergence_probability_gain": 0.0,
                        "convergence_requests_saved": 0.0,
                    }
                )

        # Cache each unique donor/strategy combination once for this target.
        unique_donors = sorted(
            {
                donor_maps[seed][target]
                for seed in cluster_seeds
                if donor_maps[seed][target] is not None
            }
        )

        strategy_donor_cache: dict[
            tuple[str, str],
            dict[int, dict[str, float]],
        ] = {}

        for strategy, spec in STRATEGY_SPECS.items():
            formula = str(spec["formula"])
            config: Config = spec["config"]

            for donor in unique_donors:
                donor_stats = performance[donor]

                runs_by_horizon = {
                    horizon: []
                    for horizon in horizons
                }

                for rep in range(args.replicates):
                    prior, _ = build_prior(
                        mode=formula,
                        anchor_mean_reward_x86=float(
                            anchor_rewards[rep]
                        ),
                        donor_stats=donor_stats,
                    )

                    trace = simulate_tl_trace(
                        target_stats=target_stats,
                        sequences={
                            "x86": online["x86"][rep],
                            "arm64": online["arm64"][rep],
                        },
                        prior_mean_reward=prior,
                        config=config,
                        convergence_run=args.convergence_run,
                    )

                    for horizon in horizons:
                        runs_by_horizon[
                            horizon
                        ].append(
                            prefix_metrics_tl(
                                trace,
                                horizon,
                            )
                        )

                strategy_donor_cache[
                    (strategy, donor)
                ] = {
                    horizon: mean_metrics(rows)
                    for horizon, rows
                    in runs_by_horizon.items()
                }

        for cluster_seed in cluster_seeds:
            donor = donor_maps[cluster_seed][target]

            for strategy, spec in STRATEGY_SPECS.items():
                formula = str(spec["formula"])
                config: Config = spec["config"]

                for horizon in horizons:
                    base = baseline_by_horizon[horizon]

                    if donor is None:
                        metrics = dict(base)
                        comparisons = {
                            "latency_gain_percent_vs_no_transfer": 0.0,
                            "reward_pseudo_regret_saved": 0.0,
                            "optimal_arm_rate_gain": 0.0,
                            "wrong_selections_saved": 0.0,
                            "first_arm_optimal_gain": 0.0,
                            "convergence_probability_gain": 0.0,
                            "convergence_requests_saved": 0.0,
                        }
                        applied = 0.0
                        donor_name = ""
                    else:
                        metrics = strategy_donor_cache[
                            (strategy, donor)
                        ][horizon]
                        comparisons = compare_to_baseline(
                            metrics,
                            base,
                        )
                        applied = 1.0
                        donor_name = donor

                    per_target_seed_rows.append(
                        {
                            "target_function": target,
                            "cluster_seed": int(cluster_seed),
                            "strategy": strategy,
                            "formula": formula,
                            "regime": config.regime,
                            "donor_function": donor_name,
                            "transfer_applied": applied,
                            "w_r": config.w_r,
                            "w_e": config.w_e,
                            "c": config.c,
                            "horizon": int(horizon),
                            **metrics,
                            **comparisons,
                        }
                    )

        if (
            target_idx % 5 == 0
            or target_idx == total_targets
        ):
            print(
                f"target_progress="
                f"{target_idx}/{total_targets}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_target_seed = pd.DataFrame(
        per_target_seed_rows
    )

    per_target_seed_path = (
        args.output_dir
        / "tl-post-tuning-per-target-seed.csv"
    )
    per_target_seed.to_csv(
        per_target_seed_path,
        index=False,
    )

    per_target = aggregate_target_seed_to_target(
        per_target_seed
    )

    per_target_path = (
        args.output_dir
        / "tl-post-tuning-per-target.csv"
    )
    per_target.to_csv(
        per_target_path,
        index=False,
    )

    summary = build_macro_summary(
        per_target
    )

    summary_path = (
        args.output_dir
        / "tl-post-tuning-summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    paired_rows: list[dict[str, Any]] = []

    for strategy_a, strategy_b in ALL_COMPARISONS:
        for horizon in horizons:
            paired_rows.extend(
                paired_bootstrap(
                    per_target=per_target,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    horizon=horizon,
                    replicates=args.bootstrap_replicates,
                    seed=stable_seed(
                        args.bootstrap_seed,
                        strategy_a,
                        strategy_b,
                        horizon,
                    ),
                )
            )

    paired = pd.DataFrame(paired_rows)

    paired_path = (
        args.output_dir
        / "tl-post-tuning-paired-bootstrap.csv"
    )
    paired.to_csv(
        paired_path,
        index=False,
    )

    effect_rows = []

    for strategy_a, strategy_b in ALL_COMPARISONS:
        for horizon in horizons:
            effect_rows.append(
                pair_effect_size(
                    per_target=per_target,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    horizon=horizon,
                )
            )

    effects = pd.DataFrame(effect_rows)

    effects_path = (
        args.output_dir
        / "tl-post-tuning-effect-sizes.csv"
    )
    effects.to_csv(
        effects_path,
        index=False,
    )

    manifest = {
        "experiment": "final post-tuning validation replay",
        "is_tuning": False,
        "frozen_donor_variant": DONOR_VARIANT,
        "strategies": {
            "no_transfer": {
                "policy": "classic UCB1",
                "c": args.baseline_c,
            },
            **{
                strategy: {
                    "formula": spec["formula"],
                    "w_R": spec["config"].w_r,
                    "w_E": spec["config"].w_e,
                    "c": spec["config"].c,
                    "regime": spec["config"].regime,
                }
                for strategy, spec in STRATEGY_SPECS.items()
            },
        },
        "simulation": {
            "horizons": horizons,
            "replicates": int(args.replicates),
            "mc_seed": int(args.mc_seed),
            "anchor_samples": int(args.anchor_samples),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "cluster_seeds": [
                int(seed)
                for seed in cluster_seeds
            ],
            "convergence_run": int(
                args.convergence_run
            ),
        },
        "bootstrap": {
            "unit": "target_function",
            "replicates": int(
                args.bootstrap_replicates
            ),
            "seed": int(
                args.bootstrap_seed
            ),
            "ci": "percentile 95%",
        },
        "inputs": {
            "raw_x86": {
                "path": str(raw_x86_path),
                "sha256": sha256_file(
                    raw_x86_path
                ),
            },
            "raw_arm64": {
                "path": str(raw_arm64_path),
                "sha256": sha256_file(
                    raw_arm64_path
                ),
            },
            "donor_detail": {
                "path": str(donor_detail_path),
                "sha256": sha256_file(
                    donor_detail_path
                ),
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "tl-post-tuning-manifest.json"
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

    report_lines = [
        "# Final post-tuning validation",
        "",
        "This is an independent replay with frozen hyperparameters.",
        "No parameter selection occurs in this experiment.",
        "",
        "## Frozen configurations",
        "",
        "- difference_original: w_R=0.25, w_E=1.0, c=0.8",
        "- ratio_original: w_R=0.25, w_E=1.0, c=0.8",
        "- difference_tuned: w_R=0.05, w_E=64.0, c=0.2",
        "- ratio_tuned: w_R=0.01, w_E=64.0, c=0.4",
        "",
        "## Macro summary",
        "",
    ]

    for _, row in summary.iterrows():
        report_lines.append(
            f"- strategy={row['strategy']}, "
            f"H={int(row['horizon'])}, "
            f"latency_gain="
            f"{row['latency_gain_percent_vs_no_transfer_macro_mean']:.4f}%, "
            f"optimal_arm_rate="
            f"{row['optimal_arm_rate_macro_mean']:.4f}, "
            f"wrong="
            f"{row['wrong_selections_macro_mean']:.4f}, "
            f"pseudo_regret="
            f"{row['cumulative_reward_pseudo_regret_macro_mean']:.4f}, "
            f"first_arm_optimal="
            f"{row['first_arm_optimal_macro_mean']:.4f}"
        )

    report_lines += [
        "",
        "## Primary tuning uplift",
        "",
    ]

    primary_names = {
        f"{b}_minus_{a}"
        for a, b in PRIMARY_COMPARISONS
    }

    for _, row in effects[
        effects["comparison"].isin(primary_names)
    ].sort_values(
        ["comparison", "horizon"]
    ).iterrows():
        report_lines.append(
            f"- {row['comparison']}, H={int(row['horizon'])}: "
            f"latency-gain delta={row['latency_gain_delta_pp']:.4f} pp, "
            f"relative lift={row['relative_lift_in_latency_gain_percent']:.2f}%, "
            f"mean direct latency reduction="
            f"{row['mean_latency_reduction_vs_strategy_a_percent']:.4f}%, "
            f"target win rate={row['target_latency_win_rate']:.4f}"
        )

    report_path = (
        args.output_dir
        / "tl-post-tuning-report.md"
    )
    report_path.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print("FINAL POST-TUNING SUMMARY")
    focus = [
        "strategy",
        "horizon",
        "transfer_applied_macro_mean",
        "latency_gain_percent_vs_no_transfer_macro_mean",
        "optimal_arm_rate_macro_mean",
        "wrong_selections_macro_mean",
        "cumulative_reward_pseudo_regret_macro_mean",
        "first_arm_optimal_macro_mean",
        "convergence_probability_macro_mean",
        "censored_convergence_request_macro_mean",
    ]
    print(
        summary[focus].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("PRIMARY TUNING EFFECT SIZES")
    primary_effects = effects[
        effects["comparison"].isin(primary_names)
    ].sort_values(
        ["comparison", "horizon"]
    )
    print(
        primary_effects.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("PRIMARY PAIRED BOOTSTRAP")
    primary_paired = paired[
        paired["comparison"].isin(primary_names)
    ].sort_values(
        [
            "comparison",
            "horizon",
            "metric",
        ]
    )
    print(
        primary_paired.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"per_target_seed={per_target_seed_path}")
    print(f"per_target={per_target_path}")
    print(f"summary={summary_path}")
    print(f"paired_bootstrap={paired_path}")
    print(f"effect_sizes={effects_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Final fresh-seed post-tuning replay for frozen Serverledge TL "
            "difference/ratio configurations."
        )
    )

    parser.add_argument(
        "--raw-x86",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raw-arm64",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--donor-detail",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--horizons",
        default="5,10,20,50",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=2026100709,
    )
    parser.add_argument(
        "--anchor-samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--baseline-c",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--convergence-run",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026100711,
    )

    return parser


if __name__ == "__main__":
    main(
        build_parser().parse_args()
    )
