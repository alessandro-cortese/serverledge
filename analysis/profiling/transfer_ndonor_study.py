#!/usr/bin/env python3
"""
Serverledge — UCB1 Transfer Learning n_donor study.

This experiment is downstream of the frozen clustering/donor-selection and
post-tuning UCB1 pipeline. It does NOT refit clustering, donor selection or
(w_R, w_E, c).

Research question
-----------------
Does transferring the amount of donor evidence into the UCB exploration
pseudo-count improve the already-tuned Transfer Learning policy?

Professor-inspired variant
--------------------------
For each architecture j:

    w_E,j*(n_donor) = w_E * n_donor

and therefore:

    B_j =
        c * sqrt(
            ln(sum_k(n_k + w_E,k*))
            /
            (n_j + w_E,j*)
        )

Here n_donor is the number of REAL historical donor observations used per
architecture to estimate the donor architecture effect.

Main operational interpretation
-------------------------------
For each Monte Carlo replicate and each donor:
1. make one deterministic random permutation of the donor's 12 real warm
   observations for x86 and one for ARM64;
2. for n_donor in {1,2,4,6,8,10,12}, use the first n_donor samples from each
   permutation (nested CRN across n);
3. estimate the donor reward means from only those observations;
4. build the Difference or Ratio prior from those means;
5. set exploration pseudo-count to w_E * n_donor;
6. replay the SAME target online CRN as all other strategies.

Thus larger n_donor means BOTH:
- a donor effect estimated from more real evidence;
- a larger transferred exploration pseudo-count.

The n_donor=12 condition is especially important:
the donor prior mean is then exactly the full frozen 12-sample donor mean,
so the comparison against the standard tuned policy isolates the effect of
changing w_E -> w_E * 12 while keeping donor evidence identical.

Frozen UCB1 TL configurations
-----------------------------
Difference:
    w_R = 0.05
    w_E = 64
    c   = 0.20

Ratio:
    w_R = 0.01
    w_E = 64
    c   = 0.40

Primary hypothesis test
-----------------------
At each H in {5,10,20,50}:
    n_donor=12 variant  vs  standard tuned TL
for Difference and Ratio separately.

The other n_donor values form a predeclared sensitivity curve. They are NOT
used to tune/select a new deployment hyperparameter.

Simulation
----------
- donor selection frozen: weighted_vote_p2
- 5 frozen clustering seeds
- target x86 anchor: 10 bootstrap samples
- donor evidence sampling: WITHOUT replacement from the 12 real warm samples
- donor evidence RNG is separate from target anchor and online replay
- nested donor subsets across n_donor
- common random numbers across all policies
- fresh MC seed
- H = {5,10,20,50}

Outputs
-------
tl-ndonor-per-target-seed.csv
tl-ndonor-per-target.csv
tl-ndonor-summary.csv
tl-ndonor-paired-bootstrap.csv
tl-ndonor-primary-effect-sizes.csv
tl-ndonor-evidence-diagnostics.csv
tl-ndonor-report.md
tl-ndonor-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.profiling.transfer_prior_formula_comparison import (
    ARMS,
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

FROZEN = {
    "difference": Config(
        config_id="difference-tuned",
        regime="post_tuning_frozen",
        w_r=0.05,
        w_e=64.0,
        c=0.20,
    ),
    "ratio": Config(
        config_id="ratio-tuned",
        regime="post_tuning_frozen",
        w_r=0.01,
        w_e=64.0,
        c=0.40,
    ),
}


def parse_int_list(value: str) -> list[int]:
    result = sorted(
        {
            int(x.strip())
            for x in value.split(",")
            if x.strip()
        }
    )
    if not result:
        raise ValueError("empty integer list")
    if any(x <= 0 for x in result):
        raise ValueError("all n_donor values must be > 0")
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def donor_subset_reward_means(
    *,
    donor_stats: dict[str, Any],
    target: str,
    donor: str,
    n_values: list[int],
    replicates: int,
    base_seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Nested WITHOUT-replacement donor evidence subsets.

    For each arm and replicate we generate one permutation of the observed
    donor warm samples; n=1,2,... use prefixes of that same permutation.
    """
    max_n = max(n_values)

    result: dict[int, dict[str, np.ndarray]] = {
        n: {
            arm: np.empty(replicates, dtype=float)
            for arm in ARMS
        }
        for n in n_values
    }

    for arm in ARMS:
        durations = np.asarray(
            donor_stats["durations"][arm],
            dtype=float,
        )

        if len(durations) < max_n:
            raise SystemExit(
                f"donor={donor} arm={arm}: "
                f"only {len(durations)} eligible warm samples, "
                f"need at least {max_n}"
            )

        rewards = -np.log(durations)

        rng = np.random.default_rng(
            stable_seed(
                base_seed,
                target,
                donor,
                "donor-evidence",
                arm,
            )
        )

        for rep in range(replicates):
            permutation = rng.permutation(len(rewards))
            ordered = rewards[permutation]
            cumulative = np.cumsum(ordered)

            for n in n_values:
                result[n][arm][rep] = float(
                    cumulative[n - 1] / n
                )

    return result


def subset_donor_stats(
    subset_means: dict[int, dict[str, np.ndarray]],
    *,
    n_donor: int,
    rep: int,
) -> dict[str, Any]:
    return {
        "mean_reward": {
            arm: float(
                subset_means[n_donor][arm][rep]
            )
            for arm in ARMS
        }
    }


def aggregate_target_seed_to_target(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "target_function",
        "strategy",
        "formula",
        "variant_kind",
        "n_donor",
        "horizon",
    ]

    exclude = set(group_cols) | {
        "cluster_seed",
        "donor_function",
    }

    numeric = [
        c
        for c in per_seed.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(
            per_seed[c]
        )
    ]

    return (
        per_seed.groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )[numeric]
        .mean()
    )


def build_summary(
    per_target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in per_target.groupby(
        [
            "strategy",
            "formula",
            "variant_kind",
            "n_donor",
            "horizon",
        ],
        dropna=False,
    ):
        (
            strategy,
            formula,
            variant_kind,
            n_donor,
            horizon,
        ) = keys

        rows.append(
            {
                "strategy": strategy,
                "formula": formula,
                "variant_kind": variant_kind,
                "n_donor": n_donor,
                "horizon": int(horizon),
                "target_count": int(
                    group["target_function"].nunique()
                ),
                "transfer_applied": float(
                    group["transfer_applied"].mean()
                ),
                "latency_gain_percent_vs_no_transfer": float(
                    group[
                        "latency_gain_percent_vs_no_transfer"
                    ].mean()
                ),
                "optimal_arm_rate": float(
                    group["optimal_arm_rate"].mean()
                ),
                "wrong_selections": float(
                    group["wrong_selections"].mean()
                ),
                "cumulative_reward_pseudo_regret": float(
                    group[
                        "cumulative_reward_pseudo_regret"
                    ].mean()
                ),
                "first_arm_optimal": float(
                    group["first_arm_optimal"].mean()
                ),
                "convergence_probability": float(
                    group["convergence_probability"].mean()
                ),
                "censored_convergence_request": float(
                    group[
                        "censored_convergence_request"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "formula",
            "horizon",
            "variant_kind",
            "n_donor",
        ],
        na_position="first",
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
    B - A; target_function is the resampling unit.
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
    rows = []

    for metric in metrics:
        delta = np.asarray(
            [
                float(b.loc[t, metric])
                - float(a.loc[t, metric])
                for t in targets
            ],
            dtype=float,
        )

        boot = np.empty(replicates, dtype=float)

        for i in range(replicates):
            idx = rng.integers(
                0,
                len(delta),
                size=len(delta),
            )
            boot[i] = float(np.mean(delta[idx]))

        rows.append(
            {
                "comparison": (
                    f"{strategy_b}_minus_{strategy_a}"
                ),
                "horizon": int(horizon),
                "metric": metric,
                "point_delta": float(np.mean(delta)),
                "ci95_low": float(
                    np.quantile(boot, 0.025)
                ),
                "ci95_high": float(
                    np.quantile(boot, 0.975)
                ),
                "target_count": int(len(delta)),
                "bootstrap_replicates": int(replicates),
            }
        )

    return rows


def effect_size(
    *,
    per_target: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    horizon: int,
) -> dict[str, Any]:
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

    a_lat = np.asarray(
        [
            float(a.loc[t, "cumulative_latency_ms"])
            for t in targets
        ]
    )
    b_lat = np.asarray(
        [
            float(b.loc[t, "cumulative_latency_ms"])
            for t in targets
        ]
    )

    reduction = 100.0 * (a_lat - b_lat) / a_lat

    wins = int(np.sum(b_lat < a_lat))
    ties = int(
        np.sum(
            np.isclose(
                a_lat,
                b_lat,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    )

    return {
        "comparison": f"{strategy_b}_minus_{strategy_a}",
        "horizon": int(horizon),
        "target_count": int(len(targets)),
        "mean_direct_latency_reduction_percent": float(
            np.mean(reduction)
        ),
        "median_direct_latency_reduction_percent": float(
            np.median(reduction)
        ),
        "target_latency_win_rate": float(
            wins / len(targets)
        ),
        "target_latency_tie_rate": float(
            ties / len(targets)
        ),
        "latency_gain_delta_pp": float(
            np.mean(
                [
                    float(
                        b.loc[
                            t,
                            "latency_gain_percent_vs_no_transfer",
                        ]
                    )
                    - float(
                        a.loc[
                            t,
                            "latency_gain_percent_vs_no_transfer",
                        ]
                    )
                    for t in targets
                ]
            )
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

    n_values = parse_int_list(args.n_donor_values)
    horizons = parse_int_list(args.horizons)

    max_n = max(n_values)
    max_h = max(horizons)

    raw_x86_path = args.raw_x86.resolve()
    raw_arm64_path = args.raw_arm64.resolve()
    donor_detail_path = args.donor_detail.resolve()

    x86 = load_eligible_durations(raw_x86_path)
    arm64 = load_eligible_durations(raw_arm64_path)
    performance = build_performance(x86, arm64)

    donor_maps, cluster_seeds, targets = (
        load_frozen_donor_map(donor_detail_path)
    )

    donor_names = sorted(
        {
            donor
            for seed in cluster_seeds
            for donor in donor_maps[seed].values()
            if donor is not None
        }
    )

    donor_counts = []
    for donor in donor_names:
        for arm in ARMS:
            count = len(
                performance[donor]["durations"][arm]
            )
            donor_counts.append(count)
            if count < max_n:
                raise SystemExit(
                    f"donor={donor} arm={arm} "
                    f"has {count} samples; max requested "
                    f"n_donor={max_n}"
                )

    print(
        "N_DONOR PREFLIGHT "
        f"targets={len(targets)} "
        f"cluster_seeds={len(cluster_seeds)} "
        f"unique_donors={len(donor_names)} "
        f"donor_samples_min={min(donor_counts)} "
        f"donor_samples_max={max(donor_counts)}"
    )
    print(
        "N_DONOR VALUES "
        + ",".join(map(str, n_values))
    )
    print(
        "FROZEN DIFFERENCE "
        "w_R=0.05 w_E=64 c=0.2"
    )
    print(
        "FROZEN RATIO "
        "w_R=0.01 w_E=64 c=0.4"
    )

    rows: list[dict[str, Any]] = []
    evidence_diag_rows: list[dict[str, Any]] = []

    for target_idx, target in enumerate(
        targets,
        start=1,
    ):
        target_stats = performance[target]

        anchor_rewards, online = make_crn_samples(
            target=target,
            target_stats=target_stats,
            replicates=args.replicates,
            max_horizon=max_h,
            anchor_n=args.anchor_samples,
            base_seed=args.mc_seed,
        )

        baseline_runs = {
            h: []
            for h in horizons
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
            for h in horizons:
                baseline_runs[h].append(
                    prefix_metrics_baseline(trace, h)
                )

        baseline_by_h = {
            h: mean_metrics(v)
            for h, v in baseline_runs.items()
        }

        # Baseline repeated over cluster seeds to preserve a rectangular table.
        for seed in cluster_seeds:
            for h in horizons:
                base = baseline_by_h[h]
                rows.append(
                    {
                        "target_function": target,
                        "cluster_seed": int(seed),
                        "donor_function": "",
                        "strategy": "no_transfer",
                        "formula": "none",
                        "variant_kind": "baseline",
                        "n_donor": np.nan,
                        "transfer_applied": 0.0,
                        "w_r": 0.0,
                        "w_e_base": 0.0,
                        "w_e_effective": 0.0,
                        "c": args.baseline_c,
                        "horizon": int(h),
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

        unique_donors = sorted(
            {
                donor_maps[seed][target]
                for seed in cluster_seeds
                if donor_maps[seed][target] is not None
            }
        )

        # Cache donor evidence subsets once and share them across Difference/Ratio.
        subset_cache = {
            donor: donor_subset_reward_means(
                donor_stats=performance[donor],
                target=target,
                donor=donor,
                n_values=n_values,
                replicates=args.replicates,
                base_seed=args.mc_seed,
            )
            for donor in unique_donors
        }

        # Cache all strategy x donor results.
        strategy_cache: dict[
            tuple[str, str],
            dict[int, dict[str, float]],
        ] = {}

        for formula in ("difference", "ratio"):
            frozen = FROZEN[formula]

            # Standard tuned TL: full donor mean, frozen w_E.
            standard_name = f"{formula}_standard"

            for donor in unique_donors:
                donor_stats = performance[donor]
                runs = {h: [] for h in horizons}

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
                        config=frozen,
                        convergence_run=args.convergence_run,
                    )
                    for h in horizons:
                        runs[h].append(
                            prefix_metrics_tl(trace, h)
                        )

                strategy_cache[
                    (standard_name, donor)
                ] = {
                    h: mean_metrics(v)
                    for h, v in runs.items()
                }

                # n_donor sensitivity variants.
                for n in n_values:
                    strategy_name = (
                        f"{formula}_ndonor_n{n}"
                    )
                    nd_config = Config(
                        config_id=strategy_name,
                        regime="n_donor_scaled_exploration",
                        w_r=frozen.w_r,
                        w_e=frozen.w_e * n,
                        c=frozen.c,
                    )

                    nd_runs = {
                        h: []
                        for h in horizons
                    }

                    full_effect = (
                        donor_stats["mean_reward"]["arm64"]
                        - donor_stats["mean_reward"]["x86"]
                    )

                    subset_direction_matches = []

                    for rep in range(args.replicates):
                        subset_stats = subset_donor_stats(
                            subset_cache[donor],
                            n_donor=n,
                            rep=rep,
                        )

                        subset_effect = (
                            subset_stats["mean_reward"]["arm64"]
                            - subset_stats["mean_reward"]["x86"]
                        )

                        subset_direction_matches.append(
                            float(
                                np.sign(subset_effect)
                                == np.sign(full_effect)
                            )
                        )

                        prior, _ = build_prior(
                            mode=formula,
                            anchor_mean_reward_x86=float(
                                anchor_rewards[rep]
                            ),
                            donor_stats=subset_stats,
                        )

                        trace = simulate_tl_trace(
                            target_stats=target_stats,
                            sequences={
                                "x86": online["x86"][rep],
                                "arm64": online["arm64"][rep],
                            },
                            prior_mean_reward=prior,
                            config=nd_config,
                            convergence_run=args.convergence_run,
                        )

                        for h in horizons:
                            nd_runs[h].append(
                                prefix_metrics_tl(trace, h)
                            )

                    strategy_cache[
                        (strategy_name, donor)
                    ] = {
                        h: mean_metrics(v)
                        for h, v in nd_runs.items()
                    }

                    evidence_diag_rows.append(
                        {
                            "target_function": target,
                            "donor_function": donor,
                            "formula": formula,
                            "n_donor": int(n),
                            "w_e_base": frozen.w_e,
                            "w_e_effective": (
                                frozen.w_e * n
                            ),
                            "full_donor_effect_reward": float(
                                full_effect
                            ),
                            "subset_direction_match_rate": float(
                                np.mean(
                                    subset_direction_matches
                                )
                            ),
                        }
                    )

        # Materialize seed-specific donor mapping rows.
        for seed in cluster_seeds:
            donor = donor_maps[seed][target]

            for formula in ("difference", "ratio"):
                frozen = FROZEN[formula]

                strategy_specs = [
                    (
                        f"{formula}_standard",
                        "standard_tuned",
                        np.nan,
                        frozen.w_e,
                    )
                ] + [
                    (
                        f"{formula}_ndonor_n{n}",
                        "n_donor",
                        float(n),
                        frozen.w_e * n,
                    )
                    for n in n_values
                ]

                for (
                    strategy,
                    variant_kind,
                    n_value,
                    effective_we,
                ) in strategy_specs:
                    for h in horizons:
                        base = baseline_by_h[h]

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
                            metrics = strategy_cache[
                                (strategy, donor)
                            ][h]
                            comparisons = compare_to_baseline(
                                metrics,
                                base,
                            )
                            applied = 1.0
                            donor_name = donor

                        rows.append(
                            {
                                "target_function": target,
                                "cluster_seed": int(seed),
                                "donor_function": donor_name,
                                "strategy": strategy,
                                "formula": formula,
                                "variant_kind": variant_kind,
                                "n_donor": n_value,
                                "transfer_applied": applied,
                                "w_r": frozen.w_r,
                                "w_e_base": frozen.w_e,
                                "w_e_effective": effective_we,
                                "c": frozen.c,
                                "horizon": int(h),
                                **metrics,
                                **comparisons,
                            }
                        )

        if (
            target_idx % 5 == 0
            or target_idx == len(targets)
        ):
            print(
                f"target_progress={target_idx}/{len(targets)}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed = pd.DataFrame(rows)
    per_seed_path = (
        args.output_dir
        / "tl-ndonor-per-target-seed.csv"
    )
    per_seed.to_csv(per_seed_path, index=False)

    per_target = aggregate_target_seed_to_target(
        per_seed
    )
    per_target_path = (
        args.output_dir
        / "tl-ndonor-per-target.csv"
    )
    per_target.to_csv(per_target_path, index=False)

    summary = build_summary(per_target)
    summary_path = (
        args.output_dir
        / "tl-ndonor-summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    evidence_diag = pd.DataFrame(
        evidence_diag_rows
    )
    evidence_diag_path = (
        args.output_dir
        / "tl-ndonor-evidence-diagnostics.csv"
    )
    evidence_diag.to_csv(
        evidence_diag_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Paired tests:
    # every n_donor vs the same formula's standard tuned policy.
    # ------------------------------------------------------------
    paired_rows = []

    for formula in ("difference", "ratio"):
        standard = f"{formula}_standard"

        for n in n_values:
            nd = f"{formula}_ndonor_n{n}"

            for h in horizons:
                paired_rows.extend(
                    paired_bootstrap(
                        per_target=per_target,
                        strategy_a=standard,
                        strategy_b=nd,
                        horizon=h,
                        replicates=args.bootstrap_replicates,
                        seed=stable_seed(
                            args.bootstrap_seed,
                            formula,
                            n,
                            h,
                        ),
                    )
                )

    # Direct comparison between actual-evidence n=12 variants.
    if 12 in n_values:
        for h in horizons:
            paired_rows.extend(
                paired_bootstrap(
                    per_target=per_target,
                    strategy_a="difference_ndonor_n12",
                    strategy_b="ratio_ndonor_n12",
                    horizon=h,
                    replicates=args.bootstrap_replicates,
                    seed=stable_seed(
                        args.bootstrap_seed,
                        "ratio-vs-difference-n12",
                        h,
                    ),
                )
            )

    paired = pd.DataFrame(paired_rows)
    paired_path = (
        args.output_dir
        / "tl-ndonor-paired-bootstrap.csv"
    )
    paired.to_csv(paired_path, index=False)

    # Primary effect sizes: n=12 vs standard.
    primary_effects = []
    if 12 in n_values:
        for formula in ("difference", "ratio"):
            for h in horizons:
                primary_effects.append(
                    effect_size(
                        per_target=per_target,
                        strategy_a=f"{formula}_standard",
                        strategy_b=f"{formula}_ndonor_n12",
                        horizon=h,
                    )
                )

    primary_effects_df = pd.DataFrame(
        primary_effects
    )
    effects_path = (
        args.output_dir
        / "tl-ndonor-primary-effect-sizes.csv"
    )
    primary_effects_df.to_csv(
        effects_path,
        index=False,
    )

    # Compact curve for presentation: standard + all n at each H.
    curve = summary[
        summary["strategy"] != "no_transfer"
    ][
        [
            "formula",
            "variant_kind",
            "n_donor",
            "horizon",
            "latency_gain_percent_vs_no_transfer",
            "optimal_arm_rate",
            "wrong_selections",
            "cumulative_reward_pseudo_regret",
            "convergence_probability",
        ]
    ].copy()
    curve_path = (
        args.output_dir
        / "tl-ndonor-curve.csv"
    )
    curve.to_csv(curve_path, index=False)

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    report = [
        "# UCB1 Transfer Learning — n_donor study",
        "",
        "No clustering, donor-selection or hyperparameter tuning is performed here.",
        "",
        "## Frozen configurations",
        "",
        "- Difference: w_R=0.05, w_E=64, c=0.2",
        "- Ratio: w_R=0.01, w_E=64, c=0.4",
        "",
        "## n_donor semantics",
        "",
        (
            "n_donor is the number of real historical donor warm observations "
            "used PER ARCHITECTURE. The donor effect is estimated from those "
            "samples and the exploration pseudo-count becomes w_E*n_donor."
        ),
        "",
        "Nested donor subsets are sampled without replacement from the frozen 12 warm observations.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]

    if 12 in n_values:
        primary_names = {
            "difference_ndonor_n12_minus_difference_standard",
            "ratio_ndonor_n12_minus_ratio_standard",
        }

        primary_paired = paired[
            paired["comparison"].isin(primary_names)
        ].copy()

        report += [
            "## Primary n_donor=12 vs standard tuned TL",
            "",
            primary_effects_df.to_markdown(
                index=False,
                floatfmt=".4f",
            ),
            "",
            "### Paired 95% bootstrap",
            "",
            primary_paired.to_markdown(
                index=False,
                floatfmt=".4f",
            ),
            "",
        ]

    report += [
        "## Interpretation rule",
        "",
        (
            "The predeclared primary question is whether n_donor=12 improves "
            "the frozen standard tuned policy. The n={1,2,4,6,8,10,12} curve "
            "is sensitivity analysis, not a new hyperparameter search."
        ),
        "",
    ]

    report_path = (
        args.output_dir
        / "tl-ndonor-report.md"
    )
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    manifest = {
        "experiment": "UCB1 TL n_donor study",
        "is_hyperparameter_tuning": False,
        "frozen_upstream": {
            "donor_variant": DONOR_VARIANT,
            "difference": {
                "w_R": 0.05,
                "w_E": 64.0,
                "c": 0.20,
            },
            "ratio": {
                "w_R": 0.01,
                "w_E": 64.0,
                "c": 0.40,
            },
        },
        "n_donor": {
            "values": n_values,
            "unit": "real warm observations per architecture",
            "sampling": "without replacement",
            "nested_subsets": True,
            "effective_exploration_weight": "w_E * n_donor",
            "primary_value": (
                12 if 12 in n_values else None
            ),
            "primary_reason": (
                "n=12 uses the complete frozen donor evidence, "
                "so its donor mean equals the standard policy donor mean"
            ),
        },
        "simulation": {
            "horizons": horizons,
            "replicates": int(args.replicates),
            "mc_seed": int(args.mc_seed),
            "anchor_samples": int(args.anchor_samples),
            "anchor_rng_separate": True,
            "donor_evidence_rng_separate": True,
            "online_crn_shared": True,
            "cluster_seeds": [
                int(x)
                for x in cluster_seeds
            ],
        },
        "bootstrap": {
            "unit": "target_function",
            "replicates": int(
                args.bootstrap_replicates
            ),
            "seed": int(args.bootstrap_seed),
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
        / "tl-ndonor-manifest.json"
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
    print("N_DONOR SUMMARY")
    print(
        summary[
            [
                "strategy",
                "horizon",
                "latency_gain_percent_vs_no_transfer",
                "optimal_arm_rate",
                "wrong_selections",
                "cumulative_reward_pseudo_regret",
                "convergence_probability",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    if 12 in n_values:
        print()
        print("PRIMARY N_DONOR=12 EFFECT SIZES")
        print(
            primary_effects_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print()
        print("PRIMARY N_DONOR=12 PAIRED BOOTSTRAP")
        primary_names = {
            "difference_ndonor_n12_minus_difference_standard",
            "ratio_ndonor_n12_minus_ratio_standard",
        }
        primary_paired = paired[
            paired["comparison"].isin(primary_names)
        ]
        print(
            primary_paired.sort_values(
                ["comparison", "horizon", "metric"]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print()
    print(f"per_target_seed={per_seed_path}")
    print(f"per_target={per_target_path}")
    print(f"summary={summary_path}")
    print(f"curve={curve_path}")
    print(f"paired_bootstrap={paired_path}")
    print(f"primary_effect_sizes={effects_path}")
    print(f"evidence_diagnostics={evidence_diag_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen UCB1 TL n_donor experiment using real donor "
            "subsamples and w_E*n_donor exploration pseudo-counts."
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
        "--n-donor-values",
        default="1,2,4,6,8,10,12",
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
        default=2026100717,
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
        default=2026100719,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
