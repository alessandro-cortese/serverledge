#!/usr/bin/env python3
"""
Serverledge: additive-vs-ratio Transfer Learning experiment.

Purpose
-------
Compare ONLY the construction of the reference-anchored prior, keeping the
already-frozen clustering/donor pipeline and all UCB1 parameters unchanged.

Frozen donor pipeline
---------------------
variant : weighted_vote_p2
ranking : nearest Manhattan donor inside the predicted directional class
input   : already-produced directional-vote-refinement artifact

Frozen TL configuration
-----------------------
policy                 : UCB1Decoupled
reward                  : r = -ln(duration_ms)
reference architecture  : x86
w_R                     : 0.25
w_E                     : 1.00
c                       : 0.80
convergence run         : 5
horizons                : 5,10,20,50

Compared strategies
-------------------
1. no_transfer

2. difference_tl  [CURRENT FORMULA]
       Delta_D = mu_D,ARM - mu_D,x86

       mu_prior_T,x86 = mu_T,x86
       mu_prior_T,ARM = mu_T,x86 + Delta_D

3. ratio_tl       [PROFESSOR VARIANT]
       R_D = mu_D,ARM / mu_D,x86

       mu_prior_T,x86 = mu_T,x86
       mu_prior_T,ARM = mu_T,x86 * R_D

where every mu is in reward space:
       mu = E[-ln(duration_ms)]

Experimental controls
---------------------
- The donor mapping is READ from the frozen weighted_vote_p2 LOFO artifact.
  This script does NOT refit clustering and does NOT reselect donors.
- If the donor rule abstains, BOTH TL formulas fall back to the exact
  no-transfer trajectory. The target is never dropped.
- Common Random Numbers (CRN): difference_tl, ratio_tl and no_transfer reuse
  the exact same online x86/ARM bootstrap streams.
- The target x86 anchor uses a DISTINCT RNG stream from the online replay.
- The same target-x86 anchor realization is reused by difference_tl and
  ratio_tl in every replicate.
- Donor x86/ARM means are computed from the frozen eligible warm samples.
- Target ARM data are never used to CONSTRUCT the prior; they are used only
  for online replay and post-hoc evaluation.
- No parameter tuning occurs here.

Primary question
----------------
With identical donors, anchors, online requests and UCB1 parameters, does:

       mu_T,x86 * (mu_D,ARM / mu_D,x86)

outperform:

       mu_T,x86 + (mu_D,ARM - mu_D,x86)

?

Outputs
-------
tl-prior-formula-per-target-seed.csv
tl-prior-formula-per-target.csv
tl-prior-formula-summary.csv
tl-prior-formula-paired-bootstrap.csv
tl-prior-formula-prior-diagnostics.csv
tl-prior-formula-report.md
tl-prior-formula-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.profiling.transfer_decoupled_ucb1 import (
    DecoupledUCB1Policy,
)
from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    UCB1Policy,
)


DONOR_VARIANT = "weighted_vote_p2"

STRATEGY_ORDER = [
    "no_transfer",
    "difference_tl",
    "ratio_tl",
]

FORMULA_MODES = {
    "difference_tl": "difference",
    "ratio_tl": "ratio",
}

EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def parse_int_list(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("empty integer list")
    return values


def load_eligible_durations(path: Path) -> dict[str, np.ndarray]:
    """
    Keep successful warm samples explicitly eligible for performance analysis.
    """
    grouped: dict[str, list[float]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

            if row.get("warm_start") is not True:
                continue

            if row.get("execution_succeeded") is not True:
                continue

            eligibility = row.get("eligibility") or {}
            if eligibility.get("performance_analysis") is not True:
                continue

            timing = row.get("timing") or {}
            duration = timing.get("duration_ms")

            if duration is None:
                continue

            duration = float(duration)

            if not math.isfinite(duration) or duration <= 0:
                raise SystemExit(
                    f"{path}:{line_no}: invalid duration_ms={duration}"
                )

            function_name = str(row.get("function_name") or "")

            if not function_name:
                raise SystemExit(
                    f"{path}:{line_no}: missing function_name"
                )

            grouped[function_name].append(duration)

    return {
        name: np.asarray(values, dtype=float)
        for name, values in grouped.items()
    }


def build_performance(
    x86: dict[str, np.ndarray],
    arm64: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    names = sorted(set(x86) & set(arm64))

    result: dict[str, dict[str, Any]] = {}

    for name in names:
        durations = {
            "x86": x86[name],
            "arm64": arm64[name],
        }

        mean_reward = {
            arm: float(np.mean(-np.log(durations[arm])))
            for arm in ARMS
        }

        best_arm = max(
            ARMS,
            key=lambda arm: (
                mean_reward[arm],
                -ARMS.index(arm),
            ),
        )

        result[name] = {
            "durations": durations,
            "mean_reward": mean_reward,
            "mean_duration_ms": {
                arm: float(np.mean(durations[arm]))
                for arm in ARMS
            },
            "median_duration_ms": {
                arm: float(np.median(durations[arm]))
                for arm in ARMS
            },
            "best_reward_arm": best_arm,
        }

    return result


def load_frozen_donor_map(
    path: Path,
) -> tuple[
    dict[int, dict[str, str | None]],
    list[int],
    list[str],
]:
    """
    Read only the already-frozen weighted_vote_p2 mapping.
    """
    df = pd.read_csv(path)

    required = {
        "variant",
        "seed",
        "target_function",
        "selection_status",
        "donor_function",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"Donor detail missing columns: {sorted(missing)}"
        )

    df = df[df["variant"] == DONOR_VARIANT].copy()

    if df.empty:
        raise SystemExit(
            f"No rows found for variant={DONOR_VARIANT!r}"
        )

    key_counts = (
        df.groupby(["seed", "target_function"])
        .size()
    )

    if (key_counts != 1).any():
        bad = key_counts[key_counts != 1].head(20)

        raise SystemExit(
            "Expected one weighted_vote_p2 donor row "
            "per seed/target:\n"
            + bad.to_string()
        )

    seeds = sorted(int(x) for x in df["seed"].unique())
    targets = sorted(str(x) for x in df["target_function"].unique())

    maps: dict[int, dict[str, str | None]] = {}

    for seed in seeds:
        seed_rows = df[df["seed"].astype(int) == seed]

        mapping: dict[str, str | None] = {}

        for target in targets:
            row = seed_rows[
                seed_rows["target_function"].astype(str) == target
            ]

            if len(row) != 1:
                raise SystemExit(
                    f"Missing/duplicate donor row: "
                    f"seed={seed} target={target}"
                )

            r = row.iloc[0]

            selected = str(r["selection_status"]) == "selected"

            donor = (
                str(r["donor_function"])
                if selected
                and pd.notna(r["donor_function"])
                and str(r["donor_function"])
                else None
            )

            mapping[target] = donor

        maps[seed] = mapping

    return maps, seeds, targets


def donor_formula_values(
    donor_stats: dict[str, Any],
) -> dict[str, float]:
    donor_x86 = float(donor_stats["mean_reward"]["x86"])
    donor_arm = float(donor_stats["mean_reward"]["arm64"])

    if abs(donor_x86) <= EPS:
        raise ValueError(
            "ratio formula undefined because donor mu_x86 is ~0"
        )

    donor_difference = donor_arm - donor_x86
    donor_ratio = donor_arm / donor_x86

    if not all(
        math.isfinite(value)
        for value in (
            donor_x86,
            donor_arm,
            donor_difference,
            donor_ratio,
        )
    ):
        raise ValueError(
            "non-finite donor reward/effect in prior construction"
        )

    return {
        "donor_mu_x86": donor_x86,
        "donor_mu_arm64": donor_arm,
        "donor_difference": donor_difference,
        "donor_ratio": donor_ratio,
        "donor_reward_sign_crossing": float(
            np.sign(donor_x86) != np.sign(donor_arm)
        ),
    }


def build_prior(
    *,
    mode: str,
    anchor_mean_reward_x86: float,
    donor_stats: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    values = donor_formula_values(donor_stats)

    anchor = float(anchor_mean_reward_x86)

    if mode == "difference":
        arm_prior = (
            anchor
            + values["donor_difference"]
        )

    elif mode == "ratio":
        arm_prior = (
            anchor
            * values["donor_ratio"]
        )

    else:
        raise ValueError(
            f"Unsupported prior mode: {mode}"
        )

    if not math.isfinite(arm_prior):
        raise ValueError(
            f"Non-finite arm prior for mode={mode}"
        )

    prior = {
        "x86": anchor,
        "arm64": float(arm_prior),
    }

    diagnostics = {
        **values,
        "anchor_mu_x86": anchor,
        "prior_mu_x86": anchor,
        "prior_mu_arm64": float(arm_prior),
        "prior_arch_effect": float(
            arm_prior - anchor
        ),
    }

    return prior, diagnostics


def make_crn_samples(
    *,
    target: str,
    target_stats: dict[str, Any],
    replicates: int,
    max_horizon: int,
    anchor_n: int,
    base_seed: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    The anchor and online replay use independent deterministic RNG streams.
    """
    anchor_rng = np.random.default_rng(
        stable_seed(
            base_seed,
            target,
            "anchor-x86",
        )
    )

    anchor_durations = anchor_rng.choice(
        target_stats["durations"]["x86"],
        size=(replicates, anchor_n),
        replace=True,
    )

    anchor_mean_reward = np.mean(
        -np.log(anchor_durations),
        axis=1,
    )

    online: dict[str, np.ndarray] = {}

    for arm in ARMS:
        rng = np.random.default_rng(
            stable_seed(
                base_seed,
                target,
                "online",
                arm,
            )
        )

        online[arm] = rng.choice(
            target_stats["durations"][arm],
            size=(replicates, max_horizon),
            replace=True,
        )

    return anchor_mean_reward.astype(float), online


def baseline_effective_mean(
    policy: UCB1Policy,
    arm: str,
) -> float | None:
    count = float(policy.effective_count(arm))

    if count <= 0:
        return None

    return float(
        (
            policy.reward_sums[arm]
            + policy.prior_reward_sums[arm]
        )
        / count
    )


def policy_effective_mean(
    policy: Any,
    arm: str,
) -> float | None:
    if isinstance(policy, DecoupledUCB1Policy):
        return float(policy.effective_mean(arm))

    return baseline_effective_mean(policy, arm)


def first_completed_true_run(
    values: list[bool],
    run_length: int,
) -> int | None:
    streak = 0

    for idx, value in enumerate(values, start=1):
        if value:
            streak += 1

            if streak >= run_length:
                return idx

        else:
            streak = 0

    return None


def simulate_trace(
    *,
    target_stats: dict[str, Any],
    sequences: dict[str, np.ndarray],
    c: float,
    convergence_run: int,
    prior_mean_reward: dict[str, float] | None,
    reward_weight: float,
    exploration_weight: float,
) -> dict[str, Any]:
    if prior_mean_reward is None:
        policy: Any = UCB1Policy(
            c=c,
            prior_weight=0.0,
            prior_mean_reward=None,
        )

    else:
        policy = DecoupledUCB1Policy(
            c=c,
            reward_prior_weight=reward_weight,
            exploration_prior_weight=exploration_weight,
            prior_mean_reward=prior_mean_reward,
        )

    positions = {arm: 0 for arm in ARMS}

    best_arm = str(target_stats["best_reward_arm"])
    best_mean_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    chosen_arms: list[str] = []
    durations: list[float] = []
    pseudo_regret_steps: list[float] = []
    exploitation_correct: list[bool] = []

    max_horizon = len(sequences[ARMS[0]])

    for _ in range(max_horizon):
        arm = str(policy.select_arm())

        position = positions[arm]

        duration = float(
            sequences[arm][position]
        )

        positions[arm] += 1

        policy.update(
            arm,
            duration,
        )

        chosen_arms.append(arm)
        durations.append(duration)

        pseudo_regret_steps.append(
            best_mean_reward
            - float(
                target_stats[
                    "mean_reward"
                ][arm]
            )
        )

        means = {
            candidate: policy_effective_mean(
                policy,
                candidate,
            )
            for candidate in ARMS
        }

        if all(
            means[candidate] is not None
            for candidate in ARMS
        ):
            exploitation_arm = max(
                ARMS,
                key=lambda candidate: (
                    float(means[candidate]),
                    -ARMS.index(candidate),
                ),
            )

            exploitation_correct.append(
                exploitation_arm == best_arm
            )

        else:
            exploitation_correct.append(False)

    return {
        "chosen_arms": chosen_arms,
        "durations": np.asarray(
            durations,
            dtype=float,
        ),
        "pseudo_regret_steps": np.asarray(
            pseudo_regret_steps,
            dtype=float,
        ),
        "exploitation_correct": exploitation_correct,
        "best_arm": best_arm,
        "convergence_run": convergence_run,
    }


def prefix_metrics(
    trace: dict[str, Any],
    horizon: int,
) -> dict[str, float]:
    chosen = trace["chosen_arms"][:horizon]
    durations = trace["durations"][:horizon]
    pseudo_regret_steps = trace[
        "pseudo_regret_steps"
    ][:horizon]
    exploitation_correct = trace[
        "exploitation_correct"
    ][:horizon]

    best_arm = str(trace["best_arm"])

    wrong = sum(
        1
        for arm in chosen
        if arm != best_arm
    )

    convergence = first_completed_true_run(
        exploitation_correct,
        int(trace["convergence_run"]),
    )

    if convergence is None:
        convergence_probability = 0.0
        censored_convergence_request = float(
            horizon + 1
        )

    else:
        convergence_probability = 1.0
        censored_convergence_request = float(
            convergence
        )

    optimal_count = len(chosen) - wrong

    return {
        "cumulative_latency_ms": float(
            np.sum(durations)
        ),
        "optimal_arm_rate": float(
            optimal_count / horizon
        ),
        "wrong_selections": float(wrong),
        "cumulative_reward_pseudo_regret": float(
            np.sum(pseudo_regret_steps)
        ),
        "first_arm_optimal": float(
            bool(chosen)
            and chosen[0] == best_arm
        ),
        "convergence_probability": float(
            convergence_probability
        ),
        "censored_convergence_request": float(
            censored_convergence_request
        ),
    }


def aggregate_runs(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    df = pd.DataFrame(rows)

    return {
        column: float(df[column].mean())
        for column in df.columns
    }


def compare_to_baseline(
    strategy: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    baseline_latency = float(
        baseline["cumulative_latency_ms"]
    )

    strategy_latency = float(
        strategy["cumulative_latency_ms"]
    )

    latency_gain = (
        100.0
        * (
            baseline_latency
            - strategy_latency
        )
        / baseline_latency
        if baseline_latency > 0
        else np.nan
    )

    return {
        "latency_gain_percent_vs_no_transfer": float(
            latency_gain
        ),
        "optimal_arm_rate_gain": float(
            strategy["optimal_arm_rate"]
            - baseline["optimal_arm_rate"]
        ),
        "wrong_selections_saved": float(
            baseline["wrong_selections"]
            - strategy["wrong_selections"]
        ),
        "reward_pseudo_regret_saved": float(
            baseline[
                "cumulative_reward_pseudo_regret"
            ]
            - strategy[
                "cumulative_reward_pseudo_regret"
            ]
        ),
        "first_arm_optimal_gain": float(
            strategy["first_arm_optimal"]
            - baseline["first_arm_optimal"]
        ),
        "convergence_probability_gain": float(
            strategy[
                "convergence_probability"
            ]
            - baseline[
                "convergence_probability"
            ]
        ),
        "convergence_requests_saved": float(
            baseline[
                "censored_convergence_request"
            ]
            - strategy[
                "censored_convergence_request"
            ]
        ),
    }


def target_level_summary(
    per_target_seed: pd.DataFrame,
) -> pd.DataFrame:
    excluded = {
        "target_function",
        "strategy",
        "formula",
        "cluster_seed",
        "donor_function",
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
                "horizon",
            ],
            as_index=False,
        )[numeric_cols]
        .mean()
    )


def macro_summary(
    per_target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        strategy,
        formula,
        horizon,
    ), group in per_target.groupby(
        [
            "strategy",
            "formula",
            "horizon",
        ]
    ):
        row: dict[str, Any] = {
            "strategy": strategy,
            "formula": formula,
            "horizon": int(horizon),
            "target_count": int(
                group["target_function"].nunique()
            ),
        }

        for column in group.columns:
            if column in {
                "target_function",
                "strategy",
                "formula",
                "horizon",
            }:
                continue

            if pd.api.types.is_numeric_dtype(
                group[column]
            ):
                row[
                    f"{column}_macro_mean"
                ] = float(
                    group[column].mean()
                )

        rows.append(row)

    return pd.DataFrame(rows)


def paired_target_bootstrap(
    *,
    per_target: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    horizon: int,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Report B - A. Resampling unit is target_function.
    """
    a = (
        per_target[
            (
                per_target["strategy"]
                == strategy_a
            )
            & (
                per_target["horizon"]
                == horizon
            )
        ]
        .set_index("target_function")
    )

    b = (
        per_target[
            (
                per_target["strategy"]
                == strategy_b
            )
            & (
                per_target["horizon"]
                == horizon
            )
        ]
        .set_index("target_function")
    )

    targets = sorted(
        set(a.index)
        & set(b.index)
    )

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
                float(
                    b.loc[target, metric]
                )
                - float(
                    a.loc[target, metric]
                )
                for target in targets
            ],
            dtype=float,
        )

        boot_means = np.empty(
            replicates,
            dtype=float,
        )

        for i in range(replicates):
            idx = rng.integers(
                0,
                len(deltas),
                size=len(deltas),
            )

            boot_means[i] = float(
                np.mean(deltas[idx])
            )

        rows.append(
            {
                "comparison": (
                    f"{strategy_b}"
                    f"_minus_"
                    f"{strategy_a}"
                ),
                "horizon": int(horizon),
                "metric": metric,
                "point_delta": float(
                    np.mean(deltas)
                ),
                "ci95_low": float(
                    np.quantile(
                        boot_means,
                        0.025,
                    )
                ),
                "ci95_high": float(
                    np.quantile(
                        boot_means,
                        0.975,
                    )
                ),
                "target_count": int(
                    len(deltas)
                ),
                "bootstrap_replicates": int(
                    replicates
                ),
            }
        )

    return rows


def write_report(
    path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    c: float,
    reward_weight: float,
    exploration_weight: float,
    anchor_samples: int,
    mc_replicates: int,
    mc_seed: int,
) -> None:
    lines = [
        "# Additive vs ratio Transfer Learning",
        "",
        "The donor pipeline and UCB1 parameters are frozen.",
        "Only the prior-construction formula changes.",
        "",
        "## Frozen configuration",
        "",
        f"- donor variant: `{DONOR_VARIANT}`",
        f"- c = {c}",
        f"- w_R = {reward_weight}",
        f"- w_E = {exploration_weight}",
        "- reward = `-ln(duration_ms)`",
        f"- target x86 anchor bootstrap size = {anchor_samples}",
        f"- Monte Carlo replicates = {mc_replicates}",
        f"- Monte Carlo seed = {mc_seed}",
        "- CRN shared by all strategies",
        "- anchor RNG separated from online replay",
        "",
        "## Formulas",
        "",
        "Difference:",
        "`mu_prior_ARM = mu_target_x86 + (mu_donor_ARM - mu_donor_x86)`",
        "",
        "Ratio:",
        "`mu_prior_ARM = mu_target_x86 * (mu_donor_ARM / mu_donor_x86)`",
        "",
        "## Macro results",
        "",
    ]

    focus_cols = [
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

    existing = [
        column
        for column in focus_cols
        if column in summary.columns
    ]

    for _, row in summary[
        existing
    ].sort_values(
        ["horizon", "strategy"]
    ).iterrows():
        chunks = []

        for column in existing:
            value = row[column]

            if isinstance(
                value,
                (float, np.floating),
            ):
                chunks.append(
                    f"{column}={value:.4f}"
                )

            else:
                chunks.append(
                    f"{column}={value}"
                )

        lines.append(
            "- " + ", ".join(chunks)
        )

    lines += [
        "",
        "## Primary paired comparison",
        "",
        (
            "`ratio_tl - difference_tl` is the primary comparison. "
            "Positive latency-gain and optimal-arm deltas favor ratio; "
            "negative wrong-selection and pseudo-regret deltas favor ratio."
        ),
        "",
    ]

    primary = paired[
        paired["comparison"]
        == "ratio_tl_minus_difference_tl"
    ]

    for _, row in primary.sort_values(
        ["horizon", "metric"]
    ).iterrows():
        lines.append(
            f"- H={int(row['horizon'])}, "
            f"{row['metric']}: "
            f"delta={row['point_delta']:.4f}, "
            f"95% CI=[{row['ci95_low']:.4f}, "
            f"{row['ci95_high']:.4f}]"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        (
            "Do not tune w_R, w_E or c from this experiment. "
            "First compare the two formulas under the identical frozen "
            "configuration. Parameter tuning is a later, separate study."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> None:
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
    donor_detail_path = (
        args.donor_detail.resolve()
    )

    x86 = load_eligible_durations(
        raw_x86_path
    )
    arm64 = load_eligible_durations(
        raw_arm64_path
    )

    performance = build_performance(
        x86,
        arm64,
    )

    (
        donor_maps,
        cluster_seeds,
        targets,
    ) = load_frozen_donor_map(
        donor_detail_path
    )

    missing_targets = [
        target
        for target in targets
        if target not in performance
    ]

    if missing_targets:
        raise SystemExit(
            "Targets missing raw performance data: "
            + ", ".join(missing_targets)
        )

    all_donors = sorted(
        {
            donor
            for target_map in donor_maps.values()
            for donor in target_map.values()
            if donor
        }
    )

    missing_donors = [
        donor
        for donor in all_donors
        if donor not in performance
    ]

    if missing_donors:
        raise SystemExit(
            "Donors missing raw performance data: "
            + ", ".join(missing_donors)
        )

    sample_counts = []

    for function_name in sorted(
        set(targets)
        | set(all_donors)
    ):
        row = {
            "function_name": function_name,
            "x86_samples": int(
                len(
                    performance[
                        function_name
                    ]["durations"]["x86"]
                )
            ),
            "arm64_samples": int(
                len(
                    performance[
                        function_name
                    ]["durations"]["arm64"]
                )
            ),
        }

        sample_counts.append(row)

    insufficient = [
        row
        for row in sample_counts
        if row["x86_samples"] < 2
        or row["arm64_samples"] < 2
    ]

    if insufficient:
        raise SystemExit(
            "Insufficient empirical samples: "
            + json.dumps(
                insufficient[:20]
            )
        )

    # Preflight ratio diagnostics over every actually selected donor.
    selected_donor_formula_rows = []

    for seed in cluster_seeds:
        for target in targets:
            donor = donor_maps[seed][target]

            if donor is None:
                continue

            values = donor_formula_values(
                performance[donor]
            )

            selected_donor_formula_rows.append(
                {
                    "cluster_seed": int(seed),
                    "target_function": target,
                    "donor_function": donor,
                    **values,
                }
            )

    preflight = pd.DataFrame(
        selected_donor_formula_rows
    )

    sign_crossings = (
        int(
            preflight[
                "donor_reward_sign_crossing"
            ].sum()
        )
        if not preflight.empty
        else 0
    )

    donor_ratio_min = (
        float(
            preflight[
                "donor_ratio"
            ].min()
        )
        if not preflight.empty
        else np.nan
    )

    donor_ratio_max = (
        float(
            preflight[
                "donor_ratio"
            ].max()
        )
        if not preflight.empty
        else np.nan
    )

    print(
        "RATIO PREFLIGHT "
        f"selected_rows={len(preflight)} "
        f"reward_sign_crossings={sign_crossings} "
        f"ratio_min={donor_ratio_min:.6f} "
        f"ratio_max={donor_ratio_max:.6f}"
    )

    per_target_seed_rows: list[
        dict[str, Any]
    ] = []

    prior_diagnostic_rows: list[
        dict[str, Any]
    ] = []

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

        # Baseline is independent of cluster seed.
        for rep in range(args.replicates):
            sequences = {
                arm: online[arm][rep]
                for arm in ARMS
            }

            trace = simulate_trace(
                target_stats=target_stats,
                sequences=sequences,
                c=args.c,
                convergence_run=args.convergence_run,
                prior_mean_reward=None,
                reward_weight=args.reward_weight,
                exploration_weight=args.exploration_weight,
            )

            for horizon in horizons:
                baseline_runs_by_horizon[
                    horizon
                ].append(
                    prefix_metrics(
                        trace,
                        horizon,
                    )
                )

        baseline_agg = {
            horizon: aggregate_runs(rows)
            for horizon, rows in (
                baseline_runs_by_horizon.items()
            )
        }

        # Duplicate baseline across clustering seeds only to keep the paired
        # target/seed table rectangular.
        for cluster_seed in cluster_seeds:
            for horizon in horizons:
                base = baseline_agg[horizon]

                per_target_seed_rows.append(
                    {
                        "target_function": target,
                        "strategy": "no_transfer",
                        "formula": "none",
                        "cluster_seed": int(
                            cluster_seed
                        ),
                        "donor_function": "",
                        "transfer_applied": 0.0,
                        "horizon": int(horizon),
                        **base,
                        "latency_gain_percent_vs_no_transfer": 0.0,
                        "optimal_arm_rate_gain": 0.0,
                        "wrong_selections_saved": 0.0,
                        "reward_pseudo_regret_saved": 0.0,
                        "first_arm_optimal_gain": 0.0,
                        "convergence_probability_gain": 0.0,
                        "convergence_requests_saved": 0.0,
                    }
                )

        for cluster_seed in cluster_seeds:
            donor = donor_maps[
                cluster_seed
            ][target]

            # Abstention means identical no-transfer fallback for BOTH formulas.
            if donor is None:
                for strategy in (
                    "difference_tl",
                    "ratio_tl",
                ):
                    for horizon in horizons:
                        base = baseline_agg[
                            horizon
                        ]

                        per_target_seed_rows.append(
                            {
                                "target_function": target,
                                "strategy": strategy,
                                "formula": FORMULA_MODES[
                                    strategy
                                ],
                                "cluster_seed": int(
                                    cluster_seed
                                ),
                                "donor_function": "",
                                "transfer_applied": 0.0,
                                "horizon": int(
                                    horizon
                                ),
                                **base,
                                "latency_gain_percent_vs_no_transfer": 0.0,
                                "optimal_arm_rate_gain": 0.0,
                                "wrong_selections_saved": 0.0,
                                "reward_pseudo_regret_saved": 0.0,
                                "first_arm_optimal_gain": 0.0,
                                "convergence_probability_gain": 0.0,
                                "convergence_requests_saved": 0.0,
                            }
                        )

                continue

            donor_stats = performance[donor]
            donor_values = donor_formula_values(
                donor_stats
            )

            for strategy in (
                "difference_tl",
                "ratio_tl",
            ):
                mode = FORMULA_MODES[
                    strategy
                ]

                strategy_runs_by_horizon = {
                    horizon: []
                    for horizon in horizons
                }

                prior_arm_values = []

                for rep in range(
                    args.replicates
                ):
                    sequences = {
                        arm: online[arm][rep]
                        for arm in ARMS
                    }

                    (
                        prior,
                        prior_diag,
                    ) = build_prior(
                        mode=mode,
                        anchor_mean_reward_x86=float(
                            anchor_rewards[rep]
                        ),
                        donor_stats=donor_stats,
                    )

                    prior_arm_values.append(
                        prior_diag[
                            "prior_mu_arm64"
                        ]
                    )

                    trace = simulate_trace(
                        target_stats=target_stats,
                        sequences=sequences,
                        c=args.c,
                        convergence_run=args.convergence_run,
                        prior_mean_reward=prior,
                        reward_weight=args.reward_weight,
                        exploration_weight=args.exploration_weight,
                    )

                    for horizon in horizons:
                        strategy_runs_by_horizon[
                            horizon
                        ].append(
                            prefix_metrics(
                                trace,
                                horizon,
                            )
                        )

                prior_diagnostic_rows.append(
                    {
                        "target_function": target,
                        "cluster_seed": int(
                            cluster_seed
                        ),
                        "donor_function": donor,
                        "strategy": strategy,
                        "formula": mode,
                        **donor_values,
                        "anchor_mu_x86_mean": float(
                            np.mean(
                                anchor_rewards
                            )
                        ),
                        "anchor_mu_x86_std": float(
                            np.std(
                                anchor_rewards,
                                ddof=1,
                            )
                        ),
                        "prior_mu_arm64_mean": float(
                            np.mean(
                                prior_arm_values
                            )
                        ),
                        "prior_mu_arm64_std": float(
                            np.std(
                                prior_arm_values,
                                ddof=1,
                            )
                        ),
                        "prior_arch_effect_mean": float(
                            np.mean(
                                np.asarray(
                                    prior_arm_values
                                )
                                - anchor_rewards
                            )
                        ),
                    }
                )

                for horizon in horizons:
                    strategy_agg = aggregate_runs(
                        strategy_runs_by_horizon[
                            horizon
                        ]
                    )

                    comparisons = (
                        compare_to_baseline(
                            strategy_agg,
                            baseline_agg[
                                horizon
                            ],
                        )
                    )

                    per_target_seed_rows.append(
                        {
                            "target_function": target,
                            "strategy": strategy,
                            "formula": mode,
                            "cluster_seed": int(
                                cluster_seed
                            ),
                            "donor_function": donor,
                            "transfer_applied": 1.0,
                            "horizon": int(horizon),
                            **strategy_agg,
                            **comparisons,
                        }
                    )

        if (
            target_idx % 5 == 0
            or target_idx == total_targets
        ):
            print(
                f"target_progress="
                f"{target_idx}/"
                f"{total_targets}"
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
        / "tl-prior-formula-per-target-seed.csv"
    )

    per_target_seed.to_csv(
        per_target_seed_path,
        index=False,
    )

    prior_diagnostics = pd.DataFrame(
        prior_diagnostic_rows
    )

    prior_diagnostics_path = (
        args.output_dir
        / "tl-prior-formula-prior-diagnostics.csv"
    )

    prior_diagnostics.to_csv(
        prior_diagnostics_path,
        index=False,
    )

    per_target = target_level_summary(
        per_target_seed
    )

    per_target_path = (
        args.output_dir
        / "tl-prior-formula-per-target.csv"
    )

    per_target.to_csv(
        per_target_path,
        index=False,
    )

    summary = macro_summary(
        per_target
    )

    strategy_order = {
        strategy: index
        for index, strategy in enumerate(
            STRATEGY_ORDER
        )
    }

    summary["_order"] = summary[
        "strategy"
    ].map(strategy_order)

    summary = (
        summary.sort_values(
            ["horizon", "_order"]
        )
        .drop(columns=["_order"])
    )

    summary_path = (
        args.output_dir
        / "tl-prior-formula-summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    paired_rows: list[
        dict[str, Any]
    ] = []

    comparisons = [
        (
            "difference_tl",
            "ratio_tl",
        ),
        (
            "no_transfer",
            "difference_tl",
        ),
        (
            "no_transfer",
            "ratio_tl",
        ),
    ]

    for strategy_a, strategy_b in comparisons:
        for horizon in horizons:
            paired_rows.extend(
                paired_target_bootstrap(
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

    paired = pd.DataFrame(
        paired_rows
    )

    paired_path = (
        args.output_dir
        / "tl-prior-formula-paired-bootstrap.csv"
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "tl-prior-formula-report.md"
    )

    write_report(
        report_path,
        summary,
        paired,
        c=args.c,
        reward_weight=args.reward_weight,
        exploration_weight=args.exploration_weight,
        anchor_samples=args.anchor_samples,
        mc_replicates=args.replicates,
        mc_seed=args.mc_seed,
    )

    manifest = {
        "experiment": (
            "difference-vs-ratio prior construction"
        ),
        "clustering_status": (
            "frozen; donor mapping read from artifact"
        ),
        "donor_variant": DONOR_VARIANT,
        "strategies": STRATEGY_ORDER,
        "formulas": {
            "difference": (
                "mu_T_ARM_prior = mu_T_x86 + "
                "(mu_D_ARM - mu_D_x86)"
            ),
            "ratio": (
                "mu_T_ARM_prior = mu_T_x86 * "
                "(mu_D_ARM / mu_D_x86)"
            ),
        },
        "reward_space": (
            "mu = mean[-ln(duration_ms)]"
        ),
        "transfer": {
            "policy": "UCB1Decoupled",
            "reward_weight_wR": (
                args.reward_weight
            ),
            "exploration_weight_wE": (
                args.exploration_weight
            ),
            "c": args.c,
            "convergence_run": (
                args.convergence_run
            ),
            "reference_arm": "x86",
        },
        "simulation": {
            "horizons": horizons,
            "replicates": args.replicates,
            "mc_seed": args.mc_seed,
            "anchor_samples": (
                args.anchor_samples
            ),
            "anchor_rng_separate": True,
            "common_random_numbers": True,
            "same_anchor_between_formulas": True,
            "abstention_fallback": (
                "exact no-transfer for both formulas"
            ),
            "cluster_seeds": cluster_seeds,
        },
        "ratio_preflight": {
            "selected_rows": int(
                len(preflight)
            ),
            "reward_sign_crossings": int(
                sign_crossings
            ),
            "donor_ratio_min": (
                donor_ratio_min
            ),
            "donor_ratio_max": (
                donor_ratio_max
            ),
            "zero_denominator_guard": (
                EPS
            ),
        },
        "bootstrap": {
            "unit": "target_function",
            "replicates": (
                args.bootstrap_replicates
            ),
            "seed": (
                args.bootstrap_seed
            ),
            "ci": "percentile 95%",
        },
        "sample_counts": sample_counts,
        "inputs": {
            "raw_x86": {
                "path": str(
                    raw_x86_path
                ),
                "sha256": sha256_file(
                    raw_x86_path
                ),
            },
            "raw_arm64": {
                "path": str(
                    raw_arm64_path
                ),
                "sha256": sha256_file(
                    raw_arm64_path
                ),
            },
            "donor_detail": {
                "path": str(
                    donor_detail_path
                ),
                "sha256": sha256_file(
                    donor_detail_path
                ),
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "tl-prior-formula-manifest.json"
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

    focus_columns = [
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

    print()
    print(
        "ADDITIVE VS RATIO TL SUMMARY"
    )

    print(
        summary[
            focus_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "RATIO MINUS DIFFERENCE"
    )

    primary = paired[
        paired["comparison"]
        == "ratio_tl_minus_difference_tl"
    ]

    print(
        primary.sort_values(
            ["horizon", "metric"]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "FORMULA SANITY"
    )

    first_arm = primary[
        primary["metric"]
        == "first_arm_optimal"
    ]

    if not first_arm.empty:
        print(
            "ratio_minus_difference_first_arm_delta="
            f"{first_arm['point_delta'].iloc[0]:.6f}"
        )

    print(
        "Expected: usually 0 if both formulas preserve the "
        "same donor direction and rewards do not cross zero."
    )

    print()
    print(
        f"per_target_seed="
        f"{per_target_seed_path}"
    )
    print(
        f"per_target="
        f"{per_target_path}"
    )
    print(
        f"summary="
        f"{summary_path}"
    )
    print(
        f"paired="
        f"{paired_path}"
    )
    print(
        f"prior_diagnostics="
        f"{prior_diagnostics_path}"
    )
    print(
        f"report="
        f"{report_path}"
    )
    print(
        f"manifest="
        f"{manifest_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare additive vs reward-ratio reference-anchored "
            "Serverledge UCB1Decoupled priors with all other choices frozen."
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
        default=500,
    )

    parser.add_argument(
        "--mc-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--anchor-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--c",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--reward-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--convergence-run",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    return parser


if __name__ == "__main__":
    main(
        build_parser().parse_args()
    )
