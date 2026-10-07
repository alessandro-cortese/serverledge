#!/usr/bin/env python3
"""
Targeted directional-vote refinement study for Serverledge.

Goal
----
Improve architecture-direction prediction (and therefore raw best-arm agreement)
WITHOUT changing the frozen clustering representation:

    PAPER5 + {static_token_count_mean, static_function_count}
    MinMax
    K-Means K=6
    alpha=1
    LOFO
    tau=2.5%

The professor-required cluster-majority rule remains the baseline.

Predeclared refinements:
1. cluster_majority                      -- professor baseline
2. cluster_majority_gate_0p10
3. cluster_majority_gate_0p20
4. cluster_majority_gate_0p30            -- abstain on weak cluster majority
5. local_k3_majority
6. local_k5_majority
7. local_k7_majority                     -- majority among nearest directional
                                           members inside the assigned cluster
8. weighted_vote_p1
9. weighted_vote_p2                      -- inverse-Manhattan weighted vote

After a direction is predicted, the donor is still the nearest Manhattan donor
inside the assigned cluster and predicted directional class.

Important
---------
- architecture-independent functions remain in the cluster but do not vote and
  cannot be donors.
- target ground truth is never used before selection.
- this is a targeted sensitivity/ablation, not an unrestricted hyperparameter
  search. Do not pick a setting merely because it crosses an arbitrary 70%.
- report agreement together with coverage and regret.

Random references
-----------------
Raw best-arm agreement is a BINARY x86-vs-ARM outcome:
- uniform random direction baseline = 50%, NOT 20%;
- empirical-prior random baseline = p_x86^2 + p_arm^2;
- always-majority baseline = max(p_x86, p_arm).

Outputs
-------
directional-vote-refinement-per-target.csv
directional-vote-refinement-per-seed.csv
directional-vote-refinement-summary.csv
directional-vote-refinement-report.md
directional-vote-refinement-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC = [
    "static_token_count_mean",
    "static_function_count",
]

TAU = 2.5
KMEANS_K = 6
N_INIT = 50
EPS = 1e-12
DIRECTIONAL = {"x86-preferred", "arm-preferred"}

VARIANTS = [
    ("cluster_majority", {"kind": "cluster_majority"}),
    (
        "cluster_majority_gate_0p10",
        {"kind": "cluster_majority_gate", "gamma": 0.10},
    ),
    (
        "cluster_majority_gate_0p20",
        {"kind": "cluster_majority_gate", "gamma": 0.20},
    ),
    (
        "cluster_majority_gate_0p30",
        {"kind": "cluster_majority_gate", "gamma": 0.30},
    ),
    ("local_k3_majority", {"kind": "local_k", "k": 3}),
    ("local_k5_majority", {"kind": "local_k", "k": 5}),
    ("local_k7_majority", {"kind": "local_k", "k": 7}),
    ("weighted_vote_p1", {"kind": "weighted", "power": 1.0}),
    ("weighted_vote_p2", {"kind": "weighted", "power": 2.0}),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def source_available(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def fit_block(
    train: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    a = scaler.fit_transform(train[columns].astype(float))
    b = scaler.transform(target[columns].astype(float))
    return a, b


def representation(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    dyn_train, dyn_target = fit_block(train, target, PAPER5)
    stat_train, stat_target = fit_block(train, target, STATIC)

    dyn_train /= math.sqrt(len(PAPER5))
    dyn_target /= math.sqrt(len(PAPER5))
    stat_train /= math.sqrt(len(STATIC))
    stat_target /= math.sqrt(len(STATIC))

    return (
        np.column_stack([dyn_train, stat_train]),
        np.column_stack([dyn_target, stat_target]),
    )


def raw_best(x86_ms: float, arm_ms: float) -> str:
    return "x86-preferred" if x86_ms <= arm_ms else "arm-preferred"


def majority_from_indices(
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[str | None, int, int]:
    if len(indices) == 0:
        return None, 0, 0

    selected = labels[indices]
    nx = int(np.sum(selected == "x86-preferred"))
    na = int(np.sum(selected == "arm-preferred"))

    if nx == 0 and na == 0:
        return None, nx, na
    if nx == na:
        return None, nx, na

    return (
        "x86-preferred" if nx > na else "arm-preferred",
        nx,
        na,
    )


def decide_direction(
    config: dict,
    labels: np.ndarray,
    same_cluster: np.ndarray,
    distances: np.ndarray,
) -> tuple[str | None, float, int]:
    directional = same_cluster[
        np.isin(labels[same_cluster], list(DIRECTIONAL))
    ]

    if len(directional) == 0:
        return None, 0.0, 0

    kind = config["kind"]

    if kind in {"cluster_majority", "cluster_majority_gate"}:
        pred, nx, na = majority_from_indices(labels, directional)
        total = nx + na
        margin = abs(nx - na) / total if total else 0.0

        if pred is None:
            return None, margin, total

        if kind == "cluster_majority_gate":
            if margin < float(config["gamma"]):
                return None, margin, total

        return pred, margin, total

    if kind == "local_k":
        order = directional[
            np.argsort(
                distances[directional],
                kind="stable",
            )
        ]
        local = order[: min(int(config["k"]), len(order))]
        pred, nx, na = majority_from_indices(labels, local)
        total = nx + na
        margin = abs(nx - na) / total if total else 0.0
        return pred, margin, total

    if kind == "weighted":
        power = float(config["power"])
        x86_score = 0.0
        arm_score = 0.0

        for idx in directional:
            weight = 1.0 / ((float(distances[idx]) + 1e-9) ** power)
            if labels[idx] == "x86-preferred":
                x86_score += weight
            else:
                arm_score += weight

        denom = x86_score + arm_score
        if denom <= 0:
            return None, 0.0, len(directional)

        margin = abs(x86_score - arm_score) / denom

        if abs(x86_score - arm_score) <= 1e-15:
            return None, margin, len(directional)

        return (
            "x86-preferred" if x86_score > arm_score else "arm-preferred",
            margin,
            len(directional),
        )

    raise ValueError(kind)


def nearest_donor(
    prediction: str,
    labels: np.ndarray,
    same_cluster: np.ndarray,
    distances: np.ndarray,
    names: np.ndarray,
) -> tuple[int, float]:
    candidates = same_cluster[
        labels[same_cluster] == prediction
    ]

    if len(candidates) == 0:
        raise RuntimeError("Predicted class has no donor candidate")

    order = np.lexsort(
        (
            names[candidates],
            distances[candidates],
        )
    )

    idx = int(candidates[int(order[0])])
    return idx, float(distances[idx])


def outcome(
    target_info: dict,
    donor_info: dict,
    predicted_direction: str,
) -> dict:
    tx = float(target_info["x86_duration_ms"])
    ta = float(target_info["arm_duration_ms"])
    dx = float(donor_info["x86_duration_ms"])
    da = float(donor_info["arm_duration_ms"])

    target_raw = raw_best(tx, ta)
    donor_raw = raw_best(dx, da)

    target_pref = str(target_info["architecture_preference"])
    target_directional = target_pref in DIRECTIONAL

    if predicted_direction == "x86-preferred":
        chosen = tx
        alternative = ta
    else:
        chosen = ta
        alternative = tx

    optimal = min(tx, ta)
    regret = (chosen - optimal) / optimal * 100.0

    target_ratio = ta / tx
    donor_ratio = da / dx

    return {
        "target_raw_best": target_raw,
        "donor_raw_best": donor_raw,
        "raw_best_arm_agreement": target_raw == donor_raw,
        "target_threshold_label": target_pref,
        "target_is_directional": target_directional,
        "directional_preference_agreement": (
            predicted_direction == target_pref
            if target_directional
            else np.nan
        ),
        "multiplicative_mismatch_factor": math.exp(
            abs(math.log(target_ratio) - math.log(donor_ratio))
        ),
        "realized_speedup_factor": alternative / chosen,
        "regret_percent": regret,
        "zero_regret": regret <= EPS,
    }


def geometric_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    x = x[x > 0]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan


def q90(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(x, 0.90)) if len(x) else np.nan


def prepare(args):
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())
    prefs = pd.read_csv(args.preferences.resolve())

    df = (
        profiles.merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    *STATIC,
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            languages[["function_name", "language"]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            prefs[
                [
                    "function_name",
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    keep = source_available(df["static_source_available"])

    for col in PAPER5 + STATIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        keep &= df[col].notna()

    df = (
        df[keep]
        .copy()
        .sort_values("function_name")
        .reset_index(drop=True)
    )

    if len(df) != 59:
        raise SystemExit(f"Expected 59 eligible functions, got {len(df)}")

    return df


def main(args):
    df = prepare(args)
    seeds = parse_seeds(args.seeds)

    raw_labels = np.asarray(
        [
            raw_best(
                float(r.x86_duration_ms),
                float(r.arm_duration_ms),
            )
            for r in df.itertuples()
        ]
    )

    p_x86 = float(np.mean(raw_labels == "x86-preferred"))
    p_arm = 1.0 - p_x86
    uniform_random = 0.5
    empirical_random = p_x86**2 + p_arm**2
    always_majority = max(p_x86, p_arm)

    rows = []
    total = len(seeds) * len(df)
    done = 0

    for seed in seeds:
        for target_idx in range(len(df)):
            target = df.iloc[[target_idx]].copy()
            train = (
                df.drop(index=target_idx)
                .copy()
                .reset_index(drop=True)
            )

            x_train, x_target = representation(train, target)

            model = KMeans(
                n_clusters=KMEANS_K,
                n_init=N_INIT,
                random_state=seed,
            )
            cluster_labels = model.fit_predict(x_train)
            target_cluster = int(model.predict(x_target)[0])

            same_cluster = np.flatnonzero(
                cluster_labels == target_cluster
            )

            names = train["function_name"].astype(str).to_numpy()
            pref_labels = (
                train["architecture_preference"]
                .astype(str)
                .to_numpy()
            )

            distances = np.sum(
                np.abs(x_train - x_target[0]),
                axis=1,
            )

            target_info = target.iloc[0].to_dict()

            for variant_name, variant_config in VARIANTS:
                pred, vote_margin, voters = decide_direction(
                    variant_config,
                    pref_labels,
                    same_cluster,
                    distances,
                )

                base = {
                    "variant": variant_name,
                    "seed": seed,
                    "target_function": str(
                        target.iloc[0]["function_name"]
                    ),
                    "target_language": str(
                        target.iloc[0]["language"]
                    ),
                    "target_cluster": target_cluster,
                    "same_cluster_size": len(same_cluster),
                    "vote_margin": vote_margin,
                    "directional_voters_used": voters,
                    "predicted_direction": pred or "",
                }

                if pred is None:
                    rows.append(
                        {
                            **base,
                            "selection_status": "abstain",
                            "donor_function": "",
                            "donor_distance": np.nan,
                            "raw_best_arm_agreement": np.nan,
                            "target_is_directional": (
                                target_info["architecture_preference"]
                                in DIRECTIONAL
                            ),
                            "directional_preference_agreement": np.nan,
                            "multiplicative_mismatch_factor": np.nan,
                            "realized_speedup_factor": np.nan,
                            "regret_percent": np.nan,
                            "zero_regret": np.nan,
                        }
                    )
                    continue

                donor_idx, donor_distance = nearest_donor(
                    pred,
                    pref_labels,
                    same_cluster,
                    distances,
                    names,
                )

                donor_info = train.iloc[donor_idx].to_dict()
                metrics = outcome(
                    target_info,
                    donor_info,
                    pred,
                )

                rows.append(
                    {
                        **base,
                        "selection_status": "selected",
                        "donor_function": str(
                            donor_info["function_name"]
                        ),
                        "donor_distance": donor_distance,
                        **metrics,
                    }
                )

            done += 1
            if done % 100 == 0 or done == total:
                print(f"progress={done}/{total}")

    detail = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = (
        args.output_dir
        / "directional-vote-refinement-per-target.csv"
    )
    detail.to_csv(detail_path, index=False)

    per_seed_rows = []

    for (variant, seed), group in detail.groupby(
        ["variant", "seed"]
    ):
        selected = group[
            group["selection_status"] == "selected"
        ].copy()

        directional = group[
            group["target_is_directional"] == True
        ].copy()

        selected_directional = selected[
            selected["target_is_directional"] == True
        ].copy()

        per_seed_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "selection_coverage": len(selected) / len(group),
                "directional_selection_coverage": (
                    len(selected_directional) / len(directional)
                    if len(directional)
                    else np.nan
                ),
                "raw_best_arm_agreement": (
                    selected["raw_best_arm_agreement"]
                    .astype(float)
                    .mean()
                    if len(selected)
                    else np.nan
                ),
                "directional_preference_agreement": (
                    selected_directional[
                        "directional_preference_agreement"
                    ]
                    .astype(float)
                    .mean()
                    if len(selected_directional)
                    else np.nan
                ),
                "directional_geomean_realized_speedup": (
                    geometric_mean(
                        selected_directional[
                            "realized_speedup_factor"
                        ]
                    )
                    if len(selected_directional)
                    else np.nan
                ),
                "directional_mean_regret_percent": (
                    selected_directional[
                        "regret_percent"
                    ].mean()
                    if len(selected_directional)
                    else np.nan
                ),
                "directional_p90_regret_percent": (
                    q90(
                        selected_directional[
                            "regret_percent"
                        ]
                    )
                    if len(selected_directional)
                    else np.nan
                ),
                "directional_geomean_mismatch": (
                    geometric_mean(
                        selected_directional[
                            "multiplicative_mismatch_factor"
                        ]
                    )
                    if len(selected_directional)
                    else np.nan
                ),
                "mean_vote_margin": (
                    selected["vote_margin"].mean()
                    if len(selected)
                    else np.nan
                ),
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)

    per_seed_path = (
        args.output_dir
        / "directional-vote-refinement-per-seed.csv"
    )
    per_seed.to_csv(per_seed_path, index=False)

    summary = (
        per_seed.groupby("variant", as_index=False)
        .agg(
            selection_coverage_mean=("selection_coverage", "mean"),
            selection_coverage_std=("selection_coverage", "std"),
            directional_selection_coverage_mean=(
                "directional_selection_coverage",
                "mean",
            ),
            raw_best_arm_agreement_mean=(
                "raw_best_arm_agreement",
                "mean",
            ),
            raw_best_arm_agreement_std=(
                "raw_best_arm_agreement",
                "std",
            ),
            directional_preference_agreement_mean=(
                "directional_preference_agreement",
                "mean",
            ),
            directional_geomean_realized_speedup_mean=(
                "directional_geomean_realized_speedup",
                "mean",
            ),
            directional_mean_regret_percent_mean=(
                "directional_mean_regret_percent",
                "mean",
            ),
            directional_p90_regret_percent_mean=(
                "directional_p90_regret_percent",
                "mean",
            ),
            directional_geomean_mismatch_mean=(
                "directional_geomean_mismatch",
                "mean",
            ),
            mean_vote_margin_mean=("mean_vote_margin", "mean"),
        )
    )

    summary["raw_agreement_minus_uniform_random"] = (
        summary["raw_best_arm_agreement_mean"] - uniform_random
    )
    summary["raw_agreement_minus_empirical_random"] = (
        summary["raw_best_arm_agreement_mean"] - empirical_random
    )
    summary["raw_agreement_minus_always_majority"] = (
        summary["raw_best_arm_agreement_mean"] - always_majority
    )
    summary["reaches_70_percent_raw_agreement"] = (
        summary["raw_best_arm_agreement_mean"] >= 0.70
    )

    summary = summary.sort_values(
        [
            "raw_best_arm_agreement_mean",
            "selection_coverage_mean",
            "directional_mean_regret_percent_mean",
        ],
        ascending=[False, False, True],
    )

    summary_path = (
        args.output_dir
        / "directional-vote-refinement-summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# Directional vote refinement study",
        "",
        "## Random references",
        "",
        f"- uniform binary random direction: {uniform_random:.4f}",
        f"- empirical-prior random direction: {empirical_random:.4f}",
        f"- always predict empirical majority direction: {always_majority:.4f}",
        f"- empirical raw-best shares: x86={p_x86:.4f}, ARM={p_arm:.4f}",
        "",
        (
            "Raw best-arm agreement is binary, so 20% is not the relevant "
            "random baseline. Agreement must always be reported together with "
            "coverage; confidence gates can raise agreement by abstaining."
        ),
        "",
        "## Variants",
        "",
    ]

    for _, r in summary.iterrows():
        report_lines.append(
            f"- `{r['variant']}`: "
            f"coverage={r['selection_coverage_mean']:.4f}, "
            f"raw-agreement={r['raw_best_arm_agreement_mean']:.4f}, "
            f"dir-agreement={r['directional_preference_agreement_mean']:.4f}, "
            f"dir-speedup={r['directional_geomean_realized_speedup_mean']:.4f}x, "
            f"dir-regret={r['directional_mean_regret_percent_mean']:.3f}%, "
            f"p90-regret={r['directional_p90_regret_percent_mean']:.3f}%"
        )

    report_lines += [
        "",
        "## Selection rule",
        "",
        (
            "Do not choose a variant merely because raw agreement crosses 70%. "
            "Prefer a Pareto improvement or a clear agreement gain at a small, "
            "explicit coverage cost. The professor cluster-majority rule remains "
            "the mandatory baseline even if an experimental local/weighted rule "
            "is better."
        ),
        "",
    ]

    report_path = (
        args.output_dir
        / "directional-vote-refinement-report.md"
    )
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    manifest = {
        "frozen_clustering": {
            "representation": (
                "PAPER5 + static_token_count_mean + static_function_count"
            ),
            "scaler": "MinMax per block",
            "kmeans_k": KMEANS_K,
            "n_init": N_INIT,
            "threshold_percent": TAU,
        },
        "variants": {
            name: config
            for name, config in VARIANTS
        },
        "random_references": {
            "uniform_binary": uniform_random,
            "empirical_prior": empirical_random,
            "always_majority": always_majority,
            "p_x86_raw_best": p_x86,
            "p_arm_raw_best": p_arm,
        },
        "inputs": {
            "profiles": {
                "path": str(args.profiles.resolve()),
                "sha256": sha256_file(args.profiles.resolve()),
            },
            "static_metrics": {
                "path": str(args.static_metrics.resolve()),
                "sha256": sha256_file(args.static_metrics.resolve()),
            },
            "language_map": {
                "path": str(args.language_map.resolve()),
                "sha256": sha256_file(args.language_map.resolve()),
            },
            "preferences": {
                "path": str(args.preferences.resolve()),
                "sha256": sha256_file(args.preferences.resolve()),
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "directional-vote-refinement-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("RANDOM REFERENCES")
    print(
        f"uniform_binary={uniform_random:.4f} "
        f"empirical_prior={empirical_random:.4f} "
        f"always_majority={always_majority:.4f} "
        f"x86_share={p_x86:.4f} arm_share={p_arm:.4f}"
    )

    print()
    print("DIRECTIONAL VOTE REFINEMENT")
    cols = [
        "variant",
        "selection_coverage_mean",
        "directional_selection_coverage_mean",
        "raw_best_arm_agreement_mean",
        "directional_preference_agreement_mean",
        "directional_geomean_realized_speedup_mean",
        "directional_mean_regret_percent_mean",
        "directional_p90_regret_percent_mean",
        "directional_geomean_mismatch_mean",
        "reaches_70_percent_raw_agreement",
    ]
    print(
        summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(f"per_seed={per_seed_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Targeted refinement of the x86-vs-ARM vote on the frozen "
            "hybrid K-Means K=6 representation."
        )
    )

    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--static-metrics", type=Path, required=True)
    p.add_argument("--language-map", type=Path, required=True)
    p.add_argument("--preferences", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--seeds",
        default="11,23,37,41,53",
    )

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
