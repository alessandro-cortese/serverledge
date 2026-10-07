#!/usr/bin/env python3
"""
LOFO donor-quality validation for the Serverledge clustering shortlist.

This is the operational validation stage after unsupervised feature screening.

For each target function:
1. remove the target from the historical catalog;
2. fit preprocessing on the remaining functions only;
3. fit K-Means on the remaining functions only;
4. transform and assign the held-out target to a cluster;
5. select the nearest same-cluster donor by Manhattan distance;
6. also compute a global-nearest Manhattan baseline without cluster restriction;
7. evaluate the selected donor only after selection using x86/ARM ground truth.

Architecture ground truth never enters scaling, clustering, target assignment or
donor ranking.

The experiment uses the same 59 static-eligible functions for every candidate,
so dynamic/static/hybrid configurations are directly comparable.

Outputs
-------
donor-lofo-per-target.csv
donor-lofo-summary.csv
donor-lofo-cluster-vs-global.csv
donor-lofo-report.md
donor-lofo-manifest.json
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC_BLOCKS = {
    "ccn_tokens": [
        "static_ccn_mean",
        "static_token_count_mean",
    ],
    "ccn_functions": [
        "static_ccn_mean",
        "static_function_count",
    ],
    "tokens_functions": [
        "static_token_count_mean",
        "static_function_count",
    ],
    "static4": [
        "static_ccn_mean",
        "static_function_nloc_mean",
        "static_token_count_mean",
        "static_function_count",
    ],
}

# Pre-declared shortlist selected BEFORE donor-ground-truth evaluation.
# It deliberately spans:
# - historical PAPER-5 continuity;
# - a less-degenerate PAPER-5 baseline;
# - pure static;
# - three balanced hybrid alternatives;
# - one strongly balanced full-static hybrid control.
CANDIDATES = {
    "paper5_minmax_k5": {
        "kind": "dynamic",
        "scaler": "minmax",
        "k": 5,
        "static_block": None,
        "alpha": 0.0,
    },
    "paper5_minmax_k8": {
        "kind": "dynamic",
        "scaler": "minmax",
        "k": 8,
        "static_block": None,
        "alpha": 0.0,
    },
    "static_ccn_tokens_standard_k7": {
        "kind": "static",
        "scaler": "standard",
        "k": 7,
        "static_block": "ccn_tokens",
        "alpha": 1.0,
    },
    "hybrid_ccn_tokens_minmax_k7_a1": {
        "kind": "hybrid",
        "scaler": "minmax",
        "k": 7,
        "static_block": "ccn_tokens",
        "alpha": 1.0,
    },
    "hybrid_ccn_functions_standard_k7_a1": {
        "kind": "hybrid",
        "scaler": "standard",
        "k": 7,
        "static_block": "ccn_functions",
        "alpha": 1.0,
    },
    "hybrid_tokens_functions_minmax_k6_a1": {
        "kind": "hybrid",
        "scaler": "minmax",
        "k": 6,
        "static_block": "tokens_functions",
        "alpha": 1.0,
    },
    "hybrid_static4_standard_k9_a1": {
        "kind": "hybrid",
        "scaler": "standard",
        "k": 9,
        "static_block": "static4",
        "alpha": 1.0,
    },
}

THRESHOLDS = [
    (2.5, "2p5"),
    (5.0, "5"),
    (10.0, "10"),
    (15.0, "15"),
    (20.0, "20"),
    (25.0, "25"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def source_available(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def scaler_from_name(name: str):
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def fit_scaled(
    train: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    scaler_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = scaler_from_name(scaler_name)

    x_train = scaler.fit_transform(
        train[columns].astype(float)
    )
    x_target = scaler.transform(
        target[columns].astype(float)
    )

    if not np.isfinite(x_train).all():
        raise ValueError(f"Non-finite train data for columns={columns}")
    if not np.isfinite(x_target).all():
        raise ValueError(f"Non-finite target data for columns={columns}")

    return x_train, x_target


def build_representation(
    train: pd.DataFrame,
    target: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    scaler_name = config["scaler"]
    kind = config["kind"]

    if kind == "dynamic":
        x_train, x_target = fit_scaled(
            train,
            target,
            PAPER5,
            scaler_name,
        )
        factor = math.sqrt(len(PAPER5))
        return x_train / factor, x_target / factor

    static_columns = STATIC_BLOCKS[config["static_block"]]

    if kind == "static":
        x_train, x_target = fit_scaled(
            train,
            target,
            static_columns,
            scaler_name,
        )
        factor = math.sqrt(len(static_columns))
        return x_train / factor, x_target / factor

    if kind == "hybrid":
        dyn_train, dyn_target = fit_scaled(
            train,
            target,
            PAPER5,
            scaler_name,
        )
        stat_train, stat_target = fit_scaled(
            train,
            target,
            static_columns,
            scaler_name,
        )

        dyn_train = dyn_train / math.sqrt(len(PAPER5))
        dyn_target = dyn_target / math.sqrt(len(PAPER5))

        stat_train = (
            stat_train
            / math.sqrt(len(static_columns))
            * float(config["alpha"])
        )
        stat_target = (
            stat_target
            / math.sqrt(len(static_columns))
            * float(config["alpha"])
        )

        return (
            np.column_stack([dyn_train, stat_train]),
            np.column_stack([dyn_target, stat_target]),
        )

    raise ValueError(f"Unsupported kind: {kind}")


def best_arm(x86_ms: float, arm_ms: float) -> str:
    if np.isclose(x86_ms, arm_ms, rtol=0.0, atol=1e-12):
        return "tie"
    return "amd64" if x86_ms < arm_ms else "arm64"


def reward_gap(x86_ms: float, arm_ms: float) -> float:
    # reward = -ln(duration)
    # ARM - x86 reward gap = ln(x86 / ARM)
    return float(np.log(x86_ms / arm_ms))


def nearest_index(
    x_target: np.ndarray,
    x_train: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_names: np.ndarray,
) -> tuple[int, float]:
    distances = np.sum(
        np.abs(
            x_train[candidate_indices]
            - x_target.reshape(1, -1)
        ),
        axis=1,
    )

    order = np.lexsort(
        (
            candidate_names[candidate_indices],
            distances,
        )
    )

    local = int(order[0])
    idx = int(candidate_indices[local])
    return idx, float(distances[local])


def prepare_frame(
    args,
) -> tuple[
    pd.DataFrame,
    dict[float, dict[str, dict]],
    dict[str, Path],
]:
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())

    all_static = sorted(
        {
            feature
            for block in STATIC_BLOCKS.values()
            for feature in block
        }
    )

    primary_path = (
        args.preferences_dir.resolve()
        / "preferences-2p5.csv"
    )
    primary = pd.read_csv(primary_path)

    df = (
        profiles.merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    *all_static,
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
            primary[
                [
                    "function_name",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                    "architecture_preference",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    complete = source_available(df["static_source_available"])

    for column in PAPER5 + all_static:
        numeric = pd.to_numeric(df[column], errors="coerce")
        df[column] = numeric
        if column in all_static:
            complete &= numeric.notna()

    df = (
        df[complete]
        .copy()
        .sort_values("function_name")
        .reset_index(drop=True)
    )

    if len(df) != 59:
        raise SystemExit(
            f"Expected exactly 59 static-eligible functions, got {len(df)}"
        )

    for column in PAPER5 + all_static:
        if df[column].isna().any():
            raise SystemExit(f"Missing values in {column}")

    pref_maps = {}
    pref_paths = {}

    for threshold, slug in THRESHOLDS:
        path = (
            args.preferences_dir.resolve()
            / f"preferences-{slug}.csv"
        )
        table = pd.read_csv(path)

        pref_paths[slug] = path
        pref_maps[threshold] = (
            table.set_index("function_name")[
                [
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ]
            .to_dict(orient="index")
        )

    return df, pref_maps, pref_paths


def evaluate_pair(
    target_row: pd.Series,
    donor_row: pd.Series,
    pref_maps: dict[float, dict[str, dict]],
) -> dict:
    target_name = str(target_row["function_name"])
    donor_name = str(donor_row["function_name"])

    target_x86 = float(target_row["x86_duration_ms"])
    target_arm = float(target_row["arm_duration_ms"])
    donor_x86 = float(donor_row["x86_duration_ms"])
    donor_arm = float(donor_row["arm_duration_ms"])

    target_delta = float(target_row["arm_vs_x86_delta_percent"])
    donor_delta = float(donor_row["arm_vs_x86_delta_percent"])

    out = {
        "target_best_arm": best_arm(target_x86, target_arm),
        "donor_best_arm": best_arm(donor_x86, donor_arm),
        "best_arm_agreement": (
            best_arm(target_x86, target_arm)
            == best_arm(donor_x86, donor_arm)
        ),
        "target_delta_percent": target_delta,
        "donor_delta_percent": donor_delta,
        "abs_delta_error_percent": abs(
            target_delta - donor_delta
        ),
        "target_reward_gap": reward_gap(
            target_x86,
            target_arm,
        ),
        "donor_reward_gap": reward_gap(
            donor_x86,
            donor_arm,
        ),
        "abs_reward_gap_error": abs(
            reward_gap(target_x86, target_arm)
            - reward_gap(donor_x86, donor_arm)
        ),
        "cross_language_donor": (
            str(target_row["language"])
            != str(donor_row["language"])
        ),
    }

    for threshold, _slug in THRESHOLDS:
        t = pref_maps[threshold][target_name][
            "architecture_preference"
        ]
        d = pref_maps[threshold][donor_name][
            "architecture_preference"
        ]

        key = str(threshold).replace(".", "p")
        out[f"label_agreement_t{key}"] = (t == d)

    return out


def summarize_seed_metrics(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "configuration",
        "ranking_scope",
    ]

    for keys, group in detail.groupby(group_cols):
        config, scope = keys

        per_seed = (
            group.groupby("seed")
            .agg(
                coverage=("donor_function", lambda s: s.notna().mean()),
                best_arm_agreement=(
                    "best_arm_agreement",
                    "mean",
                ),
                mean_abs_delta_error_percent=(
                    "abs_delta_error_percent",
                    "mean",
                ),
                median_abs_delta_error_percent=(
                    "abs_delta_error_percent",
                    "median",
                ),
                mean_abs_reward_gap_error=(
                    "abs_reward_gap_error",
                    "mean",
                ),
                median_abs_reward_gap_error=(
                    "abs_reward_gap_error",
                    "median",
                ),
                mean_donor_distance=(
                    "donor_distance",
                    "mean",
                ),
                median_donor_distance=(
                    "donor_distance",
                    "median",
                ),
                mean_candidate_pool_size=(
                    "candidate_pool_size",
                    "mean",
                ),
                cross_language_donor_rate=(
                    "cross_language_donor",
                    "mean",
                ),
                label_agreement_t2p5=(
                    "label_agreement_t2p5",
                    "mean",
                ),
                label_agreement_t5p0=(
                    "label_agreement_t5p0",
                    "mean",
                ),
                label_agreement_t10p0=(
                    "label_agreement_t10p0",
                    "mean",
                ),
                label_agreement_t15p0=(
                    "label_agreement_t15p0",
                    "mean",
                ),
                label_agreement_t20p0=(
                    "label_agreement_t20p0",
                    "mean",
                ),
                label_agreement_t25p0=(
                    "label_agreement_t25p0",
                    "mean",
                ),
            )
            .reset_index()
        )

        row = {
            "configuration": config,
            "ranking_scope": scope,
            "target_count": int(
                group["target_function"].nunique()
            ),
            "seed_count": int(
                group["seed"].nunique()
            ),
            "coverage_mean": float(
                per_seed["coverage"].mean()
            ),
            "best_arm_agreement_mean": float(
                per_seed["best_arm_agreement"].mean()
            ),
            "best_arm_agreement_std": float(
                per_seed["best_arm_agreement"].std(ddof=0)
            ),
            "mean_abs_delta_error_percent": float(
                group["abs_delta_error_percent"].mean()
            ),
            "median_abs_delta_error_percent": float(
                group["abs_delta_error_percent"].median()
            ),
            "mean_abs_reward_gap_error": float(
                group["abs_reward_gap_error"].mean()
            ),
            "median_abs_reward_gap_error": float(
                group["abs_reward_gap_error"].median()
            ),
            "mean_donor_distance": float(
                group["donor_distance"].mean()
            ),
            "median_donor_distance": float(
                group["donor_distance"].median()
            ),
            "mean_candidate_pool_size": float(
                group["candidate_pool_size"].mean()
            ),
            "cross_language_donor_rate": float(
                group["cross_language_donor"].mean()
            ),
        }

        for threshold, _slug in THRESHOLDS:
            key = str(threshold).replace(".", "p")
            column = f"label_agreement_t{key}"
            row[f"{column}_mean"] = float(
                group[column].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_cluster_vs_global(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    cluster = (
        summary[
            summary["ranking_scope"] == "same_cluster"
        ]
        .set_index("configuration")
    )
    global_ = (
        summary[
            summary["ranking_scope"] == "global"
        ]
        .set_index("configuration")
    )

    rows = []

    metrics = [
        "best_arm_agreement_mean",
        "mean_abs_delta_error_percent",
        "median_abs_delta_error_percent",
        "mean_abs_reward_gap_error",
        "median_abs_reward_gap_error",
        "cross_language_donor_rate",
    ]

    for config in sorted(
        set(cluster.index) & set(global_.index)
    ):
        row = {"configuration": config}

        for metric in metrics:
            c = float(cluster.loc[config, metric])
            g = float(global_.loc[config, metric])

            row[f"clustered_{metric}"] = c
            row[f"global_{metric}"] = g
            row[f"delta_cluster_minus_global_{metric}"] = c - g

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
):
    clustered = (
        summary[
            summary["ranking_scope"] == "same_cluster"
        ]
        .copy()
        .sort_values(
            [
                "best_arm_agreement_mean",
                "mean_abs_reward_gap_error",
                "mean_abs_delta_error_percent",
            ],
            ascending=[False, True, True],
        )
    )

    lines = [
        "# LOFO donor-quality validation",
        "",
        (
            "Each target is removed before preprocessing and K-Means fit. "
            "The target is then assigned to a cluster and the nearest donor "
            "inside that training cluster is selected with Manhattan distance."
        ),
        "",
        (
            "Architecture ground truth is used only after donor selection to "
            "measure donor quality."
        ),
        "",
        "## Same-cluster Manhattan ranking",
        "",
    ]

    for _, r in clustered.iterrows():
        lines.append(
            f"- `{r['configuration']}`: "
            f"coverage={r['coverage_mean']:.4f}, "
            f"best-arm agreement={r['best_arm_agreement_mean']:.4f}, "
            f"mean |delta error|={r['mean_abs_delta_error_percent']:.3f} pp, "
            f"median |delta error|={r['median_abs_delta_error_percent']:.3f} pp, "
            f"mean reward-gap error={r['mean_abs_reward_gap_error']:.4f}, "
            f"cross-language donor rate={r['cross_language_donor_rate']:.4f}"
        )

    lines += [
        "",
        "## Cluster restriction vs global nearest-neighbour",
        "",
        (
            "Positive delta in best-arm agreement means cluster restriction "
            "helps; negative deltas in error metrics mean cluster restriction "
            "reduces donor error."
        ),
        "",
    ]

    for _, r in comparison.iterrows():
        lines.append(
            f"- `{r['configuration']}`: "
            f"best-arm delta="
            f"{r['delta_cluster_minus_global_best_arm_agreement_mean']:+.4f}, "
            f"mean-delta-error delta="
            f"{r['delta_cluster_minus_global_mean_abs_delta_error_percent']:+.3f} pp, "
            f"reward-gap-error delta="
            f"{r['delta_cluster_minus_global_mean_abs_reward_gap_error']:+.4f}"
        )

    lines += [
        "",
        "## Selection rule",
        "",
        (
            "No single donor-quality metric is sufficient. Prefer candidates "
            "with full coverage, high best-arm agreement, low architecture-delta "
            "and reward-gap errors, and stable conclusions across seeds. "
            "The global-nearest baseline tests whether clustering itself adds "
            "value rather than merely changing the nearest-neighbour search."
        ),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    df, pref_maps, pref_paths = prepare_frame(args)

    seeds = parse_csv_ints(args.seeds)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for config_name, config in CANDIDATES.items():
        for seed in seeds:
            for target_idx in range(len(df)):
                target = df.iloc[[target_idx]].copy()
                train = (
                    df.drop(index=target_idx)
                    .copy()
                    .reset_index(drop=True)
                )

                x_train, x_target = build_representation(
                    train=train,
                    target=target,
                    config=config,
                )

                model = KMeans(
                    n_clusters=int(config["k"]),
                    n_init=args.n_init,
                    random_state=seed,
                )

                clusters = model.fit_predict(x_train)
                target_cluster = int(
                    model.predict(x_target)[0]
                )

                names = (
                    train["function_name"]
                    .astype(str)
                    .to_numpy()
                )

                same_cluster = np.flatnonzero(
                    clusters == target_cluster
                )

                if len(same_cluster) == 0:
                    raise RuntimeError(
                        "K-Means target cluster has no training members"
                    )

                global_indices = np.arange(len(train))

                for scope, candidate_indices in [
                    ("same_cluster", same_cluster),
                    ("global", global_indices),
                ]:
                    donor_idx, donor_distance = nearest_index(
                        x_target=x_target[0],
                        x_train=x_train,
                        candidate_indices=candidate_indices,
                        candidate_names=names,
                    )

                    donor = train.iloc[donor_idx]

                    quality = evaluate_pair(
                        target_row=target.iloc[0],
                        donor_row=donor,
                        pref_maps=pref_maps,
                    )

                    rows.append(
                        {
                            "configuration": config_name,
                            "representation_kind": config["kind"],
                            "scaler": config["scaler"],
                            "k": int(config["k"]),
                            "static_block": (
                                config["static_block"] or ""
                            ),
                            "alpha": float(config["alpha"]),
                            "seed": seed,
                            "ranking_scope": scope,
                            "target_function": str(
                                target.iloc[0]["function_name"]
                            ),
                            "target_language": str(
                                target.iloc[0]["language"]
                            ),
                            "target_cluster": target_cluster,
                            "candidate_pool_size": int(
                                len(candidate_indices)
                            ),
                            "donor_function": str(
                                donor["function_name"]
                            ),
                            "donor_language": str(
                                donor["language"]
                            ),
                            "donor_distance": donor_distance,
                            **quality,
                        }
                    )

        print(f"completed={config_name}")

    detail = pd.DataFrame(rows)

    detail_path = (
        args.output_dir / "donor-lofo-per-target.csv"
    )
    detail.to_csv(detail_path, index=False)

    summary = summarize_seed_metrics(detail)

    summary_path = (
        args.output_dir / "donor-lofo-summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    comparison = build_cluster_vs_global(summary)

    comparison_path = (
        args.output_dir / "donor-lofo-cluster-vs-global.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    report_path = (
        args.output_dir / "donor-lofo-report.md"
    )
    write_report(
        path=report_path,
        summary=summary,
        comparison=comparison,
    )

    manifest = {
        "eligible_functions": len(df),
        "candidates": CANDIDATES,
        "seeds": seeds,
        "n_init": args.n_init,
        "donor_metric": "Manhattan (L1) in the candidate representation",
        "validation": (
            "Leave-one-function-out. Target excluded before scaler fit and "
            "K-Means fit. Architecture ground truth evaluated post-selection."
        ),
        "comparison_baseline": (
            "Global Manhattan nearest neighbour using the same representation "
            "and the same held-out preprocessing."
        ),
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
                slug: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for slug, path in pref_paths.items()
            },
        },
    }

    manifest_path = (
        args.output_dir / "donor-lofo-manifest.json"
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

    clustered = (
        summary[
            summary["ranking_scope"] == "same_cluster"
        ]
        .sort_values(
            [
                "best_arm_agreement_mean",
                "mean_abs_reward_gap_error",
                "mean_abs_delta_error_percent",
            ],
            ascending=[False, True, True],
        )
    )

    print()
    print("SAME-CLUSTER MANHATTAN DONOR QUALITY")
    print(
        clustered[
            [
                "configuration",
                "coverage_mean",
                "best_arm_agreement_mean",
                "best_arm_agreement_std",
                "mean_abs_delta_error_percent",
                "median_abs_delta_error_percent",
                "mean_abs_reward_gap_error",
                "median_abs_reward_gap_error",
                "mean_candidate_pool_size",
                "cross_language_donor_rate",
                "label_agreement_t2p5_mean",
                "label_agreement_t15p0_mean",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("CLUSTER RESTRICTION VS GLOBAL NEAREST")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"comparison={comparison_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "LOFO donor-quality validation for dynamic/static K-Means shortlist."
        )
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--static-metrics",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--language-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--preferences-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seeds",
        default="11,23,37,41,53",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=50,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
