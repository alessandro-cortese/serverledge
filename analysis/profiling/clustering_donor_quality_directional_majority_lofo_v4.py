#!/usr/bin/env python3
"""
Directional-majority LOFO donor selection for Serverledge.

Clarified professor-aligned rule
--------------------------------
The donor must have an architectural preference. Therefore
`architecture-independent` historical functions remain valid members of the
unsupervised cluster, but they do NOT vote for the donor architecture class and
they are NEVER eligible donors in the majority-filtered procedure.

For every held-out target and threshold:
1. remove the target from the catalog;
2. fit scaler(s) and K-Means only on the remaining historical functions;
3. assign the held-out target to one fitted cluster;
4. inside that cluster, count ONLY x86-preferred and arm-preferred members;
5. choose the unique directional majority (x86 vs ARM);
6. restrict donor candidates to same-cluster members of that majority;
7. select the nearest candidate by Manhattan distance.

If the cluster has no directional historical member, or x86 and ARM are tied,
the procedure abstains. The held-out target label is never used before donor
selection.

Important evaluation rule
-------------------------
Preference agreement is defined only when the TARGET itself is directional.
For an architecture-independent target there is no directional preference to
agree with, so preference agreement is reported as N/A. Continuous performance
gap errors are still reported for all selected targets.

Outputs
-------
directional-majority-lofo-per-target.csv
directional-majority-summary.csv
directional-majority-by-target-class.csv
directional-majority-vs-unfiltered.csv
directional-majority-report.md
directional-majority-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
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

DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
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

    x_train = scaler.fit_transform(train[columns].astype(float))
    x_target = scaler.transform(target[columns].astype(float))

    if not np.isfinite(x_train).all() or not np.isfinite(x_target).all():
        raise ValueError(f"Non-finite values for block {columns}")

    return x_train, x_target


def representation(
    train: pd.DataFrame,
    target: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if config["kind"] == "dynamic":
        a, b = fit_scaled(
            train,
            target,
            PAPER5,
            config["scaler"],
        )
        scale = math.sqrt(len(PAPER5))
        return a / scale, b / scale

    static_columns = STATIC_BLOCKS[config["static_block"]]

    if config["kind"] == "static":
        a, b = fit_scaled(
            train,
            target,
            static_columns,
            config["scaler"],
        )
        scale = math.sqrt(len(static_columns))
        return a / scale, b / scale

    dyn_train, dyn_target = fit_scaled(
        train,
        target,
        PAPER5,
        config["scaler"],
    )
    stat_train, stat_target = fit_scaled(
        train,
        target,
        static_columns,
        config["scaler"],
    )

    dyn_train /= math.sqrt(len(PAPER5))
    dyn_target /= math.sqrt(len(PAPER5))

    alpha = float(config["alpha"])
    stat_train = (
        stat_train / math.sqrt(len(static_columns)) * alpha
    )
    stat_target = (
        stat_target / math.sqrt(len(static_columns)) * alpha
    )

    return (
        np.column_stack([dyn_train, stat_train]),
        np.column_stack([dyn_target, stat_target]),
    )


def nearest_index(
    x_target: np.ndarray,
    x_train: np.ndarray,
    candidate_indices: np.ndarray,
    names: np.ndarray,
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
            names[candidate_indices],
            distances,
        )
    )

    local = int(order[0])
    idx = int(candidate_indices[local])

    return idx, float(distances[local])


def directional_majority(
    labels: np.ndarray,
    cluster_indices: np.ndarray,
) -> tuple[str | None, str, int, int, int]:
    cluster_labels = labels[cluster_indices]

    x86 = int(np.sum(cluster_labels == "x86-preferred"))
    arm = int(np.sum(cluster_labels == "arm-preferred"))
    independent = int(
        np.sum(
            cluster_labels
            == "architecture-independent"
        )
    )

    distribution = (
        f"x86-preferred:{x86}|"
        f"architecture-independent:{independent}|"
        f"arm-preferred:{arm}"
    )

    if x86 == 0 and arm == 0:
        return None, distribution, x86, independent, arm

    if x86 == arm:
        return None, distribution, x86, independent, arm

    return (
        "x86-preferred" if x86 > arm else "arm-preferred",
        distribution,
        x86,
        independent,
        arm,
    )


def reward_gap(x86_ms: float, arm_ms: float) -> float:
    return float(np.log(x86_ms / arm_ms))


def raw_best_arm(x86_ms: float, arm_ms: float) -> str:
    return "amd64" if x86_ms < arm_ms else "arm64"


def pair_quality(
    target: pd.Series,
    donor: pd.Series,
    target_label: str,
    donor_label: str,
) -> dict:
    tx = float(target["x86_duration_ms"])
    ta = float(target["arm_duration_ms"])
    dx = float(donor["x86_duration_ms"])
    da = float(donor["arm_duration_ms"])

    target_delta = float(
        target["arm_vs_x86_delta_percent"]
    )
    donor_delta = float(
        donor["arm_vs_x86_delta_percent"]
    )

    target_directional = target_label in DIRECTIONAL

    return {
        "target_is_directional": target_directional,
        "target_is_independent": (
            target_label == "architecture-independent"
        ),
        "donor_preference_label": donor_label,
        "donor_has_directional_preference": (
            donor_label in DIRECTIONAL
        ),
        "directional_preference_agreement": (
            donor_label == target_label
            if target_directional
            else np.nan
        ),
        "raw_best_arm_agreement": (
            raw_best_arm(tx, ta)
            == raw_best_arm(dx, da)
        ),
        "target_delta_percent": target_delta,
        "donor_delta_percent": donor_delta,
        "abs_delta_error_percent": abs(
            target_delta - donor_delta
        ),
        "target_reward_gap": reward_gap(tx, ta),
        "donor_reward_gap": reward_gap(dx, da),
        "abs_reward_gap_error": abs(
            reward_gap(tx, ta)
            - reward_gap(dx, da)
        ),
        "cross_language_donor": (
            str(target["language"])
            != str(donor["language"])
        ),
    }


def prepare_data(args):
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())

    all_static = sorted(
        {
            feature
            for values in STATIC_BLOCKS.values()
            for feature in values
        }
    )

    base_pref = pd.read_csv(
        args.preferences_dir.resolve()
        / "preferences-2p5.csv"
    )

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
            languages[
                ["function_name", "language"]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            base_pref[
                [
                    "function_name",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    complete = source_available(
        df["static_source_available"]
    )

    for column in PAPER5 + all_static:
        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )
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
            f"Expected 59 static-eligible functions, got {len(df)}"
        )

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
                ["architecture_preference"]
            ]["architecture_preference"]
            .to_dict()
        )

    return df, pref_maps, pref_paths


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (
        config,
        threshold,
        mode,
    ), group in detail.groupby(
        [
            "configuration",
            "threshold_percent",
            "selection_mode",
        ]
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

        selected_independent = selected[
            selected["target_is_independent"] == True
        ].copy()

        row = {
            "configuration": config,
            "threshold_percent": threshold,
            "selection_mode": mode,
            "target_rows": len(group),
            "selection_coverage": (
                len(selected) / len(group)
            ),
            "directional_target_rows": len(directional),
            "directional_selection_coverage": (
                len(selected_directional)
                / len(directional)
                if len(directional)
                else np.nan
            ),
            "independent_target_rows": int(
                np.sum(
                    group["target_is_independent"]
                    == True
                )
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
            "raw_best_arm_agreement_all": (
                selected["raw_best_arm_agreement"]
                .astype(float)
                .mean()
                if len(selected)
                else np.nan
            ),
            "mean_abs_delta_error_all": (
                selected["abs_delta_error_percent"].mean()
                if len(selected)
                else np.nan
            ),
            "median_abs_delta_error_all": (
                selected["abs_delta_error_percent"].median()
                if len(selected)
                else np.nan
            ),
            "mean_abs_reward_gap_error_all": (
                selected["abs_reward_gap_error"].mean()
                if len(selected)
                else np.nan
            ),
            "mean_abs_delta_error_directional": (
                selected_directional[
                    "abs_delta_error_percent"
                ].mean()
                if len(selected_directional)
                else np.nan
            ),
            "mean_abs_reward_gap_error_directional": (
                selected_directional[
                    "abs_reward_gap_error"
                ].mean()
                if len(selected_directional)
                else np.nan
            ),
            "mean_abs_delta_error_independent": (
                selected_independent[
                    "abs_delta_error_percent"
                ].mean()
                if len(selected_independent)
                else np.nan
            ),
            "mean_candidate_pool_size": (
                selected["candidate_pool_size"].mean()
                if len(selected)
                else np.nan
            ),
            "cross_language_donor_rate": (
                selected["cross_language_donor"]
                .astype(float)
                .mean()
                if len(selected)
                else np.nan
            ),
            "donor_directional_invariant_rate": (
                selected[
                    "donor_has_directional_preference"
                ]
                .astype(float)
                .mean()
                if len(selected)
                else np.nan
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def by_target_class(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (
        config,
        threshold,
        mode,
        target_label,
    ), group in detail.groupby(
        [
            "configuration",
            "threshold_percent",
            "selection_mode",
            "target_label",
        ]
    ):
        selected = group[
            group["selection_status"] == "selected"
        ]

        rows.append(
            {
                "configuration": config,
                "threshold_percent": threshold,
                "selection_mode": mode,
                "target_label": target_label,
                "target_rows": len(group),
                "selected_rows": len(selected),
                "selection_coverage": (
                    len(selected) / len(group)
                ),
                "directional_preference_agreement": (
                    selected[
                        "directional_preference_agreement"
                    ]
                    .astype(float)
                    .mean()
                    if target_label in DIRECTIONAL
                    and len(selected)
                    else np.nan
                ),
                "raw_best_arm_agreement": (
                    selected[
                        "raw_best_arm_agreement"
                    ]
                    .astype(float)
                    .mean()
                    if len(selected)
                    else np.nan
                ),
                "mean_abs_delta_error_percent": (
                    selected[
                        "abs_delta_error_percent"
                    ].mean()
                    if len(selected)
                    else np.nan
                ),
                "mean_abs_reward_gap_error": (
                    selected[
                        "abs_reward_gap_error"
                    ].mean()
                    if len(selected)
                    else np.nan
                ),
                "mean_candidate_pool_size": (
                    selected["candidate_pool_size"].mean()
                    if len(selected)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def compare_modes(summary: pd.DataFrame) -> pd.DataFrame:
    a = (
        summary[
            summary["selection_mode"]
            == "directional_majority"
        ]
        .set_index(
            ["configuration", "threshold_percent"]
        )
    )

    b = (
        summary[
            summary["selection_mode"]
            == "unfiltered_same_cluster"
        ]
        .set_index(
            ["configuration", "threshold_percent"]
        )
    )

    metrics = [
        "selection_coverage",
        "directional_selection_coverage",
        "directional_preference_agreement",
        "raw_best_arm_agreement_all",
        "mean_abs_delta_error_all",
        "mean_abs_reward_gap_error_all",
        "mean_abs_delta_error_directional",
        "mean_abs_reward_gap_error_directional",
    ]

    rows = []

    for key in sorted(set(a.index) & set(b.index)):
        row = {
            "configuration": key[0],
            "threshold_percent": key[1],
        }

        for metric in metrics:
            av = float(a.loc[key, metric])
            bv = float(b.loc[key, metric])

            row[f"directional_majority_{metric}"] = av
            row[f"unfiltered_{metric}"] = bv
            row[
                f"delta_directional_majority_minus_unfiltered_{metric}"
            ] = av - bv

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    summary: pd.DataFrame,
):
    filtered = summary[
        summary["selection_mode"]
        == "directional_majority"
    ]

    lines = [
        "# Directional-majority donor selection",
        "",
        (
            "`architecture-independent` functions remain in the unsupervised "
            "cluster but are excluded from the architecture vote and donor pool."
        ),
        "",
        (
            "Preference agreement is evaluated only on directional targets. "
            "Independent targets have no x86/ARM preference at the chosen "
            "threshold and therefore are not counted as agreement failures."
        ),
        "",
        "## Directional-majority results",
        "",
    ]

    for threshold, block in filtered.groupby(
        "threshold_percent"
    ):
        ranked = block.sort_values(
            [
                "directional_preference_agreement",
                "mean_abs_reward_gap_error_directional",
                "directional_selection_coverage",
            ],
            ascending=[False, True, False],
        )

        best = ranked.iloc[0]

        lines.append(
            f"- t={threshold:.1f}%: `{best['configuration']}`; "
            f"coverage={best['selection_coverage']:.4f}, "
            f"directional coverage="
            f"{best['directional_selection_coverage']:.4f}, "
            f"directional preference agreement="
            f"{best['directional_preference_agreement']:.4f}, "
            f"directional mean |delta err|="
            f"{best['mean_abs_delta_error_directional']:.3f} pp, "
            f"directional reward-gap err="
            f"{best['mean_abs_reward_gap_error_directional']:.4f}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "The primary professor-aligned outcome is directional preference "
            "agreement among directional targets, together with coverage and "
            "continuous gap errors. Aggregate three-class balanced accuracy is "
            "not the right criterion for this donor rule because independent "
            "functions are deliberately not eligible donors."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(args):
    df, pref_maps, pref_paths = prepare_data(args)
    seeds = parse_csv_ints(args.seeds)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    total_fits = (
        len(CANDIDATES) * len(seeds) * len(df)
    )
    fit_count = 0

    for config_name, config in CANDIDATES.items():
        for seed in seeds:
            for target_idx in range(len(df)):
                target = df.iloc[[target_idx]].copy()

                train = (
                    df.drop(index=target_idx)
                    .copy()
                    .reset_index(drop=True)
                )

                x_train, x_target = representation(
                    train,
                    target,
                    config,
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

                same_cluster = np.flatnonzero(
                    clusters == target_cluster
                )

                names = (
                    train["function_name"]
                    .astype(str)
                    .to_numpy()
                )

                target_name = str(
                    target.iloc[0]["function_name"]
                )

                for threshold, _slug in THRESHOLDS:
                    labels = np.asarray(
                        [
                            pref_maps[threshold][name]
                            for name in names
                        ],
                        dtype=object,
                    )

                    target_label = str(
                        pref_maps[threshold][target_name]
                    )

                    (
                        majority_label,
                        distribution,
                        x86_count,
                        independent_count,
                        arm_count,
                    ) = directional_majority(
                        labels,
                        same_cluster,
                    )

                    # Professor-aligned directional majority.
                    if majority_label is None:
                        status = (
                            "abstain_no_directional"
                            if x86_count == 0 and arm_count == 0
                            else "abstain_directional_tie"
                        )

                        rows.append(
                            {
                                "configuration": config_name,
                                "seed": seed,
                                "threshold_percent": threshold,
                                "selection_mode": "directional_majority",
                                "selection_status": status,
                                "target_function": target_name,
                                "target_language": str(
                                    target.iloc[0]["language"]
                                ),
                                "target_label": target_label,
                                "target_is_directional": (
                                    target_label in DIRECTIONAL
                                ),
                                "target_is_independent": (
                                    target_label
                                    == "architecture-independent"
                                ),
                                "target_cluster": target_cluster,
                                "same_cluster_size": len(
                                    same_cluster
                                ),
                                "x86_cluster_count": x86_count,
                                "independent_cluster_count": (
                                    independent_count
                                ),
                                "arm_cluster_count": arm_count,
                                "cluster_label_distribution": (
                                    distribution
                                ),
                                "directional_majority_label": "",
                                "candidate_pool_size": 0,
                                "donor_function": "",
                                "donor_language": "",
                                "donor_distance": np.nan,
                                "donor_preference_label": "",
                                "donor_has_directional_preference": (
                                    np.nan
                                ),
                                "directional_preference_agreement": (
                                    np.nan
                                ),
                                "raw_best_arm_agreement": np.nan,
                                "abs_delta_error_percent": np.nan,
                                "abs_reward_gap_error": np.nan,
                                "cross_language_donor": np.nan,
                            }
                        )
                    else:
                        candidate_indices = same_cluster[
                            labels[same_cluster]
                            == majority_label
                        ]

                        donor_idx, distance = nearest_index(
                            x_target=x_target[0],
                            x_train=x_train,
                            candidate_indices=candidate_indices,
                            names=names,
                        )

                        donor = train.iloc[donor_idx]
                        donor_name = str(
                            donor["function_name"]
                        )
                        donor_label = str(
                            pref_maps[threshold][donor_name]
                        )

                        if donor_label not in DIRECTIONAL:
                            raise RuntimeError(
                                "Invariant violated: majority-filtered "
                                "donor is not directional"
                            )

                        quality = pair_quality(
                            target.iloc[0],
                            donor,
                            target_label,
                            donor_label,
                        )

                        rows.append(
                            {
                                "configuration": config_name,
                                "seed": seed,
                                "threshold_percent": threshold,
                                "selection_mode": "directional_majority",
                                "selection_status": "selected",
                                "target_function": target_name,
                                "target_language": str(
                                    target.iloc[0]["language"]
                                ),
                                "target_label": target_label,
                                "target_cluster": target_cluster,
                                "same_cluster_size": len(
                                    same_cluster
                                ),
                                "x86_cluster_count": x86_count,
                                "independent_cluster_count": (
                                    independent_count
                                ),
                                "arm_cluster_count": arm_count,
                                "cluster_label_distribution": (
                                    distribution
                                ),
                                "directional_majority_label": (
                                    majority_label
                                ),
                                "candidate_pool_size": len(
                                    candidate_indices
                                ),
                                "donor_function": donor_name,
                                "donor_language": str(
                                    donor["language"]
                                ),
                                "donor_distance": distance,
                                **quality,
                            }
                        )

                    # Unfiltered same-cluster control.
                    donor_idx, distance = nearest_index(
                        x_target=x_target[0],
                        x_train=x_train,
                        candidate_indices=same_cluster,
                        names=names,
                    )

                    donor = train.iloc[donor_idx]
                    donor_name = str(
                        donor["function_name"]
                    )
                    donor_label = str(
                        pref_maps[threshold][donor_name]
                    )

                    quality = pair_quality(
                        target.iloc[0],
                        donor,
                        target_label,
                        donor_label,
                    )

                    rows.append(
                        {
                            "configuration": config_name,
                            "seed": seed,
                            "threshold_percent": threshold,
                            "selection_mode": "unfiltered_same_cluster",
                            "selection_status": "selected",
                            "target_function": target_name,
                            "target_language": str(
                                target.iloc[0]["language"]
                            ),
                            "target_label": target_label,
                            "target_cluster": target_cluster,
                            "same_cluster_size": len(
                                same_cluster
                            ),
                            "x86_cluster_count": x86_count,
                            "independent_cluster_count": (
                                independent_count
                            ),
                            "arm_cluster_count": arm_count,
                            "cluster_label_distribution": (
                                distribution
                            ),
                            "directional_majority_label": (
                                majority_label or ""
                            ),
                            "candidate_pool_size": len(
                                same_cluster
                            ),
                            "donor_function": donor_name,
                            "donor_language": str(
                                donor["language"]
                            ),
                            "donor_distance": distance,
                            **quality,
                        }
                    )

                fit_count += 1

                if (
                    fit_count % 100 == 0
                    or fit_count == total_fits
                ):
                    print(
                        f"fit_progress={fit_count}/"
                        f"{total_fits}"
                    )

        print(f"completed={config_name}")

    detail = pd.DataFrame(rows)

    detail_path = (
        args.output_dir
        / "directional-majority-lofo-per-target.csv"
    )
    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = summarize(detail)

    summary_path = (
        args.output_dir
        / "directional-majority-summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    class_table = by_target_class(detail)

    class_path = (
        args.output_dir
        / "directional-majority-by-target-class.csv"
    )
    class_table.to_csv(
        class_path,
        index=False,
    )

    comparison = compare_modes(summary)

    comparison_path = (
        args.output_dir
        / "directional-majority-vs-unfiltered.csv"
    )
    comparison.to_csv(
        comparison_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "directional-majority-report.md"
    )
    write_report(
        report_path,
        summary,
    )

    manifest = {
        "eligible_functions": len(df),
        "candidates": CANDIDATES,
        "thresholds_percent": [
            t for t, _slug in THRESHOLDS
        ],
        "seeds": seeds,
        "n_init": args.n_init,
        "rule": (
            "architecture-independent functions stay in the cluster but "
            "do not vote and are not donor candidates; directional majority "
            "is computed only between x86-preferred and arm-preferred members"
        ),
        "tie_policy": (
            "abstain if no directional member or x86/ARM directional tie"
        ),
        "agreement_rule": (
            "preference agreement is evaluated only for directional targets; "
            "independent targets are N/A for x86/ARM preference agreement"
        ),
        "inputs": {
            "profiles": {
                "path": str(args.profiles.resolve()),
                "sha256": sha256_file(
                    args.profiles.resolve()
                ),
            },
            "static_metrics": {
                "path": str(
                    args.static_metrics.resolve()
                ),
                "sha256": sha256_file(
                    args.static_metrics.resolve()
                ),
            },
            "language_map": {
                "path": str(
                    args.language_map.resolve()
                ),
                "sha256": sha256_file(
                    args.language_map.resolve()
                ),
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
        args.output_dir
        / "directional-majority-manifest.json"
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

    filtered = (
        summary[
            summary["selection_mode"]
            == "directional_majority"
        ]
        .sort_values(
            [
                "threshold_percent",
                "directional_preference_agreement",
                "mean_abs_reward_gap_error_directional",
            ],
            ascending=[True, False, True],
        )
    )

    print()
    print("DIRECTIONAL-MAJORITY DONOR QUALITY")
    print(
        filtered.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("DIRECTIONAL MAJORITY VS UNFILTERED")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"by_class={class_path}")
    print(f"comparison={comparison_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "LOFO donor selection using x86-vs-ARM directional majority."
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
