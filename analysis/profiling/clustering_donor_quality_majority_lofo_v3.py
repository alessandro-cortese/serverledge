#!/usr/bin/env python3
"""
Professor-compliant LOFO donor selection for Serverledge.

Required operational rule
-------------------------
For every held-out target:
1. remove target from the historical catalog;
2. fit scaler(s) on the remaining functions only;
3. fit K-Means on the remaining functions only;
4. assign held-out target to one of the fitted clusters;
5. inspect ONLY the known historical functions in that target cluster;
6. compute the UNIQUE majority architecture-preference class in that cluster;
7. predict that majority class for the target;
8. restrict donor candidates to same-cluster functions belonging to that
   majority class;
9. rank those candidates by Manhattan distance and select the nearest donor.

If the cluster has no unique majority (tie), the procedure abstains and applies
no donor for that target. No arbitrary tie-breaking between architecture classes
is introduced.

Important
---------
The architecture label of the held-out target is NEVER used for cluster fit,
target assignment, majority calculation, candidate filtering, or donor ranking.
It is read only after selection for evaluation.

The script also evaluates an unfiltered same-cluster Manhattan baseline using
the exact same LOFO cluster fit, so the effect of the professor-required
majority filtering can be isolated.

Outputs
-------
professor-donor-lofo-per-target.csv
professor-majority-classification-summary.csv
professor-donor-quality-summary.csv
professor-majority-vs-unfiltered.csv
professor-donor-lofo-report.md
professor-donor-lofo-manifest.json
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

# Frozen before this professor-compliant donor evaluation.
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

LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]

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

    if not np.isfinite(x_train).all():
        raise ValueError(f"Non-finite train data for {columns}")
    if not np.isfinite(x_target).all():
        raise ValueError(f"Non-finite target data for {columns}")

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

    raise ValueError(f"Unsupported representation kind: {kind}")


def nearest_index(
    x_target: np.ndarray,
    x_train: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_names: np.ndarray,
) -> tuple[int, float]:
    if len(candidate_indices) == 0:
        raise ValueError("Empty candidate set")

    distances = np.sum(
        np.abs(
            x_train[candidate_indices]
            - x_target.reshape(1, -1)
        ),
        axis=1,
    )

    # Deterministic tie break: alphabetical function name.
    order = np.lexsort(
        (
            candidate_names[candidate_indices],
            distances,
        )
    )

    local_idx = int(order[0])
    idx = int(candidate_indices[local_idx])
    return idx, float(distances[local_idx])


def unique_majority(labels: list[str]) -> tuple[str | None, float, str]:
    counts = Counter(labels)

    if not counts:
        return None, 0.0, ""

    top = max(counts.values())
    winners = [
        label
        for label in LABELS
        if counts.get(label, 0) == top
    ]

    distribution = "|".join(
        f"{label}:{counts.get(label, 0)}"
        for label in LABELS
    )

    if len(winners) != 1:
        return None, float(top / len(labels)), distribution

    return (
        winners[0],
        float(top / len(labels)),
        distribution,
    )


def best_arm(x86_ms: float, arm_ms: float) -> str:
    if np.isclose(x86_ms, arm_ms, rtol=0.0, atol=1e-12):
        return "tie"
    return "amd64" if x86_ms < arm_ms else "arm64"


def reward_gap(x86_ms: float, arm_ms: float) -> float:
    # reward = -ln(duration_ms), so reward_arm - reward_x86 = ln(x86/arm).
    return float(np.log(x86_ms / arm_ms))


def quality_for_pair(
    target_row: pd.Series,
    donor_row: pd.Series,
) -> dict:
    target_x86 = float(target_row["x86_duration_ms"])
    target_arm = float(target_row["arm_duration_ms"])
    donor_x86 = float(donor_row["x86_duration_ms"])
    donor_arm = float(donor_row["arm_duration_ms"])

    target_delta = float(
        target_row["arm_vs_x86_delta_percent"]
    )
    donor_delta = float(
        donor_row["arm_vs_x86_delta_percent"]
    )

    target_gap = reward_gap(target_x86, target_arm)
    donor_gap = reward_gap(donor_x86, donor_arm)

    return {
        "target_best_arm": best_arm(
            target_x86,
            target_arm,
        ),
        "donor_best_arm": best_arm(
            donor_x86,
            donor_arm,
        ),
        "best_arm_agreement": (
            best_arm(target_x86, target_arm)
            == best_arm(donor_x86, donor_arm)
        ),
        "target_delta_percent": target_delta,
        "donor_delta_percent": donor_delta,
        "abs_delta_error_percent": abs(
            target_delta - donor_delta
        ),
        "target_reward_gap": target_gap,
        "donor_reward_gap": donor_gap,
        "abs_reward_gap_error": abs(
            target_gap - donor_gap
        ),
        "cross_language_donor": (
            str(target_row["language"])
            != str(donor_row["language"])
        ),
    }


def classification_metrics(
    frame: pd.DataFrame,
) -> dict:
    y_true = frame["target_label"].astype(str).to_numpy()
    y_pred = (
        frame["predicted_majority_label"]
        .fillna("abstain")
        .astype(str)
        .to_numpy()
    )

    covered = y_pred != "abstain"

    recalls = {}
    f1s = []

    for label in LABELS:
        truth = y_true == label
        pred = y_pred == label

        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        recalls[label] = recall
        f1s.append(f1)

    directional = np.isin(
        y_true,
        list(DIRECTIONAL),
    )

    return {
        "classification_coverage": float(
            np.mean(covered)
        ),
        "strict_accuracy": float(
            np.mean(y_true == y_pred)
        ),
        "covered_accuracy": (
            float(
                np.mean(
                    y_true[covered]
                    == y_pred[covered]
                )
            )
            if np.any(covered)
            else np.nan
        ),
        "balanced_accuracy": float(
            np.mean(list(recalls.values()))
        ),
        "macro_f1": float(np.mean(f1s)),
        "directional_accuracy": (
            float(
                np.mean(
                    y_true[directional]
                    == y_pred[directional]
                )
            )
            if np.any(directional)
            else np.nan
        ),
        "x86_recall": recalls["x86-preferred"],
        "independent_recall": recalls[
            "architecture-independent"
        ],
        "arm_recall": recalls["arm-preferred"],
        "abstain_rate": float(
            np.mean(~covered)
        ),
    }


def prepare_data(
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
            f"Expected exactly 59 static-eligible functions, got {len(df)}"
        )

    for column in PAPER5 + all_static:
        if df[column].isna().any():
            raise SystemExit(
                f"Missing numeric values in {column}"
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


def summarize_classification(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    majority = detail[
        detail["selection_mode"] == "majority_filtered"
    ].copy()

    rows = []

    for (
        config,
        threshold,
        seed,
    ), group in majority.groupby(
        [
            "configuration",
            "threshold_percent",
            "seed",
        ]
    ):
        metrics = classification_metrics(group)
        rows.append(
            {
                "configuration": config,
                "threshold_percent": threshold,
                "seed": seed,
                **metrics,
            }
        )

    per_seed = pd.DataFrame(rows)

    return (
        per_seed.groupby(
            [
                "configuration",
                "threshold_percent",
            ],
            as_index=False,
        )
        .agg(
            classification_coverage_mean=(
                "classification_coverage",
                "mean",
            ),
            strict_accuracy_mean=(
                "strict_accuracy",
                "mean",
            ),
            strict_accuracy_std=(
                "strict_accuracy",
                "std",
            ),
            covered_accuracy_mean=(
                "covered_accuracy",
                "mean",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            directional_accuracy_mean=(
                "directional_accuracy",
                "mean",
            ),
            x86_recall_mean=(
                "x86_recall",
                "mean",
            ),
            independent_recall_mean=(
                "independent_recall",
                "mean",
            ),
            arm_recall_mean=(
                "arm_recall",
                "mean",
            ),
            abstain_rate_mean=(
                "abstain_rate",
                "mean",
            ),
        )
    )


def summarize_donor_quality(
    detail: pd.DataFrame,
) -> pd.DataFrame:
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
        ]

        row = {
            "configuration": config,
            "threshold_percent": threshold,
            "selection_mode": mode,
            "target_rows": len(group),
            "selection_coverage": float(
                len(selected) / len(group)
            ),
            "mean_majority_share": float(
                group["cluster_majority_share"].mean()
            ),
            "mean_same_cluster_size": float(
                group["same_cluster_size"].mean()
            ),
        }

        if len(selected):
            row.update(
                {
                    "best_arm_agreement_rate": float(
                        selected[
                            "best_arm_agreement"
                        ].astype(float).mean()
                    ),
                    "mean_abs_delta_error_percent": float(
                        selected[
                            "abs_delta_error_percent"
                        ].mean()
                    ),
                    "median_abs_delta_error_percent": float(
                        selected[
                            "abs_delta_error_percent"
                        ].median()
                    ),
                    "mean_abs_reward_gap_error": float(
                        selected[
                            "abs_reward_gap_error"
                        ].mean()
                    ),
                    "median_abs_reward_gap_error": float(
                        selected[
                            "abs_reward_gap_error"
                        ].median()
                    ),
                    "mean_candidate_pool_size": float(
                        selected[
                            "candidate_pool_size"
                        ].mean()
                    ),
                    "cross_language_donor_rate": float(
                        selected[
                            "cross_language_donor"
                        ].astype(float).mean()
                    ),
                }
            )
        else:
            row.update(
                {
                    "best_arm_agreement_rate": np.nan,
                    "mean_abs_delta_error_percent": np.nan,
                    "median_abs_delta_error_percent": np.nan,
                    "mean_abs_reward_gap_error": np.nan,
                    "median_abs_reward_gap_error": np.nan,
                    "mean_candidate_pool_size": np.nan,
                    "cross_language_donor_rate": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def compare_modes(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    filtered = (
        summary[
            summary["selection_mode"] == "majority_filtered"
        ]
        .set_index(
            [
                "configuration",
                "threshold_percent",
            ]
        )
    )

    unfiltered = (
        summary[
            summary["selection_mode"] == "unfiltered_same_cluster"
        ]
        .set_index(
            [
                "configuration",
                "threshold_percent",
            ]
        )
    )

    rows = []

    metrics = [
        "selection_coverage",
        "best_arm_agreement_rate",
        "mean_abs_delta_error_percent",
        "median_abs_delta_error_percent",
        "mean_abs_reward_gap_error",
        "median_abs_reward_gap_error",
        "mean_candidate_pool_size",
        "cross_language_donor_rate",
    ]

    for key in sorted(
        set(filtered.index)
        & set(unfiltered.index)
    ):
        config, threshold = key

        row = {
            "configuration": config,
            "threshold_percent": threshold,
        }

        for metric in metrics:
            f = float(filtered.loc[key, metric])
            u = float(unfiltered.loc[key, metric])

            row[f"majority_{metric}"] = f
            row[f"unfiltered_{metric}"] = u
            row[
                f"delta_majority_minus_unfiltered_{metric}"
            ] = f - u

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    classification: pd.DataFrame,
    donor_summary: pd.DataFrame,
    comparison: pd.DataFrame,
):
    lines = [
        "# Professor-compliant majority-class donor selection",
        "",
        "Operational pipeline:",
        "",
        (
            "held-out target -> LOFO cluster -> unique majority architecture "
            "class among historical members -> donor candidates restricted to "
            "that majority class -> nearest Manhattan donor."
        ),
        "",
        (
            "The held-out target's own architecture label is used only after "
            "selection for evaluation."
        ),
        "",
        "## Majority-class prediction",
        "",
    ]

    for _, r in classification.sort_values(
        [
            "threshold_percent",
            "balanced_accuracy_mean",
        ],
        ascending=[True, False],
    ).iterrows():
        lines.append(
            f"- t={r['threshold_percent']:.1f}% / "
            f"`{r['configuration']}`: "
            f"coverage={r['classification_coverage_mean']:.4f}, "
            f"strict-acc={r['strict_accuracy_mean']:.4f}, "
            f"balanced-acc={r['balanced_accuracy_mean']:.4f}, "
            f"macro-F1={r['macro_f1_mean']:.4f}, "
            f"directional-acc={r['directional_accuracy_mean']:.4f}"
        )

    lines += [
        "",
        "## Majority-filtered donor quality",
        "",
    ]

    filt = donor_summary[
        donor_summary["selection_mode"] == "majority_filtered"
    ].sort_values(
        [
            "threshold_percent",
            "best_arm_agreement_rate",
            "mean_abs_reward_gap_error",
        ],
        ascending=[True, False, True],
    )

    for _, r in filt.iterrows():
        lines.append(
            f"- t={r['threshold_percent']:.1f}% / "
            f"`{r['configuration']}`: "
            f"coverage={r['selection_coverage']:.4f}, "
            f"best-arm={r['best_arm_agreement_rate']:.4f}, "
            f"mean |delta err|={r['mean_abs_delta_error_percent']:.3f} pp, "
            f"mean reward-gap err={r['mean_abs_reward_gap_error']:.4f}, "
            f"candidate-pool={r['mean_candidate_pool_size']:.2f}"
        )

    lines += [
        "",
        "## Majority filter vs unfiltered same-cluster nearest",
        "",
        (
            "For agreement, positive deltas favor the professor-required "
            "majority filter. For error metrics, negative deltas favor it."
        ),
        "",
    ]

    for _, r in comparison.sort_values(
        [
            "threshold_percent",
            "configuration",
        ]
    ).iterrows():
        lines.append(
            f"- t={r['threshold_percent']:.1f}% / "
            f"`{r['configuration']}`: "
            f"coverage delta="
            f"{r['delta_majority_minus_unfiltered_selection_coverage']:+.4f}, "
            f"best-arm delta="
            f"{r['delta_majority_minus_unfiltered_best_arm_agreement_rate']:+.4f}, "
            f"mean-delta-error delta="
            f"{r['delta_majority_minus_unfiltered_mean_abs_delta_error_percent']:+.3f} pp, "
            f"reward-gap-error delta="
            f"{r['delta_majority_minus_unfiltered_mean_abs_reward_gap_error']:+.4f}"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        (
            "Because majority-class filtering uses historical architecture "
            "labels operationally, donor label agreement at the same threshold "
            "is partly determined by construction and is not used as the main "
            "quality metric. Continuous architecture-delta error, reward-gap "
            "error, raw best-arm agreement, coverage and threshold sensitivity "
            "are more informative for comparing donor quality."
        ),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    df, pref_maps, pref_paths = prepare_data(args)
    seeds = parse_csv_ints(args.seeds)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_fits = (
        len(CANDIDATES)
        * len(seeds)
        * len(df)
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

                train_clusters = model.fit_predict(
                    x_train
                )

                target_cluster = int(
                    model.predict(x_target)[0]
                )

                same_cluster = np.flatnonzero(
                    train_clusters == target_cluster
                )

                if len(same_cluster) == 0:
                    raise RuntimeError(
                        "Assigned target cluster has no training members"
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
                            pref_maps[threshold][name][
                                "architecture_preference"
                            ]
                            for name in names
                        ],
                        dtype=object,
                    )

                    cluster_labels = (
                        labels[same_cluster]
                        .astype(str)
                        .tolist()
                    )

                    (
                        majority_label,
                        majority_share,
                        majority_distribution,
                    ) = unique_majority(
                        cluster_labels
                    )

                    target_label = (
                        pref_maps[threshold][target_name][
                            "architecture_preference"
                        ]
                    )

                    # 1) Professor-required majority-filtered selection.
                    if majority_label is None:
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
                                "threshold_percent": threshold,
                                "selection_mode": "majority_filtered",
                                "selection_status": "abstain_majority_tie",
                                "target_function": target_name,
                                "target_language": str(
                                    target.iloc[0]["language"]
                                ),
                                "target_cluster": target_cluster,
                                "same_cluster_size": int(
                                    len(same_cluster)
                                ),
                                "cluster_majority_label": "",
                                "predicted_majority_label": "",
                                "cluster_majority_share": majority_share,
                                "cluster_label_distribution": (
                                    majority_distribution
                                ),
                                "target_label": target_label,
                                "majority_prediction_correct": False,
                                "candidate_pool_size": 0,
                                "donor_function": "",
                                "donor_language": "",
                                "donor_distance": np.nan,
                                "best_arm_agreement": np.nan,
                                "abs_delta_error_percent": np.nan,
                                "abs_reward_gap_error": np.nan,
                                "cross_language_donor": np.nan,
                            }
                        )
                    else:
                        filtered_indices = same_cluster[
                            labels[same_cluster]
                            == majority_label
                        ]

                        donor_idx, donor_distance = nearest_index(
                            x_target=x_target[0],
                            x_train=x_train,
                            candidate_indices=filtered_indices,
                            candidate_names=names,
                        )

                        donor = train.iloc[donor_idx]
                        quality = quality_for_pair(
                            target.iloc[0],
                            donor,
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
                                "threshold_percent": threshold,
                                "selection_mode": "majority_filtered",
                                "selection_status": "selected",
                                "target_function": target_name,
                                "target_language": str(
                                    target.iloc[0]["language"]
                                ),
                                "target_cluster": target_cluster,
                                "same_cluster_size": int(
                                    len(same_cluster)
                                ),
                                "cluster_majority_label": majority_label,
                                "predicted_majority_label": majority_label,
                                "cluster_majority_share": majority_share,
                                "cluster_label_distribution": (
                                    majority_distribution
                                ),
                                "target_label": target_label,
                                "majority_prediction_correct": (
                                    majority_label == target_label
                                ),
                                "candidate_pool_size": int(
                                    len(filtered_indices)
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

                    # 2) Same-cluster unfiltered control.
                    donor_idx, donor_distance = nearest_index(
                        x_target=x_target[0],
                        x_train=x_train,
                        candidate_indices=same_cluster,
                        candidate_names=names,
                    )

                    donor = train.iloc[donor_idx]
                    quality = quality_for_pair(
                        target.iloc[0],
                        donor,
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
                            "threshold_percent": threshold,
                            "selection_mode": "unfiltered_same_cluster",
                            "selection_status": "selected",
                            "target_function": target_name,
                            "target_language": str(
                                target.iloc[0]["language"]
                            ),
                            "target_cluster": target_cluster,
                            "same_cluster_size": int(
                                len(same_cluster)
                            ),
                            "cluster_majority_label": (
                                majority_label or ""
                            ),
                            "predicted_majority_label": (
                                majority_label or ""
                            ),
                            "cluster_majority_share": majority_share,
                            "cluster_label_distribution": (
                                majority_distribution
                            ),
                            "target_label": target_label,
                            "majority_prediction_correct": (
                                majority_label == target_label
                                if majority_label is not None
                                else False
                            ),
                            "candidate_pool_size": int(
                                len(same_cluster)
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

                fit_count += 1
                if (
                    fit_count % 100 == 0
                    or fit_count == total_fits
                ):
                    print(
                        f"fit_progress={fit_count}/{total_fits}"
                    )

        print(f"completed={config_name}")

    detail = pd.DataFrame(rows)

    detail_path = (
        args.output_dir
        / "professor-donor-lofo-per-target.csv"
    )
    detail.to_csv(detail_path, index=False)

    classification = summarize_classification(
        detail
    )

    classification_path = (
        args.output_dir
        / "professor-majority-classification-summary.csv"
    )
    classification.to_csv(
        classification_path,
        index=False,
    )

    donor_summary = summarize_donor_quality(
        detail
    )

    donor_summary_path = (
        args.output_dir
        / "professor-donor-quality-summary.csv"
    )
    donor_summary.to_csv(
        donor_summary_path,
        index=False,
    )

    comparison = compare_modes(
        donor_summary
    )

    comparison_path = (
        args.output_dir
        / "professor-majority-vs-unfiltered.csv"
    )
    comparison.to_csv(
        comparison_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "professor-donor-lofo-report.md"
    )
    write_report(
        path=report_path,
        classification=classification,
        donor_summary=donor_summary,
        comparison=comparison,
    )

    manifest = {
        "eligible_functions": len(df),
        "candidates": CANDIDATES,
        "thresholds_percent": [
            threshold
            for threshold, _slug in THRESHOLDS
        ],
        "seeds": seeds,
        "n_init": args.n_init,
        "professor_required_pipeline": (
            "LOFO target -> cluster -> unique majority architecture class "
            "among training members -> restrict same-cluster donors to that "
            "majority class -> Manhattan nearest donor."
        ),
        "majority_tie_policy": (
            "abstain / no donor; no arbitrary architecture-label tie break"
        ),
        "control": (
            "unfiltered same-cluster Manhattan nearest donor using the exact "
            "same LOFO scaler and K-Means fit"
        ),
        "ground_truth_leakage_control": (
            "held-out target architecture label is never used before donor "
            "selection; only training/catalog labels determine cluster majority"
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
        / "professor-donor-lofo-manifest.json"
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
    print("MAJORITY-CLASS PREDICTION")
    print(
        classification.sort_values(
            [
                "threshold_percent",
                "balanced_accuracy_mean",
            ],
            ascending=[True, False],
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("PROFESSOR-COMPLIANT DONOR QUALITY")
    print(
        donor_summary[
            donor_summary["selection_mode"]
            == "majority_filtered"
        ]
        .sort_values(
            [
                "threshold_percent",
                "best_arm_agreement_rate",
                "mean_abs_reward_gap_error",
            ],
            ascending=[True, False, True],
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("MAJORITY FILTER VS UNFILTERED SAME-CLUSTER")
    print(
        comparison.sort_values(
            [
                "threshold_percent",
                "configuration",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(f"classification={classification_path}")
    print(f"donor_summary={donor_summary_path}")
    print(f"comparison={comparison_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Professor-compliant majority-class LOFO donor selection."
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
