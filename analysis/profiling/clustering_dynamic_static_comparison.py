#!/usr/bin/env python3
"""
Compare the current dynamic representation against a minimal, pre-declared
static augmentation on the SAME 48 complete-source functions.

Feature groups
--------------
dynamic3:
    utilized_cpus
    log1p(cpu_user_delta_ms + cpu_kernel_delta_ms)
    cpu_kernel_delta_ms / (cpu_user_delta_ms + cpu_kernel_delta_ms)

static_token:
    log1p(static_token_count_sum)

static_token_functions:
    log1p(static_token_count_sum)
    log1p(static_function_count)

dynamic3_plus_token:
    dynamic3 + log1p(static_token_count_sum)

dynamic3_plus_token_functions:
    dynamic3 + log1p(static_token_count_sum) + log1p(static_function_count)

Why this deliberately small family?
-----------------------------------
The preceding signal analysis found:
* no static feature survived BH-FDR at 0.05;
* parameter-count metrics were strongly language-confounded;
* NLOC / token count / CCN sum were highly redundant;
* source-behavioral indicators were sparse and the five missing sources are
  specifically inherited external-process wrappers, making those indicators
  unsuitable as the primary fair augmentation.

This benchmark therefore tests only two low-language-confounding, conceptually
cross-language structural quantities: source size (token count) and function
count.

The held-out target is excluded from scaler and KMeans fit. Cluster prediction
uses the unique majority architecture label among training members; ties
abstain. This is an exploratory comparison over k; a later nested validation is
still required before freezing a final configuration.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

LABELS = ["x86-preferred", "architecture-independent", "arm-preferred"]
DIRECTIONAL = {"x86-preferred", "arm-preferred"}

GROUPS = [
    "dynamic3",
    "static_token",
    "static_token_functions",
    "dynamic3_plus_token",
    "dynamic3_plus_token_functions",
]


def threshold_slug(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")


def unique_majority(labels: pd.Series) -> str:
    counts = Counter(labels.tolist())
    top = max(counts.values())
    winners = [label for label in LABELS if counts.get(label, 0) == top]
    return winners[0] if len(winners) == 1 else "abstain"


def representation(frame: pd.DataFrame, group: str) -> np.ndarray:
    cpu = frame["utilized_cpus"].astype(float).to_numpy()
    user = frame["cpu_user_delta_ms"].astype(float).to_numpy()
    kernel = frame["cpu_kernel_delta_ms"].astype(float).to_numpy()
    total = user + kernel
    dynamic = np.column_stack([
        cpu,
        np.log1p(np.clip(total, 0, None)),
        kernel / np.maximum(total, 1e-12),
    ])

    tokens = np.log1p(
        np.clip(frame["static_token_count_sum"].astype(float).to_numpy(), 0, None)
    ).reshape(-1, 1)
    funcs = np.log1p(
        np.clip(frame["static_function_count"].astype(float).to_numpy(), 0, None)
    ).reshape(-1, 1)

    if group == "dynamic3":
        return dynamic
    if group == "static_token":
        return tokens
    if group == "static_token_functions":
        return np.column_stack([tokens, funcs])
    if group == "dynamic3_plus_token":
        return np.column_stack([dynamic, tokens])
    if group == "dynamic3_plus_token_functions":
        return np.column_stack([dynamic, tokens, funcs])
    raise ValueError(group)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    recalls = {}
    f1s = []
    for label in LABELS:
        truth = y == label
        pred = p == label
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = recall
        f1s.append(f1)

    directional = np.isin(y, list(DIRECTIONAL))
    return {
        "coverage": float(np.mean(p != "abstain")),
        "strict_accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": float(np.mean(f1s)),
        "directional_accuracy": (
            float(np.mean(y[directional] == p[directional]))
            if np.any(directional) else 0.0
        ),
        "x86_recall": recalls["x86-preferred"],
        "independent_recall": recalls["architecture-independent"],
        "arm_recall": recalls["arm-preferred"],
        "pred_x86": int(np.sum(p == "x86-preferred")),
        "pred_independent": int(np.sum(p == "architecture-independent")),
        "pred_arm": int(np.sum(p == "arm-preferred")),
        "pred_abstain": int(np.sum(p == "abstain")),
    }


def lofo(df: pd.DataFrame, group: str, k: int, seed: int, n_init: int):
    true, pred, cluster_sizes = [], [], []

    for idx in range(len(df)):
        train = df.drop(index=idx)
        target = df.iloc[[idx]]

        scaler = RobustScaler()
        x_train = scaler.fit_transform(representation(train, group))
        x_target = scaler.transform(representation(target, group))

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        )
        train_clusters = model.fit_predict(x_train)
        target_cluster = int(model.predict(x_target)[0])

        members = np.flatnonzero(train_clusters == target_cluster)
        prediction = unique_majority(
            train.iloc[members]["architecture_preference"]
        )

        true.append(str(target.iloc[0]["architecture_preference"]))
        pred.append(prediction)
        cluster_sizes.append(len(members))

    y = np.asarray(true, dtype=object)
    p = np.asarray(pred, dtype=object)
    out = classification_metrics(y, p)
    out["mean_target_cluster_size"] = float(np.mean(cluster_sizes))
    out["large_cluster_target_share"] = float(
        np.mean(np.asarray(cluster_sizes) >= 20)
    )
    return out


def full_fit_stats(
    df: pd.DataFrame,
    group: str,
    k: int,
    seed: int,
    n_init: int,
):
    scaler = RobustScaler()
    x = scaler.fit_transform(representation(df, group))
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    clusters = model.fit_predict(x)

    sizes = pd.Series(clusters).value_counts()
    labels = df["architecture_preference"].to_numpy()

    correct = 0
    for c in sorted(set(clusters)):
        vals = labels[clusters == c]
        correct += max(Counter(vals.tolist()).values())

    return {
        "full_purity": float(correct / len(df)),
        "silhouette": float(silhouette_score(x, clusters)),
        "davies_bouldin": float(davies_bouldin_score(x, clusters)),
        "calinski_harabasz": float(calinski_harabasz_score(x, clusters)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "max_cluster_share": float(sizes.max() / len(df)),
        "singleton_count": int(np.sum(sizes == 1)),
    }


def seed_stability(
    df: pd.DataFrame,
    group: str,
    k: int,
    seeds: list[int],
    n_init: int,
) -> float:
    scaler = RobustScaler()
    x = scaler.fit_transform(representation(df, group))
    partitions = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        partitions.append(model.fit_predict(x))

    values = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            values.append(adjusted_rand_score(partitions[i], partitions[j]))
    return float(np.mean(values)) if values else 1.0


def main(args):
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    prefs = pd.read_csv(
        args.preferences_dir.resolve()
        / f"preferences-{threshold_slug(args.threshold)}.csv"
    )

    df = (
        profiles.merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    "static_token_count_sum",
                    "static_function_count",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            prefs[["function_name", "architecture_preference"]],
            on="function_name",
            validate="one_to_one",
        )
    )

    df = df[
        df["static_source_available"].astype(bool)
        & df["static_token_count_sum"].notna()
        & df["static_function_count"].notna()
    ].copy()
    df = df.sort_values("function_name").reset_index(drop=True)

    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"complete_case_functions={len(df)}")
    print("\nCLASS COUNTS:")
    print(
        df["architecture_preference"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .to_string()
    )

    rows = []
    total = len(groups) * len(ks) * len(seeds)
    done = 0

    for group in groups:
        for k in ks:
            ari = seed_stability(df, group, k, seeds, args.n_init)
            for seed in seeds:
                result = lofo(df, group, k, seed, args.n_init)
                full = full_fit_stats(df, group, k, seed, args.n_init)
                rows.append({
                    "feature_group": group,
                    "k": k,
                    "seed": seed,
                    "mean_pairwise_seed_ari": ari,
                    **result,
                    **full,
                })
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"progress={done}/{total}")

    detail = pd.DataFrame(rows)
    detail.to_csv(
        args.output_dir / "dynamic-static-comparison-detail.csv",
        index=False,
    )

    summary = (
        detail.groupby(["feature_group", "k"])
        .agg(
            seed_count=("seed", "count"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            strict_accuracy_mean=("strict_accuracy", "mean"),
            strict_accuracy_std=("strict_accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            directional_accuracy_mean=("directional_accuracy", "mean"),
            directional_accuracy_std=("directional_accuracy", "std"),
            x86_recall_mean=("x86_recall", "mean"),
            independent_recall_mean=("independent_recall", "mean"),
            arm_recall_mean=("arm_recall", "mean"),
            pred_x86_mean=("pred_x86", "mean"),
            pred_independent_mean=("pred_independent", "mean"),
            pred_arm_mean=("pred_arm", "mean"),
            pred_abstain_mean=("pred_abstain", "mean"),
            large_cluster_target_share_mean=("large_cluster_target_share", "mean"),
            full_purity_mean=("full_purity", "mean"),
            silhouette_mean=("silhouette", "mean"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            max_cluster_share_mean=("max_cluster_share", "mean"),
            singleton_count_mean=("singleton_count", "mean"),
            mean_pairwise_seed_ari=("mean_pairwise_seed_ari", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "directional_accuracy_mean",
                "strict_accuracy_mean",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "rank_exploratory", np.arange(1, len(summary) + 1))
    summary.to_csv(
        args.output_dir / "dynamic-static-comparison-summary.csv",
        index=False,
    )

    cols = [
        "rank_exploratory",
        "feature_group",
        "k",
        "coverage_mean",
        "strict_accuracy_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "directional_accuracy_mean",
        "directional_accuracy_std",
        "x86_recall_mean",
        "independent_recall_mean",
        "arm_recall_mean",
        "large_cluster_target_share_mean",
        "max_cluster_share_mean",
        "singleton_count_mean",
        "mean_pairwise_seed_ari",
    ]

    print("\nTOP CONFIGURATIONS:")
    print(summary[cols].head(args.top_n).to_string(index=False))

    print("\nBEST CONFIGURATION PER FEATURE GROUP:")
    best = (
        summary.sort_values(
            [
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "directional_accuracy_mean",
            ],
            ascending=[False, False, False],
        )
        .groupby("feature_group", as_index=False)
        .first()
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    print(best[cols].to_string(index=False))

    print(
        "\nNOTE: all groups are evaluated on the exact same complete-case 48 "
        "functions. This is still exploratory k-selection; nested validation "
        "is required before freezing a final feature group/k."
    )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--static-metrics", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument(
        "--groups",
        default=",".join(GROUPS),
    )
    p.add_argument("--k-values", default="5,6,7,8,9,10,11,12")
    p.add_argument("--seeds", default="11,23,37,41,53")
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--top-n", type=int, default=25)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
