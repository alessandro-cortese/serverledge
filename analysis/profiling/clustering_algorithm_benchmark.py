#!/usr/bin/env python3
"""
Standard clustering algorithm benchmark for the Serverledge architecture-preference task.

Goal
----
Investigate whether the remaining errors are mainly caused by vanilla K-Means
geometry / cluster imbalance, while keeping the same LOFO evaluation protocol.

Algorithms:
  * KMeans
  * BisectingKMeans with strategies:
      - biggest_inertia
      - largest_cluster
  * GaussianMixture (diag covariance)
  * DBSCAN with fold-local eps estimated from k-distance quantiles

Important:
  * The held-out target is never used to fit scaler/clusterer.
  * Cluster labels are mapped to architecture preference by UNIQUE majority
    among training members. Ties => abstain.
  * DBSCAN target assignment uses the nearest training core point and requires
    distance <= fold eps, matching the previous LOFO approach.
  * Language is intentionally excluded here because the previous ablation
    showed no material improvement; this isolates the clustering algorithm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
try:
    from sklearn.cluster import BisectingKMeans
except ImportError:
    BisectingKMeans = None

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

LABELS = ["x86-preferred", "architecture-independent", "arm-preferred"]
DIRECTIONAL = {"x86-preferred", "arm-preferred"}


def threshold_slug(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")


def load_data(profiles: Path, preferences_dir: Path, threshold: float) -> pd.DataFrame:
    prof = pd.read_csv(profiles)
    pref = pd.read_csv(
        preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
    )
    out = prof.merge(
        pref[["function_name", "architecture_preference"]],
        on="function_name",
        validate="one_to_one",
    )
    return out.sort_values("function_name").reset_index(drop=True)


def representation(frame: pd.DataFrame, variant: str) -> np.ndarray:
    pf = frame["page_faults_delta"].astype(float).to_numpy()
    cpu = frame["utilized_cpus"].astype(float).to_numpy()
    user = frame["cpu_user_delta_ms"].astype(float).to_numpy()
    kernel = frame["cpu_kernel_delta_ms"].astype(float).to_numpy()
    total = user + kernel

    log_pf = np.log1p(np.clip(pf, 0, None))
    log_total = np.log1p(np.clip(total, 0, None))
    kernel_share = kernel / np.maximum(total, 1e-12)

    if variant == "behavioral4":
        return np.column_stack([log_pf, cpu, log_total, kernel_share])
    if variant == "behavioral4_no_page_faults":
        return np.column_stack([cpu, log_total, kernel_share])
    raise ValueError(f"unknown feature variant: {variant}")


def unique_majority(labels: pd.Series) -> str:
    counts = Counter(labels.tolist())
    top = max(counts.values())
    winners = [label for label in LABELS if counts.get(label, 0) == top]
    return winners[0] if len(winners) == 1 else "abstain"


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


def make_partition_model(algorithm: str, k: int, seed: int, n_init: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    if algorithm == "bisect_inertia":
        if BisectingKMeans is None:
            raise RuntimeError("BisectingKMeans is unavailable in this scikit-learn")
        return BisectingKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=max(1, min(n_init, 20)),
            random_state=seed,
            bisecting_strategy="biggest_inertia",
        )
    if algorithm == "bisect_largest":
        if BisectingKMeans is None:
            raise RuntimeError("BisectingKMeans is unavailable in this scikit-learn")
        return BisectingKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=max(1, min(n_init, 20)),
            random_state=seed,
            bisecting_strategy="largest_cluster",
        )
    raise ValueError(algorithm)


def predict_partition_lofo(
    df: pd.DataFrame,
    variant: str,
    algorithm: str,
    k: int,
    seed: int,
    n_init: int,
):
    y_true, y_pred = [], []
    cluster_sizes = []

    for idx in range(len(df)):
        train = df.drop(index=idx)
        target = df.iloc[[idx]]

        scaler = RobustScaler()
        x_train = scaler.fit_transform(representation(train, variant))
        x_target = scaler.transform(representation(target, variant))

        if algorithm == "gmm_diag":
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                n_init=max(1, min(n_init, 10)),
                reg_covar=1e-6,
                random_state=seed,
            )
            train_clusters = model.fit_predict(x_train)
            target_cluster = int(model.predict(x_target)[0])
        else:
            model = make_partition_model(algorithm, k, seed, n_init)
            train_clusters = model.fit_predict(x_train)
            target_cluster = int(model.predict(x_target)[0])

        members = np.flatnonzero(train_clusters == target_cluster)
        pred = unique_majority(train.iloc[members]["architecture_preference"])

        y_true.append(str(target.iloc[0]["architecture_preference"]))
        y_pred.append(pred)
        cluster_sizes.append(len(members))

    y = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    result = classification_metrics(y, p)
    result["mean_target_cluster_size"] = float(np.mean(cluster_sizes))
    result["median_target_cluster_size"] = float(np.median(cluster_sizes))
    result["large_cluster_target_share"] = float(np.mean(np.asarray(cluster_sizes) >= 20))
    return result


def dbscan_eps(x: np.ndarray, min_samples: int, quantile: float, metric: str) -> float:
    # DBSCAN min_samples includes the sample itself.
    nn = NearestNeighbors(
        n_neighbors=min(min_samples, len(x)),
        metric=metric,
    ).fit(x)
    distances, _ = nn.kneighbors(x)
    kth = distances[:, -1]
    return float(np.quantile(kth, quantile))


def predict_dbscan_lofo(
    df: pd.DataFrame,
    variant: str,
    min_samples: int,
    quantile: float,
    metric: str,
):
    y_true, y_pred = [], []
    cluster_sizes = []
    eps_values = []

    for idx in range(len(df)):
        train = df.drop(index=idx)
        target = df.iloc[[idx]]

        scaler = RobustScaler()
        x_train = scaler.fit_transform(representation(train, variant))
        x_target = scaler.transform(representation(target, variant))

        eps = dbscan_eps(x_train, min_samples, quantile, metric)
        eps_values.append(eps)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        train_clusters = model.fit_predict(x_train)

        core_indices = getattr(model, "core_sample_indices_", np.array([], dtype=int))
        if len(core_indices) == 0:
            pred = "abstain"
            cluster_sizes.append(0)
        else:
            x_core = x_train[core_indices]
            d = pairwise_distances(x_target, x_core, metric=metric)[0]
            nearest_pos = int(np.argmin(d))
            nearest_dist = float(d[nearest_pos])

            if nearest_dist > eps:
                pred = "abstain"
                cluster_sizes.append(0)
            else:
                training_index = int(core_indices[nearest_pos])
                target_cluster = int(train_clusters[training_index])
                members = np.flatnonzero(train_clusters == target_cluster)
                pred = unique_majority(
                    train.iloc[members]["architecture_preference"]
                )
                cluster_sizes.append(len(members))

        y_true.append(str(target.iloc[0]["architecture_preference"]))
        y_pred.append(pred)

    y = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    result = classification_metrics(y, p)
    sizes = np.asarray(cluster_sizes)
    result["mean_target_cluster_size"] = float(np.mean(sizes))
    result["median_target_cluster_size"] = float(np.median(sizes))
    result["large_cluster_target_share"] = float(np.mean(sizes >= 20))
    result["mean_fold_eps"] = float(np.mean(eps_values))
    return result


def full_fit_stats(
    df: pd.DataFrame,
    variant: str,
    algorithm: str,
    k: int | None,
    seed: int,
    n_init: int,
    *,
    min_samples: int | None = None,
    quantile: float | None = None,
    metric: str | None = None,
):
    scaler = RobustScaler()
    x = scaler.fit_transform(representation(df, variant))

    if algorithm == "gmm_diag":
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=max(1, min(n_init, 10)),
            reg_covar=1e-6,
            random_state=seed,
        )
        clusters = model.fit_predict(x)
    elif algorithm == "dbscan":
        eps = dbscan_eps(x, min_samples, quantile, metric)
        clusters = DBSCAN(
            eps=eps, min_samples=min_samples, metric=metric
        ).fit_predict(x)
    else:
        model = make_partition_model(algorithm, k, seed, n_init)
        clusters = model.fit_predict(x)

    valid = clusters != -1
    unique = sorted(set(clusters[valid]))
    sizes = pd.Series(clusters[valid]).value_counts() if np.any(valid) else pd.Series(dtype=int)

    if len(unique) >= 2 and np.sum(valid) > len(unique):
        sil = float(silhouette_score(x[valid], clusters[valid]))
        db = float(davies_bouldin_score(x[valid], clusters[valid]))
        ch = float(calinski_harabasz_score(x[valid], clusters[valid]))
    else:
        sil = db = ch = np.nan

    if np.any(valid):
        correct = 0
        labels = df["architecture_preference"].to_numpy()
        for c in unique:
            vals = labels[clusters == c]
            correct += max(Counter(vals.tolist()).values())
        purity_clustered = correct / int(np.sum(valid))
        purity_strict = correct / len(df)
    else:
        purity_clustered = purity_strict = 0.0

    return {
        "full_cluster_count": len(unique),
        "full_noise_count": int(np.sum(~valid)),
        "full_cluster_coverage": float(np.mean(valid)),
        "full_purity_clustered": purity_clustered,
        "full_purity_strict": purity_strict,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "min_cluster_size": int(sizes.min()) if len(sizes) else 0,
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "max_cluster_share": float(sizes.max() / len(df)) if len(sizes) else 0.0,
        "singleton_count": int(np.sum(sizes == 1)) if len(sizes) else 0,
    }


def pairwise_seed_ari(
    df: pd.DataFrame,
    variant: str,
    algorithm: str,
    k: int,
    seeds: list[int],
    n_init: int,
) -> float:
    scaler = RobustScaler()
    x = scaler.fit_transform(representation(df, variant))
    partitions = []
    for seed in seeds:
        if algorithm == "gmm_diag":
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                n_init=max(1, min(n_init, 10)),
                reg_covar=1e-6,
                random_state=seed,
            )
            partitions.append(model.fit_predict(x))
        else:
            model = make_partition_model(algorithm, k, seed, n_init)
            partitions.append(model.fit_predict(x))
    aris = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            aris.append(adjusted_rand_score(partitions[i], partitions[j]))
    return float(np.mean(aris)) if aris else 1.0


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(
        args.profiles.resolve(),
        args.preferences_dir.resolve(),
        args.threshold,
    )

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    dbscan_min_samples = [int(x) for x in args.dbscan_min_samples.split(",") if x.strip()]
    dbscan_quantiles = [float(x) for x in args.dbscan_quantiles.split(",") if x.strip()]
    dbscan_metrics = [x.strip() for x in args.dbscan_metrics.split(",") if x.strip()]

    rows = []
    total = (
        len(variants) * len([a for a in algorithms if a != "dbscan"]) * len(ks) * len(seeds)
        + (len(variants) * len(dbscan_min_samples) * len(dbscan_quantiles) * len(dbscan_metrics)
           if "dbscan" in algorithms else 0)
    )
    done = 0

    for variant in variants:
        for algorithm in algorithms:
            if algorithm == "dbscan":
                for min_samples in dbscan_min_samples:
                    for q in dbscan_quantiles:
                        for metric in dbscan_metrics:
                            result = predict_dbscan_lofo(
                                df, variant, min_samples, q, metric
                            )
                            full = full_fit_stats(
                                df, variant, "dbscan", None, 0, args.n_init,
                                min_samples=min_samples, quantile=q, metric=metric
                            )
                            rows.append({
                                "feature_variant": variant,
                                "algorithm": "dbscan",
                                "k": np.nan,
                                "seed": np.nan,
                                "dbscan_min_samples": min_samples,
                                "dbscan_eps_quantile": q,
                                "dbscan_metric": metric,
                                **result,
                                **full,
                            })
                            done += 1
                            if done % 25 == 0 or done == total:
                                print(f"progress={done}/{total}")
                continue

            if algorithm == "bisect_largest" and BisectingKMeans is None:
                print("SKIP bisect_largest: BisectingKMeans unavailable")
                continue
            if algorithm == "bisect_inertia" and BisectingKMeans is None:
                print("SKIP bisect_inertia: BisectingKMeans unavailable")
                continue

            for k in ks:
                stability = pairwise_seed_ari(
                    df, variant, algorithm, k, seeds, args.n_init
                )
                for seed in seeds:
                    result = predict_partition_lofo(
                        df, variant, algorithm, k, seed, args.n_init
                    )
                    full = full_fit_stats(
                        df, variant, algorithm, k, seed, args.n_init
                    )
                    rows.append({
                        "feature_variant": variant,
                        "algorithm": algorithm,
                        "k": k,
                        "seed": seed,
                        "dbscan_min_samples": np.nan,
                        "dbscan_eps_quantile": np.nan,
                        "dbscan_metric": "",
                        "mean_pairwise_seed_ari": stability,
                        **result,
                        **full,
                    })
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"progress={done}/{total}")

    detail = pd.DataFrame(rows)
    detail.to_csv(out / "algorithm-benchmark-detail.csv", index=False)

    # Aggregate randomized algorithms over seeds; DBSCAN configurations stay single-row.
    non_db = detail[detail["algorithm"] != "dbscan"].copy()
    grouped = (
        non_db.groupby(["feature_variant", "algorithm", "k"], dropna=False)
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
            large_cluster_target_share_mean=("large_cluster_target_share", "mean"),
            full_purity_strict_mean=("full_purity_strict", "mean"),
            silhouette_mean=("silhouette", "mean"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            max_cluster_share_mean=("max_cluster_share", "mean"),
            singleton_count_mean=("singleton_count", "mean"),
            mean_pairwise_seed_ari=("mean_pairwise_seed_ari", "mean"),
        )
        .reset_index()
    )

    db = detail[detail["algorithm"] == "dbscan"].copy()
    if len(db):
        db_summary = db.rename(columns={
            "coverage": "coverage_mean",
            "strict_accuracy": "strict_accuracy_mean",
            "balanced_accuracy": "balanced_accuracy_mean",
            "macro_f1": "macro_f1_mean",
            "directional_accuracy": "directional_accuracy_mean",
            "x86_recall": "x86_recall_mean",
            "independent_recall": "independent_recall_mean",
            "arm_recall": "arm_recall_mean",
            "large_cluster_target_share": "large_cluster_target_share_mean",
            "full_purity_strict": "full_purity_strict_mean",
            "silhouette": "silhouette_mean",
            "davies_bouldin": "davies_bouldin_mean",
            "max_cluster_share": "max_cluster_share_mean",
            "singleton_count": "singleton_count_mean",
        })
        db_summary["seed_count"] = 1
        db_summary["coverage_std"] = 0.0
        db_summary["strict_accuracy_std"] = 0.0
        db_summary["balanced_accuracy_std"] = 0.0
        db_summary["macro_f1_std"] = 0.0
        db_summary["directional_accuracy_std"] = 0.0
        db_summary["mean_pairwise_seed_ari"] = 1.0

        keep = list(grouped.columns) + [
            c for c in [
                "dbscan_min_samples", "dbscan_eps_quantile", "dbscan_metric"
            ] if c not in grouped.columns
        ]
        for c in grouped.columns:
            if c not in db_summary:
                db_summary[c] = np.nan
        grouped["dbscan_min_samples"] = np.nan
        grouped["dbscan_eps_quantile"] = np.nan
        grouped["dbscan_metric"] = ""
        db_summary = db_summary[grouped.columns]
        summary = pd.concat([grouped, db_summary], ignore_index=True)
    else:
        summary = grouped

    summary = summary.sort_values(
        [
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "directional_accuracy_mean",
            "strict_accuracy_mean",
            "coverage_mean",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "rank_exploratory", np.arange(1, len(summary) + 1))
    summary.to_csv(out / "algorithm-benchmark-summary.csv", index=False)

    cols = [
        "rank_exploratory",
        "feature_variant",
        "algorithm",
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
    print("\nTOP STANDARD-ALGORITHM CONFIGURATIONS:")
    print(summary[cols].head(args.top_n).to_string(index=False))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument(
        "--variants",
        default="behavioral4,behavioral4_no_page_faults",
    )
    p.add_argument(
        "--algorithms",
        default="kmeans,bisect_inertia,bisect_largest,gmm_diag,dbscan",
    )
    p.add_argument("--k-values", default="6,7,8,9,10,11,12")
    p.add_argument("--seeds", default="11,23,37,41,53")
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--dbscan-min-samples", default="3,4,5")
    p.add_argument("--dbscan-quantiles", default="0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--dbscan-metrics", default="euclidean,cosine")
    p.add_argument("--top-n", type=int, default=30)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
