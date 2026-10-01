#!/usr/bin/env python3
"""
Final ground-truth threshold sensitivity for the frozen Serverledge clustering.

Frozen clustering configuration:
- reference architecture: AMD64/x86
- aggregation: median
- feature set: PAPER-5
- scaler: MinMaxScaler
- clusterer: K-Means, K=5
- n_init=50, random_state=42 (already used to produce assignments)

Important:
Changing tau changes ONLY the post-hoc architectural-preference labels.
The K-Means assignments are never recomputed from the ground truth.

Outputs:
- one CSV with metrics across thresholds
- one CSV with per-cluster composition across thresholds
- one CSV with ground-truth class distributions
- PCA and UMAP coordinates
- PCA ground-truth figures for each threshold
- UMAP ground-truth figures for each threshold
- cluster-composition figures for each threshold
- threshold-sensitivity summary figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import MinMaxScaler


PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

PREFERENCE_ORDER = [
    "x86-preferred",
    "arm-preferred",
    "architecture-independent",
]


def threshold_slug(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def preference_path(root: Path, threshold: float) -> Path:
    return root / "ground_truth" / f"preferences-{threshold_slug(threshold)}.csv"


def ensure_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path}: missing columns {missing}; available={list(df.columns)}"
        )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def purity_score(labels: pd.Series, clusters: pd.Series) -> float:
    tmp = pd.DataFrame({"label": labels, "cluster": clusters})
    correct = 0
    for _, group in tmp.groupby("cluster"):
        correct += int(group["label"].value_counts().iloc[0])
    return correct / len(tmp)


def load_inputs(root: Path):
    profiles_path = root / "resource" / "x86" / "function-profiles-median.csv"
    assignments_path = (
        root
        / "kmeans-scaler-minmax"
        / "runs"
        / "paper6_no_framework_runtime_ms__minmax__k5"
        / "assignments.csv"
    )

    profiles = pd.read_csv(profiles_path)
    assignments = pd.read_csv(assignments_path)

    ensure_columns(
        profiles,
        ["function_name", *PAPER5_FEATURES],
        profiles_path,
    )
    ensure_columns(
        assignments,
        ["function_name", "cluster_label"],
        assignments_path,
    )

    if profiles["function_name"].duplicated().any():
        raise RuntimeError(f"{profiles_path}: duplicate function_name")
    if assignments["function_name"].duplicated().any():
        raise RuntimeError(f"{assignments_path}: duplicate function_name")

    names_profiles = set(profiles["function_name"])
    names_assignments = set(assignments["function_name"])
    if names_profiles != names_assignments:
        raise RuntimeError(
            "Function-name mismatch between profiles and assignments: "
            f"missing_in_assignments={sorted(names_profiles - names_assignments)}, "
            f"extra_in_assignments={sorted(names_assignments - names_profiles)}"
        )

    # Keep profile order as the canonical order.
    data = profiles[["function_name", *PAPER5_FEATURES]].merge(
        assignments[["function_name", "cluster_label"]],
        on="function_name",
        how="left",
        validate="one_to_one",
    )
    return profiles_path, assignments_path, data


def compute_visual_coordinates(
    data: pd.DataFrame,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
):
    matrix = data[PAPER5_FEATURES].astype(float).to_numpy()
    scaled = MinMaxScaler().fit_transform(matrix)

    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(scaled)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    umap_xy = reducer.fit_transform(scaled)

    coords = data[["function_name", "cluster_label"]].copy()
    coords["pc1"] = pca_xy[:, 0]
    coords["pc2"] = pca_xy[:, 1]
    coords["umap1"] = umap_xy[:, 0]
    coords["umap2"] = umap_xy[:, 1]

    return coords, pca.explained_variance_ratio_


def plot_embedding(
    merged: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for label in PREFERENCE_ORDER:
        part = merged[merged["architecture_preference"] == label]
        ax.scatter(
            part[x_col],
            part[y_col],
            label=f"{label} (n={len(part)})",
            s=42,
            alpha=0.85,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    save_figure(fig, stem)


def plot_composition(
    merged: pd.DataFrame,
    threshold: float,
    stem: Path,
) -> None:
    counts = (
        merged.groupby(["cluster_label", "architecture_preference"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PREFERENCE_ORDER, fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(counts), dtype=float)
    x = np.arange(len(counts))

    for label in PREFERENCE_ORDER:
        vals = counts[label].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals

    ax.set_title(
        f"K-Means PAPER-5 / MinMax / K=5 — composition at τ={threshold:g}%"
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Functions")
    ax.set_xticks(x, [f"C{int(c)}" for c in counts.index])
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    save_figure(fig, stem)


def plot_metric_sensitivity(metrics: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for column, label in [
        ("purity_gain_over_majority_baseline", "Purity gain"),
        ("homogeneity", "Homogeneity"),
        ("adjusted_rand_index", "ARI"),
        ("normalized_mutual_information", "NMI"),
    ]:
        ax.plot(
            metrics["threshold_percent"],
            metrics[column],
            marker="o",
            label=label,
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_title(
        "K-Means PAPER-5 / MinMax / K=5 — ground-truth sensitivity"
    )
    ax.set_xlabel("Architectural-preference threshold τ (%)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.2)
    ax.legend()
    save_figure(
        fig,
        out / "figures" / "kmeans-paper5-minmax-k5-threshold-metrics",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        metrics["threshold_percent"],
        metrics["overall_purity"],
        marker="o",
        label="Purity",
    )
    ax.plot(
        metrics["threshold_percent"],
        metrics["majority_ground_truth_share"],
        marker="o",
        label="Majority baseline",
    )
    ax.set_title(
        "K-Means PAPER-5 / MinMax / K=5 — purity vs majority baseline"
    )
    ax.set_xlabel("Architectural-preference threshold τ (%)")
    ax.set_ylabel("Share")
    ax.grid(True, alpha=0.2)
    ax.legend()
    save_figure(
        fig,
        out / "figures" / "kmeans-paper5-minmax-k5-purity-vs-baseline",
    )


def plot_ground_truth_distribution(distribution: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for col, label in [
        ("x86_preferred_count", "x86-preferred"),
        ("arm_preferred_count", "arm-preferred"),
        ("architecture_independent_count", "architecture-independent"),
    ]:
        ax.plot(
            distribution["threshold_percent"],
            distribution[col],
            marker="o",
            label=label,
        )

    ax.set_title("Ground-truth distribution across τ")
    ax.set_xlabel("Architectural-preference threshold τ (%)")
    ax.set_ylabel("Functions")
    ax.grid(True, alpha=0.2)
    ax.legend()
    save_figure(
        fig,
        out / "figures" / "ground-truth-distribution-thresholds",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[2.5, 5, 10, 15, 20, 25],
    )
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output_dir.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)

    profiles_path, assignments_path, data = load_inputs(root)

    coords, explained = compute_visual_coordinates(
        data,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )
    coords.to_csv(
        out / "kmeans-paper5-minmax-k5-pca-umap-coordinates.csv",
        index=False,
    )

    metric_rows = []
    composition_rows = []
    distribution_rows = []

    expected_names = set(data["function_name"])

    for threshold in args.thresholds:
        pref_path = preference_path(root, threshold)
        if not pref_path.is_file():
            raise FileNotFoundError(pref_path)

        pref = pd.read_csv(pref_path)
        ensure_columns(
            pref,
            [
                "function_name",
                "threshold_percent",
                "architecture_preference",
            ],
            pref_path,
        )

        actual_names = set(pref["function_name"])
        if actual_names != expected_names:
            raise RuntimeError(
                f"{pref_path}: function-name mismatch; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )

        if not np.allclose(
            pref["threshold_percent"].astype(float).to_numpy(),
            threshold,
        ):
            raise RuntimeError(
                f"{pref_path}: threshold_percent does not match requested {threshold}"
            )

        merged = coords.merge(
            pref[
                [
                    "function_name",
                    "architecture_preference",
                    "threshold_percent",
                ]
            ],
            on="function_name",
            how="inner",
            validate="one_to_one",
        )

        gt = merged["architecture_preference"]
        clusters = merged["cluster_label"]

        purity = purity_score(gt, clusters)
        majority_share = gt.value_counts(normalize=True).max()

        counts = gt.value_counts()
        distribution_rows.append(
            {
                "threshold_percent": threshold,
                "x86_preferred_count": int(counts.get("x86-preferred", 0)),
                "arm_preferred_count": int(counts.get("arm-preferred", 0)),
                "architecture_independent_count": int(
                    counts.get("architecture-independent", 0)
                ),
                "majority_ground_truth_share": float(majority_share),
            }
        )

        metric_rows.append(
            {
                "threshold_percent": threshold,
                "function_count": len(merged),
                "overall_purity": float(purity),
                "majority_ground_truth_share": float(majority_share),
                "purity_gain_over_majority_baseline": float(
                    purity - majority_share
                ),
                "homogeneity": float(
                    homogeneity_score(gt, clusters)
                ),
                "completeness": float(
                    completeness_score(gt, clusters)
                ),
                "v_measure": float(
                    v_measure_score(gt, clusters)
                ),
                "adjusted_rand_index": float(
                    adjusted_rand_score(gt, clusters)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(gt, clusters)
                ),
            }
        )

        comp = (
            merged.groupby(
                ["cluster_label", "architecture_preference"]
            )
            .size()
            .unstack(fill_value=0)
            .reindex(columns=PREFERENCE_ORDER, fill_value=0)
            .sort_index()
        )

        for cluster_label, row in comp.iterrows():
            composition_rows.append(
                {
                    "threshold_percent": threshold,
                    "cluster_label": int(cluster_label),
                    "cluster_size": int(row.sum()),
                    "x86_preferred_count": int(row["x86-preferred"]),
                    "arm_preferred_count": int(row["arm-preferred"]),
                    "architecture_independent_count": int(
                        row["architecture-independent"]
                    ),
                }
            )

        slug = threshold_slug(threshold)

        plot_embedding(
            merged,
            "pc1",
            "pc2",
            (
                "K-Means PAPER-5 / MinMax / K=5 — "
                f"architectural preference, τ={threshold:g}%"
            ),
            f"PC1 ({explained[0] * 100:.2f}% variance)",
            f"PC2 ({explained[1] * 100:.2f}% variance)",
            out
            / "figures"
            / f"kmeans-paper5-minmax-k5-pca-ground-truth-t{slug}",
        )

        plot_embedding(
            merged,
            "umap1",
            "umap2",
            (
                "K-Means PAPER-5 / MinMax / K=5 — "
                f"UMAP architectural preference, τ={threshold:g}%"
            ),
            "UMAP-1",
            "UMAP-2",
            out
            / "figures"
            / f"kmeans-paper5-minmax-k5-umap-ground-truth-t{slug}",
        )

        plot_composition(
            merged,
            threshold,
            out
            / "figures"
            / f"kmeans-paper5-minmax-k5-cluster-composition-t{slug}",
        )

    metrics = pd.DataFrame(metric_rows).sort_values("threshold_percent")
    composition = pd.DataFrame(composition_rows).sort_values(
        ["threshold_percent", "cluster_label"]
    )
    distribution = pd.DataFrame(distribution_rows).sort_values(
        "threshold_percent"
    )

    metrics.to_csv(
        out / "kmeans-paper5-minmax-k5-threshold-metrics.csv",
        index=False,
    )
    composition.to_csv(
        out / "kmeans-paper5-minmax-k5-cluster-composition-all-thresholds.csv",
        index=False,
    )
    distribution.to_csv(
        out / "ground-truth-distribution.csv",
        index=False,
    )

    plot_metric_sensitivity(metrics, out)
    plot_ground_truth_distribution(distribution, out)

    manifest = {
        "schema_version": 1,
        "purpose": (
            "Post-hoc architectural-preference threshold sensitivity "
            "for the frozen K-Means clustering."
        ),
        "clustering_is_refit_per_threshold": False,
        "reference_architecture": "amd64",
        "feature_set": {
            "name": "PAPER-5",
            "features": PAPER5_FEATURES,
        },
        "clustering": {
            "algorithm": "kmeans",
            "scaler": "minmax",
            "k": 5,
            "assignment_metric": "euclidean",
        },
        "visualization": {
            "pca_usage": "visualization only",
            "pca_explained_variance_ratio": explained.tolist(),
            "umap_usage": "visualization only",
            "umap_n_neighbors": args.n_neighbors,
            "umap_min_dist": args.min_dist,
            "umap_metric": "euclidean",
            "umap_random_state": args.random_state,
        },
        "thresholds_percent": args.thresholds,
        "inputs": {
            "profiles": str(profiles_path),
            "assignments": str(assignments_path),
            "preferences": [
                str(preference_path(root, t))
                for t in args.thresholds
            ],
        },
    }

    (out / "threshold-sensitivity-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("K-MEANS PAPER-5 / MINMAX / K=5 — GROUND-TRUTH THRESHOLD SENSITIVITY")
    print("=" * 118)

    display_cols = [
        "threshold_percent",
        "overall_purity",
        "majority_ground_truth_share",
        "purity_gain_over_majority_baseline",
        "homogeneity",
        "completeness",
        "adjusted_rand_index",
        "normalized_mutual_information",
    ]

    print(
        metrics[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "PCA explained variance: "
        f"PC1={explained[0] * 100:.2f}% "
        f"PC2={explained[1] * 100:.2f}% "
        f"total={(explained[0] + explained[1]) * 100:.2f}%"
    )
    print(f"output={out}")


if __name__ == "__main__":
    main()
