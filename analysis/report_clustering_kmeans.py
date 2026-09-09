#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]

METRICS = [
    ("silhouette", "Silhouette"),
    ("overall_purity", "Overall purity"),
    ("adjusted_rand_index", "Adjusted Rand Index"),
    ("normalized_mutual_information", "Normalized Mutual Information"),
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_metric_plots(sweep_rows, out_dir):
    by_scaler = defaultdict(list)
    for row in sweep_rows:
        if row.get("status") != "ok":
            continue
        if row.get("algorithm") != "kmeans":
            continue
        by_scaler[row["scaler"]].append(row)

    for metric, ylabel in METRICS:
        fig, ax = plt.subplots()
        for scaler, rows in sorted(by_scaler.items()):
            rows = sorted(rows, key=lambda r: int(r["k"]))
            xs = [int(r["k"]) for r in rows]
            ys = [float(r[metric]) for r in rows]
            ax.plot(xs, ys, marker="o", label=scaler)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(f"KMeans - {ylabel} per K e scaler")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Scaler")
        fig.tight_layout()
        fig.savefig(out_dir / f"kmeans_{metric}_vs_k.png", dpi=180)
        plt.close(fig)


def save_cluster_composition(cluster_rows, out_dir):
    rows = sorted(cluster_rows, key=lambda r: int(r["cluster_label"]))
    labels = [f"Cluster {r['cluster_label']}" for r in rows]
    x = np.arange(len(rows))
    values = [
        [int(r["x86_preferred_count"]) for r in rows],
        [int(r["arm_preferred_count"]) for r in rows],
        [int(r["architecture_independent_count"]) for r in rows],
    ]
    names = ["x86-preferred", "arm-preferred", "architecture-independent"]

    fig, ax = plt.subplots()
    bottom = np.zeros(len(rows))
    for name, vals in zip(names, values):
        ax.bar(x, vals, bottom=bottom, label=name)
        bottom += np.array(vals)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Numero di funzioni")
    ax.set_title("Composizione architetturale dei cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "selected_cluster_composition.png", dpi=180)
    plt.close(fig)


def save_pca_plots(assign_rows, matched_rows, out_dir):
    pref_by_fn = {r["function_name"]: r["architecture_preference"] for r in matched_rows}
    X = np.array([[float(r[f]) for f in FEATURES] for r in assign_rows], dtype=float)
    names = [r["function_name"] for r in assign_rows]
    clusters = np.array([int(r["cluster_label"]) for r in assign_rows])
    prefs = [pref_by_fn[n] for n in names]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title("PCA 2D del feature space - colore = cluster")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f"Cluster {c}" for c in sorted(set(clusters))], title="Cluster")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_pca_by_cluster.png", dpi=180)
    plt.close(fig)

    pref_labels = sorted(set(prefs))
    pref_to_int = {p: i for i, p in enumerate(pref_labels)}
    pref_values = np.array([pref_to_int[p] for p in prefs])

    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=pref_values)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title("PCA 2D del feature space - colore = ground truth")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, pref_labels, title="Preferenza")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_pca_by_preference.png", dpi=180)
    plt.close(fig)

    pca_meta = {
        "features": FEATURES,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "note": "PCA usata solo per visualizzazione; il clustering rimane nello spazio originale a 6 dimensioni.",
    }
    (out_dir / "selected_pca_metadata.json").write_text(
        json.dumps(pca_meta, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--selected-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = Path(args.selected_run_dir)

    sweep_rows = read_csv(args.sweep_summary)
    cluster_rows = read_csv(selected / "cluster-summary.csv")
    assign_rows = read_csv(selected / "assignments.csv")
    matched_rows = read_csv(selected / "matched.csv")

    save_metric_plots(sweep_rows, out_dir)
    save_cluster_composition(cluster_rows, out_dir)
    save_pca_plots(assign_rows, matched_rows, out_dir)

    print(f"Report grafico salvato in: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
