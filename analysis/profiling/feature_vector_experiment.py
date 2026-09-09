#!/usr/bin/env python3
"""
Esperimento controllato sui feature vector per il clustering Serverledge.

Obiettivo metodologico:
- NON modifica preprocess.FEATURE_NAMES né il contratto runtime/transfer.
- Usa gli stessi FunctionProfile già raccolti sulla reference architecture.
- Varia solo il sottoinsieme di feature, mantenendo KMeans e le altre
  configurazioni sotto controllo.
- Salva risultati, metriche, composizione cluster e grafici in una directory
  condivisibile.

Modalità consigliata:
1) screening: MinMax + K=4, per confrontare solo i feature vector;
2) sweep: sui vettori selezionati, variare scaler e K.

Dipendenze: numpy, scikit-learn, matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


ALL_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]

# Varianti motivate, non LOFO cieco.
FEATURE_VECTORS = {
    # Baseline originale.
    "V0_all6": list(ALL_FEATURES),

    # Rimuove la metrica di memoria libera del nodo, che è node-scoped.
    "V1_no_free_memory": [
        "page_faults_delta",
        "utilized_cpus",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
        "framework_runtime_ms",
    ],

    # Sensitivity test sulla metrica più sparsa / a range più esteso
    # nella campagna corrente.
    "V2_no_page_faults": [
        "utilized_cpus",
        "free_memory_mb",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
        "framework_runtime_ms",
    ],

    # Vettore compatto focalizzato su CPU/runtime direttamente legati
    # all'esecuzione.
    "V3_cpu_runtime": [
        "utilized_cpus",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
        "framework_runtime_ms",
    ],
}

PREFERENCE_LABELS = (
    "x86-preferred",
    "arm-preferred",
    "architecture-independent",
)

SUMMARY_HEADER = [
    "experiment_id",
    "vector_id",
    "feature_count",
    "features",
    "scaler",
    "k",
    "n_init",
    "random_state",
    "sample_count",
    "cluster_count",
    "min_cluster_size",
    "max_cluster_size",
    "singleton_count",
    "cluster_size_distribution",
    "silhouette",
    "inertia",
    "overall_purity",
    "homogeneity",
    "completeness",
    "v_measure",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "preference_threshold_percent",
    "input_profiles_sha256",
    "preferences_sha256",
    "run_dir",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def require_columns(fieldnames: list[str], required: list[str], source: str) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"{source}: colonne mancanti: {missing}")


def fit_scaler(name: str, matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    if name == "none":
        return matrix.copy(), {"name": "none"}

    if name == "standard":
        scaler = StandardScaler()
    elif name == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    elif name == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=False)
    else:
        raise ValueError(f"scaler non supportato: {name}")

    transformed = scaler.fit_transform(matrix)

    state = {"name": name}
    if name == "standard":
        state.update({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        })
    elif name == "robust":
        state.update({
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
            "quantile_range": [25.0, 75.0],
        })
    elif name == "minmax":
        state.update({
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
        })

    return transformed, state


def load_preferences(path: Path) -> tuple[dict[str, dict[str, str]], float]:
    rows, fields = read_csv(path)
    require_columns(
        fields,
        [
            "function_name",
            "architecture_preference",
            "arm_vs_x86_delta_percent",
            "threshold_percent",
        ],
        str(path),
    )

    result: dict[str, dict[str, str]] = {}
    thresholds = set()

    for row in rows:
        name = row["function_name"].strip()
        pref = row["architecture_preference"].strip()
        if pref not in PREFERENCE_LABELS:
            raise ValueError(f"{path}: preferenza non supportata: {pref!r}")
        if name in result:
            raise ValueError(f"{path}: funzione duplicata: {name}")
        result[name] = row
        thresholds.add(float(row["threshold_percent"]))

    if len(thresholds) != 1:
        raise ValueError(f"{path}: threshold_percent non uniforme")

    return result, thresholds.pop()


def purity_score(true_labels: list[str], cluster_labels: np.ndarray) -> float:
    correct = 0
    for cluster in sorted(set(int(v) for v in cluster_labels)):
        indexes = np.where(cluster_labels == cluster)[0]
        counts = Counter(true_labels[i] for i in indexes)
        correct += max(counts.values())
    return correct / len(true_labels)


def save_assignments(
    path: Path,
    profile_rows: list[dict[str, str]],
    transformed: np.ndarray,
    labels: np.ndarray,
    features: list[str],
    vector_id: str,
    scaler: str,
    k: int,
) -> None:
    metadata = [
        "vector_id",
        "scaler",
        "k",
        "function_name",
        "machine_tag",
        "configured_cpus",
        "configured_memory_mb",
        "sample_count",
        "cluster_label",
    ]
    header = metadata + features

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for idx, row in enumerate(profile_rows):
            out = {
                "vector_id": vector_id,
                "scaler": scaler,
                "k": k,
                "function_name": row["function_name"],
                "machine_tag": row.get("machine_tag", ""),
                "configured_cpus": row.get("configured_cpus", ""),
                "configured_memory_mb": row.get("configured_memory_mb", ""),
                "sample_count": row.get("sample_count", ""),
                "cluster_label": int(labels[idx]),
            }
            for j, feature in enumerate(features):
                out[feature] = format(float(transformed[idx, j]), ".17g")
            writer.writerow(out)


def save_matched_and_summary(
    run_dir: Path,
    profile_rows: list[dict[str, str]],
    preferences: dict[str, dict[str, str]],
    labels: np.ndarray,
) -> tuple[list[str], list[dict]]:
    matched_header = [
        "function_name",
        "cluster_label",
        "architecture_preference",
        "arm_vs_x86_delta_percent",
        "threshold_percent",
    ]

    matched_rows = []
    true_labels = []

    for idx, row in enumerate(profile_rows):
        name = row["function_name"]
        pref = preferences[name]
        true_labels.append(pref["architecture_preference"])
        matched_rows.append({
            "function_name": name,
            "cluster_label": int(labels[idx]),
            "architecture_preference": pref["architecture_preference"],
            "arm_vs_x86_delta_percent": pref["arm_vs_x86_delta_percent"],
            "threshold_percent": pref["threshold_percent"],
        })

    with (run_dir / "matched.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matched_header)
        writer.writeheader()
        writer.writerows(matched_rows)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in matched_rows:
        grouped[int(row["cluster_label"])].append(row)

    summary_rows = []
    for cluster in sorted(grouped):
        members = grouped[cluster]
        counts = Counter(r["architecture_preference"] for r in members)
        top_count = max(counts.values())
        top_labels = sorted(k for k, v in counts.items() if v == top_count)

        summary_rows.append({
            "cluster_label": cluster,
            "cluster_size": len(members),
            "x86_preferred_count": counts["x86-preferred"],
            "arm_preferred_count": counts["arm-preferred"],
            "architecture_independent_count": counts["architecture-independent"],
            "plurality_preference": "|".join(top_labels),
            "cluster_purity": top_count / len(members),
        })

    summary_header = [
        "cluster_label",
        "cluster_size",
        "x86_preferred_count",
        "arm_preferred_count",
        "architecture_independent_count",
        "plurality_preference",
        "cluster_purity",
    ]

    with (run_dir / "cluster-summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        writer.writerows(summary_rows)

    return true_labels, summary_rows


def plot_clean_pca(
    run_dir: Path,
    transformed: np.ndarray,
    labels: np.ndarray,
    true_labels: list[str],
    title_suffix: str,
) -> dict:
    if transformed.shape[1] < 2:
        return {}

    pca = PCA(n_components=2)
    z = pca.fit_transform(transformed)

    # Cluster plot.
    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster in sorted(set(int(v) for v in labels)):
        mask = labels == cluster
        pts = z[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=72,
            alpha=0.82,
            label=f"Cluster {cluster} (n={len(pts)})",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title(f"KMeans — cluster — {title_suffix}")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(run_dir / "pca-by-cluster.png", dpi=180)
    plt.close(fig)

    # Ground truth plot on the exact same PCA coordinates.
    fig, ax = plt.subplots(figsize=(9, 7))
    true_array = np.asarray(true_labels, dtype=object)
    for pref in PREFERENCE_LABELS:
        mask = true_array == pref
        pts = z[mask]
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=72,
            alpha=0.82,
            label=f"{pref} (n={len(pts)})",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title(f"Ground truth — stessa PCA — {title_suffix}")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(run_dir / "pca-by-preference.png", dpi=180)
    plt.close(fig)

    return {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_pc1_pc2":
            float(pca.explained_variance_ratio_.sum()),
    }


def plot_composition(run_dir: Path, cluster_summary: list[dict], title_suffix: str) -> None:
    labels = [
        f"C{row['cluster_label']} (n={row['cluster_size']})"
        for row in cluster_summary
    ]
    sizes = np.asarray([row["cluster_size"] for row in cluster_summary], dtype=float)

    counts_by_pref = {
        "x86-preferred": np.asarray(
            [row["x86_preferred_count"] for row in cluster_summary],
            dtype=float,
        ),
        "arm-preferred": np.asarray(
            [row["arm_preferred_count"] for row in cluster_summary],
            dtype=float,
        ),
        "architecture-independent": np.asarray(
            [row["architecture_independent_count"] for row in cluster_summary],
            dtype=float,
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(cluster_summary))
    for pref in PREFERENCE_LABELS:
        counts = counts_by_pref[pref]
        pct = np.divide(
            counts * 100.0,
            sizes,
            out=np.zeros_like(counts),
            where=sizes != 0,
        )
        ax.bar(labels, pct, bottom=bottom, label=pref)
        bottom += pct

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Composizione (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Composizione architetturale — {title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(run_dir / "cluster-composition-percent.png", dpi=180)
    plt.close(fig)


def run_one(
    experiment_id: str,
    output_dir: Path,
    profile_rows: list[dict[str, str]],
    preferences: dict[str, dict[str, str]],
    threshold: float,
    input_hash: str,
    preference_hash: str,
    vector_id: str,
    features: list[str],
    scaler_name: str,
    k: int,
    n_init: int,
    random_state: int,
) -> dict:
    raw = np.asarray(
        [[float(row[f]) for f in features] for row in profile_rows],
        dtype=np.float64,
    )

    if not np.isfinite(raw).all():
        raise ValueError(f"{vector_id}: feature non finite")

    transformed, scaler_state = fit_scaler(scaler_name, raw)

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(transformed)

    configuration_id = f"{vector_id}__{scaler_name}__k{k}"
    run_dir = output_dir / "runs" / configuration_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_assignments(
        run_dir / "assignments.csv",
        profile_rows,
        transformed,
        labels,
        features,
        vector_id,
        scaler_name,
        k,
    )

    true_labels, cluster_summary = save_matched_and_summary(
        run_dir,
        profile_rows,
        preferences,
        labels,
    )

    encoded_true = [
        PREFERENCE_LABELS.index(label)
        for label in true_labels
    ]

    metrics = {
        "silhouette": float(silhouette_score(transformed, labels)),
        "inertia": float(model.inertia_),
        "overall_purity": float(purity_score(true_labels, labels)),
        "homogeneity": float(homogeneity_score(encoded_true, labels)),
        "completeness": float(completeness_score(encoded_true, labels)),
        "v_measure": float(v_measure_score(encoded_true, labels)),
        "adjusted_rand_index": float(adjusted_rand_score(encoded_true, labels)),
        "normalized_mutual_information":
            float(normalized_mutual_info_score(encoded_true, labels)),
    }

    pca_meta = plot_clean_pca(
        run_dir,
        transformed,
        labels,
        true_labels,
        f"{vector_id}, {scaler_name}, K={k}",
    )
    plot_composition(
        run_dir,
        cluster_summary,
        f"{vector_id}, {scaler_name}, K={k}",
    )

    document = {
        "experiment_id": experiment_id,
        "configuration_id": configuration_id,
        "vector_id": vector_id,
        "feature_names": features,
        "feature_count": len(features),
        "scaler": scaler_name,
        "scaler_state": scaler_state,
        "algorithm": "kmeans",
        "k": k,
        "n_init": n_init,
        "random_state": random_state,
        "sample_count": len(profile_rows),
        "metrics": metrics,
        "ground_truth_threshold_percent": threshold,
        "input_profiles_sha256": input_hash,
        "preferences_sha256": preference_hash,
        "sklearn_version": sklearn.__version__,
        **pca_meta,
    }

    (run_dir / "evaluation.json").write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )

    cluster_sizes = sorted(
        (int(np.sum(labels == cluster)) for cluster in set(int(v) for v in labels)),
        reverse=True,
    )

    return {
        "experiment_id": experiment_id,
        "vector_id": vector_id,
        "feature_count": len(features),
        "features": "|".join(features),
        "scaler": scaler_name,
        "k": k,
        "n_init": n_init,
        "random_state": random_state,
        "sample_count": len(profile_rows),
        "cluster_count": len(cluster_sizes),
        "min_cluster_size": min(cluster_sizes),
        "max_cluster_size": max(cluster_sizes),
        "singleton_count": sum(size == 1 for size in cluster_sizes),
        "cluster_size_distribution": "|".join(str(size) for size in cluster_sizes),
        **metrics,
        "preference_threshold_percent": threshold,
        "input_profiles_sha256": input_hash,
        "preferences_sha256": preference_hash,
        "run_dir": str(run_dir.resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--preferences", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--experiment-id", required=True)

    ap.add_argument(
        "--vector",
        action="append",
        choices=sorted(FEATURE_VECTORS),
        default=[],
        help="Ripetibile. Se omesso usa V0,V1,V2,V3.",
    )
    ap.add_argument(
        "--scaler",
        action="append",
        choices=["none", "standard", "robust", "minmax"],
        default=[],
        help="Ripetibile. Default: minmax.",
    )
    ap.add_argument(
        "--k",
        action="append",
        type=int,
        default=[],
        help="Ripetibile. Default: 4.",
    )
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()

    profiles_path = Path(args.profiles).expanduser().resolve()
    preferences_path = Path(args.preferences).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    profile_rows, profile_fields = read_csv(profiles_path)
    require_columns(
        profile_fields,
        [
            "function_name",
            "machine_tag",
            "configured_cpus",
            "configured_memory_mb",
            "sample_count",
            *ALL_FEATURES,
        ],
        str(profiles_path),
    )

    if not profile_rows:
        raise SystemExit("FunctionProfile CSV vuoto")

    names = [r["function_name"] for r in profile_rows]
    if len(names) != len(set(names)):
        raise SystemExit(
            "Il file reference deve contenere una sola riga per funzione."
        )

    preferences, threshold = load_preferences(preferences_path)

    missing_preferences = sorted(set(names) - set(preferences))
    if missing_preferences:
        raise SystemExit(
            "Ground truth mancante per: "
            + ", ".join(missing_preferences)
        )

    vectors = args.vector or list(FEATURE_VECTORS)
    scalers = args.scaler or ["minmax"]
    ks = args.k or [4]

    for k in ks:
        if k < 2 or k >= len(profile_rows):
            raise SystemExit(
                f"K={k} non valido per {len(profile_rows)} funzioni"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    input_hash = sha256_file(profiles_path)
    preference_hash = sha256_file(preferences_path)

    summary_rows = []

    for vector_id in vectors:
        features = FEATURE_VECTORS[vector_id]
        for scaler_name in scalers:
            for k in ks:
                row = run_one(
                    args.experiment_id,
                    output_dir,
                    profile_rows,
                    preferences,
                    threshold,
                    input_hash,
                    preference_hash,
                    vector_id,
                    features,
                    scaler_name,
                    k,
                    args.n_init,
                    args.random_state,
                )
                summary_rows.append(row)

                print(
                    f"[ok] vector={vector_id} "
                    f"features={len(features)} "
                    f"scaler={scaler_name} "
                    f"k={k} "
                    f"silhouette={row['silhouette']:.6f} "
                    f"purity={row['overall_purity']:.6f} "
                    f"ari={row['adjusted_rand_index']:.6f} "
                    f"nmi={row['normalized_mutual_information']:.6f}"
                )

    summary_path = output_dir / "feature-vector-summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "experiment_id": args.experiment_id,
        "profiles": str(profiles_path),
        "profiles_sha256": input_hash,
        "preferences": str(preferences_path),
        "preferences_sha256": preference_hash,
        "preference_threshold_percent": threshold,
        "algorithm": "kmeans",
        "vectors": {
            key: FEATURE_VECTORS[key]
            for key in vectors
        },
        "scalers": scalers,
        "k_values": ks,
        "n_init": args.n_init,
        "random_state": args.random_state,
        "sample_count": len(profile_rows),
        "sklearn_version": sklearn.__version__,
        "methodological_note": (
            "Feature-vector sensitivity experiment. "
            "The production profiling/transfer feature contract is not modified."
        ),
    }

    (output_dir / "feature-vector-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Summary:  {summary_path}")
    print(f"Manifest: {output_dir / 'feature-vector-manifest.json'}")


if __name__ == "__main__":
    main()
