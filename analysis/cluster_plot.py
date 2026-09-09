#!/usr/bin/env python3
"""
Genera una visualizzazione "Clustering-style" per ogni configurazione di uno
sweep di clustering Serverledge.

Per ogni run riuscita:
- legge assignments.csv;
- usa le 6 feature già preprocessate presenti nell'assegnamento;
- proietta in 2D con PCA SOLO per visualizzazione;
- colora i punti per cluster;
- disegna il centroide del cluster con una X;
- disegna un'ellisse di dispersione (2 deviazioni standard) quando possibile;
- salva un PNG per configurazione;
- salva un indice CSV con metriche e percorso dell'immagine.

Non usa pandas.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=float("nan")):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def ellipse_from_points(ax, points, center, n_std=2.0):
    """Disegna un'ellisse basata sulla covarianza dei punti 2D."""
    if len(points) < 2:
        return

    cov = np.cov(points, rowvar=False)

    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    if vals[0] <= 0:
        return

    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    # Ellisse solo contorno. Nessun colore specifico imposto:
    # eredita un colore della cycle di matplotlib.
    patch = Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=float(angle),
        fill=False,
        linewidth=1.8,
    )
    ax.add_patch(patch)


def plot_run(run_dir: Path, out_path: Path, title: str):
    assignments = run_dir / "assignments.csv"
    if not assignments.exists():
        raise FileNotFoundError(assignments)

    rows = read_csv(assignments)
    if not rows:
        raise ValueError(f"{assignments}: nessuna riga")

    X = np.asarray(
        [[float(r[f]) for f in FEATURES] for r in rows],
        dtype=float,
    )

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    labels = np.asarray([int(r["cluster_label"]) for r in rows], dtype=int)
    noise = np.asarray(
        [str(r.get("is_noise", "")).lower() == "true" for r in rows],
        dtype=bool,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    valid_labels = sorted(set(labels[~noise]))

    for cluster in valid_labels:
        mask = (labels == cluster) & (~noise)
        pts = Z[mask]

        scatter = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=52,
            label=f"Cluster {cluster}",
        )

        center = pts.mean(axis=0)
        cluster_color = scatter.get_facecolors()[0]

        ax.scatter(
            [center[0]],
            [center[1]],
            marker="X",
            s=180,
            linewidths=1.5,
            color=cluster_color,
            edgecolors="black",
        )

        if len(pts) >= 2:
            cov = np.cov(pts, rowvar=False)
            if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 0.0)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                if vals[0] > 0:
                    angle = math.degrees(
                        math.atan2(vecs[1, 0], vecs[0, 0])
                    )
                    width, height = 4 * np.sqrt(vals)
                    ellipse = Ellipse(
                        xy=center,
                        width=float(width),
                        height=float(height),
                        angle=float(angle),
                        fill=False,
                        linewidth=2.0,
                        edgecolor=cluster_color,
                    )
                    ax.add_patch(ellipse)

    if np.any(noise):
        pts = Z[noise]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="x",
            s=60,
            label="Noise",
        )

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var.)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "pc1_explained_variance_ratio": float(evr[0]),
        "pc2_explained_variance_ratio": float(evr[1]),
        "pc12_explained_variance_ratio": float(evr[0] + evr[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    summary_path = Path(args.sweep_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(summary_path)
    if not rows:
        raise SystemExit("sweep-summary.csv vuoto")

    index_rows = []

    for row in rows:
        if row.get("status") != "ok":
            continue

        config_id = row["configuration_id"]
        raw_run_dir = Path(row["run_dir"])
        if not raw_run_dir.is_absolute():
            # Normalmente run_dir è assoluto; fallback rispetto alla directory sweep.
            raw_run_dir = summary_path.parent / raw_run_dir

        run_dir = raw_run_dir.resolve()
        assignments = run_dir / "assignments.csv"

        if not assignments.exists():
            print(f"[skip] {config_id}: assignments.csv non trovato")
            continue

        out_png = output_dir / f"{config_id}_clustering.png"

        scaler = row.get("scaler", "")
        algorithm = row.get("algorithm", "")
        k = row.get("k", "")
        eps = row.get("eps", "")
        min_samples = row.get("min_samples", "")

        if algorithm == "kmeans":
            subtitle = f"{algorithm} | scaler={scaler} | K={k}"
        else:
            subtitle = (
                f"{algorithm} | scaler={scaler} | "
                f"eps={eps} | min_samples={min_samples}"
            )

        meta = plot_run(
            run_dir,
            out_png,
            title=f"Clustering - {subtitle}",
        )

        index_rows.append({
            "configuration_id": config_id,
            "scaler": scaler,
            "algorithm": algorithm,
            "k": k,
            "eps": eps,
            "min_samples": min_samples,
            "cluster_count": row.get("cluster_count", ""),
            "noise_count": row.get("noise_count", ""),
            "coverage": row.get("coverage", ""),
            "silhouette": row.get("silhouette", ""),
            "overall_purity": row.get("overall_purity", ""),
            "adjusted_rand_index": row.get("adjusted_rand_index", ""),
            "normalized_mutual_information":
                row.get("normalized_mutual_information", ""),
            **meta,
            "image": str(out_png),
        })

        print(f"[ok] {config_id} -> {out_png}")

    index_path = output_dir / "clustering-images-index.csv"

    if index_rows:
        with index_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(index_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(index_rows)

    metadata_path = output_dir / "visualization-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "projection": "PCA 2D",
                "projection_usage": "visualization_only",
                "ellipse": "2 standard deviations in PCA plane",
                "centroid_marker": "X",
                "source_sweep_summary": str(summary_path),
                "generated_configurations": len(index_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Immagini generate: {len(index_rows)}")
    print(f"Indice: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
