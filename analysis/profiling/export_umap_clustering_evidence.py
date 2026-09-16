#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler, RobustScaler

try:
    from umap import UMAP
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency 'umap-learn'. Install it inside the analysis venv with:\n"
        "  .venv-analysis/bin/python -m pip install umap-learn"
    ) from exc


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

FAMILY_ORDER = ["go", "python", "node", "twin", "synthetic"]
FAMILY_MARKERS = {
    "go": "o",
    "python": "s",
    "node": "^",
    "twin": "D",
    "synthetic": "P",
}
PREFERENCE_MARKERS = {
    "x86-preferred": "o",
    "arm-preferred": "^",
    "architecture-independent": "s",
}
NOISE_MARKER = "x"

SELECTED_LABELS_BASE = {
    "amd_faster",
    "arm_faster",
    "filehandle",
    "thread",
    "linpack",
    "primenumber",
    "twin-primenumber",
    "twin-readmemory",
    "twin-chacha20",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_column(
    fields: Iterable[str],
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:
    fields = list(fields)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    if required:
        raise KeyError(f"None of {list(candidates)!r} found in {fields!r}")
    return None


def same_float(a: float | str, b: float | str, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def threshold_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def classify_family(function_name: str) -> str:
    if function_name in {"amd_faster", "arm_faster"}:
        return "synthetic"
    if function_name.startswith("twin-"):
        return "twin"
    if function_name.endswith("-py"):
        return "python"
    if function_name.endswith("-node"):
        return "node"
    return "go"


def resolve_run_dir(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    for candidate in (root / p, Path.cwd() / p):
        if candidate.exists():
            return candidate.resolve()
    return (root / p).resolve()


def resolve_kmeans_run_dir(
    kmeans_dir: Path,
    feature_set: str,
    scaler: str,
    k: int,
) -> Path:
    candidate = kmeans_dir / "runs" / f"{feature_set}__{scaler}__k{k}"
    if (candidate / "assignments.csv").exists():
        return candidate.resolve()

    matches = [
        p.parent
        for p in (kmeans_dir / "runs").rglob("assignments.csv")
        if feature_set in p.parent.name
        and f"__{scaler}__" in p.parent.name
        and p.parent.name.endswith(f"__k{k}")
    ]
    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Unable to locate frozen K-Means run. "
        f"Expected {candidate / 'assignments.csv'}; matches={matches}"
    )


def load_assignments(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No assignments in {path}")

    function_col = pick_column(rows[0].keys(), ["function_name", "function", "name"])
    cluster_col = pick_column(
        rows[0].keys(),
        ["cluster_label", "cluster", "cluster_id", "label"],
    )
    return {
        row[function_col]: int(float(row[cluster_col]))
        for row in rows
    }


def load_profiles(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No profiles in {path}")

    function_col = pick_column(rows[0].keys(), ["function_name", "function", "name"])
    for feature in PAPER5_FEATURES:
        if feature not in rows[0]:
            raise KeyError(f"Missing {feature!r} in {path}")

    rows.sort(key=lambda r: r[function_col])
    names = [r[function_col] for r in rows]
    matrix = np.asarray(
        [[float(r[f]) for f in PAPER5_FEATURES] for r in rows],
        dtype=float,
    )
    return names, matrix


def load_preferences(path: Path) -> dict[str, dict[str, object]]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No preferences in {path}")

    fields = rows[0].keys()
    function_col = pick_column(fields, ["function_name", "function", "name"])
    preference_col = pick_column(
        fields,
        ["architecture_preference", "preference", "label"],
    )
    delta_col = pick_column(
        fields,
        [
            "delta_percent",
            "architecture_delta_percent",
            "arm_vs_x86_delta_percent",
        ],
    )

    return {
        row[function_col]: {
            "ground_truth_label": row[preference_col],
            "architecture_delta_percent": float(row[delta_col]),
        }
        for row in rows
    }


def selected_label_names(rows: list[dict]) -> set[str]:
    names = set(SELECTED_LABELS_BASE)
    names.update(
        row["function_name"]
        for row in rows
        if row["function_name"].startswith("twin-")
    )
    names.update(
        row["function_name"]
        for row in rows
        if bool(row["is_noise"])
    )
    return names


def default_colors(labels: list) -> dict:
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def make_umap_rows(
    names: list[str],
    matrix: np.ndarray,
    assignments: dict[str, int],
    preferences: dict[str, dict[str, object]],
    *,
    algorithm: str,
    scaler_name: str,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[list[dict], UMAP]:
    if scaler_name == "minmax":
        scaled = MinMaxScaler().fit_transform(matrix)
    elif scaler_name == "robust":
        scaled = RobustScaler().fit_transform(matrix)
    else:
        raise ValueError(f"Unsupported scaler {scaler_name!r}")

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        init="spectral",
    )
    coordinates = reducer.fit_transform(scaled)

    rows: list[dict] = []
    for i, name in enumerate(names):
        cluster = int(assignments[name])
        pref = preferences[name]
        rows.append(
            {
                "function_name": name,
                "function_family": classify_family(name),
                "algorithm": algorithm,
                "umap1": float(coordinates[i, 0]),
                "umap2": float(coordinates[i, 1]),
                "cluster": cluster,
                "is_noise": algorithm == "dbscan" and cluster == -1,
                "ground_truth_label": pref["ground_truth_label"],
                "architecture_delta_percent": pref["architecture_delta_percent"],
            }
        )
    return rows, reducer


def save_figure(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def setup_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)


def external_legend(ax, handles, title: str, anchor=(1.02, 1.0), fontsize=8):
    legend = ax.legend(
        handles=handles,
        title=title,
        loc="upper left",
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        fontsize=fontsize,
        title_fontsize=fontsize,
        frameon=True,
    )
    ax.add_artist(legend)
    return legend


def plot_clusters(rows: list[dict], title: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8))
    clusters = sorted({int(r["cluster"]) for r in rows})
    colors = default_colors(clusters)

    for cluster in clusters:
        group = [r for r in rows if int(r["cluster"]) == cluster]
        ax.scatter(
            [r["umap1"] for r in group],
            [r["umap2"] for r in group],
            marker=NOISE_MARKER if cluster == -1 else "o",
            s=70,
            alpha=0.85,
            c=[colors[cluster]],
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )

    setup_axes(ax, title)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    save_figure(fig, stem)


def plot_ground_truth(rows: list[dict], title: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8))
    colors = default_colors(PREFERENCE_ORDER)

    for pref in PREFERENCE_ORDER:
        group = [r for r in rows if r["ground_truth_label"] == pref]
        ax.scatter(
            [r["umap1"] for r in group],
            [r["umap2"] for r in group],
            marker="o",
            s=70,
            alpha=0.85,
            c=[colors[pref]],
            label=pref,
        )

    setup_axes(ax, title)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    save_figure(fig, stem)


def plot_cluster_family(rows: list[dict], title: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = sorted({int(r["cluster"]) for r in rows})
    colors = default_colors(clusters)

    for row in rows:
        cluster = int(row["cluster"])
        marker = NOISE_MARKER if bool(row["is_noise"]) else FAMILY_MARKERS[row["function_family"]]
        ax.scatter(
            row["umap1"],
            row["umap2"],
            marker=marker,
            s=76,
            alpha=0.88,
            c=[colors[cluster]],
        )

    cluster_handles = [
        Line2D(
            [0], [0],
            marker=NOISE_MARKER if cluster == -1 else "o",
            linestyle="",
            markerfacecolor=colors[cluster],
            markeredgecolor=colors[cluster],
            markersize=8,
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )
        for cluster in clusters
    ]
    families_present = [
        family
        for family in FAMILY_ORDER
        if any(r["function_family"] == family for r in rows)
    ]
    family_handles = [
        Line2D(
            [0], [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
            label=family,
        )
        for family in families_present
    ]

    setup_axes(ax, title)
    external_legend(ax, cluster_handles, "cluster", anchor=(1.02, 1.0))
    ax.legend(
        handles=family_handles,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.54),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    save_figure(fig, stem)


def plot_cluster_preference(rows: list[dict], title: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = sorted({int(r["cluster"]) for r in rows})
    colors = default_colors(clusters)

    for row in rows:
        cluster = int(row["cluster"])
        marker = NOISE_MARKER if bool(row["is_noise"]) else PREFERENCE_MARKERS[row["ground_truth_label"]]
        ax.scatter(
            row["umap1"],
            row["umap2"],
            marker=marker,
            s=76,
            alpha=0.88,
            c=[colors[cluster]],
        )

    cluster_handles = [
        Line2D(
            [0], [0],
            marker=NOISE_MARKER if cluster == -1 else "o",
            linestyle="",
            markerfacecolor=colors[cluster],
            markeredgecolor=colors[cluster],
            markersize=8,
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )
        for cluster in clusters
    ]
    preference_handles = [
        Line2D(
            [0], [0],
            marker=PREFERENCE_MARKERS[pref],
            linestyle="",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
            label=pref,
        )
        for pref in PREFERENCE_ORDER
    ]

    setup_axes(ax, title)
    external_legend(ax, cluster_handles, "cluster", anchor=(1.02, 1.0))
    ax.legend(
        handles=preference_handles,
        title="ground truth",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.54),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    save_figure(fig, stem)


def plot_selected_labels(rows: list[dict], title: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = sorted({int(r["cluster"]) for r in rows})
    colors = default_colors(clusters)
    selected = selected_label_names(rows)

    for row in rows:
        cluster = int(row["cluster"])
        marker = NOISE_MARKER if bool(row["is_noise"]) else FAMILY_MARKERS[row["function_family"]]
        ax.scatter(
            row["umap1"],
            row["umap2"],
            marker=marker,
            s=78,
            alpha=0.9,
            c=[colors[cluster]],
        )
        if row["function_name"] in selected:
            ax.annotate(
                row["function_name"],
                (row["umap1"], row["umap2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    cluster_handles = [
        Line2D(
            [0], [0],
            marker=NOISE_MARKER if cluster == -1 else "o",
            linestyle="",
            markerfacecolor=colors[cluster],
            markeredgecolor=colors[cluster],
            markersize=8,
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )
        for cluster in clusters
    ]
    families_present = [
        family
        for family in FAMILY_ORDER
        if any(r["function_family"] == family for r in rows)
    ]
    family_handles = [
        Line2D(
            [0], [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
            label=family,
        )
        for family in families_present
    ]

    setup_axes(ax, title)
    external_legend(ax, cluster_handles, "cluster", anchor=(1.02, 1.0))
    ax.legend(
        handles=family_handles,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.54),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    save_figure(fig, stem)


def export_algorithm(
    *,
    algorithm: str,
    key: str,
    title: str,
    names: list[str],
    matrix: np.ndarray,
    assignments: dict[str, int],
    preferences: dict[str, dict[str, object]],
    scaler_name: str,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    output_dir: Path,
    figures_dir: Path,
) -> list[dict]:
    rows, _ = make_umap_rows(
        names,
        matrix,
        assignments,
        preferences,
        algorithm=algorithm,
        scaler_name=scaler_name,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    write_csv(output_dir / f"{key}-umap-coordinates.csv", rows)

    plot_clusters(
        rows,
        title + " — clusters",
        figures_dir / f"{key}-umap-clusters",
    )
    plot_ground_truth(
        rows,
        title + " — architectural preference",
        figures_dir / f"{key}-umap-ground-truth",
    )
    plot_cluster_family(
        rows,
        title + " — cluster × family",
        figures_dir / f"{key}-umap-cluster-family",
    )
    plot_cluster_preference(
        rows,
        title + " — cluster × architectural preference",
        figures_dir / f"{key}-umap-cluster-preference",
    )
    plot_selected_labels(
        rows,
        title + " — selected labels",
        figures_dir / f"{key}-umap-selected-labels",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    profiles_path = root / "resource/x86/function-profiles-median.csv"
    preferences_path = root / "ground_truth" / f"preferences-{threshold_slug(args.threshold)}.csv"
    kmeans_dir = root / "kmeans-scaler-minmax"
    dbscan_dir = root / "dbscan-scaler-robust"

    if not profiles_path.exists():
        raise FileNotFoundError(profiles_path)
    if not preferences_path.exists():
        raise FileNotFoundError(preferences_path)

    km_internal = read_csv(kmeans_dir / "kmeans-internal-summary.csv")
    db_internal = read_csv(dbscan_dir / "dbscan-internal-summary.csv")

    km_matches = [
        row
        for row in km_internal
        if row.get("feature_set") == "paper6_no_framework_runtime_ms"
        and row.get("scaler") == "minmax"
        and int(float(row["k"])) == 5
    ]
    db_matches = [
        row
        for row in db_internal
        if row.get("feature_set") == "paper6_no_framework_runtime_ms"
        and row.get("scaler") == "robust"
        and row.get("metric") == "cosine"
        and int(float(row["min_samples"])) == 4
        and same_float(row["eps_quantile"], 0.80)
    ]
    if len(km_matches) != 1 or len(db_matches) != 1:
        raise RuntimeError(
            f"Expected one frozen row each; kmeans={len(km_matches)}, dbscan={len(db_matches)}"
        )

    km_run = resolve_kmeans_run_dir(
        kmeans_dir,
        "paper6_no_framework_runtime_ms",
        "minmax",
        5,
    )
    db_row = db_matches[0]
    if not db_row.get("run_dir"):
        raise KeyError("Frozen DBSCAN summary row does not contain run_dir")
    db_run = resolve_run_dir(root, db_row["run_dir"])

    km_assignments_path = km_run / "assignments.csv"
    db_assignments_path = db_run / "assignments.csv"

    names, matrix = load_profiles(profiles_path)
    preferences = load_preferences(preferences_path)
    km_assignments = load_assignments(km_assignments_path)
    db_assignments = load_assignments(db_assignments_path)

    expected = set(names)
    for label, values in [
        ("preferences", preferences),
        ("kmeans", km_assignments),
        ("dbscan", db_assignments),
    ]:
        if set(values) != expected:
            raise RuntimeError(
                f"{label} names differ from profiles: "
                f"missing={sorted(expected - set(values))}, "
                f"extra={sorted(set(values) - expected)}"
            )

    km_key = "kmeans-paper5-minmax-k5"
    db_key = "dbscan-paper5-robust-cosine-ms4-q80"

    km_rows = export_algorithm(
        algorithm="kmeans",
        key=km_key,
        title=(
            "K-Means PAPER-5 / MinMax / K=5 — UMAP "
            f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"
        ),
        names=names,
        matrix=matrix,
        assignments=km_assignments,
        preferences=preferences,
        scaler_name="minmax",
        metric="euclidean",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        output_dir=out,
        figures_dir=figures,
    )

    db_rows = export_algorithm(
        algorithm="dbscan",
        key=db_key,
        title=(
            "DBSCAN PAPER-5 / Robust / cosine / ms=4 / q=.80 — UMAP "
            f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})"
        ),
        names=names,
        matrix=matrix,
        assignments=db_assignments,
        preferences=preferences,
        scaler_name="robust",
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        output_dir=out,
        figures_dir=figures,
    )

    config_rows = [
        {
            "algorithm": "kmeans",
            "feature_set": "PAPER-5",
            "scaler": "minmax",
            "umap_metric": "euclidean",
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "ground_truth_threshold_percent": args.threshold,
            "cluster_configuration": "K=5",
        },
        {
            "algorithm": "dbscan",
            "feature_set": "PAPER-5",
            "scaler": "robust",
            "umap_metric": "cosine",
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "ground_truth_threshold_percent": args.threshold,
            "cluster_configuration": "min_samples=4; eps_quantile=0.80",
        },
    ]
    write_csv(out / "umap-visualization-config.csv", config_rows)

    manifest = {
        "schema_version": 1,
        "purpose": (
            "Complementary nonlinear 2-D visualization of the frozen clustering. "
            "UMAP is visualization-only and never changes cluster assignments."
        ),
        "feature_set": "PAPER-5",
        "ground_truth_threshold_percent": args.threshold,
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "kmeans_metric": "euclidean",
            "dbscan_metric": "cosine",
            "umap_learn_version": importlib.metadata.version("umap-learn"),
        },
        "marker_policy": {
            "family_markers": FAMILY_MARKERS,
            "preference_markers": PREFERENCE_MARKERS,
            "noise_marker": NOISE_MARKER,
            "selected_labels": sorted(selected_label_names(db_rows) | selected_label_names(km_rows)),
        },
        "inputs": {
            "profiles": str(profiles_path),
            "profiles_sha256": sha256(profiles_path),
            "preferences": str(preferences_path),
            "preferences_sha256": sha256(preferences_path),
            "kmeans_assignments": str(km_assignments_path),
            "kmeans_assignments_sha256": sha256(km_assignments_path),
            "dbscan_assignments": str(db_assignments_path),
            "dbscan_assignments_sha256": sha256(db_assignments_path),
        },
        "outputs": {
            "kmeans_coordinates": f"{km_key}-umap-coordinates.csv",
            "dbscan_coordinates": f"{db_key}-umap-coordinates.csv",
            "config_csv": "umap-visualization-config.csv",
            "figures": "figures/*.png and figures/*.svg",
        },
    }
    (out / "umap-clustering-evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"functions={len(names)} "
        f"threshold={args.threshold:g}% "
        f"n_neighbors={args.n_neighbors} "
        f"min_dist={args.min_dist} "
        f"random_state={args.random_state} "
        f"output={out}"
    )
    print(
        "K-Means UMAP: MinMaxScaler + euclidean; "
        "DBSCAN UMAP: RobustScaler + cosine; assignments unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
