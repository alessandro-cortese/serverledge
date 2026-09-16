#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize

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
GROUND_TRUTH_THRESHOLDS = [2.5, 5.0, 10.0, 15.0, 20.0, 25.0]
FAMILY_ORDER = ["go", "python", "node", "twin", "synthetic"]
FAMILY_MARKERS = {
    "go": "o",
    "python": "s",
    "node": "^",
    "twin": "D",
    "synthetic": "P",
}
SELECTED_LABELS_BASE = {
    "amd_faster",
    "arm_faster",
    "filehandle",
    "thread",
    "linpack",
    "primenumber",
    "twin-primenumber",
    "twin-chacha20",
    "twin-readmemory",
}
PREFERENCE_COLORS = {
    "x86-preferred": "tab:blue",
    "arm-preferred": "tab:orange",
    "architecture-independent": "tab:green",
}
NOISE_MARKER = "x"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_column(fields: Iterable[str], candidates: Iterable[str], required=True):
    fields = list(fields)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    if required:
        raise KeyError(f"None of {list(candidates)!r} found in {fields!r}")
    return None


def get_float(row, candidates, default=None):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    return default


def same_float(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def resolve_run_dir(root: Path, value: str):
    p = Path(value)
    if p.is_absolute():
        return p
    for candidate in (root / p, Path.cwd() / p):
        if candidate.exists():
            return candidate.resolve()
    return (root / p).resolve()


def resolve_kmeans_run_dir(kmeans_dir: Path, feature_set: str, scaler: str, k: int) -> Path:
    candidate = kmeans_dir / "runs" / f"{feature_set}__{scaler}__k{k}"
    assignments = candidate / "assignments.csv"
    if assignments.exists():
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
        "Unable to locate the frozen K-Means run directory. "
        f"Expected {assignments}; fallback matches={matches}"
    )


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


def default_selected_labels(rows):
    selected = set(SELECTED_LABELS_BASE)
    selected.update(r["function_name"] for r in rows if r.get("is_noise"))
    selected.update(
        r["function_name"]
        for r in rows
        if r["function_name"].startswith("twin-")
    )
    return selected


def load_assignments(path: Path):
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No assignments in {path}")
    function_col = pick_column(rows[0].keys(), ["function_name", "function", "name"])
    cluster_col = pick_column(rows[0].keys(), ["cluster_label", "cluster", "cluster_id", "label"])
    return {r[function_col]: int(float(r[cluster_col])) for r in rows}


def load_preferences(path: Path):
    rows = read_csv(path)
    fields = rows[0].keys()
    function_col = pick_column(fields, ["function_name", "function", "name"])
    preference_col = pick_column(fields, ["architecture_preference", "preference", "label"])
    delta_col = pick_column(fields, ["delta_percent", "architecture_delta_percent", "arm_vs_x86_delta_percent"])
    x86_col = pick_column(fields, ["x86_ms", "x86_median_ms", "amd64_ms", "x86_duration_ms"], False)
    arm_col = pick_column(fields, ["arm_ms", "arm_median_ms", "arm64_ms", "arm_duration_ms"], False)

    out = {}
    for row in rows:
        name = row[function_col]
        out[name] = {
            "ground_truth_label": row[preference_col],
            "architecture_delta_percent": float(row[delta_col]),
            "x86_duration_ms": float(row[x86_col]) if x86_col and row.get(x86_col) else "",
            "arm_duration_ms": float(row[arm_col]) if arm_col and row.get(arm_col) else "",
        }
    return out


def load_profiles(path: Path):
    rows = read_csv(path)
    function_col = pick_column(rows[0].keys(), ["function_name", "function", "name"])
    for feature in PAPER5_FEATURES:
        if feature not in rows[0]:
            raise KeyError(f"Missing {feature} in {path}")
    rows.sort(key=lambda r: r[function_col])
    names = [r[function_col] for r in rows]
    matrix = np.array([[float(r[f]) for f in PAPER5_FEATURES] for r in rows], dtype=float)
    return names, matrix


def cluster_summary(algorithm, assignments, preferences):
    grouped = defaultdict(list)
    for name, cluster in assignments.items():
        grouped[cluster].append(name)

    out = []
    for cluster in sorted(grouped):
        names = grouped[cluster]
        counts = Counter(preferences[n]["ground_truth_label"] for n in names)
        is_noise = algorithm == "dbscan" and cluster == -1
        if is_noise:
            majority_label, majority_share = "noise", ""
        else:
            majority_label, majority_count = max(
                ((p, counts.get(p, 0)) for p in PREFERENCE_ORDER),
                key=lambda x: (x[1], -PREFERENCE_ORDER.index(x[0])),
            )
            majority_share = majority_count / len(names)

        out.append(
            {
                "algorithm": algorithm,
                "cluster": cluster,
                "is_noise": is_noise,
                "size": len(names),
                "x86_preferred_count": counts.get("x86-preferred", 0),
                "arm_preferred_count": counts.get("arm-preferred", 0),
                "architecture_independent_count": counts.get("architecture-independent", 0),
                "majority_label": majority_label,
                "majority_share": majority_share,
                "cluster_purity": majority_share,
            }
        )
    return out


def membership_rows(algorithm, assignments, preferences, summary):
    summary_by_cluster = {int(r["cluster"]): r for r in summary}
    out = []
    for name in sorted(assignments):
        cluster = assignments[name]
        pref = preferences[name]
        cluster_info = summary_by_cluster[cluster]
        out.append(
            {
                "function_name": name,
                "function_family": classify_family(name),
                "cluster": cluster,
                "is_noise": algorithm == "dbscan" and cluster == -1,
                "ground_truth_label": pref["ground_truth_label"],
                "architecture_delta_percent": pref["architecture_delta_percent"],
                "x86_duration_ms": pref["x86_duration_ms"],
                "arm_duration_ms": pref["arm_duration_ms"],
                "cluster_majority_label": cluster_info["majority_label"],
                "cluster_majority_share": cluster_info["majority_share"],
                "cluster_size": cluster_info["size"],
            }
        )
    return out


def pca_data(names, x_scaled, assignments, preferences):
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(x_scaled)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "function_name": name,
                "function_family": classify_family(name),
                "pc1": float(coordinates[i, 0]),
                "pc2": float(coordinates[i, 1]),
                "cluster": assignments[name],
                "is_noise": assignments[name] == -1,
                "ground_truth_label": preferences[name]["ground_truth_label"],
                "architecture_delta_percent": preferences[name]["architecture_delta_percent"],
            }
        )
    return rows, pca


def family_summary(rows):
    counts = Counter(classify_family(r["function_name"]) for r in rows)
    return [
        {"function_family": family, "count": counts.get(family, 0)}
        for family in FAMILY_ORDER
        if counts.get(family, 0) > 0
    ]


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def color_map_for_clusters(rows):
    clusters = sorted({int(r["cluster"]) for r in rows})
    base = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    colors = {}
    idx = 0
    for cluster in clusters:
        if cluster == -1:
            colors[cluster] = "tab:gray"
        else:
            colors[cluster] = base[idx % len(base)]
            idx += 1
    return colors


def add_external_legend(ax, handles, title, anchor=(1.02, 1.0), fontsize=8):
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


def finalise_axes(ax, title, xlabel="PC1", ylabel="PC2"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def plot_pca_clusters(rows, title, stem):
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_colors = color_map_for_clusters(rows)
    for cluster in sorted({int(r["cluster"]) for r in rows}):
        group = [r for r in rows if int(r["cluster"]) == cluster]
        ax.scatter(
            [r["pc1"] for r in group],
            [r["pc2"] for r in group],
            c=[cluster_colors[cluster]],
            marker=NOISE_MARKER if cluster == -1 else "o",
            s=64,
            alpha=0.8,
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )
    finalise_axes(ax, title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    save(fig, stem)


def plot_pca_preference(rows, title, stem):
    fig, ax = plt.subplots(figsize=(10, 8))
    for pref in PREFERENCE_ORDER:
        group = [r for r in rows if r["ground_truth_label"] == pref]
        ax.scatter(
            [r["pc1"] for r in group],
            [r["pc2"] for r in group],
            c=PREFERENCE_COLORS[pref],
            marker="o",
            s=64,
            alpha=0.8,
            label=pref,
        )
    finalise_axes(ax, title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    save(fig, stem)


def plot_pca_cluster_family(rows, title, stem):
    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_colors = color_map_for_clusters(rows)

    for row in rows:
        cluster = int(row["cluster"])
        family = row["function_family"]
        marker = NOISE_MARKER if cluster == -1 else FAMILY_MARKERS[family]
        ax.scatter(
            row["pc1"],
            row["pc2"],
            c=[cluster_colors[cluster]],
            marker=marker,
            s=70,
            alpha=0.85,
        )

    cluster_handles = [
        Line2D([0], [0], marker=(NOISE_MARKER if c == -1 else "o"), color="w", markerfacecolor=cluster_colors[c], markeredgecolor=cluster_colors[c], markersize=8, linestyle="", label=("noise" if c == -1 else f"cluster {c}"))
        for c in sorted(cluster_colors)
    ]
    family_handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], color="black", markerfacecolor="lightgray", markeredgecolor="black", markersize=8, linestyle="", label=f)
        for f in FAMILY_ORDER
        if any(r["function_family"] == f for r in rows)
    ]

    finalise_axes(ax, title)
    add_external_legend(ax, cluster_handles, "cluster", anchor=(1.02, 1.0))
    ax.legend(
        handles=family_handles,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    save(fig, stem)


def plot_pca_selected_labels(rows, title, stem, selected_labels=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_colors = color_map_for_clusters(rows)
    selected_labels = selected_labels or default_selected_labels(rows)

    for row in rows:
        cluster = int(row["cluster"])
        family = row["function_family"]
        marker = NOISE_MARKER if cluster == -1 else FAMILY_MARKERS[family]
        ax.scatter(
            row["pc1"],
            row["pc2"],
            c=[cluster_colors[cluster]],
            marker=marker,
            s=72,
            alpha=0.88,
        )
        if row["function_name"] in selected_labels:
            ax.annotate(
                row["function_name"],
                (row["pc1"], row["pc2"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    cluster_handles = [
        Line2D([0], [0], marker=(NOISE_MARKER if c == -1 else "o"), color="w", markerfacecolor=cluster_colors[c], markeredgecolor=cluster_colors[c], markersize=8, linestyle="", label=("noise" if c == -1 else f"cluster {c}"))
        for c in sorted(cluster_colors)
    ]
    family_handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], color="black", markerfacecolor="lightgray", markeredgecolor="black", markersize=8, linestyle="", label=f)
        for f in FAMILY_ORDER
        if any(r["function_family"] == f for r in rows)
    ]

    finalise_axes(ax, title)
    add_external_legend(ax, cluster_handles, "cluster", anchor=(1.02, 1.0))
    ax.legend(
        handles=family_handles,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    save(fig, stem)


def plot_composition(summary, title, stem):
    rows = sorted(summary, key=lambda r: (bool(r["is_noise"]), int(r["cluster"])))
    labels = ["noise" if r["is_noise"] else f"C{r['cluster']}" for r in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    for pref, key in [
        ("x86-preferred", "x86_preferred_count"),
        ("arm-preferred", "arm_preferred_count"),
        ("architecture-independent", "architecture_independent_count"),
    ]:
        values = np.array([r[key] for r in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, label=pref)
        bottom += values

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Functions")
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    save(fig, stem)


def threshold_summary_rows(rows, algorithm):
    out = []
    for row in rows:
        threshold = get_float(row, ["threshold_percent"])
        purity = get_float(row, ["overall_purity_clustered", "overall_purity"])
        homogeneity = get_float(row, ["homogeneity_clustered", "homogeneity"])
        ari = get_float(row, ["adjusted_rand_index_clustered", "adjusted_rand_index"])
        nmi = get_float(row, ["normalized_mutual_information_clustered", "normalized_mutual_information"])
        coverage = get_float(row, ["coverage"], 1.0 if algorithm == "kmeans" else None)
        baseline = get_float(
            row,
            [
                "clustered_majority_ground_truth_share",
                "majority_ground_truth_share",
                "majority_baseline",
            ],
            None,
        )
        gain = get_float(
            row,
            [
                "purity_gain_over_clustered_majority_baseline",
                "purity_gain_over_majority_baseline",
                "purity_gain",
            ],
            None,
        )
        if baseline is None and purity is not None and gain is not None:
            baseline = purity - gain
        if gain is None and purity is not None and baseline is not None:
            gain = purity - baseline

        out.append(
            {
                "algorithm": algorithm,
                "threshold_percent": threshold,
                "coverage": coverage,
                "purity": purity,
                "majority_baseline": baseline,
                "purity_gain": gain,
                "homogeneity": homogeneity,
                "adjusted_rand_index": ari,
                "normalized_mutual_information": nmi,
            }
        )
    out.sort(key=lambda r: float(r["threshold_percent"]))
    return out


def plot_threshold_core(rows, title, stem):
    thresholds = [r["threshold_percent"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    series = [
        ("purity", "purity"),
        ("homogeneity", "homogeneity"),
        ("adjusted_rand_index", "ARI"),
        ("normalized_mutual_information", "NMI"),
    ]
    for key, label in series:
        values = [r[key] for r in rows]
        ax.plot(thresholds, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Ground-truth threshold τ (%)")
    ax.set_ylabel("metric value")
    ax.set_xticks(thresholds)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    save(fig, stem)


def plot_threshold_support(rows, title, stem, include_coverage=True):
    thresholds = [r["threshold_percent"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    if include_coverage:
        ax.plot(thresholds, [r["coverage"] for r in rows], marker="o", label="coverage")
    if all(r["majority_baseline"] is not None for r in rows):
        ax.plot(thresholds, [r["majority_baseline"] for r in rows], marker="o", label="majority baseline")
    if all(r["purity_gain"] is not None for r in rows):
        ax.plot(thresholds, [r["purity_gain"] for r in rows], marker="o", label="purity gain")
    ax.set_title(title)
    ax.set_xlabel("Ground-truth threshold τ (%)")
    ax.set_ylabel("metric value")
    ax.set_xticks(thresholds)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
    save(fig, stem)


def export_threshold_sensitivity(out_dir: Path, figures_dir: Path, km_dir: Path, db_dir: Path):
    km_rows = read_csv(km_dir / "kmeans-ground-truth-summary.csv")
    db_rows = read_csv(db_dir / "dbscan-ground-truth-summary.csv")

    km_selected = [
        r for r in km_rows
        if r.get("feature_set") == "paper6_no_framework_runtime_ms"
           and r.get("scaler") == "minmax"
           and int(float(r["k"])) == 5
           and any(same_float(r["threshold_percent"], t) for t in GROUND_TRUTH_THRESHOLDS)
    ]
    db_selected = [
        r for r in db_rows
        if r.get("feature_set") == "paper6_no_framework_runtime_ms"
           and r.get("scaler") == "robust"
           and r.get("metric") == "cosine"
           and int(float(r["min_samples"])) == 4
           and same_float(r["eps_quantile"], 0.80)
           and any(same_float(r["threshold_percent"], t) for t in GROUND_TRUTH_THRESHOLDS)
    ]

    if len(km_selected) != len(GROUND_TRUTH_THRESHOLDS):
        raise RuntimeError(f"Expected {len(GROUND_TRUTH_THRESHOLDS)} K-Means threshold rows, found {len(km_selected)}")
    if len(db_selected) != len(GROUND_TRUTH_THRESHOLDS):
        raise RuntimeError(f"Expected {len(GROUND_TRUTH_THRESHOLDS)} DBSCAN threshold rows, found {len(db_selected)}")

    km_export = threshold_summary_rows(km_selected, "kmeans")
    db_export = threshold_summary_rows(db_selected, "dbscan")
    write_csv(out_dir / "kmeans-threshold-sensitivity.csv", km_export)
    write_csv(out_dir / "dbscan-threshold-sensitivity.csv", db_export)

    plot_threshold_core(
        km_export,
        "K-Means PAPER-5 / MinMax / K=5 — threshold sensitivity",
        figures_dir / "kmeans-paper5-minmax-k5-threshold-sensitivity-core",
        )
    plot_threshold_support(
        km_export,
        "K-Means PAPER-5 / MinMax / K=5 — baseline/gain vs threshold",
        figures_dir / "kmeans-paper5-minmax-k5-threshold-sensitivity-support",
        include_coverage=False,
        )
    plot_threshold_core(
        db_export,
        "DBSCAN PAPER-5 / Robust / cosine / ms=4 / q=.80 — threshold sensitivity",
        figures_dir / "dbscan-paper5-robust-cosine-ms4-q80-threshold-sensitivity-core",
        )
    plot_threshold_support(
        db_export,
        "DBSCAN PAPER-5 / Robust / cosine / ms=4 / q=.80 — coverage/baseline/gain vs threshold",
        figures_dir / "dbscan-paper5-robust-cosine-ms4-q80-threshold-sensitivity-support",
        include_coverage=True,
        )

    return km_export, db_export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=15.0)
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.output_dir.resolve()
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    profiles = root / "resource/x86/function-profiles-median.csv"
    preferences = root / "ground_truth" / f"preferences-{str(args.threshold).replace('.', 'p')}.csv"
    if not preferences.exists() and same_float(args.threshold, 15.0):
        fallback = root / "ground_truth/preferences-15.csv"
        if fallback.exists():
            preferences = fallback
    kmeans_dir = root / "kmeans-scaler-minmax"
    dbscan_dir = root / "dbscan-scaler-robust"

    km_rows = read_csv(kmeans_dir / "kmeans-internal-summary.csv")
    db_rows = read_csv(dbscan_dir / "dbscan-internal-summary.csv")

    km_matches = [
        r for r in km_rows
        if r.get("feature_set") == "paper6_no_framework_runtime_ms"
           and r.get("scaler") == "minmax"
           and int(float(r["k"])) == 5
    ]
    db_matches = [
        r for r in db_rows
        if r.get("feature_set") == "paper6_no_framework_runtime_ms"
           and r.get("scaler") == "robust"
           and r.get("metric") == "cosine"
           and int(float(r["min_samples"])) == 4
           and same_float(r["eps_quantile"], 0.80)
    ]
    if len(km_matches) != 1 or len(db_matches) != 1:
        raise RuntimeError(
            f"Expected one frozen row each, got kmeans={len(km_matches)}, dbscan={len(db_matches)}"
        )

    km_row = km_matches[0]
    db_row = db_matches[0]
    km_run = resolve_kmeans_run_dir(kmeans_dir, "paper6_no_framework_runtime_ms", "minmax", 5)
    if "run_dir" not in db_row or not db_row["run_dir"]:
        raise KeyError("dbscan-internal-summary.csv does not contain run_dir for the frozen DBSCAN configuration")
    db_run = resolve_run_dir(root, db_row["run_dir"])

    km_assignments_path = km_run / "assignments.csv"
    db_assignments_path = db_run / "assignments.csv"
    km_assignments = load_assignments(km_assignments_path)
    db_assignments = load_assignments(db_assignments_path)
    pref = load_preferences(preferences)
    names, x = load_profiles(profiles)
    expected = set(names)
    for label, values in [("kmeans", km_assignments), ("dbscan", db_assignments), ("preferences", pref)]:
        if set(values) != expected:
            raise RuntimeError(
                f"{label} names differ from profiles: missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}"
            )

    km_summary = cluster_summary("kmeans", km_assignments, pref)
    db_summary = cluster_summary("dbscan", db_assignments, pref)
    km_membership = membership_rows("kmeans", km_assignments, pref, km_summary)
    db_membership = membership_rows("dbscan", db_assignments, pref, db_summary)
    write_csv(out / "kmeans-final-membership-t15.csv", km_membership)
    write_csv(out / "dbscan-final-membership-t15.csv", db_membership)
    write_csv(out / "cluster-label-summary-t15.csv", km_summary + db_summary)
    write_csv(out / "kmeans-family-summary.csv", family_summary(km_membership))
    write_csv(out / "dbscan-family-summary.csv", family_summary(db_membership))

    km_scaled = MinMaxScaler().fit_transform(x)
    km_pca_rows, km_pca_model = pca_data(names, km_scaled, km_assignments, pref)

    db_robust = RobustScaler().fit_transform(x)
    db_cosine_geometry = normalize(db_robust, norm="l2")
    db_pca_rows, db_pca_model = pca_data(names, db_cosine_geometry, db_assignments, pref)

    write_csv(out / "kmeans-pca-coordinates.csv", km_pca_rows)
    write_csv(out / "dbscan-cosine-aligned-pca-coordinates.csv", db_pca_rows)

    km_key = "kmeans-paper5-minmax-k5"
    km_title = "K-Means PAPER-5 / MinMax / K=5"
    plot_pca_clusters(km_pca_rows, km_title + " — clusters", figures_dir / f"{km_key}-pca-clusters")
    plot_pca_preference(km_pca_rows, km_title + " — architectural preference", figures_dir / f"{km_key}-pca-ground-truth")
    plot_pca_cluster_family(km_pca_rows, km_title + " — cluster × family", figures_dir / f"{km_key}-pca-cluster-family")
    plot_pca_selected_labels(km_pca_rows, km_title + " — selected labels", figures_dir / f"{km_key}-pca-selected-labels")
    plot_composition(km_summary, "K-Means PAPER-5 / MinMax / K=5 — composition at τ=15%", figures_dir / f"{km_key}-cluster-composition-t15")

    db_key = "dbscan-paper5-robust-cosine-ms4-q80"
    db_title = "DBSCAN PAPER-5 / Robust / cosine / ms=4 / q=.80 — cosine-aligned PCA"
    plot_pca_clusters(db_pca_rows, db_title + " — clusters", figures_dir / f"{db_key}-cosine-pca-clusters")
    plot_pca_preference(db_pca_rows, db_title + " — architectural preference", figures_dir / f"{db_key}-cosine-pca-ground-truth")
    plot_pca_cluster_family(db_pca_rows, db_title + " — cluster × family", figures_dir / f"{db_key}-cosine-pca-cluster-family")
    plot_pca_selected_labels(db_pca_rows, db_title + " — selected labels", figures_dir / f"{db_key}-cosine-pca-selected-labels")
    plot_composition(db_summary, "DBSCAN PAPER-5 / Robust / cosine / ms=4 / q=.80 — composition at τ=15%", figures_dir / f"{db_key}-cluster-composition-t15")

    km_threshold_rows, db_threshold_rows = export_threshold_sensitivity(out, figures_dir, kmeans_dir, dbscan_dir)

    manifest = {
        "schema_version": 2,
        "purpose": "Frozen clustering evidence before donor-selection evaluation; ground truth joined post-hoc only.",
        "reference_architecture": "amd64",
        "aggregation": "median",
        "ground_truth_threshold_percent": args.threshold,
        "ground_truth_sensitivity_thresholds": GROUND_TRUTH_THRESHOLDS,
        "feature_set": {"name": "PAPER-5", "features": PAPER5_FEATURES},
        "family_definition": "synthetic={amd_faster,arm_faster}; twin=names starting with twin-; python=names ending -py; node=names ending -node; otherwise go",
        "kmeans": {
            "scaler": "minmax",
            "k": 5,
            "run_dir": str(km_run),
            "silhouette": float(km_row["silhouette"]),
            "cluster_size_distribution": km_row["cluster_size_distribution"],
        },
        "dbscan": {
            "scaler": "robust",
            "metric": "cosine",
            "min_samples": 4,
            "eps_quantile": 0.80,
            "eps": float(db_row["eps"]),
            "run_dir": str(db_run),
            "silhouette_clustered": float(db_row["silhouette_clustered"]),
            "coverage": float(db_row["coverage"]),
            "cluster_size_distribution": db_row["cluster_size_distribution"],
        },
        "visualisation": {
            "kmeans_preprocessing": "MinMaxScaler -> PCA",
            "dbscan_preprocessing": "RobustScaler -> L2 normalize -> PCA (aligned with cosine geometry)",
            "selected_label_policy": "noise + twin-* + synthetic anchors + slow/representative functions",
            "marker_policy": "family-coded markers with external legends",
            "kmeans_explained_variance_ratio": km_pca_model.explained_variance_ratio_.tolist(),
            "dbscan_cosine_aligned_explained_variance_ratio": db_pca_model.explained_variance_ratio_.tolist(),
        },
        "threshold_sensitivity": {
            "kmeans_rows": len(km_threshold_rows),
            "dbscan_rows": len(db_threshold_rows),
        },
        "inputs": {
            "profiles": str(profiles),
            "profiles_sha256": sha256(profiles),
            "preferences": str(preferences),
            "preferences_sha256": sha256(preferences),
            "kmeans_assignments_sha256": sha256(km_assignments_path),
            "dbscan_assignments_sha256": sha256(db_assignments_path),
        },
    }
    (out / "final-clustering-evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"functions={len(names)} kmeans_clusters={len(set(km_assignments.values()))} "
        f"dbscan_clusters={len({c for c in db_assignments.values() if c != -1})} "
        f"dbscan_noise={sum(c == -1 for c in db_assignments.values())} output={out}"
    )
    print("\nK-Means cluster labels @ tau=15:")
    for row in km_summary:
        print(
            f"  C{row['cluster']}: n={row['size']} x86={row['x86_preferred_count']} "
            f"ARM={row['arm_preferred_count']} ind={row['architecture_independent_count']} "
            f"majority={row['majority_label']} share={row['majority_share']:.4f}"
        )
    print("\nDBSCAN cluster labels @ tau=15:")
    for row in db_summary:
        name = "noise" if row["is_noise"] else f"C{row['cluster']}"
        share = "n/a" if row["is_noise"] else f"{row['majority_share']:.4f}"
        print(
            f"  {name}: n={row['size']} x86={row['x86_preferred_count']} "
            f"ARM={row['arm_preferred_count']} ind={row['architecture_independent_count']} "
            f"majority={row['majority_label']} share={share}"
        )


if __name__ == "__main__":
    main()