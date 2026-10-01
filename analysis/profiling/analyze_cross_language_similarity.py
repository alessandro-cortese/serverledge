#!/usr/bin/env python3
"""
Cross-language similarity analysis for the frozen Serverledge clustering.

Goal
----
Test whether different implementations of the same logical workload are
closer in PAPER-5 / MinMax space than implementations of different workloads.

This is a post-hoc sensitivity/interpretation experiment. It does NOT refit
or modify the frozen clustering configuration.

Primary comparison:
- intra-workload Manhattan distances
- inter-workload Manhattan distances

Operational comparison:
- intra-workload pairs that are eligible under the frozen same-cluster rule
- inter-workload pairs that are eligible under the same-cluster rule

Additional diagnostics:
- nearest-neighbour recovery globally
- nearest-neighbour recovery under the same-cluster restriction
- permutation test on logical-workload labels preserving group sizes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

GROUPS = {
    "base64stream": [
        ("base64stream", "Go"),
        ("base64stream-py", "Python"),
    ],
    "compression": [
        ("compression", "Go"),
        ("compression-py", "Python"),
        ("compression-node", "Node.js"),
    ],
    "dna-visualisation": [
        ("dna-visualisation", "Go"),
        ("dna-visualisation-py", "Python"),
    ],
    "dynamic-html": [
        ("dynamichtml", "Go"),
        ("dynamic-html-py", "Python"),
        ("dynamic-html-node", "Node.js"),
    ],
    "graph-bfs": [
        ("graph-bfs", "Go"),
        ("graph-bfs-py", "Python"),
        ("graph-bfs-node", "Node.js"),
    ],
    "graph-mst": [
        ("graph-mst", "Go"),
        ("graph-mst-py", "Python"),
    ],
    "graph-pagerank": [
        ("graph-pagerank", "Go"),
        ("graph-pagerank-py", "Python"),
        ("graph-pagerank-node", "Node.js"),
    ],
    "hashing": [
        ("hashing", "Go"),
        ("hashing-py", "Python"),
    ],
    "jsonparse": [
        ("jsonparse", "Go"),
        ("jsonparse-py", "Python"),
    ],
    "pointerchase": [
        ("pointerchase", "Go"),
        ("pointerchase-py", "Python"),
    ],
    "randomaccess": [
        ("randomaccess", "Go"),
        ("randomaccess-py", "Python"),
    ],
    "thumbnailer": [
        ("thumbnailer", "Go"),
        ("thumbnailer-py", "Python"),
        ("thumbnailer-node", "Node.js"),
    ],
    "json-dumps": [
        ("json-dumps-py", "Python"),
        ("json-dumps-node", "Node.js"),
    ],
    "float-ops": [
        ("float-ops-py", "Python"),
        ("float-ops-node", "Node.js"),
    ],
    "memory-rw": [
        ("memory-rw-py", "Python"),
        ("memory-rw-node", "Node.js"),
    ],
}


def pct_rank(value: float, background: np.ndarray) -> float:
    """Percent of background distances <= value."""
    if len(background) == 0:
        return float("nan")
    return float(np.mean(background <= value) * 100.0)


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

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

    required = ["function_name", *PAPER5_FEATURES]
    missing = [c for c in required if c not in profiles.columns]
    if missing:
        raise RuntimeError(
            f"{profiles_path}: missing columns {missing}; "
            f"available={list(profiles.columns)}"
        )

    if "cluster_label" not in assignments.columns:
        raise RuntimeError(
            f"{assignments_path}: missing cluster_label"
        )

    # Fit MinMax on the entire frozen 53-function AMD64 reference corpus,
    # exactly as required by the final clustering geometry.
    all_names = profiles["function_name"].tolist()
    X = profiles[PAPER5_FEATURES].astype(float).to_numpy()
    Xs = MinMaxScaler().fit_transform(X)

    scaled = pd.DataFrame(Xs, columns=PAPER5_FEATURES)
    scaled.insert(0, "function_name", all_names)
    scaled = scaled.merge(
        assignments[["function_name", "cluster_label"]],
        on="function_name",
        how="left",
        validate="one_to_one",
    )

    # Build metadata only for logical workloads with >=2 implementations.
    meta_rows = []
    for workload, impls in GROUPS.items():
        for function_name, runtime in impls:
            meta_rows.append(
                {
                    "function_name": function_name,
                    "logical_workload": workload,
                    "runtime": runtime,
                }
            )
    meta = pd.DataFrame(meta_rows)

    missing_functions = sorted(
        set(meta["function_name"]) - set(scaled["function_name"])
    )
    if missing_functions:
        raise RuntimeError(
            f"Functions missing from final profiles: {missing_functions}"
        )

    cross = meta.merge(
        scaled,
        on="function_name",
        how="left",
        validate="one_to_one",
    )

    # Pairwise Manhattan distances among the 35 cross-language implementations.
    pairs = []
    rows = cross.reset_index(drop=True)

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = rows.iloc[i]
            b = rows.iloc[j]

            va = a[PAPER5_FEATURES].astype(float).to_numpy()
            vb = b[PAPER5_FEATURES].astype(float).to_numpy()

            distance = float(np.abs(va - vb).sum())
            same_workload = (
                a["logical_workload"] == b["logical_workload"]
            )
            same_cluster = (
                int(a["cluster_label"]) == int(b["cluster_label"])
            )

            pairs.append(
                {
                    "function_a": a["function_name"],
                    "runtime_a": a["runtime"],
                    "workload_a": a["logical_workload"],
                    "cluster_a": int(a["cluster_label"]),
                    "function_b": b["function_name"],
                    "runtime_b": b["runtime"],
                    "workload_b": b["logical_workload"],
                    "cluster_b": int(b["cluster_label"]),
                    "same_workload": bool(same_workload),
                    "same_cluster": bool(same_cluster),
                    "manhattan_distance": distance,
                }
            )

    pairs = pd.DataFrame(pairs)
    intra = pairs[pairs["same_workload"]].copy()
    inter = pairs[~pairs["same_workload"]].copy()

    # Operationally eligible pairs under frozen same-cluster donor restriction.
    intra_eligible = intra[intra["same_cluster"]].copy()
    inter_eligible = inter[inter["same_cluster"]].copy()

    inter_dist = inter["manhattan_distance"].to_numpy()
    inter_eligible_dist = inter_eligible["manhattan_distance"].to_numpy()

    intra["percentile_vs_all_inter"] = intra["manhattan_distance"].map(
        lambda x: pct_rank(x, inter_dist)
    )
    intra["percentile_vs_same_cluster_inter"] = intra.apply(
        lambda r: (
            pct_rank(r["manhattan_distance"], inter_eligible_dist)
            if r["same_cluster"]
            else np.nan
        ),
        axis=1,
    )

    intra = intra.sort_values("manhattan_distance").reset_index(drop=True)

    # Per-workload summaries.
    workload_rows = []
    for workload, g in intra.groupby("workload_a", sort=True):
        impl_count = int(
            cross[cross["logical_workload"] == workload].shape[0]
        )
        clusters = sorted(
            cross.loc[
                cross["logical_workload"] == workload, "cluster_label"
            ].astype(int).unique().tolist()
        )

        workload_rows.append(
            {
                "logical_workload": workload,
                "implementation_count": impl_count,
                "pair_count": int(len(g)),
                "cluster_set": "|".join(map(str, clusters)),
                "all_same_cluster": len(clusters) == 1,
                "mean_intra_distance": float(
                    g["manhattan_distance"].mean()
                ),
                "median_intra_distance": float(
                    g["manhattan_distance"].median()
                ),
                "min_intra_distance": float(
                    g["manhattan_distance"].min()
                ),
                "max_intra_distance": float(
                    g["manhattan_distance"].max()
                ),
                "mean_percentile_vs_all_inter": float(
                    g["percentile_vs_all_inter"].mean()
                ),
            }
        )

    workload_summary = pd.DataFrame(workload_rows)

    # Nearest-neighbour recovery.
    nn_rows = []

    for _, target in rows.iterrows():
        candidates = rows[
            rows["function_name"] != target["function_name"]
        ].copy()

        vt = target[PAPER5_FEATURES].astype(float).to_numpy()

        candidates["distance"] = candidates.apply(
            lambda r: float(
                np.abs(
                    vt
                    - r[PAPER5_FEATURES].astype(float).to_numpy()
                ).sum()
            ),
            axis=1,
        )

        nearest_global = candidates.sort_values(
            ["distance", "function_name"]
        ).iloc[0]

        eligible = candidates[
            candidates["cluster_label"] == target["cluster_label"]
        ].copy()

        nearest_cluster = None
        if len(eligible):
            nearest_cluster = eligible.sort_values(
                ["distance", "function_name"]
            ).iloc[0]

        nn_rows.append(
            {
                "target_function": target["function_name"],
                "target_workload": target["logical_workload"],
                "target_runtime": target["runtime"],
                "target_cluster": int(target["cluster_label"]),
                "nearest_global": nearest_global["function_name"],
                "nearest_global_workload": nearest_global[
                    "logical_workload"
                ],
                "nearest_global_distance": float(
                    nearest_global["distance"]
                ),
                "nearest_global_same_workload": bool(
                    nearest_global["logical_workload"]
                    == target["logical_workload"]
                ),
                "nearest_same_cluster": (
                    nearest_cluster["function_name"]
                    if nearest_cluster is not None
                    else ""
                ),
                "nearest_same_cluster_workload": (
                    nearest_cluster["logical_workload"]
                    if nearest_cluster is not None
                    else ""
                ),
                "nearest_same_cluster_distance": (
                    float(nearest_cluster["distance"])
                    if nearest_cluster is not None
                    else np.nan
                ),
                "nearest_same_cluster_same_workload": (
                    bool(
                        nearest_cluster["logical_workload"]
                        == target["logical_workload"]
                    )
                    if nearest_cluster is not None
                    else False
                ),
            }
        )

    nn = pd.DataFrame(nn_rows)

    # Permutation test:
    # Shuffle logical-workload labels across the same 35 vectors while
    # preserving exact group-size distribution. Statistic = mean within-group
    # Manhattan distance. Smaller means same-workload implementations are
    # unusually close.
    rng = np.random.default_rng(args.random_state)

    coords = rows[PAPER5_FEATURES].astype(float).to_numpy()
    original_labels = rows["logical_workload"].to_numpy()

    # Full distance matrix.
    dist_matrix = np.abs(
        coords[:, None, :] - coords[None, :, :]
    ).sum(axis=2)

    def within_mean(labels: np.ndarray) -> float:
        vals = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    vals.append(dist_matrix[i, j])
        return float(np.mean(vals))

    observed = within_mean(original_labels)

    perm_stats = np.empty(args.permutations, dtype=float)
    for pidx in range(args.permutations):
        shuffled = rng.permutation(original_labels)
        perm_stats[pidx] = within_mean(shuffled)

    # One-sided p: same-workload pairs are closer than random grouping.
    p_value = float(
        (1 + np.sum(perm_stats <= observed))
        / (args.permutations + 1)
    )

    # Effect-size-like descriptive ratios.
    intra_desc = describe(intra["manhattan_distance"].to_numpy())
    inter_desc = describe(inter["manhattan_distance"].to_numpy())
    intra_eligible_desc = describe(
        intra_eligible["manhattan_distance"].to_numpy()
    )
    inter_eligible_desc = describe(
        inter_eligible["manhattan_distance"].to_numpy()
    )

    summary = {
        "schema_version": 1,
        "configuration": {
            "reference_architecture": "amd64",
            "aggregation": "median",
            "feature_set": "PAPER-5",
            "scaler": "MinMaxScaler",
            "distance": "Manhattan",
            "clusterer": "K-Means",
            "k": 5,
            "same_cluster_donor_restriction": True,
        },
        "sample": {
            "logical_workloads": int(len(GROUPS)),
            "implementations": int(len(cross)),
            "intra_workload_pairs": int(len(intra)),
            "inter_workload_pairs": int(len(inter)),
            "eligible_intra_workload_pairs": int(len(intra_eligible)),
            "eligible_inter_workload_pairs": int(len(inter_eligible)),
        },
        "distance_summary": {
            "intra_workload": intra_desc,
            "inter_workload": inter_desc,
            "intra_workload_same_cluster": intra_eligible_desc,
            "inter_workload_same_cluster": inter_eligible_desc,
            "median_ratio_intra_over_inter": float(
                intra_desc["median"] / inter_desc["median"]
            ),
            "mean_ratio_intra_over_inter": float(
                intra_desc["mean"] / inter_desc["mean"]
            ),
            "median_ratio_intra_over_same_cluster_inter": float(
                intra_eligible_desc["median"]
                / inter_eligible_desc["median"]
            ),
        },
        "nearest_neighbour": {
            "global_same_workload_hits": int(
                nn["nearest_global_same_workload"].sum()
            ),
            "global_targets": int(len(nn)),
            "global_same_workload_rate": float(
                nn["nearest_global_same_workload"].mean()
            ),
            "same_cluster_same_workload_hits": int(
                nn["nearest_same_cluster_same_workload"].sum()
            ),
            "same_cluster_targets": int(len(nn)),
            "same_cluster_same_workload_rate": float(
                nn["nearest_same_cluster_same_workload"].mean()
            ),
        },
        "permutation_test": {
            "null": (
                "Logical-workload labels are exchangeable across the 35 "
                "cross-language implementations, preserving group sizes."
            ),
            "statistic": "mean within-workload Manhattan distance",
            "alternative": (
                "observed same-workload mean distance is smaller than random"
            ),
            "observed": observed,
            "permutations": int(args.permutations),
            "random_state": int(args.random_state),
            "null_mean": float(np.mean(perm_stats)),
            "null_median": float(np.median(perm_stats)),
            "p_value_one_sided": p_value,
        },
    }

    # Save artifacts.
    cross.to_csv(out / "cross-language-implementations.csv", index=False)
    intra.to_csv(out / "intra-workload-pairs.csv", index=False)
    inter.to_csv(out / "inter-workload-pairs.csv", index=False)
    workload_summary.to_csv(
        out / "cross-language-workload-summary.csv",
        index=False,
    )
    nn.to_csv(out / "nearest-neighbour-recovery.csv", index=False)
    np.savetxt(
        out / "permutation-null-mean-intra-distance.csv",
        perm_stats,
        delimiter=",",
        header="permuted_mean_intra_distance",
        comments="",
    )
    (out / "cross-language-similarity-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # Console report.
    print()
    print("=" * 108)
    print("CROSS-LANGUAGE SIMILARITY — PAPER-5 / MINMAX / MANHATTAN")
    print("=" * 108)
    print(
        f"logical workloads       = {len(GROUPS)}\n"
        f"implementations         = {len(cross)}\n"
        f"intra-workload pairs    = {len(intra)}\n"
        f"inter-workload pairs    = {len(inter)}"
    )

    print()
    print("DISTANCE DISTRIBUTIONS")
    print("-" * 108)
    for name, d in [
        ("INTRA workload", intra_desc),
        ("INTER workload", inter_desc),
        ("INTRA same-cluster", intra_eligible_desc),
        ("INTER same-cluster", inter_eligible_desc),
    ]:
        print(
            f"{name:21s} "
            f"n={d['n']:3d} "
            f"mean={d['mean']:.6f} "
            f"median={d['median']:.6f} "
            f"q25={d['q25']:.6f} "
            f"q75={d['q75']:.6f}"
        )

    print()
    print(
        "median intra / inter                = "
        f"{summary['distance_summary']['median_ratio_intra_over_inter']:.4f}"
    )
    print(
        "mean intra / inter                  = "
        f"{summary['distance_summary']['mean_ratio_intra_over_inter']:.4f}"
    )
    print(
        "median intra / same-cluster inter   = "
        f"{summary['distance_summary']['median_ratio_intra_over_same_cluster_inter']:.4f}"
    )

    print()
    print("PERMUTATION TEST")
    print("-" * 108)
    print(f"observed mean intra distance = {observed:.6f}")
    print(f"null mean                    = {np.mean(perm_stats):.6f}")
    print(f"null median                  = {np.median(perm_stats):.6f}")
    print(f"one-sided p-value            = {p_value:.6f}")

    print()
    print("NEAREST-NEIGHBOUR RECOVERY")
    print("-" * 108)
    print(
        "global nearest neighbour same workload       = "
        f"{int(nn['nearest_global_same_workload'].sum())}/{len(nn)} "
        f"({100 * nn['nearest_global_same_workload'].mean():.2f}%)"
    )
    print(
        "same-cluster nearest neighbour same workload = "
        f"{int(nn['nearest_same_cluster_same_workload'].sum())}/{len(nn)} "
        f"({100 * nn['nearest_same_cluster_same_workload'].mean():.2f}%)"
    )

    print()
    print("PER-WORKLOAD")
    print("-" * 108)
    print(
        workload_summary.to_string(
            index=False,
            formatters={
                "mean_intra_distance": lambda x: f"{x:.6f}",
                "median_intra_distance": lambda x: f"{x:.6f}",
                "min_intra_distance": lambda x: f"{x:.6f}",
                "max_intra_distance": lambda x: f"{x:.6f}",
                "mean_percentile_vs_all_inter": lambda x: f"{x:.2f}%",
            },
        )
    )

    print()
    print("INTRA-WORKLOAD PAIRS")
    print("-" * 108)
    cols = [
        "workload_a",
        "function_a",
        "runtime_a",
        "cluster_a",
        "function_b",
        "runtime_b",
        "cluster_b",
        "same_cluster",
        "manhattan_distance",
        "percentile_vs_all_inter",
        "percentile_vs_same_cluster_inter",
    ]
    print(
        intra[cols].to_string(
            index=False,
            formatters={
                "manhattan_distance": lambda x: f"{x:.6f}",
                "percentile_vs_all_inter": lambda x: f"{x:.2f}%",
                "percentile_vs_same_cluster_inter": (
                    lambda x: "" if pd.isna(x) else f"{x:.2f}%"
                ),
            },
        )
    )

    print()
    print(f"output={out}")


if __name__ == "__main__":
    main()
