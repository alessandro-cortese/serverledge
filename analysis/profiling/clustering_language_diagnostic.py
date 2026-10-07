#!/usr/bin/env python3
"""Auditable language-feature diagnostic for Serverledge K-Means clustering.

This script reproduces the language-vs-architecture diagnostic as a permanent
analysis artifact instead of an ad-hoc terminal snippet. Architecture labels
are used only after clustering for interpretation; they never enter K-Means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score

from analysis.profiling.clustering_dynamic_static_comparison_v2 import (
    PAPER5,
    STATIC4,
    feature_matrices,
    source_available,
    threshold_slug,
)

CONFIGS = [
    ("paper5_minmax_k5", "paper5", "minmax", 0.0, 5),
    ("static4_minmax_k10", "static4", "minmax", 0.0, 10),
    ("static4_lang_minmax_k6_w1", "static4_plus_language", "minmax", 1.0, 6),
    ("static4_lang_standard_k7_w1", "static4_plus_language", "standard", 1.0, 7),
    ("static4_lang_minmax_k8_w1", "static4_plus_language", "minmax", 1.0, 8),
    ("static4_lang_minmax_k10_w1", "static4_plus_language", "minmax", 1.0, 10),
    ("paper5_lang_minmax_k8_w1", "paper5_plus_language", "minmax", 1.0, 8),
    (
        "paper5_static4_lang_minmax_k9_w025",
        "paper5_plus_static4_plus_language",
        "minmax",
        0.25,
        9,
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for c in np.unique(clusters):
        _, counts = np.unique(labels[clusters == c], return_counts=True)
        total += int(counts.max())
    return float(total / len(labels))


def prepare_frame(args) -> tuple[pd.DataFrame, Path]:
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())
    prefs_path = args.preferences_dir.resolve() / f"preferences-{threshold_slug(args.threshold)}.csv"
    prefs = pd.read_csv(prefs_path)

    df = (
        profiles.merge(
            static[["function_name", "static_source_available", *STATIC4]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(languages[["function_name", "language"]], on="function_name", validate="one_to_one")
        .merge(
            prefs[["function_name", "architecture_preference"]],
            on="function_name",
            validate="one_to_one",
        )
    )

    mask = source_available(df["static_source_available"])
    for col in STATIC4:
        mask &= pd.to_numeric(df[col], errors="coerce").notna()

    df = df[mask].copy().sort_values("function_name").reset_index(drop=True)
    if len(df) != 59:
        raise SystemExit(f"Expected exactly 59 static-eligible functions, got {len(df)}")

    for col in PAPER5 + STATIC4:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df, prefs_path


def main(args):
    df, prefs_path = prepare_frame(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    composition_rows = []
    assignment_rows = []

    for name, group, scaler, weight, k in CONFIGS:
        x, _ = feature_matrices(
            train=df,
            target=None,
            group=group,
            scaler_name=scaler,
            language_weight=weight,
        )

        model = KMeans(n_clusters=k, n_init=args.n_init, random_state=args.random_state)
        clusters = model.fit_predict(x)

        sizes = pd.Series(clusters).value_counts().sort_values(ascending=False)
        language = df["language"].astype(str).to_numpy()
        architecture = df["architecture_preference"].astype(str).to_numpy()

        rows.append(
            {
                "configuration": name,
                "feature_group": group,
                "scaler": scaler,
                "k": k,
                "language_weight": weight,
                "max_cluster_share": float(sizes.max() / len(df)),
                "singleton_count": int((sizes == 1).sum()),
                "language_purity": purity(language, clusters),
                "language_nmi": float(normalized_mutual_info_score(language, clusters)),
                "language_ari": float(adjusted_rand_score(language, clusters)),
                "language_homogeneity": float(homogeneity_score(language, clusters)),
                "architecture_purity": purity(architecture, clusters),
                "architecture_nmi": float(normalized_mutual_info_score(architecture, clusters)),
                "architecture_ari": float(adjusted_rand_score(architecture, clusters)),
                "cluster_size_distribution": "|".join(str(int(v)) for v in sizes.to_numpy()),
            }
        )

        tmp = df[["function_name", "language", "architecture_preference"]].copy()
        tmp["cluster"] = clusters
        for _, r in tmp.iterrows():
            assignment_rows.append({
                "configuration": name,
                "feature_group": group,
                "scaler": scaler,
                "k": k,
                "language_weight": weight,
                "function_name": r["function_name"],
                "language": r["language"],
                "architecture_preference": r["architecture_preference"],
                "cluster": int(r["cluster"]),
            })

        for cluster_id, cdf in tmp.groupby("cluster"):
            lc = Counter(cdf["language"].astype(str))
            ac = Counter(cdf["architecture_preference"].astype(str))
            composition_rows.append(
                {
                    "configuration": name,
                    "cluster": int(cluster_id),
                    "cluster_size": int(len(cdf)),
                    "go_count": lc.get("go", 0),
                    "python_count": lc.get("python", 0),
                    "nodejs_count": lc.get("nodejs", 0),
                    "x86_preferred_count": ac.get("x86-preferred", 0),
                    "independent_count": ac.get("architecture-independent", 0),
                    "arm_preferred_count": ac.get("arm-preferred", 0),
                    "functions": "|".join(cdf["function_name"].astype(str)),
                }
            )

    summary = pd.DataFrame(rows)
    composition = pd.DataFrame(composition_rows)
    assignments = pd.DataFrame(assignment_rows)

    summary_path = args.output_dir / "language-cluster-summary.csv"
    composition_path = args.output_dir / "cluster-language-composition.csv"
    assignments_path = args.output_dir / "candidate-cluster-assignments.csv"
    report_path = args.output_dir / "language-feature-evidence.md"
    manifest_path = args.output_dir / "language-feature-evidence-manifest.json"

    summary.to_csv(summary_path, index=False)
    composition.to_csv(composition_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    baseline = summary.loc[summary["configuration"] == "static4_minmax_k10"].iloc[0]
    strong = summary.loc[summary["configuration"] == "static4_lang_minmax_k6_w1"].iloc[0]
    weak = summary.loc[
        summary["configuration"] == "paper5_static4_lang_minmax_k9_w025"
    ].iloc[0]

    report = f"""# Evidence: language as a clustering feature

Threshold for post-hoc architecture labels: **{args.threshold}%**  
Static-eligible corpus: **{len(df)} functions**.

Architecture labels are **not** used to fit K-Means. They are reported only after clustering.

## Main evidence

### STATIC4 without language

`static4_minmax_k10`

- language purity: **{baseline.language_purity:.4f}**
- language NMI: **{baseline.language_nmi:.4f}**
- language ARI: **{baseline.language_ari:.4f}**
- language homogeneity: **{baseline.language_homogeneity:.4f}**
- architecture NMI: **{baseline.architecture_nmi:.4f}**
- architecture ARI: **{baseline.architecture_ari:.4f}**

### STATIC4 + language, weight 1.0

`static4_lang_minmax_k6_w1`

- language purity: **{strong.language_purity:.4f}**
- language NMI: **{strong.language_nmi:.4f}**
- language ARI: **{strong.language_ari:.4f}**
- language homogeneity: **{strong.language_homogeneity:.4f}**
- architecture NMI: **{strong.architecture_nmi:.4f}**
- architecture ARI: **{strong.architecture_ari:.4f}**

A homogeneity of 1.0 means each resulting cluster contains functions from only one language. This is strong evidence that the explicit language coordinate can dominate the partition.

### Lower language weight does not remove the effect

`paper5_static4_lang_minmax_k9_w025`

- language purity: **{weak.language_purity:.4f}**
- language NMI: **{weak.language_nmi:.4f}**
- language homogeneity: **{weak.language_homogeneity:.4f}**
- architecture NMI: **{weak.architecture_nmi:.4f}**
- architecture ARI: **{weak.architecture_ari:.4f}**

## Decision for the current primary vector

Keep `language` as an ablation/sensitivity feature, but do **not** freeze it into the primary donor-clustering vector. The reason is methodological: explicit language encoding creates partitions that are strongly language-pure and can prevent cross-language donor discovery by construction, while the corresponding architecture alignment remains weak.

Inspect `cluster-language-composition.csv` to verify the actual functions and language counts inside every cluster.
"""
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "threshold_percent": args.threshold,
        "eligible_functions": len(df),
        "n_init": args.n_init,
        "random_state": args.random_state,
        "configs": CONFIGS,
        "ground_truth_usage": "post-hoc only; never used to fit K-Means",
        "inputs": {
            "profiles": {"path": str(args.profiles.resolve()), "sha256": sha256(args.profiles.resolve())},
            "static_metrics": {"path": str(args.static_metrics.resolve()), "sha256": sha256(args.static_metrics.resolve())},
            "language_map": {"path": str(args.language_map.resolve()), "sha256": sha256(args.language_map.resolve())},
            "preferences": {"path": str(prefs_path), "sha256": sha256(prefs_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"summary={summary_path}")
    print(f"composition={composition_path}")
    print(f"assignments={assignments_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--static-metrics", type=Path, required=True)
    p.add_argument("--language-map", type=Path, required=True)
    p.add_argument("--preferences-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
