#!/usr/bin/env python3
"""
Static feature subset ablation for Serverledge clustering.

Purpose
-------
Evaluate all non-empty subsets of the four pre-declared cross-language static
features on the SAME 59 static-eligible functions, without using architecture
ground-truth labels to select the clustering representation.

For each subset, scaler, k and seed the script computes:
- silhouette (higher is better)
- Davies-Bouldin (lower is better)
- Calinski-Harabasz (higher is better)
- min/max cluster size
- max cluster share (lower is better)
- singleton count (lower is better)
- multi-seed partition stability via pairwise ARI (higher is better)

Language alignment is reported separately as a diagnostic:
- purity
- NMI
- ARI
- homogeneity

Architecture-preference alignment is also reported only post-hoc and is NEVER
used in the internal Pareto selection.

Outputs
-------
static-subset-ablation-detail.csv
static-subset-ablation-summary.csv
static-subset-ablation-pareto.csv
static-subset-language-diagnostic.csv
static-subset-ablation-report.md
static-subset-ablation-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


STATIC_FEATURES = {
    "ccn": "static_ccn_mean",
    "nloc": "static_function_nloc_mean",
    "tokens": "static_token_count_mean",
    "functions": "static_function_count",
}

LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def threshold_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def source_available(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def scaler_from_name(name: str):
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def all_nonempty_subsets() -> list[tuple[str, ...]]:
    names = list(STATIC_FEATURES.keys())
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(names) + 1):
        subsets.extend(itertools.combinations(names, size))
    return subsets


def subset_name(subset: tuple[str, ...]) -> str:
    return "+".join(subset)


def subset_columns(subset: tuple[str, ...]) -> list[str]:
    return [STATIC_FEATURES[name] for name in subset]


def purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        _, counts = np.unique(
            labels[clusters == cluster],
            return_counts=True,
        )
        total += int(counts.max())
    return float(total / len(labels))


def fit_partition(
    df: pd.DataFrame,
    subset: tuple[str, ...],
    scaler_name: str,
    k: int,
    seed: int,
    n_init: int,
):
    columns = subset_columns(subset)
    scaler = scaler_from_name(scaler_name)
    x = scaler.fit_transform(df[columns].astype(float))

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite values: subset={subset_name(subset)}, scaler={scaler_name}"
        )

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=seed,
    )
    clusters = model.fit_predict(x)
    return x, clusters


def pairwise_seed_ari(
    df: pd.DataFrame,
    subset: tuple[str, ...],
    scaler_name: str,
    k: int,
    seeds: list[int],
    n_init: int,
) -> float:
    partitions = []
    for seed in seeds:
        _, clusters = fit_partition(
            df=df,
            subset=subset,
            scaler_name=scaler_name,
            k=k,
            seed=seed,
            n_init=n_init,
        )
        partitions.append(clusters)

    values = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            values.append(
                adjusted_rand_score(
                    partitions[i],
                    partitions[j],
                )
            )

    return float(np.mean(values)) if values else 1.0


def pareto_flags(summary: pd.DataFrame) -> pd.Series:
    """
    Internal Pareto only:
      silhouette               maximize
      Davies-Bouldin           minimize
      max cluster share        minimize
      singleton count          minimize
      multi-seed ARI           maximize

    CH is reported but omitted from dominance because its scale changes strongly
    with dimensionality; it remains a supporting metric, not a Pareto axis.
    """
    values = summary[
        [
            "silhouette_mean",
            "davies_bouldin_mean",
            "max_cluster_share_mean",
            "singleton_count_mean",
            "mean_pairwise_seed_ari",
        ]
    ].to_numpy(float)

    flags = []

    for i, a in enumerate(values):
        dominated = False

        for j, b in enumerate(values):
            if i == j:
                continue

            no_worse = (
                b[0] >= a[0]
                and b[1] <= a[1]
                and b[2] <= a[2]
                and b[3] <= a[3]
                and b[4] >= a[4]
            )

            strictly_better = (
                b[0] > a[0]
                or b[1] < a[1]
                or b[2] < a[2]
                or b[3] < a[3]
                or b[4] > a[4]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        flags.append(not dominated)

    return pd.Series(flags, index=summary.index)


def prepare_frame(args) -> tuple[pd.DataFrame, Path]:
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())

    prefs_path = (
        args.preferences_dir.resolve()
        / f"preferences-{threshold_slug(args.threshold)}.csv"
    )
    prefs = pd.read_csv(prefs_path)

    required_static = [
        "function_name",
        "static_source_available",
        *STATIC_FEATURES.values(),
    ]

    df = (
        static[required_static]
        .merge(
            languages[["function_name", "language"]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            prefs[["function_name", "architecture_preference"]],
            on="function_name",
            validate="one_to_one",
        )
    )

    complete = source_available(df["static_source_available"])

    for column in STATIC_FEATURES.values():
        numeric = pd.to_numeric(df[column], errors="coerce")
        complete &= numeric.notna()
        df[column] = numeric

    df = (
        df[complete]
        .copy()
        .sort_values("function_name")
        .reset_index(drop=True)
    )

    if len(df) != 59:
        raise SystemExit(
            f"Expected exactly 59 static-eligible functions, got {len(df)}"
        )

    return df, prefs_path


def write_report(
    path: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    threshold: float,
):
    full = summary[
        summary["subset"] == "ccn+nloc+tokens+functions"
    ].copy()

    best_balance = (
        summary.sort_values(
            [
                "max_cluster_share_mean",
                "singleton_count_mean",
                "davies_bouldin_mean",
                "silhouette_mean",
            ],
            ascending=[True, True, True, False],
        )
        .iloc[0]
    )

    lines = [
        "# Static feature subset ablation",
        "",
        f"- Static-eligible functions: {len(df)}",
        f"- Post-hoc architecture threshold: {threshold}%",
        "- Architecture labels are not used to fit or select K-Means.",
        "- All 15 non-empty subsets of the four static features are evaluated.",
        "",
        "## Static feature family",
        "",
        "- `ccn`: static_ccn_mean",
        "- `nloc`: static_function_nloc_mean",
        "- `tokens`: static_token_count_mean",
        "- `functions`: static_function_count",
        "",
        "## Selection rule",
        "",
        (
            "The primary screen is multi-objective. Silhouette, Davies-Bouldin, "
            "cluster balance, singleton count and multi-seed ARI are evaluated "
            "jointly. Calinski-Harabasz is reported as supporting evidence but "
            "is not a Pareto axis because its magnitude is not directly "
            "comparable across different dimensionalities."
        ),
        "",
        (
            "Language purity/NMI/ARI/homogeneity are diagnostics only. "
            "A strong language alignment is treated as a warning for donor "
            "clustering because the intended search space allows cross-language "
            "donors."
        ),
        "",
        "## Full STATIC4 reference",
        "",
    ]

    for _, r in full.sort_values(["scaler", "k"]).iterrows():
        lines.append(
            f"- {r['scaler']} k={int(r['k'])}: "
            f"sil={r['silhouette_mean']:.4f}, "
            f"DB={r['davies_bouldin_mean']:.4f}, "
            f"CH={r['calinski_harabasz_mean']:.2f}, "
            f"max-share={r['max_cluster_share_mean']:.4f}, "
            f"singletons={r['singleton_count_mean']:.2f}, "
            f"seed-ARI={r['mean_pairwise_seed_ari']:.4f}, "
            f"language-NMI={r['language_nmi_mean']:.4f}"
        )

    lines += [
        "",
        "## Most balanced observed configuration",
        "",
        (
            f"- subset `{best_balance['subset']}`, scaler "
            f"`{best_balance['scaler']}`, k={int(best_balance['k'])}"
        ),
        (
            f"- sil={best_balance['silhouette_mean']:.4f}, "
            f"DB={best_balance['davies_bouldin_mean']:.4f}, "
            f"CH={best_balance['calinski_harabasz_mean']:.2f}"
        ),
        (
            f"- max-share={best_balance['max_cluster_share_mean']:.4f}, "
            f"singletons={best_balance['singleton_count_mean']:.2f}, "
            f"seed-ARI={best_balance['mean_pairwise_seed_ari']:.4f}"
        ),
        (
            f"- language NMI={best_balance['language_nmi_mean']:.4f}, "
            f"homogeneity={best_balance['language_homogeneity_mean']:.4f}"
        ),
        "",
        "## Internal Pareto configurations",
        "",
    ]

    for _, r in pareto.head(30).iterrows():
        lines.append(
            f"- `{r['subset']}` / {r['scaler']} / k={int(r['k'])}: "
            f"sil={r['silhouette_mean']:.4f}, "
            f"DB={r['davies_bouldin_mean']:.4f}, "
            f"max-share={r['max_cluster_share_mean']:.4f}, "
            f"singletons={r['singleton_count_mean']:.2f}, "
            f"seed-ARI={r['mean_pairwise_seed_ari']:.4f}, "
            f"language-NMI={r['language_nmi_mean']:.4f}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "This experiment is an ablation/screening stage. It does not freeze "
            "the final feature vector. The next stage should retain only a small "
            "set of non-dominated, stable, non-language-dominated static subsets "
            "and compare them against PAPER-5 and a block-weighted dynamic+static "
            "representation, followed by donor-quality validation."
        ),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    df, prefs_path = prepare_frame(args)

    scalers = parse_csv_strings(args.scalers)
    ks = parse_csv_ints(args.k_values)
    seeds = parse_csv_ints(args.seeds)
    subsets = all_nonempty_subsets()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    total = len(subsets) * len(scalers) * len(ks) * len(seeds)
    done = 0

    stability_cache = {}

    for subset in subsets:
        subset_id = subset_name(subset)
        columns = subset_columns(subset)

        for scaler_name in scalers:
            for k in ks:
                stability = pairwise_seed_ari(
                    df=df,
                    subset=subset,
                    scaler_name=scaler_name,
                    k=k,
                    seeds=seeds,
                    n_init=args.n_init,
                )
                stability_cache[(subset_id, scaler_name, k)] = stability

                for seed in seeds:
                    x, clusters = fit_partition(
                        df=df,
                        subset=subset,
                        scaler_name=scaler_name,
                        k=k,
                        seed=seed,
                        n_init=args.n_init,
                    )

                    sizes = (
                        pd.Series(clusters)
                        .value_counts()
                        .sort_values(ascending=False)
                    )

                    language = df["language"].astype(str).to_numpy()
                    architecture = (
                        df["architecture_preference"]
                        .astype(str)
                        .to_numpy()
                    )

                    detail_rows.append(
                        {
                            "subset": subset_id,
                            "subset_size": len(subset),
                            "features": "|".join(columns),
                            "scaler": scaler_name,
                            "k": k,
                            "seed": seed,
                            "silhouette": float(
                                silhouette_score(x, clusters)
                            ),
                            "davies_bouldin": float(
                                davies_bouldin_score(x, clusters)
                            ),
                            "calinski_harabasz": float(
                                calinski_harabasz_score(x, clusters)
                            ),
                            "min_cluster_size": int(sizes.min()),
                            "max_cluster_size": int(sizes.max()),
                            "max_cluster_share": float(
                                sizes.max() / len(df)
                            ),
                            "singleton_count": int((sizes == 1).sum()),
                            "mean_pairwise_seed_ari": stability,
                            "language_purity": purity(
                                language,
                                clusters,
                            ),
                            "language_nmi": float(
                                normalized_mutual_info_score(
                                    language,
                                    clusters,
                                )
                            ),
                            "language_ari": float(
                                adjusted_rand_score(
                                    language,
                                    clusters,
                                )
                            ),
                            "language_homogeneity": float(
                                homogeneity_score(
                                    language,
                                    clusters,
                                )
                            ),
                            "architecture_purity": purity(
                                architecture,
                                clusters,
                            ),
                            "architecture_nmi": float(
                                normalized_mutual_info_score(
                                    architecture,
                                    clusters,
                                )
                            ),
                            "architecture_ari": float(
                                adjusted_rand_score(
                                    architecture,
                                    clusters,
                                )
                            ),
                            "cluster_size_distribution": "|".join(
                                str(int(v)) for v in sizes.to_numpy()
                            ),
                        }
                    )

                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"progress={done}/{total}")

    detail = pd.DataFrame(detail_rows)

    detail_path = (
        args.output_dir / "static-subset-ablation-detail.csv"
    )
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(
            [
                "subset",
                "subset_size",
                "features",
                "scaler",
                "k",
            ],
            as_index=False,
        )
        .agg(
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            calinski_harabasz_mean=("calinski_harabasz", "mean"),
            min_cluster_size_mean=("min_cluster_size", "mean"),
            max_cluster_size_mean=("max_cluster_size", "mean"),
            max_cluster_share_mean=("max_cluster_share", "mean"),
            singleton_count_mean=("singleton_count", "mean"),
            mean_pairwise_seed_ari=("mean_pairwise_seed_ari", "first"),
            language_purity_mean=("language_purity", "mean"),
            language_nmi_mean=("language_nmi", "mean"),
            language_ari_mean=("language_ari", "mean"),
            language_homogeneity_mean=("language_homogeneity", "mean"),
            architecture_purity_mean=("architecture_purity", "mean"),
            architecture_nmi_mean=("architecture_nmi", "mean"),
            architecture_ari_mean=("architecture_ari", "mean"),
        )
    )

    summary["internal_pareto"] = pareto_flags(summary)

    summary_path = (
        args.output_dir / "static-subset-ablation-summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    pareto = (
        summary[summary["internal_pareto"]]
        .copy()
        .sort_values(
            [
                "max_cluster_share_mean",
                "singleton_count_mean",
                "davies_bouldin_mean",
                "silhouette_mean",
            ],
            ascending=[True, True, True, False],
        )
    )

    pareto_path = (
        args.output_dir / "static-subset-ablation-pareto.csv"
    )
    pareto.to_csv(pareto_path, index=False)

    language_path = (
        args.output_dir / "static-subset-language-diagnostic.csv"
    )
    summary[
        [
            "subset",
            "subset_size",
            "features",
            "scaler",
            "k",
            "language_purity_mean",
            "language_nmi_mean",
            "language_ari_mean",
            "language_homogeneity_mean",
            "architecture_purity_mean",
            "architecture_nmi_mean",
            "architecture_ari_mean",
        ]
    ].to_csv(language_path, index=False)

    report_path = (
        args.output_dir / "static-subset-ablation-report.md"
    )
    write_report(
        path=report_path,
        df=df,
        summary=summary,
        pareto=pareto,
        threshold=args.threshold,
    )

    manifest = {
        "threshold_percent": args.threshold,
        "eligible_functions": len(df),
        "features": STATIC_FEATURES,
        "subset_count": len(subsets),
        "scalers": scalers,
        "k_values": ks,
        "seeds": seeds,
        "n_init": args.n_init,
        "selection_rule": (
            "Internal Pareto uses silhouette, Davies-Bouldin, max cluster "
            "share, singleton count, and multi-seed ARI. Architecture labels "
            "are post-hoc only. Language alignment is diagnostic only."
        ),
        "inputs": {
            "static_metrics": {
                "path": str(args.static_metrics.resolve()),
                "sha256": sha256_file(args.static_metrics.resolve()),
            },
            "language_map": {
                "path": str(args.language_map.resolve()),
                "sha256": sha256_file(args.language_map.resolve()),
            },
            "preferences": {
                "path": str(prefs_path),
                "sha256": sha256_file(prefs_path),
            },
        },
    }

    manifest_path = (
        args.output_dir / "static-subset-ablation-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("INTERNAL PARETO")
    print(
        pareto[
            [
                "subset",
                "subset_size",
                "scaler",
                "k",
                "silhouette_mean",
                "davies_bouldin_mean",
                "calinski_harabasz_mean",
                "max_cluster_share_mean",
                "singleton_count_mean",
                "mean_pairwise_seed_ari",
                "language_nmi_mean",
                "language_homogeneity_mean",
            ]
        ]
        .head(args.top_n)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"pareto={pareto_path}")
    print(f"language={language_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every non-empty subset of the four static clustering "
            "features on the same 59 eligible functions."
        )
    )

    parser.add_argument(
        "--static-metrics",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--language-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--preferences-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--scalers",
        default="minmax,standard",
    )
    parser.add_argument(
        "--k-values",
        default="5,6,7,8,9,10,11,12",
    )
    parser.add_argument(
        "--seeds",
        default="11,23,37,41,53",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
