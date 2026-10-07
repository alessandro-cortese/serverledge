#!/usr/bin/env python3
"""
Static feature subset ablation V2 for Serverledge clustering.

Changes from V1
---------------
1. Counts the number of unique feature vectors for every static subset.
2. Skips configurations where requested k exceeds the number of unique vectors.
3. Records the actual number of K-Means clusters and rejects incomplete fits.
4. Keeps single-feature experiments as attribution/ablation evidence, but
   produces a separate multi-feature Pareto shortlist for final representation
   selection.
5. Keeps architecture labels strictly post-hoc. Internal Pareto selection uses
   only unsupervised geometry, balance and multi-seed stability.
6. Reports Calinski-Harabasz but does not use it as a Pareto axis because its
   magnitude is not directly comparable across different dimensionalities.

Outputs
-------
static-subset-unique-vectors.csv
static-subset-skipped-configurations.csv
static-subset-ablation-detail.csv
static-subset-ablation-summary.csv
static-subset-ablation-pareto-all.csv
static-subset-ablation-pareto-multifeature.csv
static-subset-language-diagnostic.csv
static-subset-ablation-report.md
static-subset-ablation-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
    subsets = []
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


def count_unique_vectors(
    df: pd.DataFrame,
    subset: tuple[str, ...],
) -> int:
    columns = subset_columns(subset)
    return int(df[columns].drop_duplicates().shape[0])


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
            f"Non-finite values: subset={subset_name(subset)}, "
            f"scaler={scaler_name}"
        )

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=seed,
    )
    clusters = model.fit_predict(x)
    actual_clusters = int(np.unique(clusters).size)

    return x, clusters, actual_clusters


def pairwise_seed_ari(
    partitions: list[np.ndarray],
) -> float:
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
    Internal Pareto axes:
      silhouette          maximize
      Davies-Bouldin      minimize
      max cluster share   minimize
      singleton count     minimize
      multi-seed ARI      maximize
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


def make_unique_vector_table(
    df: pd.DataFrame,
    subsets: list[tuple[str, ...]],
) -> pd.DataFrame:
    rows = []
    for subset in subsets:
        n_unique = count_unique_vectors(df, subset)
        rows.append(
            {
                "subset": subset_name(subset),
                "subset_size": len(subset),
                "features": "|".join(subset_columns(subset)),
                "unique_feature_vectors": n_unique,
                "unique_vector_fraction": n_unique / len(df),
                "duplicate_rows": len(df) - n_unique,
                "single_feature": len(subset) == 1,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    df: pd.DataFrame,
    unique_table: pd.DataFrame,
    skipped: pd.DataFrame,
    pareto_all: pd.DataFrame,
    pareto_multi: pd.DataFrame,
    threshold: float,
):
    function_row = unique_table[
        unique_table["subset"] == "functions"
    ].iloc[0]

    lines = [
        "# Static feature subset ablation V2",
        "",
        f"- Static-eligible functions: {len(df)}",
        f"- Post-hoc architecture threshold: {threshold}%",
        "- Architecture labels are never used to fit or select K-Means.",
        "",
        "## Validity correction",
        "",
        (
            f"`functions` alone has only "
            f"{int(function_row['unique_feature_vectors'])} unique feature "
            f"vectors over {len(df)} functions. Therefore requesting k above "
            "that value cannot yield the requested number of clusters."
        ),
        (
            "V2 explicitly skips every configuration where k exceeds the "
            "number of unique feature vectors and rejects any fit whose actual "
            "cluster count is smaller than requested."
        ),
        (
            f"Skipped configurations recorded: {len(skipped)}."
        ),
        "",
        "Single-feature experiments remain useful for attribution, but the "
        "multi-feature Pareto table is the primary shortlist for a final static "
        "representation because donor similarity based on one scalar alone is "
        "too reductive to freeze without additional operational validation.",
        "",
        "## Multi-feature internal Pareto",
        "",
    ]

    if pareto_multi.empty:
        lines.append("- No valid multi-feature Pareto configurations.")
    else:
        for _, r in pareto_multi.head(30).iterrows():
            lines.append(
                f"- `{r['subset']}` / {r['scaler']} / k={int(r['k'])}: "
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
        "## Interpretation rule",
        "",
        (
            "Do not select a subset from silhouette alone. Prefer a small set "
            "of non-dominated multi-feature candidates with good stability, "
            "reasonable cluster balance, few or no singletons, and limited "
            "language alignment. Architecture alignment remains post-hoc only."
        ),
        "",
        "The next stage should compare the surviving static subsets against "
        "PAPER-5 in a block-weighted dynamic+static representation and then "
        "validate the shortlist using donor quality.",
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

    unique_table = make_unique_vector_table(df, subsets)
    unique_path = (
        args.output_dir / "static-subset-unique-vectors.csv"
    )
    unique_table.to_csv(unique_path, index=False)

    skipped_rows = []
    detail_rows = []

    total_candidate_configs = (
        len(subsets) * len(scalers) * len(ks)
    )
    config_done = 0

    language = df["language"].astype(str).to_numpy()
    architecture = (
        df["architecture_preference"]
        .astype(str)
        .to_numpy()
    )

    for subset in subsets:
        subset_id = subset_name(subset)
        columns = subset_columns(subset)
        n_unique = count_unique_vectors(df, subset)

        for scaler_name in scalers:
            for k in ks:
                config_done += 1

                if k > n_unique:
                    skipped_rows.append(
                        {
                            "subset": subset_id,
                            "subset_size": len(subset),
                            "features": "|".join(columns),
                            "scaler": scaler_name,
                            "k": k,
                            "unique_feature_vectors": n_unique,
                            "reason": "requested_k_exceeds_unique_feature_vectors",
                        }
                    )
                    continue

                seed_results = []
                partitions = []
                incomplete = False

                for seed in seeds:
                    x, clusters, actual_clusters = fit_partition(
                        df=df,
                        subset=subset,
                        scaler_name=scaler_name,
                        k=k,
                        seed=seed,
                        n_init=args.n_init,
                    )

                    if actual_clusters != k:
                        skipped_rows.append(
                            {
                                "subset": subset_id,
                                "subset_size": len(subset),
                                "features": "|".join(columns),
                                "scaler": scaler_name,
                                "k": k,
                                "unique_feature_vectors": n_unique,
                                "reason": (
                                    "actual_cluster_count_less_than_requested"
                                ),
                            }
                        )
                        incomplete = True
                        break

                    sizes = (
                        pd.Series(clusters)
                        .value_counts()
                        .sort_values(ascending=False)
                    )

                    seed_results.append(
                        {
                            "seed": seed,
                            "x": x,
                            "clusters": clusters,
                            "actual_cluster_count": actual_clusters,
                            "sizes": sizes,
                        }
                    )
                    partitions.append(clusters)

                if incomplete:
                    continue

                stability = pairwise_seed_ari(partitions)

                for result in seed_results:
                    seed = result["seed"]
                    x = result["x"]
                    clusters = result["clusters"]
                    sizes = result["sizes"]

                    detail_rows.append(
                        {
                            "subset": subset_id,
                            "subset_size": len(subset),
                            "features": "|".join(columns),
                            "unique_feature_vectors": n_unique,
                            "unique_vector_fraction": n_unique / len(df),
                            "single_feature": len(subset) == 1,
                            "scaler": scaler_name,
                            "k": k,
                            "seed": seed,
                            "actual_cluster_count": int(
                                result["actual_cluster_count"]
                            ),
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

                if (
                    config_done % 25 == 0
                    or config_done == total_candidate_configs
                ):
                    print(
                        f"config_progress={config_done}/"
                        f"{total_candidate_configs}"
                    )

    skipped = pd.DataFrame(skipped_rows).drop_duplicates()

    skipped_path = (
        args.output_dir / "static-subset-skipped-configurations.csv"
    )
    skipped.to_csv(skipped_path, index=False)

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
                "unique_feature_vectors",
                "unique_vector_fraction",
                "single_feature",
                "scaler",
                "k",
            ],
            as_index=False,
        )
        .agg(
            actual_cluster_count_min=("actual_cluster_count", "min"),
            actual_cluster_count_max=("actual_cluster_count", "max"),
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

    summary["internal_pareto_all"] = pareto_flags(summary)

    multi = summary[~summary["single_feature"]].copy()
    multi["internal_pareto_multifeature"] = pareto_flags(multi)

    summary = summary.merge(
        multi[
            [
                "subset",
                "scaler",
                "k",
                "internal_pareto_multifeature",
            ]
        ],
        on=["subset", "scaler", "k"],
        how="left",
        validate="one_to_one",
    )

    summary["internal_pareto_multifeature"] = (
        summary["internal_pareto_multifeature"]
        .fillna(False)
        .astype(bool)
    )

    summary_path = (
        args.output_dir / "static-subset-ablation-summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    pareto_all = (
        summary[summary["internal_pareto_all"]]
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

    pareto_all_path = (
        args.output_dir / "static-subset-ablation-pareto-all.csv"
    )
    pareto_all.to_csv(pareto_all_path, index=False)

    pareto_multi = (
        summary[summary["internal_pareto_multifeature"]]
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

    pareto_multi_path = (
        args.output_dir
        / "static-subset-ablation-pareto-multifeature.csv"
    )
    pareto_multi.to_csv(pareto_multi_path, index=False)

    language_path = (
        args.output_dir / "static-subset-language-diagnostic.csv"
    )
    summary[
        [
            "subset",
            "subset_size",
            "features",
            "unique_feature_vectors",
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
        unique_table=unique_table,
        skipped=skipped,
        pareto_all=pareto_all,
        pareto_multi=pareto_multi,
        threshold=args.threshold,
    )

    manifest = {
        "version": 2,
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
            "share, singleton count and multi-seed ARI. Configurations where "
            "requested k exceeds unique feature vectors or actual clusters "
            "are fewer than requested are excluded. Architecture labels are "
            "post-hoc only. Single-feature results are attribution evidence; "
            "a separate multi-feature Pareto is emitted for final selection."
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
    print("UNIQUE FEATURE VECTORS")
    print(
        unique_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("MULTI-FEATURE INTERNAL PARETO")
    print(
        pareto_multi[
            [
                "subset",
                "subset_size",
                "unique_feature_vectors",
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
    print(f"unique_vectors={unique_path}")
    print(f"skipped={skipped_path}")
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"pareto_all={pareto_all_path}")
    print(f"pareto_multifeature={pareto_multi_path}")
    print(f"language={language_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every non-empty subset of four static clustering "
            "features with explicit duplicate-vector validity checks."
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
