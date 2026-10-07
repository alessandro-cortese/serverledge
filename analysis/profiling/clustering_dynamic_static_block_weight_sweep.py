#!/usr/bin/env python3
"""
Block-weighted dynamic + static K-Means sweep for Serverledge.

The experiment combines PAPER-5 dynamic features with a small pre-declared
shortlist of static feature blocks selected after static-subset ablation.

Methodological points
---------------------
- All experiments use the SAME 59 static-eligible functions.
- Dynamic and static blocks are scaled separately.
- Each scaled block is divided by sqrt(number_of_features), so a weight of 1.0
  gives the two blocks comparable average squared-distance contribution rather
  than automatically favoring the 5-dimensional PAPER-5 block.
- The static block is then multiplied by alpha.
- Architecture preference labels are NEVER used to fit or select K-Means.
- Language and architecture alignment are diagnostics only.
- Internal Pareto selection uses silhouette, Davies-Bouldin, max-cluster share,
  singleton count and multi-seed ARI.
- Calinski-Harabasz is reported but is not a Pareto axis because its magnitude
  is not directly comparable across representations of different dimension.
- A separate donor-operational shortlist requires no singleton clusters in any
  evaluated seed (minimum cluster size >= 2 in every seed).

Outputs
-------
block-weight-sweep-detail.csv
block-weight-sweep-summary.csv
block-weight-sweep-pareto.csv
block-weight-sweep-donor-operational.csv
block-weight-sweep-language-diagnostic.csv
block-weight-sweep-report.md
block-weight-sweep-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC_BLOCKS = {
    # Primary compromise from the static ablation:
    # high resolution (57/59 unique vectors), low language alignment, stable.
    "ccn_tokens": [
        "static_ccn_mean",
        "static_token_count_mean",
    ],

    # Geometry-oriented alternative; fewer unique vectors but strong compactness.
    "ccn_functions": [
        "static_ccn_mean",
        "static_function_count",
    ],

    # Language-light / zero-singleton alternative at moderate k in the ablation.
    "tokens_functions": [
        "static_token_count_mean",
        "static_function_count",
    ],

    # Original four-feature block retained as a control to test whether
    # down-weighting can rescue the naive concatenation.
    "static4": [
        "static_ccn_mean",
        "static_function_nloc_mean",
        "static_token_count_mean",
        "static_function_count",
    ],
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


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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


def purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        _, counts = np.unique(
            labels[clusters == cluster],
            return_counts=True,
        )
        total += int(counts.max())
    return float(total / len(labels))


def prepare_frame(args) -> tuple[pd.DataFrame, Path]:
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())

    prefs_path = (
        args.preferences_dir.resolve()
        / f"preferences-{threshold_slug(args.threshold)}.csv"
    )
    prefs = pd.read_csv(prefs_path)

    all_static = sorted(
        {
            feature
            for block in STATIC_BLOCKS.values()
            for feature in block
        }
    )

    df = (
        profiles.merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    *all_static,
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            languages[["function_name", "language"]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            prefs[
                ["function_name", "architecture_preference"]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    complete = source_available(df["static_source_available"])

    for column in PAPER5 + all_static:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if column in all_static:
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

    for column in PAPER5 + all_static:
        if df[column].isna().any():
            raise SystemExit(f"Missing numeric values in {column}")

    return df, prefs_path


def scaled_block(
    df: pd.DataFrame,
    columns: list[str],
    scaler_name: str,
) -> np.ndarray:
    scaler = scaler_from_name(scaler_name)
    x = scaler.fit_transform(df[columns].astype(float))

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite block values for {columns}")

    return x / math.sqrt(len(columns))


def representation(
    df: pd.DataFrame,
    scaler_name: str,
    static_block: str | None,
    alpha: float,
) -> np.ndarray:
    dynamic = scaled_block(
        df,
        PAPER5,
        scaler_name,
    )

    if static_block is None:
        return dynamic

    static_columns = STATIC_BLOCKS[static_block]
    static = scaled_block(
        df,
        static_columns,
        scaler_name,
    )

    return np.column_stack(
        [
            dynamic,
            alpha * static,
        ]
    )


def pairwise_seed_ari(partitions: list[np.ndarray]) -> float:
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


def write_report(
    path: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    operational: pd.DataFrame,
    threshold: float,
):
    baseline = summary[summary["representation"] == "paper5"].copy()
    combined = summary[summary["representation"] != "paper5"].copy()

    lines = [
        "# Block-weighted dynamic + static clustering sweep",
        "",
        f"- Static-eligible functions: {len(df)}",
        f"- Post-hoc architecture threshold: {threshold}%",
        "- Architecture labels are never used to fit or select K-Means.",
        "- Dynamic and static blocks are scaled separately.",
        "- Each block is divided by sqrt(block dimensionality).",
        "- `alpha` multiplies only the static block.",
        "",
        "## Baseline",
        "",
    ]

    for _, r in baseline.sort_values(["scaler", "k"]).iterrows():
        lines.append(
            f"- PAPER5 / {r['scaler']} / k={int(r['k'])}: "
            f"sil={r['silhouette_mean']:.4f}, "
            f"DB={r['davies_bouldin_mean']:.4f}, "
            f"max-share={r['max_cluster_share_mean']:.4f}, "
            f"singletons={r['singleton_count_mean']:.2f}, "
            f"seed-ARI={r['mean_pairwise_seed_ari']:.4f}"
        )

    lines += [
        "",
        "## Internal Pareto",
        "",
    ]

    for _, r in pareto.head(30).iterrows():
        lines.append(
            f"- `{r['representation']}` / {r['scaler']} / "
            f"k={int(r['k'])} / alpha={r['alpha']:.3f}: "
            f"sil={r['silhouette_mean']:.4f}, "
            f"DB={r['davies_bouldin_mean']:.4f}, "
            f"max-share={r['max_cluster_share_mean']:.4f}, "
            f"singletons={r['singleton_count_mean']:.2f}, "
            f"seed-ARI={r['mean_pairwise_seed_ari']:.4f}, "
            f"language-NMI={r['language_nmi_mean']:.4f}"
        )

    lines += [
        "",
        "## Donor-operational shortlist",
        "",
        (
            "This table requires minimum cluster size >= 2 in every evaluated "
            "seed, so every historical function has at least one same-cluster "
            "peer in the full-fit partition."
        ),
        "",
    ]

    if operational.empty:
        lines.append("- No configuration satisfies the no-singleton constraint.")
    else:
        for _, r in operational.head(30).iterrows():
            lines.append(
                f"- `{r['representation']}` / {r['scaler']} / "
                f"k={int(r['k'])} / alpha={r['alpha']:.3f}: "
                f"sil={r['silhouette_mean']:.4f}, "
                f"DB={r['davies_bouldin_mean']:.4f}, "
                f"max-share={r['max_cluster_share_mean']:.4f}, "
                f"seed-ARI={r['mean_pairwise_seed_ari']:.4f}, "
                f"language-NMI={r['language_nmi_mean']:.4f}, "
                f"architecture-NMI(post-hoc)="
                f"{r['architecture_nmi_mean']:.4f}"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "This sweep tests whether static information should contribute as "
            "a weak block rather than through naive equal concatenation. "
            "Do not freeze the final vector from internal metrics alone. "
            "Retain a small non-dominated donor-operational shortlist and "
            "validate it next with actual donor-retrieval quality."
        ),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    df, prefs_path = prepare_frame(args)

    scalers = parse_csv_strings(args.scalers)
    ks = parse_csv_ints(args.k_values)
    seeds = parse_csv_ints(args.seeds)
    alphas = parse_csv_floats(args.static_weights)
    blocks = parse_csv_strings(args.static_blocks)

    unknown = sorted(set(blocks) - set(STATIC_BLOCKS))
    if unknown:
        raise SystemExit(f"Unknown static blocks: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configurations = []

    # PAPER-5 baseline only once per scaler/k.
    for scaler_name in scalers:
        for k in ks:
            configurations.append(
                {
                    "representation": "paper5",
                    "static_block": "",
                    "alpha": 0.0,
                    "scaler": scaler_name,
                    "k": k,
                }
            )

    for block in blocks:
        for scaler_name in scalers:
            for k in ks:
                for alpha in alphas:
                    configurations.append(
                        {
                            "representation": f"paper5_plus_{block}",
                            "static_block": block,
                            "alpha": alpha,
                            "scaler": scaler_name,
                            "k": k,
                        }
                    )

    language = df["language"].astype(str).to_numpy()
    architecture = (
        df["architecture_preference"]
        .astype(str)
        .to_numpy()
    )

    rows = []
    total = len(configurations)

    for config_idx, config in enumerate(configurations, start=1):
        static_block = config["static_block"] or None

        x = representation(
            df=df,
            scaler_name=config["scaler"],
            static_block=static_block,
            alpha=config["alpha"],
        )

        partitions = []
        seed_rows = []

        for seed in seeds:
            model = KMeans(
                n_clusters=config["k"],
                n_init=args.n_init,
                random_state=seed,
            )
            clusters = model.fit_predict(x)

            actual_cluster_count = int(np.unique(clusters).size)
            if actual_cluster_count != config["k"]:
                raise RuntimeError(
                    "Unexpected incomplete KMeans partition: "
                    f"{config}, seed={seed}, actual={actual_cluster_count}"
                )

            sizes = (
                pd.Series(clusters)
                .value_counts()
                .sort_values(ascending=False)
            )

            partitions.append(clusters)

            seed_rows.append(
                {
                    "seed": seed,
                    "actual_cluster_count": actual_cluster_count,
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

        stability = pairwise_seed_ari(partitions)

        for seed_row in seed_rows:
            rows.append(
                {
                    **config,
                    "mean_pairwise_seed_ari": stability,
                    **seed_row,
                }
            )

        if config_idx % 25 == 0 or config_idx == total:
            print(f"config_progress={config_idx}/{total}")

    detail = pd.DataFrame(rows)

    detail_path = (
        args.output_dir / "block-weight-sweep-detail.csv"
    )
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(
            [
                "representation",
                "static_block",
                "alpha",
                "scaler",
                "k",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            actual_cluster_count_min=("actual_cluster_count", "min"),
            actual_cluster_count_max=("actual_cluster_count", "max"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            davies_bouldin_mean=("davies_bouldin", "mean"),
            calinski_harabasz_mean=("calinski_harabasz", "mean"),
            min_cluster_size_min=("min_cluster_size", "min"),
            min_cluster_size_mean=("min_cluster_size", "mean"),
            max_cluster_size_mean=("max_cluster_size", "mean"),
            max_cluster_share_mean=("max_cluster_share", "mean"),
            singleton_count_mean=("singleton_count", "mean"),
            singleton_count_max=("singleton_count", "max"),
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
    summary["donor_operational_no_singletons"] = (
        summary["min_cluster_size_min"] >= 2
    )

    summary_path = (
        args.output_dir / "block-weight-sweep-summary.csv"
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
        args.output_dir / "block-weight-sweep-pareto.csv"
    )
    pareto.to_csv(pareto_path, index=False)

    operational = (
        summary[
            summary["donor_operational_no_singletons"]
        ]
        .copy()
        .sort_values(
            [
                "max_cluster_share_mean",
                "davies_bouldin_mean",
                "silhouette_mean",
                "mean_pairwise_seed_ari",
            ],
            ascending=[True, True, False, False],
        )
    )

    operational_path = (
        args.output_dir / "block-weight-sweep-donor-operational.csv"
    )
    operational.to_csv(operational_path, index=False)

    language_path = (
        args.output_dir / "block-weight-sweep-language-diagnostic.csv"
    )
    summary[
        [
            "representation",
            "static_block",
            "alpha",
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
        args.output_dir / "block-weight-sweep-report.md"
    )
    write_report(
        path=report_path,
        df=df,
        summary=summary,
        pareto=pareto,
        operational=operational,
        threshold=args.threshold,
    )

    manifest = {
        "threshold_percent": args.threshold,
        "eligible_functions": len(df),
        "paper5_features": PAPER5,
        "static_blocks": STATIC_BLOCKS,
        "selected_static_blocks": blocks,
        "scalers": scalers,
        "k_values": ks,
        "static_weights": alphas,
        "seeds": seeds,
        "n_init": args.n_init,
        "block_normalization": (
            "Each separately scaled block is divided by sqrt(number of "
            "features); alpha multiplies only the static block."
        ),
        "selection_rule": (
            "Internal Pareto uses silhouette, Davies-Bouldin, max cluster "
            "share, singleton count and multi-seed ARI. CH is supporting only. "
            "Language and architecture alignment are diagnostics only. "
            "Donor-operational shortlist requires min cluster size >= 2 "
            "in every seed."
        ),
        "inputs": {
            "profiles": {
                "path": str(args.profiles.resolve()),
                "sha256": sha256_file(args.profiles.resolve()),
            },
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
        args.output_dir / "block-weight-sweep-manifest.json"
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
                "representation",
                "alpha",
                "scaler",
                "k",
                "silhouette_mean",
                "davies_bouldin_mean",
                "calinski_harabasz_mean",
                "max_cluster_share_mean",
                "singleton_count_mean",
                "mean_pairwise_seed_ari",
                "language_nmi_mean",
                "architecture_nmi_mean",
            ]
        ]
        .head(args.top_n)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("DONOR-OPERATIONAL (NO SINGLETONS)")
    print(
        operational[
            [
                "representation",
                "alpha",
                "scaler",
                "k",
                "silhouette_mean",
                "davies_bouldin_mean",
                "max_cluster_share_mean",
                "min_cluster_size_min",
                "mean_pairwise_seed_ari",
                "language_nmi_mean",
                "architecture_nmi_mean",
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
    print(f"operational={operational_path}")
    print(f"language={language_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep block-weighted PAPER-5 plus selected static feature blocks."
        )
    )

    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--static-metrics", type=Path, required=True)
    parser.add_argument("--language-map", type=Path, required=True)
    parser.add_argument("--preferences-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)

    parser.add_argument(
        "--static-blocks",
        default="ccn_tokens,ccn_functions,tokens_functions,static4",
    )
    parser.add_argument(
        "--static-weights",
        default="0.1,0.25,0.5,1.0",
    )
    parser.add_argument(
        "--scalers",
        default="minmax,standard",
    )
    parser.add_argument(
        "--k-values",
        default="5,6,7,8,9",
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
