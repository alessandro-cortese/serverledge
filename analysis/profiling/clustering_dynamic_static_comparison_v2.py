#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]

DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
}

LANGUAGES = [
    "go",
    "nodejs",
    "python",
]

PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC4 = [
    "static_ccn_mean",
    "static_function_nloc_mean",
    "static_token_count_mean",
    "static_function_count",
]

GROUPS = {
    "paper5": {
        "numeric": PAPER5,
        "language": False,
    },
    "static4": {
        "numeric": STATIC4,
        "language": False,
    },
    "paper5_plus_static4": {
        "numeric": PAPER5 + STATIC4,
        "language": False,
    },
    "paper5_plus_language": {
        "numeric": PAPER5,
        "language": True,
    },
    "static4_plus_language": {
        "numeric": STATIC4,
        "language": True,
    },
    "paper5_plus_static4_plus_language": {
        "numeric": PAPER5 + STATIC4,
        "language": True,
    },
}


def threshold_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def parse_csv_ints(value: str) -> list[int]:
    return [
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    ]


def parse_csv_floats(value: str) -> list[float]:
    return [
        float(x.strip())
        for x in value.split(",")
        if x.strip()
    ]


def parse_csv_strings(value: str) -> list[str]:
    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def source_available(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def scaler_from_name(name: str):
    if name == "standard":
        return StandardScaler()

    if name == "minmax":
        return MinMaxScaler()

    raise ValueError(
        f"Unsupported scaler {name!r}. "
        "Expected standard or minmax."
    )


def one_hot_language(frame: pd.DataFrame) -> np.ndarray:
    languages = frame["language"].astype(str).str.strip().to_numpy()

    unknown = sorted(set(languages) - set(LANGUAGES))

    if unknown:
        raise ValueError(
            f"Unknown language values: {unknown}"
        )

    out = np.zeros(
        (len(frame), len(LANGUAGES)),
        dtype=float,
    )

    index = {
        language: i
        for i, language in enumerate(LANGUAGES)
    }

    for row, language in enumerate(languages):
        out[row, index[language]] = 1.0

    return out


def feature_matrices(
        train: pd.DataFrame,
        target: pd.DataFrame | None,
        group: str,
        scaler_name: str,
        language_weight: float,
):
    spec = GROUPS[group]
    numeric = spec["numeric"]

    scaler = scaler_from_name(scaler_name)

    x_train_numeric = scaler.fit_transform(
        train[numeric].astype(float)
    )

    if not np.isfinite(x_train_numeric).all():
        raise ValueError(
            f"Non-finite train values for group={group}"
        )

    x_target_numeric = None

    if target is not None:
        x_target_numeric = scaler.transform(
            target[numeric].astype(float)
        )

        if not np.isfinite(x_target_numeric).all():
            raise ValueError(
                f"Non-finite target values for group={group}"
            )

    if not spec["language"]:
        return x_train_numeric, x_target_numeric

    lang_train = (
            one_hot_language(train)
            * language_weight
    )

    x_train = np.column_stack([
        x_train_numeric,
        lang_train,
    ])

    x_target = None

    if target is not None:
        lang_target = (
                one_hot_language(target)
                * language_weight
        )

        x_target = np.column_stack([
            x_target_numeric,
            lang_target,
        ])

    return x_train, x_target


def unique_majority(labels: pd.Series) -> str:
    counts = Counter(labels.tolist())

    top = max(counts.values())

    winners = [
        label
        for label in LABELS
        if counts.get(label, 0) == top
    ]

    if len(winners) == 1:
        return winners[0]

    return "abstain"


def classification_metrics(
        y: np.ndarray,
        prediction: np.ndarray,
) -> dict:
    recalls = {}
    f1s = []

    for label in LABELS:
        truth = y == label
        pred = prediction == label

        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        recalls[label] = recall
        f1s.append(f1)

    directional = np.isin(
        y,
        list(DIRECTIONAL),
    )

    return {
        "coverage": float(
            np.mean(prediction != "abstain")
        ),
        "strict_accuracy": float(
            np.mean(y == prediction)
        ),
        "balanced_accuracy": float(
            np.mean(list(recalls.values()))
        ),
        "macro_f1": float(
            np.mean(f1s)
        ),
        "directional_accuracy": (
            float(
                np.mean(
                    y[directional]
                    == prediction[directional]
                )
            )
            if np.any(directional)
            else 0.0
        ),
        "x86_recall": recalls["x86-preferred"],
        "independent_recall": recalls[
            "architecture-independent"
        ],
        "arm_recall": recalls["arm-preferred"],
        "pred_x86": int(
            np.sum(
                prediction == "x86-preferred"
            )
        ),
        "pred_independent": int(
            np.sum(
                prediction
                == "architecture-independent"
            )
        ),
        "pred_arm": int(
            np.sum(
                prediction == "arm-preferred"
            )
        ),
        "pred_abstain": int(
            np.sum(prediction == "abstain")
        ),
    }


def lofo(
        df: pd.DataFrame,
        group: str,
        scaler_name: str,
        language_weight: float,
        k: int,
        seed: int,
        n_init: int,
):
    true = []
    prediction = []
    target_cluster_sizes = []

    for idx in range(len(df)):
        train = df.drop(index=idx)
        target = df.iloc[[idx]]

        x_train, x_target = feature_matrices(
            train=train,
            target=target,
            group=group,
            scaler_name=scaler_name,
            language_weight=language_weight,
        )

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        )

        train_clusters = model.fit_predict(
            x_train
        )

        target_cluster = int(
            model.predict(x_target)[0]
        )

        members = np.flatnonzero(
            train_clusters == target_cluster
        )

        predicted_label = unique_majority(
            train.iloc[members][
                "architecture_preference"
            ]
        )

        true.append(
            str(
                target.iloc[0][
                    "architecture_preference"
                ]
            )
        )

        prediction.append(predicted_label)
        target_cluster_sizes.append(
            len(members)
        )

    y = np.asarray(
        true,
        dtype=object,
    )

    p = np.asarray(
        prediction,
        dtype=object,
    )

    result = classification_metrics(
        y,
        p,
    )

    result["mean_target_cluster_size"] = (
        float(
            np.mean(
                target_cluster_sizes
            )
        )
    )

    result["large_cluster_target_share"] = (
        float(
            np.mean(
                np.asarray(
                    target_cluster_sizes
                ) >= 20
            )
        )
    )

    return result


def full_fit(
        df: pd.DataFrame,
        group: str,
        scaler_name: str,
        language_weight: float,
        k: int,
        seed: int,
        n_init: int,
):
    x, _ = feature_matrices(
        train=df,
        target=None,
        group=group,
        scaler_name=scaler_name,
        language_weight=language_weight,
    )

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=seed,
    )

    clusters = model.fit_predict(x)

    sizes = (
        pd.Series(clusters)
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    labels = (
        df[
            "architecture_preference"
        ]
        .to_numpy()
    )

    correct = 0

    for cluster in sorted(
            set(clusters)
    ):
        cluster_labels = labels[
            clusters == cluster
            ]

        correct += max(
            Counter(
                cluster_labels.tolist()
            ).values()
        )

    return {
        "full_purity": float(
            correct / len(df)
        ),
        "silhouette": float(
            silhouette_score(
                x,
                clusters,
            )
        ),
        "davies_bouldin": float(
            davies_bouldin_score(
                x,
                clusters,
            )
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(
                x,
                clusters,
            )
        ),
        "min_cluster_size": int(
            sizes.min()
        ),
        "max_cluster_size": int(
            sizes.max()
        ),
        "max_cluster_share": float(
            sizes.max() / len(df)
        ),
        "singleton_count": int(
            np.sum(
                sizes.to_numpy() == 1
            )
        ),
        "cluster_size_distribution": "|".join(
            str(int(v))
            for v in sizes.to_numpy()
        ),
    }


def seed_stability(
        df: pd.DataFrame,
        group: str,
        scaler_name: str,
        language_weight: float,
        k: int,
        seeds: list[int],
        n_init: int,
) -> float:
    x, _ = feature_matrices(
        train=df,
        target=None,
        group=group,
        scaler_name=scaler_name,
        language_weight=language_weight,
    )

    partitions = []

    for seed in seeds:
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        )

        partitions.append(
            model.fit_predict(x)
        )

    values = []

    for i in range(
            len(partitions)
    ):
        for j in range(
                i + 1,
                len(partitions),
        ):
            values.append(
                adjusted_rand_score(
                    partitions[i],
                    partitions[j],
                )
            )

    return (
        float(np.mean(values))
        if values
        else 1.0
    )


def pareto_flags(
        summary: pd.DataFrame,
) -> pd.Series:
    values = summary[
        [
            "silhouette_mean",
            "davies_bouldin_mean",
            "max_cluster_share_mean",
            "singleton_count_mean",
            "mean_pairwise_seed_ari",
        ]
    ].to_numpy(float)

    result = []

    for i in range(len(values)):
        a = values[i]

        dominated = False

        for j in range(len(values)):
            if i == j:
                continue

            b = values[j]

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

            if (
                    no_worse
                    and strictly_better
            ):
                dominated = True
                break

        result.append(
            not dominated
        )

    return pd.Series(
        result,
        index=summary.index,
    )


def main(args):
    profiles = pd.read_csv(
        args.profiles.resolve()
    )

    static = pd.read_csv(
        args.static_metrics.resolve()
    )

    languages = pd.read_csv(
        args.language_map.resolve()
    )

    prefs_path = (
            args.preferences_dir.resolve()
            / (
                "preferences-"
                f"{threshold_slug(args.threshold)}"
                ".csv"
            )
    )

    prefs = pd.read_csv(
        prefs_path
    )

    needed_static = [
        "function_name",
        "static_source_available",
        *STATIC4,
    ]

    df = (
        profiles.merge(
            static[needed_static],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            languages[
                [
                    "function_name",
                    "language",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            prefs[
                [
                    "function_name",
                    "architecture_preference",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    available = source_available(
        df["static_source_available"]
    )

    complete = available.copy()

    for column in STATIC4:
        complete &= (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .notna()
        )

    df = (
        df[complete]
        .copy()
        .sort_values(
            "function_name"
        )
        .reset_index(drop=True)
    )

    if len(df) != 59:
        raise SystemExit(
            "Expected exactly 59 "
            "static-eligible functions, "
            f"got {len(df)}"
        )

    for column in PAPER5 + STATIC4:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    unknown_languages = (
            set(
                df["language"]
                .astype(str)
                .str.strip()
            )
            - set(LANGUAGES)
    )

    if unknown_languages:
        raise SystemExit(
            "Unexpected languages: "
            f"{sorted(unknown_languages)}"
        )

    ks = parse_csv_ints(
        args.k_values
    )

    seeds = parse_csv_ints(
        args.seeds
    )

    scalers = parse_csv_strings(
        args.scalers
    )

    language_weights = (
        parse_csv_floats(
            args.language_weights
        )
    )

    groups = parse_csv_strings(
        args.groups
    )

    for group in groups:
        if group not in GROUPS:
            raise SystemExit(
                f"Unknown group: {group}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eligible_path = (
            args.output_dir
            / "eligible-functions.csv"
    )

    df[
        [
            "function_name",
            "language",
            "architecture_preference",
        ]
    ].to_csv(
        eligible_path,
        index=False,
    )

    print(
        f"eligible_functions={len(df)}"
    )

    print("\nLANGUAGE COUNTS:")
    print(
        df["language"]
        .value_counts()
        .to_string()
    )

    print("\nCLASS COUNTS:")
    print(
        df[
            "architecture_preference"
        ]
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0,
        )
        .to_string()
    )

    majority_baseline = float(
        df[
            "architecture_preference"
        ]
        .value_counts()
        .max()
        / len(df)
    )

    configurations = []

    for group in groups:
        if GROUPS[group]["language"]:
            weights = (
                language_weights
            )
        else:
            weights = [0.0]

        for scaler_name in scalers:
            for weight in weights:
                for k in ks:
                    configurations.append(
                        (
                            group,
                            scaler_name,
                            weight,
                            k,
                        )
                    )

    total = (
            len(configurations)
            * len(seeds)
    )

    rows = []
    done = 0

    for (
            group,
            scaler_name,
            language_weight,
            k,
    ) in configurations:

        stability = seed_stability(
            df=df,
            group=group,
            scaler_name=scaler_name,
            language_weight=language_weight,
            k=k,
            seeds=seeds,
            n_init=args.n_init,
        )

        for seed in seeds:
            external = lofo(
                df=df,
                group=group,
                scaler_name=scaler_name,
                language_weight=language_weight,
                k=k,
                seed=seed,
                n_init=args.n_init,
            )

            internal = full_fit(
                df=df,
                group=group,
                scaler_name=scaler_name,
                language_weight=language_weight,
                k=k,
                seed=seed,
                n_init=args.n_init,
            )

            rows.append(
                {
                    "feature_group": group,
                    "scaler": scaler_name,
                    "k": k,
                    "language_weight": (
                        language_weight
                    ),
                    "seed": seed,
                    "majority_baseline": (
                        majority_baseline
                    ),
                    "strict_gain_over_majority_baseline": (
                            external[
                                "strict_accuracy"
                            ]
                            - majority_baseline
                    ),
                    "mean_pairwise_seed_ari": (
                        stability
                    ),
                    **external,
                    **internal,
                }
            )

            done += 1

            if (
                    done % 25 == 0
                    or done == total
            ):
                print(
                    f"progress={done}/{total}"
                )

    detail = pd.DataFrame(rows)

    detail_path = (
            args.output_dir
            / "dynamic-static-v2-detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = (
        detail.groupby(
            [
                "feature_group",
                "scaler",
                "k",
                "language_weight",
            ],
            as_index=False,
        )
        .agg(
            coverage_mean=(
                "coverage",
                "mean",
            ),
            strict_accuracy_mean=(
                "strict_accuracy",
                "mean",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            directional_accuracy_mean=(
                "directional_accuracy",
                "mean",
            ),
            full_purity_mean=(
                "full_purity",
                "mean",
            ),
            silhouette_mean=(
                "silhouette",
                "mean",
            ),
            silhouette_std=(
                "silhouette",
                "std",
            ),
            davies_bouldin_mean=(
                "davies_bouldin",
                "mean",
            ),
            calinski_harabasz_mean=(
                "calinski_harabasz",
                "mean",
            ),
            min_cluster_size_mean=(
                "min_cluster_size",
                "mean",
            ),
            max_cluster_size_mean=(
                "max_cluster_size",
                "mean",
            ),
            max_cluster_share_mean=(
                "max_cluster_share",
                "mean",
            ),
            singleton_count_mean=(
                "singleton_count",
                "mean",
            ),
            mean_pairwise_seed_ari=(
                "mean_pairwise_seed_ari",
                "first",
            ),
        )
    )

    summary[
        "internal_pareto"
    ] = pareto_flags(summary)

    summary_path = (
            args.output_dir
            / "dynamic-static-v2-summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pareto = (
        summary[
            summary["internal_pareto"]
        ]
        .copy()
        .sort_values(
            [
                "max_cluster_share_mean",
                "singleton_count_mean",
                "davies_bouldin_mean",
                "silhouette_mean",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
    )

    pareto_path = (
            args.output_dir
            / "dynamic-static-v2-pareto.csv"
    )

    pareto.to_csv(
        pareto_path,
        index=False,
    )

    manifest = {
        "threshold_percent": (
            args.threshold
        ),
        "eligible_functions": (
            len(df)
        ),
        "paper5_features": PAPER5,
        "static4_features": STATIC4,
        "language_categories": LANGUAGES,
        "groups": groups,
        "scalers": scalers,
        "k_values": ks,
        "language_weights": (
            language_weights
        ),
        "seeds": seeds,
        "n_init": args.n_init,
        "selection_note": (
            "Internal Pareto status excludes "
            "ground-truth metrics. "
            "Architecture labels are post-hoc "
            "evaluation only."
        ),
    }

    manifest_path = (
            args.output_dir
            / "dynamic-static-v2-manifest.json"
    )

    with manifest_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    columns = [
        "feature_group",
        "scaler",
        "k",
        "language_weight",
        "silhouette_mean",
        "davies_bouldin_mean",
        "calinski_harabasz_mean",
        "max_cluster_share_mean",
        "singleton_count_mean",
        "mean_pairwise_seed_ari",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "directional_accuracy_mean",
    ]

    print(
        "\nINTERNAL PARETO "
        "(ground truth NOT used for selection):"
    )

    print(
        pareto[columns]
        .head(args.top_n)
        .to_string(index=False)
    )

    print(
        "\nOutputs:"
    )
    print(detail_path)
    print(summary_path)
    print(pareto_path)
    print(manifest_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PAPER-5, cross-language "
            "static source metrics, and language "
            "on the same static-eligible corpus."
        )
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        required=True,
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
        "--groups",
        default=",".join(GROUPS.keys()),
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
        "--language-weights",
        default="0.25,0.5,1.0",
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
    main(
        build_parser().parse_args()
    )