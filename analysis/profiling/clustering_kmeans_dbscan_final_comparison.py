#!/usr/bin/env python3
"""
Final K-Means vs DBSCAN operational comparison for Serverledge.

Purpose
-------
This experiment is intentionally narrow. It compares the three representations
that survived the previous screening:

1. PAPER-5 dynamic baseline
2. PAPER-5 + {token_count_mean, function_count}
3. PAPER-5 + STATIC4

The comparison uses the professor-aligned directional-majority donor rule and
the downstream metrics already selected:
- directional preference agreement
- |Delta_target - Delta_donor|
- multiplicative ARM/x86 ratio mismatch
- realized speedup/slowdown
- response-time regret

Critical fairness rule
----------------------
Clustering geometry and donor ranking are DECOUPLED.

The clustering algorithm uses its own preprocessing:
- K-Means: the already shortlisted scaler/K configuration
- DBSCAN: RobustScaler + cosine, min_samples=4

The donor ranking ALWAYS uses a separately fitted MinMax representation and
Manhattan distance, independently of the clustering scaler/metric. This matches
the chosen donor-distance methodology and avoids confounding the algorithm
comparison with a different donor-ranking geometry.

DBSCAN protocol
---------------
Primary setting:
- RobustScaler
- cosine distance
- min_samples=4
- eps = 80th percentile of the min_samples-neighbour distance distribution

Sensitivity only:
- q75 and q85

Out-of-sample assignment for the held-out target:
- find the nearest DBSCAN core sample by cosine distance;
- assign the target to that core sample's cluster only if distance <= eps;
- otherwise the target is treated as noise/unassigned.

The q80 setting is predeclared as the primary DBSCAN comparison. q75/q85 are
reported as sensitivity and are not selected using architecture ground truth.

Architecture-independent historical functions remain cluster members but:
- do not vote in the x86-vs-ARM majority;
- are not eligible donors.

If x86 and ARM directional votes tie, or no directional member exists, donor
selection abstains.

Outputs
-------
final-algorithm-comparison-per-target.csv
final-algorithm-comparison-per-replicate.csv
final-algorithm-comparison-summary.csv
final-algorithm-comparison-primary.csv
final-algorithm-comparison-report.md
final-algorithm-comparison-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC_BLOCKS = {
    "tokens_functions": [
        "static_token_count_mean",
        "static_function_count",
    ],
    "static4": [
        "static_ccn_mean",
        "static_function_nloc_mean",
        "static_token_count_mean",
        "static_function_count",
    ],
}

REPRESENTATIONS = {
    "paper5": {
        "kind": "dynamic",
        "static_block": None,
        "alpha": 0.0,
        "kmeans_scaler": "minmax",
        "kmeans_k": 5,
    },
    "hybrid_tokens_functions": {
        "kind": "hybrid",
        "static_block": "tokens_functions",
        "alpha": 1.0,
        "kmeans_scaler": "minmax",
        "kmeans_k": 6,
    },
    "hybrid_static4": {
        "kind": "hybrid",
        "static_block": "static4",
        "alpha": 1.0,
        "kmeans_scaler": "standard",
        "kmeans_k": 9,
    },
}

THRESHOLDS = [
    (2.5, "2p5"),
    (5.0, "5"),
    (10.0, "10"),
    (15.0, "15"),
    (20.0, "20"),
    (25.0, "25"),
]

DIRECTIONAL = {
    "x86-preferred",
    "arm-preferred",
}

DBSCAN_SCALER = "robust"
DBSCAN_METRIC = "cosine"
DBSCAN_MIN_SAMPLES = 4
DBSCAN_QUANTILES = [0.75, 0.80, 0.85]
DBSCAN_PRIMARY_QUANTILE = 0.80
DONOR_SCALER = "minmax"
DONOR_METRIC = "manhattan"
EPS_TOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def fit_scaled_block(
    train: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    scaler_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = scaler_from_name(scaler_name)

    x_train = scaler.fit_transform(
        train[columns].astype(float)
    )
    x_target = scaler.transform(
        target[columns].astype(float)
    )

    if not np.isfinite(x_train).all():
        raise ValueError(
            f"Non-finite training representation for {columns}"
        )
    if not np.isfinite(x_target).all():
        raise ValueError(
            f"Non-finite target representation for {columns}"
        )

    return x_train, x_target


def build_representation(
    train: pd.DataFrame,
    target: pd.DataFrame,
    representation_name: str,
    scaler_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    config = REPRESENTATIONS[representation_name]

    dyn_train, dyn_target = fit_scaled_block(
        train,
        target,
        PAPER5,
        scaler_name,
    )

    dyn_train = dyn_train / math.sqrt(len(PAPER5))
    dyn_target = dyn_target / math.sqrt(len(PAPER5))

    if config["kind"] == "dynamic":
        return dyn_train, dyn_target

    static_columns = STATIC_BLOCKS[
        config["static_block"]
    ]

    stat_train, stat_target = fit_scaled_block(
        train,
        target,
        static_columns,
        scaler_name,
    )

    alpha = float(config["alpha"])

    stat_train = (
        stat_train
        / math.sqrt(len(static_columns))
        * alpha
    )
    stat_target = (
        stat_target
        / math.sqrt(len(static_columns))
        * alpha
    )

    return (
        np.column_stack([dyn_train, stat_train]),
        np.column_stack([dyn_target, stat_target]),
    )


def prepare_data(args):
    profiles = pd.read_csv(args.profiles.resolve())
    static = pd.read_csv(args.static_metrics.resolve())
    languages = pd.read_csv(args.language_map.resolve())

    all_static = sorted(
        {
            feature
            for block in STATIC_BLOCKS.values()
            for feature in block
        }
    )

    base_pref_path = (
        args.preferences_dir.resolve()
        / "preferences-2p5.csv"
    )
    base_pref = pd.read_csv(base_pref_path)

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
            languages[
                ["function_name", "language"]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            base_pref[
                [
                    "function_name",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    complete = source_available(
        df["static_source_available"]
    )

    for column in PAPER5 + all_static:
        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )
        df[column] = numeric

        if column in all_static:
            complete &= numeric.notna()

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
            raise SystemExit(
                f"Missing numeric values in {column}"
            )

    pref_maps = {}
    pref_paths = {}

    for threshold, slug in THRESHOLDS:
        path = (
            args.preferences_dir.resolve()
            / f"preferences-{slug}.csv"
        )
        table = pd.read_csv(path)

        required = {
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
            "arm_vs_x86_delta_percent",
        }
        missing = required - set(table.columns)
        if missing:
            raise SystemExit(
                f"{path}: missing columns {sorted(missing)}"
            )

        pref_maps[threshold] = (
            table.set_index("function_name")[
                [
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ].to_dict(orient="index")
        )
        pref_paths[slug] = path

    return df, pref_maps, pref_paths


def nearest_manhattan(
    target_vector: np.ndarray,
    train_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    names: np.ndarray,
) -> tuple[int, float]:
    if len(candidate_indices) == 0:
        raise ValueError("Empty donor candidate pool")

    distances = np.sum(
        np.abs(
            train_matrix[candidate_indices]
            - target_vector.reshape(1, -1)
        ),
        axis=1,
    )

    # deterministic tie-break
    order = np.lexsort(
        (
            names[candidate_indices],
            distances,
        )
    )

    local = int(order[0])
    idx = int(candidate_indices[local])

    return idx, float(distances[local])


def directional_majority(
    labels: np.ndarray,
    cluster_indices: np.ndarray,
) -> tuple[str | None, int, int, int]:
    cluster_labels = labels[cluster_indices]

    x86_count = int(
        np.sum(
            cluster_labels == "x86-preferred"
        )
    )
    arm_count = int(
        np.sum(
            cluster_labels == "arm-preferred"
        )
    )
    independent_count = int(
        np.sum(
            cluster_labels
            == "architecture-independent"
        )
    )

    if x86_count == 0 and arm_count == 0:
        return (
            None,
            x86_count,
            independent_count,
            arm_count,
        )

    if x86_count == arm_count:
        return (
            None,
            x86_count,
            independent_count,
            arm_count,
        )

    majority = (
        "x86-preferred"
        if x86_count > arm_count
        else "arm-preferred"
    )

    return (
        majority,
        x86_count,
        independent_count,
        arm_count,
    )


def dbscan_eps(
    x_train: np.ndarray,
    quantile: float,
) -> float:
    nn = NearestNeighbors(
        n_neighbors=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC,
    )
    nn.fit(x_train)

    distances, _ = nn.kneighbors(x_train)

    kdist = distances[:, -1]
    eps = float(
        np.quantile(
            kdist,
            quantile,
        )
    )

    return max(eps, 1e-12)


def dbscan_target_assignment(
    model: DBSCAN,
    x_train: np.ndarray,
    x_target: np.ndarray,
    train_names: np.ndarray,
    eps: float,
) -> tuple[int | None, float | None]:
    core_indices = np.asarray(
        model.core_sample_indices_,
        dtype=int,
    )

    if len(core_indices) == 0:
        return None, None

    core_vectors = x_train[core_indices]

    distances = pairwise_distances(
        x_target.reshape(1, -1),
        core_vectors,
        metric=DBSCAN_METRIC,
    )[0]

    # deterministic nearest-core tie-break by function name
    order = np.lexsort(
        (
            train_names[core_indices],
            distances,
        )
    )

    local = int(order[0])
    core_idx = int(core_indices[local])
    distance = float(distances[local])

    if distance > eps + EPS_TOL:
        return None, distance

    label = int(model.labels_[core_idx])

    if label < 0:
        raise RuntimeError(
            "DBSCAN core sample unexpectedly labelled as noise"
        )

    return label, distance


def raw_best_architecture(
    x86_ms: float,
    arm_ms: float,
) -> tuple[str, float]:
    if x86_ms <= arm_ms:
        return "amd64", x86_ms
    return "arm64", arm_ms


def outcome_metrics(
    target_info: dict,
    donor_info: dict,
) -> dict:
    target_label = str(
        target_info["architecture_preference"]
    )
    donor_label = str(
        donor_info["architecture_preference"]
    )

    if donor_label not in DIRECTIONAL:
        raise RuntimeError(
            "Directional-majority donor is not directional"
        )

    tx = float(target_info["x86_duration_ms"])
    ta = float(target_info["arm_duration_ms"])
    dx = float(donor_info["x86_duration_ms"])
    da = float(donor_info["arm_duration_ms"])

    target_delta = float(
        target_info["arm_vs_x86_delta_percent"]
    )
    donor_delta = float(
        donor_info["arm_vs_x86_delta_percent"]
    )

    target_ratio = ta / tx
    donor_ratio = da / dx

    log_mismatch = abs(
        math.log(target_ratio)
        - math.log(donor_ratio)
    )

    multiplicative_mismatch = math.exp(
        log_mismatch
    )

    target_directional = (
        target_label in DIRECTIONAL
    )

    directional_agreement = (
        donor_label == target_label
        if target_directional
        else np.nan
    )

    if donor_label == "x86-preferred":
        chosen_arch = "amd64"
        chosen_ms = tx
        alternative_ms = ta
    else:
        chosen_arch = "arm64"
        chosen_ms = ta
        alternative_ms = tx

    realized_speedup = (
        alternative_ms / chosen_ms
    )

    realized_latency_change_percent = (
        (alternative_ms - chosen_ms)
        / alternative_ms
        * 100.0
    )

    optimal_arch, optimal_ms = (
        raw_best_architecture(tx, ta)
    )

    regret_ms = chosen_ms - optimal_ms
    regret_percent = (
        regret_ms / optimal_ms * 100.0
    )

    return {
        "target_is_directional": target_directional,
        "directional_preference_agreement": (
            directional_agreement
        ),
        "target_delta_percent": target_delta,
        "donor_delta_percent": donor_delta,
        "abs_delta_error_percent": abs(
            target_delta - donor_delta
        ),
        "target_ratio_arm_over_x86": target_ratio,
        "donor_ratio_arm_over_x86": donor_ratio,
        "log_ratio_mismatch": log_mismatch,
        "multiplicative_mismatch_factor": (
            multiplicative_mismatch
        ),
        "chosen_architecture": chosen_arch,
        "chosen_duration_ms": chosen_ms,
        "alternative_duration_ms": alternative_ms,
        "realized_speedup_factor": (
            realized_speedup
        ),
        "realized_latency_change_percent": (
            realized_latency_change_percent
        ),
        "optimal_architecture_raw": (
            optimal_arch
        ),
        "optimal_duration_ms": optimal_ms,
        "regret_ms": regret_ms,
        "regret_percent": regret_percent,
        "zero_regret": (
            regret_percent <= EPS_TOL
        ),
    }


def geometric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().to_numpy(float)

    values = values[
        values > 0
    ]

    if not len(values):
        return np.nan

    return float(
        np.exp(
            np.mean(
                np.log(values)
            )
        )
    )


def q90(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().to_numpy(float)

    if not len(values):
        return np.nan

    return float(
        np.quantile(
            values,
            0.90,
        )
    )


def run_one_partition(
    *,
    algorithm: str,
    algorithm_config: str,
    representation_name: str,
    replicate: str,
    seed: int | None,
    dbscan_quantile: float | None,
    train: pd.DataFrame,
    target: pd.DataFrame,
    pref_maps,
) -> list[dict]:
    rep_config = REPRESENTATIONS[
        representation_name
    ]

    train_names = (
        train["function_name"]
        .astype(str)
        .to_numpy()
    )

    target_name = str(
        target.iloc[0]["function_name"]
    )

    # Clustering space.
    if algorithm == "kmeans":
        cluster_train, cluster_target = (
            build_representation(
                train,
                target,
                representation_name,
                rep_config["kmeans_scaler"],
            )
        )

        model = KMeans(
            n_clusters=int(
                rep_config["kmeans_k"]
            ),
            n_init=50,
            random_state=int(seed),
        )

        train_labels = model.fit_predict(
            cluster_train
        )

        target_cluster = int(
            model.predict(
                cluster_target
            )[0]
        )

        target_assignment_distance = (
            float(
                np.min(
                    np.linalg.norm(
                        model.cluster_centers_
                        - cluster_target[0],
                        axis=1,
                    )
                )
            )
        )

        training_noise_fraction = 0.0
        training_cluster_count = int(
            np.unique(train_labels).size
        )
        eps_value = np.nan

    elif algorithm == "dbscan":
        cluster_train, cluster_target = (
            build_representation(
                train,
                target,
                representation_name,
                DBSCAN_SCALER,
            )
        )

        eps_value = dbscan_eps(
            cluster_train,
            float(dbscan_quantile),
        )

        model = DBSCAN(
            eps=eps_value,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric=DBSCAN_METRIC,
        )

        train_labels = model.fit_predict(
            cluster_train
        )

        non_noise_labels = (
            train_labels[
                train_labels >= 0
            ]
        )

        training_cluster_count = int(
            np.unique(
                non_noise_labels
            ).size
        )

        training_noise_fraction = float(
            np.mean(
                train_labels == -1
            )
        )

        (
            target_cluster,
            target_assignment_distance,
        ) = dbscan_target_assignment(
            model=model,
            x_train=cluster_train,
            x_target=cluster_target[0],
            train_names=train_names,
            eps=eps_value,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm {algorithm}"
        )

    # Donor ranking space is deliberately independent of clustering.
    donor_train, donor_target = (
        build_representation(
            train,
            target,
            representation_name,
            DONOR_SCALER,
        )
    )

    rows = []

    for threshold, _slug in THRESHOLDS:
        target_info = pref_maps[
            threshold
        ][target_name]

        target_pref = str(
            target_info[
                "architecture_preference"
            ]
        )

        base = {
            "algorithm": algorithm,
            "algorithm_config": algorithm_config,
            "representation": representation_name,
            "replicate": replicate,
            "seed": (
                int(seed)
                if seed is not None
                else np.nan
            ),
            "dbscan_quantile": (
                float(dbscan_quantile)
                if dbscan_quantile is not None
                else np.nan
            ),
            "dbscan_eps": eps_value,
            "threshold_percent": threshold,
            "target_function": target_name,
            "target_language": str(
                target.iloc[0]["language"]
            ),
            "target_preference_label": (
                target_pref
            ),
            "target_is_directional": (
                target_pref in DIRECTIONAL
            ),
            "training_cluster_count": (
                training_cluster_count
            ),
            "training_noise_fraction": (
                training_noise_fraction
            ),
            "target_assignment_distance": (
                target_assignment_distance
                if target_assignment_distance
                is not None
                else np.nan
            ),
        }

        if target_cluster is None:
            rows.append(
                {
                    **base,
                    "cluster_assignment_status": (
                        "unassigned_noise"
                    ),
                    "target_cluster": -1,
                    "same_cluster_size": 0,
                    "x86_cluster_count": 0,
                    "independent_cluster_count": 0,
                    "arm_cluster_count": 0,
                    "directional_majority_label": "",
                    "selection_status": (
                        "abstain_unassigned_noise"
                    ),
                    "candidate_pool_size": 0,
                    "donor_function": "",
                    "donor_language": "",
                    "donor_distance": np.nan,
                    "donor_preference_label": "",
                    "directional_preference_agreement": np.nan,
                    "abs_delta_error_percent": np.nan,
                    "multiplicative_mismatch_factor": np.nan,
                    "realized_speedup_factor": np.nan,
                    "realized_latency_change_percent": np.nan,
                    "regret_ms": np.nan,
                    "regret_percent": np.nan,
                    "zero_regret": np.nan,
                }
            )
            continue

        same_cluster = np.flatnonzero(
            train_labels
            == target_cluster
        )

        if len(same_cluster) == 0:
            raise RuntimeError(
                "Assigned target cluster has no historical members"
            )

        labels = np.asarray(
            [
                pref_maps[threshold][name][
                    "architecture_preference"
                ]
                for name in train_names
            ],
            dtype=object,
        )

        (
            majority_label,
            x86_count,
            independent_count,
            arm_count,
        ) = directional_majority(
            labels,
            same_cluster,
        )

        if majority_label is None:
            status = (
                "abstain_no_directional"
                if x86_count == 0
                and arm_count == 0
                else "abstain_directional_tie"
            )

            rows.append(
                {
                    **base,
                    "cluster_assignment_status": (
                        "assigned"
                    ),
                    "target_cluster": int(
                        target_cluster
                    ),
                    "same_cluster_size": int(
                        len(same_cluster)
                    ),
                    "x86_cluster_count": (
                        x86_count
                    ),
                    "independent_cluster_count": (
                        independent_count
                    ),
                    "arm_cluster_count": (
                        arm_count
                    ),
                    "directional_majority_label": "",
                    "selection_status": status,
                    "candidate_pool_size": 0,
                    "donor_function": "",
                    "donor_language": "",
                    "donor_distance": np.nan,
                    "donor_preference_label": "",
                    "directional_preference_agreement": np.nan,
                    "abs_delta_error_percent": np.nan,
                    "multiplicative_mismatch_factor": np.nan,
                    "realized_speedup_factor": np.nan,
                    "realized_latency_change_percent": np.nan,
                    "regret_ms": np.nan,
                    "regret_percent": np.nan,
                    "zero_regret": np.nan,
                }
            )
            continue

        candidate_indices = (
            same_cluster[
                labels[same_cluster]
                == majority_label
            ]
        )

        donor_idx, donor_distance = (
            nearest_manhattan(
                donor_target[0],
                donor_train,
                candidate_indices,
                train_names,
            )
        )

        donor_name = str(
            train.iloc[
                donor_idx
            ]["function_name"]
        )

        donor_info = pref_maps[
            threshold
        ][donor_name]

        donor_pref = str(
            donor_info[
                "architecture_preference"
            ]
        )

        if donor_pref not in DIRECTIONAL:
            raise RuntimeError(
                "Invariant violation: selected donor is not directional"
            )

        metrics = outcome_metrics(
            target_info,
            donor_info,
        )

        rows.append(
            {
                **base,
                "cluster_assignment_status": (
                    "assigned"
                ),
                "target_cluster": int(
                    target_cluster
                ),
                "same_cluster_size": int(
                    len(same_cluster)
                ),
                "x86_cluster_count": (
                    x86_count
                ),
                "independent_cluster_count": (
                    independent_count
                ),
                "arm_cluster_count": (
                    arm_count
                ),
                "directional_majority_label": (
                    majority_label
                ),
                "selection_status": "selected",
                "candidate_pool_size": int(
                    len(candidate_indices)
                ),
                "donor_function": donor_name,
                "donor_language": str(
                    train.iloc[
                        donor_idx
                    ]["language"]
                ),
                "donor_distance": (
                    donor_distance
                ),
                "donor_preference_label": (
                    donor_pref
                ),
                **metrics,
            }
        )

    return rows


def summarize_partition(
    group: pd.DataFrame,
) -> dict:
    assigned = group[
        group["cluster_assignment_status"]
        == "assigned"
    ]

    selected = group[
        group["selection_status"]
        == "selected"
    ]

    directional_targets = group[
        group["target_is_directional"]
        == True
    ]

    selected_directional = selected[
        selected["target_is_directional"]
        == True
    ]

    def mean_col(df, col):
        if not len(df):
            return np.nan
        return float(
            pd.to_numeric(
                df[col],
                errors="coerce",
            ).mean()
        )

    def median_col(df, col):
        if not len(df):
            return np.nan
        return float(
            pd.to_numeric(
                df[col],
                errors="coerce",
            ).median()
        )

    return {
        "target_rows": int(len(group)),
        "cluster_assignment_coverage": (
            len(assigned) / len(group)
        ),
        "selection_coverage": (
            len(selected) / len(group)
        ),
        "directional_target_rows": int(
            len(directional_targets)
        ),
        "directional_selection_coverage": (
            len(selected_directional)
            / len(directional_targets)
            if len(directional_targets)
            else np.nan
        ),
        "directional_preference_agreement": (
            mean_col(
                selected_directional,
                "directional_preference_agreement",
            )
        ),
        "mean_abs_delta_error": (
            mean_col(
                selected,
                "abs_delta_error_percent",
            )
        ),
        "directional_mean_abs_delta_error": (
            mean_col(
                selected_directional,
                "abs_delta_error_percent",
            )
        ),
        "geomean_multiplicative_mismatch": (
            geometric_mean(
                selected[
                    "multiplicative_mismatch_factor"
                ]
            )
            if len(selected)
            else np.nan
        ),
        "directional_geomean_multiplicative_mismatch": (
            geometric_mean(
                selected_directional[
                    "multiplicative_mismatch_factor"
                ]
            )
            if len(selected_directional)
            else np.nan
        ),
        "geomean_realized_speedup": (
            geometric_mean(
                selected[
                    "realized_speedup_factor"
                ]
            )
            if len(selected)
            else np.nan
        ),
        "directional_geomean_realized_speedup": (
            geometric_mean(
                selected_directional[
                    "realized_speedup_factor"
                ]
            )
            if len(selected_directional)
            else np.nan
        ),
        "mean_regret_percent": (
            mean_col(
                selected,
                "regret_percent",
            )
        ),
        "directional_mean_regret_percent": (
            mean_col(
                selected_directional,
                "regret_percent",
            )
        ),
        "directional_p90_regret_percent": (
            q90(
                selected_directional[
                    "regret_percent"
                ]
            )
            if len(selected_directional)
            else np.nan
        ),
        "directional_zero_regret_rate": (
            mean_col(
                selected_directional,
                "zero_regret",
            )
        ),
        "mean_candidate_pool_size": (
            mean_col(
                selected,
                "candidate_pool_size",
            )
        ),
        "mean_same_cluster_size": (
            mean_col(
                assigned,
                "same_cluster_size",
            )
        ),
        "mean_training_cluster_count": (
            mean_col(
                group,
                "training_cluster_count",
            )
        ),
        "mean_training_noise_fraction": (
            mean_col(
                group,
                "training_noise_fraction",
            )
        ),
    }


def build_per_replicate(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        algorithm,
        algorithm_config,
        representation,
        replicate,
        threshold,
    ), group in detail.groupby(
        [
            "algorithm",
            "algorithm_config",
            "representation",
            "replicate",
            "threshold_percent",
        ]
    ):
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_config": (
                    algorithm_config
                ),
                "representation": representation,
                "replicate": replicate,
                "threshold_percent": float(
                    threshold
                ),
                **summarize_partition(group),
            }
        )

    return pd.DataFrame(rows)


def build_summary(
    per_replicate: pd.DataFrame,
) -> pd.DataFrame:
    id_cols = {
        "algorithm",
        "algorithm_config",
        "representation",
        "replicate",
        "threshold_percent",
    }

    metric_cols = [
        c for c in per_replicate.columns
        if c not in id_cols
    ]

    agg = {}

    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    summary = (
        per_replicate.groupby(
            [
                "algorithm",
                "algorithm_config",
                "representation",
                "threshold_percent",
            ],
            as_index=False,
        )
        .agg(**agg)
    )

    return summary


def primary_flag(
    algorithm: str,
    config: str,
) -> bool:
    if algorithm == "kmeans":
        return True
    return "_q80_" in config


def write_report(
    path: Path,
    primary: pd.DataFrame,
):
    lines = [
        "# Final K-Means vs DBSCAN operational comparison",
        "",
        "## Fixed methodology",
        "",
        (
            "- K-Means uses the previously shortlisted scaler/K for each "
            "representation."
        ),
        (
            "- DBSCAN primary configuration uses RobustScaler + cosine, "
            "min_samples=4 and q80 k-distance eps."
        ),
        (
            "- q75/q85 are sensitivity only and are not selected from "
            "architecture ground truth."
        ),
        (
            "- Donor ranking is always Manhattan in a separately fitted "
            "MinMax donor space, independently of clustering algorithm."
        ),
        (
            "- architecture-independent functions remain cluster members "
            "but do not vote and cannot be donors."
        ),
        "",
        "## Primary comparison at 2.5% and 15%",
        "",
    ]

    focus = primary[
        primary["threshold_percent"].isin(
            [2.5, 15.0]
        )
    ].sort_values(
        [
            "threshold_percent",
            "directional_preference_agreement_mean",
            "directional_mean_regret_percent_mean",
        ],
        ascending=[True, False, True],
    )

    for _, r in focus.iterrows():
        lines.append(
            f"- t={r['threshold_percent']:.1f}% / "
            f"`{r['algorithm_config']}`: "
            f"assignment={r['cluster_assignment_coverage_mean']:.4f}, "
            f"donor-coverage={r['selection_coverage_mean']:.4f}, "
            f"dir-agreement={r['directional_preference_agreement_mean']:.4f}, "
            f"dir-M="
            f"{r['directional_geomean_multiplicative_mismatch_mean']:.4f}, "
            f"dir-speedup="
            f"{r['directional_geomean_realized_speedup_mean']:.4f}x, "
            f"dir-regret="
            f"{r['directional_mean_regret_percent_mean']:.3f}%, "
            f"noise={r['mean_training_noise_fraction_mean']:.4f}"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        (
            "Do not select the algorithm from one metric alone. K-Means is "
            "favored only if it remains competitive on donor outcomes while "
            "providing higher out-of-sample assignment/selection coverage and "
            "avoiding DBSCAN noise. DBSCAN should be retained if its operational "
            "donor outcomes materially compensate for lower/noisier assignment."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(args):
    df, pref_maps, pref_paths = (
        prepare_data(args)
    )

    seeds = parse_csv_ints(args.seeds)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    # K-Means fixed shortlist.
    k_total = (
        len(REPRESENTATIONS)
        * len(seeds)
        * len(df)
    )
    k_count = 0

    for rep_name, rep_config in (
        REPRESENTATIONS.items()
    ):
        algorithm_config = (
            f"kmeans_{rep_name}_"
            f"{rep_config['kmeans_scaler']}_"
            f"k{rep_config['kmeans_k']}"
        )

        for seed in seeds:
            replicate = f"seed-{seed}"

            for target_idx in range(len(df)):
                target = df.iloc[
                    [target_idx]
                ].copy()

                train = (
                    df.drop(index=target_idx)
                    .copy()
                    .reset_index(drop=True)
                )

                rows.extend(
                    run_one_partition(
                        algorithm="kmeans",
                        algorithm_config=(
                            algorithm_config
                        ),
                        representation_name=(
                            rep_name
                        ),
                        replicate=replicate,
                        seed=seed,
                        dbscan_quantile=None,
                        train=train,
                        target=target,
                        pref_maps=pref_maps,
                    )
                )

                k_count += 1

                if (
                    k_count % 100 == 0
                    or k_count == k_total
                ):
                    print(
                        f"kmeans_progress="
                        f"{k_count}/{k_total}"
                    )

    # DBSCAN primary q80 + q75/q85 sensitivity.
    d_total = (
        len(REPRESENTATIONS)
        * len(DBSCAN_QUANTILES)
        * len(df)
    )
    d_count = 0

    for rep_name in REPRESENTATIONS:
        for quantile in DBSCAN_QUANTILES:
            q_int = int(
                round(
                    quantile * 100
                )
            )

            algorithm_config = (
                f"dbscan_{rep_name}_robust_"
                f"cosine_q{q_int}_m"
                f"{DBSCAN_MIN_SAMPLES}"
            )

            replicate = "deterministic"

            for target_idx in range(len(df)):
                target = df.iloc[
                    [target_idx]
                ].copy()

                train = (
                    df.drop(index=target_idx)
                    .copy()
                    .reset_index(drop=True)
                )

                rows.extend(
                    run_one_partition(
                        algorithm="dbscan",
                        algorithm_config=(
                            algorithm_config
                        ),
                        representation_name=(
                            rep_name
                        ),
                        replicate=replicate,
                        seed=None,
                        dbscan_quantile=(
                            quantile
                        ),
                        train=train,
                        target=target,
                        pref_maps=pref_maps,
                    )
                )

                d_count += 1

                if (
                    d_count % 100 == 0
                    or d_count == d_total
                ):
                    print(
                        f"dbscan_progress="
                        f"{d_count}/{d_total}"
                    )

    detail = pd.DataFrame(rows)

    detail_path = (
        args.output_dir
        / "final-algorithm-comparison-per-target.csv"
    )
    detail.to_csv(
        detail_path,
        index=False,
    )

    per_replicate = build_per_replicate(
        detail
    )

    per_replicate_path = (
        args.output_dir
        / "final-algorithm-comparison-per-replicate.csv"
    )
    per_replicate.to_csv(
        per_replicate_path,
        index=False,
    )

    summary = build_summary(
        per_replicate
    )

    summary["primary_algorithm_setting"] = [
        primary_flag(a, c)
        for a, c in zip(
            summary["algorithm"],
            summary["algorithm_config"],
        )
    ]

    summary_path = (
        args.output_dir
        / "final-algorithm-comparison-summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    primary = summary[
        summary[
            "primary_algorithm_setting"
        ]
    ].copy()

    primary_path = (
        args.output_dir
        / "final-algorithm-comparison-primary.csv"
    )
    primary.to_csv(
        primary_path,
        index=False,
    )

    report_path = (
        args.output_dir
        / "final-algorithm-comparison-report.md"
    )
    write_report(
        report_path,
        primary,
    )

    manifest = {
        "eligible_functions": len(df),
        "representations": REPRESENTATIONS,
        "kmeans_seeds": seeds,
        "kmeans_n_init": 50,
        "dbscan": {
            "scaler": DBSCAN_SCALER,
            "metric": DBSCAN_METRIC,
            "min_samples": (
                DBSCAN_MIN_SAMPLES
            ),
            "eps_rule": (
                "quantile of min_samples-neighbour "
                "distance distribution on LOFO training set"
            ),
            "quantiles": DBSCAN_QUANTILES,
            "primary_quantile": (
                DBSCAN_PRIMARY_QUANTILE
            ),
            "out_of_sample_assignment": (
                "nearest core sample if cosine distance <= eps; "
                "otherwise unassigned/noise"
            ),
        },
        "donor_ranking": {
            "scaler": DONOR_SCALER,
            "metric": DONOR_METRIC,
            "note": (
                "separate LOFO MinMax donor space, "
                "decoupled from clustering scaler/metric"
            ),
        },
        "thresholds_percent": [
            threshold
            for threshold, _slug in THRESHOLDS
        ],
        "directional_majority_rule": (
            "independent functions remain cluster members "
            "but do not vote and cannot be donors"
        ),
        "inputs": {
            "profiles": {
                "path": str(
                    args.profiles.resolve()
                ),
                "sha256": sha256_file(
                    args.profiles.resolve()
                ),
            },
            "static_metrics": {
                "path": str(
                    args.static_metrics.resolve()
                ),
                "sha256": sha256_file(
                    args.static_metrics.resolve()
                ),
            },
            "language_map": {
                "path": str(
                    args.language_map.resolve()
                ),
                "sha256": sha256_file(
                    args.language_map.resolve()
                ),
            },
            "preferences": {
                slug: {
                    "path": str(path),
                    "sha256": (
                        sha256_file(path)
                    ),
                }
                for slug, path in pref_paths.items()
            },
        },
    }

    manifest_path = (
        args.output_dir
        / "final-algorithm-comparison-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    focus = primary[
        primary["threshold_percent"].isin(
            [2.5, 15.0]
        )
    ].sort_values(
        [
            "threshold_percent",
            "directional_preference_agreement_mean",
            "directional_mean_regret_percent_mean",
        ],
        ascending=[True, False, True],
    )

    display_cols = [
        "algorithm_config",
        "threshold_percent",
        "cluster_assignment_coverage_mean",
        "selection_coverage_mean",
        "directional_selection_coverage_mean",
        "directional_preference_agreement_mean",
        "directional_geomean_multiplicative_mismatch_mean",
        "directional_geomean_realized_speedup_mean",
        "directional_mean_regret_percent_mean",
        "directional_p90_regret_percent_mean",
        "mean_training_cluster_count_mean",
        "mean_training_noise_fraction_mean",
    ]

    print()
    print(
        "PRIMARY FINAL COMPARISON: "
        "KMEANS VS DBSCAN-Q80 @ 2.5 / 15"
    )
    print(
        focus[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(f"detail={detail_path}")
    print(
        f"per_replicate={per_replicate_path}"
    )
    print(f"summary={summary_path}")
    print(f"primary={primary_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Final operational K-Means vs DBSCAN "
            "comparison with decoupled MinMax+Manhattan donor ranking."
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
        "--seeds",
        default="11,23,37,41,53",
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
