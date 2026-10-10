#!/usr/bin/env python3
"""
Operational donor selector for the final UCB1 GCP validation.

Frozen pipeline
---------------
Features:
    dynamic:
        page_faults_delta
        utilized_cpus
        free_memory_mb
        cpu_user_delta_ms
        cpu_kernel_delta_ms

    static:
        static_token_count_mean
        static_function_count

Preprocessing:
    - target excluded from training/catalog
    - MinMaxScaler fitted ONLY on donor/reference catalog
    - KMeans geometry:
          dynamic coordinates / sqrt(5)
          static coordinates  / sqrt(2)
    - KMeans:
          K=6
          n_init=50
          random_state=11

Donor selection:
    - target assigned to KMeans cluster
    - donor distance uses PLAIN Manhattan on the 7 MinMax coordinates
      (NO /sqrt(p) weighting here)
    - directional vote:
          sum 1 / (d + 1e-9)^2
    - only x86-preferred / arm-preferred catalog members vote
    - donor = nearest Manhattan candidate in predicted directional class

Important:
    - target ARM ground truth is NEVER read.
    - target is removed from scaler fit, KMeans fit and donor catalog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


DYNAMIC_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

STATIC_FEATURES = [
    "static_token_count_mean",
    "static_function_count",
]

FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES

DIRECTIONS = (
    "x86-preferred",
    "arm-preferred",
)

K = 6
N_INIT = 50
RANDOM_STATE = 11
EPSILON = 1e-9


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"file non trovato: {path}")

    return pd.read_csv(path)


def require_columns(
        df: pd.DataFrame,
        path: Path,
        columns: list[str],
) -> None:
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        fail(
            f"{path}: colonne mancanti: "
            + ", ".join(missing)
        )


def finite_numeric(
        df: pd.DataFrame,
        columns: list[str],
) -> pd.DataFrame:
    result = df.copy()

    for name in columns:
        result[name] = pd.to_numeric(
            result[name],
            errors="coerce",
        )

    mask = np.ones(len(result), dtype=bool)

    for name in columns:
        mask &= np.isfinite(
            result[name].to_numpy(dtype=float)
        )

    return result.loc[mask].copy()


def source_available_mask(
        series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def exactly_one_target(
        df: pd.DataFrame,
        target: str,
        source: str,
) -> pd.DataFrame:
    rows = df[
        df["function_name"].astype(str) == target
        ].copy()

    if len(rows) != 1:
        fail(
            f"{source}: target={target!r}, "
            f"attesa esattamente 1 riga, trovate {len(rows)}"
        )

    return rows.reset_index(drop=True)


def load_catalog(
        historical_profiles_path: Path,
        static_metrics_path: Path,
        preferences_path: Path,
        target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build donor/reference catalog.

    The target is removed BEFORE scaler fit, KMeans fit and donor selection.
    Target preference is never used.
    """

    profiles = read_csv(
        historical_profiles_path
    )

    static = read_csv(
        static_metrics_path
    )

    preferences = read_csv(
        preferences_path
    )

    require_columns(
        profiles,
        historical_profiles_path,
        ["function_name", *DYNAMIC_FEATURES],
    )

    require_columns(
        static,
        static_metrics_path,
        [
            "function_name",
            "static_source_available",
            *STATIC_FEATURES,
        ],
    )

    require_columns(
        preferences,
        preferences_path,
        [
            "function_name",
            "architecture_preference",
        ],
    )

    for label, df in (
            ("historical profiles", profiles),
            ("static metrics", static),
            ("preferences", preferences),
    ):
        if df["function_name"].duplicated().any():
            duplicates = (
                df.loc[
                    df["function_name"].duplicated(),
                    "function_name",
                ]
                .astype(str)
                .tolist()
            )

            fail(
                f"{label}: function_name duplicati: "
                + ", ".join(duplicates)
            )

    # Static row for target:
    # architecture-independent information, allowed as input feature.
    target_static = exactly_one_target(
        static,
        target,
        str(static_metrics_path),
    )

    if not bool(
            source_available_mask(
                target_static["static_source_available"]
            ).iloc[0]
    ):
        fail(
            f"target {target!r}: metriche statiche "
            "non disponibili"
        )

    target_static = finite_numeric(
        target_static,
        STATIC_FEATURES,
    )

    if len(target_static) != 1:
        fail(
            f"target {target!r}: metriche statiche "
            "non finite"
        )

    # Remove target BEFORE any fit.
    profiles = profiles[
        profiles["function_name"].astype(str)
        != target
        ].copy()

    static = static[
        static["function_name"].astype(str)
        != target
        ].copy()

    preferences = preferences[
        preferences["function_name"].astype(str)
        != target
        ].copy()

    catalog = (
        profiles[
            ["function_name", *DYNAMIC_FEATURES]
        ]
        .merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    *STATIC_FEATURES,
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            preferences[
                [
                    "function_name",
                    "architecture_preference",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )

    catalog = catalog[
        source_available_mask(
            catalog["static_source_available"]
        )
    ].copy()

    catalog = finite_numeric(
        catalog,
        FEATURES,
    )

    catalog = (
        catalog.sort_values("function_name")
        .reset_index(drop=True)
    )

    if len(catalog) < K:
        fail(
            f"catalogo troppo piccolo per K={K}: "
            f"{len(catalog)} righe"
        )

    return catalog, target_static


def load_target_dynamic(
        target_profiles_path: Path,
        target: str,
) -> pd.DataFrame:
    target_profiles = read_csv(
        target_profiles_path
    )

    require_columns(
        target_profiles,
        target_profiles_path,
        ["function_name", *DYNAMIC_FEATURES],
    )

    target_row = exactly_one_target(
        target_profiles,
        target,
        str(target_profiles_path),
    )

    target_row = finite_numeric(
        target_row,
        DYNAMIC_FEATURES,
    )

    if len(target_row) != 1:
        fail(
            f"target {target!r}: feature dinamiche "
            "non finite"
        )

    return target_row


def make_target_row(
        target_dynamic: pd.DataFrame,
        target_static: pd.DataFrame,
        target: str,
) -> pd.DataFrame:
    row = {
        "function_name": target,
    }

    for name in DYNAMIC_FEATURES:
        row[name] = float(
            target_dynamic.iloc[0][name]
        )

    for name in STATIC_FEATURES:
        row[name] = float(
            target_static.iloc[0][name]
        )

    return pd.DataFrame([row])


def cluster_transform(
        plain: np.ndarray,
) -> np.ndarray:
    """
    Block weighting ONLY for KMeans.

    Because KMeans uses squared Euclidean distance:

        sum((delta_i / sqrt(p))^2)
        = (1/p) * sum(delta_i^2)

    This balances average squared contribution of each block.
    """

    scales = np.array(
        [math.sqrt(len(DYNAMIC_FEATURES))]
        * len(DYNAMIC_FEATURES)
        + [math.sqrt(len(STATIC_FEATURES))]
        * len(STATIC_FEATURES),
        dtype=float,
        )

    return plain / scales


def choose_donor(
        catalog: pd.DataFrame,
        x_plain: np.ndarray,
        target_plain: np.ndarray,
        cluster_labels: np.ndarray,
        target_cluster: int,
) -> tuple[
    str,
    str,
    float,
    dict[str, float],
    list[dict],
]:
    same_cluster = np.flatnonzero(
        cluster_labels == target_cluster
    )

    if len(same_cluster) == 0:
        fail(
            f"cluster target {target_cluster}: "
            "nessun membro"
        )

    prefs = (
        catalog["architecture_preference"]
        .astype(str)
        .to_numpy()
    )

    names = (
        catalog["function_name"]
        .astype(str)
        .to_numpy()
    )

    distances = np.abs(
        x_plain - target_plain[0]
    ).sum(axis=1)

    excluded_donors = np.isin(
        names[same_cluster],
        ["amd_faster", "arm_faster"],
    )

    directional = same_cluster[
        np.isin(
            prefs[same_cluster],
            DIRECTIONS,
        )
        & ~excluded_donors
    ]

    if len(directional) == 0:
        fail(
            "cluster target senza donor direzionali"
        )

    votes: dict[str, float] = {}

    for direction in DIRECTIONS:
        members = directional[
            prefs[directional] == direction
            ]

        if len(members) == 0:
            votes[direction] = 0.0
        else:
            votes[direction] = float(
                np.sum(
                    1.0
                    / (
                            distances[members]
                            + EPSILON
                    )
                    ** 2
                )
            )

    if math.isclose(
            votes[DIRECTIONS[0]],
            votes[DIRECTIONS[1]],
            rel_tol=0.0,
            abs_tol=1e-15,
    ):
        fail(
            "tie nel voto direzionale: "
            f"{votes}"
        )

    predicted = max(
        DIRECTIONS,
        key=lambda d: votes[d],
    )

    pool = directional[
        prefs[directional] == predicted
        ]

    if len(pool) == 0:
        fail(
            f"nessun donor nella classe "
            f"predetta {predicted}"
        )

    # Deterministic tie break:
    # first distance, then function name.
    order = np.lexsort(
        (
            names[pool],
            distances[pool],
        )
    )

    selected_index = int(
        pool[order[0]]
    )

    ranking: list[dict] = []

    ranked_cluster = sorted(
        directional.tolist(),
        key=lambda idx: (
            float(distances[idx]),
            names[idx],
        ),
    )

    for rank, idx in enumerate(
            ranked_cluster,
            start=1,
    ):
        ranking.append(
            {
                "rank": rank,
                "function_name": names[idx],
                "architecture_preference": prefs[idx],
                "distance": float(
                    distances[idx]
                ),
                "vote_weight": float(
                    1.0
                    / (
                            distances[idx]
                            + EPSILON
                    )
                    ** 2
                ),
                "in_predicted_class":
                    prefs[idx] == predicted,
                "selected":
                    idx == selected_index,
            }
        )

    return (
        names[selected_index],
        predicted,
        float(distances[selected_index]),
        votes,
        ranking,
    )


def write_json(
        path: Path,
        document: dict,
) -> None:
    path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--historical-profiles",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target-profiles",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--static-metrics",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--preferences",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target",
        required=True,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-donor",
        default="",
        help=(
            "Optional local reproducibility gate. "
            "Do NOT use on the live GCP target."
        ),
    )

    args = parser.parse_args()

    target = args.target.strip()

    if not target:
        fail("--target vuoto")

    if not args.run_id.strip():
        fail("--run-id vuoto")

    catalog, target_static = load_catalog(
        args.historical_profiles,
        args.static_metrics,
        args.preferences,
        target,
    )

    target_dynamic = load_target_dynamic(
        args.target_profiles,
        target,
    )

    target_row = make_target_row(
        target_dynamic,
        target_static,
        target,
    )

    scaler = MinMaxScaler()

    x_plain = scaler.fit_transform(
        catalog[FEATURES].to_numpy(
            dtype=float
        )
    )

    target_plain = scaler.transform(
        target_row[FEATURES].to_numpy(
            dtype=float
        )
    )

    x_cluster = cluster_transform(
        x_plain
    )

    target_cluster_vector = (
        cluster_transform(
            target_plain
        )
    )

    kmeans = KMeans(
        n_clusters=K,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = kmeans.fit_predict(
        x_cluster
    )

    target_cluster = int(
        kmeans.predict(
            target_cluster_vector
        )[0]
    )

    (
        selected_donor,
        prediction,
        selected_distance,
        votes,
        ranking,
    ) = choose_donor(
        catalog,
        x_plain,
        target_plain,
        labels,
        target_cluster,
    )

    if (
            args.expected_donor
            and selected_donor
            != args.expected_donor
    ):
        fail(
            "donor mismatch: "
            f"ottenuto={selected_donor}, "
            f"atteso={args.expected_donor}"
        )

    output = args.output_dir.resolve()
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
            output / "selection.json"
    )

    ranking_path = (
            output / "selection.csv"
    )

    model_path = (
            output / "model.json"
    )

    model = {
        "schema_version": 1,
        "features": FEATURES,
        "dynamic_features":
            DYNAMIC_FEATURES,
        "static_features":
            STATIC_FEATURES,
        "scaler": {
            "type": "MinMaxScaler",
            "data_min":
                scaler.data_min_.tolist(),
            "data_max":
                scaler.data_max_.tolist(),
            "scale":
                scaler.scale_.tolist(),
            "min":
                scaler.min_.tolist(),
            "fit_rows": len(catalog),
            "target_excluded": True,
        },
        "clustering": {
            "algorithm": "KMeans",
            "k": K,
            "n_init": N_INIT,
            "random_state":
                RANDOM_STATE,
            "geometry":
                "minmax_block_sqrt",
            "dynamic_divisor":
                math.sqrt(
                    len(
                        DYNAMIC_FEATURES
                    )
                ),
            "static_divisor":
                math.sqrt(
                    len(
                        STATIC_FEATURES
                    )
                ),
            "cluster_centers":
                kmeans.cluster_centers_.tolist(),
        },
    }

    write_json(
        model_path,
        model,
    )

    with ranking_path.open(
            "w",
            newline="",
            encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "function_name",
                "architecture_preference",
                "distance",
                "vote_weight",
                "in_predicted_class",
                "selected",
            ],
        )

        writer.writeheader()
        writer.writerows(ranking)

    selection = {
        "schema_version": 1,
        "status": "selected",
        "reason": "",
        "selection_run_id":
            args.run_id,
        "target_function": target,
        "target_cluster":
            target_cluster,
        "catalog_rows":
            len(catalog),
        "candidate_count":
            len(ranking),
        "prediction":
            prediction,
        "selected_donor": {
            "function_name":
                selected_donor,
            "distance":
                selected_distance,
        },
        "directional_vote": {
            "method":
                "inverse_distance_squared",
            "epsilon":
                EPSILON,
            "x86-preferred":
                votes[
                    "x86-preferred"
                ],
            "arm-preferred":
                votes[
                    "arm-preferred"
                ],
        },
        "pipeline": {
            "features":
                FEATURES,
            "reference_architecture":
                "amd64",
            "preprocessing":
                "train-only MinMax",
            "clustering": {
                "algorithm":
                    "KMeans",
                "k":
                    K,
                "n_init":
                    N_INIT,
                "random_state":
                    RANDOM_STATE,
                "geometry":
                    "block_sqrt",
                "dynamic_weight":
                    "1/sqrt(5)",
                "static_weight":
                    "1/sqrt(2)",
            },
            "donor_selection": {
                "require_same_cluster":
                    True,
                "distance":
                    "manhattan_plain_minmax",
                "directional_vote":
                    "1/(d+1e-9)^2",
                "excluded_functions": [
                    "amd_faster",
                    "arm_faster",
                ],
                "nearest_within_predicted_class":
                    True,
            },
            "target_ground_truth_used":
                False,
            "target_excluded_from_fit":
                True,
        },
        "target": {
            "raw_features": {
                feature: float(
                    target_row.iloc[0][
                        feature
                    ]
                )
                for feature in FEATURES
            },
            "minmax_features":
                target_plain[
                    0
                ].tolist(),
            "cluster_features":
                target_cluster_vector[
                    0
                ].tolist(),
        },
        "sources": {
            "historical_profiles": {
                "path": str(
                    args.historical_profiles.resolve()
                ),
                "sha256": sha256(
                    args.historical_profiles
                ),
            },
            "target_profiles": {
                "path": str(
                    args.target_profiles.resolve()
                ),
                "sha256": sha256(
                    args.target_profiles
                ),
            },
            "static_metrics": {
                "path": str(
                    args.static_metrics.resolve()
                ),
                "sha256": sha256(
                    args.static_metrics
                ),
            },
            "preferences": {
                "path": str(
                    args.preferences.resolve()
                ),
                "sha256": sha256(
                    args.preferences
                ),
            },
            "model": {
                "path":
                    str(model_path),
                "sha256":
                    sha256(model_path),
            },
            "ranking": {
                "path":
                    str(ranking_path),
                "sha256":
                    sha256(ranking_path),
            },
        },
    }

    write_json(
        selection_path,
        selection,
    )

    print(
        json.dumps(
            {
                "status":
                    "selected",
                "target":
                    target,
                "target_cluster":
                    target_cluster,
                "prediction":
                    prediction,
                "selected_donor":
                    selected_donor,
                "selected_distance":
                    selected_distance,
                "candidate_count":
                    len(ranking),
                "selection_json":
                    str(selection_path),
                "model_json":
                    str(model_path),
                "ranking_csv":
                    str(ranking_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()