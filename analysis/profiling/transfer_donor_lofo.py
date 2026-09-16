#!/usr/bin/env python3
"""
Leave-one-function-out donor-selection evaluation for Serverledge.

For each target function:
  * remove the target from the known-function catalogue;
  * refit the frozen PAPER-5 clustering configuration on the remaining functions;
  * assign the target using ONLY its x86 PAPER-5 profile;
  * choose the nearest known function inside the assigned cluster.

Frozen configurations:
  K-Means: PAPER-5 + MinMaxScaler + K=5 + Euclidean distance.
  DBSCAN:  PAPER-5 + RobustScaler + cosine + min_samples=4 +
           eps derived from the 0.80 quantile of training k-distances.

Ground truth and ARM observations are used ONLY after donor selection for
evaluation. They never participate in clustering or donor choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler

PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

DEFAULT_THRESHOLDS = [2.5, 5.0, 10.0, 15.0, 20.0, 25.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def threshold_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def load_profiles(path: Path) -> tuple[list[str], np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No profiles in {path}")

    fields = set(rows[0])
    if "function_name" not in fields:
        raise KeyError(f"function_name missing from {path}")
    for feature in PAPER5_FEATURES:
        if feature not in fields:
            raise KeyError(f"{feature} missing from {path}")

    rows.sort(key=lambda r: r["function_name"])
    names = [r["function_name"] for r in rows]
    x = np.asarray(
        [[float(r[f]) for f in PAPER5_FEATURES] for r in rows],
        dtype=float,
    )
    return names, x


def load_preferences(
    preferences_dir: Path,
    thresholds: list[float],
) -> dict[float, dict[str, dict[str, Any]]]:
    result: dict[float, dict[str, dict[str, Any]]] = {}
    for threshold in thresholds:
        path = preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
        rows = read_csv(path)
        if not rows:
            raise RuntimeError(f"No preference rows in {path}")

        per_function: dict[str, dict[str, Any]] = {}
        for row in rows:
            per_function[row["function_name"]] = {
                "label": row["architecture_preference"],
                "delta_percent": float(row["arm_vs_x86_delta_percent"]),
                "x86_duration_ms": float(row["x86_duration_ms"]),
                "arm_duration_ms": float(row["arm_duration_ms"]),
            }
        result[threshold] = per_function
    return result


def sample_is_eligible(sample: dict[str, Any]) -> bool:
    profile = sample.get("profile") or {}
    eligibility = sample.get("eligibility") or {}
    config = sample.get("function_configuration") or {}

    return (
        sample.get("warm_start") is True
        and sample.get("execution_succeeded") is True
        and profile.get("valid") is True
        and profile.get("exclusive_container") is True
        and eligibility.get("resource_clustering") is True
        and eligibility.get("performance_analysis") is True
        and abs(float(config.get("configured_cpus", 0.0)) - 1.0) <= 1e-9
    )


def load_raw_durations(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not sample_is_eligible(sample):
                continue

            function_name = sample["function_name"]
            duration_ms = float((sample.get("timing") or {})["duration_ms"])
            if not math.isfinite(duration_ms) or duration_ms <= 0:
                raise ValueError(
                    f"Invalid duration at {path}:{line_no}: {duration_ms}"
                )
            out[function_name].append(duration_ms)

    return dict(out)


def build_performance_stats(
    x86: dict[str, list[float]],
    arm: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    names = sorted(set(x86) & set(arm))
    result: dict[str, dict[str, float]] = {}

    for name in names:
        x = np.asarray(x86[name], dtype=float)
        a = np.asarray(arm[name], dtype=float)

        x_reward = -np.log(x)
        a_reward = -np.log(a)

        x_mean_reward = float(np.mean(x_reward))
        a_mean_reward = float(np.mean(a_reward))

        result[name] = {
            "x86_sample_count": len(x),
            "arm_sample_count": len(a),
            "x86_mean_ms": float(np.mean(x)),
            "arm_mean_ms": float(np.mean(a)),
            "x86_median_ms": float(np.median(x)),
            "arm_median_ms": float(np.median(a)),
            "x86_mean_reward": x_mean_reward,
            "arm_mean_reward": a_mean_reward,
            "reward_gap_x86_minus_arm": x_mean_reward - a_mean_reward,
            "best_reward_arm": (
                "x86" if x_mean_reward >= a_mean_reward else "arm64"
            ),
            "best_mean_latency_arm": (
                "x86" if float(np.mean(x)) <= float(np.mean(a)) else "arm64"
            ),
        }

    return result


def kmeans_select_donor(
    names: list[str],
    x: np.ndarray,
    target_index: int,
    k: int,
    n_init: int,
    random_state: int,
) -> dict[str, Any]:
    mask = np.arange(len(names)) != target_index
    train_names = np.asarray(names, dtype=object)[mask]
    x_train_raw = x[mask]
    x_target_raw = x[target_index : target_index + 1]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_target = scaler.transform(x_target_raw)

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(x_train)
    target_cluster = int(model.predict(x_target)[0])

    centroid_distance = float(
        np.linalg.norm(x_target[0] - model.cluster_centers_[target_cluster])
    )

    candidate_indices = np.flatnonzero(labels == target_cluster)
    candidate_matrix = x_train[candidate_indices]
    distances = np.linalg.norm(candidate_matrix - x_target[0], axis=1)

    order = np.lexsort(
        (
            train_names[candidate_indices],
            distances,
        )
    )
    selected_local = candidate_indices[int(order[0])]
    donor = str(train_names[selected_local])
    donor_distance = float(
        np.linalg.norm(x_train[selected_local] - x_target[0])
    )

    return {
        "selection_status": "selected",
        "selection_reason": "",
        "target_cluster": target_cluster,
        "target_assignment_distance": centroid_distance,
        "cluster_size": int(np.sum(labels == target_cluster)),
        "donor_function": donor,
        "donor_distance": donor_distance,
        "eps": "",
        "core_point_count": "",
    }


def dbscan_eps(
    x_train: np.ndarray,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> float:
    nn = NearestNeighbors(
        n_neighbors=min_samples,
        metric=metric,
    )
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_train)
    k_distances = distances[:, -1]
    return float(np.quantile(k_distances, eps_quantile))


def dbscan_select_donor(
    names: list[str],
    x: np.ndarray,
    target_index: int,
    min_samples: int,
    eps_quantile: float,
    metric: str,
) -> dict[str, Any]:
    mask = np.arange(len(names)) != target_index
    train_names = np.asarray(names, dtype=object)[mask]
    x_train_raw = x[mask]
    x_target_raw = x[target_index : target_index + 1]

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_target = scaler.transform(x_target_raw)

    eps = dbscan_eps(
        x_train,
        min_samples=min_samples,
        eps_quantile=eps_quantile,
        metric=metric,
    )
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )
    labels = model.fit_predict(x_train)

    core_indices = np.asarray(model.core_sample_indices_, dtype=int)
    if len(core_indices) == 0:
        return {
            "selection_status": "no-transfer",
            "selection_reason": "no_core_points",
            "target_cluster": -1,
            "target_assignment_distance": "",
            "cluster_size": 0,
            "donor_function": "",
            "donor_distance": "",
            "eps": eps,
            "core_point_count": 0,
        }

    core_matrix = x_train[core_indices]
    core_distances = pairwise_distances(
        x_target,
        core_matrix,
        metric=metric,
    )[0]
    within = np.flatnonzero(core_distances <= eps + 1e-12)

    if len(within) == 0:
        return {
            "selection_status": "no-transfer",
            "selection_reason": "no_core_within_eps",
            "target_cluster": -1,
            "target_assignment_distance": float(np.min(core_distances)),
            "cluster_size": 0,
            "donor_function": "",
            "donor_distance": "",
            "eps": eps,
            "core_point_count": len(core_indices),
        }

    candidate_core_global = core_indices[within]
    candidate_core_distances = core_distances[within]
    candidate_core_names = train_names[candidate_core_global]
    order = np.lexsort(
        (
            candidate_core_names,
            candidate_core_distances,
        )
    )
    nearest_core_global = int(candidate_core_global[int(order[0])])
    target_cluster = int(labels[nearest_core_global])
    assignment_distance = float(
        pairwise_distances(
            x_target,
            x_train[nearest_core_global : nearest_core_global + 1],
            metric=metric,
        )[0, 0]
    )

    candidate_indices = np.flatnonzero(labels == target_cluster)
    candidate_matrix = x_train[candidate_indices]
    donor_distances = pairwise_distances(
        x_target,
        candidate_matrix,
        metric=metric,
    )[0]
    candidate_names = train_names[candidate_indices]
    order = np.lexsort((candidate_names, donor_distances))
    selected_global = int(candidate_indices[int(order[0])])

    return {
        "selection_status": "selected",
        "selection_reason": "",
        "target_cluster": target_cluster,
        "target_assignment_distance": assignment_distance,
        "cluster_size": int(np.sum(labels == target_cluster)),
        "donor_function": str(train_names[selected_global]),
        "donor_distance": float(donor_distances[int(order[0])]),
        "eps": eps,
        "core_point_count": len(core_indices),
    }


def enrich_selection(
    algorithm: str,
    target: str,
    selection: dict[str, Any],
    preferences: dict[float, dict[str, dict[str, Any]]],
    primary_threshold: float,
    performance: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "algorithm": algorithm,
        "target_function": target,
        **selection,
    }

    primary_target = preferences[primary_threshold][target]
    row["primary_threshold_percent"] = primary_threshold
    row["target_ground_truth_label"] = primary_target["label"]
    row["target_architecture_delta_percent"] = primary_target["delta_percent"]

    for threshold in sorted(preferences):
        slug = threshold_slug(threshold)
        target_pref = preferences[threshold][target]
        row[f"target_label_t{slug}"] = target_pref["label"]

    target_perf = performance[target]
    row.update(
        {
            "target_x86_mean_ms": target_perf["x86_mean_ms"],
            "target_arm_mean_ms": target_perf["arm_mean_ms"],
            "target_x86_median_ms": target_perf["x86_median_ms"],
            "target_arm_median_ms": target_perf["arm_median_ms"],
            "target_x86_mean_reward": target_perf["x86_mean_reward"],
            "target_arm_mean_reward": target_perf["arm_mean_reward"],
            "target_reward_gap_x86_minus_arm": target_perf[
                "reward_gap_x86_minus_arm"
            ],
            "target_best_reward_arm": target_perf["best_reward_arm"],
            "target_best_mean_latency_arm": target_perf[
                "best_mean_latency_arm"
            ],
        }
    )

    donor = selection.get("donor_function") or ""
    if not donor:
        row.update(
            {
                "donor_ground_truth_label": "",
                "label_agreement": "",
                "donor_architecture_delta_percent": "",
                "abs_architecture_delta_error_percent": "",
                "donor_x86_mean_ms": "",
                "donor_arm_mean_ms": "",
                "donor_x86_mean_reward": "",
                "donor_arm_mean_reward": "",
                "donor_reward_gap_x86_minus_arm": "",
                "abs_reward_gap_error": "",
                "donor_best_reward_arm": "",
                "best_reward_arm_agreement": "",
                "donor_best_mean_latency_arm": "",
                "best_mean_latency_arm_agreement": "",
            }
        )
        for threshold in sorted(preferences):
            slug = threshold_slug(threshold)
            row[f"donor_label_t{slug}"] = ""
            row[f"label_agreement_t{slug}"] = ""
        return row

    donor_primary = preferences[primary_threshold][donor]
    donor_perf = performance[donor]

    row.update(
        {
            "donor_ground_truth_label": donor_primary["label"],
            "label_agreement": (
                primary_target["label"] == donor_primary["label"]
            ),
            "donor_architecture_delta_percent": donor_primary["delta_percent"],
            "abs_architecture_delta_error_percent": abs(
                primary_target["delta_percent"] - donor_primary["delta_percent"]
            ),
            "donor_x86_mean_ms": donor_perf["x86_mean_ms"],
            "donor_arm_mean_ms": donor_perf["arm_mean_ms"],
            "donor_x86_mean_reward": donor_perf["x86_mean_reward"],
            "donor_arm_mean_reward": donor_perf["arm_mean_reward"],
            "donor_reward_gap_x86_minus_arm": donor_perf[
                "reward_gap_x86_minus_arm"
            ],
            "abs_reward_gap_error": abs(
                target_perf["reward_gap_x86_minus_arm"]
                - donor_perf["reward_gap_x86_minus_arm"]
            ),
            "donor_best_reward_arm": donor_perf["best_reward_arm"],
            "best_reward_arm_agreement": (
                target_perf["best_reward_arm"] == donor_perf["best_reward_arm"]
            ),
            "donor_best_mean_latency_arm": donor_perf[
                "best_mean_latency_arm"
            ],
            "best_mean_latency_arm_agreement": (
                target_perf["best_mean_latency_arm"]
                == donor_perf["best_mean_latency_arm"]
            ),
        }
    )

    for threshold in sorted(preferences):
        slug = threshold_slug(threshold)
        target_pref = preferences[threshold][target]
        donor_pref = preferences[threshold][donor]
        row[f"donor_label_t{slug}"] = donor_pref["label"]
        row[f"label_agreement_t{slug}"] = (
            target_pref["label"] == donor_pref["label"]
        )

    return row


def bool_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [
        1.0 if r[field] is True else 0.0
        for r in rows
        if r.get(field) not in ("", None)
    ]
    return float(np.mean(values)) if values else float("nan")


def numeric_summary(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[float, float]:
    values = [
        float(r[field])
        for r in rows
        if r.get(field) not in ("", None)
    ]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.median(values))


def summarize(
    rows: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for algorithm in sorted({r["algorithm"] for r in rows}):
        all_rows = [r for r in rows if r["algorithm"] == algorithm]
        selected = [
            r for r in all_rows if r["selection_status"] == "selected"
        ]

        mean_delta, median_delta = numeric_summary(
            selected,
            "abs_architecture_delta_error_percent",
        )
        mean_gap, median_gap = numeric_summary(
            selected,
            "abs_reward_gap_error",
        )
        mean_distance, median_distance = numeric_summary(
            selected,
            "donor_distance",
        )

        row: dict[str, Any] = {
            "algorithm": algorithm,
            "target_count": len(all_rows),
            "selected_count": len(selected),
            "no_transfer_count": len(all_rows) - len(selected),
            "selection_coverage": (
                len(selected) / len(all_rows) if all_rows else float("nan")
            ),
            "primary_label_agreement_rate": bool_mean(
                selected,
                "label_agreement",
            ),
            "best_reward_arm_agreement_rate": bool_mean(
                selected,
                "best_reward_arm_agreement",
            ),
            "best_mean_latency_arm_agreement_rate": bool_mean(
                selected,
                "best_mean_latency_arm_agreement",
            ),
            "mean_abs_architecture_delta_error_percent": mean_delta,
            "median_abs_architecture_delta_error_percent": median_delta,
            "mean_abs_reward_gap_error": mean_gap,
            "median_abs_reward_gap_error": median_gap,
            "mean_donor_distance": mean_distance,
            "median_donor_distance": median_distance,
        }

        for threshold in thresholds:
            slug = threshold_slug(threshold)
            row[f"label_agreement_rate_t{slug}"] = bool_mean(
                selected,
                f"label_agreement_t{slug}",
            )

        output.append(row)

    return output


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--raw-x86", required=True, type=Path)
    p.add_argument("--raw-arm64", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--primary-threshold", type=float, default=15.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dbscan-min-samples", type=int, default=4)
    p.add_argument("--dbscan-eps-quantile", type=float, default=0.80)
    p.add_argument("--dbscan-metric", default="cosine")
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    names, x = load_profiles(args.profiles.resolve())
    preferences = load_preferences(
        args.preferences_dir.resolve(),
        DEFAULT_THRESHOLDS,
    )
    if args.primary_threshold not in preferences:
        raise ValueError(
            f"Primary threshold {args.primary_threshold} was not loaded"
        )

    raw_x86 = load_raw_durations(args.raw_x86.resolve())
    raw_arm = load_raw_durations(args.raw_arm64.resolve())
    performance = build_performance_stats(raw_x86, raw_arm)

    expected = set(names)
    if set(performance) != expected:
        raise RuntimeError(
            "Raw-performance/profile function mismatch: "
            f"missing={sorted(expected - set(performance))}, "
            f"extra={sorted(set(performance) - expected)}"
        )

    for threshold, table in preferences.items():
        if set(table) != expected:
            raise RuntimeError(
                f"Preference/profile mismatch at threshold {threshold}: "
                f"missing={sorted(expected - set(table))}, "
                f"extra={sorted(set(table) - expected)}"
            )

    rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(names):
        km = kmeans_select_donor(
            names,
            x,
            target_index,
            k=args.k,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        rows.append(
            enrich_selection(
                "kmeans",
                target,
                km,
                preferences,
                args.primary_threshold,
                performance,
            )
        )

        db = dbscan_select_donor(
            names,
            x,
            target_index,
            min_samples=args.dbscan_min_samples,
            eps_quantile=args.dbscan_eps_quantile,
            metric=args.dbscan_metric,
        )
        rows.append(
            enrich_selection(
                "dbscan",
                target,
                db,
                preferences,
                args.primary_threshold,
                performance,
            )
        )

    rows.sort(key=lambda r: (r["algorithm"], r["target_function"]))
    summary = summarize(rows, DEFAULT_THRESHOLDS)

    write_csv(out / "donor-selection-lofo.csv", rows)
    write_csv(out / "donor-selection-summary.csv", summary)

    manifest = {
        "schema_version": 1,
        "method": "leave-one-function-out donor selection",
        "feature_set": "PAPER-5",
        "features": PAPER5_FEATURES,
        "primary_ground_truth_threshold_percent": args.primary_threshold,
        "ground_truth_sensitivity_thresholds_percent": DEFAULT_THRESHOLDS,
        "kmeans": {
            "scaler": "minmax",
            "k": args.k,
            "metric": "euclidean",
            "n_init": args.n_init,
            "random_state": args.random_state,
            "selection_rule": (
                "nearest donor by Euclidean distance inside the target's "
                "predicted K-Means cluster"
            ),
        },
        "dbscan": {
            "scaler": "robust",
            "metric": args.dbscan_metric,
            "min_samples": args.dbscan_min_samples,
            "eps_quantile": args.dbscan_eps_quantile,
            "eps_policy": (
                "recomputed inside each LOFO fold from the training-only "
                "k-distance distribution"
            ),
            "target_assignment": (
                "cluster of nearest core point within eps; otherwise no-transfer"
            ),
            "selection_rule": (
                "nearest donor by cosine distance inside the assigned cluster"
            ),
        },
        "leakage_control": (
            "target function excluded before scaler fit, clustering fit, eps "
            "calibration and donor selection; ARM/ground truth used post-hoc only"
        ),
        "reward": "-ln(duration_ms)",
        "inputs": {
            "profiles": str(args.profiles.resolve()),
            "preferences_dir": str(args.preferences_dir.resolve()),
            "raw_x86": str(args.raw_x86.resolve()),
            "raw_arm64": str(args.raw_arm64.resolve()),
        },
        "outputs": [
            "donor-selection-lofo.csv",
            "donor-selection-summary.csv",
        ],
    }
    (out / "donor-selection-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"functions={len(names)} "
        f"rows={len(rows)} "
        f"output={out}"
    )
    for row in summary:
        print(
            f"{row['algorithm']}: "
            f"coverage={row['selection_coverage']:.4f} "
            f"best_reward_arm_agreement="
            f"{row['best_reward_arm_agreement_rate']:.4f} "
            f"label_agreement_t15="
            f"{row['primary_label_agreement_rate']:.4f} "
            f"median_delta_error="
            f"{row['median_abs_architecture_delta_error_percent']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
