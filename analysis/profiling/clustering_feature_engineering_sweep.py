#!/usr/bin/env python3
"""Exploratory, leakage-safe LOFO K-Means sweep for improving the
Serverledge clustering-as-classification stage.

Ground-truth labels are never used in fitting a fold. They are used only after
prediction to score each configuration. Because many configurations are
compared on the same 53 targets, this script is explicitly exploratory. A
winning configuration must later be revalidated (preferably nested LOFO)
before it is treated as final evidence.

The sweep tests transformations motivated by the observed heavy-tailed dynamic
features and a compact derived behavioural representation:

  paper5_raw:
      original PAPER-5 values

  paper5_log:
      log1p(page_faults_delta, cpu_user_delta_ms, cpu_kernel_delta_ms),
      plus utilized_cpus and free_memory_mb

  behavioral4:
      log1p(page_faults_delta)
      utilized_cpus
      log1p(cpu_user_delta_ms + cpu_kernel_delta_ms)
      kernel_share = kernel / (user + kernel)

Each representation can optionally include one-hot language.  Language has an
explicit weight so its geometry can be tested without forcing a full unit of
Euclidean distance.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

LABELS = ["x86-preferred", "architecture-independent", "arm-preferred"]
DIRECTIONAL = {"x86-preferred", "arm-preferred"}
PAPER5 = ["page_faults_delta", "utilized_cpus", "free_memory_mb", "cpu_user_delta_ms", "cpu_kernel_delta_ms"]
SCALERS = {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler}


def threshold_slug(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")


def load_data(profiles: Path, preferences_dir: Path, threshold: float, language_map: Path | None) -> pd.DataFrame:
    p = pd.read_csv(profiles)
    missing = sorted({"function_name", *PAPER5} - set(p.columns))
    if missing:
        raise ValueError(f"profiles missing columns {missing}")
    pref_path = preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
    pref = pd.read_csv(pref_path)
    df = p.merge(pref[["function_name", "architecture_preference"]], on="function_name", validate="one_to_one")
    if language_map is not None:
        lang = pd.read_csv(language_map)[["function_name", "language"]]
        if lang["language"].isna().any():
            raise ValueError("language map contains missing values")
        df = df.merge(lang, on="function_name", validate="one_to_one")
    return df.sort_values("function_name").reset_index(drop=True)


def numeric_representation(frame: pd.DataFrame, variant: str) -> np.ndarray:
    pf = frame["page_faults_delta"].astype(float).to_numpy()
    cpu = frame["utilized_cpus"].astype(float).to_numpy()
    free = frame["free_memory_mb"].astype(float).to_numpy()
    user = frame["cpu_user_delta_ms"].astype(float).to_numpy()
    kernel = frame["cpu_kernel_delta_ms"].astype(float).to_numpy()
    total = user + kernel

    if variant == "paper5_raw":
        return np.column_stack([pf, cpu, free, user, kernel])
    if variant == "paper5_no_free_raw":
        return np.column_stack([pf, cpu, user, kernel])
    if variant == "paper5_log":
        return np.column_stack([np.log1p(np.clip(pf, 0, None)), cpu, free, np.log1p(np.clip(user, 0, None)), np.log1p(np.clip(kernel, 0, None))])
    if variant == "paper5_no_free_log":
        return np.column_stack([np.log1p(np.clip(pf, 0, None)), cpu, np.log1p(np.clip(user, 0, None)), np.log1p(np.clip(kernel, 0, None))])
    if variant == "behavioral4":
        kernel_share = kernel / np.maximum(total, 1e-12)
        return np.column_stack([
            np.log1p(np.clip(pf, 0, None)),
            cpu,
            np.log1p(np.clip(total, 0, None)),
            kernel_share,
        ])
    raise ValueError(f"unknown feature variant {variant}")


def add_language(train_x: np.ndarray, target_x: np.ndarray, train_lang: pd.Series, target_lang: str, weight: float) -> tuple[np.ndarray, np.ndarray]:
    vocab = sorted(train_lang.astype(str).unique())
    idx = {v: i for i, v in enumerate(vocab)}
    a = np.zeros((len(train_lang), len(vocab)), dtype=float)
    b = np.zeros((1, len(vocab)), dtype=float)
    for row, value in enumerate(train_lang.astype(str)):
        a[row, idx[value]] = weight
    if str(target_lang) in idx:
        b[0, idx[str(target_lang)]] = weight
    return np.hstack([train_x, a]), np.hstack([target_x, b])


def unique_majority(values: pd.Series) -> str:
    counts = Counter(values.tolist())
    maximum = max(counts.values())
    winners = [label for label in LABELS if counts.get(label, 0) == maximum]
    return winners[0] if len(winners) == 1 else "abstain"


def score_predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    recalls = []
    f1s = []
    for label in LABELS:
        truth = y == label
        pred = p == label
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    directional_mask = np.isin(y, list(DIRECTIONAL))
    return {
        "strict_accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "directional_accuracy": float(np.mean(y[directional_mask] == p[directional_mask])),
        "coverage": float(np.mean(p != "abstain")),
        "pred_x86": int(np.sum(p == "x86-preferred")),
        "pred_independent": int(np.sum(p == "architecture-independent")),
        "pred_arm": int(np.sum(p == "arm-preferred")),
        "pred_abstain": int(np.sum(p == "abstain")),
    }


def purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    good = clusters >= 0
    if not np.any(good):
        return 0.0
    correct = 0
    for cluster in sorted(set(clusters[good])):
        vals = labels[clusters == cluster]
        correct += max(Counter(vals.tolist()).values())
    return float(correct / len(labels))


def evaluate_config(df: pd.DataFrame, *, variant: str, scaler_name: str, k: int, include_language: bool, language_weight: float, n_init: int, random_state: int) -> tuple[dict, list[dict]]:
    scaler_cls = SCALERS[scaler_name]
    y_true: list[str] = []
    y_pred: list[str] = []
    per_target: list[dict] = []

    for target_idx in range(len(df)):
        train = df.drop(index=target_idx)
        target = df.iloc[[target_idx]]
        x_train_raw = numeric_representation(train, variant)
        x_target_raw = numeric_representation(target, variant)
        scaler = scaler_cls()
        x_train = scaler.fit_transform(x_train_raw)
        x_target = scaler.transform(x_target_raw)
        if include_language:
            x_train, x_target = add_language(x_train, x_target, train["language"], str(target.iloc[0]["language"]), language_weight)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        cluster_train = km.fit_predict(x_train)
        target_cluster = int(km.predict(x_target)[0])
        members = np.flatnonzero(cluster_train == target_cluster)
        predicted = unique_majority(train.iloc[members]["architecture_preference"])
        truth = str(target.iloc[0]["architecture_preference"])
        y_true.append(truth)
        y_pred.append(predicted)
        per_target.append({
            "feature_variant": variant,
            "scaler": scaler_name,
            "k": k,
            "include_language": include_language,
            "language_weight": language_weight if include_language else 0.0,
            "function_name": str(target.iloc[0]["function_name"]),
            "true_label": truth,
            "predicted_label": predicted,
            "target_cluster": target_cluster,
            "cluster_size": len(members),
        })

    y = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    result = score_predictions(y, p)

    # Full-fit diagnostics are secondary, but useful for understanding geometry.
    x_raw = numeric_representation(df, variant)
    scaler = scaler_cls()
    x = scaler.fit_transform(x_raw)
    if include_language:
        vocab = sorted(df["language"].astype(str).unique())
        idx = {v: i for i, v in enumerate(vocab)}
        oh = np.zeros((len(df), len(vocab)), dtype=float)
        for row, value in enumerate(df["language"].astype(str)):
            oh[row, idx[value]] = language_weight
        x = np.hstack([x, oh])
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    clusters = km.fit_predict(x)
    result.update({
        "feature_variant": variant,
        "scaler": scaler_name,
        "k": k,
        "include_language": include_language,
        "language_weight": language_weight if include_language else 0.0,
        "n_init": n_init,
        "full_fit_purity": purity(df["architecture_preference"].to_numpy(), clusters),
        "silhouette": float(silhouette_score(x, clusters)) if len(set(clusters)) > 1 else float("nan"),
        "davies_bouldin": float(davies_bouldin_score(x, clusters)) if len(set(clusters)) > 1 else float("nan"),
        "calinski_harabasz": float(calinski_harabasz_score(x, clusters)) if len(set(clusters)) > 1 else float("nan"),
    })
    majority_share = float(df["architecture_preference"].value_counts(normalize=True).max())
    result["majority_baseline"] = majority_share
    result["strict_gain_over_majority_baseline"] = result["strict_accuracy"] - majority_share
    return result, per_target


def run(args: argparse.Namespace) -> None:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.profiles.resolve(), args.preferences_dir.resolve(), args.threshold, args.language_map.resolve() if args.language_map else None)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    ks = list(range(args.k_min, args.k_max + 1))
    language_weights = [float(v) for v in args.language_weights.split(",") if v.strip()]

    for scaler in scalers:
        if scaler not in SCALERS:
            raise ValueError(f"unsupported scaler {scaler}")
    if args.language_map is None:
        language_modes = [(False, 0.0)]
    else:
        language_modes = [(False, 0.0)] + [(True, w) for w in language_weights]

    summaries: list[dict] = []
    targets: list[dict] = []
    total = len(variants) * len(scalers) * len(ks) * len(language_modes)
    done = 0
    for variant in variants:
        for scaler in scalers:
            for k in ks:
                for include_language, language_weight in language_modes:
                    summary, per_target = evaluate_config(
                        df,
                        variant=variant,
                        scaler_name=scaler,
                        k=k,
                        include_language=include_language,
                        language_weight=language_weight,
                        n_init=args.n_init,
                        random_state=args.random_state,
                    )
                    summaries.append(summary)
                    targets.extend(per_target)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"progress={done}/{total}")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["balanced_accuracy", "macro_f1", "directional_accuracy", "strict_accuracy", "coverage"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank_exploratory", np.arange(1, len(summary_df) + 1))
    summary_df.to_csv(out / "kmeans-feature-engineering-sweep-summary.csv", index=False)
    pd.DataFrame(targets).to_csv(out / "kmeans-feature-engineering-sweep-per-target.csv", index=False)
    summary_df.head(args.top_n).to_csv(out / "kmeans-feature-engineering-top.csv", index=False)

    manifest = {
        "schema_version": 1,
        "purpose": "exploratory LOFO K-Means hyperparameter/feature-engineering sweep; winner requires nested revalidation",
        "threshold_percent": args.threshold,
        "target_count": len(df),
        "variants": variants,
        "scalers": scalers,
        "k_range": [args.k_min, args.k_max],
        "language_weights": language_weights if args.language_map else [],
        "n_init": args.n_init,
        "random_state": args.random_state,
        "ranking": ["balanced_accuracy", "macro_f1", "directional_accuracy", "strict_accuracy", "coverage"],
    }
    (out / "kmeans-feature-engineering-sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nTOP CONFIGURATIONS (EXPLORATORY):")
    cols = [
        "rank_exploratory", "feature_variant", "scaler", "k", "include_language", "language_weight",
        "coverage", "strict_accuracy", "strict_gain_over_majority_baseline", "balanced_accuracy", "macro_f1",
        "directional_accuracy", "pred_x86", "pred_independent", "pred_arm", "pred_abstain",
        "full_fit_purity", "silhouette", "davies_bouldin", "calinski_harabasz",
    ]
    print(summary_df[cols].head(args.top_n).to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--language-map", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--variants", default="paper5_raw,paper5_no_free_raw,paper5_log,paper5_no_free_log,behavioral4")
    p.add_argument("--scalers", default="minmax,robust,standard")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--language-weights", default="0.25,0.5,1.0")
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--top-n", type=int, default=20)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
