#!/usr/bin/env python3
"""Ablation study for the derived behavioral feature representation."""
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
    prof = pd.read_csv(profiles)
    pref = pd.read_csv(preferences_dir / f"preferences-{threshold_slug(threshold)}.csv")
    frame = prof.merge(pref[["function_name", "architecture_preference"]], on="function_name", validate="one_to_one")
    if language_map is not None:
        lang = pd.read_csv(language_map)[["function_name", "language"]]
        if lang["language"].isna().any():
            raise ValueError("language map contains missing values")
        frame = frame.merge(lang, on="function_name", validate="one_to_one")
    return frame.sort_values("function_name").reset_index(drop=True)


def representation(frame: pd.DataFrame, variant: str) -> np.ndarray:
    pf = frame["page_faults_delta"].astype(float).to_numpy()
    cpu = frame["utilized_cpus"].astype(float).to_numpy()
    free = frame["free_memory_mb"].astype(float).to_numpy()
    user = frame["cpu_user_delta_ms"].astype(float).to_numpy()
    kernel = frame["cpu_kernel_delta_ms"].astype(float).to_numpy()
    total = user + kernel

    log_pf = np.log1p(np.clip(pf, 0, None))
    log_user = np.log1p(np.clip(user, 0, None))
    log_kernel = np.log1p(np.clip(kernel, 0, None))
    log_total = np.log1p(np.clip(total, 0, None))
    kernel_share = kernel / np.maximum(total, 1e-12)

    variants = {
        "paper5_raw": np.column_stack([pf, cpu, free, user, kernel]),
        "behavioral4": np.column_stack([log_pf, cpu, log_total, kernel_share]),
        "behavioral4_no_page_faults": np.column_stack([cpu, log_total, kernel_share]),
        "behavioral4_no_utilized_cpus": np.column_stack([log_pf, log_total, kernel_share]),
        "behavioral4_no_total_cpu": np.column_stack([log_pf, cpu, kernel_share]),
        "behavioral4_no_kernel_share": np.column_stack([log_pf, cpu, log_total]),
        "behavioral4_raw_page_faults": np.column_stack([pf, cpu, log_total, kernel_share]),
        "behavioral4_separate_cpu": np.column_stack([log_pf, cpu, log_user, log_kernel]),
    }
    if variant not in variants:
        raise ValueError(f"unknown variant {variant}")
    return variants[variant]


def add_language(train_x, target_x, train_lang, target_lang, weight):
    vocab = sorted(train_lang.astype(str).unique())
    idx = {v: i for i, v in enumerate(vocab)}
    a = np.zeros((len(train_lang), len(vocab)), dtype=float)
    b = np.zeros((1, len(vocab)), dtype=float)
    for row, value in enumerate(train_lang.astype(str)):
        a[row, idx[value]] = weight
    if str(target_lang) in idx:
        b[0, idx[str(target_lang)]] = weight
    return np.hstack([train_x, a]), np.hstack([target_x, b])


def unique_majority(values: pd.Series):
    counts = Counter(values.tolist())
    ordered = sorted(counts.values(), reverse=True)
    first = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    winners = [label for label in LABELS if counts.get(label, 0) == first]
    pred = winners[0] if len(winners) == 1 else "abstain"
    margin = (first - second) / max(len(values), 1)
    return pred, first, second, margin


def score_predictions(y, p):
    recalls, f1s = [], []
    class_recalls = {}
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
        class_recalls[label] = recall
    directional = np.isin(y, list(DIRECTIONAL))
    return {
        "coverage": float(np.mean(p != "abstain")),
        "strict_accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "directional_accuracy": float(np.mean(y[directional] == p[directional])),
        "pred_x86": int(np.sum(p == "x86-preferred")),
        "pred_independent": int(np.sum(p == "architecture-independent")),
        "pred_arm": int(np.sum(p == "arm-preferred")),
        "pred_abstain": int(np.sum(p == "abstain")),
        "x86_recall": class_recalls["x86-preferred"],
        "independent_recall": class_recalls["architecture-independent"],
        "arm_recall": class_recalls["arm-preferred"],
    }


def purity(labels, clusters):
    correct = 0
    for cluster in sorted(set(clusters)):
        values = labels[clusters == cluster]
        correct += max(Counter(values.tolist()).values())
    return correct / len(labels)


def evaluate(df, *, variant, scaler_name, k, include_language, language_weight, n_init, random_state):
    scaler_cls = SCALERS[scaler_name]
    y_true, y_pred, targets = [], [], []

    for target_idx in range(len(df)):
        train = df.drop(index=target_idx)
        target = df.iloc[[target_idx]]
        raw_train = representation(train, variant)
        raw_target = representation(target, variant)
        scaler = scaler_cls()
        x_train = scaler.fit_transform(raw_train)
        x_target = scaler.transform(raw_target)
        if include_language:
            x_train, x_target = add_language(x_train, x_target, train["language"], str(target.iloc[0]["language"]), language_weight)

        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        train_clusters = km.fit_predict(x_train)
        target_cluster = int(km.predict(x_target)[0])
        members = np.flatnonzero(train_clusters == target_cluster)
        pred, first, second, margin = unique_majority(train.iloc[members]["architecture_preference"])
        truth = str(target.iloc[0]["architecture_preference"])
        y_true.append(truth)
        y_pred.append(pred)
        targets.append({
            "feature_variant": variant,
            "scaler": scaler_name,
            "k": k,
            "include_language": include_language,
            "language_weight": language_weight if include_language else 0.0,
            "function_name": str(target.iloc[0]["function_name"]),
            "true_label": truth,
            "predicted_label": pred,
            "target_cluster": target_cluster,
            "cluster_size": len(members),
            "majority_count": first,
            "second_count": second,
            "majority_margin": margin,
        })

    y = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    summary = score_predictions(y, p)

    raw = representation(df, variant)
    scaler = scaler_cls()
    x = scaler.fit_transform(raw)
    if include_language:
        vocab = sorted(df["language"].astype(str).unique())
        idx = {v: i for i, v in enumerate(vocab)}
        oh = np.zeros((len(df), len(vocab)), dtype=float)
        for row, value in enumerate(df["language"].astype(str)):
            oh[row, idx[value]] = language_weight
        x = np.hstack([x, oh])

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    clusters = km.fit_predict(x)
    counts = pd.Series(clusters).value_counts()
    summary.update({
        "feature_variant": variant,
        "scaler": scaler_name,
        "k": k,
        "include_language": include_language,
        "language_weight": language_weight if include_language else 0.0,
        "full_fit_purity": purity(df["architecture_preference"].to_numpy(), clusters),
        "silhouette": float(silhouette_score(x, clusters)),
        "davies_bouldin": float(davies_bouldin_score(x, clusters)),
        "calinski_harabasz": float(calinski_harabasz_score(x, clusters)),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "singleton_cluster_count": int(np.sum(counts == 1)),
    })
    baseline = float(df["architecture_preference"].value_counts(normalize=True).max())
    summary["majority_baseline"] = baseline
    summary["strict_gain_over_majority_baseline"] = summary["strict_accuracy"] - baseline
    return summary, targets


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.profiles.resolve(), args.preferences_dir.resolve(), args.threshold, args.language_map.resolve() if args.language_map else None)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    ks = [int(v) for v in args.k_values.split(",") if v.strip()]
    scalers = [v.strip() for v in args.scalers.split(",") if v.strip()]
    language_modes = [(False, 0.0)]
    if args.language_map:
        language_modes += [(True, float(w)) for w in args.language_weights.split(",") if w.strip()]

    summaries, targets = [], []
    total = len(variants) * len(ks) * len(scalers) * len(language_modes)
    done = 0
    for variant in variants:
        for scaler in scalers:
            for k in ks:
                for include_language, weight in language_modes:
                    summary, rows = evaluate(
                        df, variant=variant, scaler_name=scaler, k=k,
                        include_language=include_language, language_weight=weight,
                        n_init=args.n_init, random_state=args.random_state
                    )
                    summaries.append(summary)
                    targets.extend(rows)
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"progress={done}/{total}")

    sdf = pd.DataFrame(summaries).sort_values(
        ["balanced_accuracy", "macro_f1", "directional_accuracy", "strict_accuracy", "coverage"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)
    sdf.insert(0, "rank_exploratory", np.arange(1, len(sdf) + 1))
    sdf.to_csv(out / "behavioral4-ablation-summary.csv", index=False)
    pd.DataFrame(targets).to_csv(out / "behavioral4-ablation-per-target.csv", index=False)
    (out / "behavioral4-ablation-manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "purpose": "behavioral4 feature ablation; exploratory only",
        "threshold_percent": args.threshold,
        "variants": variants,
        "scalers": scalers,
        "k_values": ks,
        "n_init": args.n_init,
        "random_state": args.random_state
    }, indent=2) + "\n", encoding="utf-8")

    cols = [
        "rank_exploratory","feature_variant","scaler","k","include_language","language_weight",
        "coverage","strict_accuracy","strict_gain_over_majority_baseline","balanced_accuracy",
        "macro_f1","directional_accuracy","x86_recall","independent_recall","arm_recall",
        "pred_x86","pred_independent","pred_arm","pred_abstain","full_fit_purity",
        "silhouette","davies_bouldin","calinski_harabasz","min_cluster_size",
        "max_cluster_size","singleton_cluster_count"
    ]
    print("\nTOP ABLATION CONFIGURATIONS:")
    print(sdf[cols].head(args.top_n).to_string(index=False))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--language-map", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--variants", default="paper5_raw,behavioral4,behavioral4_no_page_faults,behavioral4_no_utilized_cpus,behavioral4_no_total_cpu,behavioral4_no_kernel_share,behavioral4_raw_page_faults,behavioral4_separate_cpu")
    p.add_argument("--scalers", default="robust")
    p.add_argument("--k-values", default="7,8,9,10")
    p.add_argument("--language-weights", default="0.25")
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--top-n", type=int, default=30)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
