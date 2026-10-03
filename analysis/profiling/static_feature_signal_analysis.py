#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, RobustScaler

ARCH_LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]

def threshold_slug(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")

def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out

def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    if n <= k:
        return float("nan")
    return float(max(0.0, (H - k + 1) / (n - k)))

def safe_kruskal(values: pd.Series, groups: pd.Series):
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    grouped = [g["v"].to_numpy(dtype=float) for _, g in frame.groupby("g", sort=False) if len(g)]
    if len(grouped) < 2:
        return float("nan"), float("nan"), float("nan")
    if frame["v"].nunique() <= 1:
        return 0.0, 1.0, 0.0
    H, p = kruskal(*grouped)
    return float(H), float(p), epsilon_squared_kruskal(float(H), len(frame), len(grouped))

def mutual_information_one_feature(values, labels, discrete, seed):
    frame = pd.DataFrame({"v": values, "y": labels}).dropna()
    if frame["v"].nunique() <= 1 or frame["y"].nunique() <= 1:
        return 0.0
    x = frame[["v"]].to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(frame["y"].astype(str))
    if not discrete:
        x = RobustScaler().fit_transform(x)
    mi = mutual_info_classif(
        x, y, discrete_features=[discrete], random_state=seed
    )
    return float(mi[0])

def label_stats(df, metric):
    out = {}
    for label in ARCH_LABELS:
        vals = df.loc[df["architecture_preference"] == label, metric].dropna()
        prefix = "x86" if label == "x86-preferred" else ("independent" if label == "architecture-independent" else "arm")
        out[f"{prefix}_n"] = int(len(vals))
        out[f"{prefix}_median"] = float(vals.median()) if len(vals) else np.nan
        out[f"{prefix}_mean"] = float(vals.mean()) if len(vals) else np.nan
    return out

def build_metric_table(df, metrics, seed):
    rows = []
    for metric in metrics:
        is_behavioral = metric.endswith("_signal_count")
        arch_H, arch_p, arch_eps = safe_kruskal(df[metric], df["architecture_preference"])
        lang_H, lang_p, lang_eps = safe_kruskal(df[metric], df["language"])
        row = {
            "metric": metric,
            "family": "source_behavioral" if is_behavioral else "structural",
            "n": int(df[metric].notna().sum()),
            "unique_values": int(df[metric].nunique(dropna=True)),
            "zero_fraction": float((df[metric].fillna(0) == 0).mean()),
            "arch_kruskal_H": arch_H,
            "arch_p": arch_p,
            "arch_epsilon_squared": arch_eps,
            "arch_mutual_information": mutual_information_one_feature(
                df[metric], df["architecture_preference"], is_behavioral, seed
            ),
            "language_kruskal_H": lang_H,
            "language_p": lang_p,
            "language_epsilon_squared": lang_eps,
        }
        row.update(label_stats(df, metric))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["arch_q_bh"] = bh_fdr(out["arch_p"].fillna(1.0).to_numpy())
        out["language_q_bh"] = bh_fdr(out["language_p"].fillna(1.0).to_numpy())
        out = out.sort_values(
            ["arch_q_bh", "arch_epsilon_squared", "arch_mutual_information"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return out

def correlation_table(df, metrics):
    rows = []
    for i, a in enumerate(metrics):
        for b in metrics[i+1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < 3 or pair[a].nunique() <= 1 or pair[b].nunique() <= 1:
                continue
            rho, p = spearmanr(pair[a], pair[b])
            rows.append({
                "metric_a": a,
                "metric_b": b,
                "n": len(pair),
                "spearman_rho": float(rho),
                "abs_spearman_rho": float(abs(rho)),
                "p": float(p),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)
    return out

def prevalence_table(df, behavioral):
    rows = []
    for metric in behavioral:
        for label in ARCH_LABELS:
            sub = df[df["architecture_preference"] == label]
            rows.append({
                "metric": metric,
                "architecture_preference": label,
                "n": len(sub),
                "nonzero_n": int((sub[metric].fillna(0) > 0).sum()),
                "nonzero_fraction": float((sub[metric].fillna(0) > 0).mean()) if len(sub) else np.nan,
                "median_count": float(sub[metric].median()) if len(sub) else np.nan,
                "mean_count": float(sub[metric].mean()) if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)

def main(args):
    static = pd.read_csv(args.static_metrics.resolve())
    pref = pd.read_csv(args.preferences_dir.resolve() / f"preferences-{threshold_slug(args.threshold)}.csv")

    df = static.merge(
        pref[["function_name", "architecture_preference"]],
        on="function_name",
        validate="one_to_one",
    )
    complete = df[df["static_source_available"].astype(bool)].copy()

    structural = [
        c for c in complete.columns
        if c.startswith("static_")
        and c != "static_source_available"
        and pd.api.types.is_numeric_dtype(complete[c])
    ]
    behavioral = [
        c for c in complete.columns
        if c.endswith("_signal_count")
        and pd.api.types.is_numeric_dtype(complete[c])
    ]
    metrics = structural + behavioral

    args.output_dir.mkdir(parents=True, exist_ok=True)

    signal = build_metric_table(complete, metrics, args.seed)
    signal.to_csv(args.output_dir / "static-feature-signal.csv", index=False)

    corr = correlation_table(complete, metrics)
    corr.to_csv(args.output_dir / "static-feature-spearman.csv", index=False)

    prevalence = prevalence_table(complete, behavioral)
    prevalence.to_csv(args.output_dir / "static-behavioral-prevalence.csv", index=False)

    print(
        f"rows_total={len(df)} "
        f"rows_complete_source={len(complete)} "
        f"rows_missing_source={len(df)-len(complete)} "
        f"structural_metrics={len(structural)} "
        f"behavioral_metrics={len(behavioral)}"
    )

    print("\nCOMPLETE-CASE CLASS COUNTS:")
    print(
        complete["architecture_preference"]
        .value_counts()
        .reindex(ARCH_LABELS, fill_value=0)
        .to_string()
    )

    print("\nCOMPLETE-CASE LANGUAGE COUNTS:")
    print(complete["language"].value_counts().to_string())

    display_cols = [
        "metric","family","unique_values","zero_fraction",
        "arch_kruskal_H","arch_p","arch_q_bh","arch_epsilon_squared",
        "arch_mutual_information","language_p","language_q_bh",
        "language_epsilon_squared","x86_median","independent_median","arm_median",
    ]
    print("\nSTATIC FEATURE SIGNAL — SORTED BY ARCHITECTURE FDR:")
    print(signal[display_cols].to_string(index=False))

    print("\nHIGH STATIC REDUNDANCY (|Spearman rho| >= 0.80):")
    if len(corr):
        high = corr[corr["abs_spearman_rho"] >= 0.80]
        if len(high):
            print(high[["metric_a","metric_b","spearman_rho","p"]].to_string(index=False))
        else:
            print("none")
    else:
        print("none")

    print("\nSOURCE-BEHAVIORAL NONZERO PREVALENCE BY ARCHITECTURE:")
    print(prevalence.to_string(index=False) if len(prevalence) else "none")

    print(
        "\nIMPORTANT: exploratory ground-truth-aware signal analysis only. "
        "The next step must compare predefined feature groups with LOFO/nested validation."
    )

def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--static-metrics", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    return p

if __name__ == "__main__":
    main(parser().parse_args())
