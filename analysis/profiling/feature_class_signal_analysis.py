#!/usr/bin/env python3
"""Diagnostic analysis of how much architecture-preference signal is present
in the current Serverledge profiling features.

This script does not alter the clustering pipeline.  It measures whether the
current x86-reference features contain information about the post-hoc
three-class architecture ground truth and provides a small supervised probe as
a diagnostic only (not as the final solution).

Outputs:
  numeric-feature-class-stats.csv
  numeric-feature-signal-summary.csv
  numeric-feature-vs-architecture-delta.csv
  language-ground-truth-crosstab.csv
  language-ground-truth-summary.csv
  supervised-probe-summary.csv
  supervised-probe-confusion-matrix.csv
  feature-signal-manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PAPER5_FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]
LABELS = [
    "x86-preferred",
    "architecture-independent",
    "arm-preferred",
]
DIRECTIONAL = {"x86-preferred", "arm-preferred"}


def threshold_slug(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment."""
    arr = np.asarray(pvalues, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        running = min(running, candidate)
        adjusted[i] = min(1.0, running)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def effect_epsilon_squared(h: float, n: int, k: int) -> float:
    if n <= k:
        return float("nan")
    return max(0.0, float((h - k + 1.0) / (n - k)))


def load_data(profiles: Path, preferences_dir: Path, threshold: float, language_map: Path | None) -> pd.DataFrame:
    prof = pd.read_csv(profiles)
    required = {"function_name", *PAPER5_FEATURES}
    missing = sorted(required - set(prof.columns))
    if missing:
        raise ValueError(f"{profiles}: missing columns {missing}")

    pref_path = preferences_dir / f"preferences-{threshold_slug(threshold)}.csv"
    pref = pd.read_csv(pref_path)
    pref_required = {"function_name", "architecture_preference", "arm_vs_x86_delta_percent"}
    missing = sorted(pref_required - set(pref.columns))
    if missing:
        raise ValueError(f"{pref_path}: missing columns {missing}")

    df = prof.merge(
        pref[["function_name", "architecture_preference", "arm_vs_x86_delta_percent"]],
        on="function_name",
        how="inner",
        validate="one_to_one",
    )
    if len(df) != len(prof):
        raise ValueError("profile/preference function sets do not match")

    if language_map is not None:
        lang = pd.read_csv(language_map)
        if not {"function_name", "language"}.issubset(lang.columns):
            raise ValueError("language map must contain function_name,language")
        lang = lang[["function_name", "language"]].copy()
        if lang["language"].isna().any() or (lang["language"].astype(str).str.strip() == "").any():
            missing_names = lang.loc[
                lang["language"].isna() | (lang["language"].astype(str).str.strip() == ""),
                "function_name",
            ].tolist()
            raise ValueError(f"language map contains unresolved functions: {missing_names}")
        lang["language"] = lang["language"].astype(str).str.strip().str.lower()
        df = df.merge(lang, on="function_name", how="left", validate="one_to_one")
        if df["language"].isna().any():
            raise ValueError("language map does not cover all profiled functions")
    return df


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isin(y_true, list(DIRECTIONAL))
    return float(np.mean(y_true[mask] == y_pred[mask])) if np.any(mask) else float("nan")


def probe_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str | int]:
    return {
        "model": name,
        "target_count": len(y_true),
        "strict_accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def run(args: argparse.Namespace) -> None:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.profiles.resolve(), args.preferences_dir.resolve(), args.threshold, args.language_map.resolve() if args.language_map else None)

    # ---- Per-class numeric descriptions ----
    stats_rows: list[dict] = []
    signal_rows: list[dict] = []
    delta_rows: list[dict] = []
    kw_pvalues: list[float] = []
    rho_pvalues: list[float] = []

    y_codes = pd.Categorical(df["architecture_preference"], categories=LABELS).codes
    mi = mutual_info_classif(df[PAPER5_FEATURES].astype(float).to_numpy(), y_codes, random_state=args.random_state)

    for feature, mi_value in zip(PAPER5_FEATURES, mi):
        grouped = []
        for label in LABELS:
            values = df.loc[df["architecture_preference"] == label, feature].astype(float).to_numpy()
            grouped.append(values)
            stats_rows.append({
                "feature": feature,
                "class_label": label,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
        h, p = kruskal(*grouped)
        kw_pvalues.append(float(p))
        signal_rows.append({
            "feature": feature,
            "kruskal_h": float(h),
            "kruskal_p": float(p),
            "kruskal_epsilon_squared": effect_epsilon_squared(float(h), len(df), len(LABELS)),
            "mutual_information": float(mi_value),
        })

        rho, p_rho = spearmanr(df[feature].astype(float), df["arm_vs_x86_delta_percent"].astype(float))
        rho_pvalues.append(float(p_rho))
        delta_rows.append({
            "feature": feature,
            "spearman_rho_vs_architecture_delta": float(rho),
            "spearman_p": float(p_rho),
        })

    kw_q = bh_adjust(kw_pvalues)
    for row, q in zip(signal_rows, kw_q):
        row["kruskal_fdr_bh_q"] = q
    rho_q = bh_adjust(rho_pvalues)
    for row, q in zip(delta_rows, rho_q):
        row["spearman_fdr_bh_q"] = q

    pd.DataFrame(stats_rows).to_csv(out / "numeric-feature-class-stats.csv", index=False)
    pd.DataFrame(signal_rows).sort_values(["kruskal_fdr_bh_q", "mutual_information"], ascending=[True, False]).to_csv(out / "numeric-feature-signal-summary.csv", index=False)
    pd.DataFrame(delta_rows).sort_values("spearman_fdr_bh_q").to_csv(out / "numeric-feature-vs-architecture-delta.csv", index=False)

    # ---- Language association ----
    language_summary = []
    if "language" in df.columns:
        table = pd.crosstab(df["language"], df["architecture_preference"]).reindex(columns=LABELS, fill_value=0)
        table_out = table.reset_index()
        table_out["total"] = table_out[LABELS].sum(axis=1)
        for label in LABELS:
            table_out[f"share_{label}"] = table_out[label] / table_out["total"]
        table_out.to_csv(out / "language-ground-truth-crosstab.csv", index=False)

        chi2, p, dof, _ = chi2_contingency(table.to_numpy())
        n = int(table.to_numpy().sum())
        r, c = table.shape
        denom = n * max(1, min(r - 1, c - 1))
        cramers_v = math.sqrt(float(chi2) / denom) if denom > 0 else float("nan")
        language_summary.append({
            "n": n,
            "language_count": r,
            "class_count": c,
            "chi_square": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(p),
            "cramers_v": float(cramers_v),
        })
        pd.DataFrame(language_summary).to_csv(out / "language-ground-truth-summary.csv", index=False)

    # ---- Small supervised probe: diagnostic only ----
    y = df["architecture_preference"].to_numpy()
    loo = LeaveOneOut()
    probe_rows: list[dict] = []
    confusion_rows: list[dict] = []

    majority = df["architecture_preference"].value_counts().idxmax()
    majority_pred = np.repeat(majority, len(df))
    probes: list[tuple[str, np.ndarray]] = [("majority_baseline", majority_pred)]

    numeric_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=args.random_state)),
    ])
    numeric_pred = cross_val_predict(numeric_pipeline, df[PAPER5_FEATURES], y, cv=loo)
    probes.append(("balanced_logistic_numeric", numeric_pred))

    if "language" in df.columns:
        pre = ColumnTransformer([
            ("numeric", StandardScaler(), PAPER5_FEATURES),
            ("language", OneHotEncoder(handle_unknown="ignore"), ["language"]),
        ])
        language_pipeline = Pipeline([
            ("preprocess", pre),
            ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=args.random_state)),
        ])
        lang_pred = cross_val_predict(language_pipeline, df[PAPER5_FEATURES + ["language"]], y, cv=loo)
        probes.append(("balanced_logistic_numeric_plus_language", lang_pred))

    for name, pred in probes:
        probe_rows.append(probe_metrics(name, y, pred))
        for truth in LABELS:
            for predicted in LABELS:
                confusion_rows.append({
                    "model": name,
                    "true_label": truth,
                    "predicted_label": predicted,
                    "count": int(np.sum((y == truth) & (pred == predicted))),
                })

    pd.DataFrame(probe_rows).to_csv(out / "supervised-probe-summary.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "supervised-probe-confusion-matrix.csv", index=False)

    manifest = {
        "schema_version": 1,
        "purpose": "diagnostic feature-to-architecture signal analysis; supervised probes are diagnostics only and are not the clustering pipeline",
        "threshold_percent": args.threshold,
        "features": PAPER5_FEATURES,
        "target_count": len(df),
        "labels": LABELS,
        "language_map": str(args.language_map.resolve()) if args.language_map else None,
        "random_state": args.random_state,
    }
    (out / "feature-signal-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"targets={len(df)} output={out}")
    print("\nNumeric feature signal:")
    show = pd.DataFrame(signal_rows).sort_values(["kruskal_fdr_bh_q", "mutual_information"], ascending=[True, False])
    print(show.to_string(index=False))
    if language_summary:
        print("\nLanguage x ground truth:")
        print(pd.read_csv(out / "language-ground-truth-crosstab.csv").to_string(index=False))
        print(pd.DataFrame(language_summary).to_string(index=False))
    print("\nSupervised diagnostic probes (LOFO):")
    print(pd.DataFrame(probe_rows).to_string(index=False))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", required=True, type=Path)
    p.add_argument("--preferences-dir", required=True, type=Path)
    p.add_argument("--language-map", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--random-state", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
