#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PAPER6 = "paper6"
PAPER5 = "paper6_no_framework_runtime_ms"

INTERNAL_METRICS = [
    ("silhouette", "Silhouette", "higher"),
    ("davies_bouldin", "Davies-Bouldin", "lower"),
    ("calinski_harabasz", "Calinski-Harabasz", "higher"),
]

EXTERNAL_METRICS = [
    ("overall_purity", "Purity", "higher"),
    ("purity_gain_over_majority_baseline", "Purity gain", "higher"),
    ("homogeneity", "Homogeneity", "higher"),
    ("completeness", "Completeness", "higher"),
    ("v_measure", "V-measure", "higher"),
    ("adjusted_rand_index", "GT-ARI", "higher"),
    ("normalized_mutual_information", "NMI", "higher"),
]


def improvement(old: float, new: float, direction: str, tol: float = 1e-12) -> str:
    diff = new - old
    if abs(diff) <= tol:
        return "same"
    if direction == "higher":
        return "better" if diff > 0 else "worse"
    if direction == "lower":
        return "better" if diff < 0 else "worse"
    raise ValueError(direction)


def fmt(v: float, digits: int = 6) -> str:
    return f"{v:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export PAPER-6 vs PAPER-5 ablation tables from the clustering "
            "study CSVs. PAPER-5 = PAPER-6 without framework_runtime_ms."
        )
    )
    ap.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="Directory containing kmeans-internal-summary.csv and kmeans-ground-truth-summary.csv",
    )
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument(
        "--focus-k",
        type=int,
        default=5,
        help="K highlighted in the detailed metric table",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    study = args.study_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    internal_path = study / "kmeans-internal-summary.csv"
    external_path = study / "kmeans-ground-truth-summary.csv"

    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)

    required_sets = {PAPER6, PAPER5}
    if not required_sets.issubset(set(internal["feature_set"])):
        raise RuntimeError(
            f"Missing required feature sets in {internal_path}: {required_sets}"
        )

    # ---------------- all-K internal comparison ----------------
    p6i = internal[internal["feature_set"] == PAPER6].copy()
    p5i = internal[internal["feature_set"] == PAPER5].copy()

    keep_i = [
        "k",
        "cluster_size_distribution",
        "singleton_count",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "inertia",
    ]
    p6i = p6i[keep_i].rename(
        columns={c: f"paper6_{c}" for c in keep_i if c != "k"}
    )
    p5i = p5i[keep_i].rename(
        columns={c: f"paper5_{c}" for c in keep_i if c != "k"}
    )

    merged_i = p6i.merge(p5i, on="k", validate="one_to_one").sort_values("k")

    sweep_rows = []
    for _, r in merged_i.iterrows():
        row = {
            "K": int(r["k"]),
            "PAPER-6 clusters": r["paper6_cluster_size_distribution"],
            "PAPER-5 clusters": r["paper5_cluster_size_distribution"],
            "PAPER-6 singleton": int(r["paper6_singleton_count"]),
            "PAPER-5 singleton": int(r["paper5_singleton_count"]),
        }
        for col, label, direction in INTERNAL_METRICS:
            old = float(r[f"paper6_{col}"])
            new = float(r[f"paper5_{col}"])
            row[f"{label} P6"] = old
            row[f"{label} P5"] = new
            row[f"Delta {label}"] = new - old
            row[f"Effect {label}"] = improvement(old, new, direction)
        # Keep inertia for provenance, but do not label it better/worse because
        # the feature dimensionality differs (6D vs 5D).
        row["Inertia P6"] = float(r["paper6_inertia"])
        row["Inertia P5"] = float(r["paper5_inertia"])
        row["Delta Inertia"] = float(r["paper5_inertia"]) - float(r["paper6_inertia"])
        row["Inertia comparability"] = "not used for P6-vs-P5 decision (different dimensionality)"
        sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    sweep_csv = out / "framework-runtime-ablation-all-k.csv"
    sweep.to_csv(sweep_csv, index=False)

    # ---------------- focus K with external metrics ----------------
    p6e = external[
        (external["feature_set"] == PAPER6)
        & (external["k"].astype(int) == args.focus_k)
        & (external["threshold_percent"].astype(float).sub(args.threshold).abs() < 1e-12)
    ]
    p5e = external[
        (external["feature_set"] == PAPER5)
        & (external["k"].astype(int) == args.focus_k)
        & (external["threshold_percent"].astype(float).sub(args.threshold).abs() < 1e-12)
    ]

    p6if = internal[
        (internal["feature_set"] == PAPER6)
        & (internal["k"].astype(int) == args.focus_k)
    ]
    p5if = internal[
        (internal["feature_set"] == PAPER5)
        & (internal["k"].astype(int) == args.focus_k)
    ]

    if not (len(p6e) == len(p5e) == len(p6if) == len(p5if) == 1):
        raise RuntimeError(
            f"Expected exactly one row per feature set at K={args.focus_k}, tau={args.threshold}"
        )

    p6e, p5e = p6e.iloc[0], p5e.iloc[0]
    p6if, p5if = p6if.iloc[0], p5if.iloc[0]

    detail_rows = []

    for col, label, direction in INTERNAL_METRICS:
        old = float(p6if[col])
        new = float(p5if[col])
        detail_rows.append(
            {
                "family": "internal",
                "metric": label,
                "direction": "higher is better" if direction == "higher" else "lower is better",
                "PAPER-6_with_framework": old,
                "PAPER-5_without_framework": new,
                "delta_P5_minus_P6": new - old,
                "effect_of_removal": improvement(old, new, direction),
            }
        )

    for col, label, direction in EXTERNAL_METRICS:
        old = float(p6e[col])
        new = float(p5e[col])
        detail_rows.append(
            {
                "family": "external_posthoc",
                "metric": label,
                "direction": "higher is better",
                "PAPER-6_with_framework": old,
                "PAPER-5_without_framework": new,
                "delta_P5_minus_P6": new - old,
                "effect_of_removal": improvement(old, new, direction),
            }
        )

    detail = pd.DataFrame(detail_rows)
    detail_csv = out / f"framework-runtime-ablation-k{args.focus_k}-tau{args.threshold:g}.csv"
    detail.to_csv(detail_csv, index=False)

    # Markdown-friendly copies
    sweep_md = sweep.copy()
    numeric_cols = [
        c for c in sweep_md.columns
        if c.startswith("Silhouette")
        or c.startswith("Davies-Bouldin")
        or c.startswith("Calinski-Harabasz")
        or c.startswith("Delta Silhouette")
        or c.startswith("Delta Davies-Bouldin")
        or c.startswith("Delta Calinski-Harabasz")
    ]
    for c in numeric_cols:
        sweep_md[c] = sweep_md[c].map(lambda x: fmt(float(x), 4))

    detail_md = detail.copy()
    for c in [
        "PAPER-6_with_framework",
        "PAPER-5_without_framework",
        "delta_P5_minus_P6",
    ]:
        detail_md[c] = detail_md[c].map(lambda x: fmt(float(x), 6))

    md = f"""# Ablation di `framework_runtime_ms`

Sorgente dati:

- `{internal_path}`
- `{external_path}`

Confronto:

- PAPER-6 = feature set completo con `framework_runtime_ms`
- PAPER-5 = PAPER-6 senza `framework_runtime_ms`

## Metriche interne su tutto lo sweep di K

{markdown_table(sweep_md[[
    "K",
    "PAPER-6 clusters",
    "PAPER-5 clusters",
    "Silhouette P6",
    "Silhouette P5",
    "Delta Silhouette",
    "Effect Silhouette",
    "Davies-Bouldin P6",
    "Davies-Bouldin P5",
    "Delta Davies-Bouldin",
    "Effect Davies-Bouldin",
    "Calinski-Harabasz P6",
    "Calinski-Harabasz P5",
    "Delta Calinski-Harabasz",
    "Effect Calinski-Harabasz",
]])}

### Nota sull'inertia

L'inertia viene esportata nel CSV per completezza, ma non viene usata come prova
PAPER-6 vs PAPER-5 perché i due spazi hanno dimensionalità diversa (6 feature vs 5).
Una riduzione dell'inertia può dipendere meccanicamente anche dalla rimozione di una dimensione.

## Dettaglio a K={args.focus_k}, tau={args.threshold:g}%

{markdown_table(detail_md)}

## Interpretazione metodologica

La scelta di rimuovere `framework_runtime_ms` non viene basata sull'ottimizzazione
delle metriche esterne. La motivazione primaria è semantica: la feature rappresenta
overhead del profiler/framework e non comportamento applicativo della funzione.

L'ablation fornisce evidenza quantitativa di supporto: le metriche interne mostrano
come cambia la geometria quando la feature viene rimossa. Le metriche esterne sono
riportate soltanto post-hoc e non vengono usate per scegliere il feature set, evitando
data leakage dalla ground truth architetturale.

Se a un determinato K le metriche esterne restano identiche, ciò è coerente con una
membership invariata: non è un problema, ma indica che il miglioramento osservato è
geometrico e non deriva dall'aver adattato il clustering alla ground truth.
"""

    md_path = out / "framework-runtime-ablation.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"PASS")
    print(f"all_k_csv={sweep_csv}")
    print(f"focus_csv={detail_csv}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
