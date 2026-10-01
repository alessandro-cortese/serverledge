#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


PAPER5 = "paper6_no_framework_runtime_ms"


def first_csv_with_columns(directory: Path, required: set[str]) -> Path:
    candidates = []
    for path in sorted(directory.rglob("*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=2).columns)
        except Exception:
            continue
        if required.issubset(cols):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one CSV in {directory} containing {sorted(required)}, "
            f"found {candidates}"
        )
    return candidates[0]


def one(df: pd.DataFrame, mask, label: str) -> pd.Series:
    rows = df[mask]
    if len(rows) != 1:
        raise RuntimeError(f"{label}: expected one row, got {len(rows)}")
    return rows.iloc[0]


def pick(row: pd.Series, *names: str):
    for name in names:
        if name in row.index:
            return row[name]
    raise KeyError(f"None of {names} found in {list(row.index)}")


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
        description="Export final K-Means vs DBSCAN evidence from collected Serverledge CSVs."
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--kmeans-transfer-dir",
        type=Path,
        default=Path("data/profiling/analysis-transfer-selective-kmeans-final"),
    )
    ap.add_argument(
        "--dbscan-transfer-dir",
        type=Path,
        default=Path("data/profiling/analysis-transfer-selective-dbscan-final"),
    )
    ap.add_argument(
        "--donor-summary",
        type=Path,
        default=None,
        help="Defaults to ROOT/transfer-donor-lofo-v1/donor-selection-summary.csv",
    )
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--reward-weight", type=float, default=0.25)
    ap.add_argument("--exploration-weight", type=float, default=1.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    km_internal_path = root / "kmeans-scaler-minmax" / "kmeans-internal-summary.csv"
    db_internal_path = root / "dbscan-scaler-robust" / "dbscan-internal-summary.csv"
    donor_path = (
        args.donor_summary.resolve()
        if args.donor_summary
        else root / "transfer-donor-lofo-v1" / "donor-selection-summary.csv"
    )

    km_internal = pd.read_csv(km_internal_path)
    db_internal = pd.read_csv(db_internal_path)
    donor = pd.read_csv(donor_path)

    km = one(
        km_internal,
        (km_internal["feature_set"] == PAPER5)
        & (km_internal["k"].astype(int) == 5),
        "K-Means clustering row",
    )

    db_mask = (
        (db_internal["feature_set"] == PAPER5)
        & (db_internal["metric"].astype(str) == "cosine")
        & (db_internal["min_samples"].astype(int) == 4)
        & (db_internal["eps_quantile"].astype(float).sub(0.80).abs() < 1e-12)
    )
    db = one(db_internal, db_mask, "DBSCAN clustering row")

    donor_km = one(donor, donor["algorithm"].astype(str) == "kmeans", "K-Means donor row")
    donor_db = one(donor, donor["algorithm"].astype(str) == "dbscan", "DBSCAN donor row")

    transfer_required = {
        "horizon",
        "reward_prior_weight",
        "exploration_prior_weight",
        "macro_mean_latency_gain_pct",
        "macro_mean_wrong_choices_saved",
        "macro_mean_convergence_requests_saved",
        "macro_first_arm_optimal_probability",
        "macro_mean_reward_pseudo_regret",
    }

    km_transfer_file = first_csv_with_columns(args.kmeans_transfer_dir.resolve(), transfer_required)
    db_transfer_file = first_csv_with_columns(args.dbscan_transfer_dir.resolve(), transfer_required)

    km_t = pd.read_csv(km_transfer_file)
    db_t = pd.read_csv(db_transfer_file)

    def final_row(df: pd.DataFrame, label: str) -> pd.Series:
        mask = (
            (df["horizon"].astype(int) == args.horizon)
            & (df["reward_prior_weight"].astype(float).sub(args.reward_weight).abs() < 1e-12)
            & (df["exploration_prior_weight"].astype(float).sub(args.exploration_weight).abs() < 1e-12)
        )
        if "formula_mode" in df.columns:
            mask &= df["formula_mode"].astype(str).isin(["decoupled", "transfer-learning"])
        return one(df, mask, label)

    kt = final_row(km_t, "K-Means Transfer Learning row")
    dt = final_row(db_t, "DBSCAN Transfer Learning row")

    def pct(v: float) -> str:
        return f"{100.0 * float(v):.2f}%"

    rows = [
        {
            "family": "clustering",
            "metric": "Global clustering coverage",
            "direction": "higher",
            "K-Means": pct(1.0),
            "DBSCAN selective": pct(float(pick(db, "coverage"))),
            "note": "DBSCAN can label points as noise",
        },
        {
            "family": "clustering",
            "metric": "Noise points",
            "direction": "lower",
            "K-Means": "0",
            "DBSCAN selective": str(int(float(pick(db, "noise_count", "noise_points", "noise_size")))),
            "note": "K-Means always assigns every point",
        },
        {
            "family": "clustering",
            "metric": "Silhouette",
            "direction": "higher",
            "K-Means": f"{float(pick(km, 'silhouette')):.6f}",
            "DBSCAN selective": f"{float(pick(db, 'silhouette_clustered', 'silhouette')):.6f}",
            "note": "DBSCAN silhouette is computed on non-noise points only",
        },
        {
            "family": "donor_selection",
            "metric": "Transfer coverage LOFO",
            "direction": "higher",
            "K-Means": pct(float(pick(donor_km, "selection_coverage"))),
            "DBSCAN selective": pct(float(pick(donor_db, "selection_coverage"))),
            "note": "Share of targets for which a donor is available",
        },
        {
            "family": "donor_selection",
            "metric": "Best-reward-arm agreement",
            "direction": "higher",
            "K-Means": f"{float(pick(donor_km, 'best_reward_arm_agreement_rate')):.4f}",
            "DBSCAN selective": f"{float(pick(donor_db, 'best_reward_arm_agreement_rate')):.4f}",
            "note": "DBSCAN is slightly more selective here",
        },
        {
            "family": "donor_selection",
            "metric": "Label agreement @ tau=15%",
            "direction": "higher",
            "K-Means": f"{float(pick(donor_km, 'label_agreement_rate_t15', 'primary_label_agreement_rate')):.4f}",
            "DBSCAN selective": f"{float(pick(donor_db, 'label_agreement_rate_t15', 'primary_label_agreement_rate')):.4f}",
            "note": "",
        },
        {
            "family": "donor_selection",
            "metric": "Median architecture-delta error",
            "direction": "lower",
            "K-Means": f"{float(pick(donor_km, 'median_abs_architecture_delta_error_percent')):.4f} pp",
            "DBSCAN selective": f"{float(pick(donor_db, 'median_abs_architecture_delta_error_percent')):.4f} pp",
            "note": "DBSCAN is slightly better on accepted donors",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Transfer coverage",
            "direction": "higher",
            "K-Means": pct(float(pick(kt, "transfer_coverage"))),
            "DBSCAN selective": pct(float(pick(dt, "transfer_coverage"))),
            "note": "",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Mean latency gain",
            "direction": "higher",
            "K-Means": f"{float(pick(kt, 'macro_mean_latency_gain_pct')):+.4f}%",
            "DBSCAN selective": f"{float(pick(dt, 'macro_mean_latency_gain_pct')):+.4f}%",
            "note": "Compared with each method's no-transfer baseline",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Latency gain CI95",
            "direction": "",
            "K-Means": (
                f"[{float(pick(kt, 'macro_latency_gain_ci95_low')):+.4f}%, "
                f"{float(pick(kt, 'macro_latency_gain_ci95_high')):+.4f}%]"
            ),
            "DBSCAN selective": (
                f"[{float(pick(dt, 'macro_latency_gain_ci95_low')):+.4f}%, "
                f"{float(pick(dt, 'macro_latency_gain_ci95_high')):+.4f}%]"
            ),
            "note": "These CIs are not a paired K-Means-vs-DBSCAN significance test",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Wrong choices saved",
            "direction": "higher",
            "K-Means": f"{float(pick(kt, 'macro_mean_wrong_choices_saved')):+.4f}",
            "DBSCAN selective": f"{float(pick(dt, 'macro_mean_wrong_choices_saved')):+.4f}",
            "note": "",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Convergence requests saved",
            "direction": "higher",
            "K-Means": f"{float(pick(kt, 'macro_mean_convergence_requests_saved')):+.4f}",
            "DBSCAN selective": f"{float(pick(dt, 'macro_mean_convergence_requests_saved')):+.4f}",
            "note": "",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "First-arm optimal probability",
            "direction": "higher",
            "K-Means": f"{float(pick(kt, 'macro_first_arm_optimal_probability')):.4f}",
            "DBSCAN selective": f"{float(pick(dt, 'macro_first_arm_optimal_probability')):.4f}",
            "note": "",
        },
        {
            "family": f"transfer_learning_H{args.horizon}",
            "metric": "Reward pseudo-regret",
            "direction": "lower",
            "K-Means": f"{float(pick(kt, 'macro_mean_reward_pseudo_regret')):.6f}",
            "DBSCAN selective": f"{float(pick(dt, 'macro_mean_reward_pseudo_regret')):.6f}",
            "note": "Cumulative pseudo-regret in reward space",
        },
    ]

    table = pd.DataFrame(rows)
    csv_path = out / "kmeans-vs-dbscan-final-evidence.csv"
    table.to_csv(csv_path, index=False)

    md = f"""# K-Means vs DBSCAN — evidenza finale

Sorgenti:

- `{km_internal_path}`
- `{db_internal_path}`
- `{donor_path}`
- `{km_transfer_file}`
- `{db_transfer_file}`

Configurazione Transfer Learning:

- horizon = {args.horizon}
- reward prior weight = {args.reward_weight}
- exploration pseudo-count = {args.exploration_weight}

{markdown_table(table)}

## Lettura del trade-off

DBSCAN mostra una piccola selettività utile nella donor selection: sui donor accettati
può aumentare leggermente il best-arm agreement e ridurre leggermente l'errore del
delta architetturale.

Il costo è una coverage inferiore. Nel replay finale questa selettività non si
trasforma in un vantaggio downstream maggiore: la pipeline K-Means conserva coverage
completa e mostra valori osservati più favorevoli nelle principali metriche H={args.horizon}.

La conclusione corretta è quindi specifica al corpus e alla pipeline Serverledge:
l'astensione DBSCAN non compensa, nei dati raccolti, la perdita di coverage.

Nota: gli intervalli di confidenza riportati misurano il gain di ciascuna pipeline
rispetto al proprio baseline. Non costituiscono un test paired diretto della differenza
K-Means-vs-DBSCAN.
"""
    md_path = out / "kmeans-vs-dbscan-final-evidence.md"
    md_path.write_text(md, encoding="utf-8")

    print("PASS")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
