#!/usr/bin/env python3
"""
Final DBSCAN sensitivity review for the Serverledge algorithm decision.

Reads the already-generated `final-algorithm-comparison-summary.csv`.
No clustering is rerun.

It extracts:
- shortlisted K-Means configurations;
- DBSCAN q75/q80/q85 sensitivity;
- thresholds 2.5%, 5%, 15%.

For every representation/threshold, each DBSCAN sensitivity setting is compared
against the corresponding frozen K-Means configuration on:
- cluster assignment coverage
- donor selection coverage
- directional selection coverage
- directional preference agreement
- multiplicative mismatch (lower is better)
- realized speedup (higher is better)
- mean regret (lower is better)
- p90 regret (lower is better)
- DBSCAN noise

No weighted scalar score is created. A strict Pareto-style dominance flag is
reported only as a descriptive check.

Outputs
-------
dbscan-sensitivity-final.csv
dbscan-vs-kmeans-paired.csv
dbscan-sensitivity-final-report.md
dbscan-sensitivity-final-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


FOCUS_THRESHOLDS = [2.5, 5.0, 15.0]

MAXIMIZE = [
    "cluster_assignment_coverage_mean",
    "selection_coverage_mean",
    "directional_selection_coverage_mean",
    "directional_preference_agreement_mean",
    "directional_geomean_realized_speedup_mean",
]

MINIMIZE = [
    "directional_geomean_multiplicative_mismatch_mean",
    "directional_mean_regret_percent_mean",
    "directional_p90_regret_percent_mean",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q_from_config(config: str) -> int | None:
    m = re.search(r"_q(75|80|85)_", config)
    return int(m.group(1)) if m else None


def strict_dominates(a: pd.Series, b: pd.Series) -> bool:
    no_worse = True
    strictly = False

    for col in MAXIMIZE:
        av = float(a[col])
        bv = float(b[col])
        if av < bv - 1e-12:
            no_worse = False
        if av > bv + 1e-12:
            strictly = True

    for col in MINIMIZE:
        av = float(a[col])
        bv = float(b[col])
        if av > bv + 1e-12:
            no_worse = False
        if av < bv - 1e-12:
            strictly = True

    return no_worse and strictly


def main(args):
    summary_path = args.summary_csv.resolve()
    df = pd.read_csv(summary_path)

    required = {
        "algorithm",
        "algorithm_config",
        "representation",
        "threshold_percent",
        *MAXIMIZE,
        *MINIMIZE,
        "mean_training_noise_fraction_mean",
        "mean_training_cluster_count_mean",
    }

    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    focus = df[
        df["threshold_percent"].isin(FOCUS_THRESHOLDS)
    ].copy()

    dbscan = focus[focus["algorithm"] == "dbscan"].copy()
    dbscan["dbscan_quantile"] = (
        dbscan["algorithm_config"].astype(str).map(q_from_config)
    )
    dbscan = dbscan[
        dbscan["dbscan_quantile"].isin([75, 80, 85])
    ].copy()

    kmeans = focus[focus["algorithm"] == "kmeans"].copy()

    out_cols = [
        "algorithm_config",
        "representation",
        "threshold_percent",
        "dbscan_quantile",
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

    sensitivity = dbscan[out_cols].sort_values(
        ["representation", "threshold_percent", "dbscan_quantile"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_path = (
        args.output_dir / "dbscan-sensitivity-final.csv"
    )
    sensitivity.to_csv(sensitivity_path, index=False)

    paired_rows = []

    for _, drow in dbscan.iterrows():
        matches = kmeans[
            (kmeans["representation"] == drow["representation"])
            & (
                kmeans["threshold_percent"]
                == drow["threshold_percent"]
            )
        ]

        if len(matches) != 1:
            raise SystemExit(
                "Expected exactly one K-Means row for "
                f"{drow['representation']} / "
                f"{drow['threshold_percent']}, got {len(matches)}"
            )

        krow = matches.iloc[0]

        row = {
            "representation": drow["representation"],
            "threshold_percent": float(
                drow["threshold_percent"]
            ),
            "dbscan_quantile": int(
                drow["dbscan_quantile"]
            ),
            "dbscan_config": drow["algorithm_config"],
            "kmeans_config": krow["algorithm_config"],
            "dbscan_strictly_dominates_kmeans": (
                strict_dominates(drow, krow)
            ),
            "kmeans_strictly_dominates_dbscan": (
                strict_dominates(krow, drow)
            ),
            "dbscan_noise_fraction": float(
                drow["mean_training_noise_fraction_mean"]
            ),
        }

        for col in MAXIMIZE + MINIMIZE:
            dv = float(drow[col])
            kv = float(krow[col])

            row[f"dbscan_{col}"] = dv
            row[f"kmeans_{col}"] = kv
            row[
                f"delta_dbscan_minus_kmeans_{col}"
            ] = dv - kv

        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows).sort_values(
        ["representation", "threshold_percent", "dbscan_quantile"]
    )

    paired_path = (
        args.output_dir / "dbscan-vs-kmeans-paired.csv"
    )
    paired.to_csv(paired_path, index=False)

    report_lines = [
        "# Final DBSCAN sensitivity review",
        "",
        (
            "This report uses only already-computed q75/q80/q85 DBSCAN "
            "sensitivity results. It does not retune eps using architecture "
            "ground truth."
        ),
        "",
        "## Strict dominance check",
        "",
    ]

    for (
        rep,
        threshold,
    ), block in paired.groupby(
        ["representation", "threshold_percent"]
    ):
        db_dom = block[
            block["dbscan_strictly_dominates_kmeans"]
        ]["dbscan_quantile"].tolist()

        km_dom = block[
            block["kmeans_strictly_dominates_dbscan"]
        ]["dbscan_quantile"].tolist()

        report_lines.append(
            f"- `{rep}`, t={threshold:.1f}%: "
            f"DBSCAN quantiles strictly dominating K-Means="
            f"{db_dom or 'none'}; "
            f"K-Means strictly dominating DBSCAN quantiles="
            f"{km_dom or 'none'}."
        )

    report_lines += [
        "",
        "## Decision rule",
        "",
        (
            "Do not select q75/q85 retrospectively merely because one produces "
            "the best architecture metric. q80 remains the predeclared primary "
            "DBSCAN setting. q75/q85 are robustness checks. If the qualitative "
            "K-Means-vs-DBSCAN conclusion is stable across them, the clustering "
            "algorithm can be frozen. If sensitivity reverses the conclusion, "
            "report the instability rather than tuning eps post hoc."
        ),
        "",
    ]

    report_path = (
        args.output_dir / "dbscan-sensitivity-final-report.md"
    )
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    manifest = {
        "source_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "focus_thresholds": FOCUS_THRESHOLDS,
        "dbscan_quantiles": [75, 80, 85],
        "primary_dbscan_quantile": 80,
        "decision_metrics_maximize": MAXIMIZE,
        "decision_metrics_minimize": MINIMIZE,
        "selection_note": (
            "q75/q85 are sensitivity only; no post-hoc eps retuning."
        ),
    }

    manifest_path = (
        args.output_dir / "dbscan-sensitivity-final-manifest.json"
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

    print()
    print("DBSCAN q75/q80/q85 SENSITIVITY")
    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("PAIRED STRICT-DOMINANCE CHECK")
    print(
        paired[
            [
                "representation",
                "threshold_percent",
                "dbscan_quantile",
                "dbscan_strictly_dominates_kmeans",
                "kmeans_strictly_dominates_dbscan",
                "dbscan_noise_fraction",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"sensitivity={sensitivity_path}")
    print(f"paired={paired_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Review q75/q80/q85 DBSCAN sensitivity from the already-computed "
            "final K-Means-vs-DBSCAN summary."
        )
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
