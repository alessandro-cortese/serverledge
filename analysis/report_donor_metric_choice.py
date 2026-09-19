from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_DIR = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "analysis-transfer-donor-metrics-20260918"
)

SUMMARY_CSV = INPUT_DIR / "donor-metric-summary.csv"

OUTPUT_DIR = INPUT_DIR / "presentation_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

if not SUMMARY_CSV.exists():
    raise SystemExit(
        f"File non trovato:\n{SUMMARY_CSV}"
    )

df = pd.read_csv(SUMMARY_CSV)

required = {
    "algorithm",
    "ranker",
    "coverage",
    "best_reward_arm_agreement_rate",
    "mean_abs_reward_gap_error",
    "median_abs_reward_gap_error",
}

missing = required - set(df.columns)
if missing:
    raise SystemExit(
        "Colonne mancanti nel CSV: "
        + ", ".join(sorted(missing))
    )


# ============================================================
# Export raw summary
# ============================================================

df.to_csv(
    OUTPUT_DIR / "donor_quality_full.csv",
    index=False,
    )


# ============================================================
# KMeans only
# ============================================================

kmeans = (
    df[df["algorithm"] == "kmeans"]
    .copy()
    .sort_values(
        "best_reward_arm_agreement_rate",
        ascending=False,
    )
)

kmeans.to_csv(
    OUTPUT_DIR / "donor_quality_kmeans.csv",
    index=False,
    )


# ============================================================
# Manhattan vs Euclidean
# ============================================================

comparison = kmeans[
    kmeans["ranker"].isin(
        ["euclidean", "manhattan"]
    )
].copy()

comparison.to_csv(
    OUTPUT_DIR
    / "kmeans_manhattan_vs_euclidean.csv",
    index=False,
    )

print("\n===== KMEANS: MANHATTAN VS EUCLIDEAN =====")
print(
    comparison[
        [
            "ranker",
            "coverage",
            "best_reward_arm_agreement_rate",
            "median_abs_reward_gap_error",
        ]
    ].to_string(index=False)
)


# ============================================================
# Figure 1:
# direct Manhattan vs Euclidean arm agreement
# ============================================================

comparison = comparison.sort_values("ranker")

labels = comparison["ranker"].tolist()

agreement_pct = (
        comparison[
            "best_reward_arm_agreement_rate"
        ].to_numpy()
        * 100.0
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    labels,
    agreement_pct,
)

plt.ylabel("Best-arm agreement (%)")
plt.title(
    "KMeans donor selection:\n"
    "Manhattan vs Euclidean"
)
plt.ylim(0, 100)
plt.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
        bars,
        agreement_pct,
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "kmeans_manhattan_vs_euclidean.png",
    dpi=200,
    )

plt.close()


# ============================================================
# Figure 2:
# KMeans vs DBSCAN using best donor ranker
# ============================================================

best_rows = []

for algorithm in [
    "kmeans",
    "dbscan",
]:
    subset = df[
        df["algorithm"] == algorithm
        ].copy()

    best = subset.sort_values(
        "best_reward_arm_agreement_rate",
        ascending=False,
    ).iloc[0]

    best_rows.append(best)

best_df = pd.DataFrame(best_rows)

best_df.to_csv(
    OUTPUT_DIR
    / "best_configuration_per_clusterer.csv",
    index=False,
    )

labels = [
    (
        f"{row.algorithm}\n"
        f"({row.ranker})"
    )
    for row in best_df.itertuples()
]

coverage_pct = (
        best_df["coverage"].to_numpy()
        * 100.0
)

agreement_pct = (
        best_df[
            "best_reward_arm_agreement_rate"
        ].to_numpy()
        * 100.0
)

x = np.arange(
    len(labels)
)

width = 0.35

plt.figure(
    figsize=(8, 5),
)

plt.bar(
    x - width / 2,
    coverage_pct,
    width,
    label="Coverage",
    )

plt.bar(
    x + width / 2,
    agreement_pct,
    width,
    label="Best-arm agreement",
    )

plt.xticks(
    x,
    labels,
)

plt.ylabel("Percentage (%)")
plt.title(
    "Operational donor-selection comparison"
)
plt.ylim(0, 105)
plt.legend()
plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "kmeans_vs_dbscan_best_configuration.png",
    dpi=200,
    )

plt.close()


# ============================================================
# Markdown summary
# ============================================================

euclidean = comparison[
    comparison["ranker"]
    == "euclidean"
    ].iloc[0]

manhattan = comparison[
    comparison["ranker"]
    == "manhattan"
    ].iloc[0]

delta_pp = (
                   manhattan[
                       "best_reward_arm_agreement_rate"
                   ]
                   - euclidean[
                       "best_reward_arm_agreement_rate"
                   ]
           ) * 100.0

best_kmeans = best_df[
    best_df["algorithm"]
    == "kmeans"
    ].iloc[0]

best_dbscan = best_df[
    best_df["algorithm"]
    == "dbscan"
    ].iloc[0]

markdown = f"""# Donor-selection decision

## KMeans: Manhattan vs Euclidean

| Ranker | Coverage | Best-arm agreement | Median gap error |
|---|---:|---:|---:|
| Euclidean | {euclidean['coverage']:.4f} | {euclidean['best_reward_arm_agreement_rate']:.4f} | {euclidean['median_abs_reward_gap_error']:.4f} |
| Manhattan | {manhattan['coverage']:.4f} | {manhattan['best_reward_arm_agreement_rate']:.4f} | {manhattan['median_abs_reward_gap_error']:.4f} |

Manhattan improves best-arm agreement by **{delta_pp:.2f} percentage points**
relative to Euclidean while preserving the same full coverage.

## KMeans vs DBSCAN

Best KMeans configuration:

- ranker: `{best_kmeans['ranker']}`
- coverage: `{best_kmeans['coverage']:.4f}`
- best-arm agreement: `{best_kmeans['best_reward_arm_agreement_rate']:.4f}`

Best DBSCAN configuration:

- ranker: `{best_dbscan['ranker']}`
- coverage: `{best_dbscan['coverage']:.4f}`
- best-arm agreement: `{best_dbscan['best_reward_arm_agreement_rate']:.4f}`

## Methodological decision

KMeans remains the operational clustering method because it provides full
coverage for unseen targets.

The new target is assigned to a KMeans cluster using the Euclidean geometry
intrinsic to KMeans.

After cluster assignment, donor selection is treated as a separate
nearest-neighbour ranking problem. Among the tested KMeans donor rankers,
Manhattan produces the highest observed best-arm agreement while preserving
100% coverage.

Therefore the frozen transfer pipeline is:

x86 PAPER-5 profile
→ MinMax scaling
→ KMeans assignment using Euclidean geometry
→ same-cluster donor candidates
→ Manhattan donor ranking
→ weak-prior transfer.
"""

(
        OUTPUT_DIR
        / "donor_choice_summary.md"
).write_text(
    markdown,
    encoding="utf-8",
)

print(
    "\nOutput salvato in:",
    OUTPUT_DIR,
)

for path in sorted(
        OUTPUT_DIR.iterdir()
):
    print(" -", path.name)