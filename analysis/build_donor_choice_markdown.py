from pathlib import Path
import pandas as pd

INPUT = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports/donor_quality_full.csv")
OUTFILE = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports/donor_choice_summary.md")

df = pd.read_csv(INPUT)

kmeans = df[df["clusterer"].str.lower() == "kmeans"].copy()
dbscan = df[df["clusterer"].str.lower() == "dbscan"].copy()

best_kmeans = kmeans.sort_values("arm_match", ascending=False).iloc[0]
best_dbscan = dbscan.sort_values("arm_match", ascending=False).iloc[0]

euclidean = kmeans[kmeans["distance"].str.lower() == "euclidean"].iloc[0]
manhattan = kmeans[kmeans["distance"].str.lower() == "manhattan"].iloc[0]

text = f"""# Donor selection summary

## Key results

### KMeans vs DBSCAN
- Best KMeans configuration:
  - distance = {best_kmeans['distance']}
  - coverage = {best_kmeans['coverage']:.4f}
  - arm_match = {best_kmeans['arm_match']:.4f}

- Best DBSCAN configuration:
  - distance = {best_dbscan['distance']}
  - coverage = {best_dbscan['coverage']:.4f}
  - arm_match = {best_dbscan['arm_match']:.4f}

### Manhattan vs Euclidean inside KMeans
- Euclidean:
  - coverage = {euclidean['coverage']:.4f}
  - arm_match = {euclidean['arm_match']:.4f}

- Manhattan:
  - coverage = {manhattan['coverage']:.4f}
  - arm_match = {manhattan['arm_match']:.4f}

## Interpretation
KMeans is preferred as operational clustering model because it provides full coverage on all targets.
Within KMeans, Manhattan is preferred for donor ranking because it increases best-arm agreement with respect to Euclidean while preserving full coverage.
"""

OUTFILE.write_text(text, encoding="utf-8")
print(f"Markdown salvato in: {OUTFILE}")