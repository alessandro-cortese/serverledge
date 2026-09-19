from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports/donor_quality_kmeans_only.csv")
OUTDIR = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Teniamo solo euclidean e manhattan
df = df[df["distance"].str.lower().isin(["euclidean", "manhattan"])].copy()

if df.empty:
    raise SystemExit("Nessun dato KMeans Euclidean/Manhattan trovato.")

df["distance"] = df["distance"].str.lower()
df = df.sort_values("distance")

labels = df["distance"].tolist()
coverage = df["coverage"].tolist()
arm_match = df["arm_match"].tolist()

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, coverage, width, label="Coverage")
plt.bar(x + width/2, arm_match, width, label="Best-arm agreement")

plt.xticks(x, labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("KMeans donor selection: Manhattan vs Euclidean")
plt.legend()
plt.grid(axis="y", alpha=0.3)

for i, v in enumerate(coverage):
    plt.text(i - width/2, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(arm_match):
    plt.text(i + width/2, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
outfile = OUTDIR / "kmeans_manhattan_vs_euclidean.png"
plt.savefig(outfile, dpi=200)
print(f"Grafico salvato in: {outfile}")