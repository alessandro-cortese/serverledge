from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports/donor_quality_full.csv")
OUTDIR = Path("data/profiling/analysis-transfer-donor-metrics-20260918/presentation_exports")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Per ogni clusterer prendiamo la riga con arm_match massimo
best_rows = (
    df.sort_values(["clusterer", "arm_match"], ascending=[True, False])
    .groupby("clusterer", as_index=False)
    .first()
)

labels = best_rows["clusterer"].tolist()
coverage = best_rows["coverage"].tolist()
arm_match = best_rows["arm_match"].tolist()
distance = best_rows["distance"].tolist()

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, coverage, width, label="Coverage")
plt.bar(x + width/2, arm_match, width, label="Best-arm agreement")

plt.xticks(x, [f"{lab}\n(best: {dist})" for lab, dist in zip(labels, distance)])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Best donor-selection configuration per clusterer")
plt.legend()
plt.grid(axis="y", alpha=0.3)

for i, v in enumerate(coverage):
    plt.text(i - width/2, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(arm_match):
    plt.text(i + width/2, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
outfile = OUTDIR / "clusterer_comparison_best_config.png"
plt.savefig(outfile, dpi=200)
print(f"Grafico salvato in: {outfile}")