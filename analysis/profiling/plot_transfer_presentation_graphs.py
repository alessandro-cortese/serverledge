#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("data/profiling/gcp-transfer-final/analysis/presentation_graphs")
OUT.mkdir(parents=True, exist_ok=True)

# 1) Architecture choices: aggregate R02+R03, 50 target requests each.
labels = ["No transfer\nc = 0.8", "Transfer\nc = 0", "Transfer\nc = 0.8"]
amd = np.array([6, 100, 6], dtype=float)
arm = np.array([94, 0, 94], dtype=float)

fig, ax = plt.subplots(figsize=(9.2, 5.4))
x = np.arange(len(labels))
ax.bar(x, amd, label="amd64")
ax.bar(x, arm, bottom=amd, label="arm64")
ax.set_title("Effetto del transfer e dell'esplorazione su randomaccess")
ax.set_ylabel("Scelte architetturali (%)")
ax.set_xticks(x, labels)
ax.set_ylim(0, 100)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.25)

for i, (a, r) in enumerate(zip(amd, arm)):
    if a > 0:
        ax.text(i, a / 2, f"{a:.0f}% AMD", ha="center", va="center", fontsize=10)
    if r > 0:
        ax.text(i, a + r / 2, f"{r:.0f}% ARM", ha="center", va="center", fontsize=10)

fig.text(
    0.5, 0.01,
    "Aggregazione R02+R03, 100 richieste per configurazione. "
    "Il prior deriva dal donor compression.",
    ha="center", fontsize=9
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "transfer_exploration_architecture_choices.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "transfer_exploration_architecture_choices.svg", bbox_inches="tight")
plt.close(fig)

# 2) Donor -> weak-prior gap preservation.
replicas = ["R02", "R03"]
donor_gap = np.array([0.037145464, 0.046355027])
prior_gap = np.array([0.037145572, 0.046355046])

x = np.arange(len(replicas))
width = 0.36

fig, ax = plt.subplots(figsize=(8.6, 5.2))
bars1 = ax.bar(x - width/2, donor_gap, width, label="Gap reward donor")
bars2 = ax.bar(x + width/2, prior_gap, width, label="Gap reward weak prior")
ax.set_title("Preservazione della preferenza del donor nel weak prior")
ax.set_ylabel("Reward gap (amd64 − arm64)")
ax.set_xticks(x, replicas)
ax.legend()
ax.grid(axis="y", alpha=0.25)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.6f}",
                ha="center", va="bottom", fontsize=9)

fig.text(
    0.5, 0.01,
    "Un gap positivo indica che il donor/prior favorisce amd64. "
    "La quasi identità dei valori mostra che il transfer preserva l'informazione del donor.",
    ha="center", fontsize=9
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / "donor_prior_gap_preservation.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "donor_prior_gap_preservation.svg", bbox_inches="tight")
plt.close(fig)

print(f"PASS — presentation graphs generated in {OUT}")
