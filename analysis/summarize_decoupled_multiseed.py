from pathlib import Path

import json
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_ROOT = REPO_ROOT / "data" / "profiling"

SEEDS = [11, 23, 42, 77, 101]

OUT = DATA_ROOT / "analysis-transfer-final-evidence"
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


rows = []

for seed in SEEDS:
    run_dir = DATA_ROOT / f"analysis-transfer-decoupled-seed-{seed}"
    summary_path = run_dir / "decoupled-ucb1-summary.csv"

    if not summary_path.exists():
        raise SystemExit(f"Missing: {summary_path}")

    df = pd.read_csv(summary_path)

    h10 = df[df["horizon"] == 10].copy()

    coupled = h10[
        (h10["reward_prior_weight"] == 0.25)
        & (h10["exploration_prior_weight"] == 0.25)
        ].iloc[0]

    decoupled = h10[
        (h10["reward_prior_weight"] == 0.25)
        & (h10["exploration_prior_weight"] == 1.00)
        ].iloc[0]

    rows.append(
        {
            "seed": seed,

            "coupled_gain_pct":
                coupled["macro_mean_latency_gain_pct"],

            "coupled_ci95_low":
                coupled["macro_latency_gain_ci95_low"],

            "coupled_ci95_high":
                coupled["macro_latency_gain_ci95_high"],

            "coupled_wrong_saved":
                coupled["macro_mean_wrong_choices_saved"],

            "decoupled_gain_pct":
                decoupled["macro_mean_latency_gain_pct"],

            "decoupled_ci95_low":
                decoupled["macro_latency_gain_ci95_low"],

            "decoupled_ci95_high":
                decoupled["macro_latency_gain_ci95_high"],

            "decoupled_wrong_saved":
                decoupled["macro_mean_wrong_choices_saved"],

            "delta_gain_pp":
                decoupled["macro_mean_latency_gain_pct"]
                - coupled["macro_mean_latency_gain_pct"],

            "delta_wrong_saved":
                decoupled["macro_mean_wrong_choices_saved"]
                - coupled["macro_mean_wrong_choices_saved"],
        }
    )


runs = pd.DataFrame(rows)

runs.to_csv(
    OUT / "decoupled-multiseed-per-seed.csv",
    index=False,
    )


summary = pd.DataFrame(
    [
        {
            "configuration": "coupled w=0.25",
            "mean_gain_pct": runs["coupled_gain_pct"].mean(),
            "std_across_seeds": runs["coupled_gain_pct"].std(ddof=1),
            "min_gain_pct": runs["coupled_gain_pct"].min(),
            "max_gain_pct": runs["coupled_gain_pct"].max(),
            "mean_wrong_saved": runs["coupled_wrong_saved"].mean(),
        },
        {
            "configuration": "decoupled wR=0.25 wE=1.0",
            "mean_gain_pct": runs["decoupled_gain_pct"].mean(),
            "std_across_seeds": runs["decoupled_gain_pct"].std(ddof=1),
            "min_gain_pct": runs["decoupled_gain_pct"].min(),
            "max_gain_pct": runs["decoupled_gain_pct"].max(),
            "mean_wrong_saved": runs["decoupled_wrong_saved"].mean(),
        },
        {
            "configuration": "decoupled minus coupled",
            "mean_gain_pct": runs["delta_gain_pp"].mean(),
            "std_across_seeds": runs["delta_gain_pp"].std(ddof=1),
            "min_gain_pct": runs["delta_gain_pp"].min(),
            "max_gain_pct": runs["delta_gain_pp"].max(),
            "mean_wrong_saved": runs["delta_wrong_saved"].mean(),
        },
    ]
)

summary.to_csv(
    OUT / "decoupled-multiseed-summary.csv",
    index=False,
    )


# ------------------------------------------------------------
# Figure: gain across seeds
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    runs["seed"],
    runs["coupled_gain_pct"],
    marker="o",
    label="Coupled w=0.25",
)

ax.plot(
    runs["seed"],
    runs["decoupled_gain_pct"],
    marker="o",
    label="Decoupled wR=0.25, wE=1.0",
)

ax.set_xlabel("Random seed")
ax.set_ylabel("Mean latency gain at H=10 (%)")
ax.set_title("Multi-seed stability of transfer configurations")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    FIG / "multiseed-gain-h10.png",
    dpi=200,
    )

plt.close(fig)


# ------------------------------------------------------------
# Figure: delta decoupled - coupled
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    runs["seed"].astype(str),
    runs["delta_gain_pp"],
)

ax.axhline(0, linewidth=1)

ax.set_xlabel("Random seed")
ax.set_ylabel("Gain difference (percentage points)")
ax.set_title(
    "Decoupled improvement over coupled at H=10"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG / "multiseed-decoupled-minus-coupled.png",
    dpi=200,
    )

plt.close(fig)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

manifest = {
    "analysis": "decoupled_ucb1_multiseed_validation",
    "seeds": SEEDS,
    "horizon": 10,
    "c": 0.8,
    "coupled": {
        "reward_prior_weight": 0.25,
        "exploration_prior_weight": 0.25,
    },
    "decoupled": {
        "reward_prior_weight": 0.25,
        "exploration_prior_weight": 1.0,
    },
    "clusterer": "kmeans",
    "donor_ranker": "manhattan",
    "prior_mode": "reference-anchored",
    "reward": "-ln(duration_ms)",
    "replicates_per_seed": 500,
    "purpose": (
        "Validate that the selected decoupled configuration "
        "is stable across random seeds before online GCP evaluation."
    ),
}

with (
        OUT / "multiseed-manifest.json"
).open("w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print("\n===== PER SEED =====")
print(runs.to_string(index=False))

print("\n===== SUMMARY =====")
print(summary.to_string(index=False))

print("\nOutput:")
print(OUT)