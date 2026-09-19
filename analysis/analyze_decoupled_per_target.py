from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEC_DIR = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "analysis-transfer-decoupled-20260919"
)

PER_TARGET = DEC_DIR / "decoupled-ucb1-per-target.csv"
SUMMARY = DEC_DIR / "decoupled-ucb1-summary.csv"

FINAL_ROOT = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "final-20260913-analysis-01"
)

OUT = DEC_DIR / "presentation_exports"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load
# ============================================================

per_target = pd.read_csv(PER_TARGET)
summary = pd.read_csv(SUMMARY)


# ============================================================
# Selected configuration
# ============================================================

WR = 0.25
WE = 1.00

selected = per_target[
    np.isclose(
        per_target["reward_prior_weight"],
        WR,
    )
    & np.isclose(
        per_target["exploration_prior_weight"],
        WE,
    )
    ].copy()

selected.to_csv(
    OUT / "selected-decoupled-per-target.csv",
    index=False,
    )


# ============================================================
# Try to find ground-truth preference file
# ============================================================

preference_candidates = [
    FINAL_ROOT / "ground_truth" / "preferences-15.csv",
    FINAL_ROOT / "ground-truth" / "preferences-15.csv",
    FINAL_ROOT / "ground_truth" / "preferenze-15.csv",
    FINAL_ROOT / "ground-truth" / "preferenze-15.csv",
    ]

# Fallback recursive search
for p in sorted(FINAL_ROOT.rglob("*.csv")):
    name = p.name.lower()

    if (
            ("preference" in name or "preferenze" in name)
            and "15" in name
    ):
        preference_candidates.append(p)

preferences_path = next(
    (
        p
        for p in preference_candidates
        if p.exists()
    ),
    None,
)

preferences = None
function_col = None
preference_col = None
delta_col = None


def normalize_preference(value):
    s = str(value).strip().lower()

    if (
            "independent" in s
            or "neutral" in s
            or "indip" in s
    ):
        return "architecture-independent"

    if "arm" in s:
        return "arm64-preferred"

    if (
            "x86" in s
            or "amd" in s
    ):
        return "amd64-preferred"

    return s


if preferences_path:
    preferences = pd.read_csv(
        preferences_path
    )

    print(
        "\nGround truth trovato:",
        preferences_path,
    )

    print(
        "Colonne ground truth:",
        list(preferences.columns),
    )

    # Detect function column
    for candidate in [
        "function_name",
        "function",
        "target_function",
        "name",
    ]:
        if candidate in preferences.columns:
            function_col = candidate
            break

    # Detect preference column
    for candidate in [
        "preference",
        "architecture_preference",
        "label",
        "classification",
        "class",
    ]:
        if candidate in preferences.columns:
            preference_col = candidate
            break

    # More robust preference-column detection
    if preference_col is None:
        for col in preferences.columns:
            if preferences[col].dtype != object:
                continue

            values = (
                preferences[col]
                .dropna()
                .astype(str)
                .str.lower()
            )

            joined = " ".join(
                values.unique().tolist()
            )

            if (
                    ("arm" in joined)
                    and (
                    "x86" in joined
                    or "amd" in joined
            )
            ):
                preference_col = col
                break

    # Detect delta column
    for col in preferences.columns:
        lc = col.lower()

        if (
                "delta" in lc
                and pd.api.types.is_numeric_dtype(
            preferences[col]
        )
        ):
            delta_col = col
            break


# ============================================================
# Merge preference labels when available
# ============================================================

if (
        preferences is not None
        and function_col is not None
        and preference_col is not None
):
    cols = [
        function_col,
        preference_col,
    ]

    if delta_col:
        cols.append(delta_col)

    pref = preferences[
        cols
    ].copy()

    pref = pref.rename(
        columns={
            function_col: "target_function",
            preference_col: "architecture_preference",
        }
    )

    pref[
        "architecture_preference"
    ] = pref[
        "architecture_preference"
    ].map(
        normalize_preference
    )

    if delta_col:
        pref = pref.rename(
            columns={
                delta_col: "architecture_delta"
            }
        )

    selected = selected.merge(
        pref,
        on="target_function",
        how="left",
    )

else:
    print(
        "\nWARNING: impossibile identificare automaticamente "
        "la ground truth preference."
    )


selected.to_csv(
    OUT / "selected-decoupled-per-target-with-preference.csv",
    index=False,
    )


# ============================================================
# Horizon summary for selected configuration
# ============================================================

horizon_summary = (
    selected
    .groupby("horizon")
    .agg(
        targets=("target_function", "nunique"),
        mean_gain_pct=(
            "mean_latency_gain_pct",
            "mean",
        ),
        median_gain_pct=(
            "mean_latency_gain_pct",
            "median",
        ),
        positive_transfer_rate=(
            "mean_latency_gain_pct",
            lambda x: (
                    np.asarray(x) > 0
            ).mean(),
        ),
        negative_transfer_rate=(
            "mean_latency_gain_pct",
            lambda x: (
                    np.asarray(x) < 0
            ).mean(),
        ),
        mean_wrong_saved=(
            "mean_wrong_choices_saved",
            "mean",
        ),
        mean_regret=(
            "mean_reward_pseudo_regret",
            "mean",
        ),
    )
    .reset_index()
)

horizon_summary.to_csv(
    OUT / "selected-horizon-summary.csv",
    index=False,
    )

print(
    "\n===== SELECTED CONFIGURATION ====="
)
print(
    horizon_summary.to_string(
        index=False
    )
)


# ============================================================
# H10 per-target analysis
# ============================================================

h10 = selected[
    selected["horizon"] == 10
    ].copy()

h10 = h10.sort_values(
    "mean_latency_gain_pct",
    ascending=False,
)

h10.to_csv(
    OUT / "h10-per-target-ranked.csv",
    index=False,
    )


print(
    "\n===== TOP 15 H10 ====="
)

print(
    h10[
        [
            "target_function",
            "donor_function",
            "mean_latency_gain_pct",
            "mean_wrong_choices_saved",
            "first_arm_optimal_probability",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print(
    "\n===== BOTTOM 15 H10 ====="
)

print(
    h10[
        [
            "target_function",
            "donor_function",
            "mean_latency_gain_pct",
            "mean_wrong_choices_saved",
            "first_arm_optimal_probability",
        ]
    ]
    .tail(15)
    .to_string(index=False)
)


# ============================================================
# First-arm correctness groups
# ============================================================

h10[
    "first_arm_group"
] = np.where(
    h10[
        "first_arm_optimal_probability"
    ] >= 0.999,
    "donor direction correct",
    np.where(
        h10[
            "first_arm_optimal_probability"
        ] <= 0.001,
        "donor direction wrong",
        "mixed",
        ),
    )

direction_summary = (
    h10
    .groupby("first_arm_group")
    .agg(
        target_count=(
            "target_function",
            "nunique",
        ),
        mean_gain_pct=(
            "mean_latency_gain_pct",
            "mean",
        ),
        median_gain_pct=(
            "mean_latency_gain_pct",
            "median",
        ),
        positive_transfer_rate=(
            "mean_latency_gain_pct",
            lambda x: (
                    np.asarray(x) > 0
            ).mean(),
        ),
        mean_wrong_saved=(
            "mean_wrong_choices_saved",
            "mean",
        ),
    )
    .reset_index()
)

direction_summary.to_csv(
    OUT
    / "h10-gain-by-donor-direction.csv",
    index=False,
    )

print(
    "\n===== H10 BY DONOR DIRECTION ====="
)

print(
    direction_summary.to_string(
        index=False
    )
)


# ============================================================
# Preference-group analysis
# ============================================================

if (
        "architecture_preference"
        in selected.columns
):

    preference_summary = (
        selected
        .groupby(
            [
                "architecture_preference",
                "horizon",
            ]
        )
        .agg(
            target_count=(
                "target_function",
                "nunique",
            ),
            mean_gain_pct=(
                "mean_latency_gain_pct",
                "mean",
            ),
            median_gain_pct=(
                "mean_latency_gain_pct",
                "median",
            ),
            positive_transfer_rate=(
                "mean_latency_gain_pct",
                lambda x: (
                        np.asarray(x) > 0
                ).mean(),
            ),
            negative_transfer_rate=(
                "mean_latency_gain_pct",
                lambda x: (
                        np.asarray(x) < 0
                ).mean(),
            ),
            mean_wrong_saved=(
                "mean_wrong_choices_saved",
                "mean",
            ),
        )
        .reset_index()
    )

    preference_summary.to_csv(
        OUT
        / "gain-by-architecture-preference.csv",
        index=False,
        )

    print(
        "\n===== BY ARCHITECTURE PREFERENCE ====="
    )

    print(
        preference_summary[
            preference_summary[
                "horizon"
            ].isin(
                [5, 10, 20, 50]
            )
        ].to_string(
            index=False
        )
    )


# ============================================================
# Figure 1:
# macro gain by horizon
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

configs = [
    (
        0.25,
        0.25,
        "Coupled 0.25",
    ),
    (
        0.25,
        0.50,
        "Decoupled 0.25 / 0.50",
    ),
    (
        0.25,
        1.00,
        "Decoupled 0.25 / 1.00",
    ),
]

for wr, we, label in configs:
    subset = summary[
        np.isclose(
            summary[
                "reward_prior_weight"
            ],
            wr,
        )
        & np.isclose(
            summary[
                "exploration_prior_weight"
            ],
            we,
        )
        ].sort_values("horizon")

    ax.plot(
        subset["horizon"],
        subset[
            "macro_mean_latency_gain_pct"
        ],
        marker="o",
        label=label,
    )

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Horizon (requests)"
)

ax.set_ylabel(
    "Mean latency gain vs no-transfer (%)"
)

ax.set_title(
    "Transfer benefit over the request horizon"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT
    / "gain-vs-horizon.png",
    dpi=200,
    )

plt.close(fig)


# ============================================================
# Figure 2:
# heatmap wR x wE at H10
# ============================================================

h10_summary = summary[
    summary["horizon"] == 10
    ].copy()

matrix = (
    h10_summary
    .pivot(
        index="reward_prior_weight",
        columns="exploration_prior_weight",
        values="macro_mean_latency_gain_pct",
    )
    .sort_index()
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

image = ax.imshow(
    matrix.values,
    aspect="auto",
)

ax.set_xticks(
    np.arange(
        len(matrix.columns)
    )
)

ax.set_xticklabels(
    [
        str(x)
        for x in matrix.columns
    ]
)

ax.set_yticks(
    np.arange(
        len(matrix.index)
    )
)

ax.set_yticklabels(
    [
        str(x)
        for x in matrix.index
    ]
)

ax.set_xlabel(
    "Exploration prior weight (wE)"
)

ax.set_ylabel(
    "Reward prior weight (wR)"
)

ax.set_title(
    "Decoupled UCB1 gain at H=10"
)

for i in range(
        matrix.shape[0]
):
    for j in range(
            matrix.shape[1]
    ):
        ax.text(
            j,
            i,
            f"{matrix.iloc[i, j]:+.2f}%",
            ha="center",
            va="center",
        )

fig.colorbar(
    image,
    ax=ax,
    label="Mean latency gain (%)",
)

fig.tight_layout()

fig.savefig(
    OUT
    / "decoupled-grid-h10.png",
    dpi=200,
    )

plt.close(fig)


# ============================================================
# Figure 3:
# ranked per-target gains H10
# ============================================================

ranked = h10.sort_values(
    "mean_latency_gain_pct",
    ascending=True,
)

fig_height = max(
    8,
    len(ranked) * 0.28,
    )

fig, ax = plt.subplots(
    figsize=(
        9,
        fig_height,
    )
)

ax.barh(
    ranked[
        "target_function"
    ],
    ranked[
        "mean_latency_gain_pct"
    ],
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_xlabel(
    "Mean latency gain vs no-transfer (%)"
)

ax.set_ylabel(
    "Target function"
)

ax.set_title(
    "Per-target transfer gain at H=10\n"
    "wR=0.25, wE=1.0"
)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT
    / "per-target-gain-h10.png",
    dpi=200,
    )

plt.close(fig)


# ============================================================
# Figure 4:
# architecture preference H10
# ============================================================

if (
        "architecture_preference"
        in h10.columns
):

    pref_h10 = (
        h10
        .groupby(
            "architecture_preference"
        )[
            "mean_latency_gain_pct"
        ]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        pref_h10.index,
        pref_h10.values,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "Mean latency gain vs no-transfer (%)"
    )

    ax.set_title(
        "Transfer gain by architecture preference\n"
        "H=10, wR=0.25, wE=1.0"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / "gain-by-preference-h10.png",
        dpi=200,
        )

    plt.close(fig)


# ============================================================
# Figure 5:
# negative transfer rate
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    horizon_summary[
        "horizon"
    ],
    horizon_summary[
        "negative_transfer_rate"
    ] * 100.0,
    marker="o",
    )

ax.set_xlabel(
    "Horizon (requests)"
)

ax.set_ylabel(
    "Targets with negative transfer (%)"
)

ax.set_title(
    "Negative-transfer rate\n"
    "wR=0.25, wE=1.0"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT
    / "negative-transfer-rate.png",
    dpi=200,
    )

plt.close(fig)


# ============================================================
# Final
# ============================================================

print(
    "\nOutput scritto in:"
)

print(OUT)

for p in sorted(
        OUT.iterdir()
):
    print(
        " -",
        p.name,
    )