from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEC_DIR = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "analysis-transfer-decoupled-20260919"
)

INPUT = (
        DEC_DIR
        / "presentation_exports"
        / "selected-decoupled-per-target-with-preference.csv"
)

PREFERENCES = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "final-20260913-analysis-01"
        / "ground_truth"
        / "preferences-15.csv"
)

OUT_DIR = (
        REPO_ROOT
        / "data"
        / "profiling"
        / "analysis-transfer-final-evidence"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


dec = pd.read_csv(INPUT)
pref = pd.read_csv(PREFERENCES)

# Usiamo H=10 solo per recuperare donor e indicatori.
# Il gain NON entra nei criteri di selezione.
h10 = dec[dec["horizon"] == 10].copy()

ground_truth = pref[
    [
        "function_name",
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
        "architecture_preference",
    ]
].rename(
    columns={
        "function_name": "target_function",
        "arm_vs_x86_delta_percent": "architecture_delta_percent",
    }
)

df = h10.merge(
    ground_truth,
    on="target_function",
    how="left",
    suffixes=("", "_gt"),
)

# Se la preference era già presente nel file decoupled,
# manteniamo quella della ground truth originale.
if "architecture_preference_gt" in df.columns:
    df["architecture_preference"] = df[
        "architecture_preference_gt"
    ]

df["abs_architecture_delta_percent"] = (
    df["architecture_delta_percent"].abs()
)

df["donor_direction_correct"] = np.isclose(
    df["first_arm_optimal_probability"],
    1.0,
)

df["excluded_special"] = (
        df["target_function"].isin(
            ["amd_faster", "arm_faster"]
        )
        | df["target_function"].str.startswith("twin-")
)

eligible = df[~df["excluded_special"]].copy()


def category(row):
    pref = row["architecture_preference"]
    delta = row["abs_architecture_delta_percent"]
    correct = row["donor_direction_correct"]

    if (
            pref == "arm64-preferred"
            and delta >= 15
            and correct
    ):
        return "arm64-preferred-aligned"

    if (
            pref == "amd64-preferred"
            and delta >= 15
            and correct
    ):
        return "amd64-preferred-aligned"

    if (
            pref == "architecture-independent"
            and delta <= 5
    ):
        return "neutral-control"

    if (
            delta >= 15
            and not correct
    ):
        return "robustness-wrong-donor"

    return "other"


eligible["validation_category"] = eligible.apply(
    category,
    axis=1,
)

cols = [
    "validation_category",
    "target_function",
    "donor_function",
    "architecture_preference",
    "architecture_delta_percent",
    "abs_architecture_delta_percent",
    "x86_duration_ms",
    "arm_duration_ms",
    "donor_direction_correct",
    # Manteniamo il gain come informazione descrittiva,
    # NON come criterio di selezione.
    "mean_latency_gain_pct",
    "mean_wrong_choices_saved",
]

eligible[cols].to_csv(
    OUT_DIR / "gcp-target-candidates.csv",
    index=False,
    )

shortlist = eligible[
    eligible["validation_category"] != "other"
    ].copy()

shortlist = shortlist.sort_values(
    [
        "validation_category",
        "abs_architecture_delta_percent",
    ],
    ascending=[True, False],
)

shortlist[cols].to_csv(
    OUT_DIR / "gcp-target-shortlist.csv",
    index=False,
    )

print("\n===== GCP TARGET SHORTLIST =====")
print(
    shortlist[cols].to_string(
        index=False
    )
)

print("\nIMPORTANT:")
print(
    "mean_latency_gain_pct is descriptive only and "
    "was NOT used for eligibility or ranking."
)