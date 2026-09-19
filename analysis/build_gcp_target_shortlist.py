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


def normalize_preference(value: str) -> str:
    value = str(value).strip().lower()

    if value in {
        "arm-preferred",
        "arm64-preferred",
        "arm",
        "arm64",
    }:
        return "arm64-preferred"

    if value in {
        "x86-preferred",
        "amd64-preferred",
        "x86",
        "amd64",
    }:
        return "amd64-preferred"

    if value in {
        "architecture-independent",
        "independent",
        "neutral",
    }:
        return "architecture-independent"

    return value


def is_special_function(name: str) -> bool:
    name = str(name)

    if name in {
        "amd_faster",
        "arm_faster",
    }:
        return True

    if name.startswith("twin-"):
        return True

    return False


dec = pd.read_csv(INPUT)
pref = pd.read_csv(PREFERENCES)

# H=10 è usato come punto di riferimento per le informazioni
# MAB descrittive. Il gain NON viene usato per selezionare i target.
h10 = dec[
    dec["horizon"] == 10
    ].copy()

ground_truth = pref[
    [
        "function_name",
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
        "architecture_preference",
    ]
].copy()

ground_truth = ground_truth.rename(
    columns={
        "function_name": "target_function",
        "arm_vs_x86_delta_percent":
            "architecture_delta_percent",
        "architecture_preference":
            "architecture_preference_ground_truth",
    }
)

ground_truth[
    "architecture_preference_ground_truth"
] = ground_truth[
    "architecture_preference_ground_truth"
].map(normalize_preference)

df = h10.merge(
    ground_truth,
    on="target_function",
    how="left",
)

# Ground truth canonica
df["architecture_preference"] = df[
    "architecture_preference_ground_truth"
]

df["abs_architecture_delta_percent"] = (
    df["architecture_delta_percent"].abs()
)

df["donor_direction_correct"] = np.isclose(
    df["first_arm_optimal_probability"],
    1.0,
)

df["target_is_special"] = df[
    "target_function"
].map(is_special_function)

df["donor_is_special"] = df[
    "donor_function"
].map(is_special_function)

df["eligible_normal_pair"] = (
        ~df["target_is_special"]
        & ~df["donor_is_special"]
)


def category(row):
    if not row["eligible_normal_pair"]:
        return "excluded-special"

    pref = row["architecture_preference"]
    delta = row["abs_architecture_delta_percent"]
    correct = row["donor_direction_correct"]

    # Positive-transfer validation:
    # architecture-sensitive + donor direction aligned.
    if (
            pref == "arm64-preferred"
            and delta >= 15.0
            and correct
    ):
        return "arm64-preferred-aligned"

    if (
            pref == "amd64-preferred"
            and delta >= 15.0
            and correct
    ):
        return "amd64-preferred-aligned"

    # Neutral control:
    # deliberately no meaningful architecture preference.
    if (
            pref == "architecture-independent"
            and delta <= 5.0
    ):
        return "neutral-control"

    # Robustness case:
    # strong architecture preference but wrong donor direction.
    if (
            delta >= 15.0
            and not correct
    ):
        return "robustness-wrong-donor"

    return "other"


df["validation_category"] = df.apply(
    category,
    axis=1,
)


def selection_reason(row):
    category = row["validation_category"]

    if category == "arm64-preferred-aligned":
        return (
            "ARM64-sensitive target; |architecture delta| >= 15%; "
            "same-cluster Manhattan donor predicts the correct best arm."
        )

    if category == "amd64-preferred-aligned":
        return (
            "AMD64-sensitive target; |architecture delta| >= 15%; "
            "same-cluster Manhattan donor predicts the correct best arm."
        )

    if category == "neutral-control":
        return (
            "Architecture-independent target with |architecture delta| <= 5%; "
            "used as neutral/control case."
        )

    if category == "robustness-wrong-donor":
        return (
            "Architecture-sensitive target with |architecture delta| >= 15%, "
            "but donor predicts the wrong initial best arm; robustness case."
        )

    if category == "excluded-special":
        return (
            "Excluded because target or donor is a sentinel/twin benchmark."
        )

    return "Does not satisfy a pre-registered validation category."


df["selection_reason"] = df.apply(
    selection_reason,
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
    "target_is_special",
    "donor_is_special",
    "mean_latency_gain_pct",
    "mean_wrong_choices_saved",
    "selection_reason",
]


# Tutto il dataset classificato, utile per audit.
df[cols].to_csv(
    OUT_DIR / "gcp-target-candidates-all.csv",
    index=False,
    )


# Solo categorie candidabili.
shortlist = df[
    df["validation_category"].isin(
        [
            "arm64-preferred-aligned",
            "amd64-preferred-aligned",
            "neutral-control",
            "robustness-wrong-donor",
        ]
    )
].copy()


# IMPORTANTE:
# non ordiniamo in base al gain offline.
#
# Per target architecture-sensitive:
# maggiore |delta| = caso architetturalmente più netto.
#
# Per neutral:
# minore |delta| = controllo più neutrale.
shortlist["selection_score"] = np.where(
    shortlist["validation_category"]
    == "neutral-control",
    -shortlist["abs_architecture_delta_percent"],
    shortlist["abs_architecture_delta_percent"],
    )

category_order = {
    "arm64-preferred-aligned": 0,
    "amd64-preferred-aligned": 1,
    "neutral-control": 2,
    "robustness-wrong-donor": 3,
}

shortlist["category_order"] = shortlist[
    "validation_category"
].map(category_order)

shortlist = shortlist.sort_values(
    [
        "category_order",
        "selection_score",
        "target_function",
    ],
    ascending=[
        True,
        False,
        True,
    ],
)

shortlist[cols].to_csv(
    OUT_DIR / "gcp-target-shortlist.csv",
    index=False,
    )


# Compact top candidates.
# Anche qui il gain NON entra nella scelta.
selected_rows = []

for category in [
    "arm64-preferred-aligned",
    "amd64-preferred-aligned",
    "neutral-control",
    "robustness-wrong-donor",
]:
    subset = shortlist[
        shortlist["validation_category"]
        == category
        ]

    # Salviamo al massimo i primi 3 per categoria.
    selected_rows.append(
        subset.head(3)
    )

compact = pd.concat(
    selected_rows,
    ignore_index=True,
)

compact[cols].to_csv(
    OUT_DIR / "gcp-target-shortlist-compact.csv",
    index=False,
    )


print(
    "\n===== FULL GCP TARGET SHORTLIST ====="
)

print(
    shortlist[cols].to_string(
        index=False
    )
)

print(
    "\n===== COMPACT PRE-GCP CANDIDATES ====="
)

print(
    compact[cols].to_string(
        index=False
    )
)

print(
    "\nIMPORTANT:"
)

print(
    "mean_latency_gain_pct is descriptive only. "
    "It was NOT used for eligibility or ranking."
)

print(
    "Sentinel/twin targets and sentinel/twin donors "
    "are excluded from final validation candidates."
)