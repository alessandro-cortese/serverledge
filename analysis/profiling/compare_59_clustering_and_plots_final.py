#!/usr/bin/env python3
"""Final controlled OLD -> NEW clustering/donor comparison for Serverledge.

Methodological goals
--------------------
* Same eligible corpus for every configuration.
* LOFO: target excluded before scaler fit, KMeans fit and donor selection.
* Ground-truth labels are never features.
* Primary ground-truth threshold: 2.5%; 15% is sensitivity only.
* Final KMeans geometry: 5 dynamic MinMax coordinates / sqrt(5),
  2 static MinMax coordinates / sqrt(2), K=6, n_init=50.
* Final donor geometry: PLAIN Manhattan on the seven MinMax coordinates.
* Final donor rule: inverse-distance directional vote 1/(d+1e-9)^2,
  then nearest donor in the predicted directional class.
* Synthetic functions amd_faster / arm_faster are not donor candidates.

Architecture-independent targets
--------------------------------
They have no meaningful directional classification error. Therefore:
* directional agreement excludes them from its denominator;
* they remain in continuous donor-effect errors (delta / reward-gap);
* their induced arm-choice regret is reported separately;
* raw best-arm agreement is diagnostic only, because a tiny timing delta can
  flip the raw winner while the function is still inside the neutral band.

Figures are descriptive full-corpus fits. LOFO metrics are the inferential /
validation quantities and must not be replaced by full-fit plot statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler

PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]
STATIC2 = ["static_token_count_mean", "static_function_count"]
FEATURE7 = PAPER5 + STATIC2

SEEDS = (11, 23, 37, 41, 53)
PRIMARY_THRESHOLD = 2.5
SENSITIVITY_THRESHOLD = 15.0
PREF_ORDER = ("x86-preferred", "architecture-independent", "arm-preferred")
DIRECTIONAL = ("x86-preferred", "arm-preferred")
SYNTHETIC_DONORS = {"amd_faster", "arm_faster"}
EPSILON = 1e-9

# Controlled decomposition:
# A -> B isolates representation/clustering, keeping nearest-donor rule.
# B -> C isolates final donor rule, keeping clustering fixed.
CONFIGS = {
    "A_legacy_K5": {"space": "legacy", "k": 5, "donor_rule": "nearest"},
    "B_hybrid_K6_nearest": {"space": "hybrid", "k": 6, "donor_rule": "nearest"},
    "C_final_plain": {"space": "hybrid", "k": 6, "donor_rule": "weighted_vote_p2"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_unique(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df["function_name"].duplicated().any():
        raise SystemExit(f"{path}: duplicate function_name")
    return df


def prepare(root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = {
        "profiles": root / "resource/x86/function-profiles-median.csv",
        "static": root / "static-code-metrics.csv",
        "language": root / "function-languages.csv",
        "pref_2p5": root / "ground_truth/preferences-2p5.csv",
        "pref_15": root / "ground_truth/preferences-15.csv",
    }

    profiles = load_unique(paths["profiles"], ["function_name", *PAPER5])
    static = load_unique(
        paths["static"],
        ["function_name", "static_source_available", *STATIC2],
    )
    languages = load_unique(paths["language"], ["function_name", "language"])
    pref25 = load_unique(
        paths["pref_2p5"],
        [
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
            "arm_vs_x86_delta_percent",
        ],
    )
    pref15 = load_unique(
        paths["pref_15"],
        ["function_name", "architecture_preference"],
    )

    df = (
        profiles[["function_name", *PAPER5]]
        .merge(
            static[["function_name", "static_source_available", *STATIC2]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            languages[["function_name", "language"]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            pref25[
                [
                    "function_name",
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                    "arm_vs_x86_delta_percent",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            pref15[["function_name", "architecture_preference"]].rename(
                columns={"architecture_preference": "preference_15"}
            ),
            on="function_name",
            validate="one_to_one",
        )
    )

    ok = (
        df["static_source_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    numeric = [
        *FEATURE7,
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        ok &= np.isfinite(df[column].to_numpy(float))

    df = df.loc[ok].copy().sort_values("function_name").reset_index(drop=True)

    if len(df) != 59:
        raise SystemExit(
            f"Expected exactly 59 static-eligible functions, got {len(df)}"
        )
    if not (df[["x86_duration_ms", "arm_duration_ms"]] > 0).all().all():
        raise SystemExit("Non-positive duration found")
    for field in ("architecture_preference", "preference_15"):
        unexpected = sorted(set(df[field]) - set(PREF_ORDER))
        if unexpected:
            raise SystemExit(f"Unexpected labels in {field}: {unexpected}")

    return df, paths


def fit_spaces(
        train: pd.DataFrame,
        target: pd.DataFrame,
        space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train_plain, target_plain, train_cluster, target_cluster."""
    if space == "legacy":
        columns = PAPER5
    elif space == "hybrid":
        columns = FEATURE7
    else:
        raise ValueError(space)

    scaler = MinMaxScaler()
    train_plain = scaler.fit_transform(train[columns].to_numpy(float))
    target_plain = scaler.transform(target[columns].to_numpy(float))

    train_cluster = train_plain.copy()
    target_cluster = target_plain.copy()

    if space == "hybrid":
        # EXACT frozen KMeans block geometry.
        train_cluster[:, : len(PAPER5)] /= math.sqrt(len(PAPER5))
        train_cluster[:, len(PAPER5) :] /= math.sqrt(len(STATIC2))
        target_cluster[:, : len(PAPER5)] /= math.sqrt(len(PAPER5))
        target_cluster[:, len(PAPER5) :] /= math.sqrt(len(STATIC2))

    return train_plain, target_plain, train_cluster, target_cluster


def raw_best_label(row: pd.Series) -> str:
    return (
        "x86-preferred"
        if float(row["x86_duration_ms"]) <= float(row["arm_duration_ms"])
        else "arm-preferred"
    )


def reward_gap(x86_ms: float, arm_ms: float) -> float:
    # r = -ln(ms), therefore r_ARM - r_x86 = ln(x86 / arm).
    return float(np.log(x86_ms / arm_ms))


def choose_donor(
        train: pd.DataFrame,
        labels: np.ndarray,
        target_cluster: int,
        train_plain: np.ndarray,
        target_plain: np.ndarray,
        donor_rule: str,
) -> tuple[int | None, str, str, float | None]:
    same = np.flatnonzero(labels == target_cluster)
    if not len(same):
        return None, "", "no_cluster_member", None

    names = train["function_name"].astype(str).to_numpy()
    prefs = train["architecture_preference"].astype(str).to_numpy()
    distances = np.abs(train_plain - target_plain[0]).sum(axis=1)

    # Hold donor eligibility constant across A/B/C so A->B and B->C do not
    # accidentally measure the synthetic-donor policy change.
    same = same[~np.isin(names[same], list(SYNTHETIC_DONORS))]
    if not len(same):
        return None, "", "no_admissible_cluster_member", None

    if donor_rule == "nearest":
        pool = same
        predicted = ""
    elif donor_rule == "weighted_vote_p2":
        directional = same[np.isin(prefs[same], DIRECTIONAL)]
        if not len(directional):
            return None, "", "no_directional_member", None

        votes = {}
        for direction in DIRECTIONAL:
            members = directional[prefs[directional] == direction]
            votes[direction] = float(
                np.sum(1.0 / np.power(distances[members] + EPSILON, 2))
            ) if len(members) else 0.0

        if math.isclose(
                votes[DIRECTIONAL[0]],
                votes[DIRECTIONAL[1]],
                rel_tol=0.0,
                abs_tol=1e-15,
        ):
            return None, "", "tied_vote", None

        predicted = max(DIRECTIONAL, key=lambda d: votes[d])
        pool = directional[prefs[directional] == predicted]
    else:
        raise ValueError(donor_rule)

    order = np.lexsort((names[pool], distances[pool]))
    idx = int(pool[order[0]])

    if donor_rule == "nearest":
        # Legacy nearest donor has no explicit class-vote stage. For arm-choice
        # diagnostics we use the raw faster arm of the selected donor.
        predicted = raw_best_label(train.iloc[idx])

    return idx, predicted, "selected", float(distances[idx])


def pair_outcome(
        target: pd.Series,
        donor: pd.Series,
        predicted: str,
        threshold: float = PRIMARY_THRESHOLD,
) -> dict:
    tx = float(target["x86_duration_ms"])
    ta = float(target["arm_duration_ms"])
    dx = float(donor["x86_duration_ms"])
    da = float(donor["arm_duration_ms"])
    td = float(target["arm_vs_x86_delta_percent"])
    dd = float(donor["arm_vs_x86_delta_percent"])

    target_label = str(target["architecture_preference"])
    donor_label = str(donor["architecture_preference"])
    target_directional = target_label in DIRECTIONAL
    target_independent = target_label == "architecture-independent"

    # Compatibility semantics for directional targets:
    # - same directional donor => compatible
    # - architecture-independent donor => compatible / non-conflicting
    # - opposite directional donor => conflict
    # For architecture-independent targets there is no directional requirement,
    # so these classification-style fields are undefined and continuous/regret
    # diagnostics are used instead.
    if target_directional:
        strict_donor_direction_match = float(donor_label == target_label)
        donor_independent_on_directional_target = float(
            donor_label == "architecture-independent"
        )
        directional_conflict = float(
            donor_label in DIRECTIONAL and donor_label != target_label
        )
        directional_compatibility = 1.0 - directional_conflict
    else:
        strict_donor_direction_match = np.nan
        donor_independent_on_directional_target = np.nan
        directional_conflict = np.nan
        directional_compatibility = np.nan

    chosen = tx if predicted == "x86-preferred" else ta
    alternative = ta if predicted == "x86-preferred" else tx
    best = min(tx, ta)

    return {
        "target_is_directional": target_directional,
        "target_is_independent": target_independent,
        "donor_preference": donor_label,
        # Diagnostic only: compares the raw faster arm, even inside the neutral band.
        "raw_best_arm_agreement": float(raw_best_label(donor) == raw_best_label(target)),
        # Routing/prediction agreement: kept for continuity with previous analyses.
        # For legacy nearest-donor cases `predicted` is the donor raw faster arm.
        "predicted_direction_agreement": (
            float(predicted == target_label) if target_directional else np.nan
        ),
        # Donor-label semantics requested for the final report.
        # An architecture-independent donor is compatible with either directional
        # target because it does not encode the opposite architectural preference.
        "strict_donor_direction_match": strict_donor_direction_match,
        "donor_independent_on_directional_target": donor_independent_on_directional_target,
        "directional_conflict": directional_conflict,
        "directional_compatibility": directional_compatibility,
        # Continuous effect errors remain meaningful for ALL target classes.
        "target_delta_percent": td,
        "donor_delta_percent": dd,
        "abs_delta_error_pp": abs(td - dd),
        "target_reward_gap": reward_gap(tx, ta),
        "donor_reward_gap": reward_gap(dx, da),
        "abs_reward_gap_error": abs(reward_gap(tx, ta) - reward_gap(dx, da)),
        "multiplicative_mismatch": math.exp(
            abs(math.log(ta / tx) - math.log(da / dx))
        ),
        "regret_percent": (chosen / best - 1.0) * 100.0,
        "realized_speedup": alternative / chosen,
        # On independent targets this measures how far the selected donor's
        # effect sits outside the neutral ground-truth band. It is NOT a
        # classification error and is reported only as a magnitude diagnostic.
        "neutral_band_excess_pp": max(0.0, abs(dd) - threshold),
        "same_language": float(str(donor["language"]) == str(target["language"])),
    }


def lofo(df: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict] = []

    for seed in seeds:
        print(f"LOFO seed {seed}...", flush=True)
        for target_idx in range(len(df)):
            target = df.iloc[[target_idx]].copy()
            train = df.drop(index=target_idx).reset_index(drop=True)

            cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

            for config_name, cfg in CONFIGS.items():
                key = (cfg["space"], cfg["k"])
                if key not in cache:
                    x_plain, xt_plain, x_cluster, xt_cluster = fit_spaces(
                        train, target, cfg["space"]
                    )
                    model = KMeans(
                        n_clusters=cfg["k"],
                        n_init=50,
                        random_state=seed,
                    )
                    labels = model.fit_predict(x_cluster)
                    target_cluster = int(model.predict(xt_cluster)[0])
                    cache[key] = x_plain, xt_plain, labels, target_cluster

                x_plain, xt_plain, labels, target_cluster = cache[key]
                donor_idx, predicted, status, selected_distance = choose_donor(
                    train,
                    labels,
                    target_cluster,
                    x_plain,
                    xt_plain,
                    cfg["donor_rule"],
                )

                base = {
                    "config": config_name,
                    "seed": seed,
                    "target_function": str(target.iloc[0]["function_name"]),
                    "target_preference": str(target.iloc[0]["architecture_preference"]),
                    "target_language": str(target.iloc[0]["language"]),
                    "target_is_synthetic": str(target.iloc[0]["function_name"]) in SYNTHETIC_DONORS,
                    "status": status,
                    "prediction": predicted,
                    "donor_function": "",
                    "selected_distance": selected_distance,
                }

                if donor_idx is not None:
                    donor = train.iloc[donor_idx]
                    base["donor_function"] = str(donor["function_name"])
                    base.update(pair_outcome(target.iloc[0], donor, predicted))

                rows.append(base)

    result = pd.DataFrame(rows)
    expected = len(df) * len(seeds) * len(CONFIGS)
    if len(result) != expected:
        raise AssertionError(f"Unexpected LOFO row count {len(result)} != {expected}")
    if result.duplicated(["config", "seed", "target_function"]).any():
        raise AssertionError("Duplicate LOFO key")
    leaked = (
            (result["status"] == "selected")
            & (result["donor_function"] == result["target_function"])
    )
    if leaked.any():
        raise AssertionError("Target leaked into donor selection")
    if result["donor_function"].isin(SYNTHETIC_DONORS).any():
        raise AssertionError("Synthetic donor selected despite exclusion")

    return result


def _safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _safe_geomean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def metrics(detail: pd.DataFrame, *, exclude_synthetic_targets: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = detail.copy()
    if exclude_synthetic_targets:
        data = data[data["target_is_synthetic"] == False].copy()

    rows = []
    for (config, seed), group in data.groupby(["config", "seed"], sort=False):
        selected = group[group["status"] == "selected"].copy()
        directional_all = group[group["target_preference"].isin(DIRECTIONAL)]
        directional = selected[selected["target_is_directional"] == True]
        independent_all = group[group["target_preference"] == "architecture-independent"]
        independent = selected[selected["target_is_independent"] == True]

        rows.append(
            {
                "config": config,
                "seed": int(seed),
                "target_count": len(group),
                "selection_coverage": len(selected) / len(group) if len(group) else np.nan,
                "directional_target_count": len(directional_all),
                "directional_selection_coverage": (
                    len(directional) / len(directional_all) if len(directional_all) else np.nan
                ),
                "independent_target_count": len(independent_all),
                "independent_selection_coverage": (
                    len(independent) / len(independent_all) if len(independent_all) else np.nan
                ),
                "predicted_direction_agreement": _safe_mean(directional["predicted_direction_agreement"]),
                "directional_compatibility": _safe_mean(directional["directional_compatibility"]),
                "directional_conflict_rate": _safe_mean(directional["directional_conflict"]),
                "strict_donor_direction_match": _safe_mean(directional["strict_donor_direction_match"]),
                "independent_donor_rate_on_directional_targets": _safe_mean(
                    directional["donor_independent_on_directional_target"]
                ),
                "raw_best_arm_agreement_diagnostic": _safe_mean(selected["raw_best_arm_agreement"]),
                "mean_abs_delta_error_pp_all": _safe_mean(selected["abs_delta_error_pp"]),
                "mean_abs_delta_error_pp_directional": _safe_mean(directional["abs_delta_error_pp"]),
                "mean_abs_delta_error_pp_independent": _safe_mean(independent["abs_delta_error_pp"]),
                "mean_abs_reward_gap_error_all": _safe_mean(selected["abs_reward_gap_error"]),
                "mean_abs_reward_gap_error_directional": _safe_mean(directional["abs_reward_gap_error"]),
                "mean_abs_reward_gap_error_independent": _safe_mean(independent["abs_reward_gap_error"]),
                "geomean_multiplicative_mismatch_all": _safe_geomean(selected["multiplicative_mismatch"]),
                "directional_geomean_speedup": _safe_geomean(directional["realized_speedup"]),
                "directional_mean_regret_percent": _safe_mean(directional["regret_percent"]),
                "independent_mean_regret_percent": _safe_mean(independent["regret_percent"]),
                "independent_mean_neutral_band_excess_pp": _safe_mean(independent["neutral_band_excess_pp"]),
                "same_language_rate": _safe_mean(selected["same_language"]),
            }
        )

    per_seed = pd.DataFrame(rows)
    metric_columns = [
        c for c in per_seed.columns if c not in {"config", "seed"}
    ]
    summary = (
        per_seed.groupby("config", as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    summary.insert(1, "seed_count", per_seed.groupby("config")["seed"].nunique().values)
    summary.insert(2, "target_scope", "non_synthetic" if exclude_synthetic_targets else "all_59")
    return per_seed, summary


def comparison_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("A_legacy_K5", "B_hybrid_K6_nearest", "B_minus_A_new_clustering"),
        ("B_hybrid_K6_nearest", "C_final_plain", "C_minus_B_final_donor"),
        ("A_legacy_K5", "C_final_plain", "C_minus_A_total_pipeline"),
    ]
    metrics_to_diff = [
        "selection_coverage",
        "directional_selection_coverage",
        "predicted_direction_agreement",
        "directional_compatibility",
        "directional_conflict_rate",
        "strict_donor_direction_match",
        "independent_donor_rate_on_directional_targets",
        "mean_abs_delta_error_pp_all",
        "mean_abs_reward_gap_error_all",
        "directional_mean_regret_percent",
        "independent_mean_regret_percent",
        "same_language_rate",
    ]
    rows = []
    for a, b, label in pairs:
        aa = per_seed[per_seed["config"] == a].set_index("seed")
        bb = per_seed[per_seed["config"] == b].set_index("seed")
        common = aa.index.intersection(bb.index)
        row = {"comparison": label}
        for metric in metrics_to_diff:
            row[f"delta_{metric}"] = float((bb.loc[common, metric] - aa.loc[common, metric]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_panel(
        ax,
        coords: np.ndarray,
        labels: np.ndarray,
        categories: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
) -> None:
    for category in categories:
        mask = labels == category
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=f"{category} ({int(mask.sum())})",
                s=38,
                alpha=0.82,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)


def save_projection_pair(
        pca: np.ndarray,
        umap_coords: np.ndarray,
        labels: np.ndarray,
        categories: list[str],
        title: str,
        output_base: Path,
) -> None:
    # Combined panel for slides.
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.6), layout="constrained")
    plot_panel(axes[0], pca, labels, categories, "PCA", "PC1", "PC2")
    plot_panel(axes[1], umap_coords, labels, categories, "UMAP", "UMAP1", "UMAP2")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_base.with_suffix(".png"), dpi=190)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)

    # Separate figures give more freedom when composing final slides.
    for suffix, coords, name, xlabel, ylabel in (
            ("pca", pca, "PCA", "PC1", "PC2"),
            ("umap", umap_coords, "UMAP", "UMAP1", "UMAP2"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.8), layout="constrained")
        plot_panel(ax, coords, labels, categories, name, xlabel, ylabel)
        fig.suptitle(title, fontsize=13)
        fig.savefig(output_base.parent / f"{output_base.name}_{suffix}.png", dpi=190)
        fig.savefig(output_base.parent / f"{output_base.name}_{suffix}.svg")
        plt.close(fig)


def full_fit_space(df: pd.DataFrame, space: str) -> tuple[np.ndarray, np.ndarray]:
    if space == "legacy":
        columns = PAPER5
    elif space == "hybrid":
        columns = FEATURE7
    else:
        raise ValueError(space)

    plain = MinMaxScaler().fit_transform(df[columns].to_numpy(float))
    cluster = plain.copy()
    if space == "hybrid":
        cluster[:, : len(PAPER5)] /= math.sqrt(len(PAPER5))
        cluster[:, len(PAPER5) :] /= math.sqrt(len(STATIC2))
    return plain, cluster


def plot_full(df: pd.DataFrame, out: Path, seed: int, umap_neighbors: int) -> pd.DataFrame:
    try:
        import umap
    except ImportError as exc:
        raise SystemExit("UMAP unavailable: pip install umap-learn") from exc

    out.mkdir(parents=True, exist_ok=True)
    rows = []

    presentation_names = {
        "A_legacy_K5": "Legacy clustering — K=5",
        "C_final_plain": "Final clustering — K=6",
    }

    # A and C are the two presentation configurations; B is still exported as
    # quantitative cluster metrics for the controlled A->B->C decomposition.
    for config_name, cfg in CONFIGS.items():
        _, x_cluster = full_fit_space(df, cfg["space"])
        labels = KMeans(
            n_clusters=cfg["k"],
            n_init=50,
            random_state=seed,
        ).fit_predict(x_cluster)

        pca_model = PCA(n_components=2)
        pca = pca_model.fit_transform(x_cluster)
        um = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.1,
            random_state=seed,
        ).fit_transform(x_cluster)

        cluster_labels = np.asarray([f"cluster {int(x)}" for x in labels])
        cluster_categories = [f"cluster {int(x)}" for x in sorted(np.unique(labels))]

        if config_name in {"A_legacy_K5", "C_final_plain"}:
            presentation_name = presentation_names[config_name]
            save_projection_pair(
                pca,
                um,
                cluster_labels,
                cluster_categories,
                f"{presentation_name} | cluster assegnati | n=59",
                out / f"{config_name}_clusters_pca_umap",
                )
            for slug, field, threshold in (
                    ("2p5", "architecture_preference", PRIMARY_THRESHOLD),
                    ("15", "preference_15", SENSITIVITY_THRESHOLD),
            ):
                save_projection_pair(
                    pca,
                    um,
                    df[field].astype(str).to_numpy(),
                    list(PREF_ORDER),
                    f"{presentation_name} | ground truth τ={threshold:g}% | n=59",
                    out / f"{config_name}_groundtruth_t{slug}_pca_umap",
                    )

        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "config": config_name,
                "seed": seed,
                "n": len(df),
                "k": cfg["k"],
                "silhouette": silhouette_score(x_cluster, labels),
                "davies_bouldin": davies_bouldin_score(x_cluster, labels),
                "calinski_harabasz": calinski_harabasz_score(x_cluster, labels),
                "max_cluster_size": int(sizes.max()),
                "singleton_clusters": int((sizes == 1).sum()),
                "pca_2_explained_variance": float(pca_model.explained_variance_ratio_.sum()),
                "ari_groundtruth_2p5": adjusted_rand_score(df["architecture_preference"], labels),
                "nmi_groundtruth_2p5": normalized_mutual_info_score(df["architecture_preference"], labels),
                "ari_groundtruth_15": adjusted_rand_score(df["preference_15"], labels),
                "nmi_groundtruth_15": normalized_mutual_info_score(df["preference_15"], labels),
                "nmi_language": normalized_mutual_info_score(df["language"], labels),
            }
        )

        pd.DataFrame(
            {
                "function_name": df["function_name"],
                "config": config_name,
                "cluster": labels,
                "language": df["language"],
                "preference_2p5": df["architecture_preference"],
                "preference_15": df["preference_15"],
                "pca1": pca[:, 0],
                "pca2": pca[:, 1],
                "umap1": um[:, 0],
                "umap2": um[:, 1],
            }
        ).to_csv(out / f"{config_name}_fullfit_coordinates.csv", index=False)

    return pd.DataFrame(rows)


def write_report(
        out: Path,
        all_summary: pd.DataFrame,
        primary_summary: pd.DataFrame,
        comparisons: pd.DataFrame,
        cluster_metrics: pd.DataFrame,
        paths: dict[str, Path],
        seeds: tuple[int, ...],
) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Final clustering / donor validation",
        "",
        "## Frozen semantics",
        "",
        "- Primary ground truth: tau=2.5%; tau=15% is sensitivity.",
        "- Architecture-independent targets are NOT counted as directional-agreement failures.",
        "- Their donor/target mismatch is evaluated with continuous effect errors and induced regret.",
        "- Final clustering uses block-sqrt geometry; final donor distance uses PLAIN Manhattan.",
        "- `amd_faster` and `arm_faster` are excluded as donor candidates in every A/B/C case so the controlled comparison does not confound donor eligibility.",
        f"- LOFO KMeans seeds: {', '.join(map(str, seeds))}; n_init=50.",
        "",
        "## Primary donor-quality summary (non-synthetic targets)",
        "",
        table(primary_summary),
        "",
        "## All-59 sensitivity summary",
        "",
        table(all_summary),
        "",
        "## Controlled A -> B -> C deltas",
        "",
        table(comparisons),
        "",
        "## Descriptive full-fit cluster metrics",
        "",
        table(cluster_metrics),
        "",
        "## Reading independent targets",
        "",
        "Directional classification-style metrics use only x86-preferred / arm-preferred targets. For those targets, `directional_compatibility` counts a neutral donor as non-conflicting; `strict_donor_direction_match` requires exact donor/target class equality; and `directional_conflict_rate` counts only opposite directional donors as conflicts. For an architecture-independent target there is no correct directional class to predict. The relevant quantities are `mean_abs_delta_error_pp_independent`, `mean_abs_reward_gap_error_independent`, `independent_mean_regret_percent`, and `independent_mean_neutral_band_excess_pp`.",
        "",
        "## Provenance",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{path}`; SHA256 `{sha256(path)}`")
    (out / "FINAL_CLUSTERING_DONOR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="11,23,37,41,53")
    parser.add_argument("--plot-seed", type=int, default=11)
    parser.add_argument("--umap-neighbors", type=int, default=10)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df, paths = prepare(root)
    df[[
        "function_name",
        "language",
        "architecture_preference",
        "preference_15",
        "arm_vs_x86_delta_percent",
    ]].to_csv(out / "eligible_59_functions.csv", index=False)

    detail = lofo(df, seeds)
    detail.to_csv(out / "lofo_59_per_target_seed.csv", index=False)

    all_seed, all_summary = metrics(detail, exclude_synthetic_targets=False)
    primary_seed, primary_summary = metrics(detail, exclude_synthetic_targets=True)
    all_seed.to_csv(out / "lofo_59_per_seed_all_targets.csv", index=False)
    all_summary.to_csv(out / "lofo_59_summary_all_targets.csv", index=False)
    primary_seed.to_csv(out / "lofo_59_per_seed_primary_nonsynthetic.csv", index=False)
    primary_summary.to_csv(out / "lofo_59_summary_primary_nonsynthetic.csv", index=False)

    comparisons = comparison_table(primary_seed)
    comparisons.to_csv(out / "lofo_59_controlled_comparisons.csv", index=False)

    cluster_metrics = pd.DataFrame()
    if not args.skip_plots:
        cluster_metrics = plot_full(df, out / "figures", args.plot_seed, args.umap_neighbors)
        cluster_metrics.to_csv(out / "fullfit_cluster_metrics.csv", index=False)

    write_report(
        out,
        all_summary,
        primary_summary,
        comparisons,
        cluster_metrics,
        paths,
        seeds,
    )

    manifest = {
        "eligible_cluster_functions": 59,
        "primary_donor_target_scope": "non-synthetic targets",
        "synthetic_donor_exclusion": sorted(SYNTHETIC_DONORS),
        "primary_ground_truth_threshold_percent": PRIMARY_THRESHOLD,
        "sensitivity_ground_truth_threshold_percent": SENSITIVITY_THRESHOLD,
        "lofo_seeds": list(seeds),
        "plot_seed": args.plot_seed,
        "kmeans_n_init": 50,
        "final_k": 6,
        "final_cluster_geometry": "MinMax 7D; dynamic/sqrt(5), static/sqrt(2)",
        "final_donor_geometry": "plain Manhattan on 7 MinMax coordinates",
        "final_donor_rule": "directional vote 1/(d+1e-9)^2 + nearest in predicted class",
        "independent_target_directional_classification": "undefined/excluded",
        "directional_target_independent_donor_semantics": "compatible/non-conflicting",
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    print("PASS — final clustering/donor comparison generated")
    print("PRIMARY (non-synthetic targets):")
    print(primary_summary.to_string(index=False))
    if not cluster_metrics.empty:
        print("FULL-FIT CLUSTER METRICS:")
        print(cluster_metrics.to_string(index=False))
    print("REPORT:", out / "FINAL_CLUSTERING_DONOR_REPORT.md")


if __name__ == "__main__":
    main()
