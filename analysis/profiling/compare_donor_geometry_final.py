#!/usr/bin/env python3
"""Controlled donor-geometry ablation for the frozen Serverledge pipeline.

Everything except donor distance is held fixed:
- 7 features: PAPER5 + 2 static features
- LOFO target exclusion before scaler/KMeans/donor selection
- train-only MinMaxScaler
- KMeans space: dynamic/sqrt(5), static/sqrt(2)
- K=6, n_init=50, seeds 11,23,37,41,53
- synthetic donors amd_faster/arm_faster excluded
- same-cluster directional vote sum 1/(d+1e-9)^2
- donor = nearest candidate inside predicted directional class

Compared donor distances:
- euclidean_cluster_space: Euclidean in the block-sqrt KMeans space
- manhattan_block_mean: mean L1 per block, dyn/5 + static/2
- manhattan_sqrt: L1 in the block-sqrt KMeans space
- manhattan_plain: L1 over the seven plain MinMax coordinates (FINAL)

Primary report scope excludes the two synthetic functions as targets. Architecture-
independent targets have no directional classification error; they remain in
continuous mismatch and induced-regret diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

DYN = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]
STAT = ["static_token_count_mean", "static_function_count"]
FEATURES = DYN + STAT
DIRECTIONS = ("x86-preferred", "arm-preferred")
SYNTHETIC = {"amd_faster", "arm_faster"}
EPS = 1e-9
PRIMARY_THRESHOLD = 2.5

VARIANTS = (
    "euclidean_cluster_space",
    "manhattan_block_mean",
    "manhattan_sqrt",
    "manhattan_plain",
)
FINAL_VARIANT = "manhattan_plain"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_unique(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["function_name"].duplicated().any():
        raise ValueError(f"{path}: duplicate function_name")
    return df


def load(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = {
        "profiles": root / "resource/x86/function-profiles-median.csv",
        "static": root / "static-code-metrics.csv",
        "ground_truth": root / "ground_truth/preferences-2p5.csv",
        "language": root / "function-languages.csv",
    }

    p = load_unique(paths["profiles"], ["function_name", *DYN])
    s = load_unique(
        paths["static"],
        ["function_name", "static_source_available", *STAT],
    )
    g = load_unique(
        paths["ground_truth"],
        [
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
            "arm_vs_x86_delta_percent",
        ],
    )
    l = load_unique(paths["language"], ["function_name", "language"])

    df = (
        p[["function_name", *DYN]]
        .merge(
            s[["function_name", "static_source_available", *STAT]],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            g[
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
        .merge(l[["function_name", "language"]], on="function_name", validate="one_to_one")
    )

    mask = (
        df["static_source_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    for c in [*FEATURES, "x86_duration_ms", "arm_duration_ms", "arm_vs_x86_delta_percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        mask &= np.isfinite(df[c].to_numpy(float))

    df = df.loc[mask].sort_values("function_name").reset_index(drop=True)
    if len(df) != 59:
        raise ValueError(f"expected 59 eligible functions, got {len(df)}")
    if not (df[["x86_duration_ms", "arm_duration_ms"]] > 0).all().all():
        raise ValueError("non-positive ground-truth duration")

    hashes = {str(path.relative_to(root)): sha256(path) for path in paths.values()}
    return df, hashes


def block_sqrt(x_plain: np.ndarray) -> np.ndarray:
    scales = np.array(
        [math.sqrt(len(DYN))] * len(DYN)
        + [math.sqrt(len(STAT))] * len(STAT),
        dtype=float,
        )
    return x_plain / scales


def distances(
        variant: str,
        x_plain: np.ndarray,
        xt_plain: np.ndarray,
        x_cluster: np.ndarray,
        xt_cluster: np.ndarray,
) -> np.ndarray:
    if variant == "manhattan_plain":
        return np.abs(x_plain - xt_plain[0]).sum(axis=1)
    if variant == "manhattan_sqrt":
        return np.abs(x_cluster - xt_cluster[0]).sum(axis=1)
    if variant == "manhattan_block_mean":
        delta = np.abs(x_plain - xt_plain[0])
        return (
                delta[:, : len(DYN)].sum(axis=1) / len(DYN)
                + delta[:, len(DYN) :].sum(axis=1) / len(STAT)
        )
    if variant == "euclidean_cluster_space":
        return np.sqrt(((x_cluster - xt_cluster[0]) ** 2).sum(axis=1))
    raise ValueError(variant)


def choose(
        train: pd.DataFrame,
        labels: np.ndarray,
        target_cluster: int,
        dist: np.ndarray,
) -> tuple[int | None, str, str]:
    same = np.flatnonzero(labels == target_cluster)
    if not len(same):
        return None, "", "no_cluster_member"

    names = train["function_name"].astype(str).to_numpy()
    prefs = train["architecture_preference"].astype(str).to_numpy()

    same = same[~np.isin(names[same], list(SYNTHETIC))]
    directional = same[np.isin(prefs[same], DIRECTIONS)]
    if not len(directional):
        return None, "", "no_directional_member"

    votes: dict[str, float] = {}
    for direction in DIRECTIONS:
        members = directional[prefs[directional] == direction]
        votes[direction] = (
            float(np.sum(1.0 / np.power(dist[members] + EPS, 2)))
            if len(members)
            else 0.0
        )

    if math.isclose(votes[DIRECTIONS[0]], votes[DIRECTIONS[1]], rel_tol=0.0, abs_tol=1e-15):
        return None, "", "tied_vote"

    predicted = max(DIRECTIONS, key=lambda d: votes[d])
    pool = directional[prefs[directional] == predicted]
    order = np.lexsort((names[pool], dist[pool]))
    return int(pool[order[0]]), predicted, "selected"


def raw_best_label(row: pd.Series) -> str:
    return "x86-preferred" if float(row.x86_duration_ms) <= float(row.arm_duration_ms) else "arm-preferred"


def reward_gap(x86_ms: float, arm_ms: float) -> float:
    # Reward r=-ln(ms), so ARM-x86 reward gap = ln(x86/arm).
    return float(np.log(x86_ms / arm_ms))


def outcome(target: pd.Series, donor: pd.Series, predicted: str) -> dict:
    tx, ta = float(target.x86_duration_ms), float(target.arm_duration_ms)
    dx, da = float(donor.x86_duration_ms), float(donor.arm_duration_ms)
    td = float(target.arm_vs_x86_delta_percent)
    dd = float(donor.arm_vs_x86_delta_percent)
    target_pref = str(target.architecture_preference)
    donor_pref = str(donor.architecture_preference)
    directional = target_pref in DIRECTIONS
    independent = target_pref == "architecture-independent"

    chosen = tx if predicted == "x86-preferred" else ta
    alternative = ta if predicted == "x86-preferred" else tx
    best = min(tx, ta)

    # Donors are directional by construction in this ablation, therefore for
    # directional targets compatibility == strict match == prediction agreement.
    agreement = float(predicted == target_pref) if directional else np.nan

    return {
        "target_is_directional": directional,
        "target_is_independent": independent,
        "donor_preference": donor_pref,
        "predicted_direction_agreement": agreement,
        "directional_compatibility": agreement,
        "directional_conflict_rate": (1.0 - agreement) if directional else np.nan,
        "strict_donor_direction_match": float(donor_pref == target_pref) if directional else np.nan,
        "raw_best_arm_agreement_diagnostic": float(raw_best_label(donor) == raw_best_label(target)),
        "target_delta_percent": td,
        "donor_delta_percent": dd,
        "abs_delta_error_pp": abs(td - dd),
        "target_reward_gap": reward_gap(tx, ta),
        "donor_reward_gap": reward_gap(dx, da),
        "abs_reward_gap_error": abs(reward_gap(tx, ta) - reward_gap(dx, da)),
        "multiplicative_mismatch": math.exp(abs(math.log(ta / tx) - math.log(da / dx))),
        "regret_percent": (chosen / best - 1.0) * 100.0,
        "realized_speedup": alternative / chosen,
        "neutral_band_excess_pp": max(0.0, abs(dd) - PRIMARY_THRESHOLD),
        "same_language": float(str(target.language) == str(donor.language)),
    }


def evaluate(df: pd.DataFrame, seeds: tuple[int, ...], n_init: int) -> pd.DataFrame:
    rows: list[dict] = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            print(f"LOFO seed {seed}...", flush=True)
            for i in range(len(df)):
                target = df.iloc[[i]].copy()
                train = df.drop(index=i).reset_index(drop=True)
                t = target.iloc[0]

                scaler = MinMaxScaler()
                x_plain = scaler.fit_transform(train[FEATURES].to_numpy(float))
                xt_plain = scaler.transform(target[FEATURES].to_numpy(float))
                x_cluster = block_sqrt(x_plain)
                xt_cluster = block_sqrt(xt_plain)

                km = KMeans(n_clusters=6, n_init=n_init, random_state=seed)
                labels = km.fit_predict(x_cluster)
                target_cluster = int(km.predict(xt_cluster)[0])

                for variant in VARIANTS:
                    dist = distances(variant, x_plain, xt_plain, x_cluster, xt_cluster)
                    j, predicted, status = choose(train, labels, target_cluster, dist)
                    base = {
                        "variant": variant,
                        "seed": seed,
                        "target_function": str(t.function_name),
                        "target_preference": str(t.architecture_preference),
                        "target_is_synthetic": str(t.function_name) in SYNTHETIC,
                        "status": status,
                        "donor_function": "",
                        "prediction": predicted,
                        "selected_distance": np.nan,
                    }
                    if j is not None:
                        donor = train.iloc[j]
                        base["donor_function"] = str(donor.function_name)
                        base["selected_distance"] = float(dist[j])
                        base.update(outcome(t, donor, predicted))
                    rows.append(base)

    result = pd.DataFrame(rows)
    expected = len(df) * len(seeds) * len(VARIANTS)
    if len(result) != expected:
        raise AssertionError(f"unexpected row count {len(result)} != {expected}")
    if result.duplicated(["variant", "seed", "target_function"]).any():
        raise AssertionError("duplicate LOFO key")
    if result["donor_function"].isin(SYNTHETIC).any():
        raise AssertionError("synthetic donor selected")
    if ((result.status == "selected") & (result.donor_function == result.target_function)).any():
        raise AssertionError("target leaked into donor selection")
    return result


def safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def safe_geomean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def summarize(rows: pd.DataFrame, exclude_synthetic_targets: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = rows.copy()
    if exclude_synthetic_targets:
        data = data[data.target_is_synthetic == False].copy()

    per_seed_rows: list[dict] = []
    for (variant, seed), g in data.groupby(["variant", "seed"], sort=False):
        selected = g[g.status == "selected"].copy()
        directional_all = g[g.target_preference.isin(DIRECTIONS)]
        directional = selected[selected.target_is_directional == True]
        independent_all = g[g.target_preference == "architecture-independent"]
        independent = selected[selected.target_is_independent == True]

        per_seed_rows.append({
            "variant": variant,
            "seed": int(seed),
            "target_count": len(g),
            "selection_coverage": len(selected) / len(g),
            "directional_target_count": len(directional_all),
            "directional_selection_coverage": len(directional) / len(directional_all) if len(directional_all) else np.nan,
            "independent_target_count": len(independent_all),
            "independent_selection_coverage": len(independent) / len(independent_all) if len(independent_all) else np.nan,
            "predicted_direction_agreement": safe_mean(directional.predicted_direction_agreement),
            "directional_compatibility": safe_mean(directional.directional_compatibility),
            "directional_conflict_rate": safe_mean(directional.directional_conflict_rate),
            "strict_donor_direction_match": safe_mean(directional.strict_donor_direction_match),
            "raw_best_arm_agreement_diagnostic": safe_mean(selected.raw_best_arm_agreement_diagnostic),
            "mean_abs_delta_error_pp_all": safe_mean(selected.abs_delta_error_pp),
            "mean_abs_delta_error_pp_directional": safe_mean(directional.abs_delta_error_pp),
            "mean_abs_delta_error_pp_independent": safe_mean(independent.abs_delta_error_pp),
            "mean_abs_reward_gap_error_all": safe_mean(selected.abs_reward_gap_error),
            "mean_abs_reward_gap_error_directional": safe_mean(directional.abs_reward_gap_error),
            "mean_abs_reward_gap_error_independent": safe_mean(independent.abs_reward_gap_error),
            "geomean_multiplicative_mismatch_all": safe_geomean(selected.multiplicative_mismatch),
            "directional_geomean_speedup": safe_geomean(directional.realized_speedup),
            "directional_mean_regret_percent": safe_mean(directional.regret_percent),
            "independent_mean_regret_percent": safe_mean(independent.regret_percent),
            "independent_mean_neutral_band_excess_pp": safe_mean(independent.neutral_band_excess_pp),
            "same_language_rate": safe_mean(selected.same_language),
        })

    per_seed = pd.DataFrame(per_seed_rows)
    metric_cols = [c for c in per_seed.columns if c not in {"variant", "seed"}]
    summary = per_seed.groupby("variant", as_index=False)[metric_cols].mean(numeric_only=True)
    summary.insert(1, "seed_count", per_seed.groupby("variant")["seed"].nunique().reindex(summary.variant).to_numpy())
    summary.insert(2, "target_scope", "non_synthetic" if exclude_synthetic_targets else "all_59")
    order = {v: i for i, v in enumerate(VARIANTS)}
    summary["_order"] = summary.variant.map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return per_seed, summary


def bootstrap_paired(rows: pd.DataFrame, nboot: int, seed: int) -> pd.DataFrame:
    data = rows[rows.target_is_synthetic == False].copy()
    ref = data[data.variant == FINAL_VARIANT].set_index(["target_function", "seed"])
    rng = np.random.default_rng(seed)
    metrics = [
        "predicted_direction_agreement",
        "regret_percent",
        "realized_speedup",
        "abs_reward_gap_error",
        "abs_delta_error_pp",
    ]
    out: list[dict] = []

    for variant in VARIANTS:
        if variant == FINAL_VARIANT:
            continue
        oth = data[data.variant == variant].set_index(["target_function", "seed"])
        idx = ref.index.intersection(oth.index)
        a = ref.loc[idx]
        b = oth.loc[idx]
        common = (a.status == "selected") & (b.status == "selected")
        donor_diff = int((a.loc[common, "donor_function"] != b.loc[common, "donor_function"]).sum())
        pred_diff = int((a.loc[common, "prediction"] != b.loc[common, "prediction"]).sum())

        for metric in metrics:
            valid = common & a[metric].notna() & b[metric].notna()
            if metric in {"predicted_direction_agreement", "regret_percent", "realized_speedup"}:
                # Primary arm-choice metrics apply only to directional targets.
                valid &= a["target_is_directional"] == True
                valid &= b["target_is_directional"] == True
            if not valid.any():
                continue

            # Delta is FINAL plain minus comparator. Aggregate seeds within target,
            # then bootstrap targets so repeated seeds are not treated as independent functions.
            delta = (a.loc[valid, metric] - b.loc[valid, metric]).groupby(level=0).mean().to_numpy(float)
            boot = np.empty(nboot, dtype=float)
            for start in range(0, nboot, 1000):
                k = min(1000, nboot - start)
                draw = rng.integers(0, len(delta), size=(k, len(delta)))
                boot[start:start+k] = delta[draw].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({
                "comparison": f"{FINAL_VARIANT}_minus_{variant}",
                "metric": metric,
                "n_targets": int(len(delta)),
                "n_common_target_seeds": int(valid.sum()),
                "delta_final_minus_comparator": float(delta.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "donor_differences_common": donor_diff,
                "prediction_differences_common": pred_diff,
            })
    return pd.DataFrame(out)


def presentation_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant",
        "selection_coverage",
        "predicted_direction_agreement",
        "directional_mean_regret_percent",
        "directional_geomean_speedup",
        "mean_abs_reward_gap_error_directional",
        "mean_abs_delta_error_pp_directional",
        "same_language_rate",
    ]
    return summary[cols].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seeds", default="11,23,37,41,53")
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261011)
    args = ap.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    df, hashes = load(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate(df, seeds, args.n_init)
    per_seed, summary = summarize(rows, exclude_synthetic_targets=True)
    paired = bootstrap_paired(rows, args.bootstrap_replicates, args.bootstrap_seed)
    slide = presentation_table(summary)

    rows.to_csv(args.output_dir / "donor_geometry_per_target_seed.csv", index=False)
    per_seed.to_csv(args.output_dir / "donor_geometry_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "donor_geometry_summary.csv", index=False)
    paired.to_csv(args.output_dir / "donor_geometry_paired_vs_plain.csv", index=False)
    slide.to_csv(args.output_dir / "donor_geometry_presentation_table.csv", index=False)

    manifest = {
        "input_hashes": hashes,
        "target_count_total": 59,
        "primary_target_scope": "57 non-synthetic targets",
        "synthetic_functions": sorted(SYNTHETIC),
        "synthetic_donor_policy": "excluded for every variant",
        "seeds": seeds,
        "k": 6,
        "n_init": args.n_init,
        "features": FEATURES,
        "clustering_space": "train-only MinMax; dynamic/sqrt(5), static/sqrt(2)",
        "donor_rule": "same cluster; directional vote sum 1/(d+1e-9)^2; nearest in predicted class",
        "distance_variants": {
            "euclidean_cluster_space": "Euclidean on block-sqrt KMeans coordinates",
            "manhattan_block_mean": "sum(|dyn delta|)/5 + sum(|static delta|)/2 on plain MinMax",
            "manhattan_sqrt": "Manhattan on block-sqrt KMeans coordinates",
            "manhattan_plain": "Manhattan over all 7 plain MinMax coordinates (FINAL)",
        },
        "final_variant": FINAL_VARIANT,
        "ground_truth_threshold_percent": PRIMARY_THRESHOLD,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "unit": "target after averaging seeds",
        },
        "note": "All four variants reuse exactly the same LOFO KMeans clustering for each target/seed; only donor distance changes. Target ground truth is posthoc only.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("PASS — final donor-geometry ablation generated")
    print("PRIMARY SUMMARY (57 non-synthetic targets):")
    print(slide.to_string(index=False))
    print("\nPAIRED FINAL PLAIN vs alternatives:")
    focus = paired[paired.metric.isin([
        "predicted_direction_agreement",
        "regret_percent",
        "realized_speedup",
        "abs_reward_gap_error",
    ])]
    print(focus.to_string(index=False))
    print("OUTPUT:", args.output_dir)


if __name__ == "__main__":
    main()
