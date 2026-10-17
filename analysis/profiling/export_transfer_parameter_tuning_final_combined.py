#!/usr/bin/env python3
"""Combine the two saved TL tuning phases into reproducible + slide-ready tables.

This script DOES NOT retune or select new hyperparameters.
It reads the already-produced tuning artifacts from:
  1) the initial outer-CV tuning phase;
  2) the boundary-extension tuning phase;
and exports a single provenance-preserving package.

Important methodological note
-----------------------------
The historical tuning artifacts freeze the donor mapping referenced in each
manifest. They therefore document how the parameter values were selected at
that stage. A later confirmatory replay on the final plain-Manhattan donor
pipeline should be reported separately when isolating the final deployed
configuration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

OLD = {
    "difference": (0.25, 1.0, 0.8),
    "ratio": (0.25, 1.0, 0.8),
}
FINAL = {
    "difference": (0.05, 64.0, 0.2),
    "ratio": (0.01, 64.0, 0.4),
}

PHASES = (
    ("initial_cv", "initial"),
    ("boundary_extension", "boundary"),
)


def close(a: Any, b: float) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")
    return path


def matches(row: pd.Series, spec: tuple[float, float, float]) -> bool:
    wr, we, c = spec
    return close(row.get("w_r"), wr) and close(row.get("w_e"), we) and close(row.get("c"), c)


def load_phase(path: Path, phase: str) -> dict[str, Any]:
    path = path.resolve()
    ranking = pd.read_csv(require(path / "tl-tuning-all-target-ranking.csv"))
    grid = pd.read_csv(require(path / "tl-tuning-grid.csv"))
    rec = pd.read_csv(require(path / "tl-tuning-full-data-recommendation.csv"))
    folds = pd.read_csv(require(path / "tl-tuning-fold-selections.csv"))
    manifest = json.loads(require(path / "tl-tuning-manifest.json").read_text(encoding="utf-8"))

    for df in (ranking, grid, rec, folds):
        df.insert(0, "phase", phase)

    ranking["source_dir"] = str(path)
    donor = manifest.get("frozen_upstream", {}).get("donor_variant")
    ranking["source_donor_variant"] = donor
    donor_detail = manifest.get("inputs", {}).get("donor_detail", {})
    ranking["source_donor_detail_sha256"] = donor_detail.get("sha256", "")

    return {
        "path": path,
        "ranking": ranking,
        "grid": grid,
        "recommendation": rec,
        "folds": folds,
        "manifest": manifest,
    }


def metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "phase", "formula", "config_id", "regime", "w_r", "w_e", "c",
        "rank_within_formula", "target_count", "latency_gain",
        "reward_pseudo_regret_saved", "optimal_arm_rate_gain",
        "wrong_selections_saved", "first_arm_optimal_gain",
        "is_old_config", "is_frozen_final", "is_full_data_recommendation",
    ]
    return [c for c in preferred if c in df.columns]


def phase_space(grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (phase, regime), g in grid.groupby(["phase", "regime"], dropna=False):
        rows.append({
            "phase": phase,
            "regime": regime,
            "config_count_per_formula": len(g),
            "w_R_values": ", ".join(f"{x:g}" for x in sorted(g["w_r"].dropna().unique())),
            "w_E_values": ", ".join(f"{x:g}" for x in sorted(g["w_e"].dropna().unique())),
            "c_values": ", ".join(f"{x:g}" for x in sorted(g["c"].dropna().unique())),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-dir", type=Path, required=True)
    ap.add_argument("--boundary-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    phases = [
        load_phase(args.initial_dir, "initial_cv"),
        load_phase(args.boundary_dir, "boundary_extension"),
    ]

    ranking = pd.concat([p["ranking"] for p in phases], ignore_index=True, sort=False)
    grid = pd.concat([p["grid"] for p in phases], ignore_index=True, sort=False)
    rec = pd.concat([p["recommendation"] for p in phases], ignore_index=True, sort=False)
    folds = pd.concat([p["folds"] for p in phases], ignore_index=True, sort=False)

    required = {
        "phase", "formula", "config_id", "regime", "w_r", "w_e", "c",
        "rank_within_formula", "latency_gain", "reward_pseudo_regret_saved",
        "optimal_arm_rate_gain", "wrong_selections_saved",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise SystemExit(f"Combined ranking missing columns: {missing}")

    ranking = ranking.sort_values(
        ["phase", "formula", "rank_within_formula", "w_r", "w_e", "c"]
    ).reset_index(drop=True)

    # Transparent flags. Historical config belongs to the initial phase; frozen
    # final configs should be present in the boundary-extension phase.
    ranking["is_old_config"] = False
    ranking["is_frozen_final"] = False
    ranking["is_top_n_within_phase_formula"] = ranking["rank_within_formula"] <= args.top_n

    for formula, spec in OLD.items():
        mask = (
                (ranking["phase"] == "initial_cv")
                & (ranking["formula"].astype(str) == formula)
                & ranking.apply(lambda r: matches(r, spec), axis=1)
        )
        ranking.loc[mask, "is_old_config"] = True

    for formula, spec in FINAL.items():
        mask = (
                (ranking["phase"] == "boundary_extension")
                & (ranking["formula"].astype(str) == formula)
                & ranking.apply(lambda r: matches(r, spec), axis=1)
        )
        ranking.loc[mask, "is_frozen_final"] = True

    rec_keys = set(
        zip(
            rec["phase"].astype(str),
            rec["formula"].astype(str),
            rec["config_id"].astype(str),
        )
    )
    ranking["is_full_data_recommendation"] = [
        (str(ph), str(f), str(cid)) in rec_keys
        for ph, f, cid in zip(ranking["phase"], ranking["formula"], ranking["config_id"])
    ]

    ranking.to_csv(out / "tuning_all_tested_configurations_both_phases_h10.csv", index=False)

    tested_space = phase_space(grid)
    tested_space.to_csv(out / "tuning_tested_parameter_space_both_phases.csv", index=False)

    # Recommendations and fold selections are copied with phase labels.
    rec.to_csv(out / "tuning_full_data_recommendations_both_phases.csv", index=False)
    folds.to_csv(out / "tuning_fold_selected_configurations_both_phases.csv", index=False)

    shortlist = ranking[
        ranking[[
            "is_top_n_within_phase_formula",
            "is_old_config",
            "is_frozen_final",
            "is_full_data_recommendation",
        ]].any(axis=1)
    ].copy()
    shortlist = shortlist.sort_values(
        ["formula", "phase", "rank_within_formula", "w_r", "w_e", "c"]
    )
    shortlist.to_csv(out / "tuning_presentation_shortlist_h10.csv", index=False)

    # One compact row for the exact historical and frozen-final configurations.
    key_rows = ranking[ranking["is_old_config"] | ranking["is_frozen_final"]].copy()
    key_rows = key_rows.sort_values(["formula", "phase"])
    key_rows.to_csv(out / "tuning_old_vs_frozen_final_h10.csv", index=False)

    # Human-readable report.
    lines = [
        "# Transfer Learning parameter tuning — two-phase history",
        "",
        "## Tested parameter spaces",
        "",
        tested_space.to_markdown(index=False),
        "",
        "## Full-data recommendations",
        "",
        rec.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Historical vs frozen-final configurations",
        "",
    ]
    if len(key_rows):
        lines.append(key_rows[metric_columns(key_rows)].to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No exact OLD/FINAL rows were found in the saved grids.")

    lines += [
        "",
        "## Presentation shortlist",
        "",
        shortlist[metric_columns(shortlist)].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Methodological interpretation",
        "",
        "- `initial_cv` is the first broad parameter search.",
        "- `boundary_extension` extends the search toward smaller w_R, larger w_E and smaller c.",
        "- The two phases are preserved rather than deduplicated: the phase is part of provenance.",
        "- These tables document the historical tuning study and its saved donor mapping.",
        "- A separate matched-c confirmatory replay on the final plain-Manhattan donor pipeline should be used to isolate final TL performance from the effect of changing c.",
        "",
        "## Frozen configurations",
        "",
        "- Historical: w_R=0.25, w_E=1, c=0.8 (both formulas).",
        "- Final Difference: w_R=0.05, w_E=64, c=0.2.",
        "- Final Ratio: w_R=0.01, w_E=64, c=0.4.",
    ]
    (out / "TUNING_TWO_PHASE_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifests = {}
    for p in phases:
        m = p["manifest"]
        manifests[p["ranking"]["phase"].iloc[0]] = {
            "source_dir": str(p["path"]),
            "parameter_grid": m.get("parameter_grid", {}),
            "primary_horizon": m.get("selection", {}).get("primary_horizon"),
            "validation": m.get("validation", {}),
            "frozen_upstream": m.get("frozen_upstream", {}),
            "donor_detail": m.get("inputs", {}).get("donor_detail", {}),
        }

    provenance = {
        "schema": "serverledge.tl-tuning-two-phase-export.v1",
        "phases": manifests,
        "old_configs": OLD,
        "frozen_final_configs": FINAL,
        "top_n": args.top_n,
        "warning": (
            "This export combines the saved historical tuning phases only. "
            "Do not reinterpret their scores as a matched-c confirmatory result on a later donor mapping."
        ),
    }
    (out / "tuning_two_phase_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    # Strong sanity checks for the rows we expect to exist.
    for formula in ("difference", "ratio"):
        if not ((ranking["formula"] == formula) & ranking["is_old_config"]).any():
            print(f"WARNING — historical config not found for {formula}")
        if not ((ranking["formula"] == formula) & ranking["is_frozen_final"]).any():
            print(f"WARNING — frozen final config not found for {formula}")

    print("PASS — combined two-phase tuning tables exported")
    print("full=", out / "tuning_all_tested_configurations_both_phases_h10.csv")
    print("space=", out / "tuning_tested_parameter_space_both_phases.csv")
    print("key=", out / "tuning_old_vs_frozen_final_h10.csv")
    print("shortlist=", out / "tuning_presentation_shortlist_h10.csv")
    print("report=", out / "TUNING_TWO_PHASE_TABLES.md")


if __name__ == "__main__":
    main()
