#!/usr/bin/env python3
"""Export complete + presentation-ready TL tuning tables from saved tuning artifacts.

This script DOES NOT tune or select new parameters. It only reads the already
completed tuning directory and exposes:
  1) the full H=10 ranking (all configurations actually tested),
  2) the exact tested parameter space,
  3) a compact presentation shortlist containing the historical config,
     the frozen final configs, the full-data recommendation, and top-N rows.

Use the full CSV in the thesis/reproducibility package; use the compact table in
slides. This avoids cherry-picking while keeping the slide readable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

OLD = {
    "difference": (0.25, 1.0, 0.8),
    "ratio": (0.25, 1.0, 0.8),
}
FINAL = {
    "difference": (0.05, 64.0, 0.2),
    "ratio": (0.01, 64.0, 0.4),
}


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")
    return path


def matches(row: pd.Series, spec: tuple[float, float, float]) -> bool:
    wr, we, c = spec
    return close(row["w_r"], wr) and close(row["w_e"], we) and close(row["c"], c)


def compact_metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "formula", "config_id", "regime", "w_r", "w_e", "c",
        "rank_within_formula", "target_count", "latency_gain",
        "reward_pseudo_regret_saved", "optimal_arm_rate_gain",
        "wrong_selections_saved", "first_arm_optimal_gain",
    ]
    return [c for c in preferred if c in df.columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    tuning = args.tuning_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ranking_path = require(tuning / "tl-tuning-all-target-ranking.csv")
    grid_path = require(tuning / "tl-tuning-grid.csv")
    recommendation_path = require(tuning / "tl-tuning-full-data-recommendation.csv")
    folds_path = require(tuning / "tl-tuning-fold-selections.csv")
    manifest_path = require(tuning / "tl-tuning-manifest.json")

    ranking = pd.read_csv(ranking_path)
    grid = pd.read_csv(grid_path)
    recommendation = pd.read_csv(recommendation_path)
    folds = pd.read_csv(folds_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {
        "formula", "config_id", "regime", "w_r", "w_e", "c",
        "rank_within_formula", "latency_gain",
        "reward_pseudo_regret_saved", "optimal_arm_rate_gain",
        "wrong_selections_saved",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise SystemExit(f"Ranking missing columns: {missing}")

    ranking = ranking.sort_values(["formula", "rank_within_formula"]).reset_index(drop=True)
    ranking.to_csv(out / "tuning_all_tested_configurations_h10.csv", index=False)

    # Tested-space summary: exact unique values, not a hand-written description.
    space_rows = []
    for regime, group in grid.groupby("regime", dropna=False):
        space_rows.append({
            "regime": regime,
            "config_count_per_formula": len(group),
            "w_R_values": ", ".join(f"{x:g}" for x in sorted(group["w_r"].unique())),
            "w_E_values": ", ".join(f"{x:g}" for x in sorted(group["w_e"].unique())),
            "c_values": ", ".join(f"{x:g}" for x in sorted(group["c"].unique())),
        })
    tested_space = pd.DataFrame(space_rows)
    tested_space.to_csv(out / "tuning_tested_parameter_space.csv", index=False)

    # Flag rows for a transparent presentation shortlist.
    ranking["is_top_n"] = ranking["rank_within_formula"] <= args.top_n
    ranking["is_old_config"] = False
    ranking["is_frozen_final"] = False
    ranking["is_full_data_recommendation"] = False

    for formula, spec in OLD.items():
        mask = (ranking["formula"] == formula) & ranking.apply(lambda r: matches(r, spec), axis=1)
        ranking.loc[mask, "is_old_config"] = True

    for formula, spec in FINAL.items():
        mask = (ranking["formula"] == formula) & ranking.apply(lambda r: matches(r, spec), axis=1)
        ranking.loc[mask, "is_frozen_final"] = True

    rec_keys = set(
        zip(
            recommendation["formula"].astype(str),
            recommendation["config_id"].astype(str),
        )
    )
    ranking["is_full_data_recommendation"] = [
        (str(f), str(cid)) in rec_keys
        for f, cid in zip(ranking["formula"], ranking["config_id"])
    ]

    shortlist = ranking[
        ranking[["is_top_n", "is_old_config", "is_frozen_final", "is_full_data_recommendation"]].any(axis=1)
    ].copy()
    shortlist = shortlist.sort_values(["formula", "rank_within_formula", "w_r", "w_e", "c"])
    shortlist.to_csv(out / "tuning_presentation_shortlist_h10.csv", index=False)

    fold_cols = [
        c for c in [
            "outer_fold", "formula", "config_id", "regime", "w_r", "w_e", "c",
            "training_latency_gain",
        ] if c in folds.columns
    ]
    folds[fold_cols].to_csv(out / "tuning_fold_selected_configurations.csv", index=False)

    # Markdown presentation package.
    show = shortlist[compact_metric_columns(shortlist) + [
        "is_old_config", "is_frozen_final", "is_full_data_recommendation"
    ]]
    lines = [
        "# Transfer Learning parameter tuning — H=10",
        "",
        "## Tested parameter space",
        "",
        tested_space.to_markdown(index=False),
        "",
        "## Presentation shortlist",
        "",
        show.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- The complete CSV contains every configuration that was actually tested.",
        "- The slide shortlist contains top-N rows plus the historical and frozen-final configurations.",
        "- `latency_gain` is the tuning-study score and must be interpreted with the baseline used by that study.",
        "- The later matched-c baseline replay is the confirmatory comparison used to isolate TL from the effect of changing c.",
        "",
        "## Frozen final configurations",
        "",
        "- Difference: w_R=0.05, w_E=64, c=0.2.",
        "- Ratio: w_R=0.01, w_E=64, c=0.4.",
        "- Historical comparison: w_R=0.25, w_E=1, c=0.8.",
    ]
    (out / "TUNING_PRESENTATION_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "source_tuning_dir": str(tuning),
        "primary_horizon": manifest.get("selection", {}).get("primary_horizon", 10),
        "parameter_grid": manifest.get("parameter_grid", {}),
        "note_baseline_c": (
            "Older tuning manifests may not record baseline_c. transfer_parameter_tuning_cv.py default is 0.8; "
            "verify the exact historical invocation/log before claiming it for a saved run."
        ),
        "old_configs": OLD,
        "frozen_final_configs": FINAL,
        "top_n": args.top_n,
    }
    (out / "tuning_table_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    print("PASS — tuning tables exported")
    print("full=", out / "tuning_all_tested_configurations_h10.csv")
    print("space=", out / "tuning_tested_parameter_space.csv")
    print("shortlist=", out / "tuning_presentation_shortlist_h10.csv")
    print("report=", out / "TUNING_PRESENTATION_TABLES.md")


if __name__ == "__main__":
    main()
