#!/usr/bin/env python3
"""
Additional dynamic metrics for the Serverledge transfer-learning campaign.

This script analyzes the MAIN transfer campaign only:
    No Transfer vs Coupled vs Decoupled, R01-R03, c=0.8.

It reads measure/mab-requests.csv from each run and computes metrics that
complement cumulative execution duration:

- best-arm usage share;
- number of executions on the suboptimal arm;
- total execution time spent on the suboptimal arm;
- first request executed on the best arm;
- first W-request window in which the best arm reaches a chosen dominance
  threshold;
- last suboptimal-arm execution;
- longest consecutive best-arm streak.

IMPORTANT:
"suboptimal arm" is defined relative to an independently established best
architecture. For randomaccess the current experiment uses arm64 as best arm.
This can be overridden with --best-arm.

The "dominant window" metric is intentionally NOT called true convergence:
UCB keeps exploring, so it need not permanently stop selecting other arms.

Outputs:
  transfer_dynamics_metrics.csv
  04_best_arm_share_over_requests.png/.svg
  05_suboptimal_arm_pulls.png/.svg
  06_suboptimal_arm_execution_time.png/.svg
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Run:
    replica: str
    variant: str
    run_dir: str


RUNS = (
    Run("R01", "no_transfer",
        "gcp-transfer-final-randomaccess-no-transfer-20260920_154452"),
    Run("R01", "coupled",
        "gcp-transfer-final-randomaccess-coupled-20260920_154804"),
    Run("R01", "decoupled",
        "gcp-transfer-final-randomaccess-decoupled-20260920_155555"),

    Run("R02", "no_transfer",
        "gcp-transfer-final-randomaccess-no-transfer-20260920_161356"),
    Run("R02", "coupled",
        "gcp-transfer-final-randomaccess-coupled-20260920_161638"),
    Run("R02", "decoupled",
        "gcp-transfer-final-randomaccess-decoupled-20260920_160825"),

    Run("R03", "no_transfer",
        "gcp-transfer-final-randomaccess-no-transfer-20260920_163626"),
    Run("R03", "coupled",
        "gcp-transfer-final-randomaccess-coupled-20260920_162514"),
    Run("R03", "decoupled",
        "gcp-transfer-final-randomaccess-decoupled-20260920_163051"),
)

LABELS = {
    "no_transfer": "No Transfer",
    "coupled": "Coupled",
    "decoupled": "Decoupled",
}


def longest_true_streak(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def first_dominant_window_end(
    best_flags: list[bool],
    window: int,
    threshold: float,
) -> int | None:
    if len(best_flags) < window:
        return None

    need = int(np.ceil(window * threshold))
    for end in range(window, len(best_flags) + 1):
        current = best_flags[end-window:end]
        if sum(current) >= need:
            return end
    return None


def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "request_index",
        "selected_arm",
        "execution_arm",
        "fallback",
        "duration_ms",
        "reward",
        "warm_start",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path}: colonne mancanti: {sorted(missing)}"
        )

    df = df.sort_values("request_index").reset_index(drop=True)

    if list(df["request_index"]) != list(range(1, len(df) + 1)):
        raise ValueError(f"{path}: request_index non consecutivi")

    return df


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/profiling/gcp-transfer-final"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/analysis/graphs"
        ),
    )
    parser.add_argument(
        "--best-arm",
        choices=("amd64", "arm64"),
        default="arm64",
        help=(
            "Best architecture established independently from the "
            "online MAB run. Default: arm64."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Window size for dominant-arm identification. Default: 10.",
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.9,
        help=(
            "Required best-arm share inside the window. "
            "Default: 0.9."
        ),
    )
    args = parser.parse_args()

    if not (0 < args.dominance_threshold <= 1):
        raise SystemExit("STOP — dominance-threshold deve essere in (0,1].")
    if args.window <= 0:
        raise SystemExit("STOP — window deve essere > 0.")

    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    trajectories: list[pd.DataFrame] = []
    missing: list[Path] = []

    for run in RUNS:
        path = root / run.run_dir / "measure" / "mab-requests.csv"
        if not path.is_file():
            missing.append(path)
            continue

        df = load_run(path)
        best = df["execution_arm"].eq(args.best_arm)
        suboptimal = ~best

        cumulative_best_share = (
            best.astype(int).cumsum()
            / np.arange(1, len(df) + 1)
            * 100.0
        )

        first_best = (
            int(df.loc[best, "request_index"].iloc[0])
            if best.any() else None
        )
        last_suboptimal = (
            int(df.loc[suboptimal, "request_index"].iloc[-1])
            if suboptimal.any() else None
        )
        dominant_end = first_dominant_window_end(
            best.tolist(),
            args.window,
            args.dominance_threshold,
        )

        metrics.append({
            "replica": run.replica,
            "variant": run.variant,
            "best_arm": args.best_arm,
            "request_count": len(df),
            "best_arm_executions": int(best.sum()),
            "best_arm_share_pct": float(best.mean() * 100.0),
            "suboptimal_arm_executions": int(suboptimal.sum()),
            "suboptimal_arm_share_pct": float(suboptimal.mean() * 100.0),
            "suboptimal_arm_duration_ms":
                float(df.loc[suboptimal, "duration_ms"].sum()),
            "suboptimal_arm_duration_s":
                float(df.loc[suboptimal, "duration_ms"].sum() / 1000.0),
            "first_best_arm_request": first_best,
            "dominant_window_size": args.window,
            "dominance_threshold_pct":
                args.dominance_threshold * 100.0,
            "first_dominant_window_end_request": dominant_end,
            "last_suboptimal_request": last_suboptimal,
            "longest_best_arm_streak":
                longest_true_streak(best.tolist()),
            "fallback_count":
                int(df["fallback"].astype(str).str.lower().eq("true").sum()),
            "all_warm":
                bool(df["warm_start"].astype(str).str.lower().eq("true").all()),
            "source_run": run.run_dir,
        })

        t = pd.DataFrame({
            "replica": run.replica,
            "variant": run.variant,
            "request_index": df["request_index"].astype(int),
            "cumulative_best_arm_share_pct": cumulative_best_share,
        })
        trajectories.append(t)

    if missing:
        print("ERROR — file mancanti:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(
            f"STOP — {len(missing)} run mancanti; "
            "non genero risultati parziali."
        )

    metrics_df = pd.DataFrame(metrics)
    traj_df = pd.concat(trajectories, ignore_index=True)

    metrics_path = out / "transfer_dynamics_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ------------------------------------------------------------
    # Figure 04: cumulative best-arm share over request index
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    for variant in ("no_transfer", "coupled", "decoupled"):
        d = traj_df[traj_df["variant"] == variant]
        agg = (
            d.groupby("request_index")[
                "cumulative_best_arm_share_pct"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = agg["request_index"].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].fillna(0).to_numpy()

        ax.plot(
            x,
            mean,
            linewidth=2,
            label=LABELS[variant],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12,
        )

    ax.axhline(
        args.dominance_threshold * 100.0,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_ylim(0, 102)
    ax.set_xlabel("Indice della richiesta")
    ax.set_ylabel(
        f"Quota cumulativa su {args.best_arm} (%)"
    )
    ax.set_title(
        "Velocità di adattamento verso l'architettura migliore"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.01, -0.17,
        "Media su R01–R03; area = ±1 deviazione standard. "
        "La quota cumulativa mostra quanto rapidamente il MAB "
        "privilegia il braccio migliore.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "04_best_arm_share_over_requests")

    # ------------------------------------------------------------
    # Figure 05: suboptimal arm executions
    # ------------------------------------------------------------
    grouped = (
        metrics_df.groupby("variant")[
            "suboptimal_arm_executions"
        ]
        .agg(["mean", "std"])
        .reindex(["no_transfer", "coupled", "decoupled"])
    )

    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    bars = ax.bar(
        np.arange(3),
        grouped["mean"].to_numpy(),
        yerr=grouped["std"].fillna(0).to_numpy(),
        capsize=4,
    )
    ax.set_xticks(
        np.arange(3),
        [LABELS[v] for v in grouped.index],
    )
    ax.set_ylabel(
        f"Esecuzioni sul braccio non-{args.best_arm} (su 50)"
    )
    ax.set_title("Esposizione al braccio subottimale")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, grouped["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{value:.1f}/50",
            ha="center",
            va="bottom",
        )

    ax.text(
        0.01, -0.16,
        "Media su R01–R03; barre = deviazione standard.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "05_suboptimal_arm_pulls")

    # ------------------------------------------------------------
    # Figure 06: time spent executing on suboptimal arm
    # ------------------------------------------------------------
    grouped_time = (
        metrics_df.groupby("variant")[
            "suboptimal_arm_duration_s"
        ]
        .agg(["mean", "std"])
        .reindex(["no_transfer", "coupled", "decoupled"])
    )

    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    bars = ax.bar(
        np.arange(3),
        grouped_time["mean"].to_numpy(),
        yerr=grouped_time["std"].fillna(0).to_numpy(),
        capsize=4,
    )
    ax.set_xticks(
        np.arange(3),
        [LABELS[v] for v in grouped_time.index],
    )
    ax.set_ylabel("Tempo sul braccio subottimale (s)")
    ax.set_title(
        "Tempo di esecuzione speso sul braccio subottimale"
    )
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, grouped_time["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
        )

    ax.text(
        0.01, -0.16,
        "Non è un regret controfattuale: è il tempo realmente "
        "osservato nelle esecuzioni sul braccio subottimale.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out, "06_suboptimal_arm_execution_time")

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------
    print(f"PASS — run analizzati: {len(metrics_df)}")
    print(f"best_arm={args.best_arm}")
    print(
        f"dominance_window={args.window}, "
        f"threshold={args.dominance_threshold:.2f}"
    )

    print("\n===== METRICHE MEDIE R01-R03 =====")
    summary = (
        metrics_df.groupby("variant")[
            [
                "best_arm_share_pct",
                "suboptimal_arm_executions",
                "suboptimal_arm_duration_s",
                "first_best_arm_request",
                "first_dominant_window_end_request",
                "last_suboptimal_request",
                "longest_best_arm_streak",
            ]
        ]
        .mean()
        .reindex(["no_transfer", "coupled", "decoupled"])
    )
    print(summary.to_string())

    print("\n===== OUTPUT =====")
    print(metrics_path)
    for stem in (
        "04_best_arm_share_over_requests",
        "05_suboptimal_arm_pulls",
        "06_suboptimal_arm_execution_time",
    ):
        print(out / f"{stem}.png")
        print(out / f"{stem}.svg")


if __name__ == "__main__":
    main()
