#!/usr/bin/env python3
"""
Presentation-ready transfer-learning analysis for Serverledge/randomaccess.

Generates figures with presentation labels:
  - No Transfer
  - Original Transfer   (internal implementation: coupled)
  - Improved Transfer   (internal implementation: decoupled)

The "decoupled" implementation name is intentionally kept out of the
presentation figures. It remains only an internal/code-level identifier.

Main comparison:
  R01-R03, c=0.8

Exploration ablation:
  R02-R03, Improved Transfer only, c in {0.0, 0.8}

Metrics:
  - cumulative execution duration H5/H10/H20/H50
  - best-arm usage/share
  - suboptimal-arm pulls
  - observed execution time on suboptimal arm
  - first best-arm request
  - first dominant W-request window
  - stable-dominance proxy
  - last suboptimal request
  - longest best-arm streak
  - cumulative suboptimal pulls over request index
  - cumulative suboptimal execution time over request index

"Stable dominance" is a convergence proxy, not mathematical convergence:
it is the earliest request from which every remaining full sliding window
of size W contains at least the requested fraction of executions on the
independently established best arm.

Outputs both PNG and SVG, plus a Markdown summary table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEST_ARM_DEFAULT = "arm64"
HORIZONS = [5, 10, 20, 50]

DISPLAY = {
    "no_transfer": "No Transfer",
    "coupled": "Original Transfer",
    "decoupled": "Improved Transfer",
}


@dataclass(frozen=True)
class Run:
    replica: str
    variant: str
    run_dir: str


MAIN_RUNS = (
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


ABLATION_RUNS = (
    Run("R02", "improved_c0",
        "gcp-transfer-materialized-randomaccess-decoupled-c00-20260922_174716"),
    Run("R02", "improved_c08",
        "gcp-transfer-materialized-randomaccess-decoupled-c08-20260922_174404"),
    Run("R03", "improved_c0",
        "gcp-transfer-materialized-randomaccess-decoupled-c00-20260923_080510"),
    Run("R03", "improved_c08",
        "gcp-transfer-materialized-randomaccess-decoupled-c08-20260923_080146"),
)


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_mab_requests(path: Path) -> pd.DataFrame:
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
        raise ValueError(f"{path}: colonne mancanti: {sorted(missing)}")

    df = df.sort_values("request_index").reset_index(drop=True)

    expected = list(range(1, len(df) + 1))
    if list(df["request_index"].astype(int)) != expected:
        raise ValueError(f"{path}: request_index non consecutivi")

    return df


def first_dominant_window_end(
    best_flags: list[bool],
    window: int,
    threshold: float,
) -> int | None:
    if len(best_flags) < window:
        return None
    need = int(np.ceil(window * threshold))
    for end in range(window, len(best_flags) + 1):
        if sum(best_flags[end-window:end]) >= need:
            return end
    return None


def stable_dominance_start(
    best_flags: list[bool],
    window: int,
    threshold: float,
) -> int | None:
    """
    Earliest 1-based request index s such that every full sliding window
    starting at s or later has >= threshold best-arm executions.
    """
    n = len(best_flags)
    if n < window:
        return None

    need = int(np.ceil(window * threshold))
    last_start = n - window + 1

    for start in range(1, last_start + 1):
        ok = True
        for wstart in range(start, last_start + 1):
            chunk = best_flags[wstart-1:wstart-1+window]
            if sum(chunk) < need:
                ok = False
                break
        if ok:
            return start

    return None


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


def analyze_main(
    root: Path,
    best_arm: str,
    window: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trajectories = []
    missing = []

    for run in MAIN_RUNS:
        path = root / run.run_dir / "measure" / "mab-requests.csv"
        if not path.is_file():
            missing.append(path)
            continue

        df = load_mab_requests(path)

        best = df["execution_arm"].eq(best_arm)
        sub = ~best

        first_best = (
            int(df.loc[best, "request_index"].iloc[0])
            if best.any() else None
        )
        last_sub = (
            int(df.loc[sub, "request_index"].iloc[-1])
            if sub.any() else None
        )

        first_dom_end = first_dominant_window_end(
            best.tolist(), window, threshold
        )
        stable_start = stable_dominance_start(
            best.tolist(), window, threshold
        )
        stable_end = (
            stable_start + window - 1
            if stable_start is not None else None
        )
        stable_time_s = (
            float(df.loc[df["request_index"] <= stable_end,
                         "duration_ms"].sum() / 1000.0)
            if stable_end is not None else None
        )

        rows.append({
            "replica": run.replica,
            "variant": run.variant,
            "presentation_label": DISPLAY[run.variant],
            "best_arm": best_arm,
            "request_count": len(df),
            "best_arm_executions": int(best.sum()),
            "best_arm_share_pct": float(best.mean() * 100.0),
            "suboptimal_arm_executions": int(sub.sum()),
            "suboptimal_arm_share_pct": float(sub.mean() * 100.0),
            "suboptimal_arm_duration_s":
                float(df.loc[sub, "duration_ms"].sum() / 1000.0),
            "first_best_arm_request": first_best,
            "first_dominant_window_end_request": first_dom_end,
            "stable_dominance_start_request": stable_start,
            "stable_dominance_end_request": stable_end,
            "stable_dominance_execution_time_s": stable_time_s,
            "last_suboptimal_request": last_sub,
            "longest_best_arm_streak":
                longest_true_streak(best.tolist()),
            "fallback_count":
                int(df["fallback"].astype(str).str.lower().eq("true").sum()),
            "all_warm":
                bool(df["warm_start"].astype(str).str.lower().eq("true").all()),
            "source_run": run.run_dir,
        })

        n = len(df)
        trajectories.append(pd.DataFrame({
            "replica": run.replica,
            "variant": run.variant,
            "request_index": df["request_index"].astype(int),
            "cumulative_best_arm_share_pct":
                best.astype(int).cumsum().to_numpy()
                / np.arange(1, n + 1) * 100.0,
            "cumulative_suboptimal_pulls":
                sub.astype(int).cumsum().to_numpy(),
            "cumulative_suboptimal_duration_s":
                (df["duration_ms"].where(sub, 0.0).cumsum()
                 / 1000.0).to_numpy(),
        }))

    if missing:
        print("ERROR — file mancanti:")
        for p in missing:
            print(f"  {p}")
        raise SystemExit(
            f"STOP — {len(missing)} run main mancanti; "
            "non genero risultati parziali."
        )

    return (
        pd.DataFrame(rows),
        pd.concat(trajectories, ignore_index=True),
    )


def analyze_ablation(root: Path) -> pd.DataFrame:
    rows = []
    missing = []

    for run in ABLATION_RUNS:
        path = root / run.run_dir / "measure" / "summary.json"
        if not path.is_file():
            missing.append(path)
            continue

        import json
        with path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

        h50 = summary["horizons"]["50"]
        rows.append({
            "replica": run.replica,
            "variant": run.variant,
            "c": 0.0 if run.variant == "improved_c0" else 0.8,
            "cumulative_duration_s":
                float(h50["cumulative_duration_ms"]) / 1000.0,
            "arm64_executions":
                int(h50.get("arm64_executions", 0)),
            "amd64_executions":
                int(h50.get("amd64_executions", 0)),
        })

    if missing:
        print("ERROR — file ablation mancanti:")
        for p in missing:
            print(f"  {p}")
        raise SystemExit(
            f"STOP — {len(missing)} run ablation mancanti."
        )

    return pd.DataFrame(rows)


def plot_main_cumulative(
    aggregate_csv: Path,
    out_dir: Path,
) -> None:
    df = pd.read_csv(aggregate_csv)
    main = df[df["campaign"] == "transfer_main"].copy()

    x = np.arange(len(HORIZONS))
    xlabels = [f"H{h}" for h in HORIZONS]

    # Complete comparison
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in ("no_transfer", "coupled", "decoupled"):
        d = (
            main[main["variant"] == variant]
            .set_index("horizon_requests")
            .loc[HORIZONS]
        )
        ax.errorbar(
            x,
            d["cumulative_duration_mean_s"].to_numpy(),
            yerr=d["cumulative_duration_stdev_ms"].to_numpy()/1000.0,
            marker="o",
            linewidth=2,
            capsize=4,
            label=DISPLAY[variant],
        )

    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Orizzonte di richieste")
    ax.set_ylabel("Durata cumulativa di esecuzione (s)")
    ax.set_title("Transfer learning: confronto completo")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.01, -0.17,
        "Media su R01–R03; barre = deviazione standard; c = 0.8.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "01_transfer_confronto_completo")

    # Baseline vs Improved Transfer
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in ("no_transfer", "decoupled"):
        d = (
            main[main["variant"] == variant]
            .set_index("horizon_requests")
            .loc[HORIZONS]
        )
        ax.errorbar(
            x,
            d["cumulative_duration_mean_s"].to_numpy(),
            yerr=d["cumulative_duration_stdev_ms"].to_numpy()/1000.0,
            marker="o",
            linewidth=2,
            capsize=4,
            label=DISPLAY[variant],
        )

    improved = (
        main[main["variant"] == "decoupled"]
        .set_index("horizon_requests")
        .loc[HORIZONS]
    )

    offsets = [(0, 14), (0, 14), (0, -26), (0, -26)]
    for i, h in enumerate(HORIZONS):
        gain = float(improved.loc[h, "gain_vs_baseline_mean_pct"])
        y = float(improved.loc[h, "cumulative_duration_mean_s"])
        label = (
            f"−{gain:.2f}% costo"
            if gain >= 0
            else f"+{abs(gain):.2f}% costo"
        )
        ax.annotate(
            label,
            (x[i], y),
            xytext=offsets[i],
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x, xlabels)
    ax.set_xlabel("Orizzonte di richieste")
    ax.set_ylabel("Durata cumulativa di esecuzione (s)")
    ax.set_title("Beneficio del transfer learning")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.01, -0.19,
        "No Transfer vs Improved Transfer. "
        "Metrica: cumulative execution duration; media R01–R03; c = 0.8.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "02_baseline_vs_improved_cumulative_duration")


def plot_ablation(
    ablation: pd.DataFrame,
    out_dir: Path,
) -> None:
    grouped = (
        ablation.groupby("c")
        .agg(
            cumulative_mean_s=("cumulative_duration_s", "mean"),
            cumulative_std_s=("cumulative_duration_s", "std"),
            arm64_mean=("arm64_executions", "mean"),
        )
        .loc[[0.0, 0.8]]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(2)
    bars = ax.bar(
        x,
        grouped["cumulative_mean_s"].to_numpy(),
        yerr=grouped["cumulative_std_s"].fillna(0).to_numpy(),
        capsize=4,
    )

    labels = ["Improved Transfer\nc=0", "Improved Transfer\nc=0.8"]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Durata cumulativa H50 (s)")
    ax.set_title("Ablation sull'esplorazione")
    ax.grid(True, axis="y", alpha=0.25)

    max_y = float(
        (grouped["cumulative_mean_s"]
         + grouped["cumulative_std_s"].fillna(0)).max()
    )
    ax.set_ylim(0, max_y + 10)

    for bar, c in zip(bars, [0.0, 0.8]):
        mean_s = float(grouped.loc[c, "cumulative_mean_s"])
        std_s = float(grouped.loc[c, "cumulative_std_s"])
        arm = float(grouped.loc[c, "arm64_mean"])
        ax.text(
            bar.get_x() + bar.get_width()/2,
            mean_s + (0 if np.isnan(std_s) else std_s) + 1.0,
            f"{mean_s:.2f} s\nARM {arm:.1f}/50",
            ha="center",
            va="bottom",
        )

    ax.text(
        0.01, -0.17,
        "Media su R02–R03. c=0 rimuove il termine di esplorazione; "
        "c=0.8 lo mantiene.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "03_exploration_ablation_improved_only")


def plot_dynamics(
    metrics: pd.DataFrame,
    trajectories: pd.DataFrame,
    out_dir: Path,
    best_arm: str,
    threshold: float,
) -> None:
    keep = ("no_transfer", "decoupled")

    # 04 cumulative best-arm share
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in keep:
        d = trajectories[trajectories["variant"] == variant]
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

        ax.plot(x, mean, linewidth=2, label=DISPLAY[variant])
        ax.fill_between(x, mean-std, mean+std, alpha=0.12)

    ax.axhline(
        threshold*100.0,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_ylim(0, 102)
    ax.set_xlabel("Indice della richiesta")
    ax.set_ylabel(f"Quota cumulativa su {best_arm} (%)")
    ax.set_title("Velocità di adattamento verso l'architettura migliore")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.01, -0.17,
        "No Transfer vs Improved Transfer; media R01–R03; "
        "area = ±1 deviazione standard.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "04_best_arm_share_baseline_vs_improved")

    # 05 final suboptimal pulls
    d = (
        metrics[metrics["variant"].isin(keep)]
        .groupby("variant")["suboptimal_arm_executions"]
        .agg(["mean", "std"])
        .reindex(keep)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bars = ax.bar(
        np.arange(2),
        d["mean"].to_numpy(),
        yerr=d["std"].fillna(0).to_numpy(),
        capsize=4,
    )
    ax.set_xticks(np.arange(2), [DISPLAY[v] for v in keep])
    ax.set_ylabel(f"Esecuzioni sul braccio non-{best_arm} (su 50)")
    ax.set_title("Esposizione al braccio subottimale")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, d["mean"]):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.12,
            f"{value:.1f}/50",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save(fig, out_dir, "05_suboptimal_pulls_baseline_vs_improved")

    # 06 final suboptimal time
    d = (
        metrics[metrics["variant"].isin(keep)]
        .groupby("variant")["suboptimal_arm_duration_s"]
        .agg(["mean", "std"])
        .reindex(keep)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bars = ax.bar(
        np.arange(2),
        d["mean"].to_numpy(),
        yerr=d["std"].fillna(0).to_numpy(),
        capsize=4,
    )
    ax.set_xticks(np.arange(2), [DISPLAY[v] for v in keep])
    ax.set_ylabel("Tempo sul braccio subottimale (s)")
    ax.set_title("Tempo speso sul braccio subottimale")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, d["mean"]):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.08,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save(fig, out_dir, "06_suboptimal_time_baseline_vs_improved")

    # 07 cumulative suboptimal pulls
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in keep:
        d = trajectories[trajectories["variant"] == variant]
        agg = (
            d.groupby("request_index")["cumulative_suboptimal_pulls"]
            .agg(["mean", "std"])
            .reset_index()
        )
        x = agg["request_index"].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].fillna(0).to_numpy()
        ax.plot(x, mean, linewidth=2, label=DISPLAY[variant])
        ax.fill_between(x, mean-std, mean+std, alpha=0.12)

    ax.set_xlabel("Indice della richiesta")
    ax.set_ylabel("Pull cumulativi del braccio subottimale")
    ax.set_title("Accumulo delle selezioni subottimali")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "07_cumulative_suboptimal_pulls")

    # 08 cumulative suboptimal execution time
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in keep:
        d = trajectories[trajectories["variant"] == variant]
        agg = (
            d.groupby("request_index")[
                "cumulative_suboptimal_duration_s"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        x = agg["request_index"].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].fillna(0).to_numpy()
        ax.plot(x, mean, linewidth=2, label=DISPLAY[variant])
        ax.fill_between(x, mean-std, mean+std, alpha=0.12)

    ax.set_xlabel("Indice della richiesta")
    ax.set_ylabel("Tempo cumulativo sul braccio subottimale (s)")
    ax.set_title("Costo cumulativo delle selezioni subottimali")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "08_cumulative_suboptimal_time")


def write_summary_table(
    aggregate_csv: Path,
    metrics: pd.DataFrame,
    out_dir: Path,
) -> None:
    agg = pd.read_csv(aggregate_csv)
    main = agg[agg["campaign"] == "transfer_main"]

    base = (
        main[main["variant"] == "no_transfer"]
        .set_index("horizon_requests")
    )
    improved = (
        main[main["variant"] == "decoupled"]
        .set_index("horizon_requests")
    )

    dyn = (
        metrics[metrics["variant"].isin(("no_transfer", "decoupled"))]
        .groupby("variant")
        .mean(numeric_only=True)
    )

    rows = []

    for h in HORIZONS:
        rows.append({
            "Metrica": f"Durata cumulativa H{h}",
            "No Transfer":
                f'{float(base.loc[h, "cumulative_duration_mean_s"]):.3f} s',
            "Improved Transfer":
                f'{float(improved.loc[h, "cumulative_duration_mean_s"]):.3f} s',
            "Differenza":
                f'−{float(improved.loc[h, "gain_vs_baseline_mean_pct"]):.2f}% costo',
        })

    b = dyn.loc["no_transfer"]
    i = dyn.loc["decoupled"]

    rows.extend([
        {
            "Metrica": "Quota architettura migliore",
            "No Transfer": f'{b["best_arm_share_pct"]:.1f}%',
            "Improved Transfer": f'{i["best_arm_share_pct"]:.1f}%',
            "Differenza":
                f'+{i["best_arm_share_pct"]-b["best_arm_share_pct"]:.1f} pp',
        },
        {
            "Metrica": "Esecuzioni sul braccio subottimale",
            "No Transfer": f'{b["suboptimal_arm_executions"]:.1f}/50',
            "Improved Transfer": f'{i["suboptimal_arm_executions"]:.1f}/50',
            "Differenza":
                f'{(i["suboptimal_arm_executions"]/b["suboptimal_arm_executions"]-1)*100:.1f}%',
        },
        {
            "Metrica": "Tempo sul braccio subottimale",
            "No Transfer": f'{b["suboptimal_arm_duration_s"]:.3f} s',
            "Improved Transfer": f'{i["suboptimal_arm_duration_s"]:.3f} s',
            "Differenza":
                f'{(i["suboptimal_arm_duration_s"]/b["suboptimal_arm_duration_s"]-1)*100:.1f}%',
        },
        {
            "Metrica": "Prima finestra dominante (fine)",
            "No Transfer":
                f'richiesta {b["first_dominant_window_end_request"]:.1f}',
            "Improved Transfer":
                f'richiesta {i["first_dominant_window_end_request"]:.1f}',
            "Differenza":
                f'{i["first_dominant_window_end_request"]-b["first_dominant_window_end_request"]:+.1f} richieste',
        },
        {
            "Metrica": "Dominanza stabile (inizio)",
            "No Transfer":
                f'richiesta {b["stable_dominance_start_request"]:.1f}',
            "Improved Transfer":
                f'richiesta {i["stable_dominance_start_request"]:.1f}',
            "Differenza":
                f'{i["stable_dominance_start_request"]-b["stable_dominance_start_request"]:+.1f} richieste',
        },
        {
            "Metrica": "Tempo fino alla dominanza stabile",
            "No Transfer":
                f'{b["stable_dominance_execution_time_s"]:.3f} s',
            "Improved Transfer":
                f'{i["stable_dominance_execution_time_s"]:.3f} s',
            "Differenza":
                f'{i["stable_dominance_execution_time_s"]-b["stable_dominance_execution_time_s"]:+.3f} s',
        },
        {
            "Metrica": "Ultima esecuzione subottimale",
            "No Transfer":
                f'richiesta {b["last_suboptimal_request"]:.1f}',
            "Improved Transfer":
                f'richiesta {i["last_suboptimal_request"]:.1f}',
            "Differenza":
                f'{i["last_suboptimal_request"]-b["last_suboptimal_request"]:+.1f} richieste',
        },
        {
            "Metrica": "Serie massima sul braccio migliore",
            "No Transfer":
                f'{b["longest_best_arm_streak"]:.1f}',
            "Improved Transfer":
                f'{i["longest_best_arm_streak"]:.1f}',
            "Differenza":
                f'{i["longest_best_arm_streak"]-b["longest_best_arm_streak"]:+.1f}',
        },
    ])

    table_df = pd.DataFrame(rows)

    md_path = out_dir / "09_transfer_summary_table.md"
    md_path.write_text(
        table_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    fig_height = 0.55 * len(table_df) + 1.7
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title(
        "Transfer learning — sintesi No Transfer vs Improved Transfer",
        pad=16,
    )
    fig.tight_layout()
    save(fig, out_dir, "09_transfer_summary_table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/profiling/gcp-transfer-final"),
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/analysis/"
            "transfer_learning_results_aggregate.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/analysis/"
            "presentation_graphs"
        ),
    )
    parser.add_argument(
        "--best-arm",
        choices=("amd64", "arm64"),
        default=BEST_ARM_DEFAULT,
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.90,
    )
    args = parser.parse_args()

    if args.window <= 0:
        raise SystemExit("STOP — window deve essere > 0")
    if not 0 < args.dominance_threshold <= 1:
        raise SystemExit(
            "STOP — dominance-threshold deve essere in (0,1]"
        )

    root = args.root.resolve()
    aggregate_csv = args.aggregate_csv.resolve()
    out_dir = args.output_dir.resolve()

    if not root.is_dir():
        raise SystemExit(f"STOP — root non trovata: {root}")
    if not aggregate_csv.is_file():
        raise SystemExit(
            f"STOP — aggregate CSV non trovato: {aggregate_csv}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, trajectories = analyze_main(
        root,
        args.best_arm,
        args.window,
        args.dominance_threshold,
    )
    ablation = analyze_ablation(root)

    # Keep detailed data for reproducibility, but presentation uses figures/table.
    metrics.to_csv(
        out_dir / "presentation_transfer_metrics.csv",
        index=False,
    )
    trajectories.to_csv(
        out_dir / "presentation_transfer_trajectories.csv",
        index=False,
    )

    plot_main_cumulative(aggregate_csv, out_dir)
    plot_ablation(ablation, out_dir)
    plot_dynamics(
        metrics,
        trajectories,
        out_dir,
        args.best_arm,
        args.dominance_threshold,
    )
    write_summary_table(
        aggregate_csv,
        metrics,
        out_dir,
    )

    print("PASS — presentation analysis generated")
    print(f"best_arm={args.best_arm}")
    print(
        f"dominance_window={args.window}, "
        f"threshold={args.dominance_threshold:.2f}"
    )
    print("\n===== OUTPUT DIR =====")
    print(out_dir)

    print("\n===== FIGURE =====")
    for p in sorted(out_dir.glob("*.png")):
        print(p)

    print("\n===== TABLE =====")
    print(out_dir / "09_transfer_summary_table.md")


if __name__ == "__main__":
    main()
