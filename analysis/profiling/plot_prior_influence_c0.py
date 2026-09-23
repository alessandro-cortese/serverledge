#!/usr/bin/env python3
"""
Generate presentation-ready figures for the c=0 control experiment.

Purpose
-------
Compare:
  - No Transfer, c=0
  - Improved Transfer, c=0

This isolates the effect of transferred prior knowledge when the explicit
UCB exploration bonus is disabled.

The script auto-discovers completed runs under:
  data/profiling/gcp-transfer-final

Expected completed run patterns:
  gcp-transfer-control-randomaccess-no-transfer-c00-*
  gcp-transfer-materialized-randomaccess-decoupled-c00-*

Incomplete/failed directories without measure/summary.json and
measure/mab-requests.csv are ignored.

Outputs:
  10_prior_influence_arm_usage.png/.svg
  11_prior_influence_trajectory.png/.svg
  12_prior_influence_h50_duration.png/.svg
  13_prior_influence_summary_table.png/.svg
  13_prior_influence_summary_table.md
  prior_influence_c0_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "no_transfer": "No Transfer",
    "improved": "Improved Transfer",
}


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def completed_runs(root: Path, pattern: str) -> list[Path]:
    runs = []
    for d in sorted(root.glob(pattern)):
        if (
            (d / "measure" / "summary.json").is_file()
            and (d / "measure" / "mab-requests.csv").is_file()
        ):
            runs.append(d)
    return runs


def load_run(run_dir: Path, variant: str) -> tuple[dict, pd.DataFrame]:
    with (run_dir / "measure" / "summary.json").open(
        "r", encoding="utf-8"
    ) as f:
        summary = json.load(f)

    req = pd.read_csv(run_dir / "measure" / "mab-requests.csv")
    req = req.sort_values("request_index").reset_index(drop=True)

    required = {
        "request_index",
        "execution_arm",
        "duration_ms",
    }
    missing = required.difference(req.columns)
    if missing:
        raise SystemExit(
            f"STOP — {run_dir}: colonne mancanti {sorted(missing)}"
        )

    if len(req) != 50:
        raise SystemExit(
            f"STOP — {run_dir}: attese 50 richieste, trovate {len(req)}"
        )

    h50 = summary["horizons"]["50"]

    metrics = {
        "variant": variant,
        "label": LABELS[variant],
        "run_dir": run_dir.name,
        "amd64_executions": int(h50.get("amd64_executions", 0)),
        "arm64_executions": int(h50.get("arm64_executions", 0)),
        "cumulative_duration_s":
            float(h50["cumulative_duration_ms"]) / 1000.0,
        "mean_duration_ms": float(h50["mean_duration_ms"]),
        "median_duration_ms": float(h50["median_duration_ms"]),
        "all_warm": bool(summary.get("all_warm", False)),
        "fallback_count": int(summary.get("fallback_count", 0)),
    }

    return metrics, req


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
            "data/profiling/gcp-transfer-final/analysis/"
            "presentation_graphs"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    no_transfer_dirs = completed_runs(
        root,
        "gcp-transfer-control-randomaccess-no-transfer-c00-*",
    )
    improved_dirs = completed_runs(
        root,
        "gcp-transfer-materialized-randomaccess-decoupled-c00-*",
    )

    if len(no_transfer_dirs) < 2:
        raise SystemExit(
            "STOP — servono almeno 2 run completi No Transfer c=0; "
            f"trovati {len(no_transfer_dirs)}"
        )

    if len(improved_dirs) < 2:
        raise SystemExit(
            "STOP — servono almeno 2 run completi Improved Transfer c=0; "
            f"trovati {len(improved_dirs)}"
        )

    # Use all completed runs if there are exactly the intended repetitions.
    # If older completed repetitions exist, stop instead of silently mixing them.
    if len(no_transfer_dirs) != 2:
        raise SystemExit(
            "STOP — trovati più di 2 run completi No Transfer c=0. "
            "Specificare/filtrare i run prima di aggregare:\n"
            + "\n".join(str(p) for p in no_transfer_dirs)
        )

    if len(improved_dirs) != 2:
        raise SystemExit(
            "STOP — trovati più di 2 run completi Improved Transfer c=0. "
            "Specificare/filtrare i run prima di aggregare:\n"
            + "\n".join(str(p) for p in improved_dirs)
        )

    rows = []
    trajectories = []

    for variant, dirs in (
        ("no_transfer", no_transfer_dirs),
        ("improved", improved_dirs),
    ):
        for idx, run_dir in enumerate(dirs, start=1):
            metrics, req = load_run(run_dir, variant)
            metrics["replicate"] = idx
            rows.append(metrics)

            amd64 = req["execution_arm"].eq("amd64").astype(int)
            trajectories.append(
                pd.DataFrame({
                    "variant": variant,
                    "replicate": idx,
                    "request_index": req["request_index"].astype(int),
                    "cumulative_amd64_share_pct":
                        amd64.cumsum().to_numpy()
                        / np.arange(1, len(req) + 1)
                        * 100.0,
                    "cumulative_duration_s":
                        req["duration_ms"].cumsum().to_numpy() / 1000.0,
                })
            )

    metrics_df = pd.DataFrame(rows)
    traj_df = pd.concat(trajectories, ignore_index=True)

    metrics_df.to_csv(
        out_dir / "prior_influence_c0_metrics.csv",
        index=False,
    )

    agg = (
        metrics_df
        .groupby(["variant", "label"], as_index=False)
        .agg(
            amd64_mean=("amd64_executions", "mean"),
            amd64_std=("amd64_executions", "std"),
            arm64_mean=("arm64_executions", "mean"),
            arm64_std=("arm64_executions", "std"),
            h50_mean_s=("cumulative_duration_s", "mean"),
            h50_std_s=("cumulative_duration_s", "std"),
            mean_duration_ms=("mean_duration_ms", "mean"),
            median_duration_ms=("median_duration_ms", "mean"),
        )
        .set_index("variant")
        .loc[["no_transfer", "improved"]]
    )

    # ------------------------------------------------------------------
    # 10 — Arm usage, 100% stacked
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(2)
    amd = agg["amd64_mean"].to_numpy() / 50.0 * 100.0
    arm = agg["arm64_mean"].to_numpy() / 50.0 * 100.0

    ax.bar(x, amd, label="amd64")
    ax.bar(x, arm, bottom=amd, label="arm64")

    ax.set_xticks(x, ["No Transfer\nc=0", "Improved Transfer\nc=0"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Quota delle esecuzioni (%)")
    ax.set_title("Influenza del prior trasferito senza bonus di esplorazione")
    ax.legend()

    for i in range(2):
        if amd[i] > 0:
            ax.text(
                x[i], amd[i] / 2,
                f"amd64\n{amd[i]:.0f}%",
                ha="center", va="center",
            )
        if arm[i] > 0:
            ax.text(
                x[i], amd[i] + arm[i] / 2,
                f"arm64\n{arm[i]:.0f}%",
                ha="center", va="center",
            )

    ax.text(
        0.01, -0.20,
        "Donor preference transferred to target: amd64. "
        "Observed target preference without transfer: arm64.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "10_prior_influence_arm_usage")

    # ------------------------------------------------------------------
    # 11 — Cumulative amd64 share over requests
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for variant in ("no_transfer", "improved"):
        d = traj_df[traj_df["variant"] == variant]
        a = (
            d.groupby("request_index")["cumulative_amd64_share_pct"]
            .agg(["mean", "std"])
            .reset_index()
        )
        xx = a["request_index"].to_numpy()
        mean = a["mean"].to_numpy()
        std = a["std"].fillna(0).to_numpy()

        ax.plot(
            xx, mean, linewidth=2,
            label=LABELS[variant],
        )
        ax.fill_between(
            xx, mean - std, mean + std, alpha=0.12
        )

    ax.set_ylim(0, 102)
    ax.set_xlabel("Indice della richiesta")
    ax.set_ylabel("Quota cumulativa su amd64 (%)")
    ax.set_title("Traiettoria decisionale con c=0")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.01, -0.18,
        "Media su 2 ripetizioni per configurazione; area = ±1 deviazione standard.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "11_prior_influence_trajectory")

    # ------------------------------------------------------------------
    # 12 — H50 cumulative duration
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    means = agg["h50_mean_s"].to_numpy()
    stds = agg["h50_std_s"].fillna(0).to_numpy()

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )
    ax.set_xticks(x, ["No Transfer\nc=0", "Improved Transfer\nc=0"])
    ax.set_ylabel("Durata cumulativa H50 (s)")
    ax.set_title("Conseguenza prestazionale della preferenza ereditata")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    save(fig, out_dir, "12_prior_influence_h50_duration")

    # ------------------------------------------------------------------
    # 13 — Summary table
    # ------------------------------------------------------------------
    base = agg.loc["no_transfer"]
    imp = agg.loc["improved"]

    delta_h50_pct = (
        (imp["h50_mean_s"] - base["h50_mean_s"])
        / base["h50_mean_s"] * 100.0
    )

    table_rows = [
        [
            "amd64 executions",
            f'{base["amd64_mean"]:.1f}/50',
            f'{imp["amd64_mean"]:.1f}/50',
        ],
        [
            "arm64 executions",
            f'{base["arm64_mean"]:.1f}/50',
            f'{imp["arm64_mean"]:.1f}/50',
        ],
        [
            "amd64 share",
            f'{base["amd64_mean"]/50*100:.1f}%',
            f'{imp["amd64_mean"]/50*100:.1f}%',
        ],
        [
            "H50 cumulative duration",
            f'{base["h50_mean_s"]:.3f} s',
            f'{imp["h50_mean_s"]:.3f} s',
        ],
        [
            "Mean request duration",
            f'{base["mean_duration_ms"]:.1f} ms',
            f'{imp["mean_duration_ms"]:.1f} ms',
        ],
        [
            "Median request duration",
            f'{base["median_duration_ms"]:.1f} ms',
            f'{imp["median_duration_ms"]:.1f} ms',
        ],
    ]

    md = [
        "| Metrica | No Transfer, c=0 | Improved Transfer, c=0 |",
        "|---|---:|---:|",
    ]
    for row in table_rows:
        md.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    md.append("")
    md.append(
        f"H50 Improved vs No Transfer: +{delta_h50_pct:.2f}% "
        "di durata cumulativa in questa condizione c=0."
    )
    md.append("")
    md.append(
        "Interpretazione: il test c=0 isola l'influenza del prior "
        "sulla traiettoria decisionale; non è la configurazione operativa "
        "finale del sistema."
    )

    (out_dir / "13_prior_influence_summary_table.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Metrica",
            "No Transfer, c=0",
            "Improved Transfer, c=0",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    ax.set_title(
        "Controllo c=0 — effetto del prior trasferito",
        pad=16,
    )
    fig.tight_layout()
    save(fig, out_dir, "13_prior_influence_summary_table")

    print("PASS — c=0 prior-influence analysis generated")
    print("\n===== RUN NO TRANSFER =====")
    for p in no_transfer_dirs:
        print(p)
    print("\n===== RUN IMPROVED TRANSFER =====")
    for p in improved_dirs:
        print(p)

    print("\n===== AGGREGATE =====")
    print(
        agg[
            [
                "amd64_mean",
                "arm64_mean",
                "h50_mean_s",
                "h50_std_s",
            ]
        ].to_string()
    )
    print(
        f"\nH50 delta Improved vs No Transfer: "
        f"+{delta_h50_pct:.2f}%"
    )

    print("\n===== OUTPUT =====")
    for stem in (
        "10_prior_influence_arm_usage",
        "11_prior_influence_trajectory",
        "12_prior_influence_h50_duration",
        "13_prior_influence_summary_table",
    ):
        print(out_dir / f"{stem}.png")
    print(out_dir / "13_prior_influence_summary_table.md")


if __name__ == "__main__":
    main()
