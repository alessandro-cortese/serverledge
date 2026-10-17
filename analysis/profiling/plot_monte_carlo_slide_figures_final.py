#!/usr/bin/env python3
"""
Serverledge — definitive slide-style Monte Carlo figures.

Style goal:
- same visual logic as the GCP slide: cumulative execution time vs request horizon;
- actual aggregate Monte Carlo latency, not a normalized/artificial y-axis;
- percentage benefit annotated directly on each horizon;
- exact values read from matched_multiseed_summary.csv.

Main figures
------------
01_mc_notl_vs_difference_frozen
    No-TL c=.2 vs frozen Difference (.05,64,.2)

02_mc_notl_vs_ratio_frozen
    No-TL c=.4 vs frozen Ratio (.01,64,.4)

03_mc_notl_vs_difference_vs_ratio_matched_A
    No-TL c=.2 vs Difference/Ratio with identical (.01,2,.2)

04_mc_ratio_vs_difference_matched_families
    Three panels, each a pure formula comparison with identical parameters:
      A = (.01,2,.2)
      B = (.05,64,.2)
      C = (.01,64,.4)

Also writes:
- monte_carlo_plot_values.csv
- MONTE_CARLO_FIGURE_NOTES.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [5, 10, 20, 50]


def fmt_params(w_r: float, w_e: float, c: float) -> str:
    def f(x: float) -> str:
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s[1:] if s.startswith("0.") else s
    return f"({f(w_r)}, {f(w_e)}, {f(c)})"


def get_strategy(summary: pd.DataFrame, name: str) -> pd.DataFrame:
    x = (
        summary[summary["strategy"] == name]
        .copy()
        .sort_values("horizon")
    )
    x = x[x["horizon"].isin(HORIZONS)].copy()
    got = x["horizon"].astype(int).tolist()
    if got != HORIZONS:
        raise SystemExit(
            f"{name}: expected horizons {HORIZONS}, got {got}"
        )
    return x


def sec(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / 1000.0


def gain(base_ms: np.ndarray, alt_ms: np.ndarray) -> np.ndarray:
    return 100.0 * (base_ms - alt_ms) / base_ms


def style_ax(ax, title: str, ylabel: str = "Cumulative execution time (s)") -> None:
    ax.set_title(title)
    ax.set_xlabel("Request horizon H")
    ax.set_ylabel(ylabel)
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)


def annotate_gain(
        ax,
        xs,
        ys,
        gains,
        *,
        prefix: str = "",
        yoffsets=None,
        fontsize: int = 10,
) -> None:
    if yoffsets is None:
        yoffsets = [8] * len(xs)
    for x, y, g, dy in zip(xs, ys, gains, yoffsets):
        label = f"{prefix}{g:+.2f}%"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=fontsize,
            fontweight="semibold",
        )


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)

    # Fixed strategies / baselines.
    nt02 = get_strategy(summary, "no_transfer_c02")
    nt04 = get_strategy(summary, "no_transfer_c04")

    dA = get_strategy(summary, "difference_A")
    rA = get_strategy(summary, "ratio_A")
    dB = get_strategy(summary, "difference_B")
    rB = get_strategy(summary, "ratio_B")
    dC = get_strategy(summary, "difference_C")
    rC = get_strategy(summary, "ratio_C")

    # ------------------------------------------------------------------
    # Figure 1 — No-TL vs frozen Difference
    # ------------------------------------------------------------------
    base = nt02["cumulative_latency_ms"].to_numpy(float)
    alt = dB["cumulative_latency_ms"].to_numpy(float)
    g = gain(base, alt)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(HORIZONS, sec(base), marker="o", linewidth=2.4, markersize=7,
            label="No Transfer (c=.2)")
    ax.plot(HORIZONS, sec(alt), marker="o", linewidth=2.4, markersize=7,
            label="Difference (.05, 64, .2)")
    style_ax(ax, "Monte Carlo — No-TL vs Difference")
    ax.legend(loc="upper left")
    annotate_gain(ax, HORIZONS, sec(alt), g,
                  prefix="", yoffsets=[-20, -20, -20, -20])
    save(fig, out, "01_mc_notl_vs_difference_frozen")

    # ------------------------------------------------------------------
    # Figure 2 — No-TL vs frozen Ratio
    # ------------------------------------------------------------------
    base = nt04["cumulative_latency_ms"].to_numpy(float)
    alt = rC["cumulative_latency_ms"].to_numpy(float)
    g = gain(base, alt)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(HORIZONS, sec(base), marker="o", linewidth=2.4, markersize=7,
            label="No Transfer (c=.4)")
    ax.plot(HORIZONS, sec(alt), marker="o", linewidth=2.4, markersize=7,
            label="Ratio (.01, 64, .4)")
    style_ax(ax, "Monte Carlo — No-TL vs Ratio")
    ax.legend(loc="upper left")
    annotate_gain(ax, HORIZONS, sec(alt), g,
                  prefix="", yoffsets=[-20, -20, -20, -20])
    save(fig, out, "02_mc_notl_vs_ratio_frozen")

    # ------------------------------------------------------------------
    # Figure 3 — matched family A, NoTL vs both formulas
    # ------------------------------------------------------------------
    base = nt02["cumulative_latency_ms"].to_numpy(float)
    d = dA["cumulative_latency_ms"].to_numpy(float)
    r = rA["cumulative_latency_ms"].to_numpy(float)
    gd = gain(base, d)
    gr = gain(base, r)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(HORIZONS, sec(base), marker="o", linewidth=2.4, markersize=7,
            label="No Transfer (c=.2)")
    ax.plot(HORIZONS, sec(d), marker="o", linewidth=2.4, markersize=7,
            label="Difference (.01, 2, .2)")
    ax.plot(HORIZONS, sec(r), marker="o", linewidth=2.4, markersize=7,
            label="Ratio (.01, 2, .2)")
    style_ax(ax, "Monte Carlo — No-TL vs Difference vs Ratio")
    ax.legend(loc="upper left")
    annotate_gain(ax, HORIZONS, sec(d), gd,
                  prefix="D ", yoffsets=[-26, -26, -26, -26], fontsize=9)
    annotate_gain(ax, HORIZONS, sec(r), gr,
                  prefix="R ", yoffsets=[9, 9, 9, 9], fontsize=9)
    save(fig, out, "03_mc_notl_vs_difference_vs_ratio_matched_A")

    # ------------------------------------------------------------------
    # Figure 4 — pure formula effect for all three matched families
    # ------------------------------------------------------------------
    fams = [
        ("A", dA, rA, "(.01, 2, .2)"),
        ("B", dB, rB, "(.05, 64, .2)"),
        ("C", dC, rC, "(.01, 64, .4)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)

    for ax, (letter, ddf, rdf, params) in zip(axes, fams):
        d = ddf["cumulative_latency_ms"].to_numpy(float)
        r = rdf["cumulative_latency_ms"].to_numpy(float)
        rgain = gain(d, r)  # positive => Ratio faster/lower latency

        ax.plot(HORIZONS, sec(d), marker="o", linewidth=2.2, markersize=6,
                label="Difference")
        ax.plot(HORIZONS, sec(r), marker="o", linewidth=2.2, markersize=6,
                label="Ratio")
        ax.set_title(f"{letter}: {params}")
        ax.set_xlabel("Request horizon H")
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.22)
        annotate_gain(
            ax,
            HORIZONS,
            sec(r),
            rgain,
            prefix="R−D ",
            yoffsets=[9, 9, 9, 9],
            fontsize=8,
        )

    axes[0].set_ylabel("Cumulative execution time (s)")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Monte Carlo — Ratio vs Difference with identical parameters",
        fontsize=18,
        y=1.03,
    )
    save(fig, out, "04_mc_ratio_vs_difference_matched_families")

    # ------------------------------------------------------------------
    # Exact table used by the figures.
    # ------------------------------------------------------------------
    rows = []

    def add_pair(fig_id, baseline_name, strategy_name, base_df, alt_df):
        b = base_df["cumulative_latency_ms"].to_numpy(float)
        a = alt_df["cumulative_latency_ms"].to_numpy(float)
        gg = gain(b, a)
        for h, bm, am, gv in zip(HORIZONS, b, a, gg):
            rows.append({
                "figure": fig_id,
                "baseline": baseline_name,
                "strategy": strategy_name,
                "horizon": h,
                "baseline_cumulative_latency_ms": bm,
                "strategy_cumulative_latency_ms": am,
                "latency_gain_percent": gv,
            })

    add_pair("01", "no_transfer_c02", "difference_B", nt02, dB)
    add_pair("02", "no_transfer_c04", "ratio_C", nt04, rC)
    add_pair("03D", "no_transfer_c02", "difference_A", nt02, dA)
    add_pair("03R", "no_transfer_c02", "ratio_A", nt02, rA)
    add_pair("04A", "difference_A", "ratio_A", dA, rA)
    add_pair("04B", "difference_B", "ratio_B", dB, rB)
    add_pair("04C", "difference_C", "ratio_C", dC, rC)

    values = pd.DataFrame(rows)
    values.to_csv(out / "monte_carlo_plot_values.csv", index=False)

    # ------------------------------------------------------------------
    # Notes for slide preparation.
    # ------------------------------------------------------------------
    def h10_gain(base_df, alt_df) -> float:
        b = float(base_df.loc[base_df["horizon"] == 10, "cumulative_latency_ms"].iloc[0])
        a = float(alt_df.loc[alt_df["horizon"] == 10, "cumulative_latency_ms"].iloc[0])
        return 100.0 * (b - a) / b

    notes = f"""# Monte Carlo figure notes

## Plotting convention

The figures show **mean cumulative execution time across the 57 primary non-synthetic target functions**.

The percentage labels are computed from the same aggregate cumulative latency:

`100 * (baseline - strategy) / baseline`.

This presentation intentionally mirrors the GCP slide style: the curves show absolute cumulative cost, while labels expose the relative benefit.

## Figure 1 — frozen Difference

Configuration: Difference `(.05,64,.2)` vs matched No-TL `c=.2`.

H10 gain: **{h10_gain(nt02, dB):.3f}%**.

## Figure 2 — frozen Ratio

Configuration: Ratio `(.01,64,.4)` vs matched No-TL `c=.4`.

H10 gain: **{h10_gain(nt04, rC):.3f}%**.

## Figure 3 — pure matched formula comparison

Difference and Ratio both use `(.01,2,.2)`, with No-TL `c=.2`.

H10 Difference gain: **{h10_gain(nt02, dA):.3f}%**.

H10 Ratio gain: **{h10_gain(nt02, rA):.3f}%**.

## Figure 4 — direct formula effect

Each panel holds `w_R`, `w_E`, and `c` fixed; only the transfer formula changes.

The direct Ratio-vs-Difference differences are intentionally small compared with the full TL-vs-NoTL effect. This is expected: both formulas transfer very similar donor information, while the main effect is the presence and strength of transfer/exploration priors.

## Why the aggregate gain can look small

The headline curves are macro-aggregates across 57 functions. They hide target-level heterogeneity: some functions receive a strong gain, some are nearly neutral, and a minority can experience negative transfer. The transfer benefit is also concentrated in the early horizon and is progressively amortized as H grows.

Therefore:
- use these figures to show the **overall trend**;
- use per-target tables/distributions to show the **heterogeneity behind the mean**;
- interpret the GCP campaign similarly, but more cautiously because it contains only four held-out target functions.
"""
    (out / "MONTE_CARLO_FIGURE_NOTES.md").write_text(notes, encoding="utf-8")

    print("PASS — definitive Monte Carlo slide figures generated")
    print()
    print(values.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    for path in sorted(out.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
