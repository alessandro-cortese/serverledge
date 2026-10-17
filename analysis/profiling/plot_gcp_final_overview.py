#!/usr/bin/env python3
"""
Serverledge — final accepted GCP campaign overview plots.

Discovers the FINAL accepted GCP runs directly from:
    data/profiling/gcp-ucb1-final/runs/**/manifest.json

Acceptance gates:
- exact frozen final runtime commit
- request_count == 50
- all_warm == True
- fallback_count == 0
- one accepted run for each of 4 targets x 4 conditions

Conditions:
- notl-difference : No-TL c=.2
- difference      : Difference (.05,64,.2)
- notl-ratio      : No-TL c=.4
- ratio           : Ratio (.01,64,.4)

Generates:
1) GCP aggregate cumulative duration: No-TL c=.2 vs Difference
2) GCP aggregate cumulative duration: No-TL c=.4 vs Ratio
3) H10 per-target gain: Difference vs Ratio

The aggregate curves show the mean cumulative duration across the four held-out
targets. Percentage annotations are the MACRO mean of per-target gains, i.e.
each held-out function receives equal weight.

Outputs:
    gcp_01_aggregate_difference.png/.pdf
    gcp_02_aggregate_ratio.png/.pdf
    gcp_03_h10_per_target_gain.png/.pdf
    gcp_final_plot_data.csv
    gcp_final_aggregate.csv
    GCP_FINAL_PLOT_NOTES.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_RUNTIME = "4c9f427a76e188c9cb026d0b65ccf44b4d116d6f"
HORIZONS = [5, 10, 20, 50]

TARGETS = [
    "billibench-knn",
    "dynamic-html-py",
    "functionbench-gzip",
    "graph-bfs",
]

CONDITIONS = [
    "notl-difference",
    "difference",
    "notl-ratio",
    "ratio",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def discover_accepted_runs(root: Path) -> pd.DataFrame:
    rows = []

    for manifest_path in root.glob("runs/**/manifest.json"):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue

        target = str(manifest.get("target_function", ""))
        condition = str(manifest.get("condition", ""))
        commit = str(manifest.get("serverledge_commit", ""))

        if target not in TARGETS:
            continue
        if condition not in CONDITIONS:
            continue
        if commit != FINAL_RUNTIME:
            continue
        if int(manifest.get("request_count", -1)) != 50:
            continue
        if manifest.get("all_warm") is not True:
            continue
        if int(manifest.get("fallback_count", -1)) != 0:
            continue

        summary_path = manifest_path.parent / "measure" / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(
                f"accepted-looking run has no summary: {summary_path}"
            )

        summary = load_json(summary_path)

        if summary.get("all_warm") is not True:
            continue
        if int(summary.get("fallback_count", -1)) != 0:
            continue

        horizons = summary.get("horizons", {})

        for h in HORIZONS:
            hd = horizons.get(str(h))
            if hd is None:
                raise SystemExit(
                    f"{summary_path}: missing horizon {h}"
                )

            rows.append({
                "target_function": target,
                "condition": condition,
                "horizon": h,
                "cumulative_duration_ms": float(
                    hd["cumulative_duration_ms"]
                ),
                "amd64_executions": int(
                    hd["amd64_executions"]
                ),
                "arm64_executions": int(
                    hd["arm64_executions"]
                ),
                "fallbacks": int(hd["fallbacks"]),
                "run_id": manifest.get("run_id"),
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "serverledge_commit": commit,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit(
            f"no accepted final GCP runs found under {root}"
        )

    # Exactly one accepted run per target/condition.
    base = (
        df[["target_function", "condition", "run_id"]]
        .drop_duplicates()
    )

    dup = (
        base.groupby(["target_function", "condition"])
        .size()
        .reset_index(name="n")
    )
    bad_dup = dup[dup["n"] != 1]
    if not bad_dup.empty:
        raise SystemExit(
            "expected exactly one accepted run per target/condition:\n"
            + bad_dup.to_string(index=False)
        )

    expected = {
        (t, c)
        for t in TARGETS
        for c in CONDITIONS
    }
    got = set(
        zip(base["target_function"], base["condition"])
    )

    missing = sorted(expected - got)
    extra = sorted(got - expected)

    if missing or extra:
        raise SystemExit(
            f"accepted-run matrix mismatch. missing={missing} extra={extra}"
        )

    if len(base) != 16:
        raise SystemExit(
            f"expected 16 accepted runs, got {len(base)}"
        )

    return df.sort_values(
        ["target_function", "condition", "horizon"]
    ).reset_index(drop=True)


def pair_gain_table(
        df: pd.DataFrame,
        baseline: str,
        transfer: str,
        formula: str,
) -> pd.DataFrame:
    b = df[df["condition"] == baseline][
        ["target_function", "horizon", "cumulative_duration_ms"]
    ].rename(
        columns={"cumulative_duration_ms": "baseline_ms"}
    )

    t = df[df["condition"] == transfer][
        ["target_function", "horizon", "cumulative_duration_ms"]
    ].rename(
        columns={"cumulative_duration_ms": "transfer_ms"}
    )

    x = b.merge(
        t,
        on=["target_function", "horizon"],
        validate="one_to_one",
    )

    x["gain_percent"] = (
            100.0
            * (x["baseline_ms"] - x["transfer_ms"])
            / x["baseline_ms"]
    )
    x["formula"] = formula

    return x


def aggregate_pair(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in HORIZONS:
        x = pair[pair["horizon"] == h].copy()

        baseline_mean_ms = float(x["baseline_ms"].mean())
        transfer_mean_ms = float(x["transfer_ms"].mean())

        macro_gain = float(x["gain_percent"].mean())
        pooled_gain = (
                100.0
                * (baseline_mean_ms - transfer_mean_ms)
                / baseline_mean_ms
        )

        rows.append({
            "formula": str(x["formula"].iloc[0]),
            "horizon": h,
            "target_count": int(x["target_function"].nunique()),
            "baseline_mean_cumulative_ms": baseline_mean_ms,
            "transfer_mean_cumulative_ms": transfer_mean_ms,
            "macro_mean_target_gain_percent": macro_gain,
            "pooled_gain_percent": pooled_gain,
        })

    return pd.DataFrame(rows)


def annotate_macro_gain(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        gain: np.ndarray,
) -> None:
    for h, yy, gg in zip(x, y, gain):
        dy = -18 if h != 50 else -20
        ax.annotate(
            f"{gg:+.2f}%",
            xy=(h, yy),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


def plot_aggregate(
        agg: pd.DataFrame,
        *,
        title: str,
        baseline_label: str,
        transfer_label: str,
        out_png: Path,
        out_pdf: Path,
) -> None:
    x = agg["horizon"].to_numpy(int)
    b = (
            agg["baseline_mean_cumulative_ms"]
            .to_numpy(float) / 1000.0
    )
    t = (
            agg["transfer_mean_cumulative_ms"]
            .to_numpy(float) / 1000.0
    )
    g = agg["macro_mean_target_gain_percent"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.6, 5.7))

    ax.plot(
        x,
        b,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=baseline_label,
    )
    ax.plot(
        x,
        t,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=transfer_label,
    )

    annotate_macro_gain(ax, x, t, g)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Request horizon H", fontsize=14)
    ax.set_ylabel(
        "Mean cumulative execution time across held-out targets (s)",
        fontsize=13,
    )
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_h10_per_target(
        diff_pair: pd.DataFrame,
        ratio_pair: pd.DataFrame,
        out_png: Path,
        out_pdf: Path,
) -> pd.DataFrame:
    d = diff_pair[diff_pair["horizon"] == 10][
        ["target_function", "gain_percent"]
    ].rename(columns={"gain_percent": "difference_gain_percent"})

    r = ratio_pair[ratio_pair["horizon"] == 10][
        ["target_function", "gain_percent"]
    ].rename(columns={"gain_percent": "ratio_gain_percent"})

    x = d.merge(
        r,
        on="target_function",
        validate="one_to_one",
    )

    order = TARGETS
    x["target_function"] = pd.Categorical(
        x["target_function"],
        categories=order,
        ordered=True,
    )
    x = x.sort_values("target_function")

    pos = np.arange(len(x))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.4, 5.7))

    bars_d = ax.bar(
        pos - width / 2,
        x["difference_gain_percent"],
        width,
        label="Difference (.05, 64, .2)",
        )
    bars_r = ax.bar(
        pos + width / 2,
        x["ratio_gain_percent"],
        width,
        label="Ratio (.01, 64, .4)",
        )

    for bars in (bars_d, bars_r):
        for bar in bars:
            value = float(bar.get_height())
            offset = 4 if value >= 0 else -13
            va = "bottom" if value >= 0 else "top"
            ax.annotate(
                f"{value:+.2f}%",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(
        "GCP — H10 latency gain by held-out target",
        fontsize=18,
        pad=12,
    )
    ax.set_ylabel(
        "Latency gain vs matched No-Transfer (%)",
        fontsize=13,
    )
    ax.set_xticks(pos)
    ax.set_xticklabels(
        [str(v) for v in x["target_function"]],
        rotation=15,
        ha="right",
    )
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcp-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    root = args.gcp_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = discover_accepted_runs(root)

    diff_pair = pair_gain_table(
        data,
        "notl-difference",
        "difference",
        "difference",
    )
    ratio_pair = pair_gain_table(
        data,
        "notl-ratio",
        "ratio",
        "ratio",
    )

    diff_agg = aggregate_pair(diff_pair)
    ratio_agg = aggregate_pair(ratio_pair)

    plot_aggregate(
        diff_agg,
        title="GCP — No-TL vs Difference",
        baseline_label="No Transfer (c=.2)",
        transfer_label="Difference (.05, 64, .2)",
        out_png=out / "gcp_01_aggregate_difference.png",
        out_pdf=out / "gcp_01_aggregate_difference.pdf",
    )

    plot_aggregate(
        ratio_agg,
        title="GCP — No-TL vs Ratio",
        baseline_label="No Transfer (c=.4)",
        transfer_label="Ratio (.01, 64, .4)",
        out_png=out / "gcp_02_aggregate_ratio.png",
        out_pdf=out / "gcp_02_aggregate_ratio.pdf",
    )

    h10 = plot_h10_per_target(
        diff_pair,
        ratio_pair,
        out / "gcp_03_h10_per_target_gain.png",
        out / "gcp_03_h10_per_target_gain.pdf",
        )

    data_path = out / "gcp_final_plot_data.csv"
    data.to_csv(data_path, index=False)

    aggregate = pd.concat(
        [diff_agg, ratio_agg],
        ignore_index=True,
    )
    agg_path = out / "gcp_final_aggregate.csv"
    aggregate.to_csv(agg_path, index=False)

    h10_path = out / "gcp_final_h10_per_target.csv"
    h10.to_csv(h10_path, index=False)

    notes = out / "GCP_FINAL_PLOT_NOTES.md"
    notes.write_text(
        "# Final GCP plotting notes\n\n"
        f"Frozen runtime commit: `{FINAL_RUNTIME}`.\n\n"
        "Accepted runs are selected only when: 50 requests, all warm, "
        "fallback_count=0, exact frozen runtime commit, and exactly one run "
        "exists for each target/condition pair.\n\n"
        "Aggregate line plots:\n"
        "- curve y-value = mean cumulative duration across the four held-out targets;\n"
        "- percentage annotation = macro mean of the four per-target gains, "
        "so every held-out function has equal weight;\n"
        "- `gcp_final_aggregate.csv` also records the pooled gain based on "
        "the ratio of mean durations.\n\n"
        "The H10 bar chart is the most transparent breakdown because it shows "
        "the individual held-out functions rather than hiding heterogeneity "
        "inside the mean.\n",
        encoding="utf-8",
    )

    print("PASS — final GCP overview plots generated")
    print()
    print("===== ACCEPTED RUN MATRIX =====")
    print(
        data[
            ["target_function", "condition", "run_id"]
        ]
        .drop_duplicates()
        .sort_values(["target_function", "condition"])
        .to_string(index=False)
    )

    print()
    print("===== AGGREGATE =====")
    print(
        aggregate.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print()
    print("===== H10 PER TARGET =====")
    print(
        h10.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print()
    for p in sorted(out.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
