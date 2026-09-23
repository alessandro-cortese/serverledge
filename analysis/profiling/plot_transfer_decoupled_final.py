#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("data/profiling/gcp-transfer-final")

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "presentation_graphs"
)

RUNS = {
    "wrong_c0": ROOT
    / "gcp-transfer-materialized-randomaccess-decoupled-c00-20260923_080510",

    "wrong_c08": ROOT
    / "gcp-transfer-materialized-randomaccess-decoupled-c08-20260923_080146",

    "good_c0": ROOT
    / "gcp-transfer-materialized-hashing-decoupled-c00-20260923_152020",

    "good_c08": ROOT
    / "gcp-transfer-materialized-hashing-decoupled-c08-20260923_152332",
}

META = {
    "wrong_c0": {
        "case": "Wrong donor",
        "target": "randomaccess",
        "donor": "compression",
        "c": 0.0,
    },
    "wrong_c08": {
        "case": "Wrong donor",
        "target": "randomaccess",
        "donor": "compression",
        "c": 0.8,
    },
    "good_c0": {
        "case": "Good donor",
        "target": "hashing",
        "donor": "hashing-py",
        "c": 0.0,
    },
    "good_c08": {
        "case": "Good donor",
        "target": "hashing",
        "donor": "hashing-py",
        "c": 0.8,
    },
}

# Entrambi i target hanno ground truth ARM64.
OPTIMAL_ARM = "arm64"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(run: Path) -> dict:
    path = run / "measure" / "summary.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    return load_json(path)


def load_requests(run: Path) -> list[dict]:
    path = run / "measure" / "mab-requests.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def selected_counts(summary: dict) -> tuple[int, int]:
    counts = summary.get(
        "selected_arm_counts",
        {},
    )

    return (
        int(counts.get("amd64", 0)),
        int(counts.get("arm64", 0)),
    )


def cumulative_series(
    requests: list[dict],
) -> tuple[list[int], list[float], list[int]]:

    indices = []
    arm_shares = []
    suboptimal_counts = []

    arm_count = 0
    suboptimal_count = 0

    for index, row in enumerate(
        requests,
        start=1,
    ):
        selected = row["selected_arm"].strip()

        if selected == OPTIMAL_ARM:
            arm_count += 1
        else:
            suboptimal_count += 1

        indices.append(index)

        arm_shares.append(
            arm_count / index * 100.0
        )

        suboptimal_counts.append(
            suboptimal_count
        )

    return (
        indices,
        arm_shares,
        suboptimal_counts,
    )


def save(fig, basename: str) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{basename}.png",
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{basename}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_architecture_choices(
    summaries: dict[str, dict],
) -> None:

    order = [
        "wrong_c0",
        "wrong_c08",
        "good_c0",
        "good_c08",
    ]

    labels = [
        "Wrong donor\nc = 0",
        "Wrong donor\nc = 0.8",
        "Good donor\nc = 0",
        "Good donor\nc = 0.8",
    ]

    amd_pct = []
    arm_pct = []

    for key in order:
        amd, arm = selected_counts(
            summaries[key]
        )

        total = amd + arm

        amd_pct.append(
            100.0 * amd / total
        )

        arm_pct.append(
            100.0 * arm / total
        )

    y = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    ax.barh(
        y,
        amd_pct,
        label="AMD64",
    )

    ax.barh(
        y,
        arm_pct,
        left=amd_pct,
        label="ARM64",
    )

    for index, (amd, arm) in enumerate(
        zip(amd_pct, arm_pct)
    ):
        if amd > 0:
            ax.text(
                amd / 2,
                index,
                f"{amd:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
            )

        if arm > 0:
            ax.text(
                amd + arm / 2,
                index,
                f"{arm:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    ax.set_xlim(0, 100)

    ax.set_xlabel(
        "Architecture selections over 50 target requests (%)"
    )

    ax.set_title(
        "Transfer Learning\n"
        "Effect of donor quality and exploration"
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    save(
        fig,
        "06_transfer_learning_good_vs_wrong_architecture_choices",
    )


def plot_arm_share(
    requests: dict[str, list[dict]],
) -> None:

    series = [
        (
            "Wrong donor — c=0",
            "wrong_c0",
            "--",
        ),
        (
            "Wrong donor — c=0.8",
            "wrong_c08",
            "-",
        ),
        (
            "Good donor — c=0",
            "good_c0",
            "--",
        ),
        (
            "Good donor — c=0.8",
            "good_c08",
            "-",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    for label, key, linestyle in series:
        x, arm_share, _ = cumulative_series(
            requests[key]
        )

        ax.plot(
            x,
            arm_share,
            linewidth=2.3,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlim(1, 50)
    ax.set_ylim(-3, 103)

    ax.set_xlabel(
        "Target request index"
    )

    ax.set_ylabel(
        "Cumulative ARM64 selection share (%)"
    )

    ax.set_title(
        "Transfer Learning adaptation over target requests\n"
        "ARM64 is the ground-truth optimal architecture in both cases"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        loc="center right",
        frameon=True,
    )

    save(
        fig,
        "07_transfer_learning_arm_share_over_requests",
    )


def plot_suboptimal_choices(
    requests: dict[str, list[dict]],
) -> None:

    series = [
        (
            "Wrong donor — c=0",
            "wrong_c0",
            "--",
        ),
        (
            "Wrong donor — c=0.8",
            "wrong_c08",
            "-",
        ),
        (
            "Good donor — c=0",
            "good_c0",
            "--",
        ),
        (
            "Good donor — c=0.8",
            "good_c08",
            "-",
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    for label, key, linestyle in series:
        x, _, suboptimal = cumulative_series(
            requests[key]
        )

        ax.plot(
            x,
            suboptimal,
            linewidth=2.3,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlim(1, 50)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(
        "Target request index"
    )

    ax.set_ylabel(
        "Cumulative suboptimal architecture selections"
    )

    ax.set_title(
        "Cost of a misaligned donor prior\n"
        "Suboptimal = AMD64, because both targets are ARM64-preferred"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    save(
        fig,
        "08_transfer_learning_cumulative_suboptimal_choices",
    )


def build_summary_csv(
    summaries: dict[str, dict],
) -> None:

    output = (
        OUTPUT_DIR
        / "transfer_learning_final.csv"
    )

    fields = [
        "case",
        "target",
        "donor",
        "policy",
        "c",
        "amd64_selections",
        "arm64_selections",
        "arm64_share_percent",
        "suboptimal_selections",
        "mean_duration_ms",
        "median_duration_ms",
        "fallback_count",
        "all_warm",
    ]

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for key in (
            "wrong_c0",
            "wrong_c08",
            "good_c0",
            "good_c08",
        ):
            summary = summaries[key]
            meta = META[key]

            amd, arm = selected_counts(
                summary
            )

            total = amd + arm

            horizon = summary[
                "horizons"
            ]["50"]

            writer.writerow(
                {
                    "case": meta["case"],
                    "target": meta["target"],
                    "donor": meta["donor"],
                    "policy": summary["policy"],
                    "c": meta["c"],
                    "amd64_selections": amd,
                    "arm64_selections": arm,
                    "arm64_share_percent": (
                        100.0 * arm / total
                    ),
                    "suboptimal_selections": amd,
                    "mean_duration_ms": horizon[
                        "mean_duration_ms"
                    ],
                    "median_duration_ms": horizon[
                        "median_duration_ms"
                    ],
                    "fallback_count": summary[
                        "fallback_count"
                    ],
                    "all_warm": summary[
                        "all_warm"
                    ],
                }
            )


def print_analysis(
    summaries: dict[str, dict],
) -> None:

    print()
    print("=" * 72)
    print("FINAL TRANSFER LEARNING COMPARISON")
    print("=" * 72)

    for key in (
        "wrong_c0",
        "wrong_c08",
        "good_c0",
        "good_c08",
    ):
        summary = summaries[key]
        meta = META[key]

        amd, arm = selected_counts(
            summary
        )

        h50 = summary["horizons"]["50"]

        print(
            f"{meta['case']:11s} "
            f"c={meta['c']:<3} "
            f"target={meta['target']:12s} "
            f"donor={meta['donor']:10s} "
            f"AMD={amd:2d} "
            f"ARM={arm:2d} "
            f"mean={h50['mean_duration_ms']:.3f} ms"
        )

    print()
    print(
        "Wrong donor, c=0: transferred knowledge is followed "
        "without correction."
    )

    print(
        "Wrong donor, c=0.8: exploration exposes the ARM64 "
        "advantage and corrects the initial prior."
    )

    print(
        "Good donor, c=0 and c=0.8: transferred knowledge is "
        "already aligned with the target and ARM64 is selected "
        "for all 50 requests."
    )


def main() -> None:

    summaries = {
        key: load_summary(path)
        for key, path in RUNS.items()
    }

    requests = {
        key: load_requests(path)
        for key, path in RUNS.items()
    }

    # Strong comparability gates.
    for key, summary in summaries.items():
        if summary["policy"] != "UCB1Decoupled":
            raise RuntimeError(
                f"{key}: expected UCB1Decoupled, "
                f"got {summary['policy']}"
            )

        if summary["request_count"] != 50:
            raise RuntimeError(
                f"{key}: expected 50 requests"
            )

        if not summary["all_warm"]:
            raise RuntimeError(
                f"{key}: cold request detected"
            )

        if summary["fallback_count"] != 0:
            raise RuntimeError(
                f"{key}: fallback detected"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_final_architecture_choices(
        summaries
    )

    plot_arm_share(
        requests
    )

    plot_suboptimal_choices(
        requests
    )

    build_summary_csv(
        summaries
    )

    print_analysis(
        summaries
    )

    print()
    print(
        "PASS — final transfer learning analysis generated"
    )

    print(
        f"output={OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
