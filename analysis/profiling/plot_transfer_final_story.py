#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZONS = (5, 10, 20, 50)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def latest_summary(root: Path, target: str, c_tag: str) -> Path:
    patterns = [
        f"gcp-transfer-materialized-{target}-coupled-c{c_tag}-*/measure/summary.json",
        f"gcp-transfer-materialized-{target}-c{c_tag}-*/measure/summary.json",
    ]

    candidates: list[Path] = []

    for pattern in patterns:
        candidates.extend(root.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"nessun summary trovato per target={target}, c={c_tag}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_prior(path: Path) -> dict:
    document = load_json(path)
    return document.get("prior", document)


def prior_means(prior: dict) -> tuple[float, float]:
    arms = prior["arms"]

    amd = float(arms["amd64"]["ucb1"]["mean_reward"])
    arm = float(arms["arm64"]["ucb1"]["mean_reward"])

    return amd, arm


def winner_from_rewards(amd: float, arm: float) -> str:
    return "ARM64" if arm > amd else "AMD64"


def counts(summary: dict) -> tuple[int, int]:
    selected = summary.get("selected_arm_counts", {})

    return (
        int(selected.get("amd64", 0)),
        int(selected.get("arm64", 0)),
    )


def arm_share(summary: dict, horizon: int) -> float:
    block = summary["horizons"][str(horizon)]

    amd = int(block.get("amd64_executions", 0))
    arm = int(block.get("arm64_executions", 0))

    total = amd + arm

    if total == 0:
        return 0.0

    return arm / total * 100.0


def save_figure(fig, output_dir: Path, basename: str) -> None:
    fig.tight_layout()

    fig.savefig(
        output_dir / f"{basename}.png",
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{basename}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_architecture_choices(
    wrong_c00: dict,
    wrong_c08: dict,
    good_c00: dict,
    good_c08: dict,
    output_dir: Path,
) -> None:

    labels = [
        "Wrong donor\nc = 0",
        "Wrong donor\nc = 0.8",
        "Good donor\nc = 0",
        "Good donor\nc = 0.8",
    ]

    summaries = [
        wrong_c00,
        wrong_c08,
        good_c00,
        good_c08,
    ]

    amd_percent = []
    arm_percent = []

    for summary in summaries:
        amd, arm = counts(summary)
        total = amd + arm

        amd_percent.append(100.0 * amd / total)
        arm_percent.append(100.0 * arm / total)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    ax.barh(
        y,
        amd_percent,
        label="AMD64",
        color="#4C78A8",
    )

    ax.barh(
        y,
        arm_percent,
        left=amd_percent,
        label="ARM64",
        color="#F58518",
    )

    for index, (amd, arm) in enumerate(
        zip(amd_percent, arm_percent)
    ):
        if amd > 0:
            ax.text(
                amd / 2,
                index,
                f"{amd:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        if arm > 0:
            ax.text(
                amd + arm / 2,
                index,
                f"{arm:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    ax.set_xlim(0, 100)

    ax.set_xlabel("Architecture selections over 50 target requests (%)")

    ax.set_title(
        "Transfer learning: wrong vs good donor\n"
        "Effect of exploration on architecture selection"
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

    save_figure(
        fig,
        output_dir,
        "03_good_vs_wrong_donor_architecture_choices",
    )


def plot_horizon_evolution(
    wrong_c00: dict,
    wrong_c08: dict,
    good_c00: dict,
    good_c08: dict,
    output_dir: Path,
) -> None:

    series = [
        (
            "Wrong donor — c=0",
            wrong_c00,
            "#D62728",
            "--",
        ),
        (
            "Wrong donor — c=0.8",
            wrong_c08,
            "#FF9896",
            "-",
        ),
        (
            "Good donor — c=0",
            good_c00,
            "#2CA02C",
            "--",
        ),
        (
            "Good donor — c=0.8",
            good_c08,
            "#98DF8A",
            "-",
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for label, summary, color, linestyle in series:
        values = [
            arm_share(summary, horizon)
            for horizon in HORIZONS
        ]

        ax.plot(
            HORIZONS,
            values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            linestyle=linestyle,
            color=color,
            label=label,
        )

        for horizon, value in zip(HORIZONS, values):
            ax.annotate(
                f"{value:.0f}%",
                (horizon, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
            )

    ax.set_ylim(-5, 105)

    ax.set_xticks(HORIZONS)

    ax.set_xlabel("Number of target requests")
    ax.set_ylabel("ARM64 selection share (%)")

    ax.set_title(
        "How exploration validates or corrects transferred knowledge"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        loc="lower right",
        frameon=True,
    )

    save_figure(
        fig,
        output_dir,
        "04_arm_share_over_requests",
    )


def plot_ground_truth(
    wrong_prior: dict,
    good_prior: dict,
    wrong_x86_ms: float,
    wrong_arm_ms: float,
    good_x86_ms: float,
    good_arm_ms: float,
    output_dir: Path,
) -> None:

    labels = [
        "Wrong donor case\nrandomaccess",
        "Good donor case\nhashing",
    ]

    x86 = [
        wrong_x86_ms,
        good_x86_ms,
    ]

    arm = [
        wrong_arm_ms,
        good_arm_ms,
    ]

    wrong_amd_reward, wrong_arm_reward = prior_means(
        wrong_prior
    )

    good_amd_reward, good_arm_reward = prior_means(
        good_prior
    )

    donor_preferences = [
        winner_from_rewards(
            wrong_amd_reward,
            wrong_arm_reward,
        ),
        winner_from_rewards(
            good_amd_reward,
            good_arm_reward,
        ),
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    bars_x86 = ax.bar(
        x - width / 2,
        x86,
        width,
        label="AMD64 ground truth",
        color="#4C78A8",
    )

    bars_arm = ax.bar(
        x + width / 2,
        arm,
        width,
        label="ARM64 ground truth",
        color="#F58518",
    )

    for bars in (bars_x86, bars_arm):
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.0f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    y_max = max(x86 + arm)

    for index, preference in enumerate(
        donor_preferences
    ):
        target_winner = (
            "ARM64"
            if arm[index] < x86[index]
            else "AMD64"
        )

        relation = (
            "aligned"
            if preference == target_winner
            else "misaligned"
        )

        ax.text(
            index,
            y_max * 1.10,
            (
                f"Transferred donor prior: {preference}\n"
                f"Target ground truth: {target_winner} "
                f"({relation})"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, y_max * 1.30)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Mean execution duration (ms)")

    ax.set_title(
        "Transferred prior vs empirical target performance"
    )

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(
        fig,
        output_dir,
        "05_prior_vs_target_ground_truth",
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    fieldnames = [
        "case",
        "target",
        "donor",
        "c",
        "amd64_selections",
        "arm64_selections",
        "arm64_share_percent",
        "arm64_share_h5",
        "arm64_share_h10",
        "arm64_share_h20",
        "arm64_share_h50",
        "median_duration_h50_ms",
        "mean_duration_h50_ms",
        "fallback_count",
        "all_warm",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_row(
    case: str,
    target: str,
    donor: str,
    c: float,
    summary: dict,
) -> dict:

    amd, arm = counts(summary)

    total = amd + arm

    h50 = summary["horizons"]["50"]

    return {
        "case": case,
        "target": target,
        "donor": donor,
        "c": c,
        "amd64_selections": amd,
        "arm64_selections": arm,
        "arm64_share_percent": (
            100.0 * arm / total
        ),
        "arm64_share_h5": arm_share(
            summary,
            5,
        ),
        "arm64_share_h10": arm_share(
            summary,
            10,
        ),
        "arm64_share_h20": arm_share(
            summary,
            20,
        ),
        "arm64_share_h50": arm_share(
            summary,
            50,
        ),
        "median_duration_h50_ms": h50[
            "median_duration_ms"
        ],
        "mean_duration_h50_ms": h50[
            "mean_duration_ms"
        ],
        "fallback_count": summary[
            "fallback_count"
        ],
        "all_warm": summary[
            "all_warm"
        ],
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate final presentation graphs for "
            "Serverledge transfer-learning exploration ablation."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/"
            "analysis/presentation_graphs"
        ),
    )

    parser.add_argument(
        "--wrong-materialized",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/"
            "exploration-ablation/"
            "randomaccess-r03-materialized"
        ),
    )

    parser.add_argument(
        "--good-materialized",
        type=Path,
        default=Path(
            "data/profiling/gcp-transfer-final/"
            "exploration-ablation/"
            "hashing-r02-materialized"
        ),
    )

    # Frozen ground-truth measurements already collected
    # in the final dual-architecture campaign.
    parser.add_argument(
        "--wrong-x86-ms",
        type=float,
        default=1647.718275,
    )

    parser.add_argument(
        "--wrong-arm-ms",
        type=float,
        default=817.547201,
    )

    parser.add_argument(
        "--good-x86-ms",
        type=float,
        default=3436.7184005,
    )

    parser.add_argument(
        "--good-arm-ms",
        type=float,
        default=709.105813,
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wrong_c00_path = latest_summary(
        root,
        "randomaccess",
        "00",
    )

    wrong_c08_path = latest_summary(
        root,
        "randomaccess",
        "08",
    )

    good_c00_path = latest_summary(
        root,
        "hashing",
        "00",
    )

    good_c08_path = latest_summary(
        root,
        "hashing",
        "08",
    )

    wrong_c00 = load_json(wrong_c00_path)
    wrong_c08 = load_json(wrong_c08_path)
    good_c00 = load_json(good_c00_path)
    good_c08 = load_json(good_c08_path)

    wrong_prior = load_prior(
        args.wrong_materialized
        / "frozen-prior.json"
    )

    good_prior = load_prior(
        args.good_materialized
        / "frozen-prior.json"
    )

    plot_architecture_choices(
        wrong_c00,
        wrong_c08,
        good_c00,
        good_c08,
        output_dir,
    )

    plot_horizon_evolution(
        wrong_c00,
        wrong_c08,
        good_c00,
        good_c08,
        output_dir,
    )

    plot_ground_truth(
        wrong_prior,
        good_prior,
        args.wrong_x86_ms,
        args.wrong_arm_ms,
        args.good_x86_ms,
        args.good_arm_ms,
        output_dir,
    )

    rows = [
        build_row(
            "wrong-donor",
            "randomaccess",
            "compression",
            0.0,
            wrong_c00,
        ),
        build_row(
            "wrong-donor",
            "randomaccess",
            "compression",
            0.8,
            wrong_c08,
        ),
        build_row(
            "good-donor",
            "hashing",
            "hashing-py",
            0.0,
            good_c00,
        ),
        build_row(
            "good-donor",
            "hashing",
            "hashing-py",
            0.8,
            good_c08,
        ),
    ]

    csv_path = (
        output_dir
        / "transfer_final_story.csv"
    )

    write_csv(
        csv_path,
        rows,
    )

    wrong_amd_reward, wrong_arm_reward = prior_means(
        wrong_prior
    )

    good_amd_reward, good_arm_reward = prior_means(
        good_prior
    )

    print("PASS — final transfer story generated")
    print()
    print(f"wrong c=0:   {wrong_c00_path}")
    print(f"wrong c=.8:  {wrong_c08_path}")
    print(f"good c=0:    {good_c00_path}")
    print(f"good c=.8:   {good_c08_path}")
    print()

    print(
        "wrong prior preference:",
        winner_from_rewards(
            wrong_amd_reward,
            wrong_arm_reward,
        ),
    )

    print(
        "good prior preference:",
        winner_from_rewards(
            good_amd_reward,
            good_arm_reward,
        ),
    )

    print()

    print(
        "graphs:",
        output_dir,
    )

    print(
        "csv:",
        csv_path,
    )


if __name__ == "__main__":
    main()
