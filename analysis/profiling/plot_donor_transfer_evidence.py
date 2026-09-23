#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def parse_donor_log(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if "function=compression" not in line:
            continue
        arm = re.search(r"arm=([^ ]+)", line)
        dur = re.search(r"duration_ms=([0-9.]+)", line)
        rew = re.search(r"reward=([-0-9.]+)", line)
        if arm and dur and rew:
            rows.append({
                "arm": arm.group(1),
                "duration_ms": float(dur.group(1)),
                "reward": float(rew.group(1)),
            })
    return pd.DataFrame(rows)


def complete_dirs(gcp_root: Path, pattern: str) -> list[Path]:
    return sorted(
        d for d in gcp_root.glob(pattern)
        if (d / "measure" / "summary.json").is_file()
    )


def aggregate_condition(gcp_root: Path, pattern: str) -> dict:
    dirs = complete_dirs(gcp_root, pattern)
    if len(dirs) != 2:
        raise SystemExit(
            f"STOP — attesi 2 run completi per {pattern}, trovati {len(dirs)}"
        )

    rows = []
    for d in dirs:
        s = json.load(open(d / "measure" / "summary.json"))
        h50 = s["horizons"]["50"]
        rows.append({
            "amd64": int(h50.get("amd64_executions", 0)),
            "arm64": int(h50.get("arm64_executions", 0)),
            "h50_s": float(h50["cumulative_duration_ms"]) / 1000.0,
        })

    df = pd.DataFrame(rows)
    return {
        "dirs": [d.name for d in dirs],
        "amd64_mean": float(df["amd64"].mean()),
        "arm64_mean": float(df["arm64"].mean()),
        "h50_mean_s": float(df["h50_s"].mean()),
        "h50_std_s": float(df["h50_s"].std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("data/profiling"),
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

    profiling_root = args.profiling_root.resolve()
    gcp_root = profiling_root / "gcp-transfer-final"
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    membership_path = (
        profiling_root
        / "final-20260913-analysis-01"
        / "final-clustering-evidence-v2"
        / "kmeans-final-membership-t15.csv"
    )
    membership = pd.read_csv(membership_path)
    hist = membership[
        membership["function_name"].isin(["compression", "randomaccess"])
    ].set_index("function_name")

    compression_hist = hist.loc["compression"]
    randomaccess_hist = hist.loc["randomaccess"]

    replica_info = {}
    for rid in ["randomaccess-r02", "randomaccess-r03"]:
        selection = json.load(
            open(gcp_root / "replicas" / rid / "selection" / "selection.json")
        )
        prior_path = (
            gcp_root
            / "exploration-ablation"
            / f"{rid}-decoupled-materialized"
            / "frozen-prior.json"
        )
        prior = json.load(open(prior_path))

        replica_info[rid] = {
            "selected_donor": selection["selected_donor"]["function_name"],
            "distance": float(selection["selected_donor"]["distance"]),
            "cluster": int(selection["query"]["cluster_label"]),
            "prior_amd64": float(prior["arms"]["amd64"]["ucb1"]["mean_reward"]),
            "prior_arm64": float(prior["arms"]["arm64"]["ucb1"]["mean_reward"]),
            "prior_gap": float(
                prior["arms"]["amd64"]["ucb1"]["mean_reward"]
                - prior["arms"]["arm64"]["ucb1"]["mean_reward"]
            ),
            "source_amd64_n": int(
                prior["arms"]["amd64"]["source_real_observation_count"]
            ),
            "source_arm64_n": int(
                prior["arms"]["arm64"]["source_real_observation_count"]
            ),
            "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        }

    source_runs = []
    for d in sorted(gcp_root.glob("gcp-transfer-materialize-randomaccess-*")):
        log = d / "donor" / "update-reward.log"
        if not log.is_file():
            continue

        df = parse_donor_log(log)
        if len(df) != 30:
            continue

        a = df[df["arm"] == "amd64"]
        b = df[df["arm"] == "arm64"]

        source_runs.append({
            "dir": d,
            "amd64_n": len(a),
            "arm64_n": len(b),
            "amd64_duration_mean_ms": float(a["duration_ms"].mean()),
            "arm64_duration_mean_ms": float(b["duration_ms"].mean()),
            "amd64_reward_mean": float(a["reward"].mean()),
            "arm64_reward_mean": float(b["reward"].mean()),
            "reward_gap": float(a["reward"].mean() - b["reward"].mean()),
        })

    for rid, info in replica_info.items():
        matches = [
            r for r in source_runs
            if r["amd64_n"] == info["source_amd64_n"]
            and r["arm64_n"] == info["source_arm64_n"]
            and abs(r["reward_gap"] - info["prior_gap"]) < 1e-5
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"STOP — source donor run non univoco per {rid}: {len(matches)}"
            )
        info["source"] = matches[0]

    no_transfer_c0 = aggregate_condition(
        gcp_root,
        "gcp-transfer-control-randomaccess-no-transfer-c00-*",
    )
    improved_c0 = aggregate_condition(
        gcp_root,
        "gcp-transfer-materialized-randomaccess-decoupled-c00-*",
    )
    improved_c08 = aggregate_condition(
        gcp_root,
        "gcp-transfer-materialized-randomaccess-decoupled-c08-*",
    )

    # Check that paired c=0 / c=.8 improved runs use exactly the same priors.
    c00 = complete_dirs(
        gcp_root,
        "gcp-transfer-materialized-randomaccess-decoupled-c00-*",
    )
    c08 = complete_dirs(
        gcp_root,
        "gcp-transfer-materialized-randomaccess-decoupled-c08-*",
    )
    sha00 = sorted(
        hashlib.sha256((d / "transfer" / "frozen-prior.json").read_bytes()).hexdigest()
        for d in c00
    )
    sha08 = sorted(
        hashlib.sha256((d / "transfer" / "frozen-prior.json").read_bytes()).hexdigest()
        for d in c08
    )
    if sha00 != sha08:
        raise SystemExit("STOP — c=0 e c=.8 non usano gli stessi prior")

    evidence_rows = []
    for rid, info in replica_info.items():
        s = info["source"]
        evidence_rows.append({
            "replica": rid.replace("randomaccess-", "").upper(),
            "selected_donor": info["selected_donor"],
            "donor_distance": info["distance"],
            "cluster": info["cluster"],
            "donor_amd64_n": s["amd64_n"],
            "donor_arm64_n": s["arm64_n"],
            "donor_amd64_duration_mean_ms": s["amd64_duration_mean_ms"],
            "donor_arm64_duration_mean_ms": s["arm64_duration_mean_ms"],
            "donor_amd64_reward_mean": s["amd64_reward_mean"],
            "donor_arm64_reward_mean": s["arm64_reward_mean"],
            "donor_reward_gap_amd64_minus_arm64": s["reward_gap"],
            "prior_amd64_mean_reward": info["prior_amd64"],
            "prior_arm64_mean_reward": info["prior_arm64"],
            "prior_reward_gap_amd64_minus_arm64": info["prior_gap"],
            "gap_preservation_error": info["prior_gap"] - s["reward_gap"],
            "prior_sha256": info["prior_sha256"],
            "source_run": s["dir"].name,
        })

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv(out_dir / "donor_prior_evidence.csv", index=False)

    # 14
    vals = [
        float(compression_hist["architecture_delta_percent"]),
        float(randomaccess_hist["architecture_delta_percent"]),
    ]
    labels = ["Donor: compression", "Target: randomaccess"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(np.arange(2), vals)
    ax.axhline(0, linewidth=1)
    ax.axhline(15, linestyle="--", linewidth=1)
    ax.axhline(-15, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("Δ architetturale ARM vs x86 (%)")
    ax.set_title("Donor e target hanno effetti architetturali diversi")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, v in zip(bars, vals):
        if v >= 0:
            y, va = v + 1.8, "bottom"
        else:
            y, va = v / 2, "center"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            y,
            f"{v:+.1f}%",
            ha="center",
            va=va,
            fontweight="bold",
        )

    ax.text(
        0.01, -0.20,
        "Segno positivo: ARM più lento di x86. Segno negativo: ARM più veloce. "
        "Linee tratteggiate: soglia ±15%.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "14_donor_target_architecture_effect")

    # 15
    cats = []
    values = []
    for rid in ["randomaccess-r02", "randomaccess-r03"]:
        short = rid.split("-")[-1].upper()
        info = replica_info[rid]
        cats += [f"{short}\nDonor live", f"{short}\nPrior trasferito"]
        values += [info["source"]["reward_gap"], info["prior_gap"]]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(np.arange(len(cats)), values)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(cats)), cats)
    ax.set_ylabel("Vantaggio reward amd64 − arm64")
    ax.set_title("Dal donor al prior: il vantaggio relativo viene preservato")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            v + 0.0012,
            f"{v:.5f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.text(
        0.01, -0.20,
        "Valore > 0: il donor/prior favorisce amd64. "
        "Il prior è ancorato al target ma conserva il gap relativo del donor.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "15_donor_to_prior_reward_gap")

    # 16
    conditions = [
        ("No Transfer\nc=0", no_transfer_c0),
        ("Improved Transfer\nc=0", improved_c0),
        ("Improved Transfer\nc=0.8", improved_c08),
    ]
    amd = np.array([c[1]["amd64_mean"]/50*100 for c in conditions])
    arm = np.array([c[1]["arm64_mean"]/50*100 for c in conditions])
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(x, amd, label="amd64")
    ax.bar(x, arm, bottom=amd, label="arm64")
    ax.set_xticks(x, [c[0] for c in conditions])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Quota delle esecuzioni (%)")
    ax.set_title("Influenza del prior e correzione tramite esplorazione")
    ax.legend()

    for i in range(3):
        if amd[i] >= 15:
            ax.text(x[i], amd[i]/2, f"amd64\n{amd[i]:.0f}%", ha="center", va="center")
        elif amd[i] > 0:
            ax.text(x[i], 7, f"amd64 {amd[i]:.0f}%", ha="center", va="center", fontweight="bold")

        if arm[i] >= 15:
            ax.text(
                x[i],
                amd[i] + arm[i]/2,
                f"arm64\n{arm[i]:.0f}%",
                ha="center",
                va="center",
            )
        elif arm[i] > 0:
            ax.text(x[i], 93, f"arm64 {arm[i]:.0f}%", ha="center", va="center", fontweight="bold")

    ax.text(
        0.01, -0.20,
        "Media su R02–R03. Improved c=0 e c=0.8 usano, per replica, "
        "lo stesso prior materializzato (SHA identico).",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, out_dir, "16_prior_influence_and_online_correction")

    # 17
    h50 = [
        no_transfer_c0["h50_mean_s"],
        improved_c0["h50_mean_s"],
        improved_c08["h50_mean_s"],
    ]
    h50_std = [
        no_transfer_c0["h50_std_s"],
        improved_c0["h50_std_s"],
        improved_c08["h50_std_s"],
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    bars = ax.bar(x, h50, yerr=h50_std, capsize=4)
    ax.set_xticks(x, [c[0] for c in conditions])
    ax.set_ylabel("Durata cumulativa H50 (s)")
    ax.set_title("Effetto prestazionale del prior e sua correzione online")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, v in zip(bars, h50):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            v+1.0,
            f"{v:.2f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    save(fig, out_dir, "17_prior_influence_h50_and_recovery")

    donor_x86 = np.mean(
        [replica_info[r]["source"]["amd64_duration_mean_ms"] for r in replica_info]
    )
    donor_arm = np.mean(
        [replica_info[r]["source"]["arm64_duration_mean_ms"] for r in replica_info]
    )
    donor_delta_live = (donor_arm-donor_x86)/donor_x86*100
    prior_gap_mean = np.mean([replica_info[r]["prior_gap"] for r in replica_info])

    # 18
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.axis("off")

    steps = [
        (
            0.12,
            "1. Donor selezionato\ncompression\nrank #1 in R02 e R03\nstesso cluster del target",
        ),
        (
            0.38,
            f"2. Segnale trasferito\ndonor live: amd64 ≈ {donor_delta_live:.1f}% più veloce\n"
            f"prior: reward amd64 > arm64\ngap medio = {prior_gap_mean:.5f}",
        ),
        (
            0.65,
            "3. Target senza bonus di esplorazione\nNo Transfer, c=0: 2% amd64\n"
            "Improved, c=0: 100% amd64\n→ il prior cambia la traiettoria",
        ),
        (
            0.89,
            "4. Correzione online\nstesso prior, c=0.8\n5% amd64 / 95% ARM\n"
            "→ feedback + esplorazione correggono il bias",
        ),
    ]

    for xpos, txt in steps:
        ax.text(
            xpos, 0.52, txt,
            ha="center", va="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fill=False),
            fontsize=9.5,
        )

    for left, right in zip(steps[:-1], steps[1:]):
        ax.annotate(
            "",
            xy=(right[0]-0.105, 0.52),
            xytext=(left[0]+0.105, 0.52),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

    ax.text(
        0.5, 0.13,
        f"Ground truth storico: compression = {compression_hist['architecture_delta_percent']:+.1f}% "
        f"(architecture-independent a τ=15%); randomaccess = "
        f"{randomaccess_hist['architecture_delta_percent']:+.1f}% (ARM-preferred).",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
    )
    ax.set_title(
        "Catena di evidenza: donor → prior → comportamento del target",
        fontsize=14,
        pad=18,
    )
    fig.tight_layout()
    save(fig, out_dir, "18_donor_prior_target_evidence_chain")

    print("PASS — donor transfer evidence generated")
    print(f"output={out_dir}")
    print(evidence_df.to_string(index=False))


if __name__ == "__main__":
    main()
