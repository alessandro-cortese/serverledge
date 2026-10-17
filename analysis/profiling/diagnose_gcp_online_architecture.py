#!/usr/bin/env python3
"""
Serverledge — diagnostic of realized online architecture performance in the
final accepted GCP campaign.

This is a POST-HOC diagnostic. It does not alter donor selection or the
experimental results.

What it does
------------
1. Discovers only the 16 accepted final runs:
   - frozen runtime commit
   - 50 requests
   - all warm
   - fallback_count == 0
   - exactly one run for every target x condition
2. Reads measure/requests.csv, whose observed endpoint is client_elapsed_ms.
3. Computes per-run x86/ARM descriptive statistics.
4. Pools ONLY the two No-TL runs (c=.2 and c=.4) for a less transfer-biased
   view of the online architecture behavior.
5. Compares that online diagnostic with the frozen offline ground truth.
6. Prints the request-level sequence for graph-bfs, because that is the most
   interesting apparent mismatch.

Important
---------
The offline ground truth uses function duration_ms, whereas requests.csv uses
end-to-end client_elapsed_ms. Therefore the online comparison is diagnostic,
not a replacement for the frozen ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_RUNTIME = "4c9f427a76e188c9cb026d0b65ccf44b4d116d6f"

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

NOTL_CONDITIONS = [
    "notl-difference",
    "notl-ratio",
]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def discover_runs(root: Path) -> pd.DataFrame:
    rows = []

    for manifest_path in root.glob("runs/**/manifest.json"):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue

        target = str(manifest.get("target_function", ""))
        condition = str(manifest.get("condition", ""))

        if target not in TARGETS or condition not in CONDITIONS:
            continue

        if str(manifest.get("serverledge_commit", "")) != FINAL_RUNTIME:
            continue
        if int(manifest.get("request_count", -1)) != 50:
            continue
        if manifest.get("all_warm") is not True:
            continue
        if int(manifest.get("fallback_count", -1)) != 0:
            continue

        req_path = manifest_path.parent / "measure" / "requests.csv"
        summary_path = manifest_path.parent / "measure" / "summary.json"

        if not req_path.is_file():
            raise SystemExit(f"missing requests.csv: {req_path}")
        if not summary_path.is_file():
            raise SystemExit(f"missing summary.json: {summary_path}")

        rows.append({
            "target_function": target,
            "condition": condition,
            "run_id": manifest.get("run_id"),
            "manifest_path": str(manifest_path),
            "requests_path": str(req_path),
            "summary_path": str(summary_path),
        })

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit(f"no accepted final runs under {root}")

    counts = (
        runs.groupby(["target_function", "condition"])
        .size()
        .reset_index(name="n")
    )
    bad = counts[counts["n"] != 1]
    if not bad.empty:
        raise SystemExit(
            "expected exactly one accepted run per target/condition:\n"
            + bad.to_string(index=False)
        )

    expected = {(t, c) for t in TARGETS for c in CONDITIONS}
    got = set(zip(runs["target_function"], runs["condition"]))
    if got != expected:
        raise SystemExit(
            f"run matrix mismatch: missing={sorted(expected-got)}, "
            f"extra={sorted(got-expected)}"
        )

    return runs.sort_values(
        ["target_function", "condition"]
    ).reset_index(drop=True)


def load_requests(runs: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for row in runs.itertuples(index=False):
        x = pd.read_csv(row.requests_path)

        required = {
            "request_index",
            "timestamp_utc",
            "client_elapsed_ms",
            "node_arch",
            "success",
        }
        missing = required - set(x.columns)
        if missing:
            raise SystemExit(
                f"{row.requests_path}: missing columns {sorted(missing)}"
            )

        if len(x) != 50:
            raise SystemExit(
                f"{row.requests_path}: expected 50 rows, got {len(x)}"
            )

        x = x.copy()
        x["target_function"] = row.target_function
        x["condition"] = row.condition
        x["run_id"] = row.run_id
        x["client_elapsed_ms"] = pd.to_numeric(
            x["client_elapsed_ms"], errors="raise"
        )
        x["request_index"] = pd.to_numeric(
            x["request_index"], errors="raise"
        ).astype(int)

        frames.append(x)

    return pd.concat(frames, ignore_index=True)


def summarize_by_run_arm(req: pd.DataFrame) -> pd.DataFrame:
    def q10(s):
        return float(s.quantile(0.10))

    def q90(s):
        return float(s.quantile(0.90))

    out = (
        req.groupby(
            ["target_function", "condition", "node_arch"],
            as_index=False,
        )
        .agg(
            n=("client_elapsed_ms", "size"),
            mean_ms=("client_elapsed_ms", "mean"),
            median_ms=("client_elapsed_ms", "median"),
            std_ms=("client_elapsed_ms", "std"),
            p10_ms=("client_elapsed_ms", q10),
            p90_ms=("client_elapsed_ms", q90),
            first_request=("request_index", "min"),
            last_request=("request_index", "max"),
        )
    )

    return out


def pooled_notl(req: pd.DataFrame) -> pd.DataFrame:
    x = req[req["condition"].isin(NOTL_CONDITIONS)].copy()

    stats = (
        x.groupby(
            ["target_function", "node_arch"],
            as_index=False,
        )
        .agg(
            n=("client_elapsed_ms", "size"),
            mean_ms=("client_elapsed_ms", "mean"),
            median_ms=("client_elapsed_ms", "median"),
            std_ms=("client_elapsed_ms", "std"),
        )
    )

    return stats


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path)

    required = {
        "function_name",
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
        "architecture_preference",
    }
    missing = required - set(gt.columns)
    if missing:
        raise SystemExit(
            f"ground truth missing columns: {sorted(missing)}"
        )

    gt = gt[gt["function_name"].isin(TARGETS)].copy()

    if len(gt) != len(TARGETS):
        raise SystemExit(
            f"expected {len(TARGETS)} ground-truth rows, got {len(gt)}"
        )

    return gt


def make_arch_diagnostic(
        gt: pd.DataFrame,
        notl: pd.DataFrame,
) -> pd.DataFrame:
    amd = notl[notl["node_arch"] == "amd64"].copy()
    arm = notl[notl["node_arch"] == "arm64"].copy()

    amd = amd.rename(columns={
        "n": "online_amd64_n",
        "mean_ms": "online_amd64_mean_ms",
        "median_ms": "online_amd64_median_ms",
        "std_ms": "online_amd64_std_ms",
    }).drop(columns=["node_arch"])

    arm = arm.rename(columns={
        "n": "online_arm64_n",
        "mean_ms": "online_arm64_mean_ms",
        "median_ms": "online_arm64_median_ms",
        "std_ms": "online_arm64_std_ms",
    }).drop(columns=["node_arch"])

    x = (
        gt.rename(columns={"function_name": "target_function"})
        .merge(amd, on="target_function", how="left", validate="one_to_one")
        .merge(arm, on="target_function", how="left", validate="one_to_one")
    )

    x["online_arm_vs_x86_mean_delta_percent"] = (
            100.0
            * (
                    x["online_arm64_mean_ms"]
                    - x["online_amd64_mean_ms"]
            )
            / x["online_amd64_mean_ms"]
    )

    x["online_arm_vs_x86_median_delta_percent"] = (
            100.0
            * (
                    x["online_arm64_median_ms"]
                    - x["online_amd64_median_ms"]
            )
            / x["online_amd64_median_ms"]
    )

    def sign_pref(delta: float) -> str:
        if pd.isna(delta):
            return "unknown"
        if delta < 0:
            return "arm-faster"
        if delta > 0:
            return "x86-faster"
        return "equal"

    x["online_pref_by_mean"] = (
        x["online_arm_vs_x86_mean_delta_percent"].map(sign_pref)
    )
    x["online_pref_by_median"] = (
        x["online_arm_vs_x86_median_delta_percent"].map(sign_pref)
    )

    x["online_low_sample_warning"] = (
            (x["online_amd64_n"] < 10)
            | (x["online_arm64_n"] < 10)
    )

    cols = [
        "target_function",
        "architecture_preference",
        "x86_duration_ms",
        "arm_duration_ms",
        "arm_vs_x86_delta_percent",
        "online_amd64_n",
        "online_arm64_n",
        "online_amd64_mean_ms",
        "online_arm64_mean_ms",
        "online_arm_vs_x86_mean_delta_percent",
        "online_pref_by_mean",
        "online_amd64_median_ms",
        "online_arm64_median_ms",
        "online_arm_vs_x86_median_delta_percent",
        "online_pref_by_median",
        "online_low_sample_warning",
    ]
    return x[cols].sort_values("target_function")


def make_sequence(req: pd.DataFrame) -> pd.DataFrame:
    # All four graph-bfs accepted runs, request by request.
    x = req[req["target_function"] == "graph-bfs"].copy()
    return x[
        [
            "condition",
            "request_index",
            "node_arch",
            "client_elapsed_ms",
            "timestamp_utc",
        ]
    ].sort_values(
        ["condition", "request_index"]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gcp-root", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.gcp_root.resolve())
    req = load_requests(runs)
    by_run_arm = summarize_by_run_arm(req)
    notl = pooled_notl(req)
    gt = load_ground_truth(args.ground_truth.resolve())
    diag = make_arch_diagnostic(gt, notl)
    seq = make_sequence(req)

    runs.to_csv(out / "accepted_runs.csv", index=False)
    by_run_arm.to_csv(out / "per_run_arch_stats.csv", index=False)
    notl.to_csv(out / "pooled_notl_arch_stats.csv", index=False)
    diag.to_csv(out / "offline_vs_online_architecture.csv", index=False)
    seq.to_csv(out / "graph_bfs_request_sequence.csv", index=False)

    print("PASS — GCP online architecture diagnostic completed")
    print()
    print("===== POOLED No-TL ARCHITECTURE DIAGNOSTIC =====")
    print(
        diag.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )

    print()
    print("===== PER-RUN / PER-ARCH STATS =====")
    print(
        by_run_arm.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )

    print()
    print("===== GRAPH-BFS REQUEST SEQUENCE =====")
    print(
        seq.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        )
    )

    print()
    print("NOTE:")
    print(
        "Offline ground truth uses function duration_ms; "
        "online diagnostic uses end-to-end client_elapsed_ms."
    )
    print(
        "Treat online arm deltas as post-hoc diagnostics, "
        "not as a replacement for the frozen ground truth."
    )

    print()
    for name in [
        "accepted_runs.csv",
        "per_run_arch_stats.csv",
        "pooled_notl_arch_stats.csv",
        "offline_vs_online_architecture.csv",
        "graph_bfs_request_sequence.csv",
    ]:
        print(out / name)


if __name__ == "__main__":
    main()
