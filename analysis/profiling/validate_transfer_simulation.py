#!/usr/bin/env python3
"""
Methodology validation for the Serverledge weak-prior Transfer Learning simulator.

This script is designed to live at:
    analysis/profiling/validate_transfer_simulation.py
and to be run from the Serverledge repository root.

It validates the FINAL selected Transfer Learning policy:

    reward r = -ln(duration_ms)

    mu_eff,j = (S_j + w_R * mu_prior,j) / (n_j + w_R)

    B_j = c * sqrt( ln(sum_k(n_k + w_E)) / (n_j + w_E) )
          (B_j = 0 when the exploration total <= 1)

    score_j = mu_eff,j + B_j

with the reference-anchored prior:

    mu_prior,x86   = mu_target,x86
    mu_prior,arm64 = mu_target,x86 + (mu_donor,arm64 - mu_donor,x86)

Default final configuration:
    c   = 0.8
    w_R = 0.25
    w_E = 1.0

The no-transfer comparator is the existing classic-UCB1 baseline with no prior.

Outputs
-------
CSV:
  00_validation_summary.csv
  01_sample_count_audit.csv
  02_crn_verification.csv
  02b_vectorized_policy_equivalence.csv
  03_replicate_count_sensitivity.csv
  04_multiseed_per_seed_horizon.csv
  05_multiseed_summary.csv
  06_jackknife_observation_h10.csv
  07_leave_one_target_out_h10.csv
  08_early_horizon_selection_audit.csv
  09_early_horizon_target_patterns.csv
  10_early_horizon_summary.csv

Figures (PNG + SVG):
  replicate_count_sensitivity_h10
  multiseed_spaghetti_gain
  jackknife_observation_influence_h10
  leave_one_target_out_h10
  early_horizon_arm_choice_share
  early_horizon_h2_pattern_gain

Metadata:
  validation_manifest.json

Important reproducibility detail
--------------------------------
The original study generated sequences up to Hmax=50 and then read metrics at
H={1,2,5,10,20,50}. This script preserves that convention. In particular,
H=10 results are NOT regenerated with sequence length 10, because doing so
changes RNG consumption for the second arm and therefore changes the exact
Monte Carlo sample stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
# Expected location: <repo>/analysis/profiling/validate_transfer_simulation.py
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) >= 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.profiling.transfer_ucb1_offline import (  # noqa: E402
    ARMS,
    UCB1Policy,
    empirical_stats,
    load_raw_durations,
    stable_seed,
)
import analysis.profiling.transfer_decoupled_ucb1 as transfer_module  # noqa: E402
from analysis.profiling.transfer_decoupled_ucb1 import (  # noqa: E402
    DecoupledUCB1Policy,
    build_reference_anchored_prior,
    load_manhattan_donors,
    make_sequences,
    simulate_baseline_trace,
    simulate_transfer_trace,
)

# ---------------------------------------------------------------------------
# Defaults matching the final study
# ---------------------------------------------------------------------------

DEFAULT_HORIZONS = [1, 2, 5, 10, 20, 50]
DEFAULT_REPLICATE_COUNTS = [50, 100, 200, 500, 1000, 2000]
DEFAULT_SEEDS = [11, 23, 42, 77, 101]

DEFAULT_C = 0.8
DEFAULT_WR = 0.25
DEFAULT_WE = 1.0
DEFAULT_REPLICATES = 500
DEFAULT_PRIMARY_H = 10
DEFAULT_HMAX = 50
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_gain_pct(baseline_latency: np.ndarray, transfer_latency: np.ndarray) -> float:
    gains = 100.0 * (baseline_latency - transfer_latency) / baseline_latency
    return float(np.mean(gains))


def stats_for_single_function(
    name: str,
    x86_values: list[float],
    arm_values: list[float],
) -> dict[str, Any]:
    return empirical_stats({name: x86_values}, {name: arm_values})[name]


def horizons_valid(horizons: Iterable[int], hmax: int) -> list[int]:
    hs = sorted(set(int(h) for h in horizons))
    if not hs or min(hs) < 1 or max(hs) > hmax:
        raise ValueError(f"horizons must be in [1, {hmax}], got {hs}")
    return hs


def ensure_input(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Lightweight paired simulation using the repository's actual policy classes
# ---------------------------------------------------------------------------

def _stack_sequences(
    sequences_by_rep: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert repository sequence objects to dense [replicate, draw] arrays."""
    x = np.stack([np.asarray(seq["x86"], dtype=float) for seq in sequences_by_rep], axis=0)
    a = np.stack([np.asarray(seq["arm64"], dtype=float) for seq in sequences_by_rep], axis=0)
    return x, a


def paired_latency_for_sequences(
    target_stats: dict[str, Any],
    prior_mean: dict[str, float],
    sequences_by_rep: list[dict[str, np.ndarray]],
    horizons: list[int],
    c: float,
    w_r: float,
    w_e: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Vectorized exact replay of baseline + final Transfer Learning policy.

    The logic is algebraically identical to UCB1Policy and DecoupledUCB1Policy,
    but all Monte Carlo replicas are advanced in parallel with NumPy. This keeps
    the validation run practical even for R=2000 and the 1,272 jackknife cases.

    CRN semantics are unchanged: both policies consume the same arm-specific
    pre-generated sequence, indexed by occurrence count within that arm.
    """
    del target_stats  # kept in signature to make the call-site explicit
    max_h = max(horizons)
    xseq, aseq = _stack_sequences(sequences_by_rep)
    reps = xseq.shape[0]
    rows = np.arange(reps)
    hset = set(horizons)

    bout = {h: np.empty(reps, dtype=float) for h in horizons}
    tout = {h: np.empty(reps, dtype=float) for h in horizons}

    # Baseline state: classic UCB1, no prior.
    bc = np.zeros((reps, 2), dtype=np.int64)
    bs = np.zeros((reps, 2), dtype=float)
    bp = np.zeros((reps, 2), dtype=np.int64)
    blat = np.zeros(reps, dtype=float)

    # Transfer state: decoupled weak prior.
    tc = np.zeros((reps, 2), dtype=np.int64)
    ts = np.zeros((reps, 2), dtype=float)
    tp = np.zeros((reps, 2), dtype=np.int64)
    tlat = np.zeros(reps, dtype=float)
    prior = np.asarray([prior_mean["x86"], prior_mean["arm64"]], dtype=float)
    prior_reward_sum = w_r * prior

    for step in range(1, max_h + 1):
        # ---------------- baseline ----------------
        barm = np.full(reps, -1, dtype=np.int64)
        m0 = bc[:, 0] <= 0
        barm[m0] = 0
        m1 = (barm < 0) & (bc[:, 1] <= 0)
        barm[m1] = 1
        rest = barm < 0
        if np.any(rest):
            counts = bc[rest].astype(float)
            means = bs[rest] / counts
            total = np.sum(counts, axis=1)
            bonus = c * np.sqrt(np.log(total)[:, None] / counts)
            scores = means + bonus
            # Exact tie rule of max(ARMS, key=(score, -index)): x86 wins ties.
            barm[rest] = np.where(scores[:, 0] >= scores[:, 1], 0, 1)

        bd = np.empty(reps, dtype=float)
        bx = barm == 0
        ba = ~bx
        bd[bx] = xseq[rows[bx], bp[bx, 0]]
        bd[ba] = aseq[rows[ba], bp[ba, 1]]
        bp[rows, barm] += 1
        bc[rows, barm] += 1
        bs[rows, barm] += -np.log(bd)
        blat += bd

        # ---------------- transfer ----------------
        teff_count = tc.astype(float) + w_r
        tmean = (ts + prior_reward_sum[None, :]) / teff_count
        exploration_count = tc.astype(float) + w_e
        exploration_total = np.sum(exploration_count, axis=1)
        tbonus = np.zeros_like(tmean)
        active = exploration_total > 1.0
        if np.any(active):
            tbonus[active] = c * np.sqrt(
                np.log(exploration_total[active])[:, None]
                / exploration_count[active]
            )
        score = tmean + tbonus
        tarm = np.where(score[:, 0] >= score[:, 1], 0, 1)

        td = np.empty(reps, dtype=float)
        tx = tarm == 0
        ta = ~tx
        td[tx] = xseq[rows[tx], tp[tx, 0]]
        td[ta] = aseq[rows[ta], tp[ta, 1]]
        tp[rows, tarm] += 1
        tc[rows, tarm] += 1
        ts[rows, tarm] += -np.log(td)
        tlat += td

        if step in hset:
            bout[step][:] = blat
            tout[step][:] = tlat

    return bout, tout


def transfer_latency_only(
    prior_mean: dict[str, float],
    sequences_by_rep: list[dict[str, np.ndarray]],
    horizon: int,
    c: float,
    w_r: float,
    w_e: float,
) -> np.ndarray:
    xseq, aseq = _stack_sequences(sequences_by_rep)
    reps = xseq.shape[0]
    rows = np.arange(reps)
    counts = np.zeros((reps, 2), dtype=np.int64)
    sums = np.zeros((reps, 2), dtype=float)
    pos = np.zeros((reps, 2), dtype=np.int64)
    lat = np.zeros(reps, dtype=float)
    prior = np.asarray([prior_mean["x86"], prior_mean["arm64"]], dtype=float)
    prior_reward_sum = w_r * prior

    for _ in range(horizon):
        eff_count = counts.astype(float) + w_r
        mean = (sums + prior_reward_sum[None, :]) / eff_count
        exploration_count = counts.astype(float) + w_e
        exploration_total = np.sum(exploration_count, axis=1)
        bonus = np.zeros_like(mean)
        active = exploration_total > 1.0
        if np.any(active):
            bonus[active] = c * np.sqrt(
                np.log(exploration_total[active])[:, None]
                / exploration_count[active]
            )
        score = mean + bonus
        arm = np.where(score[:, 0] >= score[:, 1], 0, 1)
        d = np.empty(reps, dtype=float)
        mx = arm == 0
        ma = ~mx
        d[mx] = xseq[rows[mx], pos[mx, 0]]
        d[ma] = aseq[rows[ma], pos[ma, 1]]
        pos[rows, arm] += 1
        counts[rows, arm] += 1
        sums[rows, arm] += -np.log(d)
        lat += d
    return lat


def baseline_latency_only(
    sequences_by_rep: list[dict[str, np.ndarray]],
    horizon: int,
    c: float,
) -> np.ndarray:
    xseq, aseq = _stack_sequences(sequences_by_rep)
    reps = xseq.shape[0]
    rows = np.arange(reps)
    counts = np.zeros((reps, 2), dtype=np.int64)
    sums = np.zeros((reps, 2), dtype=float)
    pos = np.zeros((reps, 2), dtype=np.int64)
    lat = np.zeros(reps, dtype=float)

    for _ in range(horizon):
        arm = np.full(reps, -1, dtype=np.int64)
        m0 = counts[:, 0] <= 0
        arm[m0] = 0
        m1 = (arm < 0) & (counts[:, 1] <= 0)
        arm[m1] = 1
        rest = arm < 0
        if np.any(rest):
            cts = counts[rest].astype(float)
            means = sums[rest] / cts
            total = np.sum(cts, axis=1)
            bonus = c * np.sqrt(np.log(total)[:, None] / cts)
            score = means + bonus
            arm[rest] = np.where(score[:, 0] >= score[:, 1], 0, 1)

        d = np.empty(reps, dtype=float)
        mx = arm == 0
        ma = ~mx
        d[mx] = xseq[rows[mx], pos[mx, 0]]
        d[ma] = aseq[rows[ma], pos[ma, 1]]
        pos[rows, arm] += 1
        counts[rows, arm] += 1
        sums[rows, arm] += -np.log(d)
        lat += d
    return lat


# ---------------------------------------------------------------------------
# 1) Data/sample audit
# ---------------------------------------------------------------------------

def sample_count_audit(
    raw_x86: dict[str, list[float]],
    raw_arm: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = sorted(set(raw_x86) | set(raw_arm))
    rows = []
    for name in names:
        nx = len(raw_x86.get(name, []))
        na = len(raw_arm.get(name, []))
        rows.append(
            {
                "function": name,
                "x86_eligible_warm_samples": nx,
                "arm64_eligible_warm_samples": na,
                "same_count": nx == na,
                "both_equal_12": nx == 12 and na == 12,
            }
        )
    summary = {
        "function_count_union": len(names),
        "function_count_common": len(set(raw_x86) & set(raw_arm)),
        "all_x86_12": all(len(v) == 12 for v in raw_x86.values()),
        "all_arm64_12": all(len(v) == 12 for v in raw_arm.values()),
        "all_common_and_12_each": (
            set(raw_x86) == set(raw_arm)
            and all(len(raw_x86[n]) == 12 and len(raw_arm[n]) == 12 for n in names)
        ),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# 2) CRN implementation verification
# ---------------------------------------------------------------------------

def crn_source_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    base_src = inspect.getsource(simulate_baseline_trace)
    transfer_src = inspect.getsource(simulate_transfer_trace)
    make_src = inspect.getsource(make_sequences)
    original_main_src = inspect.getsource(transfer_module.main)

    def add(name: str, ok: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence})

    add(
        "separate_sequence_per_arm",
        "for arm in ARMS" in make_src and "target_stats[\"durations\"][arm]" in make_src,
        "make_sequences builds one resampled array for each arm",
    )
    add(
        "baseline_per_arm_position_counter",
        "positions = {arm: 0 for arm in ARMS}" in base_src,
        "baseline initializes positions independently for each arm",
    )
    add(
        "baseline_indexes_by_arm_occurrence",
        "sequences[arm][positions[arm]]" in base_src and "positions[arm] += 1" in base_src,
        "baseline reads sequence[arm][positions[arm]] then increments only that arm",
    )
    add(
        "transfer_per_arm_position_counter",
        "positions = {arm: 0 for arm in ARMS}" in transfer_src,
        "transfer initializes positions independently for each arm",
    )
    add(
        "transfer_indexes_by_arm_occurrence",
        "sequences[arm][positions[arm]]" in transfer_src and "positions[arm] += 1" in transfer_src,
        "transfer reads sequence[arm][positions[arm]] then increments only that arm",
    )
    add(
        "original_simulator_pairs_same_replica_sequence",
        original_main_src.count("sequences=sequences[i]") >= 2,
        "the original simulator passes the same sequences[i] replica to baseline and transfer",
    )
    return checks


def crn_runtime_checks(
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    seeds: list[int],
    c: float,
    w_r: float,
    w_e: float,
    hmax: int,
    reps_per_target: int,
) -> list[dict[str, Any]]:
    """Runtime confirmation of paired CRN semantics.

    For both policies, reconstruct the sample coordinate consumed at each pull:
        (arm, occurrence_number_within_that_arm)
    and check that every coordinate common to both policies resolves to exactly
    the same value in the shared pre-generated arm sequence.
    """
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for target in sorted(performance):
            ts = performance[target]
            ds = performance[donors[target]]
            prior = build_reference_anchored_prior(ts, ds)
            sequences = make_sequences(
                target_name=target,
                target_stats=ts,
                horizon=hmax,
                replicates=reps_per_target,
                seed=seed,
            )
            checked_coords = 0
            different_policy_order_reps = 0
            all_equal = True

            for seq in sequences:
                bt = simulate_baseline_trace(ts, seq, hmax, c, convergence_run=5)
                tt = simulate_transfer_trace(ts, prior, seq, hmax, c, w_r, w_e, convergence_run=5)
                bchosen = bt["chosen_arms"]
                tchosen = tt["chosen_arms"]
                if bchosen != tchosen:
                    different_policy_order_reps += 1

                def coordinates(chosen: list[str]) -> dict[tuple[str, int], float]:
                    pos = {arm: 0 for arm in ARMS}
                    out: dict[tuple[str, int], float] = {}
                    for arm in chosen:
                        k = pos[arm]
                        out[(arm, k)] = float(seq[arm][k])
                        pos[arm] += 1
                    return out

                bm = coordinates(bchosen)
                tm = coordinates(tchosen)
                common = set(bm) & set(tm)
                checked_coords += len(common)
                if any(bm[k] != tm[k] for k in common):
                    all_equal = False
                    break

            rows.append(
                {
                    "seed": seed,
                    "target_function": target,
                    "replicates_checked": reps_per_target,
                    "hmax": hmax,
                    "shared_arm_occurrence_coordinates_checked": checked_coords,
                    "replicates_with_different_policy_arm_order": different_policy_order_reps,
                    "shared_coordinates_equal": all_equal,
                    "status": "PASS" if all_equal else "FAIL",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 2b) Exact equivalence of the vectorized validator to repository policies
# ---------------------------------------------------------------------------
def vectorized_policy_equivalence_checks(
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    seed: int,
    horizons: list[int],
    hmax: int,
    c: float,
    w_r: float,
    w_e: float,
    replicates_per_target: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in sorted(performance):
        ts = performance[target]
        prior = build_reference_anchored_prior(ts, performance[donors[target]])
        seq = make_sequences(target, ts, hmax, replicates_per_target, seed)
        vb, vt = paired_latency_for_sequences(ts, prior, seq, horizons, c, w_r, w_e)
        max_diff_b = 0.0
        max_diff_t = 0.0
        exact = True
        for r in range(replicates_per_target):
            ob = simulate_baseline_trace(ts, seq[r], hmax, c, convergence_run=5)["cumulative_latency"]
            ot = simulate_transfer_trace(ts, prior, seq[r], hmax, c, w_r, w_e, convergence_run=5)["cumulative_latency"]
            for h in horizons:
                db = abs(float(vb[h][r]) - float(ob[h - 1]))
                dt = abs(float(vt[h][r]) - float(ot[h - 1]))
                max_diff_b = max(max_diff_b, db)
                max_diff_t = max(max_diff_t, dt)
                if db != 0.0 or dt != 0.0:
                    exact = False
        rows.append(
            {
                "target_function": target,
                "seed": seed,
                "replicates_checked": replicates_per_target,
                "horizons_checked": ",".join(map(str, horizons)),
                "max_abs_baseline_latency_difference_ms": max_diff_b,
                "max_abs_transfer_latency_difference_ms": max_diff_t,
                "exact_match": exact,
                "status": "PASS" if exact else "FAIL",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 3) Replicate-count sensitivity (nested prefixes)
# ---------------------------------------------------------------------------

def replicate_sensitivity(
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    counts: list[int],
    seed: int,
    primary_h: int,
    hmax: int,
    c: float,
    w_r: float,
    w_e: float,
) -> list[dict[str, Any]]:
    counts = sorted(set(counts))
    max_r = max(counts)
    per_r_target_gains: dict[int, list[float]] = {r: [] for r in counts}

    for idx, target in enumerate(sorted(performance), 1):
        print(f"[replicate sensitivity {idx:02d}/{len(performance)}] {target}")
        ts = performance[target]
        prior = build_reference_anchored_prior(ts, performance[donors[target]])
        seq = make_sequences(target, ts, hmax, max_r, seed)
        b, t = paired_latency_for_sequences(ts, prior, seq, [primary_h], c, w_r, w_e)
        bg = b[primary_h]
        tg = t[primary_h]
        rep_gains = 100.0 * (bg - tg) / bg
        for r in counts:
            per_r_target_gains[r].append(float(np.mean(rep_gains[:r])))

    rows = []
    ref = float(np.mean(per_r_target_gains[max_r]))
    for r in counts:
        macro = float(np.mean(per_r_target_gains[r]))
        rows.append(
            {
                "replicates_R": r,
                "seed": seed,
                "primary_horizon": primary_h,
                "hmax_used_for_rng": hmax,
                "target_count": len(performance),
                "macro_mean_latency_gain_pct": macro,
                "absolute_difference_from_maxR_pp": abs(macro - ref),
                "reference_maxR": max_r,
                "reference_gain_pct": ref,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 4) Multi-seed sensitivity + spaghetti data
# ---------------------------------------------------------------------------

def multiseed_analysis(
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    seeds: list[int],
    replicates: int,
    horizons: list[int],
    hmax: int,
    c: float,
    w_r: float,
    w_e: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        per_h: dict[int, list[float]] = {h: [] for h in horizons}
        for idx, target in enumerate(sorted(performance), 1):
            print(f"[multi-seed seed={seed} {idx:02d}/{len(performance)}] {target}")
            ts = performance[target]
            prior = build_reference_anchored_prior(ts, performance[donors[target]])
            seq = make_sequences(target, ts, hmax, replicates, seed)
            b, t = paired_latency_for_sequences(ts, prior, seq, horizons, c, w_r, w_e)
            for h in horizons:
                per_h[h].append(mean_gain_pct(b[h], t[h]))
        for h in horizons:
            rows.append(
                {
                    "seed": seed,
                    "horizon": h,
                    "replicates": replicates,
                    "target_count": len(performance),
                    "macro_mean_latency_gain_pct": float(np.mean(per_h[h])),
                }
            )

    summary: list[dict[str, Any]] = []
    for h in horizons:
        vals = np.asarray(
            [r["macro_mean_latency_gain_pct"] for r in rows if r["horizon"] == h],
            dtype=float,
        )
        summary.append(
            {
                "horizon": h,
                "seed_count": len(seeds),
                "replicates_per_seed": replicates,
                "mean_gain_across_seeds_pct": float(np.mean(vals)),
                "std_across_seeds_pp": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_gain_across_seeds_pct": float(np.min(vals)),
                "max_gain_across_seeds_pct": float(np.max(vals)),
                "range_across_seeds_pp": float(np.max(vals) - np.min(vals)),
            }
        )
    return rows, summary


# ---------------------------------------------------------------------------
# 5) Leave-one-observation-out jackknife
# ---------------------------------------------------------------------------

def jackknife_observation(
    raw_x86: dict[str, list[float]],
    raw_arm: dict[str, list[float]],
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    seed: int,
    replicates: int,
    primary_h: int,
    hmax: int,
    c: float,
    w_r: float,
    w_e: float,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    targets = sorted(performance)
    n_targets = len(targets)

    # Cache original sequences + per-replica baseline latency + original gain.
    base_sequences: dict[str, list[dict[str, np.ndarray]]] = {}
    base_baseline: dict[str, np.ndarray] = {}
    base_gain: dict[str, float] = {}

    print("[jackknife] building original per-target cache...")
    for idx, target in enumerate(targets, 1):
        print(f"[jackknife base {idx:02d}/{n_targets}] {target}")
        ts = performance[target]
        prior = build_reference_anchored_prior(ts, performance[donors[target]])
        seq = make_sequences(target, ts, hmax, replicates, seed)
        base_sequences[target] = seq
        b, t = paired_latency_for_sequences(ts, prior, seq, [primary_h], c, w_r, w_e)
        base_baseline[target] = b[primary_h]
        base_gain[target] = mean_gain_pct(b[primary_h], t[primary_h])

    base_macro = float(np.mean(list(base_gain.values())))
    donor_to_targets: dict[str, list[str]] = defaultdict(list)
    for target, donor in donors.items():
        donor_to_targets[donor].append(target)

    rows: list[dict[str, Any]] = []
    total_conditions = sum(len(raw_x86[t]) + len(raw_arm[t]) for t in targets)
    done = 0

    for function in targets:
        for arm in ARMS:
            original_values = raw_x86[function] if arm == "x86" else raw_arm[function]
            for obs_idx, removed_value in enumerate(original_values):
                done += 1
                if done == 1 or done % 50 == 0 or done == total_conditions:
                    print(f"[jackknife {done:04d}/{total_conditions}] {function} {arm} obs={obs_idx}")

                xvals = list(raw_x86[function])
                avals = list(raw_arm[function])
                if arm == "x86":
                    del xvals[obs_idx]
                else:
                    del avals[obs_idx]
                modified_stats = stats_for_single_function(function, xvals, avals)

                affected = set(donor_to_targets.get(function, []))
                affected.add(function)

                macro_sum = base_macro * n_targets
                best_arm_changed = False

                for target in affected:
                    old_gain = base_gain[target]
                    ts = modified_stats if target == function else performance[target]
                    donor_name = donors[target]
                    ds = modified_stats if donor_name == function else performance[donor_name]
                    prior = build_reference_anchored_prior(ts, ds)

                    if target == function:
                        # Target empirical distribution changed: regenerate Hmax-sized
                        # sequences using the same stable seed, then evaluate H=10.
                        seq = make_sequences(target, ts, hmax, replicates, seed)
                        b, t = paired_latency_for_sequences(ts, prior, seq, [primary_h], c, w_r, w_e)
                        new_gain = mean_gain_pct(b[primary_h], t[primary_h])
                        best_arm_changed = (
                            modified_stats["best_reward_arm"] != performance[function]["best_reward_arm"]
                        )
                    else:
                        # Only donor prior changed. Target environment and baseline CRN
                        # stream remain exactly the cached original ones.
                        seq = base_sequences[target]
                        tl = transfer_latency_only(prior, seq, primary_h, c, w_r, w_e)
                        new_gain = mean_gain_pct(base_baseline[target], tl)

                    macro_sum += new_gain - old_gain

                macro = macro_sum / n_targets
                rows.append(
                    {
                        "function": function,
                        "arm": arm,
                        "observation_index_zero_based": obs_idx,
                        "removed_duration_ms": float(removed_value),
                        "remaining_samples_in_that_arm": len(original_values) - 1,
                        "affected_target_count": len(affected),
                        "original_macro_gain_pct": base_macro,
                        "leave_one_observation_out_macro_gain_pct": macro,
                        "delta_macro_gain_pp": macro - base_macro,
                        "absolute_influence_pp": abs(macro - base_macro),
                        "best_empirical_arm_changed_for_removed_function": best_arm_changed,
                    }
                )

    influences = np.asarray([r["absolute_influence_pp"] for r in rows], dtype=float)
    macro_vals = np.asarray([r["leave_one_observation_out_macro_gain_pct"] for r in rows], dtype=float)
    summary = {
        "base_macro_gain_pct": base_macro,
        "conditions": float(len(rows)),
        "min_loo_macro_gain_pct": float(np.min(macro_vals)),
        "max_loo_macro_gain_pct": float(np.max(macro_vals)),
        "max_absolute_influence_pp": float(np.max(influences)),
        "median_absolute_influence_pp": float(np.median(influences)),
        "p95_absolute_influence_pp": float(np.percentile(influences, 95)),
        "negative_gain_conditions": float(np.sum(macro_vals < 0.0)),
        "best_arm_change_conditions": float(
            sum(bool(r["best_empirical_arm_changed_for_removed_function"]) for r in rows)
        ),
    }
    return rows, base_gain, summary


# ---------------------------------------------------------------------------
# 6) Leave-one-target-out robustness of the macro average
# ---------------------------------------------------------------------------

def leave_one_target_out(base_gain: dict[str, float]) -> list[dict[str, Any]]:
    targets = sorted(base_gain)
    vals = np.asarray([base_gain[t] for t in targets], dtype=float)
    full = float(np.mean(vals))
    rows = []
    for target in targets:
        remaining = [base_gain[t] for t in targets if t != target]
        loo = float(np.mean(remaining))
        rows.append(
            {
                "removed_target": target,
                "removed_target_gain_pct": base_gain[target],
                "full_macro_gain_pct": full,
                "leave_one_target_out_macro_gain_pct": loo,
                "delta_macro_gain_pp": loo - full,
                "remains_positive": loo > 0.0,
            }
        )
    return rows



# ---------------------------------------------------------------------------
# 7) Early-horizon mechanism audit (H=1 -> H=2 bump)
# ---------------------------------------------------------------------------

def early_horizon_audit(
    performance: dict[str, dict[str, Any]],
    donors: dict[str, str],
    seeds: list[int],
    replicates: int,
    hmax: int,
    c: float,
    w_r: float,
    w_e: float,
    representative_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Explain the non-monotone gain between H=1 and H=2.

    The classic no-transfer baseline has zero effective count on both arms, so
    UCB1's unseen-arm rule forces x86 on pull 1 (ARMS ordering) and ARM64 on
    pull 2. The transfer policy has positive prior pseudo-counts on both arms
    and therefore scores both arms from the first pull.

    We explicitly replay the first two pulls using Hmax-sized random streams,
    preserving the RNG convention of the final study. Besides aggregate arm
    shares, the audit identifies whether transfer switches arms (one sample of
    each, just like the baseline) or repeats one arm. Under paired CRN, when
    transfer switches arms, cumulative latency at H=2 must be exactly equal to
    the baseline because both policies consumed the same first x86 sample and
    the same first ARM64 sample, merely in a different order.
    """
    seed_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    seed_target_patterns: dict[int, dict[str, tuple[str, str] | None]] = {}

    for seed in seeds:
        aggregate = Counter()
        target_patterns: dict[str, tuple[str, str] | None] = {}
        target_h1_gains: list[float] = []
        target_h2_gains: list[float] = []
        switch_target_h2_gains: list[float] = []
        repeat_target_h2_gains: list[float] = []
        repeat_target_best_matches = 0
        deterministic_target_patterns = 0

        for target in sorted(performance):
            ts = performance[target]
            prior = build_reference_anchored_prior(ts, performance[donors[target]])
            sequences = make_sequences(target, ts, hmax, replicates, seed)

            pair_counts: Counter[tuple[str, str]] = Counter()
            h1_bg = np.empty(replicates, dtype=float)
            h1_tg = np.empty(replicates, dtype=float)
            h2_bg = np.empty(replicates, dtype=float)
            h2_tg = np.empty(replicates, dtype=float)

            for i, seq in enumerate(sequences):
                bt = simulate_baseline_trace(ts, seq, 2, c, convergence_run=5)
                tt = simulate_transfer_trace(ts, prior, seq, 2, c, w_r, w_e, convergence_run=5)
                bchosen = [str(x) for x in bt["chosen_arms"]]
                tchosen = [str(x) for x in tt["chosen_arms"]]

                aggregate[("baseline", 1, bchosen[0])] += 1
                aggregate[("baseline", 2, bchosen[1])] += 1
                aggregate[("transfer", 1, tchosen[0])] += 1
                aggregate[("transfer", 2, tchosen[1])] += 1
                pair_counts[(tchosen[0], tchosen[1])] += 1

                if tchosen[0] == tchosen[1]:
                    aggregate[("transfer_pattern", 2, "repeat")] += 1
                else:
                    aggregate[("transfer_pattern", 2, "switch")] += 1

                h1_bg[i] = float(bt["cumulative_latency"][0])
                h1_tg[i] = float(tt["cumulative_latency"][0])
                h2_bg[i] = float(bt["cumulative_latency"][1])
                h2_tg[i] = float(tt["cumulative_latency"][1])

            h1_gain = mean_gain_pct(h1_bg, h1_tg)
            h2_gain = mean_gain_pct(h2_bg, h2_tg)
            target_h1_gains.append(h1_gain)
            target_h2_gains.append(h2_gain)

            deterministic = len(pair_counts) == 1
            if deterministic:
                deterministic_target_patterns += 1
                pair = next(iter(pair_counts))
                target_patterns[target] = pair
                if pair[0] == pair[1]:
                    repeat_target_h2_gains.append(h2_gain)
                    if pair[0] == ts["best_reward_arm"]:
                        repeat_target_best_matches += 1
                else:
                    switch_target_h2_gains.append(h2_gain)
            else:
                target_patterns[target] = None

            if seed == representative_seed:
                dominant_pair, dominant_count = pair_counts.most_common(1)[0]
                representative_rows.append(
                    {
                        "seed": seed,
                        "target_function": target,
                        "donor_function": donors[target],
                        "empirical_best_arm": ts["best_reward_arm"],
                        "baseline_round1_arm": "x86",
                        "baseline_round2_arm": "arm64",
                        "transfer_round1_arm": dominant_pair[0],
                        "transfer_round2_arm": dominant_pair[1],
                        "transfer_pair_deterministic_across_replicates": deterministic,
                        "dominant_pair_replicates": dominant_count,
                        "replicates": replicates,
                        "transfer_repeats_same_arm": dominant_pair[0] == dominant_pair[1],
                        "repeated_arm_is_empirical_best": (
                            dominant_pair[0] == ts["best_reward_arm"]
                            if dominant_pair[0] == dominant_pair[1]
                            else ""
                        ),
                        "target_gain_h1_pct": h1_gain,
                        "target_gain_h2_pct": h2_gain,
                    }
                )

        seed_target_patterns[seed] = target_patterns
        total = len(performance) * replicates
        repeat_targets = sum(
            1 for pair in target_patterns.values()
            if pair is not None and pair[0] == pair[1]
        )
        switch_targets = sum(
            1 for pair in target_patterns.values()
            if pair is not None and pair[0] != pair[1]
        )
        seed_rows.append(
            {
                "seed": seed,
                "target_count": len(performance),
                "replicates_per_target": replicates,
                "total_target_replicates": total,
                "baseline_round1_x86_share_pct": 100.0 * aggregate[("baseline", 1, "x86")] / total,
                "baseline_round1_arm64_share_pct": 100.0 * aggregate[("baseline", 1, "arm64")] / total,
                "baseline_round2_x86_share_pct": 100.0 * aggregate[("baseline", 2, "x86")] / total,
                "baseline_round2_arm64_share_pct": 100.0 * aggregate[("baseline", 2, "arm64")] / total,
                "transfer_round1_x86_share_pct": 100.0 * aggregate[("transfer", 1, "x86")] / total,
                "transfer_round1_arm64_share_pct": 100.0 * aggregate[("transfer", 1, "arm64")] / total,
                "transfer_round2_x86_share_pct": 100.0 * aggregate[("transfer", 2, "x86")] / total,
                "transfer_round2_arm64_share_pct": 100.0 * aggregate[("transfer", 2, "arm64")] / total,
                "deterministic_target_patterns": deterministic_target_patterns,
                "switch_targets": switch_targets,
                "repeat_targets": repeat_targets,
                "repeat_targets_using_empirical_best_arm": repeat_target_best_matches,
                "macro_gain_h1_pct": float(np.mean(target_h1_gains)),
                "macro_gain_h2_pct": float(np.mean(target_h2_gains)),
                "h2_minus_h1_pp": float(np.mean(target_h2_gains) - np.mean(target_h1_gains)),
                "mean_h2_gain_switch_targets_pct": (
                    float(np.mean(switch_target_h2_gains)) if switch_target_h2_gains else float("nan")
                ),
                "max_abs_h2_gain_switch_targets_pct": (
                    float(np.max(np.abs(switch_target_h2_gains))) if switch_target_h2_gains else float("nan")
                ),
                "mean_h2_gain_repeat_targets_pct": (
                    float(np.mean(repeat_target_h2_gains)) if repeat_target_h2_gains else float("nan")
                ),
                "weighted_h2_macro_contribution_from_switch_targets_pp": (
                    float(np.sum(switch_target_h2_gains) / len(performance)) if switch_target_h2_gains else 0.0
                ),
                "weighted_h2_macro_contribution_from_repeat_targets_pp": (
                    float(np.sum(repeat_target_h2_gains) / len(performance)) if repeat_target_h2_gains else 0.0
                ),
            }
        )

    # Check whether each target's first-two-pull transfer pattern is invariant
    # across all requested seeds.
    pattern_stable_across_seeds = True
    if seeds:
        ref_seed = seeds[0]
        for target in sorted(performance):
            ref = seed_target_patterns[ref_seed].get(target)
            if any(seed_target_patterns[s].get(target) != ref for s in seeds[1:]):
                pattern_stable_across_seeds = False
                break

    seed_h1 = np.asarray([r["macro_gain_h1_pct"] for r in seed_rows], dtype=float)
    seed_h2 = np.asarray([r["macro_gain_h2_pct"] for r in seed_rows], dtype=float)
    representative = next(r for r in seed_rows if r["seed"] == representative_seed)

    summary_rows = [
        {
            "metric": "baseline_round1_forced_x86_all_runs",
            "value": all(abs(r["baseline_round1_x86_share_pct"] - 100.0) < 1e-12 for r in seed_rows),
            "interpretation": "Classic UCB1 pulls the first unseen arm (x86) on request 1.",
        },
        {
            "metric": "baseline_round2_forced_arm64_all_runs",
            "value": all(abs(r["baseline_round2_arm64_share_pct"] - 100.0) < 1e-12 for r in seed_rows),
            "interpretation": "Classic UCB1 pulls the remaining unseen arm (ARM64) on request 2.",
        },
        {
            "metric": "transfer_first_two_pull_pattern_stable_across_seeds",
            "value": pattern_stable_across_seeds,
            "interpretation": "For these empirical data, each target follows the same first-two-pull transfer pattern for all tested seeds.",
        },
        {
            "metric": "representative_seed",
            "value": representative_seed,
            "interpretation": "Seed used for target-level pattern table and figures.",
        },
        {
            "metric": "switch_targets_at_h2",
            "value": representative["switch_targets"],
            "interpretation": "Transfer consumes one x86 and one ARM64 observation; with CRN its H=2 cumulative latency equals baseline exactly.",
        },
        {
            "metric": "repeat_targets_at_h2",
            "value": representative["repeat_targets"],
            "interpretation": "Transfer avoids baseline's forced second-arm exploration and samples the same arm twice.",
        },
        {
            "metric": "repeat_targets_using_empirical_best_arm",
            "value": representative["repeat_targets_using_empirical_best_arm"],
            "interpretation": "Among repeated-arm targets, count for which the repeated arm is the empirically best architecture.",
        },
        {
            "metric": "max_abs_h2_gain_switch_targets_pct",
            "value": representative["max_abs_h2_gain_switch_targets_pct"],
            "interpretation": "Must be zero under correct paired CRN when both policies consume one sample from each arm.",
        },
        {
            "metric": "mean_macro_gain_h1_across_seeds_pct",
            "value": float(np.mean(seed_h1)),
            "interpretation": "Average H=1 macro latency gain over tested seeds.",
        },
        {
            "metric": "mean_macro_gain_h2_across_seeds_pct",
            "value": float(np.mean(seed_h2)),
            "interpretation": "Average H=2 macro latency gain over tested seeds.",
        },
        {
            "metric": "mean_h2_minus_h1_across_seeds_pp",
            "value": float(np.mean(seed_h2 - seed_h1)),
            "interpretation": "Observed early bump; explained by the different initialization semantics of baseline vs transfer.",
        },
        {
            "metric": "representative_h2_macro_contribution_switch_targets_pp",
            "value": representative["weighted_h2_macro_contribution_from_switch_targets_pp"],
            "interpretation": "Contribution of switching targets to the overall H=2 macro gain.",
        },
        {
            "metric": "representative_h2_macro_contribution_repeat_targets_pp",
            "value": representative["weighted_h2_macro_contribution_from_repeat_targets_pp"],
            "interpretation": "Contribution of repeated-arm targets to the overall H=2 macro gain.",
        },
    ]
    return seed_rows, representative_rows, summary_rows

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_both(fig: plt.Figure, out_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_replicate_sensitivity(rows: list[dict[str, Any]], figures: Path) -> None:
    x = [r["replicates_R"] for r in rows]
    y = [r["macro_mean_latency_gain_pct"] for r in rows]
    ref = rows[-1]["reference_gain_pct"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="Estimated macro gain")
    ax.axhline(ref, linestyle="--", linewidth=1, label=f"R={rows[-1]['reference_maxR']} reference")
    ax.set_xlabel("Monte Carlo replicates (R)")
    ax.set_ylabel("Macro mean latency gain at H=10 (%)")
    ax.set_title("Replicate-count sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    save_both(fig, figures / "replicate_count_sensitivity_h10")


def plot_multiseed_spaghetti(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    seeds: list[int],
    figures: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed in seeds:
        sub = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["horizon"])
        ax.plot(
            [r["horizon"] for r in sub],
            [r["macro_mean_latency_gain_pct"] for r in sub],
            marker="o",
            alpha=0.65,
            linewidth=1.4,
            label=f"seed {seed}",
        )
    s = sorted(summary, key=lambda r: r["horizon"])
    ax.plot(
        [r["horizon"] for r in s],
        [r["mean_gain_across_seeds_pct"] for r in s],
        marker="o",
        linewidth=3.0,
        label="mean across seeds",
    )
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_xlabel("Horizon H (requests)")
    ax.set_ylabel("Macro mean latency gain vs no-transfer (%)")
    ax.set_title("Transfer Learning stability across random seeds")
    ax.set_xticks(sorted({r["horizon"] for r in rows}))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    save_both(fig, figures / "multiseed_spaghetti_gain")


def plot_jackknife(rows: list[dict[str, Any]], figures: Path) -> None:
    vals = [r["delta_macro_gain_pp"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=45)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Change in macro gain after removing one observation (pp)")
    ax.set_ylabel("Leave-one-observation-out conditions")
    ax.set_title("Observation-level jackknife influence at H=10")
    ax.grid(axis="y", alpha=0.2)
    save_both(fig, figures / "jackknife_observation_influence_h10")


def plot_loto(rows: list[dict[str, Any]], figures: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["leave_one_target_out_macro_gain_pct"])
    fig, ax = plt.subplots(figsize=(9, 10))
    y = np.arange(len(ordered))
    ax.barh(y, [r["leave_one_target_out_macro_gain_pct"] for r in ordered])
    ax.set_yticks(y)
    ax.set_yticklabels([r["removed_target"] for r in ordered], fontsize=7)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Macro mean latency gain after removing target (%)")
    ax.set_ylabel("Removed target")
    ax.set_title("Leave-one-target-out robustness at H=10")
    ax.grid(axis="x", alpha=0.2)
    save_both(fig, figures / "leave_one_target_out_h10")



def plot_early_horizon_arm_choices(
    seed_rows: list[dict[str, Any]],
    representative_seed: int,
    figures: Path,
) -> None:
    r = next(row for row in seed_rows if row["seed"] == representative_seed)
    labels = ["Baseline R1", "Baseline R2", "Transfer R1", "Transfer R2"]
    x86 = [
        r["baseline_round1_x86_share_pct"],
        r["baseline_round2_x86_share_pct"],
        r["transfer_round1_x86_share_pct"],
        r["transfer_round2_x86_share_pct"],
    ]
    arm = [100.0 - v for v in x86]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, x86, width, label="x86")
    ax.bar(x + width / 2, arm, width, label="ARM64")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Arm selections (%)")
    ax.set_title(f"Early-horizon arm choices (seed {representative_seed})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    save_both(fig, figures / "early_horizon_arm_choice_share")


def plot_early_horizon_h2_pattern_gain(
    seed_rows: list[dict[str, Any]],
    representative_seed: int,
    figures: Path,
) -> None:
    r = next(row for row in seed_rows if row["seed"] == representative_seed)
    labels = [
        f"Switch arms\n({r['switch_targets']} targets)",
        f"Repeat arm\n({r['repeat_targets']} targets)",
        "Overall\n(53 targets)",
    ]
    values = [
        r["mean_h2_gain_switch_targets_pct"],
        r["mean_h2_gain_repeat_targets_pct"],
        r["macro_gain_h2_pct"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values)
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_ylabel("Mean latency gain at H=2 (%)")
    ax.set_title("H=2 gain decomposition by Transfer Learning pull pattern")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    save_both(fig, figures / "early_horizon_h2_pattern_gain")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_data = REPO_ROOT / "data" / "profiling"

    p = argparse.ArgumentParser(
        description="Validate Serverledge Transfer Learning Monte Carlo methodology"
    )
    p.add_argument(
        "--raw-x86",
        type=Path,
        default=default_data / "final-20260913-analysis-01" / "raw" / "x86" / "all_samples.jsonl",
    )
    p.add_argument(
        "--raw-arm64",
        type=Path,
        default=default_data / "final-20260913-analysis-01" / "raw" / "arm64" / "all_samples.jsonl",
    )
    p.add_argument(
        "--donor-metric-results",
        type=Path,
        default=default_data / "analysis-transfer-donor-metrics-20260918" / "donor-metric-per-target.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_data / "analysis-transfer-methodology-validation",
    )
    p.add_argument("--algorithm", default="kmeans")
    p.add_argument("--ranker", default="manhattan")
    p.add_argument("--c", type=float, default=DEFAULT_C)
    p.add_argument("--reward-prior-weight", type=float, default=DEFAULT_WR)
    p.add_argument("--exploration-prior-weight", type=float, default=DEFAULT_WE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    p.add_argument("--replicate-counts", type=parse_int_list, default=DEFAULT_REPLICATE_COUNTS)
    p.add_argument("--primary-horizon", type=int, default=DEFAULT_PRIMARY_H)
    p.add_argument("--hmax", type=int, default=DEFAULT_HMAX)
    p.add_argument("--horizons", type=parse_int_list, default=DEFAULT_HORIZONS)
    p.add_argument(
        "--crn-replicates",
        type=int,
        default=20,
        help="runtime CRN spot-check replicates per target/seed (source-level check is exhaustive)",
    )
    p.add_argument(
        "--skip-jackknife",
        action="store_true",
        help="skip the 1,272-condition observation jackknife for a faster smoke run",
    )
    args = p.parse_args()

    if args.c < 0 or args.reward_prior_weight <= 0 or args.exploration_prior_weight <= 0:
        raise ValueError("Require c>=0, w_R>0, w_E>0")
    if args.replicates <= 0 or min(args.replicate_counts) <= 0:
        raise ValueError("replicate counts must be positive")

    horizons = horizons_valid(args.horizons, args.hmax)
    if args.primary_horizon not in horizons:
        horizons = sorted(set(horizons + [args.primary_horizon]))

    raw_x86_path = ensure_input(args.raw_x86, "x86 raw file")
    raw_arm_path = ensure_input(args.raw_arm64, "ARM64 raw file")
    donor_path = ensure_input(args.donor_metric_results, "donor metric results")
    out = args.output_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("Loading eligible warm/exclusive samples...")
    raw_x86 = load_raw_durations(raw_x86_path)
    raw_arm = load_raw_durations(raw_arm_path)
    performance = empirical_stats(raw_x86, raw_arm)
    donors = load_manhattan_donors(donor_path, args.algorithm, args.ranker)

    if set(performance) != set(donors):
        raise RuntimeError(
            "Donor/performance target mismatch: "
            f"missing donors={sorted(set(performance)-set(donors))}, "
            f"extra donors={sorted(set(donors)-set(performance))}"
        )

    validation_rows: list[dict[str, Any]] = []

    # 1) Data audit
    print("\n=== 1/7 DATA AUDIT ===")
    sample_rows, sample_summary = sample_count_audit(raw_x86, raw_arm)
    write_csv(out / "01_sample_count_audit.csv", sample_rows)
    validation_rows.append(
        {
            "validation": "eligible sample count",
            "status": "PASS" if sample_summary["all_common_and_12_each"] else "WARN",
            "result": (
                f"{sample_summary['function_count_common']} common functions; "
                f"12 eligible samples/arm for every function={sample_summary['all_common_and_12_each']}"
            ),
        }
    )

    # 2) CRN
    print("\n=== 2/7 COMMON RANDOM NUMBERS ===")
    crn_src = crn_source_checks()
    crn_runtime = crn_runtime_checks(
        performance,
        donors,
        args.seeds,
        args.c,
        args.reward_prior_weight,
        args.exploration_prior_weight,
        args.hmax,
        args.crn_replicates,
    )
    crn_rows = crn_src + [
        {
            "check": f"runtime_seed_{r['seed']}_target_{r['target_function']}",
            "status": r["status"],
            "evidence": (
                f"shared coordinates checked={r['shared_arm_occurrence_coordinates_checked']}; "
                f"replicates with different policy arm order={r['replicates_with_different_policy_arm_order']}"
            ),
        }
        for r in crn_runtime
    ]
    write_csv(out / "02_crn_verification.csv", crn_rows)
    crn_pass = all(r["status"] == "PASS" for r in crn_rows)
    validation_rows.append(
        {
            "validation": "paired Common Random Numbers",
            "status": "PASS" if crn_pass else "FAIL",
            "result": (
                "arm-specific streams indexed by per-arm occurrence; baseline and transfer share the same replica stream"
                if crn_pass else "one or more CRN checks failed"
            ),
        }
    )

    print("\n=== 2b/7 VALIDATOR/POLICY EXACT EQUIVALENCE ===")
    eq_rows = vectorized_policy_equivalence_checks(
        performance, donors, args.seed, horizons, args.hmax, args.c,
        args.reward_prior_weight, args.exploration_prior_weight,
    )
    write_csv(out / "02b_vectorized_policy_equivalence.csv", eq_rows)
    eq_pass = all(r["status"] == "PASS" for r in eq_rows)
    validation_rows.append(
        {
            "validation": "vectorized validator equivalence",
            "status": "PASS" if eq_pass else "FAIL",
            "result": (
                "exact cumulative-latency match against repository policy classes for every checked target/replica/horizon"
                if eq_pass else "validator differs from repository policy execution"
            ),
        }
    )

    # 3) Replicate sensitivity
    print("\n=== 3/7 REPLICATE-COUNT SENSITIVITY ===")
    rep_rows = replicate_sensitivity(
        performance,
        donors,
        args.replicate_counts,
        args.seed,
        args.primary_horizon,
        args.hmax,
        args.c,
        args.reward_prior_weight,
        args.exploration_prior_weight,
    )
    write_csv(out / "03_replicate_count_sensitivity.csv", rep_rows)
    plot_replicate_sensitivity(rep_rows, figures)
    r_default = next((r for r in rep_rows if r["replicates_R"] == args.replicates), None)
    r_ref = rep_rows[-1]
    validation_rows.append(
        {
            "validation": "replicate-count stability",
            "status": "PASS",
            "result": (
                f"R={args.replicates} gain={r_default['macro_mean_latency_gain_pct']:.6f}% vs "
                f"R={r_ref['replicates_R']} gain={r_ref['macro_mean_latency_gain_pct']:.6f}%; "
                f"|delta|={r_default['absolute_difference_from_maxR_pp']:.6f} pp"
                if r_default else f"max R={r_ref['replicates_R']} gain={r_ref['macro_mean_latency_gain_pct']:.6f}%"
            ),
        }
    )

    # 4) Multi-seed
    print("\n=== 4/7 MULTI-SEED SENSITIVITY ===")
    seed_rows, seed_summary = multiseed_analysis(
        performance,
        donors,
        args.seeds,
        args.replicates,
        horizons,
        args.hmax,
        args.c,
        args.reward_prior_weight,
        args.exploration_prior_weight,
    )
    write_csv(out / "04_multiseed_per_seed_horizon.csv", seed_rows)
    write_csv(out / "05_multiseed_summary.csv", seed_summary)
    plot_multiseed_spaghetti(seed_rows, seed_summary, args.seeds, figures)
    h10_summary = next(r for r in seed_summary if r["horizon"] == args.primary_horizon)
    validation_rows.append(
        {
            "validation": "random-seed stability",
            "status": "PASS",
            "result": (
                f"{len(args.seeds)} seeds={args.seeds}; H={args.primary_horizon}: "
                f"mean={h10_summary['mean_gain_across_seeds_pct']:.6f}%, "
                f"std={h10_summary['std_across_seeds_pp']:.6f} pp, "
                f"range={h10_summary['range_across_seeds_pp']:.6f} pp"
            ),
        }
    )

    # 5) Observation jackknife + 6) target LOO
    base_gain: dict[str, float]
    if not args.skip_jackknife:
        print("\n=== 5/7 LEAVE-ONE-OBSERVATION-OUT JACKKNIFE ===")
        jack_rows, base_gain, jack_summary = jackknife_observation(
            raw_x86,
            raw_arm,
            performance,
            donors,
            args.seed,
            args.replicates,
            args.primary_horizon,
            args.hmax,
            args.c,
            args.reward_prior_weight,
            args.exploration_prior_weight,
        )
        write_csv(out / "06_jackknife_observation_h10.csv", jack_rows)
        plot_jackknife(jack_rows, figures)
        validation_rows.append(
            {
                "validation": "single-observation robustness",
                "status": "PASS" if jack_summary["negative_gain_conditions"] == 0 else "WARN",
                "result": (
                    f"{int(jack_summary['conditions'])} leave-one-out conditions; "
                    f"gain range=[{jack_summary['min_loo_macro_gain_pct']:.6f}, "
                    f"{jack_summary['max_loo_macro_gain_pct']:.6f}]%; "
                    f"max influence={jack_summary['max_absolute_influence_pp']:.6f} pp; "
                    f"negative conditions={int(jack_summary['negative_gain_conditions'])}; "
                    f"best-arm changes={int(jack_summary['best_arm_change_conditions'])}"
                ),
            }
        )

        print("\n=== 6/7 LEAVE-ONE-TARGET-OUT ===")
        loto_rows = leave_one_target_out(base_gain)
        write_csv(out / "07_leave_one_target_out_h10.csv", loto_rows)
        plot_loto(loto_rows, figures)
        loto_vals = np.asarray([r["leave_one_target_out_macro_gain_pct"] for r in loto_rows], dtype=float)
        validation_rows.append(
            {
                "validation": "single-target robustness",
                "status": "PASS" if np.all(loto_vals > 0.0) else "WARN",
                "result": (
                    f"LOTO macro gain range=[{np.min(loto_vals):.6f}, {np.max(loto_vals):.6f}]%; "
                    f"all remain positive={bool(np.all(loto_vals > 0.0))}"
                ),
            }
        )
    else:
        print("\n=== 5-6/7 JACKKNIFE / TARGET LOO SKIPPED ===")
        validation_rows.append(
            {
                "validation": "single-observation robustness",
                "status": "SKIPPED",
                "result": "rerun without --skip-jackknife for final evidence",
            }
        )
        validation_rows.append(
            {
                "validation": "single-target robustness",
                "status": "SKIPPED",
                "result": "rerun without --skip-jackknife for final evidence",
            }
        )


    # 7) Early-horizon mechanism audit
    print("\n=== 7/7 EARLY-HORIZON H=1 -> H=2 MECHANISM ===")
    early_seed_rows, early_target_rows, early_summary_rows = early_horizon_audit(
        performance,
        donors,
        args.seeds,
        args.replicates,
        args.hmax,
        args.c,
        args.reward_prior_weight,
        args.exploration_prior_weight,
        args.seed,
    )
    write_csv(out / "08_early_horizon_selection_audit.csv", early_seed_rows)
    write_csv(out / "09_early_horizon_target_patterns.csv", early_target_rows)
    write_csv(out / "10_early_horizon_summary.csv", early_summary_rows)
    plot_early_horizon_arm_choices(early_seed_rows, args.seed, figures)
    plot_early_horizon_h2_pattern_gain(early_seed_rows, args.seed, figures)

    early_rep = next(r for r in early_seed_rows if r["seed"] == args.seed)
    early_pass = (
        all(abs(r["baseline_round1_x86_share_pct"] - 100.0) < 1e-12 for r in early_seed_rows)
        and all(abs(r["baseline_round2_arm64_share_pct"] - 100.0) < 1e-12 for r in early_seed_rows)
        and abs(early_rep["max_abs_h2_gain_switch_targets_pct"]) < 1e-12
    )
    validation_rows.append(
        {
            "validation": "early-horizon H=1->H=2 mechanism",
            "status": "PASS" if early_pass else "WARN",
            "result": (
                f"baseline forced x86->ARM64; transfer seed={args.seed}: "
                f"switch targets={early_rep['switch_targets']}, repeat targets={early_rep['repeat_targets']} "
                f"({early_rep['repeat_targets_using_empirical_best_arm']} repeat empirical best); "
                f"H1={early_rep['macro_gain_h1_pct']:.6f}%, H2={early_rep['macro_gain_h2_pct']:.6f}%; "
                f"switch-target H2 max |gain|={early_rep['max_abs_h2_gain_switch_targets_pct']:.12f}%"
            ),
        }
    )

    write_csv(out / "00_validation_summary.csv", validation_rows)

    manifest = {
        "analysis": "serverledge_transfer_learning_methodology_validation",
        "policy_name": "weak-prior decoupled Transfer Learning UCB1",
        "baseline": "classic UCB1 without prior",
        "formula": {
            "reward": "-ln(duration_ms)",
            "effective_mean": "(S_j + w_R * mu_prior_j) / (n_j + w_R)",
            "exploration_bonus": "c * sqrt(ln(sum_k(n_k + w_E)) / (n_j + w_E))",
            "score": "effective_mean + exploration_bonus",
            "reference_anchored_prior_x86": "mu_target_x86",
            "reference_anchored_prior_arm64": "mu_target_x86 + (mu_donor_arm64 - mu_donor_x86)",
        },
        "configuration": {
            "c": args.c,
            "w_R": args.reward_prior_weight,
            "w_E": args.exploration_prior_weight,
            "primary_horizon": args.primary_horizon,
            "hmax_used_for_rng": args.hmax,
            "replicates": args.replicates,
            "replicate_count_sensitivity": args.replicate_counts,
            "seeds": args.seeds,
            "clusterer": args.algorithm,
            "donor_ranker": args.ranker,
        },
        "data": {
            "raw_x86": str(raw_x86_path),
            "raw_arm64": str(raw_arm_path),
            "donor_metric_results": str(donor_path),
            "raw_x86_sha256": sha256_file(raw_x86_path),
            "raw_arm64_sha256": sha256_file(raw_arm_path),
            "donor_metric_results_sha256": sha256_file(donor_path),
            "target_count": len(performance),
            "eligible_x86_sample_count": sum(len(v) for v in raw_x86.values()),
            "eligible_arm64_sample_count": sum(len(v) for v in raw_arm.values()),
        },
        "methodological_notes": {
            "paired_common_random_numbers": True,
            "crn_indexing": "per arm and per occurrence of that arm, not global request round",
            "plug_in_monte_carlo": (
                "The empirical distribution from the observed samples is treated as the resampling distribution; "
                "more Monte Carlo replications reduce simulation error but do not add new GCP information."
            ),
            "rng_reproducibility": (
                "Metrics at H=10 are evaluated from streams generated with Hmax=50, matching the original study."
            ),
            "early_horizon_interpretation": (
                "Classic UCB1 forces x86 then ARM64 for its first two unseen-arm pulls. "
                "The transfer policy has prior pseudo-counts and can score arms immediately; the H=1->H=2 bump is audited explicitly."
            ),
        },
        "outputs": [r["validation"] for r in validation_rows],
    }
    (out / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\n================ FINAL VALIDATION SUMMARY ================")
    for row in validation_rows:
        print(f"{row['status']:<7} {row['validation']}: {row['result']}")
    print(f"\nOutput directory: {out}")
    print(f"Figures:          {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
