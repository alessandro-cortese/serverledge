#!/usr/bin/env python3
"""
Serverledge — Monte Carlo IID / RNG / CRN validation.

This is a validation/audit utility, NOT a tuning or performance experiment.

What it validates
-----------------
A) Construction of the simulator
   - bootstrap sampling is with replacement
   - anchor and online streams use distinct deterministic seeds
   - x86 and arm64 online streams use distinct deterministic seeds
   - all stream seeds used by the final multi-seed experiment are collision-free
   - same seed => exact reproducibility
   - different seed => different generated streams
   - final matched-parameter runner reuses the same online arrays across strategies
     (CRN audit by source inspection + source SHA256)

B) Empirical sanity checks on generated bootstrap streams
   - lag autocorrelations close to zero on long synthetic/empirical-resampling streams
   - correlations between different MC-seed streams close to zero

C) Plausibility checks on the REAL warm input sequences
   - lag-1 / max absolute ACF
   - Ljung-Box test where sample size is sufficient
   - runs test around the median
   - Spearman trend test versus acquisition/loader order
   - Benjamini-Hochberg correction across target-arm series

Important
---------
No statistical test can prove that real measurements are IID.
The simulated draws are IID *by construction* conditional on the empirical
distribution, because NumPy choice is called with replace=True and no probability
vector, from a fixed target/arm empirical sample array. The real-data tests are
diagnostics for obvious serial dependence / drift only.

Outputs
-------
iid_generator_checks.json
iid_seed_stream_correlations.csv
iid_real_series_tests.csv
iid_validation_summary.json
IID_VALIDATION_REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr

from analysis.profiling.transfer_prior_formula_comparison import (
    ARMS,
    build_performance,
    load_eligible_durations,
    make_crn_samples,
    stable_seed,
)

FINAL_VARIANT = "manhattan_plain"
SYNTHETIC_TARGETS = {
    "amd_faster", "arm_faster", "amd-faster", "arm-faster"
}


def parse_ints(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    if len(vals) != len(set(vals)):
        raise ValueError("duplicate seeds are not allowed")
    return vals


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_synthetic(name: str) -> bool:
    return str(name).strip().lower() in SYNTHETIC_TARGETS


def load_plain_targets(path: Path, donor_seed: int) -> list[str]:
    df = pd.read_csv(path)
    required = {"variant", "seed", "target_function"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"donor detail missing columns: {sorted(missing)}"
        )

    x = df[
        (df["variant"].astype(str) == FINAL_VARIANT)
        & (df["seed"].astype(int) == int(donor_seed))
        ].copy()

    if len(x) != 59:
        raise SystemExit(
            f"expected 59 rows for {FINAL_VARIANT}/seed={donor_seed}; "
            f"got {len(x)}"
        )

    targets = sorted(x["target_function"].astype(str).tolist())
    if len(set(targets)) != 59:
        raise SystemExit("duplicate target_function in donor detail")
    return targets


def acf_values(x: np.ndarray, max_lag: int) -> dict[int, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return {}
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return {}
    out: dict[int, float] = {}
    for lag in range(1, min(max_lag, n - 1) + 1):
        out[lag] = float(
            np.dot(centered[:-lag], centered[lag:]) / denom
        )
    return out


def ljung_box(x: np.ndarray, lags: int) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = min(int(lags), max(1, n // 5))
    if n < 20 or h < 1 or np.std(x) == 0:
        return math.nan, math.nan, h
    acf = acf_values(x, h)
    if len(acf) < h:
        return math.nan, math.nan, h
    q = n * (n + 2.0) * sum(
        (acf[k] ** 2) / (n - k) for k in range(1, h + 1)
    )
    p = float(chi2.sf(q, df=h))
    return float(q), p, h


def runs_test_median(x: np.ndarray) -> tuple[float, float, int]:
    """
    Wald-Wolfowitz-style runs test after removing values equal to the median.
    Returns z, two-sided p, effective n.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return math.nan, math.nan, len(x)

    med = float(np.median(x))
    signs = x[x != med] > med
    n = len(signs)
    if n < 10:
        return math.nan, math.nan, n

    n1 = int(np.sum(signs))
    n0 = int(n - n1)
    if n1 == 0 or n0 == 0:
        return math.nan, math.nan, n

    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 1.0 + (2.0 * n1 * n0) / n
    var = (
            2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
            / (n**2 * (n - 1.0))
    )
    if var <= 0:
        return math.nan, math.nan, n
    z = (runs - expected) / math.sqrt(var)
    p = 2.0 * float(norm.sf(abs(z)))
    return float(z), p, n


def bh_adjust(pvals: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR adjusted p-values. NaNs stay NaN.
    """
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna().astype(float)
    m = len(valid)
    if m == 0:
        return out

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out.loc[order] = q
    return out


def source_audit(runner_module_name: str) -> dict[str, Any]:
    """
    Audit exact source used for generator and final matched runner.
    """
    gen_src = inspect.getsource(make_crn_samples)

    # Import runner dynamically.
    import importlib
    runner = importlib.import_module(runner_module_name)
    runner_src = inspect.getsource(runner.run_one_seed)

    generator_checks = {
        "generator_source_sha256": sha256_text(gen_src),
        "runner_source_sha256": sha256_text(runner_src),
        "generator_uses_replace_true":
            "replace=True" in gen_src,
        "generator_uses_choice":
            ".choice(" in gen_src,
        "generator_has_anchor_stream_label":
            '"anchor-x86"' in gen_src or "'anchor-x86'" in gen_src,
        "generator_has_online_stream_label":
            '"online"' in gen_src or "'online'" in gen_src,
        "generator_separates_arm_in_seed":
            "arm" in gen_src and "stable_seed" in gen_src,
        "runner_calls_make_crn_once_in_function":
            runner_src.count("make_crn_samples(") == 1,
        "runner_reuses_online_x86":
            runner_src.count('online["x86"][rep]') >= 2,
        "runner_reuses_online_arm64":
            runner_src.count('online["arm64"][rep]') >= 2,
        "runner_source_contains_crn_comment":
            "CRN" in runner_src or "same CRN" in runner_src,
    }
    generator_checks["source_audit_pass"] = bool(
        generator_checks["generator_uses_replace_true"]
        and generator_checks["generator_uses_choice"]
        and generator_checks["generator_has_anchor_stream_label"]
        and generator_checks["generator_has_online_stream_label"]
        and generator_checks["generator_separates_arm_in_seed"]
        and generator_checks["runner_calls_make_crn_once_in_function"]
        and generator_checks["runner_reuses_online_x86"]
        and generator_checks["runner_reuses_online_arm64"]
    )
    return generator_checks


def stream_seed_audit(
        targets: list[str],
        mc_seeds: list[int],
) -> dict[str, Any]:
    records = []
    for base in mc_seeds:
        for target in targets:
            records.append((
                base, target, "anchor-x86",
                stable_seed(base, target, "anchor-x86")
            ))
            for arm in ARMS:
                records.append((
                    base, target, f"online-{arm}",
                    stable_seed(base, target, "online", arm)
                ))

    values = [r[3] for r in records]
    collision_count = len(values) - len(set(values))

    # Exact reproducibility and different-seed nonidentity on a synthetic target.
    synthetic_stats = {
        "durations": {
            "x86": np.arange(1.0, 9.0, dtype=float),
            "arm64": np.arange(101.0, 109.0, dtype=float),
        }
    }
    a1, o1 = make_crn_samples(
        target="__iid_validation__",
        target_stats=synthetic_stats,
        replicates=2000,
        max_horizon=50,
        anchor_n=10,
        base_seed=mc_seeds[0],
    )
    a2, o2 = make_crn_samples(
        target="__iid_validation__",
        target_stats=synthetic_stats,
        replicates=2000,
        max_horizon=50,
        anchor_n=10,
        base_seed=mc_seeds[0],
    )
    other_seed = mc_seeds[1] if len(mc_seeds) > 1 else mc_seeds[0] + 1
    a3, o3 = make_crn_samples(
        target="__iid_validation__",
        target_stats=synthetic_stats,
        replicates=2000,
        max_horizon=50,
        anchor_n=10,
        base_seed=other_seed,
    )

    same_seed_exact = bool(
        np.array_equal(a1, a2)
        and all(np.array_equal(o1[k], o2[k]) for k in ARMS)
    )
    different_seed_different = bool(
        not np.array_equal(a1, a3)
        and all(not np.array_equal(o1[k], o3[k]) for k in ARMS)
    )

    # Serial ACF sanity on long generated streams.
    serial = {}
    for arm in ARMS:
        flat = o1[arm].reshape(-1)
        acfs = acf_values(flat, 20)
        serial[arm] = {
            "n": int(len(flat)),
            "lag1_acf": float(acfs.get(1, math.nan)),
            "max_abs_acf_lag1_20": float(
                max(abs(v) for v in acfs.values())
            ) if acfs else math.nan,
        }

    return {
        "stream_seed_records": len(records),
        "unique_stream_seeds": len(set(values)),
        "seed_collision_count": int(collision_count),
        "same_seed_exact_reproducibility": same_seed_exact,
        "different_seed_produces_different_stream": different_seed_different,
        "generated_serial_acf_sanity": serial,
        "bit_generator":
            type(np.random.default_rng().bit_generator).__name__,
    }


def seed_stream_correlations(
        *,
        targets: list[str],
        performance: dict[str, Any],
        mc_seeds: list[int],
        replicates: int,
        max_horizon: int,
) -> pd.DataFrame:
    """
    Empirical sanity: pairwise Pearson correlation between independently seeded
    online streams, within same target/arm. Not a proof; expected ~0.
    """
    rows = []
    primary_targets = [t for t in targets if not is_synthetic(t)]

    for target in primary_targets:
        stats = performance[target]
        streams: dict[int, dict[str, np.ndarray]] = {}
        for seed in mc_seeds:
            _, online = make_crn_samples(
                target=target,
                target_stats=stats,
                replicates=replicates,
                max_horizon=max_horizon,
                anchor_n=10,
                base_seed=seed,
            )
            streams[seed] = online

        for arm in ARMS:
            if np.std(stats["durations"][arm]) == 0:
                continue
            for s1, s2 in combinations(mc_seeds, 2):
                x = streams[s1][arm].reshape(-1).astype(float)
                y = streams[s2][arm].reshape(-1).astype(float)
                if np.std(x) == 0 or np.std(y) == 0:
                    corr = math.nan
                else:
                    corr = float(np.corrcoef(x, y)[0, 1])
                rows.append({
                    "target_function": target,
                    "arm": arm,
                    "seed_a": int(s1),
                    "seed_b": int(s2),
                    "n_draws": int(len(x)),
                    "pearson_corr": corr,
                    "abs_pearson_corr": abs(corr)
                    if np.isfinite(corr) else math.nan,
                })

    return pd.DataFrame(rows)


def real_series_tests(
        targets: list[str],
        performance: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for target in targets:
        if is_synthetic(target):
            continue
        for arm in ARMS:
            x = np.asarray(
                performance[target]["durations"][arm],
                dtype=float,
            )
            n = len(x)
            acfs = acf_values(x, min(10, max(1, n // 4)))
            lag1 = acfs.get(1, math.nan)
            max_abs = (
                max(abs(v) for v in acfs.values())
                if acfs else math.nan
            )

            q, p_lb, lb_lags = ljung_box(x, lags=10)
            z_runs, p_runs, runs_n = runs_test_median(x)

            if n >= 8 and np.std(x) > 0:
                rho, p_trend = spearmanr(
                    np.arange(n, dtype=float),
                    x,
                )
                rho = float(rho)
                p_trend = float(p_trend)
            else:
                rho = math.nan
                p_trend = math.nan

            rows.append({
                "target_function": target,
                "arm": arm,
                "n": int(n),
                "mean_ms": float(np.mean(x)) if n else math.nan,
                "std_ms": float(np.std(x, ddof=1))
                if n > 1 else math.nan,
                "lag1_acf": float(lag1),
                "max_abs_acf_tested_lags": float(max_abs),
                "ljung_box_lags": int(lb_lags),
                "ljung_box_q": float(q),
                "ljung_box_p": float(p_lb),
                "runs_effective_n": int(runs_n),
                "runs_z": float(z_runs),
                "runs_p": float(p_runs),
                "spearman_index_rho": rho,
                "spearman_index_p": p_trend,
            })

    df = pd.DataFrame(rows)

    for col in (
            "ljung_box_p",
            "runs_p",
            "spearman_index_p",
    ):
        df[col.replace("_p", "_q_bh")] = bh_adjust(df[col])

    df["any_fdr05_flag"] = (
            (df["ljung_box_q_bh"] < 0.05)
            | (df["runs_q_bh"] < 0.05)
            | (df["spearman_index_q_bh"] < 0.05)
    )
    return df


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    donor_detail = args.donor_detail.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    mc_seeds = parse_ints(args.mc_seeds)

    raw_x = root / "raw/x86/all_samples.jsonl"
    raw_a = root / "raw/arm64/all_samples.jsonl"

    for p in (raw_x, raw_a, donor_detail):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    targets = load_plain_targets(donor_detail, args.donor_seed)
    performance = build_performance(
        load_eligible_durations(raw_x),
        load_eligible_durations(raw_a),
    )

    missing = [t for t in targets if t not in performance]
    if missing:
        raise SystemExit(
            "missing target performance data: " + ", ".join(missing)
        )

    source_checks = source_audit(args.runner_module)
    seed_checks = stream_seed_audit(targets, mc_seeds)

    corr = seed_stream_correlations(
        targets=targets,
        performance=performance,
        mc_seeds=mc_seeds,
        replicates=args.correlation_replicates,
        max_horizon=args.correlation_horizon,
    )
    corr_path = out / "iid_seed_stream_correlations.csv"
    corr.to_csv(corr_path, index=False)

    real = real_series_tests(targets, performance)
    real_path = out / "iid_real_series_tests.csv"
    real.to_csv(real_path, index=False)

    generator = {
        **source_checks,
        **seed_checks,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "mc_seeds": mc_seeds,
        "target_count": len(targets),
        "primary_non_synthetic_target_count":
            sum(not is_synthetic(t) for t in targets),
        "raw_x86_sha256": sha256_file(raw_x),
        "raw_arm64_sha256": sha256_file(raw_a),
        "donor_detail_sha256": sha256_file(donor_detail),
    }

    generator_path = out / "iid_generator_checks.json"
    generator_path.write_text(
        json.dumps(generator, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    # Conservative implementation PASS: deterministic/source properties only.
    structural_pass = bool(
        source_checks["source_audit_pass"]
        and seed_checks["seed_collision_count"] == 0
        and seed_checks["same_seed_exact_reproducibility"]
        and seed_checks["different_seed_produces_different_stream"]
    )

    finite_corr = corr["abs_pearson_corr"].dropna()
    corr_summary = {
        "pairwise_stream_comparisons": int(len(finite_corr)),
        "mean_abs_correlation": float(finite_corr.mean())
        if len(finite_corr) else math.nan,
        "p95_abs_correlation": float(finite_corr.quantile(0.95))
        if len(finite_corr) else math.nan,
        "max_abs_correlation": float(finite_corr.max())
        if len(finite_corr) else math.nan,
    }

    real_summary = {
        "series_count": int(len(real)),
        "min_n": int(real["n"].min()) if len(real) else 0,
        "median_n": float(real["n"].median()) if len(real) else math.nan,
        "max_n": int(real["n"].max()) if len(real) else 0,
        "ljung_box_testable_series":
            int(real["ljung_box_p"].notna().sum()),
        "ljung_box_fdr05_flags":
            int((real["ljung_box_q_bh"] < 0.05).sum()),
        "runs_testable_series":
            int(real["runs_p"].notna().sum()),
        "runs_fdr05_flags":
            int((real["runs_q_bh"] < 0.05).sum()),
        "trend_testable_series":
            int(real["spearman_index_p"].notna().sum()),
        "trend_fdr05_flags":
            int((real["spearman_index_q_bh"] < 0.05).sum()),
        "series_with_any_fdr05_flag":
            int(real["any_fdr05_flag"].sum()),
    }

    summary = {
        "structural_iid_simulator_validation_pass": structural_pass,
        "meaning": (
            "The Monte Carlo bootstrap draws are IID by construction "
            "conditional on each fixed empirical target/arm distribution. "
            "CRN intentionally couples competing strategies within a replica."
        ),
        "generated_stream_sanity": corr_summary,
        "real_input_diagnostics": real_summary,
        "important_limitation": (
            "Failure to reject serial dependence tests on the real input "
            "does not prove that the real cloud process is IID; small sample "
            "sizes also imply limited statistical power."
        ),
    }

    summary_path = out / "iid_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )

    flagged = real[real["any_fdr05_flag"]].copy()

    report = [
        "# Monte Carlo IID / RNG / CRN validation",
        "",
        "## Scope",
        "",
        "This audit validates the random-sampling construction used by the local Monte Carlo replay. It does **not** claim that the physical cloud runtime process is provably IID.",
        "",
        "## Structural result",
        "",
        f"- structural simulator validation: **{'PASS' if structural_pass else 'FAIL'}**",
        f"- bootstrap uses `replace=True`: `{source_checks['generator_uses_replace_true']}`",
        f"- generator source SHA256: `{source_checks['generator_source_sha256']}`",
        f"- runner source SHA256: `{source_checks['runner_source_sha256']}`",
        f"- RNG bit generator: `{seed_checks['bit_generator']}`",
        f"- stream seed collisions: `{seed_checks['seed_collision_count']}` across `{seed_checks['stream_seed_records']}` derived stream seeds",
        f"- same-seed exact reproducibility: `{seed_checks['same_seed_exact_reproducibility']}`",
        f"- different-seed different stream: `{seed_checks['different_seed_produces_different_stream']}`",
        "",
        "The simulator therefore samples independently with replacement from a fixed empirical distribution for each target/architecture. Anchor and online streams, and the two online arms, are keyed by distinct deterministic seed labels.",
        "",
        "CRN is deliberate: the matched-parameter runner generates the online arrays once per target/MC seed and reuses the same per-replicate x86/arm64 sequences across No-TL, Difference and Ratio. Hence strategies are paired, not independent.",
        "",
        "## Generated-stream sanity",
        "",
        f"- pairwise independent-seed stream comparisons: `{corr_summary['pairwise_stream_comparisons']}`",
        f"- mean |correlation|: `{corr_summary['mean_abs_correlation']:.6f}`",
        f"- 95th percentile |correlation|: `{corr_summary['p95_abs_correlation']:.6f}`",
        f"- maximum |correlation|: `{corr_summary['max_abs_correlation']:.6f}`",
        "",
        "These correlations are diagnostics only; IID follows from the sampling construction, not from a failure to observe correlation.",
        "",
        "## Real warm-input diagnostics",
        "",
        f"- target/arm series: `{real_summary['series_count']}`",
        f"- n range: `{real_summary['min_n']}..{real_summary['max_n']}`; median `{real_summary['median_n']:.1f}`",
        f"- Ljung-Box testable series: `{real_summary['ljung_box_testable_series']}`; FDR<.05 flags: `{real_summary['ljung_box_fdr05_flags']}`",
        f"- runs-test testable series: `{real_summary['runs_testable_series']}`; FDR<.05 flags: `{real_summary['runs_fdr05_flags']}`",
        f"- order-trend testable series: `{real_summary['trend_testable_series']}`; FDR<.05 flags: `{real_summary['trend_fdr05_flags']}`",
        f"- series with any FDR<.05 diagnostic flag: `{real_summary['series_with_any_fdr05_flag']}`",
        "",
        "These tests assess whether obvious serial dependence, non-random runs or monotonic drift are visible in the observed warm measurements. They cannot prove that the real system is IID, and small `n` means low power.",
        "",
    ]

    if len(flagged):
        report += [
            "## Flagged real series",
            "",
            flagged[
                [
                    "target_function",
                    "arm",
                    "n",
                    "lag1_acf",
                    "ljung_box_q_bh",
                    "runs_q_bh",
                    "spearman_index_q_bh",
                ]
            ].to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    else:
        report += [
            "## Flagged real series",
            "",
            "No target/arm series was flagged at FDR 5% by the implemented diagnostics.",
            "",
        ]

    report += [
        "## Thesis-safe statement",
        "",
        "> Runtime observations in the Monte Carlo replay are generated by independent sampling with replacement from the empirical warm-runtime distribution of each function–architecture pair. The IID property is therefore an assumption of the simulator and is enforced by construction. Independent deterministic RNG streams are used for anchor and online sampling and for the two architectures, while Common Random Numbers are deliberately shared across competing policies to obtain paired, variance-reduced comparisons. Diagnostics on the original warm traces are reported separately and are interpreted as plausibility checks rather than proof that the physical cloud process is IID.",
        "",
    ]

    report_path = out / "IID_VALIDATION_REPORT.md"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("PASS" if structural_pass else "FAIL",
          "— structural Monte Carlo IID/RNG validation")
    print()
    print("===== GENERATOR / RNG =====")
    print(f"replace=True: {source_checks['generator_uses_replace_true']}")
    print(f"source audit: {source_checks['source_audit_pass']}")
    print(f"derived seed collisions: {seed_checks['seed_collision_count']}")
    print(f"same-seed reproducibility: {seed_checks['same_seed_exact_reproducibility']}")
    print(f"different-seed different stream: {seed_checks['different_seed_produces_different_stream']}")
    print(f"bit generator: {seed_checks['bit_generator']}")
    for arm, vals in seed_checks["generated_serial_acf_sanity"].items():
        print(
            f"synthetic {arm}: n={vals['n']} "
            f"lag1_acf={vals['lag1_acf']:.6f} "
            f"max|acf|1..20={vals['max_abs_acf_lag1_20']:.6f}"
        )

    print()
    print("===== DIFFERENT-SEED STREAM CORRELATION =====")
    print(
        f"pairs={corr_summary['pairwise_stream_comparisons']} "
        f"mean|r|={corr_summary['mean_abs_correlation']:.6f} "
        f"p95|r|={corr_summary['p95_abs_correlation']:.6f} "
        f"max|r|={corr_summary['max_abs_correlation']:.6f}"
    )

    print()
    print("===== REAL WARM INPUT DIAGNOSTICS =====")
    for k, v in real_summary.items():
        print(f"{k}={v}")

    print()
    print(f"generator={generator_path}")
    print(f"seed_corr={corr_path}")
    print(f"real_tests={real_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--donor-detail", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--donor-seed", type=int, default=11)
    p.add_argument(
        "--mc-seeds",
        default="2026101210,2026101211,2026101212,2026101213,2026101214",
    )
    p.add_argument(
        "--runner-module",
        default="analysis.profiling.transfer_formula_matched_multiseed",
    )
    p.add_argument(
        "--correlation-replicates",
        type=int,
        default=500,
        help="draws per target/seed are replicates * correlation-horizon",
    )
    p.add_argument("--correlation-horizon", type=int, default=50)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
