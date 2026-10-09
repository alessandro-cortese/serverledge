#!/usr/bin/env python3
"""Paired LOFO donor diagnostics on a common selected support.

Uses the per-target/per-seed CSV already written by
compare_59_clustering_and_plots.py. No new clustering or profiling.
The bootstrap resampling unit is target_function (not seed or request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [
    ('A_legacy_K5', 'B_hybrid_K6'),
    ('B_hybrid_K6', 'C_final_weighted'),
    ('A_legacy_K5', 'C_final_weighted'),
    ('C_final_weighted', 'C_lang025'),
    ('C_final_weighted', 'C_lang100'),
]
DIRECTIONAL = {'x86-preferred', 'arm-preferred'}
METRICS = [
    ('raw_agreement', 'all_selected', 100.0, 'pp'),
    ('directional_agreement', 'directional', 100.0, 'pp'),
    ('regret_percent', 'directional', 1.0, 'percentage_points'),
    ('realized_speedup', 'directional', 1.0, 'log_speedup'),
    ('same_language', 'all_selected', 100.0, 'pp'),
    ('multiplicative_mismatch', 'all_selected', 1.0, 'log_mismatch'),
]
REQUIRED = { 'config', 'seed', 'target_function', 'target_preference', 'status',
             'donor_function', *(m[0] for m in METRICS) }


def paired_view(data: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    left = data.loc[data.config.eq(a)].drop(columns=['config']).set_index(['target_function', 'seed'])
    right = data.loc[data.config.eq(b)].drop(columns=['config']).set_index(['target_function', 'seed'])
    if set(left.index) != set(right.index):
        raise ValueError(f'Unpaired target-seed keys: {a} vs {b}')
    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError('Repeated target-seed key')
    matched = left.join(right, how='inner', lsuffix='_a', rsuffix='_b', validate='one_to_one')
    if not matched.target_preference_a.eq(matched.target_preference_b).all():
        raise ValueError('Ground-truth preferences differ between configurations')
    selected = matched.status_a.eq('selected') & matched.status_b.eq('selected')
    result = matched.loc[selected].reset_index()
    if result.empty:
        raise ValueError(f'No common selected support: {a} vs {b}')
    return result


def seed_for(seed: int, a: str, b: str, metric: str) -> int:
    value = f'{seed}|{a}|{b}|{metric}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


def bootstrap(values: np.ndarray, *, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    # Small batches avoid large peak memory allocations for optional big runs.
    for start in range(0, n, 1000):
        end = min(n, start + 1000)
        draws = rng.integers(0, len(values), size=(end-start, len(values)))
        out[start:end] = values[draws].mean(axis=1)
    return tuple(map(float, np.quantile(out, [0.025, 0.975])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lofo-detail', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026100818)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit('Use >=100 bootstrap resamples')
    raw = args.lofo_detail.read_bytes()
    data = pd.read_csv(args.lofo_detail)
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    if data[['config','seed','target_function']].duplicated().any():
        raise SystemExit('Duplicate config-seed-target records')

    rows = []
    support_rows = []
    target_rows = []
    for a,b in PAIRS:
        matched = paired_view(data, a, b)
        all_keys = data.loc[data.config.eq(a), ['target_function','seed']]
        coverage = float(len(matched)/len(all_keys))
        directional = matched.target_preference_a.isin(DIRECTIONAL)
        support_rows.append(dict(comparison=f'{b}_minus_{a}', total_target_seed_keys=len(all_keys),
                                 common_selected_target_seed_keys=len(matched),
                                 common_support_fraction=coverage,
                                 common_selected_targets=matched.target_function.nunique(),
                                 common_directional_target_seed_keys=int(directional.sum()),
                                 common_directional_targets=matched.loc[directional,'target_function'].nunique()))
        for metric, subset, scale, unit in METRICS:
            frame = matched.loc[directional] if subset=='directional' else matched
            if frame.empty:
                continue
            aa, bb = frame[f'{metric}_a'].astype(float), frame[f'{metric}_b'].astype(float)
            if aa.isna().any() or bb.isna().any():
                raise ValueError(f'Missing metric {metric}: {a} vs {b}')
            if unit.startswith('log_'):
                if (aa <= 0).any() or (bb <= 0).any():
                    raise ValueError(f'Nonpositive {metric} for geometric comparison')
                diffs = np.log(bb.to_numpy()) - np.log(aa.to_numpy())
            else:
                diffs = (bb.to_numpy() - aa.to_numpy()) * scale
            by_target = pd.DataFrame({'target_function':frame.target_function,'delta':diffs})
            by_target = by_target.groupby('target_function',sort=True).delta.mean()
            vec = by_target.to_numpy(dtype=float)
            low,high = bootstrap(vec,n=args.bootstrap_replicates,seed=seed_for(args.bootstrap_seed,a,b,metric))
            point = float(vec.mean())
            if unit.startswith('log_'):
                point, low, high = [float(np.exp(v)) for v in (point,low,high)]
                final_unit = 'geometric_ratio_B_over_A'
            else:
                final_unit = unit
            rows.append(dict(comparison=f'{b}_minus_{a}',metric=metric,unit=final_unit,
                             point_delta=point,ci95_low=low,ci95_high=high,
                             paired_targets=int(len(vec)),matched_target_seed_rows=len(frame),
                             common_selected_coverage=coverage))
            for target, delta in by_target.items():
                target_rows.append(dict(comparison=f'{b}_minus_{a}',metric=metric,
                                        target_function=target,mean_seed_delta=float(delta)))

    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'donor_59_paired_bootstrap.csv',index=False)
    pd.DataFrame(support_rows).to_csv(out/'donor_59_common_support.csv',index=False)
    pd.DataFrame(target_rows).to_csv(out/'donor_59_per_target_deltas.csv',index=False)
    (out/'donor_59_paired_manifest.json').write_text(json.dumps(dict(
        input_path=str(args.lofo_detail.resolve()),
        input_sha256=hashlib.sha256(raw).hexdigest(),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        resampling_unit='target_function, seed first averaged within target',
        comparisons=[list(x) for x in PAIRS],
        directional_definition='x86-preferred or arm-preferred, tau=2.5 percent',
        use_only_common_selected_support=True,
        note='Geometric ratios use bootstrap on per-target mean log differences.'
    ),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

    key_metrics={'directional_agreement','regret_percent','raw_agreement','same_language'}
    selected_rows=[r for r in rows if r['metric'] in key_metrics]
    lines=['# Donor comparison — paired common-support analysis (59 functions)', '',
           'This report is based on the current LOFO CSV; no new clustering was performed.',
           'For each pair, keep only identical target×seed records with a selected donor in both configurations.',
           'Within each target, average over seeds; bootstrap **functions** rather than treating 5 seeds as independent observations.',
           'Consequently the numerical point deltas can differ from the original unpaired-coverage summary.',
           '', '## Support', '',
           '| Pair (B minus A) | Shared target×seed | Share | Directional target×seed |',
           '|---|---:|---:|---:|']
    for r in support_rows:
        lines.append(f"| {r['comparison']} | {r['common_selected_target_seed_keys']}/{r['total_target_seed_keys']} | {r['common_support_fraction']:.2%} | {r['common_directional_target_seed_keys']} |")
    lines+=['','## Paired differences and percentile bootstrap','',
            'Positive delta: B higher than A; for regret lower is better. CI crossing 0 is inconclusive.',
            '| Pair | Metric | Δ | CI95% | Targets |','|---|---|---:|---:|---:|']
    for r in selected_rows:
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['point_delta']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['paired_targets']} |")
    lines+=['','**Interpretation note:** these are exploratory percentile intervals without adjustment for multiple comparisons;',
            'conclusions should distinguish clustering/donor diagnostics from downstream UCB1 latency effects.',
            'Geometric ratios and all target deltas are preserved in the CSV.', '']
    (out/'DONOR_59_PAIRED_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PASS — paired donor comparison on common support')
    print(pd.DataFrame(support_rows).to_string(index=False))
    print(pd.DataFrame(rows).query('metric in ["directional_agreement", "regret_percent"]').to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('REPORT:',out/'DONOR_59_PAIRED_REPORT.md')

if __name__ == '__main__':
    main()
