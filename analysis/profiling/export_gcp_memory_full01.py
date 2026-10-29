#!/usr/bin/env python3
"""Export and analyze a FINISHED Serverledge GCP memory experiment, without touching originals.

Run from the serverledge repository root:
  python3 /path/to/export_gcp_memory_full01.py \
      --source data/profiling/gcp-memory/cloud-runs/full-01 \
      --out data/profiling/gcp-memory/analysis/full-01

Exports: memory_runs.csv, memory_requests.csv, memory_containers.csv,
         memory_mab_decisions.csv, memory_paired.csv, memory_by_condition.csv,
         memory_by_function.csv, source_manifest.csv, provenance.json,
         analysis_summary.txt, plots/*.png.

No network or cloud calls. Existing source files are opened read-only. All output
is written strictly under --out (which must be outside --source).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_TARGETS = 10
EXPECTED_CONFIGS = 9
EXPECTED_REPLICAS = 2
EXPECTED_INVOCATIONS = 50
EXPECTED_CONCURRENCY = 3
CONFIGS = ('N02', 'D02', 'R02', 'N04', 'D04', 'R04', 'N08', 'D08', 'R08')
ARCHS = ('amd64', 'arm64')
EVENT_RE = re.compile(r'(?<!\S)([A-Za-z_][A-Za-z_0-9]*)=([^\s]*)')


def must(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def rows_from_jsonl(path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_dynamic_csv(path, rows, preferred):
    extra = sorted({key for row in rows for key in row if key not in preferred})
    write_csv(path, rows, preferred + extra)


def num(value, label):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{label} is not numeric: {value!r}')
    number = float(value)
    must(math.isfinite(number), f'{label} is not finite')
    return number


def optional_map(aggregate, keys):
    for key in keys:
        candidate = aggregate.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def count_by_arm(mapping, arch):
    return int(mapping.get(arch, 0))


def quantile_nearest_linear(values, fraction):
    sorted_values = sorted(values)
    position = fraction * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def normalized_decision(raw_line):
    values = dict(EVENT_RE.findall(raw_line))
    must(values.get('event') == 'decision_resolved', 'Invalid MAB decision line')
    must(values.get('selected_arm') in ARCHS and values.get('execution_arm') in ARCHS,
         f'Invalid architecture: {raw_line}')
    must(values.get('fallback') in ('true', 'false'), 'Missing fallback flag')
    is_fallback = values['fallback'] == 'true'
    must(is_fallback == (values['selected_arm'] != values['execution_arm']),
         'MAB selected/execution/fallback mismatch')
    must(bool(values.get('request_id')), 'Missing MAB request_id')
    return {
        'request_id': values['request_id'],
        'selected_arm': values['selected_arm'],
        'execution_arm': values['execution_arm'],
        'fallback': is_fallback,
        'fallback_reason': values.get('fallback_reason', ''),
        'raw_log_line': raw_line,
    }


def pct_change(new, old):
    return (100.0 * (new - old) / old) if old else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/profiling/gcp-memory/cloud-runs/full-01'))
    p.add_argument('--out', type=Path, default=Path('data/profiling/gcp-memory/analysis/full-01'))
    p.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')
    args = p.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    must(source.is_dir(), f'Campaign source missing: {source}')
    must(source != output and source not in output.parents and output not in source.parents,
         'Output and source must be separate, non-nested directories')
    must((source/'campaign.json').is_file(), 'campaign.json not found')
    must((source/'final-verification.json').is_file(), 'final-verification.json not found')
    manifest, final = load_json(source/'campaign.json'), load_json(source/'final-verification.json')
    targets, configs = manifest['targets'], manifest['configs']
    must(manifest.get('mode') == 'full', 'Not a full campaign')
    must(len(set(targets)) == len(targets) == EXPECTED_TARGETS, 'Expected ten distinct targets')
    must(set(configs) == set(CONFIGS) and len(configs) == EXPECTED_CONFIGS, 'Unexpected conditions')
    must(manifest['replicas'] == EXPECTED_REPLICAS, 'Unexpected replica count')
    must(manifest.get('run_concurrency') == EXPECTED_CONCURRENCY
         and manifest.get('finished_grid_concurrency') == EXPECTED_CONCURRENCY,
         'Incorrect concurrency')
    must(manifest.get('deleted') is True and final.get('state') == 'DELETE_VERIFIED',
         'Final cleanup is not verified')
    must(final.get('observations') == 9000, 'Final observation count differs from 9000')
    keys = {f'{name}-{config}-r{replica:02d}-q3'
            for name in targets for config in configs for replica in (1, 2)}
    actual_keys = {p.name for p in (source/'runs').iterdir() if p.is_dir()}
    must(keys == actual_keys, f'Run grid mismatch, missing={keys-actual_keys}, unexpected={actual_keys-keys}')

    memory_runs = []
    requests_out = []
    containers_out = []
    decisions_out = []
    source_manifest = []
    all_fallback = 0

    for name in sorted(keys):
        folder = source/'runs'/name
        status, verified, s = [load_json(folder/file) for file in
                               ('status.json', 'verified.json', 'memory-summary.json')]
        must(status.get('status') == 'VERIFIED_LOCAL' and verified.get('state') == 'PASS',
             f'{name}: checkpoint status invalid')
        must(s.get('run_key') == name and status.get('key') == name, f'{name}: run identity mismatch')
        target, config, replica = status['target'], status['condition'], int(status['replica'])
        must(name == f'{target}-{config}-r{replica:02d}-q3', f'{name}: status ID mismatch')
        must(s.get('config_id') == config and s.get('policy') == 'UCB1Decoupled',
             f'{name}: wrong policy/condition')
        must(s.get('concurrency') == EXPECTED_CONCURRENCY and s.get('requests') == EXPECTED_INVOCATIONS,
             f'{name}: malformed concurrency or request count')
        must(s.get('runtime_pin') == manifest.get('runtime_pin'), f'{name}: runtime pin mismatch')

        # Recheck the original SHA-256 inventory, including raw lifecycle/MAB logs.
        checksums = verified.get('sha256') or {}
        must(checksums, f'{name}: empty checksum inventory')
        for relative, digest in checksums.items():
            rel = Path(relative)
            must(not rel.is_absolute() and '..' not in rel.parts, f'{name}: invalid checksum path')
            file = folder/rel
            must(file.is_file() and sha256(file) == digest,
                 f'{name}: missing/corrupt {relative}')

        requests = rows_from_jsonl(folder/'requests.jsonl')
        must(len(requests) == 50 and {r.get('index') for r in requests} == set(range(1, 51)),
             f'{name}: incomplete request set')
        elapsed = []
        for r in requests:
            duration = num(r.get('client_elapsed_ms'), f'{name}: client_elapsed_ms')
            must(duration > 0 and r.get('success') is True, f'{name}: invalid request')
            must(r.get('execution_header_arm') in ARCHS, f'{name}: execution header invalid')
            elapsed.append(duration)
            requests_out.append({
                'run_key': name, 'function_name': target, 'condition': config,
                'c': int(config[1:])/10, 'formula': {'N':'none','D':'difference','R':'ratio'}[config[0]],
                'replica': replica, 'concurrency': 3, 'request_index': r['index'],
                'client_elapsed_ms': duration, 'success': r['success'],
                'is_warm_start': r.get('is_warm_start'),
                'execution_header_arm': r['execution_header_arm'],
                'extra_request_fields_json': json.dumps({k:v for k,v in r.items() if k not in
                    ('index','client_elapsed_ms','success','is_warm_start','execution_header_arm')},
                    sort_keys=True, ensure_ascii=False),
            })

        with (folder/'container-residency.csv').open('r', encoding='utf-8', newline='') as f:
            containers = list(csv.DictReader(f))
        must(len(containers) == int(s['created_containers']), f'{name}: container count mismatch')
        for c in containers:
            # Preserve all original container columns using the 'source_' prefix.
            containers_out.append({
                'run_key': name, 'function_name': target, 'condition': config,
                'c': int(config[1:])/10, 'replica': replica,
                **{'source_'+key: value for key, value in c.items()}
            })

        decision_lines = (folder/'mab-decisions.log').read_text(encoding='utf-8').splitlines()
        decisions = [normalized_decision(line) for line in decision_lines if line.strip()]
        must(len(decisions) == 50 and len({d['request_id'] for d in decisions}) == 50,
             f'{name}: 50 unique MAB decisions not found')
        selected = Counter(d['selected_arm'] for d in decisions)
        executed = Counter(d['execution_arm'] for d in decisions)
        fallback = sum(d['fallback'] for d in decisions)
        for order, decision in enumerate(decisions, 1):
            decisions_out.append({
                'run_key': name, 'function_name': target, 'condition': config,
                'c': int(config[1:])/10, 'replica': replica,
                'log_order': order, **decision,
            })
        mab = s['mab']
        must(int(mab['fallbacks']) == fallback == int(verified['fallbacks']),
             f'{name}: fallback summary disagrees with decision log')
        must(int(mab['decisions']) == 50, f'{name}: wrong number of decisions')
        for arch in ARCHS:
            must(int(mab.get('selected_arm_counts', {}).get(arch, 0)) == selected[arch],
                 f'{name}: selected architecture mismatch for {arch}')
            must(int(mab.get('execution_arm_counts', {}).get(arch, 0)) == executed[arch],
                 f'{name}: execution architecture mismatch for {arch}')
        actual_executed = Counter(r['execution_header_arm'] for r in requests)
        must(executed == actual_executed, f'{name}: execution arm disagrees with HTTP headers')
        all_fallback += fallback

        peak_by_worker = s.get('peak_resident_containers_by_worker', {})
        must(isinstance(peak_by_worker, dict), f'{name}: peak data malformed')
        containers_by_arm = optional_map(s, ('containers_by_arm', 'created_containers_by_arm'))
        memory_by_arm = optional_map(s, ('memory_MB_s_by_arm', 'memory_by_arm_MB_s', 'memory_MiB_s_by_arm'))
        value = num(s['total_memory_MB_s'], f'{name}: total memory')
        must(value > 0, f'{name}: memory must be positive')
        must(math.isclose(statistics.mean(elapsed), float(s['latency_client_mean_ms']),
                          rel_tol=1e-8, abs_tol=1e-6), f'{name}: latency summary mismatch')
        must(int(s.get('warm_invocations', -1)) == sum(r.get('is_warm_start') is True for r in requests),
             f'{name}: warm invocation mismatch')
        must(int(s.get('cold_invocations', -1)) == sum(r.get('is_warm_start') is False for r in requests),
             f'{name}: cold invocation mismatch')
        source_manifest.append({
            'run_key': name,
            'verified_json_sha256': sha256(folder/'verified.json'),
            'memory_summary_sha256': sha256(folder/'memory-summary.json'),
            'request_jsonl_sha256': sha256(folder/'requests.jsonl'),
            'mab_decisions_sha256': sha256(folder/'mab-decisions.log'),
        })
        memory_runs.append({
            'run_key': name, 'function_name': target, 'condition': config,
            'formula': {'N':'none','D':'difference','R':'ratio'}[config[0]],
            'c': int(config[1:])/10, 'replica': replica, 'concurrency': 3,
            'request_count': 50, 'runtime_pin': s['runtime_pin'],
            'total_memory_MB_s': value,
            'memory_amd64_MB_s': memory_by_arm.get('amd64', ''),
            'memory_arm64_MB_s': memory_by_arm.get('arm64', ''),
            'containers_created': int(s['created_containers']),
            'containers_destroyed': int(s.get('destroyed_containers', s['created_containers'])),
            'containers_amd64': containers_by_arm.get('amd64', ''),
            'containers_arm64': containers_by_arm.get('arm64', ''),
            'peak_containers_one_worker': s.get('max_containers_one_worker', ''),
            'peak_worker_amd64': peak_by_worker.get('sl-x86-1', ''),
            'peak_worker_arm64': peak_by_worker.get('sl-arm-1', ''),
            'latency_client_mean_ms': statistics.mean(elapsed),
            'latency_client_median_ms': statistics.median(elapsed),
            'latency_client_p95_ms': quantile_nearest_linear(elapsed, .95),
            'warm_invocations': s['warm_invocations'],
            'cold_invocations': s['cold_invocations'],
            'mab_decisions': len(decisions), 'mab_fallbacks': fallback,
            'mab_fallback_rate_percent': 100*fallback/50,
            'mab_selected_amd64': selected['amd64'], 'mab_selected_arm64': selected['arm64'],
            'mab_execution_amd64': executed['amd64'], 'mab_execution_arm64': executed['arm64'],
            'mab_fallback_reasons_json': json.dumps(mab.get('fallback_reasons', {}), sort_keys=True),
            'validation_class': s.get('validation_class'),
            'source_memory_summary': str((folder/'memory-summary.json').relative_to(source)),
        })

    must(len(memory_runs) == 180 and len(requests_out) == 9000 and len(decisions_out) == 9000,
         'Final counts invalid')

    by_triplet = defaultdict(dict)
    for row in memory_runs:
        label = (row['function_name'], row['c'], row['replica'])
        must(row['formula'] not in by_triplet[label], f'Duplicate triplet formula: {label}')
        by_triplet[label][row['formula']] = row
    must(len(by_triplet) == 60 and all(set(v) == {'none','difference','ratio'} for v in by_triplet.values()),
         'Incomplete N/D/R pairing')
    paired = []
    for (target, c, replica), group in sorted(by_triplet.items()):
        baseline, diff, ratio = [group[v] for v in ('none','difference','ratio')]
        n, d, r = [x['total_memory_MB_s'] for x in (baseline,diff,ratio)]
        paired.append({
            'function_name': target, 'c': c, 'replica': replica,
            'N_memory_MB_s': n, 'D_memory_MB_s': d, 'R_memory_MB_s': r,
            'D_minus_N_MB_s': d-n, 'R_minus_N_MB_s': r-n, 'R_minus_D_MB_s': r-d,
            'D_saving_vs_N_percent': -pct_change(d,n),
            'R_saving_vs_N_percent': -pct_change(r,n),
            'D_minus_N_latency_ms': diff['latency_client_mean_ms']-baseline['latency_client_mean_ms'],
            'R_minus_N_latency_ms': ratio['latency_client_mean_ms']-baseline['latency_client_mean_ms'],
            'N_fallbacks':baseline['mab_fallbacks'],
            'D_fallbacks':diff['mab_fallbacks'],
            'R_fallbacks':ratio['mab_fallbacks'],
            'any_fallback': any(x['mab_fallbacks'] > 0 for x in (baseline,diff,ratio)),
        })

    by_condition = []
    for config in CONFIGS:
        group = [r for r in memory_runs if r['condition'] == config]
        mem = [r['total_memory_MB_s'] for r in group]
        latency = [r['latency_client_mean_ms'] for r in group]
        by_condition.append({
            'condition': config, 'formula': group[0]['formula'], 'c': group[0]['c'],
            'n_runs': len(group), 'n_functions': len({r['function_name'] for r in group}),
            'mean_memory_MB_s': statistics.mean(mem),
            'median_memory_MB_s': statistics.median(mem),
            'min_memory_MB_s': min(mem), 'max_memory_MB_s':max(mem),
            'mean_client_latency_ms':statistics.mean(latency),
            'sum_fallbacks':sum(r['mab_fallbacks'] for r in group),
            'runs_with_fallback':sum(r['mab_fallbacks']>0 for r in group),
            'mean_created_containers':statistics.mean(r['containers_created'] for r in group),
        })

    by_fn = []
    for function in sorted(targets):
        for c in (0.2,0.4,0.8):
            group = [r for r in paired if r['function_name'] == function and r['c'] == c]
            must(len(group) == 2, f'Missing two replicas for {function} c={c}')
            by_fn.append({
                'function_name': function, 'c':c, 'replicas':2,
                'mean_D_saving_vs_N_percent':statistics.mean(r['D_saving_vs_N_percent'] for r in group),
                'mean_R_saving_vs_N_percent':statistics.mean(r['R_saving_vs_N_percent'] for r in group),
                'mean_N_memory_MB_s':statistics.mean(r['N_memory_MB_s'] for r in group),
                'mean_D_memory_MB_s':statistics.mean(r['D_memory_MB_s'] for r in group),
                'mean_R_memory_MB_s':statistics.mean(r['R_memory_MB_s'] for r in group),
                'replica_savings_D_json':json.dumps([r['D_saving_vs_N_percent'] for r in group]),
                'replica_savings_R_json':json.dumps([r['R_saving_vs_N_percent'] for r in group]),
            })

    output.mkdir(parents=True, exist_ok=True)
    write_dynamic_csv(output/'memory_runs.csv', memory_runs,
        ['run_key','function_name','condition','formula','c','replica','concurrency','request_count',
         'total_memory_MB_s','containers_created','peak_containers_one_worker','mab_fallbacks',
         'latency_client_mean_ms','latency_client_median_ms','latency_client_p95_ms'])
    write_dynamic_csv(output/'memory_requests.csv', requests_out,
        ['run_key','function_name','condition','c','formula','replica','request_index',
         'client_elapsed_ms','success','is_warm_start','execution_header_arm'])
    write_dynamic_csv(output/'memory_containers.csv', containers_out,
        ['run_key','function_name','condition','c','replica'])
    write_dynamic_csv(output/'memory_mab_decisions.csv', decisions_out,
        ['run_key','function_name','condition','c','replica','log_order','request_id',
         'selected_arm','execution_arm','fallback','fallback_reason','raw_log_line'])
    write_dynamic_csv(output/'memory_paired.csv', paired,
        ['function_name','c','replica','N_memory_MB_s','D_memory_MB_s','R_memory_MB_s',
         'D_saving_vs_N_percent','R_saving_vs_N_percent','any_fallback'])
    write_dynamic_csv(output/'memory_by_condition.csv', by_condition,
        ['condition','formula','c','n_runs','mean_memory_MB_s','median_memory_MB_s',
         'mean_client_latency_ms','sum_fallbacks'])
    write_dynamic_csv(output/'memory_by_function.csv', by_fn,
        ['function_name','c','replicas','mean_D_saving_vs_N_percent','mean_R_saving_vs_N_percent'])
    write_dynamic_csv(output/'source_manifest.csv', source_manifest, ['run_key'])

    metadata = {
        'source_campaign': str(source), 'source_manifest_sha256':sha256(source/'campaign.json'),
        'final_verification_sha256':sha256(source/'final-verification.json'),
        'exporter_sha256':sha256(Path(__file__)), 'runtime_pin':manifest['runtime_pin'],
        'dataset_rows': {'memory_runs':len(memory_runs),'memory_requests':len(requests_out),
                         'memory_containers':len(containers_out),'memory_mab_decisions':len(decisions_out),
                         'memory_paired':len(paired),'memory_by_condition':len(by_condition),
                         'memory_by_function':len(by_fn)},
        'request_fallbacks_total':all_fallback,
        'definitions': {
            'total_memory_MB_s': 'Configured container memory (memory_mb) multiplied by container lifespan in seconds; not RSS or energy',
            'client_elapsed_ms':'HTTP client-observed elapsed time, not the MAB synthetic reward',
            'mab_decisions_log_order':'Order of decision_resolved lines, not necessarily original request arrival/order',
            'memory_requests_join':'No per-request join to MAB decisions is asserted; execution arm aggregate cross-checked',
            'D_saving_vs_N_percent':'100*(N-D)/N within same target, c, replica',
            'R_saving_vs_N_percent':'100*(N-R)/N within same target, c, replica',
            'replicates':'Only two GCP serving replications; do not interpret 20 runs per condition as 20 independent function-level samples',
            'missing_memory_by_arm':'Arm-specific summary left blank when the source summary does not provide a matching named mapping',
        }
    }
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    totals = []
    for config in CONFIGS:
        x = next(x for x in by_condition if x['condition'] == config)
        totals.append(f"{config}: mean_memory={x['mean_memory_MB_s']:.3f} MB.s, "
                      f"mean_latency={x['mean_client_latency_ms']:.3f} ms, "
                      f"fallbacks={x['sum_fallbacks']}")
    intro = [
        f'RUNS={len(memory_runs)} REQUESTS={len(requests_out)} '
        f'MAB_DECISIONS={len(decisions_out)} CONTAINERS={len(containers_out)}',
        f'TOTAL_FALLBACKS={all_fallback} FALLBACK_RATE={100*all_fallback/len(requests_out):.4f}%',
        f'PAIRED_TRIPLETS={len(paired)}; fully fallback-free={sum(not r["any_fallback"] for r in paired)}',
        'NOTE: Aggregated memory is configured memory × residency; do not infer RSS or energy.',
        'NOTE: D vs N and R vs N comparisons are paired by function,c,replica.',
        'NOTE: Two replicas are not enough for strong inference about variance across repetitions.',
        '', 'BY CONDITION:', *totals,
        '', 'SAVINGS VS MATCHED N (positive means less memory):',
    ]
    for c in (0.2,0.4,0.8):
        subset = [r for r in paired if r['c'] == c]
        for key in ('D','R'):
            field = key+'_saving_vs_N_percent'
            intro.append(f'c={c} {key}: mean paired saving={statistics.mean(r[field] for r in subset):.3f}%, '
                         f'median paired saving={statistics.median(r[field] for r in subset):.3f}%, '
                         f'{sum(r[field]>0 for r in subset)}/{len(subset)} pairs save memory')
    (output/'analysis_summary.txt').write_text('\n'.join(intro)+'\n',encoding='utf-8')

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError('matplotlib is needed for plots; use --no-plots to export CSV only') from exc
        plots = output/'plots'
        plots.mkdir(exist_ok=True)
        for c in (0.2,0.4,0.8):
            subset = [r for r in paired if r['c'] == c]
            fig, ax = plt.subplots(figsize=(10,4.8))
            x = list(range(len(subset)))
            ax.scatter(x,[r['D_saving_vs_N_percent'] for r in subset], s=18, label='Difference vs No-TL')
            ax.scatter(x,[r['R_saving_vs_N_percent'] for r in subset], s=18, marker='x', label='Ratio vs No-TL')
            ax.axhline(0, linewidth=1)
            ax.set(title=f'Memory change vs paired No-TL — c={c}',
                   xlabel='Paired function/replica (sorted)', ylabel='Memory saving (%)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots/f'paired_savings_c{str(c).replace(".","")}.png',dpi=170)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(10,5))
        ordered = [next(row for row in by_condition if row['condition']==code) for code in CONFIGS]
        ax.bar(range(9),[r['mean_memory_MB_s'] for r in ordered])
        ax.set_xticks(range(9),CONFIGS)
        ax.set_ylabel('Mean configured-memory residency (MB.s)')
        ax.set_title('Memory across 10 functions × 2 replicas per condition')
        fig.tight_layout()
        fig.savefig(plots/'memory_by_condition.png',dpi=170)
        plt.close(fig)

    print('\n'.join(intro))
    print(f'\nPASS: derived CSVs and analysis written to {output}')


if __name__ == '__main__':
    main()
