#!/usr/bin/env python3
"""Build reproducible, read-only-derived Serverledge architecture-comparison CSVs.

Original GCP profiling datasets, the historical 21-proxy-feature representation,
static measurements, and ground truth remain untouched. The generated feature
CSVs deliberately DO NOT contain ARM-derived ground-truth labels.

Usage (from Serverledge repository root):
    python3 analysis/profiling/build_architecture_comparison_datasets.py
    python3 analysis/profiling/build_architecture_comparison_datasets.py --output-dir data/architecture-comparison/datasets/v2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final/paper-rf-dataset.csv')
SOURCE_PROVENANCE = Path('data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final/paper-rf-provenance.csv')
PROXY = Path('data/profiling/gcp-paper-rf-schema6/analysis/paper-rf-classifier/diagnostica_21/paper_rf_dataset_21_proxy.csv')
TRUTH = Path('data/profiling/gcp-paper-rf-schema6/analysis/paper-rf-classifier/paper_ground_truth.csv')
STATIC = Path('data/profiling/final-corpus64-20261005-analysis-01/static-code-metrics.csv')
DEFAULT_OUTPUT = Path('data/architecture-comparison/datasets/v1')
EXPECTED_SOURCE_SHA256 = '50a7738aa1c8dd8d45abe0c89749dbc4351c6318f26cf6eafa7c4da86b61c3a5'
EXCLUDED_STATIC = {'chacha20', 'primenumber', 'readdisk', 'readmemory', 'thread'}
STATIC_REQUIRED = ('static_token_count_mean', 'static_function_count')
KEYS = ('function_name', 'machine_tag', 'experiment_id', 'request_id')
NATIVE_FEATURES = (
    'totalMemory', 'freeMemory', 'frameworkRuntime', 'userRuntime',
    'cpuUserDelta', 'cpuKernelDelta', 'cpuIdleDelta', 'cpuIOWaitDelta',
    'cpuSoftIrqDelta', 'cpuStealDelta', 'pageFaultsDelta', 'utilizedCPUs',
    'frameworkRuntimeDeltas', 'runtime',
)
PROXY_FEATURES = (
    'pageFaults', 'cpuNiceDelta', 'cpuIrqDelta', 'cpuGuestDelta',
    'cpuGuestNiceDelta', 'availableCPUs', 'recommendedMemory',
)
STATIC_METADATA = {'language', 'source_path', 'source_resolution', 'static_source_available'}


def fail(message: str) -> None:
    raise SystemExit('ERRORE: ' + message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path):
    if not path.is_file():
        fail(f'file mancante: {path}')
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if not header or len(header) != len(set(header)):
            fail(f'header assente o duplicato: {path}')
        rows = list(reader)
        if any(None in row for row in rows):
            fail(f'CSV con righe malformate: {path}')
    return header, rows


def unique_by(rows, cols, name):
    index = {}
    for row in rows:
        key = tuple(row[col] for col in cols)
        if key in index:
            fail(f'chiave duplicata in {name}: {key}')
        index[key] = row
    return index


def finite_number(value, description):
    try:
        n = float(value)
    except (ValueError, TypeError):
        fail(f'numero non valido in {description}: {value!r}')
    if not math.isfinite(n):
        fail(f'numero non finito in {description}: {value!r}')
    return n


def label(delta, threshold):
    if delta < -threshold:
        return 'arm-preferred'
    if delta > threshold:
        return 'x86-preferred'
    return 'architecture-independent'


def paper_label(label_serverledge):
    return {
        'arm-preferred': 'ARM-faster',
        'x86-preferred': 'ARM-slower',
        'architecture-independent': 'ARM-similar',
    }[label_serverledge]


def bool_available(value):
    return str(value).strip().lower() in ('true', '1', 'yes')


def write_csv(path, header, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def build(output: Path):
    inputs = {
        'native_gcp': SOURCE,
        'gcp_provenance': SOURCE_PROVENANCE,
        'feature21_proxy': PROXY,
        'ground_truth_original': TRUTH,
        'static_metrics_historical': STATIC,
    }
    for p in inputs.values():
        if not p.is_file():
            fail(f'input non trovato: {p}')
    source_sha = digest(SOURCE)
    if source_sha != EXPECTED_SOURCE_SHA256:
        fail(f'SHA originale cambiato: {source_sha}')

    native_cols, native = load_csv(SOURCE)
    proxy_cols, proxy = load_csv(PROXY)
    provenance_cols, provenance = load_csv(SOURCE_PROVENANCE)
    gt_cols, groundtruth = load_csv(TRUTH)
    static_cols, static = load_csv(STATIC)
    if len(native) != 13400 or len(proxy) != 13400 or len(provenance) != 13400:
        fail(f'numero campioni: native={len(native)}, proxy={len(proxy)}, provenance={len(provenance)}')
    if len(native_cols) != 22 or len(proxy_cols) != 29:
        fail(f'colonne inattese: native={len(native_cols)}, proxy={len(proxy_cols)}')
    if not set(KEYS).issubset(native_cols):
        fail('chiavi di unione mancanti dal CSV originale')
    if not set(native_cols).issubset(proxy_cols):
        fail('colonne originali mancanti nel CSV a 21 feature')
    if set(proxy_cols) - set(native_cols) != set(PROXY_FEATURES):
        fail('feature aggiunte non corrispondenti alle 7 attese')
    if not set(NATIVE_FEATURES).issubset(native_cols):
        fail('feature native mancanti')
    if not set(STATIC_REQUIRED + ('function_name','static_source_available','source_path','source_resolution')).issubset(static_cols):
        fail('feature statiche o metadati sorgente mancanti')
    if not set(('function_name','x86_median_duration_ms','arm_median_duration_ms','delta_percent','paper_class','serverledge_class')).issubset(gt_cols):
        fail('colonne ground truth mancanti')

    index_proxy = unique_by(proxy, KEYS, 'proxy21')
    index_native = unique_by(native, KEYS, 'profiling originale')
    if set(index_proxy) != set(index_native):
        fail('chiavi di invocazione native e proxy diverse')
    index_static = unique_by(static, ('function_name',), 'statiche')
    index_gt = unique_by(groundtruth, ('function_name',), 'ground truth')

    candidate_names = set(index_native[key]['function_name'] for key in index_native)
    if len(candidate_names) != 67 or set(index_gt) != {(name,) for name in candidate_names}:
        fail(f'funzioni attese=67: native={len(candidate_names)}, GT={len(index_gt)}')
    twins = {name for name in candidate_names if name.startswith('twin-')}
    excluded = EXCLUDED_STATIC | twins
    eligible = candidate_names - excluded
    if len(twins) != 3 or len(eligible) != 59:
        fail(f'selezione inattesa: twins={len(twins)}, eligible={len(eligible)}')

    # Verify all original 22 columns are preserved semantically. 'runtime'
    # differs by at most ~4.5e-13 from an earlier CSV reserialization.
    near_rounding = Counter()
    for key, original in index_native.items():
        derived = index_proxy[key]
        for col in native_cols:
            a, b = original[col], derived[col]
            if a == b:
                continue
            if col in NATIVE_FEATURES or col in ('configured_cpus', 'configured_memory_mb'):
                fa = finite_number(a, f'original/{col} {key}')
                fb = finite_number(b, f'proxy/{col} {key}')
                if not math.isclose(fa, fb, rel_tol=1e-10, abs_tol=1e-9):
                    fail(f'feature originale modificata: {key} {col}: {a} vs {b}')
                near_rounding[col] += 1
            else:
                fail(f'metadato originale modificato: {key} {col}: {a} vs {b}')
        for col in PROXY_FEATURES:
            finite_number(derived[col], f'proxy21/{col} {key}')

    counts = Counter((row['function_name'], row['machine_tag']) for row in native)
    if len(counts) != 134 or set(counts.values()) != {100}:
        fail(f'profili non bilanciati: {len(counts)} coppie e conteggi={Counter(counts.values())}')
    if set(tag for _, tag in counts) != {'amd64','arm64'}:
        fail(f'architetture non attese: {set(tag for _,tag in counts)}')

    # Every eligible function must have trustworthy static metrics.
    for name in sorted(eligible):
        if (name,) not in index_static:
            fail(f'statiche mancanti per {name}')
        st = index_static[(name,)]
        if not bool_available(st['static_source_available']):
            fail(f'sorgente statica non disponibile per {name}')
        for col in STATIC_REQUIRED:
            n = finite_number(st[col], f'{name}/{col}')
            if n < 0:
                fail(f'feature statica negativa: {name}/{col}')
    if not any(index_static[(name,)]['source_resolution'] == 'explicit-repository-override' for name in eligible):
        print('NOTA: nessun override esplicito tra le 59: verificare se il corpus statico è cambiato')

    gt_output = []
    for name in sorted(candidate_names):
        t = index_gt[(name,)]
        x86 = finite_number(t['x86_median_duration_ms'], f'{name}/x86')
        arm = finite_number(t['arm_median_duration_ms'], f'{name}/arm')
        reported = finite_number(t['delta_percent'], f'{name}/delta')
        if min(x86, arm) <= 0:
            fail(f'durata non positiva: {name}')
        calculated = 100.0 * (arm / x86 - 1.0)
        if not math.isclose(calculated, reported, rel_tol=1e-10, abs_tol=1e-8):
            fail(f'delta incoerente per {name}: {calculated} vs {reported}')
        mapped15 = label(reported, 15)
        if t['paper_class'] != paper_label(mapped15):
            fail(f'classe paper incoerente per {name}')
        if t['serverledge_class'] != mapped15:
            fail(f'colonna serverledge_class NON è mappatura 15% per {name}')
        gt_output.append({
            'function_name': name,
            'x86_median_duration_ms': t['x86_median_duration_ms'],
            'arm_median_duration_ms': t['arm_median_duration_ms'],
            'arm_vs_x86_delta_percent': t['delta_percent'],
            'paper_class_15': t['paper_class'],
            'serverledge_preference_2p5': label(reported, 2.5),
            'measured_faster_architecture': ('amd64' if x86 <= arm else 'arm64'),
        })

    gt_header = [
        'function_name', 'x86_median_duration_ms', 'arm_median_duration_ms',
        'arm_vs_x86_delta_percent', 'paper_class_15',
        'serverledge_preference_2p5', 'measured_faster_architecture',
    ]
    feature_header = native_cols + list(PROXY_FEATURES) + [col for col in static_cols if col != 'function_name'] + ['eligible_common59']
    if len(feature_header) != len(set(feature_header)):
        fail('collisione nei nomi delle colonne del dataset arricchito')
    all_rows, common_rows = [], []
    for row in native:
        name = row['function_name']
        proxy_row = index_proxy[tuple(row[col] for col in KEYS)]
        st = index_static.get((name,))
        extras = {col: proxy_row[col] for col in PROXY_FEATURES}
        static_values = {}
        for col in static_cols:
            if col == 'function_name':
                continue
            if st is None:
                static_values[col] = ''
            elif name in eligible or col in STATIC_METADATA:
                static_values[col] = st[col]
            else:
                # Do not propagate numerical static metrics of unreliable wrappers.
                static_values[col] = ''
        enriched = {
            **row,
            **extras,
            **static_values,
            'eligible_common59': int(name in eligible),
        }
        all_rows.append(enriched)
        if name in eligible:
            common_rows.append(enriched)

    if len(common_rows) != 11800:
        fail(f'dimensione common59 inattesa: {len(common_rows)}')

    eligibility = []
    for name in sorted(candidate_names):
        reason = ('static_source_unreliable' if name in EXCLUDED_STATIC
                  else 'twin_excluded' if name in twins else 'included')
        st = index_static.get((name,), {})
        eligibility.append({
            'function_name': name, 'eligible_common59': int(name in eligible),
            'reason': reason,
            'static_source_available': st.get('static_source_available', ''),
            'source_resolution': st.get('source_resolution', ''),
            'source_path': st.get('source_path', ''),
        })

    # Only now write outputs. Never overwrite existing directories or source files.
    if output.exists():
        fail(f'directory di output già presente: {output}. Usa --output-dir con una nuova versione.')
    output.mkdir(parents=True, exist_ok=False)
    written = []
    outputs = {
        'profiling_all67_enriched.csv': (feature_header, all_rows),
        'profiling_common59_enriched.csv': (feature_header, common_rows),
        'ground_truth_all67.csv': (gt_header, gt_output),
        'ground_truth_common59.csv': (gt_header, [r for r in gt_output if r['function_name'] in eligible]),
        'function_eligibility.csv': (
            ['function_name','eligible_common59','reason','static_source_available','source_resolution','source_path'],
            eligibility,
        ),
    }
    for filename, (header, rows) in outputs.items():
        p = output / filename
        write_csv(p, header, rows)
        written.append((filename, len(rows), digest(p)))

    doc = {
        'purpose': 'Controlled architecture comparison, GCP profiling and static metrics',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_files_sha256': {label: {'path': str(p), 'sha256': digest(p)} for label, p in inputs.items()},
        'source_original_22_columns_preserved': True,
        'original_22_column_numeric_rounding_in_proxy': dict(near_rounding),
        'native_features_count': len(NATIVE_FEATURES),
        'proxy_features_count': len(PROXY_FEATURES),
        'proxy_features': list(PROXY_FEATURES),
        'proxy_warning': 'Seven reconstructed/proxy columns have NOT yet been semantically validated against SAAF/paper.',
        'eligible_functions': sorted(eligible),
        'excluded_functions': sorted(excluded),
        'static_metrics_source': str(STATIC),
        'static_provenance_warning': 'Historical static metrics reused; audit source files/overrides against GCP workload implementations.',
        'label_definitions': {
            'delta_percent': '100*(ARM_median/x86_median - 1)',
            'paper_class_15': 'ARM-faster if <-15; ARM-slower if >15; otherwise ARM-similar',
            'serverledge_preference_2p5': 'arm-preferred if <-2.5; x86-preferred if >2.5; otherwise independent',
            'measured_faster_architecture': 'smaller median, amd64 on exact tie; diagnostic only, NOT a model feature',
        },
        'safety': 'No ground-truth labels in profiling feature CSVs. Fit train-only scaler and RF in group/function holdout.',
        'outputs': [{'file': name, 'rows': n, 'sha256': hash_} for name, n, hash_ in written],
    }
    (output / 'dataset_manifest.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PASS dataset architecture-comparison')
    print(f'Output: {output}')
    for name, n, _ in written:
        print(f'  {name}: {n} righe')
    print('  dataset_manifest.json: SHA-256/provenienza/etichette')
    print('Controlli: 67 funzioni, 59 ammissibili, 13400/11800 campioni, 100 per coppia, sorgenti immutati')
    print('NOTA: le sette proxy SAAF restano da validare semanticamente; dataset arricchito != replica fedele del paper.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    build(a.output_dir)


if __name__ == '__main__':
    main()
