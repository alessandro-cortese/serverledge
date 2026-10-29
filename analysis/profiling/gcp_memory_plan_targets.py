#!/usr/bin/env python3
"""Plan an auditable 10-target memory experiment from the frozen RF ground truth.

Reads labels ONLY to stratify the *evaluation* set; these labels must never be
passed to donor selection, prior computation, or MAB. No GCP operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from analysis.profiling.gcp_memory_prepare import CONFIGS, read_catalog, read_ground_truth

CLASSES = ('ARM-faster', 'ARM-slower', 'ARM-similar')
QUOTAS = {'ARM-faster': 3, 'ARM-slower': 3, 'ARM-similar': 4}
ORIGINAL_TARGETS = ('billibench-knn', 'dynamic-html-py', 'functionbench-gzip', 'graph-bfs')


def family(name: str) -> str:
    name = re.sub(r'^twin-', '', name)
    name = re.sub(r'-(py|node)$', '', name)
    return re.sub(r'[^a-z0-9]', '', name.lower())


def choose_six(catalog: dict, labels: dict, original=ORIGINAL_TARGETS):
    original = tuple(original)
    if len(set(original)) != len(original):
        raise ValueError('Original targets are not unique')
    missing = [fn for fn in original if fn not in catalog or fn not in labels]
    if missing:
        raise ValueError(f'Missing original targets in catalog/ground truth: {missing}')
    if len(original) != 4:
        raise ValueError('Exactly four already-prepared targets expected')
    candidates = set(catalog).intersection(labels).difference(original)
    if len(candidates) < 6:
        raise ValueError('Less than six eligible new functions')
    selected = list(original)
    for _ in range(6):
        counts = Counter(labels[f] for f in selected)
        fams = {family(f) for f in selected}
        runtimes = Counter(catalog[f]['runtime'] for f in selected)
        def ranking(fn):
            cls = labels[fn]
            deficit = max(QUOTAS[cls] - counts[cls], 0)
            # Strong preference for filling class deficits, then new semantic
            # families, then underrepresented runtimes. Alphabetic tie-break.
            return (-deficit, int(family(fn) in fams), runtimes[catalog[fn]['runtime']],
                    counts[cls], fn)
        choice = min(candidates, key=ranking)
        selected.append(choice)
        candidates.remove(choice)
    return selected


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_prepared(root: Path, name: str, replica: str) -> dict:
    folder = root / 'data/profiling/gcp-memory/priors' / f'{name}-{replica}'
    manifest_path = folder / 'grid-manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Preparazione assente per {name}: {manifest_path}')
    doc = json.loads(manifest_path.read_text(encoding='utf-8'))
    if doc.get('target') != name or doc.get('replica') != replica:
        raise ValueError(f'Preparazione incoerente per {name}')
    expected = {x[0] for x in CONFIGS}
    entries = doc.get('conditions') or []
    if len(entries) != 9 or {x.get('config_id') for x in entries} != expected:
        raise ValueError(f'Griglia incompleta per {name}')
    mapping = {x['config_id']: x for x in entries}
    for config_id, mode, wr, we, c in CONFIGS:
        entry = mapping[config_id]
        if (entry.get('formula') != mode or entry.get('policy') != 'UCB1Decoupled'
                or entry.get('prior_applied') is not (mode != 'none')
                or not math.isclose(float(entry.get('c', -1)), c, abs_tol=1e-12)
                or (mode != 'none' and (
                    not math.isclose(float(entry.get('w_R', -1)), wr, abs_tol=1e-12)
                    or not math.isclose(float(entry.get('w_E', -1)), we, abs_tol=1e-12)))):
            raise ValueError(f'Configurazione congelata incoerente: {name}/{config_id}')
        if entry.get('prior_applied'):
            cid = entry['config_id']
            prior_path = folder / cid / 'frozen-prior.json'
            meta_path = folder / cid / 'manifest.json'
            if not prior_path.is_file() or not meta_path.is_file():
                raise ValueError(f'Bundle mancante {name}/{cid}')
            manifest = json.loads(meta_path.read_text(encoding='utf-8'))
            got = sha256(prior_path)
            if got != entry.get('prior_sha256') or got != manifest.get('prior_sha256'):
                raise ValueError(f'SHA errato {name}/{cid}')
    if not doc.get('donor') or doc['donor'] == name:
        raise ValueError(f'Donor invalido per {name}')
    return {'manifest': str(manifest_path.relative_to(root)),
            'manifest_sha256': sha256(manifest_path), 'donor': doc.get('donor'),
            'bootstrap_commit': doc.get('bootstrap_commit')}


def build(root: Path, replica: str, dest: Path, original=ORIGINAL_TARGETS) -> dict:
    catalog = read_catalog(root)
    truth_path = (root / 'data/profiling/gcp-paper-rf-schema6/analysis/'
                  'paper-rf-classifier/paper_ground_truth.csv')
    gt = read_ground_truth(truth_path)
    if len(gt) != 67 or len(catalog) != 67:
        raise ValueError(f'Corpus non corrispondente a 67 funzioni: catalog={len(catalog)}, gt={len(gt)}')
    labels = {fn: row.get('paper_class') for fn, row in gt.items()}
    unexpected = {k: v for k, v in labels.items() if v not in CLASSES}
    if unexpected:
        raise ValueError(f'Etichette non ammesse: {unexpected}')
    if set(catalog) != set(labels):
        raise ValueError('Catalogo e ground truth non contengono le stesse funzioni')
    original_info = {fn: validate_prepared(root, fn, replica) for fn in original}
    chosen = choose_six(catalog, labels, original)
    rows = []
    for i, fn in enumerate(chosen):
        prepared = i < len(original)
        rows.append({
            'order': i+1, 'function_name': fn, 'paper_class': labels[fn],
            'runtime': catalog[fn]['runtime'], 'memory_mb': catalog[fn]['memory_mb'],
            'semantic_family': family(fn),
            'phase': 'PREPARED_BUNDLES_NEED_RUNTIME_VALIDATION' if prepared else 'NEW_TARGET_NEEDS_PROFILE_AND_DONOR_EVIDENCE',
            'donor': original_info[fn]['donor'] if prepared else '',
        })
    if dest.exists():
        raise FileExistsError(f'Non sovrascrivo directory di piano esistente: {dest}')
    dest.mkdir(parents=True, exist_ok=False)
    csv_path = dest / 'target_selection.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    info = {
        'status': 'OFFLINE_SELECTION_ONLY_NOT_GCP_READY',
        'experiment': 'gcp-memory', 'count': len(chosen),
        'existing_targets': list(original), 'new_targets': chosen[4:],
        'class_counts': dict(Counter(labels[fn] for fn in chosen)),
        'semantic_families': len({family(fn) for fn in chosen}),
        'runtimes': dict(Counter(catalog[fn]['runtime'] for fn in chosen)),
        'ground_truth': str(truth_path.relative_to(root)),
        'ground_truth_sha256': sha256(truth_path),
        'original_prior_manifests': original_info,
        'target_selection_sha256': sha256(csv_path),
        'selection_use_of_ground_truth': 'Only evaluation-strata selection; NEVER to select donor or build prior',
        'new_target_procedure': [
            'Profile target on AMD64 only (controlled warm samples) to build anchor and feature vector',
            'Run original donor selector excluding target and without target ARM results',
            'Collect donor warm reward observations on both arms; construct source evidence',
            'Materialize and verify nine conditions via existing gcp_memory_prepare',
            'Validate prior/runtime compatibility; only then run memory GCP campaign',
        ],
    }
    (dest / 'target_selection_manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('PASS: selezione offline pronta (nessuna chiamata GCP)')
    print(f"Classi: {info['class_counts']}; famiglie semantiche: {info['semantic_families']}")
    print('Sei candidati NUOVI:')
    for r in rows[4:]:
        print('  ', r['function_name'], '|', r['paper_class'], '|', r['runtime'])
    print('CSV:', csv_path)
    print('Manifest:', dest / 'target_selection_manifest.json')
    print('ATTENZIONE: i sei candidati NON hanno ancora evidence, bootstrap o prior.')
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--replica', default='r01')
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    root = args.repo_root.resolve()
    dest = args.output_dir or (root / 'data/profiling/gcp-memory/planning/target_selection_v1')
    build(root, args.replica, dest.resolve())


if __name__ == '__main__':
    main()
