#!/usr/bin/env python3
"""Offline preparation of the frozen nine-condition UCB1Decoupled memory grid.

No GCP calls, VM creation, deletion, or changes to frozen experiment artifacts.
Place this file at analysis/profiling/gcp_memory_prepare.py in Serverledge.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

CONFIGS = (
    ('N08', 'none', None, None, 0.8),
    ('D08', 'difference', 0.25, 1.0, 0.8),
    ('R08', 'ratio', 0.25, 1.0, 0.8),
    ('N02', 'none', None, None, 0.2),
    ('D02', 'difference', 0.05, 64.0, 0.2),
    ('R02', 'ratio', 0.05, 64.0, 0.2),
    ('N04', 'none', None, None, 0.4),
    ('D04', 'difference', 0.01, 64.0, 0.4),
    ('R04', 'ratio', 0.01, 64.0, 0.4),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_catalog(root: Path) -> dict:
    path = root / 'scripts/gcp_scripts/functions_catalog.sh'
    if not path.is_file():
        raise FileNotFoundError(path)
    pattern = re.compile(r'^\s*"([^"|]+)\|([^"|]+)\|([^"|]+)\|([0-9]+)\|([^"]*)"\s*$')
    functions = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = pattern.match(line)
        if m:
            name, runtime, src, memory, handler = m.groups()
            if name in functions:
                raise ValueError(f'catalog duplicate: {name}')
            functions[name] = {'runtime': runtime, 'src': src, 'memory_mb': int(memory), 'handler': handler}
    if not functions:
        raise ValueError('catalogo vuoto o formato inatteso')
    return functions


def read_ground_truth(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError('ground truth CSV senza header')
        name_col = next((x for x in ('function_name', 'function', 'name') if x in reader.fieldnames), None)
        if name_col is None:
            raise ValueError(f'ground truth: impossibile individuare colonna funzione in {reader.fieldnames}')
        result = {}
        for row in reader:
            name = row[name_col]
            if name in result:
                raise ValueError(f'ground truth duplicata: {name}')
            result[name] = row
        return result


def inspect(root: Path, ground_truth: Path) -> dict:
    catalog = read_catalog(root)
    gt = read_ground_truth(ground_truth)
    evidence_base = root / 'data/profiling/gcp-ucb1-final/replicas'
    rows = []
    for fn, details in sorted(catalog.items()):
        paths = sorted(evidence_base.glob(f'{fn}-*/evidence/evidence.json')) if evidence_base.is_dir() else []
        rows.append({'function_name': fn, **details, 'in_ground_truth': fn in gt,
                     'evidence_paths': [str(p.relative_to(root)) for p in paths],
                     'evidence_count': len(paths)})
    print(f'Catalogo: {len(catalog)} funzioni')
    print(f'Ground truth: {len(gt)} funzioni (lettura soltanto; nessuna etichetta inventata)')
    print(f'Funzioni con evidence.json: {sum(bool(r["evidence_count"]) for r in rows)}')
    for row in rows:
        if row['evidence_count']:
            print(f'  READY_CANDIDATE {row["function_name"]}: {row["evidence_paths"]}')
    print('Le evidence presenti non sono automaticamente prior validati per il runtime di misura.')
    return {'catalog': len(catalog), 'ground_truth_rows': len(gt), 'functions': rows}


def materialize(root: Path, target: str, replica: str, dest: Path) -> dict:
    # Import from the user's actual repository, not a reimplementation of the formula.
    from analysis.profiling.transfer_materialized_tuned import tuned_bundle
    catalog = read_catalog(root)
    if target not in catalog:
        raise ValueError(f'target non presente nel catalogo: {target}')
    evidence_dir = root / 'data/profiling/gcp-ucb1-final/replicas' / f'{target}-{replica}' / 'evidence'
    evidence_path = evidence_dir / 'evidence.json'
    source = evidence_dir / 'source'
    evidence = read_json(evidence_path)
    source_manifest = read_json(source / 'manifest.json')
    if evidence.get('target_function') != target or source_manifest.get('target_function') != target:
        raise ValueError('target mismatch in donor evidence/source manifest')
    donor = source_manifest.get('donor_function')
    if not donor or donor == target:
        raise ValueError('donor mancante oppure identico al target')
    donor_info = evidence.get('donor') or {}
    donor_ref = float(donor_info['amd64']['mean_reward'])
    donor_other = float(donor_info['arm64']['mean_reward'])
    bootstrap = read_json(source / 'bootstrap/bootstrap.json')
    if bootstrap.get('target_function') != target:
        raise ValueError('bootstrap target mismatch')
    if bootstrap.get('target_profile', {}).get('arm_ground_truth_used') is not False:
        raise ValueError('bootstrap: uso ARM ground truth del target non escluso')
    if source_manifest.get('target_real_feedback_before_export') not in (None, 0):
        raise ValueError('prior materializzato dopo feedback reale del target: leakage')
    source_prior_sha = sha(source / 'frozen-prior.json')
    if dest.exists():
        raise FileExistsError(f'directory di destinazione già esistente: {dest}')
    dest.mkdir(parents=True, exist_ok=False)
    conditions = []
    for config_id, mode, wr, we, c in CONFIGS:
        cfg = {'config_id': config_id, 'policy': 'UCB1Decoupled', 'formula': mode,
               'c': c, 'w_R': wr, 'w_E': we, 'prior_applied': mode != 'none',
               'target': target, 'donor': donor, 'source_prior_sha256': source_prior_sha}
        if mode != 'none':
            out = dest / config_id
            tuned_bundle(source, out, mode=mode, reward_weight=wr,
                         exploration_weight=we, donor_reward_ref=donor_ref,
                         donor_reward_other=donor_other)
            # The old materializer defaults target_c based on formula; here c is a
            # frozen independent experiment factor, so update ONLY metadata.
            manifest_path = out / 'manifest.json'
            manifest = read_json(manifest_path)
            manifest['target_c'] = c
            manifest['memory_grid_config_id'] = config_id
            manifest['memory_grid_runtime_c'] = c
            write_json(manifest_path, manifest)
            prior = read_json(out / 'frozen-prior.json')
            assert prior['policy'] == 'UCB1Decoupled'
            assert prior['config']['ucb1_reference_anchor']['mode'] == mode
            assert manifest['prior_sha256'] == sha(out / 'frozen-prior.json')
            for arm in ('amd64', 'arm64'):
                assert abs(prior['arms'][arm]['ucb1']['observation_weight'] - wr) < 1e-12
                assert abs(prior['arms'][arm]['ucb1']['exploration_observation_weight'] - we) < 1e-12
            cfg['prior_sha256'] = manifest['prior_sha256']
            cfg['prior_request'] = str((out / 'materialized-request.json').relative_to(dest))
        conditions.append(cfg)
    out_manifest = {'schema_version': 1, 'experiment': 'gcp-memory',
                    'target': target, 'replica': replica, 'donor': donor,
                    'catalog_function': catalog[target], 'source_evidence': str(evidence_path),
                    'source_evidence_sha256': sha(evidence_path),
                    'source_prior_sha256': source_prior_sha,
                    'bootstrap_commit': bootstrap.get('serverledge_commit'),
                    'measurement_commit_required': 'TO_BE_VALIDATED_SEPARATELY',
                    'conditions': conditions, 'status': 'OFFLINE_PREPARED_NOT_GCP_VALIDATED'}
    write_json(dest / 'grid-manifest.json', out_manifest)
    print(f'PASS: 9 condizioni create, 6 prior materializzati, 3 senza prior: {dest}')
    return out_manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('inspect', help='Elenca funzioni nel catalogo, ground truth e evidence disponibili')
    i.add_argument('--repo-root', type=Path, default=repo_root())
    i.add_argument('--ground-truth', type=Path)
    i.add_argument('--output-json', type=Path)
    m = sub.add_parser('materialize', help='Materializza 6 prior della griglia, senza GCP')
    m.add_argument('--repo-root', type=Path, default=repo_root())
    m.add_argument('--target', required=True)
    m.add_argument('--replica', default='r01')
    m.add_argument('--out', type=Path)
    args = p.parse_args()
    root = args.repo_root.resolve()
    if args.command == 'inspect':
        gt = args.ground_truth or root / 'data/profiling/gcp-paper-rf-schema6/analysis/paper-rf-classifier/paper_ground_truth.csv'
        result = inspect(root, gt.resolve())
        if args.output_json:
            write_json(args.output_json.resolve(), result)
    else:
        dest = args.out or root / 'data/profiling/gcp-memory/priors' / f'{args.target}-{args.replica}'
        materialize(root, args.target, args.replica, dest.resolve())


if __name__ == '__main__':
    main()
