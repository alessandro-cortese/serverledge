#!/usr/bin/env python3
"""Audit and (explicitly) materialize six GCP-memory prior bundles from frozen RF observations.

Read-only default: audit.  --build creates ONLY new gcp-memory evidence and grid
folders.  Never changes historical gcp-ucb1-final, paper RF datasets, or GCP.

Feature semantics of pinned Serverledge schema-6 exporter:
  userRuntime == sample.timing.duration_ms (in milliseconds)
  runtime == frameworkRuntime + userRuntime + frameworkRuntimeDeltas
Reward: mean(-ln(userRuntime)), not -ln(mean(userRuntime)).

Source RF provenance is archived warm profiling (NOT an interchangeable direct
GCP 10+10 donor measurement). This distinction is recorded in every output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from analysis.profiling.gcp_memory_prepare import CONFIGS, read_catalog, sha, write_json

PIN = '9f9e9cfc5631b41f525f6d91fadad4415b93a8c1'
TARGETS = (
    'graph-pagerank-node', 'amd_faster', 'filehandle',
    'compression-node', 'arm_faster', 'functionbench-linpack',
)
ARMS = ('amd64', 'arm64')
SAMPLES = 10  # fixed donor evidence budget used by the original UCB1 experiments
SOURCE_NOTE = 'rf_schema6_warm_userRuntime_ms_stratified_10_per_arm_NOT_direct_10_plus_10'


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def need(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def num(v, where: str) -> float:
    try:
        x = float(v)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{where}: invalid number {v!r}') from e
    need(math.isfinite(x), f'{where}: nonfinite number')
    return x


def path_for(base: Path, override: Path | None, suffix: str) -> Path:
    return (override or (base / suffix)).resolve()


def select_ten(items: list[dict], label: str) -> list[dict]:
    """Deterministic, round-ordered, equispaced archive selection (no RNG)."""
    need(len(items) == 100, f'{label}: expected 100 RF samples, got {len(items)}')
    ordered = sorted(items, key=lambda x: (
        int(x['proof']['round']), x['proof']['block_id'], x['request_id']))
    indices = [((2 * i + 1) * len(ordered)) // (2 * SAMPLES)
               for i in range(SAMPLES)]
    chosen = [ordered[i] for i in indices]
    need(len({v['request_id'] for v in chosen}) == SAMPLES, f'{label}: sample duplicate')
    return chosen


def audit(root: Path, dataset: Path, proof: Path, manifest: Path,
          selections: Path) -> dict:
    for p in (dataset, proof, manifest):
        need(p.is_file(), f'missing: {p}')
    doc = load(manifest)
    need(doc.get('pin') == PIN, f'RF dataset pin mismatch: {doc.get("pin")}')
    need(sha(dataset) == doc.get('dataset_sha256'), 'RF dataset SHA mismatch')
    need(sha(proof) == doc.get('provenance_sha256'), 'RF provenance SHA mismatch')
    catalog = read_catalog(root)
    chosen_donors = {}
    selection_files = {}
    for target in TARGETS:
        selpath = selections / target / 'selection.json'
        need(selpath.is_file(), f'selection missing: {selpath}')
        s = load(selpath)
        need(s.get('status') == 'selected', f'{target}: status is not selected')
        need(s.get('target_function') == target, f'{target}: target mismatch in selection')
        donor = (s.get('selected_donor') or {}).get('function_name')
        need(donor in catalog and donor != target, f'{target}: donor invalid: {donor}')
        chosen_donors[target] = donor
        selection_files[target] = {'path': str(selpath), 'sha256': sha(selpath),
                                   'prediction': s.get('prediction'),
                                   'candidate_count': s.get('candidate_count'),
                                   'target_cluster': s.get('target_cluster'),
                                   'selected_distance': (s.get('selected_donor') or {}).get('distance')}
    desired = set(TARGETS) | set(chosen_donors.values())
    lookup = defaultdict(list)
    proof_lookup = {}
    for r in rows(proof):
        request_id = (r.get('request_id') or '').strip()
        need(bool(request_id) and request_id not in proof_lookup,
             f'RF provenance duplicate or empty request_id: {request_id}')
        proof_lookup[request_id] = r
    total = 0
    ids = set()
    for r in rows(dataset):
        total += 1
        rid = (r.get('request_id') or '').strip()
        need(rid and rid not in ids, f'RF dataset duplicate/empty request_id: {rid}')
        ids.add(rid)
        name = r.get('function_name')
        arch = r.get('machine_tag')
        p = proof_lookup.get(rid)
        need(p is not None, f'dataset request_id missing provenance: {rid}')
        need(p['function'] == name and p['arch'] == arch,
             f'dataset/provenance mismatch: {rid}')
        if name not in desired:
            continue
        need(arch in ARMS, f'{name}/{rid}: invalid arm: {arch}')
        need(num(r.get('configured_cpus'), f'{name}/{rid} CPU') == 1,
             f'{name}/{rid}: CPU != 1')
        need(int(r['configured_memory_mb']) == catalog[name]['memory_mb'],
             f'{name}/{rid}: memory differs from frozen catalog')
        need(r.get('paper_rf_csv_schema_version') == '1', f'{name}/{rid}: RF CSV schema != 1')
        need(r.get('paper_rf_feature_vector_schema_version') == '1',
             f'{name}/{rid}: RF features schema != 1')
        d = num(r.get('userRuntime'), f'{name}/{rid} userRuntime')
        need(d > 0, f'{name}/{rid}: nonpositive userRuntime')
        fw = num(r.get('frameworkRuntime'), f'{name}/{rid} frameworkRuntime')
        extra = num(r.get('frameworkRuntimeDeltas'), f'{name}/{rid} frameworkRuntimeDeltas')
        rt = num(r.get('runtime'), f'{name}/{rid} runtime')
        need(fw >= 0 and extra >= 0, f'{name}/{rid}: negative profiling overhead')
        need(math.isclose(rt, fw+d+extra, rel_tol=1e-8, abs_tol=1e-5),
             f'{name}/{rid}: runtime != frameworkRuntime+userRuntime+frameworkRuntimeDeltas')
        lookup[(name, arch)].append({'request_id': rid, 'duration_ms': d,
                                    'reward': -math.log(d), 'proof': p})
    need(total == doc.get('rows'), f'RF dataset rows={total} disagree with manifest={doc.get("rows")}')
    need(ids == set(proof_lookup), 'RF dataset and provenance request_id sets differ')
    for fn in sorted(desired):
        for arch in ARMS:
            need(len(lookup[(fn, arch)]) == 100,
                 f'{fn}/{arch}: expected 100 warm RF samples, got {len(lookup[(fn, arch)])}')
    samples = {}
    for target in TARGETS:
        donor = chosen_donors[target]
        # Only reference x86 is accessed for target reward. Target ARM must NOT be an input.
        target_x86 = select_ten(lookup[(target, 'amd64')], target + '/target-amd64')
        donor_x86 = select_ten(lookup[(donor, 'amd64')], target + '/donor-amd64')
        donor_arm = select_ten(lookup[(donor, 'arm64')], target + '/donor-arm64')
        samples[target] = {
            'target_anchor': sum(v['reward'] for v in target_x86)/SAMPLES,
            'donor_amd64_mean_reward': sum(v['reward'] for v in donor_x86)/SAMPLES,
            'donor_arm64_mean_reward': sum(v['reward'] for v in donor_arm)/SAMPLES,
            'selection': selection_files[target], 'donor': donor,
            'samples': {'target_x86': target_x86, 'donor_x86': donor_x86, 'donor_arm': donor_arm},
        }
        need(abs(samples[target]['donor_amd64_mean_reward']) > 1e-12,
             f'{target}: donor x86 mean reward too close to 0 for Ratio')
    print('PASS: RF dataset/provenance SHA, PIN, request IDs, metadata, reward semantics')
    print('Source: userRuntime=DurationMs, runtime includes profiling overhead')
    print('Sampling: 10 equispaced warm RF samples per group, across archive rounds')
    print('Note: RF archive samples are NOT direct 10+10 donor measurements')
    for target in TARGETS:
        record = samples[target]
        meta = record['selection']
        print(f'{target:23s} donor={record["donor"]:22s} '
              f'distance={float(meta["selected_distance"]):.5f} '
              f'anchor={record["target_anchor"]:.6f} '
              f'donor_ref={record["donor_amd64_mean_reward"]:.6f} '
              f'donor_ARM={record["donor_arm64_mean_reward"]:.6f}')
        if float(meta['selected_distance']) > 7.0 or int(meta.get('candidate_count') or 0) < 2:
            print(f'  WARNING: LOW_CONFIDENCE_DONOR for {target}; do not discard or hide')
    return {'samples': samples, 'catalog': catalog, 'dataset': dataset,
            'provenance': proof, 'manifest': manifest,
            'manifest_sha256': sha(manifest), 'dataset_sha256': sha(dataset),
            'provenance_sha256': sha(proof)}


def build(result: dict, root: Path, evidence_root: Path, priors_root: Path) -> None:
    from analysis.profiling.transfer_materialized_prior import write_json as original_json
    from analysis.profiling.transfer_materialized_tuned import tuned_bundle

    for target in TARGETS:
        need(not (evidence_root / (target + '-r01')).exists(),
             f'evidence destination exists for {target}; refusing overwrite')
        need(not (priors_root / (target + '-r01')).exists(),
             f'priors destination exists for {target}; refusing overwrite')
    for target in TARGETS:
        sample = result['samples'][target]
        donor = sample['donor']
        anchor = sample['target_anchor']
        ref = sample['donor_amd64_mean_reward']
        other = sample['donor_arm64_mean_reward']
        gap = other - ref
        evidence_dir = evidence_root / (target + '-r01')
        source = evidence_dir / 'source'
        (source / 'bootstrap').mkdir(parents=True, exist_ok=False)
        boot = {
            'schema_version': 1, 'experiment': 'gcp-memory-rf-archive-reuse',
            'replica_id': target + '-r01', 'serverledge_commit': PIN,
            'target_function': target,
            'target_profile': {
                'reference_arm': 'amd64', 'sample_count': SAMPLES,
                'target_reference_mean_reward': anchor,
                'aggregation': 'mean(-ln(userRuntime_ms))',
                'reward_definition': '-ln(duration_ms)',
                'arm_ground_truth_used': False,
            },
            'donor_selection': {
                'selected_donor': donor,
                'selection_artifact': sample['selection']['path'],
                'selection_artifact_sha256': sample['selection']['sha256'],
                'prediction': sample['selection']['prediction'],
                'target_cluster': sample['selection']['target_cluster'],
            },
            'experimental_block': {
                'policy': 'UCB1Decoupled',
                'condition_ids': [v[0] for v in CONFIGS],
            },
            'rf_archive_source': SOURCE_NOTE,
            'target_ARM_measurements_used': False,
        }
        historical_root = root / 'data/profiling/final-corpus64-20261005-analysis-01'
        boot['donor_selection']['historical_x86_profile'] = (
            str(historical_root / 'resource/x86/function-profiles-median.csv'))
        boot['donor_selection']['historical_static_metrics'] = (
            str(historical_root / 'static-code-metrics.csv'))
        boot['donor_selection']['note'] = (
            'Frozen donor geometry uses older x86 profiles; archived RF userRuntime supplies reward anchor')
        original_json(source / 'bootstrap/bootstrap.json', boot)
        raw = sample['samples']
        for key, observations in raw.items():
            original_json(evidence_dir / f'{key}-selected.json', [
                {'request_id': v['request_id'], 'duration_ms': v['duration_ms'],
                 'reward': v['reward'], 'round': int(v['proof']['round']),
                 'block_id': v['proof']['block_id']} for v in observations])
        prior = {
            'schema_version': 1, 'policy': 'UCB1', 'has_prior': True,
            'donor_function_name': donor, 'source_real_observation_count': 2*SAMPLES,
            'source_excluded_synthetic_observation_count': 0,
            'arm_count': 2, 'transferred_arm_count': 2, 'skipped_arm_count': 0,
            'config': {
                'equivalent_observation_weight': 0.25,
                'min_real_observations_per_arm': SAMPLES,
                'ucb1_reference_anchor': {
                    'enabled': True, 'reference_arm': 'amd64',
                    'target_reference_mean_reward': anchor, 'mode': 'difference',
                },
            },
            'arms': {},
        }
        for arch, mean in (('amd64', anchor), ('arm64', anchor + gap)):
            prior['arms'][arch] = {
                'transferred': True, 'source_real_observation_count': SAMPLES,
                'source_excluded_synthetic_observation_count': 0,
                'applied_equivalent_observation_weight': 0.25,
                'applied_exploration_observation_weight': 0.25,
                'attenuation_scale': 0.25/SAMPLES,
                'ucb1': {
                    'observation_weight': 0.25, 'exploration_observation_weight': 0.25,
                    'mean_reward': mean, 'reward_sum': mean*0.25,
                },
            }
        original_json(source / 'frozen-prior.json', prior, compact=True)
        source_sha = sha(source / 'frozen-prior.json')
        (source / 'prior.sha256').write_text(source_sha + '  frozen-prior.json\n', encoding='utf-8')
        original_json(source / 'manifest.json', {
            'schema_version': 1, 'status': 'rf-archive-source',
            'target_function': target, 'donor_function': donor,
            'prior_sha256': source_sha,
            'donor_observations': {'amd64': SAMPLES, 'arm64': SAMPLES},
            'donor_reward_ref': ref, 'donor_reward_other': other,
            'donor_reward_gap': gap,
            'target_real_feedback_before_export': 0,
            'rf_source_dataset_sha256': result['dataset_sha256'],
            'rf_source_provenance_sha256': result['provenance_sha256'],
            'rf_source_manifest_sha256': result['manifest_sha256'],
            'sample_policy': SOURCE_NOTE,
        })
        original_json(evidence_dir / 'evidence.json', {
            'schema_version': 1, 'status': 'ready_from_archived_rf_data',
            'target_function': target, 'donor_function': donor,
            'target_reference_mean_reward': anchor,
            'observations': {'amd64': SAMPLES, 'arm64': SAMPLES},
            'reward_definition': '-ln(userRuntime_ms)',
            'donor': {
                'amd64': {'mean_reward': ref}, 'arm64': {'mean_reward': other},
                'reward_gap_arm64_minus_amd64': gap,
            },
            'provenance': SOURCE_NOTE,
        })
        # Reuse the ORIGINAL formula materializer, never rewrite Difference/Ratio.
        priors = priors_root / (target + '-r01')
        need(not priors.exists(), f'Unexpected prior folder for {target}')
        priors.mkdir(parents=True, exist_ok=False)
        grid = []
        for cid, mode, wr, we, c in CONFIGS:
            entry = {'config_id': cid, 'policy': 'UCB1Decoupled', 'formula': mode,
                     'c': c, 'w_R': wr, 'w_E': we, 'prior_applied': mode != 'none',
                     'target': target, 'donor': donor,
                     'source_prior_sha256': source_sha}
            if mode != 'none':
                out = priors / cid
                tuned_bundle(source, out, mode=mode, reward_weight=wr,
                             exploration_weight=we, donor_reward_ref=ref,
                             donor_reward_other=other)
                mpath = out / 'manifest.json'
                m = load(mpath)
                m.update(target_c=c, memory_grid_config_id=cid,
                         memory_grid_runtime_c=c)
                write_json(mpath, m)
                need(sha(out / 'frozen-prior.json') == m['prior_sha256'],
                     f'prior SHA mismatch {target}/{cid}')
                entry['prior_sha256'] = m['prior_sha256']
                entry['prior_request'] = f'{cid}/materialized-request.json'
            grid.append(entry)
        write_json(priors / 'grid-manifest.json', {
            'schema_version': 1, 'experiment': 'gcp-memory',
            'target': target, 'replica': 'r01', 'donor': donor,
            'catalog_function': result['catalog'][target],
            'source_evidence': str(evidence_dir / 'evidence.json'),
            'source_evidence_sha256': sha(evidence_dir / 'evidence.json'),
            'source_prior_sha256': source_sha,
            'bootstrap_commit': PIN,
            'measurement_commit_required': 'TO_BE_VALIDATED_SEPARATELY',
            'conditions': grid,
            'status': 'OFFLINE_PREPARED_NOT_GCP_VALIDATED',
            'provenance_method': SOURCE_NOTE,
            'selection_sha256': sample['selection']['sha256'],
            'donor_confidence_warning': (
                float(sample['selection']['selected_distance']) > 7 or
                int(sample['selection'].get('candidate_count') or 0) < 2),
        })
        print(f'PASS: {target}: 9 conditions / 6 prior, donor={donor}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('audit', 'build'))
    p.add_argument('--repo-root', type=Path,
                   default=Path(__file__).resolve().parents[2])
    p.add_argument('--dataset', type=Path)
    p.add_argument('--provenance', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--selection-dir', type=Path)
    p.add_argument('--evidence-out', type=Path)
    p.add_argument('--priors-out', type=Path)
    a = p.parse_args()
    root = a.repo_root.resolve()
    base = root / 'data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final'
    dataset = (a.dataset or (base/'paper-rf-dataset.csv')).resolve()
    provenance = (a.provenance or (base/'paper-rf-provenance.csv')).resolve()
    manifest = (a.manifest or (base/'dataset_manifest.json')).resolve()
    selections = (a.selection_dir or
                  (root/'data/profiling/gcp-memory/planning/donor_selection_r01')).resolve()
    result = audit(root, dataset, provenance, manifest, selections)
    if a.action == 'audit':
        print('AUDIT-ONLY: no files created, no prior changed, no GCP access')
        return
    evidence_root = (a.evidence_out or (root/'data/profiling/gcp-memory/evidence')).resolve()
    priors_root = (a.priors_out or (root/'data/profiling/gcp-memory/priors')).resolve()
    build(result, root, evidence_root, priors_root)
    print('OFFLINE RF BRIDGE: PASS. Prior are NOT verified by GCP runtime.')


if __name__ == '__main__':
    main()
