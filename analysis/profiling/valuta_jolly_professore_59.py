#!/usr/bin/env python3
"""Three-class Chen + unchanged binary Serverledge: professor's neutral-as-jolly evaluation.

Uses only ALREADY-FROZEN predictions, without training, retuning or GCP.
The neutral *ground-truth* class accepts either architecture; a predicted
neutral on a truly directional target is NOT an automatic success.
Ternary (exact-label) classification accuracy is meaningful only for the RF.

Run from Serverledge repository root:
    python3 analysis/profiling/valuta_jolly_professore_59.py self-test
    python3 analysis/profiling/valuta_jolly_professore_59.py check
    python3 analysis/profiling/valuta_jolly_professore_59.py run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path('data/architecture-comparison/comparison/frozen-rf-vs-serverledge-59-v1')
OUT = Path('data/architecture-comparison/comparison/professor-ternary-jolly-v1')
METHODS = ('paper_RF_15_original', 'paper_RF_2p5_relabelled',
           'majority_diagnostic', 'weighted_frozen')
SEEDS = (11, 23, 37, 41, 53)
CLASS_MAP = {
    'ARM-faster': 'arm-preferred',
    'ARM-slower': 'x86-preferred',
    'ARM-similar': 'architecture-independent',
}
DIRECTIONS = {'arm-preferred', 'x86-preferred'}
NEUTRAL = 'architecture-independent'
ABSTAIN = 'ABSTAIN'
N = 59


def ensure(flag: bool, message: str) -> None:
    if not flag:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mpath = SOURCE / 'manifest.json'
    ensure(mpath.is_file(), f'Missing {mpath}')
    manifest = json.loads(mpath.read_text(encoding='utf-8'))

    def checked(name: str) -> pd.DataFrame:
        path = SOURCE / name
        rec = manifest.get('outputs', {}).get(name)
        ensure(path.is_file() and rec is not None, f'Missing SHA-verified {path}')
        ensure(sha(path) == rec['sha256'], f'Changed source file: {path}')
        z = pd.read_csv(path)
        ensure(len(z) == rec['rows'], f'Incorrect number of rows: {path}')
        return z

    cohort = checked('function_comparison_summary_59.csv')
    decisions = checked('direct_decisions_all_methods.csv')
    expected = {'function_name', 'paper_class_15', 'serverledge_preference_2p5'}
    ensure(expected <= set(cohort.columns), f'Missing truth fields: {expected - set(cohort.columns)}')
    needed = {'function_name', 'seed', 'method', 'predicted_preference'}
    ensure(needed <= set(decisions.columns), f'Missing predictions: {needed - set(decisions.columns)}')
    ensure(len(cohort) == N and cohort.function_name.is_unique, 'Expected 59 distinct targets')
    ensure(len(decisions) == 2*N + 2*N*len(SEEDS), 'Expected 708 frozen predictions')
    ensure(set(decisions.method) == set(METHODS), 'Unexpected method names')
    ensure(set(decisions.function_name) == set(cohort.function_name), 'Target mismatch')

    cohort['truth_15'] = cohort.paper_class_15.map(CLASS_MAP)
    cohort['truth_2p5'] = cohort.serverledge_preference_2p5
    for c in ('truth_15', 'truth_2p5'):
        ensure(cohort[c].isin(DIRECTIONS | {NEUTRAL}).all(), f'Unexpected truth in {c}')
    truth15 = cohort.truth_15.value_counts().to_dict()
    truth25 = cohort.truth_2p5.value_counts().to_dict()
    ensure(truth15 == {'architecture-independent': 40, 'arm-preferred': 11, 'x86-preferred': 8},
           f'Changed 15% truth labels: {truth15}')
    ensure(truth25 == {'arm-preferred': 26, 'x86-preferred': 24, 'architecture-independent': 9},
           f'Changed 2.5% truth labels: {truth25}')
    return cohort, decisions, manifest


def compatible(truth: str, prediction: str) -> int:
    """Set-valued, nonvacuous validity: predicted choices ⊆ acceptable true choices.

    True neutral accepts both; predicted neutral proposes both. Therefore
    a predicted neutral is NOT valid for a directional target.
    ABSTAIN is scored as failure (not vacuously a correct empty set).
    """
    if prediction == ABSTAIN:
        return 0
    if truth == NEUTRAL:
        return int(prediction in DIRECTIONS | {NEUTRAL})
    ensure(truth in DIRECTIONS, f'Invalid target truth {truth}')
    return int(prediction == truth)


def get_seed_frame(decisions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rf_mask = decisions.method.isin(METHODS[:2])
    sl_mask = decisions.method.isin(METHODS[2:]) & (decisions.seed.astype(str) == str(seed))
    z = decisions.loc[rf_mask | sl_mask, ['function_name', 'method', 'predicted_preference']].copy()
    ensure(len(z) == N*len(METHODS), f'Seed {seed}: wrong row count')
    ensure(not z.duplicated(['method', 'function_name']).any(), f'Seed {seed}: duplicated predictions')
    return z


def score_all(cohort: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    for seed in SEEDS:
        z = get_seed_frame(decisions, seed).merge(
            cohort[['function_name', 'truth_15', 'truth_2p5']],
            on='function_name', validate='many_to_one')
        z['seed'] = seed
        for tau, gt in ((15.0, 'truth_15'), (2.5, 'truth_2p5')):
            q = z.copy()
            q['threshold_percent_EVAL_ONLY'] = tau
            q['true_preference_EVAL_ONLY'] = q[gt]
            q['model_can_choose_architecture'] = q.predicted_preference.isin(DIRECTIONS)
            q['predicted_neutral'] = q.predicted_preference.eq(NEUTRAL)
            q['abstained'] = q.predicted_preference.eq(ABSTAIN)
            ensure((q.model_can_choose_architecture | q.predicted_neutral | q.abstained).all(),
                   'Unrecognized prediction')
            q['truth_is_neutral_EVAL_ONLY'] = q[gt].eq(NEUTRAL)
            q['jolly_compatible_all59_EVAL_ONLY'] = [
                compatible(t, p) for t, p in zip(q[gt], q.predicted_preference)
            ]
            q['directional_truth_correct_all_EVAL_ONLY'] = np.where(
                ~q.truth_is_neutral_EVAL_ONLY,
                q['jolly_compatible_all59_EVAL_ONLY'], np.nan)
            # Strict 3-way categorical accuracy: defined for RF, NOT for binary Serverledge.
            q['rf_exact_ternary_correct_EVAL_ONLY'] = np.where(
                q.method.isin(METHODS[:2]),
                q.predicted_preference.eq(q[gt]).astype(float), np.nan)
            detail.append(q[['function_name', 'method', 'seed', 'threshold_percent_EVAL_ONLY',
                'true_preference_EVAL_ONLY', 'predicted_preference',
                'truth_is_neutral_EVAL_ONLY', 'model_can_choose_architecture',
                'predicted_neutral', 'abstained', 'jolly_compatible_all59_EVAL_ONLY',
                'directional_truth_correct_all_EVAL_ONLY', 'rf_exact_ternary_correct_EVAL_ONLY']])
    detail = pd.concat(detail, ignore_index=True)
    rows = []
    for (tau, seed, method), q in detail.groupby(
            ['threshold_percent_EVAL_ONLY', 'seed', 'method'], sort=True):
        directional = q[~q.truth_is_neutral_EVAL_ONLY]
        covered_directional = directional[directional.model_can_choose_architecture]
        rows.append({
            'threshold_percent': tau, 'seed': seed, 'method': method,
            'n_total': len(q), 'n_true_neutral': int(q.truth_is_neutral_EVAL_ONLY.sum()),
            'n_true_directional': len(directional),
            'n_architecture_decisions': int(q.model_can_choose_architecture.sum()),
            'n_predicted_neutral': int(q.predicted_neutral.sum()),
            'n_abstained': int(q.abstained.sum()),
            'architecture_decision_coverage': float(q.model_can_choose_architecture.mean()),
            'jolly_compatibility_fraction_all59': float(q.jolly_compatible_all59_EVAL_ONLY.mean()),
            'directional_truth_accuracy_with_abstentions_as_failure': float(
                directional.jolly_compatible_all59_EVAL_ONLY.mean()),
            'directional_truth_decision_coverage': float(
                directional.model_can_choose_architecture.mean()),
            'directional_truth_accuracy_conditional_on_action': (
                float(covered_directional.jolly_compatible_all59_EVAL_ONLY.mean())
                if len(covered_directional) else np.nan),
            'rf_exact_ternary_class_accuracy': (
                float(q.rf_exact_ternary_correct_EVAL_ONLY.mean())
                if method in METHODS[:2] else np.nan),
        })
    by_seed = pd.DataFrame(rows)
    ensure(len(by_seed) == 2*len(SEEDS)*len(METHODS), 'Invalid summary shape')
    return detail, by_seed


def summarize(by_seed: pd.DataFrame) -> pd.DataFrame:
    nums = [c for c in by_seed.columns if c not in {'method', 'threshold_percent', 'seed'}]
    return by_seed.groupby(['threshold_percent','method'], as_index=False, sort=False)[nums].mean()


def self_test() -> None:
    expect = [
        (NEUTRAL, 'arm-preferred', 1), (NEUTRAL, 'x86-preferred', 1),
        (NEUTRAL, NEUTRAL, 1), (NEUTRAL, ABSTAIN, 0),
        ('arm-preferred', 'arm-preferred', 1),
        ('arm-preferred', 'x86-preferred', 0),
        ('arm-preferred', NEUTRAL, 0), ('arm-preferred', ABSTAIN, 0),
        ('x86-preferred', 'x86-preferred', 1),
        ('x86-preferred', 'arm-preferred', 0),
        ('x86-preferred', NEUTRAL, 0),
    ]
    for truth, p, correct in expect:
        ensure(compatible(truth, p) == correct, f'Wrong jolly semantics {truth}/{p}')
    print('PASS self-test: true neutral accepts either arm; predicted neutral does not excuse directional errors; abstention fails')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['self-test', 'check', 'run'])
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
        return
    cohort, decisions, manifest = load_inputs()
    detail, by_seed = score_all(cohort, decisions)
    summary = summarize(by_seed)
    native = summary[((summary.threshold_percent == 15.0) &
             (summary.method == 'paper_RF_15_original')) |
            ((summary.threshold_percent == 2.5) &
             (summary.method == 'paper_RF_2p5_relabelled'))]
    ensure(np.isclose(float(native.loc[native.threshold_percent == 15.0,
            'rf_exact_ternary_class_accuracy'].iloc[0]), 30/59), 'Frozen RF 15% accuracy changed')
    ensure(np.isclose(float(native.loc[native.threshold_percent == 2.5,
            'rf_exact_ternary_class_accuracy'].iloc[0]), 24/59), 'Frozen RF 2.5% accuracy changed')
    print('PASS frozen inputs; 59 functions x 5 seeds, two evaluation thresholds; no refit')
    print('\n===== PROFESSOR JOLLY: ALL 59 TARGETS =====')
    print(summary[['threshold_percent', 'method', 'architecture_decision_coverage',
        'jolly_compatibility_fraction_all59', 'directional_truth_accuracy_with_abstentions_as_failure',
        'rf_exact_ternary_class_accuracy']].to_string(index=False, float_format=lambda v:f'{v:.5f}'))
    print('\nNOTE: tau=15% has 40/59 true-neutral; always-ARM compatibility baseline = 51/59; always-x86 = 48/59.')
    print('NOTE: tau=2.5% has 9/59 true-neutral; always-ARM = 35/59; always-x86 = 33/59.')
    print('NOTE: RF class-neutral output is a valid ternary prediction, but DOES NOT choose an execution arm.')
    if args.command == 'check':
        print('PASS CHECK read-only; no output')
        return
    ensure(not OUT.exists(), f'Refusing to overwrite existing output {OUT}')
    OUT.mkdir(parents=True)
    output = {
        'jolly_per_function_seed.csv': detail,
        'jolly_per_seed.csv': by_seed,
        'jolly_summary_two_thresholds.csv': summary,
    }
    out_hashes = {}
    for name, df in output.items():
        dest = OUT / name
        df.to_csv(dest, index=False)
        out_hashes[name] = {'sha256':sha(dest),'rows':len(df)}
    source_files = ('manifest.json','direct_decisions_all_methods.csv','function_comparison_summary_59.csv')
    audit = {
        'research_question': 'Frozen ternary RF vs unchanged binary Serverledge; professor true-neutral-as-jolly evaluation',
        'no_training_no_tuning_no_simulation':True,
        'evaluation_thresholds':[15.0,2.5],
        'primary_comparison_pairs':{
            'tau_15':['paper_RF_15_original','majority_diagnostic','weighted_frozen'],
            'tau_2p5':['paper_RF_2p5_relabelled','majority_diagnostic','weighted_frozen']},
        'true_neutral_rule': 'either actual architecture considered acceptable',
        'predicted_neutral_rule': 'correct if true neutral; not sufficient if target truly directional; no routed arm',
        'abstention_rule':'no action => not counted as success in all-function compatibility',
        'rf_ternary_accuracy':'only meaningful for Random Forest; not imputed for binary Serverledge',
        'seeds':list(SEEDS),
        'inputs': {p:{'sha256':sha(SOURCE / p)} for p in source_files},
        'outputs':out_hashes,
    }
    (OUT / 'manifest.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report = '''# Professor protocol — three-class Chen and binary Serverledge with neutral-as-jolly

- No models were trained or modified. All source file SHA hashes were checked.
- RF 15% (original label task), RF 2.5% (controlled relabelled task); both remain ternary.
- Serverledge Majority (diagnostic) and Weighted (frozen) remain binary on their internal 2.5% donor labels.
- **Truth neutral:** either ARM64 or x86 is an acceptable execution architecture at the indicated threshold.
- **Prediction neutral:** genuine three-class output from RF. Correct as a three-class label only when true neutral. Does not magically count as correct on directionally labelled targets, and it is not an actual execution-arm choice.
- **Serverledge abstention:** is distinct from an RF neutral prediction; abstention is NOT scored as a success.
- RF strict exact-label ternary accuracy is shown *separately*. Serverledge cannot receive a genuine ternary exact-label accuracy without changing its prediction policy.
- Jolly compatibility across **all 59** counts any non-abstaining output compatible with ground-truth allowed arch(es). For true directional targets, only an explicit matching direction succeeds. This is **not** the original paper's ternary classification accuracy.
- **Critical baseline:** at tau=15%, always choosing ARM scores (40+11)/59=86.44% under jolly compatibility; always x86 scores (40+8)/59=81.36%, even without any useful feature prediction. Interpret jolly compatibility alongside directional-truth accuracy and action coverage.
- At tau=2.5%, always ARM scores 35/59=59.32%; always x86 33/59=55.93%.
- Per-seed results are repeated clustering initialization checks, not 5 independent cohorts.
- For primary paired reporting compare RF15 vs both Serverledge rules on tau15; RF2.5 vs both Serverledge rules on tau2.5. Do not silently conflate the tasks.

## Summary (ALL METHODS x BOTH EVALUATION THRESHOLDS)

'''
    report += summary[['threshold_percent','method','architecture_decision_coverage',
        'jolly_compatibility_fraction_all59','directional_truth_accuracy_with_abstentions_as_failure',
        'rf_exact_ternary_class_accuracy']].to_markdown(index=False,floatfmt='.4f') + '\n'
    (OUT / 'PROFESSOR_JOLLY_REPORT.md').write_text(report,encoding='utf-8')
    print('PASS RUN outputs at:',OUT)


if __name__ == '__main__':
    main()
