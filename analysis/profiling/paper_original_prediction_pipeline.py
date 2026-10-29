#!/usr/bin/env python3
"""Fixed-author-model, function-held-out reproduction on Serverledge GCP corpus.

Main methods (no tuning):
    sklearn RandomForestClassifier(random_state=42), input = author's 14 retained fields;
    sklearn RandomForestRegressor(n_estimators=1000, random_state=42),
    input = cpuUserDelta/cpuKernelDelta/cpuIdleDelta, output = ARM 'runtime'.

Evaluate all three predeclared paper training strategies: All-in-one,
Resource-bound (CPU kernel fraction >10%), ARM-speed (classifier-selected).
Per-function LOFO, with runtime-rank pairing *inside each training function*.
Never use the held-out target's ARM samples/true class for prediction.

Usage, from repository root:
    python3 analysis/profiling/paper_original_prediction_pipeline.py check
    python3 analysis/profiling/paper_original_prediction_pipeline.py run

Only 'run' writes new files, under data/architecture-comparison/paper-original/v1.
Run may be computationally demanding: it fits multiple 1000-tree RF regressors
per held-out function. Model parameters and evaluation cohort are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

DATASET_ROOT = Path('data/architecture-comparison/datasets/v1')
OUTPUT_ROOT = Path('data/architecture-comparison/paper-original/v1')
PREDICTORS_14 = [
    'totalMemory', 'freeMemory', 'frameworkRuntime', 'userRuntime',
    'cpuUserDelta', 'cpuKernelDelta', 'cpuIdleDelta', 'cpuIOWaitDelta',
    'cpuSoftIrqDelta', 'cpuStealDelta', 'pageFaultsDelta', 'utilizedCPUs',
    'frameworkRuntimeDeltas', 'runtime',
]
REGRESSORS_3 = ['cpuUserDelta', 'cpuKernelDelta', 'cpuIdleDelta']
CLASSES = ('ARM-faster', 'ARM-similar', 'ARM-slower')
STRATEGIES = ('all-in-one', 'resource-bound', 'arm-speed')
MODEL_CLASSIFIER = {'random_state': 42}
MODEL_REGRESSOR = {'n_estimators': 1000, 'random_state': 42}
N_FUNCTIONS = 59
N_SAMPLES_PER_ARCH = 100


def require(assertion: bool, message: str) -> None:
    if not assertion:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    p_profiling = root / 'profiling_common59_enriched.csv'
    p_truth = root / 'ground_truth_common59.csv'
    p_manifest = root / 'dataset_manifest.json'
    for p in (p_profiling, p_truth, p_manifest):
        require(p.is_file(), f'Missing input: {p}')
    manifest = json.loads(p_manifest.read_text(encoding='utf-8'))
    file_hashes = {item['file']: item['sha256'] for item in manifest['outputs']}
    for p in (p_profiling, p_truth):
        require(digest(p) == file_hashes.get(p.name), f'SHA256 differs from dataset manifest: {p}')
    # Verify immutable original-source hash provenance, not merely the derived file.
    native = manifest['source_files_sha256']['native_gcp']
    native_path = Path(native['path'])
    require(native_path.is_file() and digest(native_path) == native['sha256'],
            f'Original GCP source missing or changed: {native_path}')
    dataset = pd.read_csv(p_profiling, low_memory=False)
    truth = pd.read_csv(p_truth, low_memory=False)
    require(len(dataset) == 11800 and len(truth) == N_FUNCTIONS,
            f'Unexpected corpus dimensions: {len(dataset)} samples, {len(truth)} truth rows')
    require(not truth['function_name'].duplicated().any(), 'Duplicate truth function')
    require(set(dataset.function_name) == set(truth.function_name), 'Functions in truth and samples differ')
    require('eligible_common59' in dataset and (dataset.eligible_common59 == 1).all(),
            'Input contains samples excluded from the common 59')
    require('paper_class_15' in truth, 'Missing native paper ground truth class')
    require(set(truth.paper_class_15) <= set(CLASSES), 'Unknown paper class')
    require(set(PREDICTORS_14).issubset(dataset.columns), 'Missing author classifier features')
    require(set(REGRESSORS_3).issubset(dataset.columns), 'Missing author regression features')
    require(all(k in dataset for k in ('machine_tag', 'experiment_id', 'request_id')), 'Missing observation keys')
    require(not dataset.duplicated(['function_name', 'machine_tag', 'experiment_id', 'request_id']).any(),
            'Duplicate request ID within function and architecture')
    sample_counts = dataset.groupby(['function_name', 'machine_tag']).size()
    require(len(sample_counts) == N_FUNCTIONS * 2 and (sample_counts == N_SAMPLES_PER_ARCH).all(),
            'Every function must have exactly 100 x86 and 100 ARM samples')
    require(set(dataset.machine_tag) == {'amd64', 'arm64'}, 'Unexpected architecture labels')
    columns = list(dict.fromkeys(PREDICTORS_14 + REGRESSORS_3))
    for c in columns:
        dataset[c] = pd.to_numeric(dataset[c], errors='coerce')
    require(np.isfinite(dataset[columns].to_numpy(dtype=float)).all(),
            'Non-finite native author-model features')
    require((dataset['runtime'].to_numpy(float) > 0).all(), 'Non-positive profiling runtime')
    # This is data validation, not an extra model-selection or preprocessing step.
    for c in REGRESSORS_3:
        require((dataset[c].to_numpy(float) >= 0).all(), f'Negative CPU time delta: {c}')
    for c in ['x86_median_duration_ms', 'arm_median_duration_ms', 'arm_vs_x86_delta_percent']:
        truth[c] = pd.to_numeric(truth[c], errors='coerce')
    require(np.isfinite(truth[['x86_median_duration_ms', 'arm_median_duration_ms',
                              'arm_vs_x86_delta_percent']].to_numpy(float)).all(),
            'Ground-truth duration/delta invalid')
    require((truth[['x86_median_duration_ms', 'arm_median_duration_ms']] > 0).all().all(),
            'Non-positive ground-truth duration')
    calc_delta = 100 * (truth.arm_median_duration_ms / truth.x86_median_duration_ms - 1)
    require(np.allclose(calc_delta, truth.arm_vs_x86_delta_percent, rtol=1e-10, atol=1e-8),
            'Ground-truth percentage inconsistent')
    expected_labels = np.where(calc_delta < -15, 'ARM-faster',
                               np.where(calc_delta > 15, 'ARM-slower', 'ARM-similar'))
    require(np.array_equal(truth.paper_class_15.to_numpy(), expected_labels),
            'Ground-truth classes not at 15%')
    return dataset, truth, manifest, {'features': columns}


def diagnostics(dataset: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    median_runtime = dataset.groupby(['function_name', 'machine_tag'])['runtime'].median().unstack()
    target = truth.set_index('function_name')
    values = pd.DataFrame(index=median_runtime.index)
    values['gcp_x86_duration_median'] = target.x86_median_duration_ms
    values['profile_x86_runtime_median'] = median_runtime.amd64
    values['gcp_arm_duration_median'] = target.arm_median_duration_ms
    values['profile_arm_runtime_median'] = median_runtime.arm64
    values['x86_relative_difference_percent'] = 100 * (
        values.profile_x86_runtime_median / values.gcp_x86_duration_median - 1)
    values['arm_relative_difference_percent'] = 100 * (
        values.profile_arm_runtime_median / values.gcp_arm_duration_median - 1)
    values['profile_faster_architecture'] = np.where(
        values.profile_arm_runtime_median < values.profile_x86_runtime_median, 'arm64', 'amd64')
    values['ground_truth_faster_architecture'] = np.where(
        values.gcp_arm_duration_median < values.gcp_x86_duration_median, 'arm64', 'amd64')
    values['arch_winner_disagreement'] = (
        values.profile_faster_architecture != values.ground_truth_faster_architecture)
    return values.reset_index()


def make_pairs(data: pd.DataFrame, train_functions: list[str]) -> pd.DataFrame:
    """Pair equal runtime ranks per FUNCTION, never across different workloads."""
    selected = data.loc[data.function_name.isin(train_functions)]
    paired = []
    for fn in train_functions:
        one = selected.loc[selected.function_name == fn]
        x = one.loc[one.machine_tag == 'amd64'].sort_values(
            ['runtime', 'experiment_id', 'request_id'], kind='mergesort').reset_index(drop=True)
        arm = one.loc[one.machine_tag == 'arm64'].sort_values(
            ['runtime', 'experiment_id', 'request_id'], kind='mergesort').reset_index(drop=True)
        require(len(x) == len(arm) == N_SAMPLES_PER_ARCH, f'Unbalanced pairing: {fn}')
        frame = x[['function_name', *REGRESSORS_3, 'runtime']].copy()
        frame = frame.rename(columns={'runtime': 'runtime_x86_ms'})
        frame['runtime_arm_ms'] = arm['runtime'].to_numpy(float)
        frame['quantile_rank'] = np.arange(N_SAMPLES_PER_ARCH, dtype=int)
        paired.append(frame)
    result = pd.concat(paired, ignore_index=True)
    require(len(result) == N_SAMPLES_PER_ARCH * len(train_functions),
            'Not every training rank was paired')
    return result


def cpu_kernel_percent(x86: pd.DataFrame) -> float:
    user = float(x86['cpuUserDelta'].sum())
    kernel = float(x86['cpuKernelDelta'].sum())
    require(user + kernel > 0, f'Cannot determine CPU user/kernel split for {x86.function_name.iloc[0]}')
    return 100 * kernel / (user + kernel)


def bound_label(kernel_pct: float) -> str:
    # In the paper, >10% CPU kernel time denotes CPU-Kernel-intensive.
    return 'cpu-kernel' if kernel_pct > 10 else 'cpu-user'


def majority_class(predictions: np.ndarray) -> str:
    counts = Counter(predictions.tolist())
    require(set(counts) <= set(CLASSES), f'Unexpected predictions: {counts}')
    # Fixed and documented tie-break rule, NOT fitted/tuned using ground truth.
    return sorted(CLASSES, key=lambda klass: (-counts[klass], CLASSES.index(klass)))[0]


def run_fixed_models(data: pd.DataFrame, truth: pd.DataFrame):
    gt = truth.set_index('function_name', drop=False)
    names = sorted(gt.index.tolist())
    class_rows, sample_rows, per_strategy = [], [], []
    for index, target in enumerate(names, 1):
        training_names = [f for f in names if f != target]
        x_train = data.loc[(data.machine_tag == 'amd64') &
                           data.function_name.isin(training_names)].copy()
        x_test = data.loc[(data.machine_tag == 'amd64') &
                          (data.function_name == target)].copy()
        require(len(x_train) == (len(names)-1)*N_SAMPLES_PER_ARCH and len(x_test) == N_SAMPLES_PER_ARCH,
                f'Invalid function-holdout split: {target}')
        require(target not in set(x_train.function_name), f'Train/test leakage: {target}')
        class_train_y = x_train.function_name.map(gt.paper_class_15).to_numpy()
        classifier = RandomForestClassifier(**MODEL_CLASSIFIER)
        classifier.fit(x_train[PREDICTORS_14], class_train_y)
        sample_classes = classifier.predict(x_test[PREDICTORS_14])
        predicted_class = majority_class(sample_classes)
        for rec, predicted in zip(x_test[['function_name','experiment_id','request_id']].to_dict('records'),
                                  sample_classes):
            class_rows.append({**rec, 'fold_target_function': target,
                               'predicted_paper_class': predicted})
        target_pct = cpu_kernel_percent(x_test)
        target_bound = bound_label(target_pct)
        train_x86_bound = {
            fn: bound_label(cpu_kernel_percent(x_train.loc[x_train.function_name == fn]))
            for fn in training_names
        }
        paired = make_pairs(data, training_names)
        x_test_sorted = x_test.sort_values(['runtime','experiment_id','request_id'], kind='mergesort').reset_index(drop=True)
        predictions_for_target = []
        for strategy in STRATEGIES:
            if strategy == 'all-in-one':
                group = 'all'
                train_funcs = training_names
            elif strategy == 'resource-bound':
                group = target_bound
                train_funcs = [f for f in training_names if train_x86_bound[f] == group]
            else:
                group = predicted_class
                train_funcs = [f for f in training_names if gt.loc[f, 'paper_class_15'] == group]
            require(target not in train_funcs, f'Target leaked to regression training: {target}')
            if not train_funcs:
                raise ValueError(f'No training functions in {strategy}/{group} for heldout target {target}; '
                                 'do NOT silently choose another model')
            fold_training = paired.loc[paired.function_name.isin(train_funcs)]
            require(len(fold_training) == N_SAMPLES_PER_ARCH * len(train_funcs), 'Regression training pair count incorrect')
            regressor = RandomForestRegressor(**MODEL_REGRESSOR)
            regressor.fit(fold_training[REGRESSORS_3], fold_training['runtime_arm_ms'])
            predicted = regressor.predict(x_test_sorted[REGRESSORS_3])
            require(np.isfinite(predicted).all() and (predicted > 0).all(),
                    f'Invalid ARM predictions for {target}/{strategy}')
            # Keep only model predictions and training metadata until *all*
            # strategies have finished inference on this held-out function.
            predictions_for_target.append((strategy, group, len(train_funcs),
                                           len(fold_training), predicted))
            del regressor

        # Only now access the target's ARM profiling and measured ground truth.
        # Neither classifier nor any regressor sees these values during fit/predict.
        target_y = gt.loc[target]
        target_arm_sorted = data.loc[(data.function_name == target) &
                                     (data.machine_tag == 'arm64')].sort_values(
                                         ['runtime','experiment_id','request_id'], kind='mergesort').reset_index(drop=True)
        real_arm_rank = target_arm_sorted['runtime'].to_numpy(dtype=float)
        real_x86_rank = x_test_sorted['runtime'].to_numpy(dtype=float)
        for strategy, group, n_train_funcs, n_training_pairs, predicted in predictions_for_target:
            for rank, (experiment_id, request_id, pred, x_real, arm_real) in enumerate(zip(
                x_test_sorted.experiment_id, x_test_sorted.request_id, predicted, real_x86_rank, real_arm_rank)):
                sample_rows.append({
                    'function_name': target, 'strategy': strategy, 'model_group': group,
                    'rank': rank, 'x86_experiment_id': experiment_id, 'x86_request_id': request_id,
                    'x86_runtime_ms': float(x_real),
                    'predicted_arm_runtime_ms': float(pred),
                    'observed_arm_runtime_rank_ms_EVAL_ONLY': float(arm_real),
                    'ape_percent_EVAL_ONLY': 100 * abs(float(pred) - float(arm_real)) / float(arm_real),
                })
            predicted_median = float(np.median(predicted))
            x86_runtime_median = float(np.median(real_x86_rank))
            observed_arm_runtime_median = float(np.median(real_arm_rank))
            chosen = 'arm64' if predicted_median < x86_runtime_median else 'amd64'
            measured_best = ('arm64' if float(target_y.arm_median_duration_ms) <
                             float(target_y.x86_median_duration_ms) else 'amd64')
            ground_truth_chosen_latency = (float(target_y.arm_median_duration_ms) if chosen == 'arm64'
                                           else float(target_y.x86_median_duration_ms))
            ground_truth_optimal_latency = min(float(target_y.arm_median_duration_ms),
                                               float(target_y.x86_median_duration_ms))
            r2_denom = float(np.sum((real_arm_rank - np.mean(real_arm_rank)) ** 2))
            r2_rank = (1 - float(np.sum((predicted - real_arm_rank) ** 2)) / r2_denom
                       if r2_denom > 0 else float('nan'))
            per_strategy.append({
                'function_name': target,
                'strategy': strategy,
                'regressor': 'RandomForestRegressor(multireg; n_estimators=1000; random_state=42)',
                'model_group': group,
                'regressor_training_functions': n_train_funcs,
                'regressor_training_pairs': n_training_pairs,
                'paper_true_class_15_EVAL_ONLY': target_y.paper_class_15,
                'paper_predicted_class_15': predicted_class,
                'classification_correct_EVAL_ONLY': bool(predicted_class == target_y.paper_class_15),
                'classification_sample_accuracy_EVAL_ONLY': float(np.mean(sample_classes == target_y.paper_class_15)),
                'classification_votes_arm_faster': int((sample_classes == 'ARM-faster').sum()),
                'classification_votes_arm_similar': int((sample_classes == 'ARM-similar').sum()),
                'classification_votes_arm_slower': int((sample_classes == 'ARM-slower').sum()),
                'cpu_kernel_percent_x86': target_pct,
                'predicted_arm_runtime_median_ms': predicted_median,
                'profile_x86_runtime_median_ms': x86_runtime_median,
                'profile_arm_runtime_median_ms_EVAL_ONLY': observed_arm_runtime_median,
                'mean_rank_ape_percent_EVAL_ONLY': float(np.mean(100 * np.abs(predicted - real_arm_rank) / real_arm_rank)),
                'mean_absolute_error_arm_ms_EVAL_ONLY': float(np.mean(np.abs(predicted - real_arm_rank))),
                'r2_rank_EVAL_ONLY': r2_rank,
                'median_pred_error_percent_EVAL_ONLY': 100 * (predicted_median / observed_arm_runtime_median - 1),
                'paper_predicted_architecture': chosen,
                'measured_faster_architecture_EVAL_ONLY': measured_best,
                'architecture_agreement_EVAL_ONLY': bool(chosen == measured_best),
                'architectural_regret_percent_EVAL_ONLY': 100 * (ground_truth_chosen_latency / ground_truth_optimal_latency - 1),
                'gcp_x86_median_duration_ms_EVAL_ONLY': float(target_y.x86_median_duration_ms),
                'gcp_arm_median_duration_ms_EVAL_ONLY': float(target_y.arm_median_duration_ms),
                'gcp_delta_percent_EVAL_ONLY': float(target_y.arm_vs_x86_delta_percent),
            })
        del classifier, paired
        gc.collect()
        print(f'[{index:02d}/{len(names)}] {target}: predicted_class={predicted_class}, '
              f'completed {len(STRATEGIES)} fixed-paper strategies', flush=True)
    return pd.DataFrame(class_rows), pd.DataFrame(sample_rows), pd.DataFrame(per_strategy)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for strategy in STRATEGIES:
        part = df.loc[df.strategy == strategy]
        require(len(part) == N_FUNCTIONS, f'Expected {N_FUNCTIONS} results for {strategy}')
        result.append({
            'strategy': strategy,
            'function_count': int(len(part)),
            'architecture_agreement_fraction': float(part.architecture_agreement_EVAL_ONLY.mean()),
            'mean_absolute_median_error_percent': float(part.median_pred_error_percent_EVAL_ONLY.abs().mean()),
            'mean_rank_mape_percent': float(part.mean_rank_ape_percent_EVAL_ONLY.mean()),
            'mean_rank_r2': float(part.r2_rank_EVAL_ONLY.mean()),
            'mean_architectural_regret_percent': float(part.architectural_regret_percent_EVAL_ONLY.mean()),
            'mean_regressor_training_functions': float(part.regressor_training_functions.mean()),
        })
    return pd.DataFrame(result)


def write_results(output: Path, data: pd.DataFrame, truth: pd.DataFrame,
                  source_manifest: dict, diagnostics_df: pd.DataFrame):
    require(not output.exists(), f'Output directory already exists; refuse overwrite: {output}')
    print('Starting fixed models: 59 function-held-out folds, classifier RF + three RF-MLR strategies.',
          flush=True)
    classes, reg_samples, per_fn = run_fixed_models(data, truth)
    summary = summarize(per_fn)
    require(len(classes) == 5900 and len(reg_samples) == 17700 and len(per_fn) == 177,
            'Unexpected prediction count')
    require(set(per_fn.function_name) == set(truth.function_name), 'Not all functions evaluated')
    # Do not emit results until the full grid is finished successfully.
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        'classifier_sample_predictions.csv': classes,
        'regression_sample_predictions.csv': reg_samples,
        'paper_predictions_by_function_and_strategy.csv': per_fn,
        'paper_metrics_by_strategy.csv': summary,
        'runtime_semantics_diagnostic.csv': diagnostics_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    document = {
        'run_completed_utc': datetime.now(timezone.utc).isoformat(),
        'original_source_git_commit': '8fdf033c99f87d1500dd37bd28c01007e34ae35c',
        'paper_reference': 'Chen et al., UCC 2024, generalized ARM-speed / Resource-bound / All-in-one',
        'model_policy': 'NO HYPERPARAMETER TUNING, NO MODEL RANK-BASED SELECTION',
        'classifier': 'sklearn.ensemble.RandomForestClassifier(random_state=42)',
        'classifier_actual_params': RandomForestClassifier(**MODEL_CLASSIFIER).get_params(),
        'classification_features': PREDICTORS_14,
        'classifier_aggregation': 'majority of all 100 held-out x86 warm sample predictions',
        'regressor': 'sklearn.ensemble.RandomForestRegressor(n_estimators=1000, random_state=42)',
        'regressor_actual_params': RandomForestRegressor(**MODEL_REGRESSOR).get_params(),
        'regression_features': REGRESSORS_3,
        'training_pairing': 'within function: independently rank x86 and ARM by runtime; pair corresponding ranks',
        'holdout': 'leave one entire function out for both classifier and regressor',
        'training_arm_labels': 'known only for training functions to partition ARM-speed regression models',
        'target_arm_labels': 'used only after prediction for scoring',
        'strategies': list(STRATEGIES),
        'resource_bound_rule': 'CPU kernel time percent = 100*sum(cpuKernelDelta)/(sum(cpuKernelDelta)+sum(cpuUserDelta)) on x86 per function; CPU-kernel if >10%',
        'arm_speed_rule': 'predict test label from 100 x86 RF classifications; fit only training functions of that class',
        'scope_difference_from_original_paper': [
            'Serverledge/GCP corpus (59 functions, 100 warm samples per architecture) rather than authors AWS Lambda corpus.',
            'LOFO generalization: all 59 functions evaluated as unseen functions.',
            'The author paper mentions 21 SAAF measurements; their public classifier training code drops 7, leaving 14 features.',
            'The author repository exposes regression primitives but not fully scripted RQ2 category orchestration.',
            'Resource-bound grouping calculation is an explicit operationalization of the published >10% CPU-kernel criterion.',
            'A single function configuration is available, so rank matching is done inside each function.',
            'Profiling runtime and separate GCP duration ground truth may be distinct quantities; inspect runtime_semantics_diagnostic.csv.',
        ],
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
                     'python': sys.version.split()[0]},
        'source_dataset_manifest_sha256': digest(DATASET_ROOT / 'dataset_manifest.json'),
        'outputs': {name: {'rows': len(frame), 'sha256': digest(output / name)}
                    for name, frame in outputs.items()},
        'source_dataset_sha256': source_manifest['source_files_sha256']['native_gcp']['sha256'],
    }
    (output / 'paper_model_manifest.json').write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print('\nPASS: fixed-paper LOFO completed')
    for name, info in document['outputs'].items():
        print(f'  {name}: {info["rows"]} rows')
    print('\nDescriptive metrics, no tuning:')
    print(summary.to_string(index=False))
    print('Output:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run'))
    args = parser.parse_args()
    data, truth, manifest, _ = load_inputs(DATASET_ROOT)
    diag = diagnostics(data, truth)
    print('PASS dataset and manifest: 59 functions, 11,800 samples, 100/architecture, 14 native features.')
    print('Paper classes:', truth.paper_class_15.value_counts().to_dict())
    print('Observed median(runtime) vs independent GCP ground truth median(duration_ms):')
    for arch in ('x86', 'arm'):
        col = f'{arch}_relative_difference_percent'
        print(f'  {arch}: median_abs_diff={diag[col].abs().median():.4f}%, '
              f'max_abs_diff={diag[col].abs().max():.4f}%')
    disagreement = diag.loc[diag.arch_winner_disagreement, 'function_name'].tolist()
    print(f'  Arch-best disagreement between profiling runtime and GCP duration: {len(disagreement)}/59')
    if disagreement:
        print('  Affected functions:', ', '.join(disagreement))
    print('  NOTE: These are distinct signals until shown equivalent; no mixing in model training.')
    if args.action == 'check':
        print('PASS CHECK (no model trained, no new files created)')
    else:
        write_results(OUTPUT_ROOT, data, truth, manifest, diag)


if __name__ == '__main__':
    main()
