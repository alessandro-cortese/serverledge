#!/usr/bin/env python3
"""Final no-tuning diagnostic + predeclared 2.5%-label sensitivity of Chen RF.

Run from Serverledge repository root; place next to paper_original_prediction_pipeline.py
and diagnose_paper_classifier_no_tuning.py in analysis/profiling/.

check: read-only validation of SHA, exact features, ground-truth at two thresholds,
       frozen 15% results, 59-function cohort, static readiness for Serverledge.
run:   repeats validation; computes DESCRIPTIVE univariate feature diagnostics;
       independently refits ONLY the author's untouched RandomForestClassifier(
       random_state=42) on the TRAINING functions' 2.5%-threshold labels,
       leave-one-entire-function-out, and saves derived CSVs under a NEW path.

The 2.5%-label run is *NOT* the authors' published 15%-threshold experiment.
It is a fixed, explicitly declared sensitivity to label definition, NOT tuning.
No RF regressor is retrained; no prior original results are overwritten.

No ARM features from test function enter fit(), preprocessing, or predict().
Ground truth for the held-out target is accessed only after all 100
out-of-fold x86 predictions have been computed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from paper_original_prediction_pipeline import (
    DATASET_ROOT, OUTPUT_ROOT as PAPER_ROOT, PREDICTORS_14,
    MODEL_CLASSIFIER, CLASSES, N_FUNCTIONS, N_SAMPLES_PER_ARCH,
    load_inputs, majority_class, digest, require,
)
from diagnose_paper_classifier_no_tuning import read_results, eta_squared

OUTPUT = Path('data/architecture-comparison/diagnostics/fixed-rf-label-thresholds-v1')
LABELS = ('ARM-faster', 'ARM-similar', 'ARM-slower')


def label_for(delta: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(delta, dtype=float)
    require(np.isfinite(values).all(), 'Non-finite delta')
    return np.where(values < -threshold, 'ARM-faster',
                    np.where(values > threshold, 'ARM-slower', 'ARM-similar'))


def class_accuracy_summary(frame: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    confusion = pd.crosstab(frame[true_col], frame[pred_col]).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    require(int(confusion.to_numpy().sum()) == len(frame), 'Confusion cardinality mismatch')
    rates = {}
    for klass in LABELS:
        support = int(confusion.loc[klass].sum())
        rates[klass] = float(confusion.loc[klass, klass] / support) if support else None
    available = [r for r in rates.values() if r is not None]
    return {
        'n': int(len(frame)),
        'accuracy': float((frame[true_col] == frame[pred_col]).mean()),
        'macro_recall': float(np.mean(available)) if available else float('nan'),
        'recall_by_class': rates,
        'confusion': confusion,
    }


def validate(data: pd.DataFrame, truth: pd.DataFrame, old_samples: pd.DataFrame,
             old_by_fn: pd.DataFrame) -> pd.DataFrame:
    req = {'function_name', 'paper_class_15', 'serverledge_preference_2p5',
           'arm_vs_x86_delta_percent', 'x86_median_duration_ms', 'arm_median_duration_ms'}
    require(req.issubset(truth), f'Ground truth missing {req-set(truth)}')
    gt = truth.copy().sort_values('function_name').reset_index(drop=True)
    delta = gt.arm_vs_x86_delta_percent.to_numpy(float)
    gt['class_15_checked'] = label_for(delta, 15.0)
    gt['class_2p5_checked'] = label_for(delta, 2.5)
    gt['different_15_vs_2p5'] = gt.class_15_checked != gt.class_2p5_checked
    require((gt.class_15_checked == gt.paper_class_15).all(), 'Bad 15 percent classes')
    map_p = {'ARM-faster': 'arm-preferred',
             'ARM-slower': 'x86-preferred', 'ARM-similar': 'architecture-independent'}
    require((gt.class_2p5_checked.map(map_p) == gt.serverledge_preference_2p5).all(),
            'Bad 2.5 percent preference labels')
    require(len(gt) == N_FUNCTIONS and len(data) == N_FUNCTIONS*2*N_SAMPLES_PER_ARCH,
            'Unexpected function and sample count')
    require(set(old_by_fn.function_name) == set(gt.function_name), 'Old predictions and truth mismatch')
    require(set(old_samples.function_name) == set(gt.function_name), 'Old sample predictions and truth mismatch')
    require((old_by_fn.set_index('function_name').loc[gt.function_name, 'paper_true_class_15_EVAL_ONLY'].to_numpy()
             == gt.paper_class_15.to_numpy()).all(), 'Frozen paper truth mismatch')
    # A preflight for future KMeans comparison only: x86 GCP native five dynamic
    # and historical/static two columns must exist and be finite in the common cohort.
    x86 = data.loc[data.machine_tag=='amd64']
    cols = ['pageFaultsDelta','utilizedCPUs','freeMemory','cpuUserDelta','cpuKernelDelta',
            'static_token_count_mean','static_function_count']
    require(set(cols).issubset(x86), f'Missing Serverledge seven features: {set(cols)-set(x86)}')
    require(np.isfinite(x86[cols].to_numpy(float)).all(), 'Non-finite Serverledge features')
    static_by_fn = x86.groupby('function_name')[['static_token_count_mean','static_function_count']].nunique()
    require((static_by_fn == 1).all().all(), 'Static feature values change within function')
    require((x86.groupby('function_name').size() == 100).all(), 'Wrong x86 samples count')
    return gt


def preflight():
    data, truth, dataset_manifest, _ = load_inputs(DATASET_ROOT)
    old_samples, old_fn, model_manifest = read_results()
    gt = validate(data, truth, old_samples, old_fn)
    print('PASS: 59 common functions; 5900 x86 + 5900 ARM; SHA of frozen inputs')
    print('15% labels:', gt.class_15_checked.value_counts().to_dict())
    print('2.5% labels:', gt.class_2p5_checked.value_counts().to_dict())
    print('Functions changing label:', int(gt.different_15_vs_2p5.sum()))
    print('PASS: future Serverledge PAPER5+STATIC2 inputs finite, static values consistent')
    print('NOTE: source-file identity is not proven by static CSV value consistency.')
    return data, gt, old_samples, old_fn


def diagnose_features(data: pd.DataFrame, gt: pd.DataFrame):
    x = data.loc[data.machine_tag == 'amd64'].copy()
    med = x.groupby('function_name', sort=True)[PREDICTORS_14].median()
    tf = gt.set_index('function_name').loc[med.index]
    feature = []
    for f in PREDICTORS_14:
        values = x.groupby('function_name')[f]
        within = values.apply(lambda s: float(np.median(np.abs(s.to_numpy(float)-np.median(s.to_numpy(float))))))
        base = med[f].to_numpy(dtype=float)
        between = float(np.median(np.abs(base-np.median(base))))
        feature.append({
            'feature':f,
            'n_distinct_sample_values':int(x[f].nunique()),
            'n_distinct_function_medians':int(med[f].nunique()),
            'n_functions_constant_in_100_calls':int((values.nunique()==1).sum()),
            'n_functions_with_MAD_zero':int((within==0).sum()),
            'median_within_function_MAD':float(within.median()),
            'between_function_medians_MAD':between,
            'eta2_class_15_descriptive':eta_squared(med[f],tf.class_15_checked),
            'eta2_class_2p5_descriptive':eta_squared(med[f],tf.class_2p5_checked),
        })
    descriptor=pd.DataFrame(feature)
    quant=[]
    for f in PREDICTORS_14:
        for threshold_col in ('class_15_checked', 'class_2p5_checked'):
            for label in LABELS:
                z=med.loc[tf[threshold_col]==label, f].to_numpy(float)
                require(len(z)>0, f'Missing {threshold_col} class {label}')
                quant.append({'feature':f,'label_definition':threshold_col,'class':label,'n_functions':len(z),
                              'q25':float(np.quantile(z,.25)),'median':float(np.median(z)),
                              'q75':float(np.quantile(z,.75)),'min':float(z.min()),'max':float(z.max())})
    perfn = tf[['class_15_checked','class_2p5_checked','different_15_vs_2p5',
                'arm_vs_x86_delta_percent','x86_median_duration_ms','arm_median_duration_ms']].copy()
    perfn['n_samples_x86'] = x.groupby('function_name').size().loc[med.index]
    perfn['n_distinct_x86_runtime_values'] = x.groupby('function_name')['runtime'].nunique().loc[med.index]
    perfn['x86_runtime_median_ms'] = med.runtime
    perfn['x86_runtime_q25_ms'] = x.groupby('function_name')['runtime'].quantile(.25).loc[med.index]
    perfn['x86_runtime_q75_ms'] = x.groupby('function_name')['runtime'].quantile(.75).loc[med.index]
    perfn['x86_runtime_iqr_over_median'] = ((perfn.x86_runtime_q75_ms-perfn.x86_runtime_q25_ms)
                                            /perfn.x86_runtime_median_ms)
    perfn['language'] = x.groupby('function_name')['language'].first().loc[med.index]
    perfn['configured_memory_mb'] = x.groupby('function_name')['configured_memory_mb'].first().loc[med.index]
    perfn['configured_cpus'] = x.groupby('function_name')['configured_cpus'].first().loc[med.index]
    perfn['n_memory_configurations'] = x.groupby('function_name')['configured_memory_mb'].nunique().loc[med.index]
    perfn['n_cpu_configurations'] = x.groupby('function_name')['configured_cpus'].nunique().loc[med.index]
    perfn['n_distinct_workload_ids'] = x.groupby('function_name')['experiment_id'].nunique().loc[med.index]
    perfn['distance_to_nearest_threshold_15_percent_points'] = abs(abs(perfn.arm_vs_x86_delta_percent)-15)
    perfn['distance_to_nearest_threshold_2p5_percent_points'] = abs(abs(perfn.arm_vs_x86_delta_percent)-2.5)
    return descriptor, pd.DataFrame(quant), perfn.reset_index()


def train_fixed_2p5(data: pd.DataFrame, gt: pd.DataFrame):
    """Same author RF, new *predeclared* training target threshold. NO tuning."""
    x86 = data.loc[data.machine_tag == 'amd64'].copy()
    gt_index = gt.set_index('function_name')
    names = sorted(gt.function_name)
    sample_rows=[]; byfn=[]
    for i, target in enumerate(names,1):
        train = x86.loc[x86.function_name != target]
        test = x86.loc[x86.function_name == target]
        require(len(train) == (len(names)-1)*N_SAMPLES_PER_ARCH and len(test) == N_SAMPLES_PER_ARCH,
                f'Invalid split for {target}')
        require(target not in set(train.function_name),f'Target leaked: {target}')
        ytrain=train.function_name.map(gt_index.class_2p5_checked)
        require(ytrain.notna().sum()==len(ytrain),'Missing training class')
        model=RandomForestClassifier(**MODEL_CLASSIFIER)
        model.fit(train[PREDICTORS_14], ytrain)
        labels=model.predict(test[PREDICTORS_14])
        predicted=majority_class(labels)
        # Evaluate only AFTER predict() so test target labels never enter fit().
        target_gt=gt_index.loc[target]
        true_label=target_gt.class_2p5_checked
        votes=Counter(labels.tolist())
        direct={'ARM-faster':'arm64','ARM-slower':'amd64','ARM-similar':None}[predicted]
        x=float(target_gt.x86_median_duration_ms); a=float(target_gt.arm_median_duration_ms)
        best='amd64' if x<=a else 'arm64'
        regret = (100* ((x if direct=='amd64' else a) / min(x,a)-1)) if direct else float('nan')
        for row, pred in zip(test[['experiment_id','request_id']].to_dict('records'), labels):
            sample_rows.append({'function_name':target,**row,'predicted_class_2p5':pred,
                                'true_class_2p5_EVAL_ONLY':true_label,
                                'correct_EVAL_ONLY':bool(pred==true_label)})
        byfn.append({'function_name':target,'true_class_2p5_EVAL_ONLY':true_label,
                     'predicted_class_2p5':predicted,'classification_correct_EVAL_ONLY':bool(predicted==true_label),
                     'sample_accuracy_EVAL_ONLY':float(np.mean(labels==true_label)),
                     'votes_faster':int(votes['ARM-faster']),'votes_similar':int(votes['ARM-similar']),
                     'votes_slower':int(votes['ARM-slower']),
                     'direct_choice_or_abstain':direct or 'ABSTAIN',
                     'measured_best_arm_EVAL_ONLY':best,
                     'direct_architecture_agreement_EVAL_ONLY':bool(direct==best) if direct else None,
                     'direct_regret_percent_EVAL_ONLY':regret})
        print(f'[2.5% {i:02d}/{len(names)}] {target}: {predicted}', flush=True)
    sample_frame=pd.DataFrame(sample_rows)
    fn_frame=pd.DataFrame(byfn)
    require(len(sample_frame)==len(names)*N_SAMPLES_PER_ARCH and len(fn_frame)==len(names),
            'Unexpected 2.5% output rows')
    return sample_frame,fn_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['self-test','check','run'])
    args=p.parse_args()
    if args.mode=='self-test':
        assert label_for(np.array([-15,-2.5,0,2.5,15]),15).tolist()==[
            'ARM-similar']*5
        assert label_for(np.array([-15,-2.5,0,2.5,15]),2.5).tolist()==[
            'ARM-faster','ARM-similar','ARM-similar','ARM-similar','ARM-slower']
        fake=pd.DataFrame({'a':['ARM-faster','ARM-similar','ARM-slower'],
                           'b':['ARM-faster','ARM-similar','ARM-slower']})
        assert class_accuracy_summary(fake,'a','b')['accuracy']==1
        fixed=RandomForestClassifier(**MODEL_CLASSIFIER)
        assert fixed.get_params()['random_state']==42
        assert fixed.get_params()['n_estimators']==100  # sklearn default, original code no n_estimators override
        print('PASS self-test: threshold strictness, class accounting and fixed author RF hyperparams')
        return
    data,gt,old_samples,old_by_fn=preflight()
    if args.mode=='check':
        print('PASS CHECK: no output, no model trained')
        return
    require(not OUTPUT.exists(),f'Refusing to overwrite: {OUTPUT}')
    feats, quant, variation=diagnose_features(data,gt)
    sample2p5, fn2p5=train_fixed_2p5(data,gt)
    stats2p5=class_accuracy_summary(fn2p5,'true_class_2p5_EVAL_ONLY','predicted_class_2p5')
    stats_samples2p5=class_accuracy_summary(sample2p5,'true_class_2p5_EVAL_ONLY','predicted_class_2p5')
    old = old_by_fn.sort_values('function_name')
    original_accuracy = float((old.paper_true_class_15_EVAL_ONLY == old.paper_predicted_class_15).mean())
    require(abs(original_accuracy-30/59)<1e-12,'Unexpected frozen original RF accuracy')
    old_samples_true=old_samples.merge(gt[['function_name','class_15_checked']],on='function_name',validate='many_to_one')
    original_sample_acc = float((old_samples_true.predicted_paper_class == old_samples_true.class_15_checked).mean())
    fixed_result_summary=pd.DataFrame([
        {'experiment':'paper_original_RF_15','label_threshold_percent':15.0,
         'class_accuracy_function_fraction':original_accuracy,
         'class_accuracy_sample_fraction':original_sample_acc,
         'macro_recall_function':class_accuracy_summary(old,'paper_true_class_15_EVAL_ONLY','paper_predicted_class_15')['macro_recall'],
         'n_direct_predictions':int((old.paper_predicted_class_15!='ARM-similar').sum()),
         'model_definition':'authors 15% label task; frozen existing training/predictions'},
        {'experiment':'fixed_author_RF_relabelled_2p5','label_threshold_percent':2.5,
         'class_accuracy_function_fraction':stats2p5['accuracy'],
         'class_accuracy_sample_fraction':stats_samples2p5['accuracy'],
         'macro_recall_function':stats2p5['macro_recall'],
         'n_direct_predictions':int((fn2p5.predicted_class_2p5!='ARM-similar').sum()),
         'model_definition':'sensitivity: same RF 14 features and hyperparams, training labels set to 2.5%; NOT original paper task'},
    ])
    # Output is atomic at the logical experiment level: only write after 59 folds.
    OUTPUT.mkdir(parents=True,exist_ok=False)
    files={
        'true_labels_both_thresholds_59.csv':gt,
        'feature_variability_both_thresholds.csv':feats,
        'feature_quantiles_by_class_threshold.csv':quant,
        'function_workload_variability_59.csv':variation,
        'classifier_2p5_sample_predictions.csv':sample2p5,
        'classifier_2p5_function_predictions.csv':fn2p5,
        'classifier_2p5_confusion_functions.csv':stats2p5['confusion'].rename_axis('true_class_2p5').reset_index(),
        'classifier_2p5_confusion_samples.csv':stats_samples2p5['confusion'].rename_axis('true_class_2p5').reset_index(),
        'frozen_15_vs_fixed_2p5_summary.csv':fixed_result_summary,
    }
    for filename,frame in files.items():
        frame.to_csv(OUTPUT/filename,index=False)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'purpose':'NO TUNING: diagnostic + a priori 2.5% label variant with same author RF parameters',
              'warning':'2.5% label run is NOT authors 15% model; thresholds are not interchangeable; results are not used to select a model',
              'fixed_author_classifier_params':RandomForestClassifier(**MODEL_CLASSIFIER).get_params(),
              'original_output_sha256':digest(PAPER_ROOT/'paper_model_manifest.json'),
              'source_dataset_sha256':digest(DATASET_ROOT/'dataset_manifest.json'),
              'source_diagnostic_sha256':digest(Path('data/architecture-comparison/diagnostics/paper-classifier-v1/diagnostic_manifest.json')),
              'versions':{'scikit-learn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
              'outputs':{fn:{'rows':len(frame),'sha256':digest(OUTPUT/fn)} for fn,frame in files.items()}}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    print('\n===== FROZEN 15% vs FIXED-RF 2.5% (distinct label tasks; no tuning) =====')
    print(fixed_result_summary.to_string(index=False))
    print('\n===== 2.5% FUNCTION CONFUSION =====')
    print(stats2p5['confusion'].to_string())
    print('\n===== FEATURE ETA² (DESCRIPTIVE, NO CAUSAL CLAIM) =====')
    print(feats[['feature','eta2_class_15_descriptive','eta2_class_2p5_descriptive',
                 'n_functions_with_MAD_zero']].sort_values('eta2_class_2p5_descriptive',ascending=False).to_string(index=False))
    print('\n===== WORKLOAD WITHIN-FUNCTION VARIATION =====')
    print(variation['x86_runtime_iqr_over_median'].describe().to_string())
    print('Function CPU configurations (unique counts):',variation.n_cpu_configurations.value_counts().to_dict())
    print('Function memory configurations (unique counts):',variation.n_memory_configurations.value_counts().to_dict())
    print('OUTPUT:',OUTPUT)
    print('PASS: original paper models, data, and result CSVs untouched; 2.5% is separately labeled.')

if __name__=='__main__':
    main()
