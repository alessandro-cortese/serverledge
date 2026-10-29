#!/usr/bin/env python3
"""Read-only inference diagnostics for the *already trained* fixed Chen et al. RF.

No fit(), no retraining, no hyperparameter optimization and no sample reweighting.

Inputs (relative to the repository root):
  data/architecture-comparison/datasets/v1/{profiling_common59_enriched.csv,
      ground_truth_common59.csv,dataset_manifest.json}
  data/architecture-comparison/paper-original/v1/{classifier_sample_predictions.csv,
      paper_predictions_by_function_and_strategy.csv,paper_model_manifest.json}

Outputs ONLY after successful validation, into a new directory:
  data/architecture-comparison/diagnostics/paper-classifier-v1

Usage:
  python3 analysis/profiling/diagnose_paper_classifier_no_tuning.py check
  python3 analysis/profiling/diagnose_paper_classifier_no_tuning.py run
  python3 analysis/profiling/diagnose_paper_classifier_no_tuning.py self-test

The direct ternary classifier abstains for ARM-similar. For transparency we also
report BOTH fixed, hypothetical completions (similar -> x86, similar -> ARM),
without picking a 'best' one. Neither is the authors' full pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from paper_original_prediction_pipeline import (
    load_inputs, digest, PREDICTORS_14, CLASSES, N_FUNCTIONS,
)

PAPER = Path('data/architecture-comparison/paper-original/v1')
OUTPUT = Path('data/architecture-comparison/diagnostics/paper-classifier-v1')
CLASS_ORDER = list(CLASSES)
ARCHES = ('amd64', 'arm64')


def require(test: bool, message: str):
    if not test:
        raise ValueError(message)


def sha(path: Path) -> str:
    return digest(path)


def read_results():
    manifest_path = PAPER / 'paper_model_manifest.json'
    require(manifest_path.is_file(), f'File mancante: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name in ('classifier_sample_predictions.csv', 'paper_predictions_by_function_and_strategy.csv'):
        path = PAPER / name
        info = manifest['outputs'].get(name)
        require(info and path.is_file(), f'File risultato mancante o non tracciato: {path}')
        require(sha(path) == info['sha256'], f'SHA-256 risultato non corrisponde al manifest: {path}')
    samples = pd.read_csv(PAPER / 'classifier_sample_predictions.csv')
    decisions_all = pd.read_csv(PAPER / 'paper_predictions_by_function_and_strategy.csv')
    required_sample = {'function_name', 'fold_target_function', 'experiment_id',
                       'request_id', 'predicted_paper_class'}
    required_decision = {'function_name', 'strategy', 'paper_true_class_15_EVAL_ONLY',
                         'paper_predicted_class_15', 'classification_correct_EVAL_ONLY',
                         'classification_sample_accuracy_EVAL_ONLY',
                         'classification_votes_arm_faster', 'classification_votes_arm_similar',
                         'classification_votes_arm_slower',
                         'paper_predicted_architecture',
                         'measured_faster_architecture_EVAL_ONLY'}
    require(required_sample <= set(samples), f'Colonne mancanti classifier_sample_predictions: {required_sample-set(samples)}')
    require(required_decision <= set(decisions_all), f'Colonne mancanti per-function: {required_decision-set(decisions_all)}')
    decision = decisions_all.loc[decisions_all.strategy == 'arm-speed'].copy()
    require(len(samples) == 5900 and len(decision) == N_FUNCTIONS,
            f'Cardinalita inattesa: samples={len(samples)}, functions={len(decision)}')
    require(not decision.function_name.duplicated().any(), 'Duplicati in decisioni arm-speed')
    require((samples.function_name == samples.fold_target_function).all(),
            'Fold target diverso dalla funzione osservata')
    require(not samples.duplicated(['function_name','experiment_id','request_id']).any(),
            'Predizioni campione con chiave duplicata')
    require(set(samples.predicted_paper_class) <= set(CLASS_ORDER), 'Classe predetta sconosciuta')
    require(set(decision.paper_predicted_class_15) <= set(CLASS_ORDER), 'Classe maggioritaria sconosciuta')
    require(set(samples.function_name) == set(decision.function_name), 'Funzioni di predizione non corrispondono')
    require((samples.groupby('function_name').size() == 100).all(), 'Numero predizioni per funzione !=100')
    require((decision['classification_votes_arm_faster'] + decision['classification_votes_arm_similar'] +
             decision['classification_votes_arm_slower'] == 100).all(), 'Voti per funzione != 100')
    return samples, decision, manifest


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Descriptive one-dimensional explained dispersion of function medians."""
    x = pd.to_numeric(values, errors='raise').to_numpy(float)
    y = pd.Series(groups).to_numpy()
    require(np.isfinite(x).all() and len(x) == len(y), 'Invalid eta-squared inputs')
    total = float(np.sum((x - x.mean())**2))
    if total <= 0:
        return float('nan')
    between = sum(int((y == g).sum()) * float((x[y == g].mean() - x.mean()) ** 2)
                  for g in pd.unique(y))
    return min(1.0, max(0.0, float(between / total)))


def direct_architecture(label: str) -> str | None:
    return {'ARM-faster': 'arm64', 'ARM-slower': 'amd64', 'ARM-similar': None}[label]


def regret(chosen: str | None, x86_ms: float, arm_ms: float) -> float:
    if chosen is None:
        return float('nan')
    optimum = min(x86_ms, arm_ms)
    return 100 * ((x86_ms if chosen == 'amd64' else arm_ms) / optimum - 1)


def describe_feature_spread(x86: pd.DataFrame, truth: pd.DataFrame):
    x = x86.copy()
    for feat in PREDICTORS_14:
        x[feat] = pd.to_numeric(x[feat], errors='raise')
    require(np.isfinite(x[PREDICTORS_14].to_numpy(float)).all(), 'Non-finite x86 features')

    grouped = x.groupby('function_name', sort=True)
    med = grouped[PREDICTORS_14].median()
    actual = truth.set_index('function_name').loc[med.index]
    med['true_class_15_EVAL_ONLY'] = actual.paper_class_15.to_numpy()
    med['delta_percent_EVAL_ONLY'] = actual.arm_vs_x86_delta_percent.to_numpy(float)
    med['language'] = grouped['language'].first().loc[med.index].to_numpy()
    med['configured_cpus_values'] = grouped['configured_cpus'].nunique().loc[med.index].to_numpy(int)
    med['configured_memory_mb_values'] = grouped['configured_memory_mb'].nunique().loc[med.index].to_numpy(int)
    med['configured_memory_mb_median'] = grouped['configured_memory_mb'].median().loc[med.index].to_numpy(float)
    require(len(med) == N_FUNCTIONS, f'Wrong function count for feature diagnostic: {len(med)}')

    spread_rows, class_rows = [], []
    for f in PREDICTORS_14:
        # MAD around per-function median measures within-function fluctuation.
        fn_mad = grouped[f].apply(lambda s: float(np.median(np.abs(s.to_numpy(float) - np.median(s.to_numpy(float))))))
        between_mad = float(np.median(np.abs(med[f].to_numpy(float) - np.median(med[f].to_numpy(float)))))
        within_median = float(fn_mad.median())
        et = eta_squared(med[f], med['true_class_15_EVAL_ONLY'])
        spread_rows.append({
            'feature': f,
            'n_distinct_sample_values': int(x[f].nunique(dropna=True)),
            'n_distinct_function_medians': int(med[f].nunique(dropna=True)),
            'functions_constant_across_100_samples': int((grouped[f].nunique(dropna=True) == 1).sum()),
            'median_within_function_MAD': within_median,
            'between_function_medians_MAD': between_mad,
            'between_over_within_MAD': (between_mad / within_median if within_median > 0 else float('nan')),
            'within_MAD_zero': bool(within_median == 0),
            'eta2_class_15_descriptive': et,
            'spearman_feature_median_vs_gcp_delta': float(
                med[f].rank().corr(med['delta_percent_EVAL_ONLY'].rank())
            ) if med[f].nunique() > 1 else float('nan'),
        })
        for cls in CLASS_ORDER:
            values = med.loc[med.true_class_15_EVAL_ONLY == cls, f].to_numpy(float)
            require(len(values) > 0, f'Missing true class {cls}')
            class_rows.append({
                'feature': f, 'true_class_15_EVAL_ONLY': cls,
                'function_count': len(values),
                'min_of_function_medians': float(values.min()),
                'q1_of_function_medians': float(np.quantile(values, 0.25)),
                'median_of_function_medians': float(np.median(values)),
                'q3_of_function_medians': float(np.quantile(values, 0.75)),
                'max_of_function_medians': float(values.max()),
            })
    return med.reset_index(), pd.DataFrame(spread_rows), pd.DataFrame(class_rows)


def compute_direct(decisions: pd.DataFrame, truth: pd.DataFrame):
    rows=[]
    t = truth.set_index('function_name')
    for row in decisions.itertuples(index=False):
        original = t.loc[row.function_name]
        chosen = direct_architecture(row.paper_predicted_class_15)
        x = float(original.x86_median_duration_ms)
        a = float(original.arm_median_duration_ms)
        winner = 'amd64' if x <= a else 'arm64'
        policies = {
            'direct_with_abstention': chosen,
            'similar_to_x86_FIXED_SENSITIVITY': chosen or 'amd64',
            'similar_to_arm_FIXED_SENSITIVITY': chosen or 'arm64',
            'paper_arm_speed_regression_ORIGINAL': row.paper_predicted_architecture,
            'always_x86_BASELINE': 'amd64',
            'always_arm_BASELINE': 'arm64',
        }
        rows.append({
            'function_name': row.function_name,
            'true_class_15_EVAL_ONLY': original.paper_class_15,
            'predicted_class_15': row.paper_predicted_class_15,
            'direct_choice_or_abstain': chosen or 'ABSTAIN',
            'direct_is_covered': chosen is not None,
            'direct_class_correct_EVAL_ONLY': row.paper_predicted_class_15 == original.paper_class_15,
            'measured_best_arm_EVAL_ONLY': winner,
            'x86_gcp_median_duration_ms_EVAL_ONLY': x,
            'arm_gcp_median_duration_ms_EVAL_ONLY': a,
            **{f'choice__{key}': value or 'ABSTAIN' for key,value in policies.items()},
            **{f'agreement__{key}_EVAL_ONLY': bool(value == winner) if value else None
               for key,value in policies.items()},
            **{f'regret_percent__{key}_EVAL_ONLY': regret(value, x, a)
               for key,value in policies.items()},
        })
    out = pd.DataFrame(rows).sort_values('function_name').reset_index(drop=True)
    summary=[]
    for key in ('direct_with_abstention','similar_to_x86_FIXED_SENSITIVITY',
                'similar_to_arm_FIXED_SENSITIVITY','paper_arm_speed_regression_ORIGINAL',
                'always_x86_BASELINE','always_arm_BASELINE'):
        choose = out[f'choice__{key}']
        covered= choose != 'ABSTAIN'
        a=out.loc[covered, f'agreement__{key}_EVAL_ONLY'].astype(float)
        r=out.loc[covered, f'regret_percent__{key}_EVAL_ONLY'].astype(float)
        summary.append({
            'policy':key,'functions_total':len(out), 'functions_covered':int(covered.sum()),
            'coverage_fraction':float(covered.mean()),
            'accuracy_on_covered_fraction_EVAL_ONLY':float(a.mean()) if len(a) else float('nan'),
            'mean_regret_percent_on_covered_EVAL_ONLY':float(r.mean()) if len(r) else float('nan'),
            'median_regret_percent_on_covered_EVAL_ONLY':float(r.median()) if len(r) else float('nan'),
            'status': ('Paper regression (fixed authors model)' if key.startswith('paper_')
                       else 'Direct abstaining ternary baseline' if key.startswith('direct_')
                       else 'Sensitivity policy, NOT authors original model' if 'SENSITIVITY' in key
                       else 'Constant architecture baseline, NOT authors original model'),
        })
    return out, pd.DataFrame(summary)


def compute_classifier(samples: pd.DataFrame, decisions: pd.DataFrame, truth: pd.DataFrame):
    joined = samples.merge(truth[['function_name','paper_class_15']], on='function_name', validate='many_to_one')
    joined['correct_EVAL_ONLY'] = joined.predicted_paper_class == joined.paper_class_15
    mat_sample = (pd.crosstab(joined.paper_class_15, joined.predicted_paper_class)
                  .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0))
    mat_fn = (pd.crosstab(decisions.paper_true_class_15_EVAL_ONLY, decisions.paper_predicted_class_15)
              .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0))
    require(int(mat_sample.values.sum()) == 5900 and int(mat_fn.values.sum()) == 59,
            'Confusion totals invalid')
    function_rows=[]
    vote_cols = {'ARM-faster':'classification_votes_arm_faster',
                 'ARM-similar':'classification_votes_arm_similar',
                 'ARM-slower':'classification_votes_arm_slower'}
    for item in decisions.itertuples(index=False):
        fn_samples = joined.loc[joined.function_name == item.function_name]
        require(len(fn_samples)==100, 'Bad sample count for function')
        for cls in CLASS_ORDER:
            require((fn_samples.predicted_paper_class == cls).sum() == getattr(item, vote_cols[cls]),
                    f'Votes inconsistent: {item.function_name}/{cls}')
        require(math.isclose(float(fn_samples.correct_EVAL_ONLY.mean()),
                             float(item.classification_sample_accuracy_EVAL_ONLY), abs_tol=1e-10),
                f'Sample accuracy inconsistent: {item.function_name}')
        function_rows.append({
            'function_name': item.function_name,
            'true_class_15_EVAL_ONLY': item.paper_true_class_15_EVAL_ONLY,
            'predicted_class_15': item.paper_predicted_class_15,
            'votes_faster': int(item.classification_votes_arm_faster),
            'votes_similar': int(item.classification_votes_arm_similar),
            'votes_slower': int(item.classification_votes_arm_slower),
            'sample_accuracy_EVAL_ONLY': float(fn_samples.correct_EVAL_ONLY.mean()),
            'function_correct_EVAL_ONLY': bool(item.paper_true_class_15_EVAL_ONLY == item.paper_predicted_class_15),
            'vote_margin_first_minus_second': int(sorted([
                int(item.classification_votes_arm_faster), int(item.classification_votes_arm_similar),
                int(item.classification_votes_arm_slower)], reverse=True)[0] -
                sorted([int(item.classification_votes_arm_faster), int(item.classification_votes_arm_similar),
                        int(item.classification_votes_arm_slower)], reverse=True)[1]),
        })
    perfn = pd.DataFrame(function_rows)
    require(set(perfn.function_name) == set(truth.function_name), 'Missing function class diagnostics')
    class_rows=[]
    for cls in CLASS_ORDER:
        ss = joined.loc[joined.paper_class_15 == cls]
        ff = perfn.loc[perfn.true_class_15_EVAL_ONLY == cls]
        class_rows.append({'true_class_15':cls,'n_functions':len(ff),'n_samples':len(ss),
                           'correct_functions':int(ff.function_correct_EVAL_ONLY.sum()),
                           'correct_samples':int(ss.correct_EVAL_ONLY.sum()),
                           'function_recall':float(ff.function_correct_EVAL_ONLY.mean()),
                           'sample_recall':float(ss.correct_EVAL_ONLY.mean())})
    summary = pd.DataFrame(class_rows)
    overview = {
        'n_functions':len(perfn),'n_samples':len(joined),
        'function_accuracy':float(perfn.function_correct_EVAL_ONLY.mean()),
        'sample_accuracy':float(joined.correct_EVAL_ONLY.mean()),
        'always_similar_baseline_fraction':float((perfn.true_class_15_EVAL_ONLY=='ARM-similar').mean()),
        'predicted_similar_functions':int((perfn.predicted_class_15 == 'ARM-similar').sum()),
        'predicted_similar_samples':int((joined.predicted_paper_class == 'ARM-similar').sum()),
        'sample_balanced_accuracy_macro_recall':float(summary.sample_recall.mean()),
        'function_balanced_accuracy_macro_recall':float(summary.function_recall.mean()),
    }
    return perfn, summary, mat_fn, mat_sample, overview


def diagnostic_markdown(overview, classes, feature_stats, direct_summary):
    top = feature_stats.sort_values('eta2_class_15_descriptive', ascending=False).head(5)
    def pct(v): return f'{100*v:.2f}%'
    result=[
        '# Diagnostica RF paper — nessun tuning', '',
        '**Unità di valutazione primaria:** funzione (LOFO). I dati ARM della funzione target servono solo a valutare.',
        '', '## Classificazione congelata', '',
        f'- Funzioni: **{overview["n_functions"]}**; campioni OOF x86: **{overview["n_samples"]}**.',
        f'- Accuracy per campione: **{pct(overview["sample_accuracy"])}**.',
        f'- Accuracy per funzione: **{pct(overview["function_accuracy"])}**.',
        f'- Recall macro per funzione: **{pct(overview["function_balanced_accuracy_macro_recall"])}**.',
        f'- Predizione ARM-similar dopo majority: **{overview["predicted_similar_functions"]}/59**.',
        f'- Baseline sempre-similar: **{pct(overview["always_similar_baseline_fraction"])}**.',
        '', '| Vera classe | n funzioni | recall per funzione | recall per campione |',
        '|---|---:|---:|---:|',
    ]
    for r in classes.itertuples(index=False):
        result.append(f'| {r.true_class_15} | {r.n_functions} | {pct(r.function_recall)} | {pct(r.sample_recall)} |')
    result.extend(['', '## RF ternaria come selettore diretto', '',
                   'La classe `ARM-similar` **non decide** quale architettura sia più veloce. La baseline diretta si astiene in questo caso.',
                   '', '| Policy | Coverage | Accuracy sui casi coperti | Regret medio sui coperti |',
                   '|---|---:|---:|---:|'])
    for r in direct_summary.itertuples(index=False):
        result.append(f'| {r.policy} | {pct(r.coverage_fraction)} | '
                      f'{pct(r.accuracy_on_covered_fraction_EVAL_ONLY)} | '
                      f'{r.mean_regret_percent_on_covered_EVAL_ONLY:.2f}% |')
    result.extend(['', '**Avvertenza:** le due policy `similar_to_*` sono soltanto analisi di sensibilità prestabilite; '
                   'non sono la pipeline originale degli autori e non ne viene scelta una a posteriori.',
                   '', '## Diagnostica descrittiva delle feature x86', '',
                   'Eta² misura la dispersione delle **59 mediane per funzione** associata alla classe reale, '
                   'considerando una caratteristica alla volta. Non è una misura di importanza RF, di accuracy o una prova causale.',
                   '', '| Feature | eta² descrittivo | MAD interfunzione / MAD intrafunzione |',
                   '|---|---:|---:|'])
    for r in top.itertuples(index=False):
        ratio = r.between_over_within_MAD
        result.append(f'| {r.feature} | {r.eta2_class_15_descriptive:.4f} | '
                      f'{ratio:.3g} |' if pd.notna(ratio) else
                      f'| {r.feature} | {r.eta2_class_15_descriptive:.4f} | n.d. (MAD interno zero) |')
    result.extend(['', 'Per interpretare la sovrapposizione delle tre classi consultare '
                   '`feature_class_descriptive.csv` e `function_x86_feature_medians.csv`.',
                   '', '## Vincoli inferenziali', '',
                   '- Le 100 invocazioni della stessa funzione sono correlate; non equivalgono a 100 funzioni diverse.',
                   '- Nessun parametro del classificatore o del regressore è stato ottimizzato o modificato.',
                   '- La diagnostica usa le etichette vere ARM **solo per la spiegazione a posteriori**, non per predire.',
                   '- Non misura l’effetto causale della piattaforma AWS/GCP né quello delle 40 configurazioni del paper.',
                   '- Nessuna nuova raccolta di workload è prevista in questa fase.',
                   '- La valutazione finale del MAB avverrà con una campagna GCP dedicata, dopo i confronti offline.',
                   '', 'Fonti: manifest originali e CSV in `data/architecture-comparison/`; '
                   'Chen et al., UCC 2024, https://faculty.washington.edu/wlloyd/papers/ucc-2024-draft.pdf.', ''])
    return '\n'.join(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','self-test'])
    args=parser.parse_args()
    if args.command=='self-test':
        assert direct_architecture('ARM-faster')=='arm64'
        assert direct_architecture('ARM-slower')=='amd64'
        assert direct_architecture('ARM-similar') is None
        assert regret(None, 1.0, 2.0) != regret(None,1.0,2.0)  # nan
        assert regret('arm64',100,80)==0
        assert math.isclose(regret('amd64',100,80),25)
        assert math.isclose(eta_squared(pd.Series([0,0,1,1]),pd.Series(['a','a','b','b'])),1)
        assert math.isnan(eta_squared(pd.Series([1,1,1]),pd.Series(['a','b','c'])))
        print('PASS self-test: class mapping, regret, eta². No data read, no model trained.')
        return
    dataset, truth, dataset_manifest, _ = load_inputs(Path('data/architecture-comparison/datasets/v1'))
    samples, decisions, original_model_manifest = read_results()
    require(set(dataset.loc[dataset.machine_tag=='amd64'].function_name)==set(decisions.function_name),
            'Mismatch between profile and predictions')
    print('PASS inputs SHA-256 and frozen model results: 59 functions / 5900 classifier predictions')
    if args.command=='check':
        print('PASS CHECK: no model trained, no new files created')
        return
    require(not OUTPUT.exists(), f'Output already exists; refusing overwrite: {OUTPUT}')
    features, spread, distributions = describe_feature_spread(dataset.loc[dataset.machine_tag=='amd64'], truth)
    perfn, class_metrics, cm_func, cm_sample, overview=compute_classifier(samples,decisions,truth)
    direct, direct_summary=compute_direct(decisions, truth)
    require(len(features)==len(perfn)==len(direct)==N_FUNCTIONS, 'Bad function count')
    with_features=perfn.merge(features, on='function_name', validate='one_to_one')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    files={
        'function_x86_feature_medians.csv': features,
        'feature_variability_and_eta2.csv': spread,
        'feature_class_descriptive.csv': distributions,
        'classifier_by_function_diagnostic.csv': with_features,
        'classifier_by_true_class.csv': class_metrics,
        'classifier_confusion_functions.csv': cm_func.rename_axis('true_class_15').reset_index(),
        'classifier_confusion_samples.csv': cm_sample.rename_axis('true_class_15').reset_index(),
        'classifier_direct_architecture.csv': direct,
        'classifier_direct_policies_summary.csv': direct_summary,
    }
    for name,frame in files.items(): frame.to_csv(OUTPUT/name,index=False)
    (OUTPUT/'diagnostic_report.md').write_text(
        diagnostic_markdown(overview, class_metrics, spread, direct_summary),encoding='utf-8')
    record={
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'policy':'PURE DIAGNOSTIC ONLY. NO TUNING. NO TRAINING. INPUTS IMMUTABLE.',
        'unit_of_analysis':'functions (59); x86 samples (5900) for sample confusion only',
        'source_classifier_manifest_sha256':sha(PAPER/'paper_model_manifest.json'),
        'source_dataset_manifest_sha256':sha(Path('data/architecture-comparison/datasets/v1/dataset_manifest.json')),
        'input_sha256':{
            'classifier_sample_predictions.csv':sha(PAPER/'classifier_sample_predictions.csv'),
            'paper_predictions_by_function_and_strategy.csv':sha(PAPER/'paper_predictions_by_function_and_strategy.csv'),
            'profiling_common59_enriched.csv':sha(Path('data/architecture-comparison/datasets/v1/profiling_common59_enriched.csv')),
            'ground_truth_common59.csv':sha(Path('data/architecture-comparison/datasets/v1/ground_truth_common59.csv')),
        },
        'summary':overview,
        'outputs_sha256':{p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        'method_note':'Descriptive univariate eta² is NOT RF feature importance. No hypotheses about causes tested.',
    }
    (OUTPUT/'diagnostic_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS: diagnostic created at',OUTPUT)
    print('Function accuracy: {:.4f}%'.format(100*overview['function_accuracy']))
    print('Sample accuracy: {:.4f}%'.format(100*overview['sample_accuracy']))
    print('Predicted ARM-similar: {}/59'.format(overview['predicted_similar_functions']))
    print('\n===== PER-CLASS RECALL =====')
    print(class_metrics.to_string(index=False))
    print('\n===== DIRECT ARCHITECTURE (non-paper sensitivities labelled) =====')
    print(direct_summary[['policy','functions_covered','accuracy_on_covered_fraction_EVAL_ONLY',
                          'mean_regret_percent_on_covered_EVAL_ONLY']].to_string(index=False))
    print('\n===== TOP FEATURE ETA² — DESCRIPTIVE ONLY =====')
    print(spread.sort_values('eta2_class_15_descriptive',ascending=False)[[
        'feature','eta2_class_15_descriptive','between_over_within_MAD'
    ]].head(8).to_string(index=False))
    print('\nNOTE: no model trained, no GCP needed, no original file modified.')


if __name__=='__main__':
    main()
