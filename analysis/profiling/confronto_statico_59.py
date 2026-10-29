#!/usr/bin/env python3
"""Produce final publication-ready support-matched tables WITHOUT retraining.

Reads the SHA-verified artifacts from compare_frozen_rf_vs_serverledge_59.py.
Four fixed methods are included: Chen RF 15%, Chen RF relabeled 2.5%,
Serverledge unweighted majority diagnostic, Serverledge frozen weighted 1/d^2.

Output A: each method on its own coverage (conditional accuracy/regret).
Output B: each pair on the SAME functions for each KMeans seed, CSV separate.
Output C: all FOUR methods on their common covered functions, per seed.
No abstention is recoded as a wrong arm. Units are functions; KMeans seeds
are repeated model-instantiation checks, not independent test workloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('data/architecture-comparison/comparison/frozen-rf-vs-serverledge-59-v1')
OUTPUT = Path('data/architecture-comparison/comparison/rf-serverledge-final-tables-v1')
METHODS = ('paper_RF_15_original', 'paper_RF_2p5_relabelled',
           'majority_diagnostic', 'weighted_frozen')
PAIRS = tuple((a,b) for i,a in enumerate(METHODS) for b in METHODS[i+1:])
SEEDS = (11,23,37,41,53)
N = 59


def ensure(ok, message):
    if not ok: raise AssertionError(message)


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def load():
    mpath=BASE/'manifest.json'
    ensure(mpath.is_file(), f'Missing {mpath}')
    manifest=json.loads(mpath.read_text(encoding='utf8'))
    def checked(filename):
        f=BASE/filename
        ensure(f.is_file() and filename in manifest['outputs'],f'Missing {f} or manifest record')
        rec=manifest['outputs'][filename]
        ensure(sha256(f)==rec['sha256'], f'SHA mismatch: {f}')
        z=pd.read_csv(f)
        ensure(len(z)==rec['rows'], f'Unexpected row count: {filename}')
        return z
    decisions=checked('direct_decisions_all_methods.csv')
    cohort=checked('function_comparison_summary_59.csv')
    ensure(len(cohort)==N and cohort.function_name.is_unique,'Bad cohort')
    ensure(set(decisions.method)==set(METHODS),'Different methods')
    ensure(len(decisions)==(2*N+2*N*len(SEEDS)),'Bad decision count')
    # Values stored as strings/booleans depending on the CSV parser version.
    decisions['coverage']=decisions.coverage.astype(str).str.lower().map(
        {'true':True,'false':False,'1':True,'0':False})
    ensure(decisions.coverage.notna().all(),'Invalid coverage')
    for field in ['arch_agreement_covered_EVAL_ONLY','regret_percent_covered_EVAL_ONLY']:
        decisions[field]=pd.to_numeric(decisions[field],errors='coerce')
    ensure(decisions.loc[decisions.coverage,'arch_agreement_covered_EVAL_ONLY'].isin((0,1)).all(),'Bad accuracy')
    ensure(decisions.loc[decisions.coverage,'regret_percent_covered_EVAL_ONLY'].ge(-1e-7).all(),'Bad regret')
    ensure(decisions.loc[decisions.coverage,'regret_percent_covered_EVAL_ONLY'].notna().all(),'Missing covered regret')
    ensure(set(decisions.function_name)==set(cohort.function_name),'Cohort misalignment')
    return decisions,mpath


def get_seed_rows(decisions,seed):
    z=decisions.loc[decisions.method.isin(METHODS)].copy()
    # Each RF row is evaluated identically against each cluster seed.
    z=z.loc[z.method.str.startswith('paper_') | (z.seed.astype(str)==str(seed))].copy()
    ensure(len(z)==N*len(METHODS),f'Seed {seed}: missing methods or functions')
    ensure(not z.duplicated(['method','function_name']).any(),f'Seed {seed}: duplicated key')
    for method in METHODS:
        ensure(set(z.loc[z.method==method,'function_name'])==set(z.function_name),
               f'Seed {seed}: function mismatch {method}')
    return z


def score(z):
    ensure(len(z)>0,'No covered function; cannot compare')
    return {'accuracy_fraction':float(z.arch_agreement_covered_EVAL_ONLY.mean()),
            'mean_regret_percent':float(z.regret_percent_covered_EVAL_ONLY.mean()),
            'median_regret_percent':float(z.regret_percent_covered_EVAL_ONLY.median())}


def summarize_group(z,group_keys,metrics):
    return (z.groupby(group_keys,as_index=False,sort=True)[metrics]
              .mean(numeric_only=True))


def build_tables(decisions):
    per_method=[]; per_pair=[]; four=[]; long_common=[]
    for seed in SEEDS:
        z=get_seed_rows(decisions,seed)
        by_name={method: z[z.method==method].set_index('function_name').sort_index()
                 for method in METHODS}
        for method, q in by_name.items():
            covered=q[q.coverage]
            per_method.append({'method':method,'seed':seed,'functions_total':N,
                'covered':int(q.coverage.sum()),'abstentions':int((~q.coverage).sum()),
                'coverage_fraction':float(q.coverage.mean()),**score(covered)})
        for a,b in PAIRS:
            qa,qb=by_name[a],by_name[b]
            common=qa.index[qa.coverage & qb.coverage]
            aa=qa.loc[common];bb=qb.loc[common]
            sa=score(aa);sb=score(bb)
            rec={'seed':seed,'method_a':a,'method_b':b,'common_covered':len(common),
                 'common_coverage_fraction':len(common)/N,
                 'a_covered_total':int(qa.coverage.sum()),'b_covered_total':int(qb.coverage.sum()),
                 'a_only':int((qa.coverage & ~qb.coverage).sum()),
                 'b_only':int((~qa.coverage & qb.coverage).sum()),
                 'both_abstain':int((~qa.coverage & ~qb.coverage).sum()),
                 'decision_disagreements_common':int((aa.predicted_architecture_or_abstain
                                                    !=bb.predicted_architecture_or_abstain).sum())}
            rec.update({f'a_{k}':v for k,v in sa.items()})
            rec.update({f'b_{k}':v for k,v in sb.items()})
            rec['b_minus_a_accuracy_pp']=100*(sb['accuracy_fraction']-sa['accuracy_fraction'])
            rec['b_minus_a_mean_regret_pp']=sb['mean_regret_percent']-sa['mean_regret_percent']
            per_pair.append(rec)
        common_all=by_name[METHODS[0]].index.copy()
        for method in METHODS:
            common_all=common_all.intersection(by_name[method].index[by_name[method].coverage])
        ensure(len(common_all)>0,f'Seed {seed}: zero all-method common functions')
        for method in METHODS:
            selected=by_name[method].loc[common_all]
            four.append({'seed':seed,'method':method,'common_covered_all_four':len(common_all),
                         'common_coverage_fraction':len(common_all)/N,**score(selected)})
            for fn,row in selected.iterrows():
                long_common.append({'seed':seed,'function_name':fn,'method':method,
                     'predicted_architecture':row.predicted_architecture_or_abstain,
                     'correct_EVAL_ONLY':int(row.arch_agreement_covered_EVAL_ONLY),
                     'regret_percent_EVAL_ONLY':float(row.regret_percent_covered_EVAL_ONLY)})
    dfm=pd.DataFrame(per_method); dfp=pd.DataFrame(per_pair)
    dff=pd.DataFrame(four); dfl=pd.DataFrame(long_common)
    by_method=summarize_group(dfm,['method'],[
        'covered','abstentions','coverage_fraction','accuracy_fraction','mean_regret_percent','median_regret_percent'])
    pair_keys=['method_a','method_b']
    pair_metrics=[c for c in dfp.select_dtypes(include='number').columns if c!='seed']
    by_pair=summarize_group(dfp,pair_keys,pair_metrics)
    four_metrics=['common_covered_all_four','common_coverage_fraction',
                  'accuracy_fraction','mean_regret_percent','median_regret_percent']
    by_four=summarize_group(dff,['method'],four_metrics)
    return {'coverage_original_per_seed.csv':dfm,
            'coverage_original_summary.csv':by_method,
            'same_coverage_pairwise_per_seed.csv':dfp,
            'same_coverage_pairwise_summary.csv':by_pair,
            'same_coverage_all_four_per_seed.csv':dff,
            'same_coverage_all_four_summary.csv':by_four,
            'same_coverage_all_four_by_function_seed.csv':dfl}


def self_test():
    # Validate the intersection: no abstentions silently transformed to errors.
    functions=[f'f{i:02d}' for i in range(N)]
    data=[]
    for method in METHODS:
        for fn in functions:
            for seed in (['fixed_RF'] if method.startswith('paper_') else SEEDS):
                covered=(fn!='f00') if method=='paper_RF_15_original' else True
                data.append({'method':method,'function_name':fn,'seed':seed,'coverage':covered,
                  'predicted_architecture_or_abstain':'amd64' if covered else 'ABSTAIN',
                  'arch_agreement_covered_EVAL_ONLY':1 if covered else np.nan,
                  'regret_percent_covered_EVAL_ONLY':0 if covered else np.nan})
    tables=build_tables(pd.DataFrame(data))
    f=tables['same_coverage_all_four_per_seed.csv']
    ensure((f.common_covered_all_four==58).all(),'Bad all-four coverage')
    q=tables['same_coverage_pairwise_per_seed.csv']
    ensure((q.query('method_a == "paper_RF_15_original"').common_covered==58).all(),'Bad pair coverage')
    ensure(len(tables['same_coverage_all_four_by_function_seed.csv'])==58*len(SEEDS)*len(METHODS),
           'Bad long table')
    print('PASS self-test: four-method intersection, no abstention imputation, seed pairing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['self-test','check','run'])
    a=p.parse_args()
    if a.command=='self-test':return self_test()
    inputs,source_manifest=load()
    tables=build_tables(inputs)
    print('===== ORIGINAL COVERAGE: ALL FOUR METHODS =====')
    print(tables['coverage_original_summary.csv'].to_string(index=False))
    print('\n===== PAIRWISE SAME COVERAGE =====')
    show=['method_a','method_b','common_covered','a_accuracy_fraction','b_accuracy_fraction',
          'a_mean_regret_percent','b_mean_regret_percent']
    print(tables['same_coverage_pairwise_summary.csv'][show].to_string(index=False))
    print('\n===== ALL FOUR ON SAME FUNCTIONS =====')
    print(tables['same_coverage_all_four_summary.csv'].to_string(index=False))
    if a.command=='check':
        print('\nPASS CHECK read-only; no files created')
        return
    ensure(not OUTPUT.exists(), f'Refusing overwrite: {OUTPUT}')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    saved={}
    for name,frame in tables.items():
        file=OUTPUT/name
        frame.to_csv(file,index=False)
        saved[name]={'rows':len(frame),'sha256':sha256(file)}
    info={'source_manifest_path':str(source_manifest), 'source_manifest_sha256':sha256(source_manifest),
          'cohort':N,'method_names':list(METHODS),'cluster_seeds':list(SEEDS),
          'no_tuning':True,'no_retraining':True,
          'unit_of_analysis':'function; KMeans seed values are repeated sensitivity checks',
          'abstention':'not classified as error; common-support intersection only',
          'output_sha256':saved}
    (OUTPUT/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    report=['# Confronto finale RF Chen vs Serverledge — 59 funzioni','',
            'Nessun riaddestramento, tuning, ricampionamento o cambio di donor.',
            'Le due soglie RF sono esperimenti distinti; majority è un controllo e weighted è la pipeline congelata.',
            'Coverage originale, confronto a copertura comune per coppia e confronto su intersezione di TUTTI e quattro i metodi sono tre letture diverse.',
            'La RF originale al 15% ha copertura limitata; l’intersezione a quattro può quindi essere molto piccola.',
            'I seed KMeans servono a verificare stabilità, non costituiscono 295 osservazioni indipendenti.',
            '', '## Copertura originale', '', tables['coverage_original_summary.csv'].to_markdown(index=False,floatfmt='.4f'),
            '', '## Copertura comune per coppia', '', tables['same_coverage_pairwise_summary.csv'].to_markdown(index=False,floatfmt='.4f'),
            '', '## Stesse funzioni per tutti e quattro i metodi', '',
            tables['same_coverage_all_four_summary.csv'].to_markdown(index=False,floatfmt='.4f'),
            '', '## Provenienza', '', f'- SHA256 manifest di origine: `{sha256(source_manifest)}`', '']
    (OUTPUT/'FINAL_COMPARISON_REPORT.md').write_text('\n'.join(report),encoding='utf8')
    print('\nPASS RUN:',OUTPUT)


if __name__=='__main__':main()
