#!/usr/bin/env python3
"""Read-only verification and evidence consolidation for frozen Chen RF vs Serverledge.

Does not refit any model, does not sample new observations, and does not change
any existing artifact. `check` prints an audit; `run` additionally writes a
new, separate evidence folder. Historical donor replay is compared only if the
old per-target/seed CSV exists. A mismatch is reported, never silently fixed.

From Serverledge repository root:
    python3 analysis/profiling/audit_final_comparison_59.py self-test
    python3 analysis/profiling/audit_final_comparison_59.py check
    python3 analysis/profiling/audit_final_comparison_59.py run
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path('data/architecture-comparison/comparison/frozen-rf-vs-serverledge-59-v1')
OLDER = Path('data/profiling/final-corpus64-20261005-analysis-01')
OLD_LOFO = OLDER / 'analysis-legacy-vs-final-59/lofo_59_per_target_seed.csv'
STATIC_CSV = OLDER / 'static-code-metrics.csv'
GCP_100 = Path('data/architecture-comparison/datasets/v1/profiling_common59_enriched.csv')
RESULT = Path('data/architecture-comparison/comparison/evidence-audit-59-v1')
EXPECTED_STATIC_SHA = 'adeba94386226b91bfe878f715ee4f10669cf7f54ab5219d98aadfa488df6bf2'
SEEDS = {11, 23, 37, 41, 53}
NAMES = ('paper_RF_15_original','paper_RF_2p5_relabelled','majority_diagnostic','weighted_frozen')
PAIRS = (
    ('paper_RF_15_original','weighted_frozen'),
    ('paper_RF_2p5_relabelled','weighted_frozen'),
    ('majority_diagnostic','weighted_frozen'),
    ('paper_RF_2p5_relabelled','majority_diagnostic'),
)


def require(ok, msg):
    if not bool(ok): raise AssertionError(msg)


def sha256(path:Path) -> str:
    require(path.is_file(),f'File non trovato: {path}')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verified_csv(name:str, manifest:dict) -> pd.DataFrame:
    require(name in manifest.get('outputs', {}),f'{name}: SHA mancante dal manifest')
    path=HERE/name
    expected=manifest['outputs'][name]['sha256']
    require(sha256(path)==expected,f'{name}: hash differente dal manifest')
    df=pd.read_csv(path, low_memory=False)
    require(len(df)==manifest['outputs'][name]['rows'],f'{name}: righe inattese')
    return df


def coerce_bool(series:pd.Series) -> pd.Series:
    vals=series.astype(str).str.strip().str.lower()
    require(vals.isin(('true','false','1','0')).all(),f'Valori booleani inattesi: {set(vals)}')
    return vals.isin(('true','1'))


def load_inputs():
    require(sha256(STATIC_CSV)==EXPECTED_STATIC_SHA, 'Le metriche statiche storiche sono cambiate')
    manifest_path=HERE/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    direct=verified_csv('direct_decisions_all_methods.csv',manifest)
    donors=verified_csv('serverledge_lofo_by_target_seed.csv',manifest)
    sources=verified_csv('source_provenance_check.csv',manifest)
    consensus=verified_csv('serverledge_seed_consensus_per_function.csv',manifest)
    wide=verified_csv('function_comparison_summary_59.csv',manifest)
    for frame, key, expected in ((wide,'function_name',59),(sources,'function_name',59)):
        require(frame[key].nunique()==expected and len(frame)==expected,'Cardinalità cohort non valida')
    require(set(direct['method'])==set(NAMES),'Metodi non corrispondenti ai quattro previsti')
    require(set(donors['method'])=={'majority_diagnostic','weighted_frozen'},'Metodi donor inattesi')
    require(set(pd.to_numeric(donors['seed']))==SEEDS,'Seed cluster non congelati')
    require(len(donors)==2*59*5 and len(direct)==2*59+2*59*5,'Cardinalità valutazione errata')
    require(not direct.duplicated(['method','seed','function_name']).any(),'Chiave decisione duplicata')
    require(not donors.duplicated(['method','seed','function_name']).any(),'Chiave donor duplicata')
    require(set(donors.function_name)==set(wide.function_name),'Cohort donor differente')
    direct=direct.copy()
    direct['coverage']=coerce_bool(direct['coverage'])
    direct['regret_percent_covered_EVAL_ONLY']=pd.to_numeric(direct['regret_percent_covered_EVAL_ONLY'],errors='coerce')
    direct['arch_agreement_covered_EVAL_ONLY']=pd.to_numeric(direct['arch_agreement_covered_EVAL_ONLY'],errors='coerce')
    require(direct.loc[direct.coverage,'regret_percent_covered_EVAL_ONLY'].notna().all(),'Regret mancante sui coperti')
    require((direct.loc[direct.coverage,'regret_percent_covered_EVAL_ONLY'] >= -1e-7).all(),'Regret negativo anomalo')
    require(direct.loc[direct.coverage,'arch_agreement_covered_EVAL_ONLY'].isin((0,1)).all(),'Accuracy invalid')
    for m in NAMES:
        temp=direct[direct.method==m]
        expected=59 if m.startswith('paper_') else 59*5
        require(len(temp)==expected, f'Numero righe {m} errato')
        require(set(temp.function_name)==set(wide.function_name),f'Funzioni discordanti {m}')
    require(sources.source_exists_now.astype(str).str.lower().isin(('true','1')).all(),
            'Alcuni sorgenti non sono ora leggibili')
    require(not coerce_bool(sources.historical_source_sha_verified).any(),
            'Flag SHA storico inatteso: esaminare manualmente')
    require(GCP_100.is_file(),'Dataset GCP a 100 campioni mancante')
    return manifest,direct,donors,sources,consensus,wide


def compare_shared(x:pd.DataFrame, a:str, b:str, seed:str|int|None=None) -> dict:
    one=x[x.method==a].set_index('function_name')
    two=x[x.method==b].set_index('function_name')
    require(set(one.index)==set(two.index) and len(one)==len(two)==59,'Funzioni non allineate')
    both=one.coverage & two.coverage
    inter=one[both].index
    result={'method_a':a,'method_b':b,'seed':str(seed if seed is not None else x.seed.unique()[0]),
            'covered_a':int(one.coverage.sum()),'covered_b':int(two.coverage.sum()),
            'common_covered':len(inter),'common_fraction_of_59':len(inter)/59,
            'both_abstain':int((~one.coverage&~two.coverage).sum()),
            'a_only':int((one.coverage&~two.coverage).sum()),
            'b_only':int((~one.coverage&two.coverage).sum())}
    for tag,frame in (('a',one),('b',two)):
        z=frame.loc[inter]
        result[f'{tag}_accuracy_common']=float(z.arch_agreement_covered_EVAL_ONLY.mean()) if len(z) else math.nan
        result[f'{tag}_mean_regret_common']=float(z.regret_percent_covered_EVAL_ONLY.mean()) if len(z) else math.nan
        result[f'{tag}_median_regret_common']=float(z.regret_percent_covered_EVAL_ONLY.median()) if len(z) else math.nan
    result['a_minus_b_accuracy_common_pp']=100*(result['a_accuracy_common']-result['b_accuracy_common'])
    result['a_minus_b_regret_common_pp']=result['a_mean_regret_common']-result['b_mean_regret_common']
    result['architecture_disagreements_on_common']=int((one.loc[inter,'predicted_architecture_or_abstain'] !=two.loc[inter,'predicted_architecture_or_abstain']).sum())
    return result


def common_support(direct:pd.DataFrame) -> pd.DataFrame:
    out=[]
    for seed in sorted(SEEDS):
        z=direct[(direct.seed.astype(str)==str(seed)) | (direct.method.str.startswith('paper_'))].copy()
        for a,b in PAIRS:
            out.append(compare_shared(z,a,b,seed))
    return pd.DataFrame(out)


def old_donor_comparison(new:pd.DataFrame):
    if not OLD_LOFO.is_file():
        return {'status':'MISSING_HISTORICAL_FILE','source':str(OLD_LOFO)},pd.DataFrame()
    old=pd.read_csv(OLD_LOFO,low_memory=False)
    require({'config','seed','target_function','status','donor_function'}.issubset(old.columns),
            'Colonne donor replay storico inattese')
    variants=[i for i in ('C_final_plain','C_final_weighted') if i in set(old.config)]
    if not variants:
        return {'status':'NO_RECOGNIZED_FROZEN_CONFIG','configurations':sorted(old.config.unique().tolist())},pd.DataFrame()
    old=old[old.config==variants[0]].copy()
    old['seed']=pd.to_numeric(old.seed,errors='raise').astype(int)
    old['old_donor']=np.where(old.status=='selected',old.donor_function.fillna(''),'')
    nw=new[new.method=='weighted_frozen'].copy()
    nw['seed']=pd.to_numeric(nw.seed,errors='raise').astype(int)
    nw['new_donor']=np.where(nw.status=='selected',nw.selected_donor.fillna(''),'')
    merged=nw[['function_name','seed','new_donor','predicted_architecture_or_abstain']].merge(
        old[['target_function','seed','old_donor']],left_on=['function_name','seed'],
        right_on=['target_function','seed'],how='outer',validate='one_to_one',indicator=True)
    require(len(merged)==295 and merged._merge.eq('both').all(),
            'Il replay storico non copre esattamente le 59 funzioni per cinque seed')
    merged['identical_donor']=merged.new_donor.fillna('')==merged.old_donor.fillna('')
    info={'status':'MATCHED','historical_configuration':variants[0],
          'historical_lofo_file_sha256':sha256(OLD_LOFO),
          'total_function_seed_pairs':len(merged),
          'identical_donor_count':int(merged.identical_donor.sum()),
          'changed_donor_count':int((~merged.identical_donor).sum()),
          'same_donor_fraction':float(merged.identical_donor.mean()),
          'functions_with_any_changed_donor':int(merged.loc[~merged.identical_donor,'function_name'].nunique()),
          'note':'Donor equality alone never establishes identical latency input distributions.'}
    return info,merged


def self_test():
    functions=['a','b','c']
    a=pd.DataFrame({'function_name':functions,'method':'mA','seed':'11',
        'coverage':[True,False,True],
        'arch_agreement_covered_EVAL_ONLY':[1.,math.nan,0.],
        'regret_percent_covered_EVAL_ONLY':[0.,math.nan,10.],
        'predicted_architecture_or_abstain':['amd64','ABSTAIN','arm64']})
    b=pd.DataFrame({'function_name':functions,'method':'mB','seed':'11',
        'coverage':[True,True,False],
        'arch_agreement_covered_EVAL_ONLY':[0.,1.,math.nan],
        'regret_percent_covered_EVAL_ONLY':[5.,0.,math.nan],
        'predicted_architecture_or_abstain':['arm64','arm64','ABSTAIN']})
    # Test intersection semantics with a 59-function fixture, not by weakening production assertion.
    rest=[f'dummy-{i}' for i in range(56)]
    for q in (a,b):
        addon=pd.DataFrame({'function_name':rest,'method':q.method.iloc[0],'seed':'11',
            'coverage':False,'arch_agreement_covered_EVAL_ONLY':math.nan,
            'regret_percent_covered_EVAL_ONLY':math.nan,
            'predicted_architecture_or_abstain':'ABSTAIN'})
        q=pd.concat([q,addon],ignore_index=True)
        if q.method.iloc[0]=='mA':aa=q
        else:bb=q
    x=pd.concat([aa,bb],ignore_index=True)
    r=compare_shared(x,'mA','mB')
    require(r['common_covered']==1 and r['a_only']==1 and r['b_only']==1 and r['both_abstain']==56,
            'Intersection logic failure')
    require(r['architecture_disagreements_on_common']==1,'Disagreement logic failure')
    print('PASS self-test: common support, abstentions, conditional accuracy and paired regret')


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('mode',choices=['self-test','check','run'])
    args=pa.parse_args()
    if args.mode=='self-test':return self_test()
    manifest,direct,donor,sources,consensus,wide=load_inputs()
    require(coerce_bool(consensus.unanimous_across_seeds).groupby(consensus.method).sum().to_dict()==
            {'majority_diagnostic':54,'weighted_frozen':58},'Seed consensus inconsistent with frozen run')
    support=common_support(direct)
    old_info,mapping=old_donor_comparison(donor)
    notes={'static_csv_sha256':sha256(STATIC_CSV),
           'historical_source_digest_available':False,
           'static_sources_now_readable':int(coerce_bool(sources.source_exists_now).sum()),
           'gcp_100_sha256':sha256(GCP_100),
           'old_replay_raw_x86_path':str(OLDER/'raw/x86/all_samples.jsonl'),
           'old_replay_raw_arm_path':str(OLDER/'raw/arm64/all_samples.jsonl'),
           'historical_replay_uses_different_input_root':True,
           'important':'The replay historical raw and GCP-100 profiling are not proven sample-identical. A new same-corpus replay is required for a definitive matched-data comparison.',
           'donor_mapping_comparison':old_info,
           'pipeline_manifest_sha256':sha256(HERE/'manifest.json')}
    print('PASS frozen files verified, source static hash unchanged, 59 target functions, 5 seeds, both RF thresholds')
    print('Static source files readable now:',notes['static_sources_now_readable'],'/59; historical individual SHA: unavailable')
    print('\n===== SAME-COVERAGE SUMMARY (MEAN ACROSS FIVE SEEDS) =====')
    wanted=['method_a','method_b','common_covered','a_accuracy_common','b_accuracy_common',
            'a_mean_regret_common','b_mean_regret_common','a_minus_b_accuracy_common_pp']
    print(support.groupby(['method_a','method_b'],as_index=False)[wanted[2:]].mean(numeric_only=True).to_string(index=False))
    print('\n===== HISTORICAL DONOR MAPPING =====')
    for k,v in old_info.items():print(f'{k}: {v}')
    if not mapping.empty and (~mapping.identical_donor).any():
        print('\nFirst 12 changed function/seed donor mappings:')
        print(mapping[~mapping.identical_donor][['function_name','seed','old_donor','new_donor']].head(12).to_string(index=False))
    print('\n===== INTERPRETATION =====')
    print('Historical raw replay and latest GCP-100 use different source roots; do NOT combine them as if sample-identical.')
    print('No tuning, fitting, simulation or GCP usage has occurred in this audit.')
    if args.mode=='check':
        print('\nPASS CHECK: read-only, no output created')
        return
    require(not RESULT.exists(),f'Refusing to overwrite {RESULT}')
    RESULT.mkdir(parents=True,exist_ok=False)
    support.to_csv(RESULT/'common_coverage_per_seed.csv',index=False)
    if not mapping.empty:mapping.to_csv(RESULT/'historical_donor_mapping_comparison.csv',index=False)
    (RESULT/'audit_manifest.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    summ=support.groupby(['method_a','method_b'],as_index=False)[wanted[2:]].mean(numeric_only=True)
    summ.to_csv(RESULT/'common_coverage_summary.csv',index=False)
    md=['# Audit consolidato: Chen RF vs Serverledge (59 funzioni)','',
       '- Confronto condizionato a **intersezione di copertura per seed**: non sostituisce le coperture totali per metodo.',
       '- I cinque seed non equivalgono a cinque nuovi target indipendenti.',
       '- Nessuna scelta a posteriori del seed o del metodo.',
       '- Il CSV delle metriche statiche è identico allo storico, ma gli SHA individuali dei sorgenti storici non sono disponibili.',
       '- Il replay UCB1 storico e la nuova campagna GCP a 100 campioni non vanno considerati la stessa distribuzione senza verifica.',
       '', '## Confronto a copertura comune (media seed)', '',summ.to_markdown(index=False,floatfmt='.4f'),'',
       '## Confronto donor storico', '', '```json',json.dumps(old_info,indent=2,ensure_ascii=False),'```','']
    (RESULT/'EVIDENCE_AUDIT_REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    print('\nPASS RUN: audit artefacts created at',RESULT)

if __name__=='__main__':main()
