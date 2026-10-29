#!/usr/bin/env python3
"""Compare frozen Chen RF decisions (15%, 2.5%) with frozen Serverledge LOFO.

All inputs read-only. Does NOT fit a classifier/regressor or tune KMeans:
uses KMeans K=6, n_init=50, seeds=11/23/37/41/53 as frozen.
Uses x86 profiling medians + historical static metrics on the SAME common59.
Serverledge frozen path: block-sqrt 5+2 KMeans geometry, *plain* 7D
MinMax Manhattan donor distance, inverse distance squared directional vote,
nearest donor belonging to the voted class. Synthetic funcs are targets but
are NEVER donor candidates. Original code reference:
compare_59_clustering_and_plots_final.py (previous final validation).

Also reports a separately labelled unweighted cluster directional majority,
NOT part of the frozen final donor policy. Abstention is NOT an error and is
never silently mapped to x86 or ARM. Only observed target ARM times may be
used AFTER prediction, for evaluation.

Run from Serverledge root after placing under analysis/profiling/:
    python3 analysis/profiling/compare_frozen_rf_vs_serverledge_59.py self-test
    python3 analysis/profiling/compare_frozen_rf_vs_serverledge_59.py check
    python3 analysis/profiling/compare_frozen_rf_vs_serverledge_59.py run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from paper_original_prediction_pipeline import DATASET_ROOT, OUTPUT_ROOT as PAPER_ROOT, load_inputs, digest

RF25_DIR = Path('data/architecture-comparison/diagnostics/fixed-rf-label-thresholds-v1')
OUT = Path('data/architecture-comparison/comparison/frozen-rf-vs-serverledge-59-v1')
DYNAMIC = ['pageFaultsDelta','utilizedCPUs','freeMemory','cpuUserDelta','cpuKernelDelta']
STATIC = ['static_token_count_mean','static_function_count']
FEATURES = DYNAMIC+STATIC
SEEDS = (11,23,37,41,53)
SYNTHETIC = {'amd_faster','arm_faster'}
DIRECTION = ('x86-preferred','arm-preferred')
EPS = 1e-9
RF_MAP = {'ARM-faster':'arm-preferred','ARM-slower':'x86-preferred','ARM-similar':'architecture-independent'}
PRED_MAP = {'arm-preferred':'arm64','x86-preferred':'amd64','architecture-independent':'ABSTAIN'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha_check_file(path, recorded):
    need(path.is_file(), f'Missing file: {path}')
    need(digest(path)==recorded, f'Input changed after frozen run: {path}')


def verify_frozen_csv(folder, manifest_file, csv_name):
    manifest_path = folder / manifest_file
    need(manifest_path.is_file(), f'Missing manifest {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    need(csv_name in manifest['outputs'], f'{csv_name}: missing manifest record')
    sha_check_file(folder/csv_name,manifest['outputs'][csv_name]['sha256'])
    return pd.read_csv(folder/csv_name,low_memory=False)


def load_checked():
    data, truth, _, _ = load_inputs(DATASET_ROOT)  # source files validated by existing code
    # 2.5% derivative must trace back to EXACT SAME frozen 15% model and GCP dataset.
    sensitivity_manifest=json.loads((RF25_DIR/'manifest.json').read_text(encoding='utf8'))
    need(sensitivity_manifest.get('original_output_sha256')==digest(PAPER_ROOT/'paper_model_manifest.json'),
         '2.5% sensitivity refers to a different frozen original RF manifest')
    need(sensitivity_manifest.get('source_dataset_sha256')==digest(DATASET_ROOT/'dataset_manifest.json'),
         '2.5% sensitivity refers to a different GCP dataset manifest')
    p15 = verify_frozen_csv(PAPER_ROOT, 'paper_model_manifest.json', 'paper_predictions_by_function_and_strategy.csv')
    p25 = verify_frozen_csv(RF25_DIR, 'manifest.json', 'classifier_2p5_function_predictions.csv')
    p15 = p15.loc[p15.strategy=='arm-speed', ['function_name','paper_predicted_class_15']].copy()
    p25 = p25[['function_name','predicted_class_2p5']].copy()
    for frame,name in ((p15,'15% RF'),(p25,'2.5% RF')):
        need(len(frame)==59 and frame.function_name.nunique()==59, f'{name}: wrong function cardinality')
        need(set(frame.function_name)==set(truth.function_name),f'{name}: function set mismatch')
    for col in (DYNAMIC+STATIC):
        need(col in data, f'Missing feature {col}')
        v=pd.to_numeric(data[col],errors='coerce')
        need(np.isfinite(v.to_numpy(float)).all(),f'Non-finite {col}')
        data[col]=v
    x86=data.loc[data.machine_tag=='amd64'].copy()
    need(len(x86)==5900,'Not exactly 5900 x86 samples')
    need((x86.groupby('function_name').size()==100).all(),'Wrong x86 samples per function')
    need((x86.groupby('function_name')[STATIC].nunique()==1).all().all(),'Static metrics vary within function')
    med=x86.groupby('function_name',sort=True)[FEATURES].median().reset_index()
    meta_cols=['function_name','language','source_resolution','source_path','static_source_available']
    need(set(meta_cols).issubset(x86.columns),'Missing static provenance columns')
    for c in meta_cols[1:]:
        need((x86.groupby('function_name')[c].nunique(dropna=False)==1).all(),f'{c}: heterogeneous within function')
    meta=x86.groupby('function_name',sort=True)[meta_cols[1:]].first().reset_index()
    merged=med.merge(meta,on='function_name',validate='one_to_one').merge(
        truth[['function_name','serverledge_preference_2p5','paper_class_15',
               'x86_median_duration_ms','arm_median_duration_ms','arm_vs_x86_delta_percent',
               'measured_faster_architecture']],on='function_name',validate='one_to_one'
        ).merge(p15,on='function_name',validate='one_to_one').merge(p25,on='function_name',validate='one_to_one')
    merged=merged.sort_values('function_name').reset_index(drop=True)
    need(len(merged)==59,'Missing cohort rows')
    need(merged[FEATURES].to_numpy(float).shape==(59,7),'Missing X matrix')
    need((merged.paper_class_15==merged.paper_predicted_class_15).sum()==30,'Frozen 15% RF accuracy unexpected')
    need((merged.predicted_class_2p5.map(RF_MAP)==merged.serverledge_preference_2p5).sum()==24,
         'Frozen 2.5% RF accuracy unexpected')
    need(set(merged.serverledge_preference_2p5)==set(DIRECTION)|{'architecture-independent'},'Class labels unexpected')
    assert float(np.nanmin(merged[STATIC].to_numpy(float)))>=0
    return merged


def source_provenance(df):
    rows=[]
    for row in df.itertuples(index=False):
        src=str(row.source_path)
        path=Path(src)
        candidates=[path,Path.cwd()/path]
        found=next((p for p in candidates if p.is_file()),None)
        rows.append({'function_name':row.function_name,
                     'static_token_count_mean':row.static_token_count_mean,
                     'static_function_count':row.static_function_count,
                     'source_resolution':row.source_resolution,
                     'source_path_recorded':src,
                     'source_exists_now':bool(found),
                     'source_sha256_NOW_ONLY':digest(found) if found else '',
                     'historical_source_sha_verified':False,
                     'note':'Historical source hash not in this audit; existence does NOT prove identity.'})
    return pd.DataFrame(rows)


def vote_distance(labels, prefs, distances, names, target_cluster, method):
    """Frozen C vote with optionally labelled, unweighted-majority diagnostic."""
    idx=np.flatnonzero(labels==target_cluster)
    idx=np.array([i for i in idx if str(names[i]) not in SYNTHETIC],dtype=int)
    if not len(idx):return None,'ABSTAIN','no_admissible_cluster_member',float('nan'),0
    good=idx[np.isin(prefs[idx],DIRECTION)]
    if not len(good):return None,'ABSTAIN','no_directional_member',float('nan'),0
    score={c:0.0 for c in DIRECTION}
    for c in DIRECTION:
        group=good[prefs[good]==c]
        score[c]=float((1/((distances[group]+EPS)**2)).sum()) if method=='weighted' else float(len(group))
    if math.isclose(score[DIRECTION[0]],score[DIRECTION[1]],rel_tol=0,abs_tol=1e-15):
        return None,'ABSTAIN','tied_vote',float('nan'),len(good)
    winner=max(DIRECTION,key=lambda c:score[c])
    pool=good[prefs[good]==winner]
    order=np.lexsort((names[pool],distances[pool]))
    donor=int(pool[order[0]])
    return donor,winner,'selected',float(distances[donor]),len(good)


def single_seed(df, seed):
    out=[]
    cols=FEATURES
    for i in range(len(df)):
        train=df.drop(index=i).reset_index(drop=True)
        target=df.iloc[[i]]
        need(target.iloc[0].function_name not in set(train.function_name),'Target leaked into train')
        # Frozen scaler: fit TRAIN ONLY; plain 7D coordinates retained for donor Manhattan.
        scaler=MinMaxScaler()
        plain=scaler.fit_transform(train[cols].to_numpy(dtype=float))
        tp=scaler.transform(target[cols].to_numpy(dtype=float))
        # Frozen KMeans geometry: block-sqrt reweight only for clustering.
        cluster=plain.copy(); tcluster=tp.copy()
        cluster[:,:5] /= math.sqrt(5);cluster[:,5:] /= math.sqrt(2)
        tcluster[:,:5] /= math.sqrt(5);tcluster[:,5:] /= math.sqrt(2)
        model=KMeans(n_clusters=6,n_init=50,random_state=int(seed))
        labels=model.fit_predict(cluster)
        cluster_id=int(model.predict(tcluster)[0])
        distances=np.abs(plain-tp[0]).sum(axis=1)  # PLAIN 7D Manhattan, NO block scaling
        names=train.function_name.to_numpy(dtype=str)
        prefs=train.serverledge_preference_2p5.to_numpy(dtype=str)
        for method in ('majority_diagnostic','weighted_frozen'):
            donor, pref, status, dist, eligible=vote_distance(
                labels,prefs,distances,names,cluster_id,
                'weighted' if method=='weighted_frozen' else 'majority')
            donor_name='' if donor is None else str(names[donor])
            if donor_name in SYNTHETIC or donor_name==str(target.iloc[0].function_name):
                raise AssertionError('Disallowed donor or LOFO leak')
            out.append({'function_name':str(target.iloc[0].function_name),'seed':int(seed),
                        'method':method,'status':status,'predicted_preference':pref,
                        'predicted_architecture_or_abstain':PRED_MAP.get(pref,'ABSTAIN'),
                        'selected_donor':donor_name,'cluster_id':cluster_id,
                        'donor_distance_plain_L1':dist,'eligible_directional_members':eligible,
                        'donor_true_preference_TRAIN_ONLY':str(prefs[donor]) if donor is not None else ''})
    return out


def to_eval(df, truth):
    z=df.merge(truth[['function_name','serverledge_preference_2p5',
           'measured_faster_architecture','x86_median_duration_ms','arm_median_duration_ms',
           'arm_vs_x86_delta_percent']],on='function_name',validate='many_to_one')
    # Ground truth is accessed only after ALL decisions have already been formed.
    best=np.minimum(z.x86_median_duration_ms.to_numpy(float),z.arm_median_duration_ms.to_numpy(float))
    selected=np.select([z.predicted_architecture_or_abstain.eq('amd64'),z.predicted_architecture_or_abstain.eq('arm64')],
                       [z.x86_median_duration_ms,z.arm_median_duration_ms],default=np.nan)
    z['coverage']=z.predicted_architecture_or_abstain!='ABSTAIN'
    z['regret_percent_covered_EVAL_ONLY']=(selected/best-1)*100
    z['arch_agreement_covered_EVAL_ONLY']=np.where(z.coverage,
          z.predicted_architecture_or_abstain==z.measured_faster_architecture,np.nan)
    z['directional_target_EVAL_ONLY']=z.serverledge_preference_2p5.isin(DIRECTION)
    z['directional_class_correct_covered_EVAL_ONLY']=np.where(z.coverage&z.directional_target_EVAL_ONLY,
          z.predicted_preference==z.serverledge_preference_2p5,np.nan)
    return z


def per_method_summary(df):
    rows=[]
    for (method,seed), z in df.groupby(['method','seed'],dropna=False,sort=True):
        covered=z[z.coverage]
        directional=z[z.directional_target_EVAL_ONLY]
        dcovered=directional[directional.coverage]
        rows.append({'method':method,'seed':seed,'target_count':len(z),
                     'covered':len(covered),'coverage_fraction':len(covered)/len(z),
                     'directional_covered':len(dcovered),'directional_target_count':len(directional),
                     'directional_coverage_fraction':len(dcovered)/len(directional),
                     'directional_accuracy_on_covered':float(dcovered.directional_class_correct_covered_EVAL_ONLY.mean()) if len(dcovered) else np.nan,
                     'arch_accuracy_on_covered':float(covered.arch_agreement_covered_EVAL_ONLY.mean()) if len(covered) else np.nan,
                     'mean_regret_percent_on_covered':float(covered.regret_percent_covered_EVAL_ONLY.mean()) if len(covered) else np.nan,
                     'median_regret_percent_on_covered':float(covered.regret_percent_covered_EVAL_ONLY.median()) if len(covered) else np.nan})
    return pd.DataFrame(rows)


def rf_decisions(df):
    frames=[]
    for method,col in [('paper_RF_15_original','paper_predicted_class_15'),
                       ('paper_RF_2p5_relabelled','predicted_class_2p5')]:
        z=df[['function_name',col]].copy()
        z['method']=method;z['seed']='fixed_RF';
        z['predicted_preference']=z[col].map(RF_MAP)
        z['predicted_architecture_or_abstain']=z.predicted_preference.map(PRED_MAP)
        z['status']=np.where(z.predicted_architecture_or_abstain=='ABSTAIN','class_similar','selected')
        z['selected_donor']='';z['donor_distance_plain_L1']=np.nan
        z=z.drop(columns=[col])
        frames.append(z)
    return pd.concat(frames,ignore_index=True)


def per_function_seed_consensus(donor):
    rows=[]
    for (name,method),z in donor.groupby(['function_name','method'],sort=True):
        row={'function_name':name,'method':method,'seeds_tested':len(z)}
        for arm in ('amd64','arm64','ABSTAIN'):
            row[f'seed_count_{arm}']=int((z.predicted_architecture_or_abstain==arm).sum())
        row['unanimous_across_seeds']=int(z.predicted_architecture_or_abstain.nunique()==1)
        rows.append(row)
    return pd.DataFrame(rows)


def run_preflight():
    data=load_checked()
    sources=source_provenance(data)
    print(f'PASS cohort: {len(data)} functions, 7 features (PAPER5+STATIC2), original 15% RF and fixed 2.5% RF frozen')
    print('Static metric summary:')
    print(data[STATIC].describe().loc[['min','50%','max']].to_string())
    print('Historical source paths currently readable:',int(sources.source_exists_now.sum()),'/59')
    print('Explicit overrides:',data.loc[data.source_resolution=='explicit-repository-override','function_name'].tolist())
    print('NOTE: source presence and static-value consistency cannot prove historical source byte identity.')
    print('15% and 2.5% RF direct coverage:',int((data.paper_predicted_class_15!='ARM-similar').sum()),
          int((data.predicted_class_2p5!='ARM-similar').sum()))
    return data,sources


def self_test():
    names=np.array(['A','B','C','D','amd_faster']);prefs=np.array([
          'arm-preferred','x86-preferred','architecture-independent','arm-preferred','arm-preferred'])
    label=np.zeros(5,dtype=int); dist=np.array([1.,3.,.2,2.,.01])
    donor,pred,status,d,_=vote_distance(label,prefs,dist,names,0,'weighted')
    need(pred=='arm-preferred' and donor==0,'Wrong frozen weighted vote/donor / synthetic not excluded')
    donor,pred,status,d,_=vote_distance(label,prefs,dist,names,0,'majority')
    need(pred=='arm-preferred' and donor==0,'Wrong diagnostic majority')
    p=MinMaxScaler().fit(np.array([[0,2],[1,3],[2,4]]))
    need(np.allclose(p.transform([[1,3]]),[[0.5,0.5]]),'Scaler smoke failure')
    need(PRED_MAP['architecture-independent']=='ABSTAIN','Neutral class must abstain')
    print('PASS self-test: donor exclusion, voting formulas, scalers, abstention')


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('command',choices=['self-test','check','run'])
    args=pa.parse_args()
    if args.command=='self-test':return self_test()
    cohort,sources=run_preflight()
    if args.command=='check':
        print('PASS CHECK read-only, no model trained, no files created')
        return
    need(not OUT.exists(),f'Refusing to overwrite existing output: {OUT}')
    rows=[]
    for seed in SEEDS:
        print(f'LOFO frozen Serverledge K=6, seed={seed}...',flush=True)
        rows.extend(single_seed(cohort,seed))
    donor=pd.DataFrame(rows)
    need(len(donor)==2*59*len(SEEDS),'Wrong Serverledge cardinality')
    rf=rf_decisions(cohort)
    decisions=pd.concat([rf,donor],ignore_index=True,sort=False)
    scored=to_eval(decisions,cohort)
    by_seed=per_method_summary(scored)
    summary=by_seed.groupby('method',as_index=False).agg({
        'target_count':'mean','coverage_fraction':'mean','directional_coverage_fraction':'mean',
        'directional_accuracy_on_covered':'mean','arch_accuracy_on_covered':'mean',
        'mean_regret_percent_on_covered':'mean','median_regret_percent_on_covered':'mean'})
    consensus=per_function_seed_consensus(donor)
    # One row per function: easy to compare with the frozen RF decisions.
    wide=cohort[['function_name','paper_class_15','serverledge_preference_2p5',
        'measured_faster_architecture','arm_vs_x86_delta_percent',
        'paper_predicted_class_15','predicted_class_2p5',
        'x86_median_duration_ms','arm_median_duration_ms',
        'static_token_count_mean','static_function_count']].copy()
    wide['rf15_architecture_or_abstain']=wide.paper_predicted_class_15.map(RF_MAP).map(PRED_MAP)
    wide['rf2p5_architecture_or_abstain']=wide.predicted_class_2p5.map(RF_MAP).map(PRED_MAP)
    for method,prefix in [('weighted_frozen','serverledge_frozen'),
                          ('majority_diagnostic','majority_diagnostic')]:
        sub=donor[donor.method==method].copy()
        info=per_function_seed_consensus(sub)
        for col in ('seed_count_amd64','seed_count_arm64','seed_count_ABSTAIN','unanimous_across_seeds'):
            wide=wide.merge(info[['function_name',col]].rename(columns={col:f'{prefix}_{col}'}),
                            on='function_name',validate='one_to_one')
        donor_lists=sub.sort_values('seed').groupby('function_name').agg(
            donor_by_seed=('selected_donor',lambda s:';'.join(s.fillna(''))),
            architecture_by_seed=('predicted_architecture_or_abstain',lambda s:';'.join(s.fillna(''))))
        donor_lists=donor_lists.rename(columns={c:f'{prefix}_{c}' for c in donor_lists.columns}).reset_index()
        wide=wide.merge(donor_lists,on='function_name',validate='one_to_one')
    need(len(wide)==59 and wide.function_name.is_unique,'Bad per-function comparison table')
    source_input_files = [DATASET_ROOT/'dataset_manifest.json',PAPER_ROOT/'paper_model_manifest.json',
                 RF25_DIR/'manifest.json']
    manifest={'role':'Frozen RF classification decisions vs original final Serverledge clustering/donor decisions',
              'no_tuning':True,'gcp_recollection':False,'primary_ground_truth':'GCP independent duration_ms, common59',
              'rf_15':'original author 15% labels; abstain if similar',
              'rf_2p5':'same author params, separately relabelled 2.5% training; abstain if similar',
              'serverledge':'frozen K6 PAPER5+STATIC2 blocksqrt KMeans; plain 7D Manhattan donor vote 1/(d+1e-9)^2',
              'serverledge_unweighted':'unweighted directional majority DIAGNOSTIC ONLY, not final frozen algorithm',
              'k':6,'n_init':50,'seeds':list(SEEDS),
              'excluded_donors':sorted(SYNTHETIC),'include_synthetic_as_targets':True,
              'no_ground_truth_used_for_target_prediction':True,
              'static_source_identity_note':'Not proven: historical source-byte digests not available',
              'input_sha256':{str(p):digest(p) for p in source_input_files},'outputs':{}}
    OUT.mkdir(parents=True,exist_ok=False)
    frames={
      'source_provenance_check.csv':sources,
      'common59_features_and_truth.csv':cohort,
      'serverledge_lofo_by_target_seed.csv':donor,
      'direct_decisions_all_methods.csv':scored,
      'metrics_per_seed.csv':by_seed,
      'metrics_summary.csv':summary,
      'serverledge_seed_consensus_per_function.csv':consensus,
      'function_comparison_summary_59.csv':wide,
    }
    for filename,frame in frames.items():
        frame.to_csv(OUT/filename,index=False)
        manifest['outputs'][filename]={'rows':len(frame),'sha256':digest(OUT/filename)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    notes=['# Confronto architetturale — RF Chen e Serverledge', '',
      '- Cohort comune: 59 funzioni, profiling GCP, 100 misure warm per architettura.',
      '- Paper originale: RF 15% congelata. Variante RF 2,5%: stessi iperparametri, etichette diverse, NON originale.',
      '- Serverledge: KMeans K=6; n_init=50; seed 11,23,37,41,53; LOFO;',
      '  MinMax fit su training; blocchi dynamic/sqrt(5), static/sqrt(2) per KMeans;',
      '  Manhattan semplice 7D per donor; voto direzionale pesato 1/(d+1e-9)^2;',
      '  nearest donor nella classe predetta; amd_faster e arm_faster esclusi dai donor, ammessi come target.',
      '- La maggioranza non pesata e un riferimento DIAGNOSTICO, non la politica finale congelata.',
      '- I casi neutrali/di astensione NON sono conteggiati come errori; coverage e accuracy condizionata sono riportate separatamente.',
      '- Le 5 repliche seed non sono 295 funzioni indipendenti: unita di valutazione = funzione.',
      '- Nessun dato ARM della target entra nel training, nella normalizzazione o nella scelta del donor.',
      '- Verifica statica: valori numerici disponibili; i file sorgente attuali possono essere controllati,',
      '  ma senza digest storici non si puo attestare identita byte-per-byte.', '',
      '## Riepilogo', '', '```', summary.to_string(index=False), '```', '',
      '## File principali', '',
      '- `function_comparison_summary_59.csv`: tabella unica, decisioni per seed, donor.',
      '- `direct_decisions_all_methods.csv`: tutte le decisioni valutate con ground truth GCP.',
      '- `metrics_per_seed.csv` / `metrics_summary.csv`: coverage, correttezza e regret condizionati.',
      '- `source_provenance_check.csv`: verifica di disponibilita dei sorgenti.',
      '- `manifest.json`: SHA e protocollo congelato.', '']
    (OUT/'COMPARISON_REPORT.md').write_text('\n'.join(notes),encoding='utf8')
    print('\n===== COMPARISON: ABSTENTION EXCLUDED FROM CONDITIONAL ACCURACY / REGRET =====')
    print(summary.to_string(index=False))
    print('\n===== SERVERLEDGE SEED CONSENSUS =====')
    print(consensus.groupby('method')['unanimous_across_seeds'].agg(['sum','count']).to_string())
    print('\nPASS: comparison frozen, outputs:', OUT)


if __name__=='__main__':
    main()
