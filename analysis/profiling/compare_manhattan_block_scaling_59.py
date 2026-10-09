#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

DYN=['page_faults_delta','utilized_cpus','free_memory_mb','cpu_user_delta_ms','cpu_kernel_delta_ms']
STAT=['static_token_count_mean','static_function_count']
DIRECTIONS=('x86-preferred','arm-preferred')
VARIANTS=('manhattan_sqrt_current','manhattan_block_mean_1_over_p','manhattan_plain','euclidean_sqrt_control')
BASE='manhattan_sqrt_current'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root:Path):
    p=root/'resource/x86/function-profiles-median.csv'; s=root/'static-code-metrics.csv'; g=root/'ground_truth/preferences-2p5.csv'; l=root/'function-languages.csv'
    for x in [p,s,g,l]:
        if not x.is_file(): raise FileNotFoundError(x)
    P=pd.read_csv(p); S=pd.read_csv(s); G=pd.read_csv(g); L=pd.read_csv(l)
    df=P[['function_name',*DYN]].merge(S[['function_name','static_source_available',*STAT]],on='function_name',validate='one_to_one').merge(G[['function_name','architecture_preference','x86_duration_ms','arm_duration_ms']],on='function_name',validate='one_to_one').merge(L[['function_name','language']],on='function_name',validate='one_to_one')
    mask=df.static_source_available.astype(str).str.lower().isin(('true','1','yes'))
    for c in [*DYN,*STAT,'x86_duration_ms','arm_duration_ms']:
        df[c]=pd.to_numeric(df[c],errors='coerce'); mask &= np.isfinite(df[c].to_numpy(float))
    df=df[mask].sort_values('function_name').reset_index(drop=True)
    if len(df)!=59: raise ValueError(f'expected 59 functions, got {len(df)}')
    return df,{str(x.relative_to(root)):sha(x) for x in [p,s,g,l]}

def distances(variant:str, x_plain:np.ndarray, xt_plain:np.ndarray, x_sqrt:np.ndarray, xt_sqrt:np.ndarray)->np.ndarray:
    if variant=='manhattan_sqrt_current':
        return np.abs(x_sqrt-xt_sqrt[0]).sum(axis=1)
    if variant=='manhattan_plain':
        return np.abs(x_plain-xt_plain[0]).sum(axis=1)
    if variant=='manhattan_block_mean_1_over_p':
        delta=np.abs(x_plain-xt_plain[0])
        return delta[:,:len(DYN)].sum(axis=1)/len(DYN) + delta[:,len(DYN):].sum(axis=1)/len(STAT)
    if variant=='euclidean_sqrt_control':
        return np.sqrt(((x_sqrt-xt_sqrt[0])**2).sum(axis=1))
    raise ValueError(variant)

def choose(train,labels,target_label,dist):
    same=np.flatnonzero(labels==target_label)
    prefs=train.architecture_preference.to_numpy(str)
    eligible=same[np.isin(prefs[same],DIRECTIONS)]
    if not len(eligible): return -1,'','no_directional_member'
    votes={p:float(np.sum(1.0/(dist[eligible[prefs[eligible]==p]]+1e-9)**2)) for p in DIRECTIONS}
    if abs(votes[DIRECTIONS[0]]-votes[DIRECTIONS[1]])<=1e-15:return -1,'','tie'
    pred=max(DIRECTIONS,key=lambda p:votes[p])
    pool=eligible[prefs[eligible]==pred]
    names=train.function_name.to_numpy(str)
    order=np.lexsort((names[pool],dist[pool]))
    return int(pool[order[0]]),pred,'selected'

def evaluate(df,seeds,n_init):
    rows=[]
    with threadpool_limits(limits=1):
        for seed in seeds:
            t0=time.monotonic()
            for i in range(len(df)):
                target=df.iloc[[i]]; train=df.drop(index=i).reset_index(drop=True); t=target.iloc[0]
                scaler=MinMaxScaler()
                cols=DYN+STAT
                x_plain=scaler.fit_transform(train[cols].to_numpy(float)); xt_plain=scaler.transform(target[cols].to_numpy(float))
                scale=np.array([math.sqrt(len(DYN))]*len(DYN)+[math.sqrt(len(STAT))]*len(STAT))
                x_sqrt=x_plain/scale; xt_sqrt=xt_plain/scale
                km=KMeans(n_clusters=6,n_init=n_init,random_state=seed)
                labels=km.fit_predict(x_sqrt); tc=int(km.predict(xt_sqrt)[0])
                for variant in VARIANTS:
                    dist=distances(variant,x_plain,xt_plain,x_sqrt,xt_sqrt)
                    j,pred,status=choose(train,labels,tc,dist)
                    row={'variant':variant,'seed':seed,'target_function':t.function_name,'target_preference':t.architecture_preference,'status':status,'donor_function':'','prediction':pred,'directional':t.architecture_preference in DIRECTIONS,'raw_agreement':np.nan,'directional_agreement':np.nan,'regret_percent':np.nan,'log_speedup':np.nan,'same_language':np.nan}
                    if j>=0:
                        d=train.iloc[j]; tx=float(t.x86_duration_ms); ta=float(t.arm_duration_ms)
                        best_raw='x86-preferred' if tx<=ta else 'arm-preferred'; dbest='x86-preferred' if d.x86_duration_ms<=d.arm_duration_ms else 'arm-preferred'
                        chosen=tx if pred=='x86-preferred' else ta; alt=ta if pred=='x86-preferred' else tx
                        row.update(donor_function=d.function_name,raw_agreement=float(best_raw==dbest),directional_agreement=float(pred==t.architecture_preference) if row['directional'] else np.nan,regret_percent=(chosen/min(tx,ta)-1)*100 if row['directional'] else np.nan,log_speedup=math.log(alt/chosen) if row['directional'] else np.nan,same_language=float(t.language==d.language))
                    rows.append(row)
            print(f'LOFO seed={seed} complete in {time.monotonic()-t0:.1f}s',flush=True)
    return pd.DataFrame(rows)

def summarize(rows):
    out=[]
    for v,g in rows.groupby('variant',sort=True):
        sel=g[g.status=='selected']; dr=sel[sel.directional]
        out.append({'variant':v,'coverage':len(sel)/len(g),'raw_agreement':sel.raw_agreement.mean(),'directional_agreement':dr.directional_agreement.mean(),'regret_percent':dr.regret_percent.mean(),'geomean_speedup':float(np.exp(dr.log_speedup.mean())),'same_language':sel.same_language.mean()})
    return pd.DataFrame(out).sort_values('regret_percent').reset_index(drop=True)

def paired(rows,nboot,seed):
    ref=rows[rows.variant==BASE].set_index(['target_function','seed']); rng=np.random.default_rng(seed); out=[]
    for v,g in rows.groupby('variant'):
        if v==BASE:continue
        oth=g.set_index(['target_function','seed']); idx=ref.index.intersection(oth.index); a=ref.loc[idx];b=oth.loc[idx]; common=(a.status=='selected')&(b.status=='selected')
        donor_diff=(a.loc[common,'donor_function']!=b.loc[common,'donor_function']).sum(); pred_diff=(a.loc[common,'prediction']!=b.loc[common,'prediction']).sum()
        for metric in ['raw_agreement','directional_agreement','regret_percent','log_speedup']:
            valid=common&a[metric].notna()&b[metric].notna(); delta=(b.loc[valid,metric]-a.loc[valid,metric]).groupby(level=0).mean().to_numpy(float)
            bs=[]
            for start in range(0,nboot,1000):
                k=min(1000,nboot-start); draw=rng.integers(0,len(delta),size=(k,len(delta)));bs.extend(delta[draw].mean(axis=1).tolist())
            lo,hi=np.percentile(bs,[2.5,97.5])
            out.append({'comparison':f'{v}_minus_{BASE}','metric':metric,'n_targets':len(delta),'n_common_target_seeds':int(valid.sum()),'delta':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi),'donor_differences_common':int(donor_diff),'prediction_differences_common':int(pred_diff)})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--seeds',default='11,23,37,41,53');ap.add_argument('--n-init',type=int,default=50);ap.add_argument('--bootstrap-replicates',type=int,default=10000);ap.add_argument('--bootstrap-seed',type=int,default=20261009);args=ap.parse_args()
    seeds=tuple(int(x) for x in args.seeds.split(','));df,hashes=load(args.root);args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=evaluate(df,seeds,args.n_init); expected=59*len(seeds)*len(VARIANTS)
    if len(rows)!=expected:raise RuntimeError(f'expected {expected} rows, got {len(rows)}')
    summary=summarize(rows); pairs=paired(rows,args.bootstrap_replicates,args.bootstrap_seed)
    rows.to_csv(args.output_dir/'manhattan_scaling_59_per_target_seed.csv',index=False);summary.to_csv(args.output_dir/'manhattan_scaling_59_summary.csv',index=False);pairs.to_csv(args.output_dir/'manhattan_scaling_59_paired.csv',index=False)
    manifest={'input_hashes':hashes,'seeds':seeds,'k':6,'n_init':args.n_init,'features':DYN+STAT,'clustering_space':'MinMax train-only, dynamic/sqrt(5), static/sqrt(2)','variants':VARIANTS,'base':BASE,'bootstrap_replicates':args.bootstrap_replicates,'bootstrap_seed':args.bootstrap_seed,'note':'All donor-distance variants reuse exactly the same KMeans clusters; only donor voting/ranking distance changes. Ground truth target used posthoc only.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'PASS: 59 target, {len(VARIANTS)} variants, {len(rows)} LOFO rows')
    print(summary.to_string(index=False))
    print('\nPaired vs current:')
    print(pairs[pairs.metric.isin(['directional_agreement','regret_percent','log_speedup'])].to_string(index=False))
    print('Output:',args.output_dir)
if __name__=='__main__':main()
