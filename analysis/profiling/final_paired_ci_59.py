#!/usr/bin/env python3
"""Paired function-level uncertainty for the frozen Chen RF vs Serverledge comparison.

No models are fitted. No UCB1 replay, no new latency sampling and no GCP.
Bootstrap resamples 59 FUNCTION identities (not 5900 invocations and not
5*59 seed-function pairs). The same function-resample is used at all five
KMeans seeds and for both methods in the pair, preserving pairing.

The target corpus is a fixed benchmark, not an iid population; interpret
confidence intervals as descriptive sensitivity under a function-resampling
assumption, not as definitive population-wide statistical proof.

Place in analysis/profiling and run from repository root:
    python3 analysis/profiling/final_paired_ci_59.py self-test
    python3 analysis/profiling/final_paired_ci_59.py check
    python3 analysis/profiling/final_paired_ci_59.py run
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from confronto_statico_59 import load, get_seed_rows, SEEDS, N, METHODS

OUT = Path('data/architecture-comparison/comparison/paired-inference-59-v1')
# The second method is the candidate. A positive accuracy delta, or a positive
# regret reduction, favours the second method on their jointly covered targets.
PAIRS = [
    ('paper_RF_2p5_relabelled','weighted_frozen'),
    ('paper_RF_2p5_relabelled','majority_diagnostic'),
    ('majority_diagnostic','weighted_frozen'),
    ('paper_RF_15_original','weighted_frozen'),
]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def matrices(decisions):
    names=sorted(decisions.function_name.unique().tolist())
    if len(names)!=N:raise AssertionError('Expected 59 distinct function IDs')
    data={}
    for seed in SEEDS:
        q=get_seed_rows(decisions,seed)
        per={}
        for method in METHODS:
            z=q[q.method==method].set_index('function_name').loc[names]
            per[method]={
                'coverage':z.coverage.to_numpy(bool),
                'accuracy':z.arch_agreement_covered_EVAL_ONLY.to_numpy(float),
                'regret':z.regret_percent_covered_EVAL_ONLY.to_numpy(float),
            }
            if np.any(per[method]['coverage'] & ~np.isfinite(per[method]['regret'])):
                raise AssertionError('Missing regret on covered row')
        data[seed]=per
    return names,data


def pair_arrays(data,a,b):
    mask=[];diff_acc=[];diff_reg=[]
    for seed in SEEDS:
        aa=data[seed][a];bb=data[seed][b]
        joint=aa['coverage'] & bb['coverage']
        mask.append(joint)
        # Positive means B has better accuracy, or B has lower regret.
        diff_acc.append(np.where(joint,bb['accuracy']-aa['accuracy'],0.0))
        diff_reg.append(np.where(joint,aa['regret']-bb['regret'],0.0))
    return np.asarray(mask,dtype=float),np.asarray(diff_acc),np.asarray(diff_reg)


def observed(mask,acc,reg):
    denom=mask.sum(axis=1)
    if np.any(denom==0): raise ValueError('Pair has a seed with zero common coverage')
    return (float(denom.mean()), float(np.mean((acc*mask).sum(axis=1)/denom)),
            float(np.mean((reg*mask).sum(axis=1)/denom)))


def bootstrap(mask,acc,reg,idx):
    # idx: (replicates, 59) resampled function identities. The same function
    # draw is used across every method and seed; seeds are NOT independent units.
    b=idx.shape[0]
    scores_acc=np.zeros((len(SEEDS),b),float)
    scores_reg=np.zeros((len(SEEDS),b),float)
    for si in range(len(SEEDS)):
        chosen_mask=mask[si][idx]
        den=chosen_mask.sum(axis=1)
        scores_acc[si]=np.divide((acc[si][idx]*chosen_mask).sum(axis=1),den,
                                 out=np.full(b,np.nan),where=den>0)
        scores_reg[si]=np.divide((reg[si][idx]*chosen_mask).sum(axis=1),den,
                                 out=np.full(b,np.nan),where=den>0)
    # Use only draws with nonzero jointly covered functions in every seed.
    valid=np.isfinite(scores_acc).all(axis=0)&np.isfinite(scores_reg).all(axis=0)
    if valid.mean()<.95: raise ValueError('Too few valid bootstrap samples')
    return scores_acc[:,valid].mean(axis=0),scores_reg[:,valid].mean(axis=0),float(valid.mean())


def compute(decisions,reps=10000,seed=20261027):
    names,data=matrices(decisions)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,N,size=(reps,N))
    rows=[]
    for a,b in PAIRS:
        mask,acc,reg=pair_arrays(data,a,b)
        coverage,dacc,dreg=observed(mask,acc,reg)
        da,dr,valid=bootstrap(mask,acc,reg,idx)
        rows.append({
            'method_a':a, 'method_b':b,
            'mean_common_functions_per_seed':coverage,
            'accuracy_b_minus_a_pp':dacc*100,
            'accuracy_delta_ci95_low_pp':float(np.quantile(da,.025))*100,
            'accuracy_delta_ci95_high_pp':float(np.quantile(da,.975))*100,
            'regret_reduction_a_minus_b_pp':dreg,
            'regret_reduction_ci95_low_pp':float(np.quantile(dr,.025)),
            'regret_reduction_ci95_high_pp':float(np.quantile(dr,.975)),
            'bootstrap_valid_fraction':valid,
            'bootstrap_functions':N,
            'bootstrap_replicates':reps,
            'bootstrap_seed':seed,
        })
    return pd.DataFrame(rows)


def self_test():
    n=59
    mask=np.ones((len(SEEDS),n))
    # B improves classification by 1 on 30 of 59 functions, and reduces regret
    # by 10 pp on all. Preserve exactly one cluster per 59 function identities.
    acc=np.zeros((len(SEEDS),n));acc[:,:30]=1
    reg=np.full((len(SEEDS),n),10.0)
    cov,aa,rr=observed(mask,acc,reg)
    assert cov==59 and np.isclose(aa,30/59) and rr==10
    rng=np.random.default_rng(17)
    idx=rng.integers(0,n,size=(250,n))
    ba,br,valid=bootstrap(mask,acc,reg,idx)
    assert valid==1 and len(ba)==250 and np.allclose(br,10)
    assert np.std(ba)>0
    # Only conditional common support is evaluated.
    mask[:,-1]=0;acc[:,-1]=50;reg[:,-1]=1000
    cov,aa,rr=observed(mask,acc,reg)
    assert cov==58 and np.isclose(aa,30/58) and rr==10
    print('PASS self-test: paired target bootstrap, fixed cluster seeds, abstention exclusion')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['self-test','check','run'])
    p.add_argument('--replicates',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261027)
    args=p.parse_args()
    if args.command=='self-test':return self_test()
    if args.replicates<100: raise ValueError('At least 100 bootstrap replicates required')
    decisions,manifest_path=load() # validates frozen CSV and SHA
    result=compute(decisions,args.replicates,args.seed)
    show=['method_a','method_b','mean_common_functions_per_seed',
          'accuracy_b_minus_a_pp','accuracy_delta_ci95_low_pp','accuracy_delta_ci95_high_pp',
          'regret_reduction_a_minus_b_pp','regret_reduction_ci95_low_pp','regret_reduction_ci95_high_pp']
    print('===== PAIRED 59-FUNCTION CI (B MINUS A ACCURACY; A MINUS B REGRET) =====')
    print(result[show].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('NOTE: intervals resample functions, not 5 seeds or 5900 invocations;')
    print('      related language variants may violate iid assumptions.')
    if args.command=='check':
        print('PASS CHECK: frozen files unchanged, no output written')
        return
    if OUT.exists():raise SystemExit(f'Refusing overwrite: {OUT}')
    OUT.mkdir(parents=True,exist_ok=False)
    csv=OUT/'paired_accuracy_regret_ci95.csv'
    result.to_csv(csv,index=False)
    (OUT/'manifest.json').write_text(json.dumps({
        'source_manifest':str(manifest_path),'source_manifest_sha256':sha256(manifest_path),
        'output_sha256':sha256(csv),'function_count':N,'kmeans_seeds':list(SEEDS),
        'bootstrap_replicates':args.replicates,'bootstrap_seed':args.seed,
        'resampling_unit':'whole function, paired across methods and frozen seeds',
        'interpretation':'Exploratory uncertainty on fixed related workload corpus; not population proof',
        'no_model_training':True,'no_ucb1_replay':True,'no_gcp':True,
    },indent=2)+'\n',encoding='utf8')
    print('PASS RUN:',OUT)

if __name__=='__main__':main()
