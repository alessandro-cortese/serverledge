#!/usr/bin/env python3
"""Offline UCB1 replay for donor-distance variants on the same 59 targets.

Reads donor mappings from compare_manhattan_block_scaling_59.py and keeps the
Transfer Learning configuration frozen to the final Difference-tuned setup:
  w_R=0.05, w_E=64, c=0.2, H=10 by default.

Requires the existing Serverledge analysis modules in analysis/profiling.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.profiling.transfer_prior_formula_comparison import (
    load_eligible_durations, build_performance, make_crn_samples, build_prior
)
from analysis.profiling.transfer_parameter_tuning_cv import (
    Config, simulate_baseline_trace, simulate_tl_trace,
    prefix_metrics_baseline, prefix_metrics_tl
)

VARIANTS = (
    'manhattan_sqrt_current',
    'manhattan_block_mean_1_over_p',
    'manhattan_plain',
    'euclidean_sqrt_control',
)
BASE='manhattan_sqrt_current'
METRICS=['gain_percent','optimal_arm_rate','wrong_selections',
         'cumulative_reward_pseudo_regret','cumulative_latency_ms','first_arm_optimal']

def stable_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'big') % 2**32

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--donor-detail',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--replicates',type=int,default=200)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--anchor-samples',type=int,default=10)
    p.add_argument('--mc-seed',type=int,default=2026100911)
    p.add_argument('--bootstrap-replicates',type=int,default=5000)
    args=p.parse_args()

    detail=pd.read_csv(args.donor_detail).fillna({'donor_function':''})
    seeds=sorted(detail.seed.unique())
    targets=sorted(detail.target_function.unique())
    if len(targets)!=59 or len(seeds)!=5:
        raise ValueError(f'Expected 59 targets and 5 seeds, got {len(targets)} / {len(seeds)}')

    lookups={}
    for variant in VARIANTS:
        for seed in seeds:
            r=detail[(detail.variant==variant)&(detail.seed==seed)]
            if len(r)!=59 or r.target_function.duplicated().any():
                raise ValueError((variant,seed,len(r)))
            lookups[(variant,seed)]={
                str(row.target_function): (str(row.donor_function) if row.status=='selected' else None)
                for row in r.itertuples()
            }

    raw_x=args.root/'raw/x86/all_samples.jsonl'
    raw_a=args.root/'raw/arm64/all_samples.jsonl'
    perf=build_performance(load_eligible_durations(raw_x),load_eligible_durations(raw_a))
    if not set(targets)<=set(perf):
        raise ValueError('Missing target functions in raw replay corpus')

    out=[]
    for i,target in enumerate(targets,1):
        anchor,online=make_crn_samples(
            target=target,target_stats=perf[target],replicates=args.replicates,
            max_horizon=args.horizon,anchor_n=args.anchor_samples,base_seed=args.mc_seed)

        baseline=[]
        for rep in range(args.replicates):
            seq={arm:online[arm][rep] for arm in ('x86','arm64')}
            trace=simulate_baseline_trace(target_stats=perf[target],sequences=seq,c=.8,convergence_run=5)
            baseline.append(prefix_metrics_baseline(trace,args.horizon))

        for seed in seeds:
            for variant in VARIANTS:
                donor=lookups[(variant,seed)][target]
                vals=[]
                for rep in range(args.replicates):
                    if donor is None:
                        m={**baseline[rep],'gain_percent':0.0}
                    else:
                        prior,_=build_prior(
                            mode='difference',anchor_mean_reward_x86=float(anchor[rep]),
                            donor_stats=perf[donor])
                        trace=simulate_tl_trace(
                            target_stats=perf[target],
                            sequences={arm:online[arm][rep] for arm in ('x86','arm64')},
                            prior_mean_reward=prior,
                            config=Config(config_id=variant,regime='frozen',w_r=.05,w_e=64.,c=.2),
                            convergence_run=5)
                        m=prefix_metrics_tl(trace,args.horizon)
                        m['gain_percent']=100.0*(baseline[rep]['cumulative_latency_ms']-m['cumulative_latency_ms'])/baseline[rep]['cumulative_latency_ms']
                    vals.append(m)

                rec={'variant':variant,'target_function':target,'seed':seed,
                     'donor_function':donor or '','tl_applied':int(bool(donor))}
                rec.update({key:float(np.mean([v[key] for v in vals])) for key in METRICS})
                out.append(rec)
        print(f'Replayed {i:02d}/59 {target}',flush=True)

    df=pd.DataFrame(out)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output_dir/'ucb1_manhattan_per_target_seed.csv',index=False)

    per_target=df.groupby(['variant','target_function'],as_index=False)[METRICS+['tl_applied']].mean()
    per_target.to_csv(args.output_dir/'ucb1_manhattan_per_target.csv',index=False)
    summary=per_target.groupby('variant',as_index=False)[METRICS+['tl_applied']].mean()
    summary.to_csv(args.output_dir/'ucb1_manhattan_summary.csv',index=False)

    base=per_target[per_target.variant==BASE].set_index('target_function').loc[targets]
    pairs=[]
    for variant in VARIANTS:
        if variant==BASE: continue
        other=per_target[per_target.variant==variant].set_index('target_function').loc[targets]
        rng=np.random.default_rng(stable_seed(variant,BASE,args.mc_seed))
        for metric in ['gain_percent','optimal_arm_rate','wrong_selections','cumulative_reward_pseudo_regret']:
            delta=(other[metric]-base[metric]).to_numpy(float)
            draws=rng.integers(0,len(delta),size=(args.bootstrap_replicates,len(delta)))
            means=delta[draws].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            pairs.append({'comparison':f'{variant}_minus_{BASE}','metric':metric,
                          'delta':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
                          'targets':len(delta)})
    paired=pd.DataFrame(pairs)
    paired.to_csv(args.output_dir/'ucb1_manhattan_paired_ci.csv',index=False)

    meta={
        'mapping_sha256':hashlib.sha256(args.donor_detail.read_bytes()).hexdigest(),
        'raw_x86_sha256':hashlib.sha256(raw_x.read_bytes()).hexdigest(),
        'raw_arm64_sha256':hashlib.sha256(raw_a.read_bytes()).hexdigest(),
        'variants':VARIANTS,'base':BASE,'replicates':args.replicates,'horizon':args.horizon,
        'anchor_samples':args.anchor_samples,'mc_seed':args.mc_seed,'cluster_seeds':list(map(int,seeds)),
        'formula':'Difference','w_R':.05,'w_E':64.,'c':.2,
        'note':'Post-hoc frozen replay with common random numbers; donor-distance rule changes only. Not an independent tuning experiment.'
    }
    (args.output_dir/'ucb1_manhattan_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    print('PASS: UCB1 donor-distance replay complete')
    print(summary.to_string(index=False))
    print('\nPaired vs current:')
    print(paired.to_string(index=False))
    print('Output:',args.output_dir)

if __name__=='__main__':
    main()
