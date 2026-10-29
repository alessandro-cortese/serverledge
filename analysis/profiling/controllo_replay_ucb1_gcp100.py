#!/usr/bin/env python3
"""Read-only check before reusing Serverledge's VALIDATED UCB1 replay engine on GCP-100.

Reports available per-call latency fields, the old loader's actual data shape,
validated simulator interfaces, and frozen new donor tables. Does not run,
tune, simulate, or rewrite anything.
"""
from __future__ import annotations
import csv
import inspect
import json
from pathlib import Path

import pandas as pd
from analysis.profiling.transfer_prior_formula_comparison import (
    load_eligible_durations, build_performance, make_crn_samples, build_prior)
from analysis.profiling.transfer_parameter_tuning_cv import (
    Config, simulate_baseline_trace, simulate_tl_trace,
    prefix_metrics_baseline, prefix_metrics_tl)

base=Path('data/architecture-comparison/datasets/v1')
prof=base/'profiling_common59_enriched.csv'
truth=base/'ground_truth_common59.csv'
old=Path('data/profiling/final-corpus64-20261005-analysis-01/raw/x86/all_samples.jsonl')
provenance=Path('data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final/paper-rf-provenance.csv')
original=Path('data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final/paper-rf-dataset.csv')
donors=Path('data/architecture-comparison/comparison/frozen-rf-vs-serverledge-59-v1/serverledge_lofo_by_target_seed.csv')

def show(value):
    s=repr(value)
    return s[:500]+(' ...' if len(s)>500 else '')

print('===== 1. NUOVI DATI: NOMI DELLE COLONNE =====')
for f in (prof,truth,provenance,original):
    print('\n',f, 'exists=',f.is_file())
    if f.is_file():
        with f.open(newline='',encoding='utf8') as h:
            header=next(csv.reader(h))
        print('columns=',header)
        print('potential duration fields=',[k for k in header if any(x in k.lower() for x in ('duration','latency','runtime','elapsed'))])

print('\n===== 2. PROFILING RUNTIME GCP-100 =====')
if prof.is_file():
    z=pd.read_csv(prof,usecols=['function_name','machine_tag','runtime'])
    print('pairs=',len(z.groupby(['function_name','machine_tag'])),
          'counts=',z.groupby(['function_name','machine_tag']).size().value_counts().to_dict())
    print('runtime min,max=',pd.to_numeric(z.runtime,errors='coerce').min(),pd.to_numeric(z.runtime,errors='coerce').max())

print('\n===== 3. VECCHIO LOADER VALIDATO: SHAPE E TIPI =====')
if old.is_file():
    loaded=load_eligible_durations(old)
    print('loader output type:',type(loaded).__name__)
    if isinstance(loaded,dict):
        print('number of function keys:',len(loaded))
        k=sorted(loaded,key=str)[0]
        item=loaded[k]
        print('first key:',show(k),'value type:',type(item).__name__)
        print('first value:',show(item[:2] if isinstance(item,(list,tuple)) else item))
    elif isinstance(loaded,pd.DataFrame):
        print('columns=',list(loaded.columns),'first=',loaded.head(2).to_dict('records'))
    elif isinstance(loaded,list):
        print('number of items:',len(loaded),'first=',show(loaded[:2]))
    else:
        print('first item unavailable for type:',type(loaded).__name__)
else:
    print('Legacy raw not found (it is only used for format inspection)')

print('\n===== 4. SIGNATURES REPLAY VALIDATO =====')
for fn in (build_performance,make_crn_samples,build_prior,Config,simulate_baseline_trace,
           simulate_tl_trace,prefix_metrics_baseline,prefix_metrics_tl):
    print(fn.__name__,inspect.signature(fn))

print('\n===== 5. NUOVI DONOR =====')
if donors.is_file():
    d=pd.read_csv(donors)
    print('columns=',list(d.columns))
    print('methods=',d.method.value_counts().to_dict())
    print('seed=',d.seed.value_counts().sort_index().to_dict())
    print('status=',d.groupby('method').status.value_counts().to_dict())
else:
    print('Missing',donors)
print('\nPASS: preflight read-only; no model or simulation modified')
