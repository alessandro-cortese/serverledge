#!/usr/bin/env python3
"""Audit di sola lettura della campagna Serverledge GCP memory full-01.

Verifica a livello di ogni container: campi, durata da timestamp, contributo
memoria-configurata x residenza, totale del riepilogo, e aggregazione per ARM.
Calcola anche la media temporale sulla finestra di lifecycle della funzione.
Non equivale ad un parser indipendente dei log raw: la fonte primaria qui e'
container-residency.csv, gia' ricostruito dai log dal runner GCP.

Da root repository:
 python3 analysis/profiling/audit_memoria_container_full01.py self-test
 python3 analysis/profiling/audit_memoria_container_full01.py check
 python3 analysis/profiling/audit_memoria_container_full01.py run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

SOURCE = Path('data/profiling/gcp-memory/cloud-runs/full-01')
OUT = Path('data/profiling/gcp-memory/analysis/lifecycle-audit-full01-v1')
CONFIGS = ('N02','D02','R02','N04','D04','R04','N08','D08','R08')


def require(p, msg):
    if not p:
        raise ValueError(msg)


def num(value, ctx):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{ctx}: numero non valido {value!r}') from exc
    require(math.isfinite(x), f'{ctx}: numero non finito')
    return x


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def per_run(folder: Path, verify_hashes=True):
    summary_path=folder/'memory-summary.json'
    residency_path=folder/'container-residency.csv'
    verified_path=folder/'verified.json'
    for p in (summary_path,residency_path,verified_path):
        require(p.is_file(), f'{folder.name}: file mancante {p.name}')
    s=load_json(summary_path)
    v=load_json(verified_path)
    require(v.get('state') == 'PASS', f'{folder.name}: verified != PASS')
    if verify_hashes:
        checks=v.get('sha256', {})
        require(bool(checks), f'{folder.name}: SHA inventory assente')
        for rel, expected in checks.items():
            rel=Path(rel)
            require(not rel.is_absolute() and '..' not in rel.parts, f'{folder.name}: path SHA non valido')
            p=folder/rel
            require(p.is_file() and sha(p)==expected, f'{folder.name}: SHA fallito per {rel}')
    rows=read_csv(residency_path)
    require(rows, f'{folder.name}: residency vuoto')
    require(len(rows)==int(s['created_containers']), f'{folder.name}: numero container incoerente')
    # Colonne effettive del runner gcp_memory_campaign_v3 (full-01).
    expected_columns={'container_id','function_name','machine_tag','memory_mb',
                      'created_ts_ms','destroyed_ts_ms','residency_s',
                      'configured_MB_s','configured_GiB_s_if_MiB'}
    require(expected_columns.issubset(rows[0]),
            f'{folder.name}: mancano colonne {expected_columns-set(rows[0])}; colonne={list(rows[0])}')
    ids=set(); total=0.; by_arm=defaultdict(float); min_ts=math.inf; max_ts=-math.inf
    for row in rows:
        cid=row['container_id']; require(cid and cid not in ids, f'{folder.name}: container_id duplicato {cid}')
        ids.add(cid)
        arch=row['machine_tag']; require(arch in ('amd64','arm64'), f'{folder.name}: arch non valida {arch}')
        function_expected=folder.name.rsplit('-',3)[0]
        require(row['function_name']==function_expected,
                f'{folder.name}: container {cid} attribuito a {row["function_name"]!r}, non a {function_expected!r}')
        mem=num(row['memory_mb'],cid)
        start=num(row['created_ts_ms'],cid);end=num(row['destroyed_ts_ms'],cid)
        duration_s=num(row['residency_s'],cid)
        stored=num(row['configured_MB_s'],cid)
        stored_gib=num(row['configured_GiB_s_if_MiB'],cid)
        require(mem>0 and end>start and duration_s>0, f'{folder.name}: memoria/durata non positiva {cid}')
        recomputed_duration_s=(end-start)/1000.
        require(math.isclose(recomputed_duration_s,duration_s,rel_tol=1e-8,abs_tol=1e-3),
                f'{folder.name}: residenza diversa dai timestamp {cid}: {recomputed_duration_s} vs {duration_s}')
        recomputed=mem*recomputed_duration_s
        require(math.isclose(recomputed,stored,rel_tol=1e-8,abs_tol=1e-3),
                f'{folder.name}: contributo errato {cid}: {recomputed} vs {stored}')
        require(math.isclose(stored/1024.,stored_gib,rel_tol=1e-8,abs_tol=1e-5),
                f'{folder.name}: conversione GiB.s errata {cid}: {stored/1024.} vs {stored_gib}')
        total+=recomputed; by_arm[arch]+=recomputed
        min_ts=min(min_ts,start);max_ts=max(max_ts,end)
    # Come nel runner: memory_mb per intervallo di residenza (s).
    # Le intestazioni storiche usano MB, mentre la conversione /1024 suggerisce MiB;
    # confrontiamo i valori numerici senza modificare la convenzione originale.
    summary_total=num(s['total_memory_MB_s'],f'{folder.name}: summary total')
    require(math.isclose(total,summary_total,rel_tol=1e-8,abs_tol=1e-4),
            f'{folder.name}: totale mismatch: {total} vs {summary_total}')
    per_arm=s.get('memory_MB_s_by_arm',s.get('memory_MiB_s_by_arm'))
    if isinstance(per_arm,dict):
        for arch in ('amd64','arm64'):
            if arch in per_arm:
                require(math.isclose(by_arm[arch],float(per_arm[arch]),rel_tol=1e-8,abs_tol=1e-4),
                        f'{folder.name}: somma per-arch mismatch {arch}')
    span_sec=(max_ts-min_ts)/1000.
    require(span_sec>0,f'{folder.name}: intervallo campagna non positivo')
    c=s['config_id']
    require(c in CONFIGS, f'{folder.name}: condizione sconosciuta {c}')
    return {
        'run_key':folder.name,'function_name':folder.name.rsplit('-',3)[0],
        'condition':c,'c':int(c[1:])/10,'replica':int(folder.name.rsplit('-',2)[1].lstrip('r')),
        'container_count':len(rows),'mem_residency_from_containers':total,
        'mem_residency_summary':summary_total,'arm64_mem_residency':by_arm['arm64'],
        'amd64_mem_residency':by_arm['amd64'],
        'lifecycle_window_s':span_sec,
        'mean_allocated_memory_lifecycle':total/span_sec,
        'start_container_ts_ms':int(min_ts),'end_container_ts_ms':int(max_ts),
        'fallbacks':int(s['mab']['fallbacks']),
        'sha256_residency':sha(residency_path),
        'sha256_summary':sha(summary_path)
    }


def calculate_pairs(rows):
    groups=defaultdict(dict)
    for row in rows:
        key=(row['function_name'],row['c'],row['replica'])
        kind=row['condition'][0]
        require(kind not in groups[key], f'duplicato {key}/{kind}')
        groups[key][kind]=row
    paired=[]
    for (fn,c,rep), values in sorted(groups.items()):
        require(set(values)=={'N','D','R'},f'{fn}/{c}/{rep}: tripla incompleta')
        n,d,r=(values[k] for k in 'NDR')
        for label, obj in (('D',d),('R',r)):
            paired.append({
                'function_name':fn, 'c':c, 'replica':rep,'strategy':label,
                'baseline_residency':n['mem_residency_from_containers'],
                'tl_residency':obj['mem_residency_from_containers'],
                'saving_residency_percent':100*(n['mem_residency_from_containers']-obj['mem_residency_from_containers'])/n['mem_residency_from_containers'],
                'baseline_mean_allocated_memory':n['mean_allocated_memory_lifecycle'],
                'tl_mean_allocated_memory':obj['mean_allocated_memory_lifecycle'],
                'saving_mean_allocated_memory_percent':100*(n['mean_allocated_memory_lifecycle']-obj['mean_allocated_memory_lifecycle'])/n['mean_allocated_memory_lifecycle'],
                'baseline_lifecycle_window_s':n['lifecycle_window_s'],
                'tl_lifecycle_window_s':obj['lifecycle_window_s'],
                'baseline_containers':n['container_count'],'tl_containers':obj['container_count'],
                'any_fallback': bool(n['fallbacks'] or obj['fallbacks'])
            })
    return paired


def save_csv(path, rows):
    require(rows, f'{path}: no rows')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f=Path(td);(f/'memory-summary.json').write_text(json.dumps({
            'created_containers':2,'total_memory_MB_s':1500.,'config_id':'R04','mab':{'fallbacks':0}}))
        rec=[
            dict(container_id='a',function_name='abc',machine_tag='amd64',memory_mb=100,
                 created_ts_ms=1000,destroyed_ts_ms=11000,residency_s=10,
                 configured_MB_s=1000,configured_GiB_s_if_MiB=1000/1024),
            dict(container_id='b',function_name='abc',machine_tag='arm64',memory_mb=100,
                 created_ts_ms=5000,destroyed_ts_ms=10000,residency_s=5,
                 configured_MB_s=500,configured_GiB_s_if_MiB=500/1024)]
        save_csv(f/'container-residency.csv',rec)
        (f/'verified.json').write_text(json.dumps({'state':'PASS','sha256':{'container-residency.csv':sha(f/'container-residency.csv')}}))
        # Lo self-test testa la parte aritmetica separatamente dai pattern del nome cartella.
        f2=f/'abc-R04-r01-q3';f2.mkdir()
        for n in ('memory-summary.json','container-residency.csv','verified.json'):
            (f2/n).write_bytes((f/n).read_bytes())
        result=per_run(f2)
        require(result['container_count']==2 and result['mem_residency_from_containers']==1500.,'self test total')
        require(math.isclose(result['mean_allocated_memory_lifecycle'],150.0),'self test mean')
        sample=[]
        for kind,val in [('N',2000.),('D',1500.),('R',1000.)]:
            sample.append({'function_name':'abc','c':.4,'replica':1,'condition':kind+'04',
                           'mem_residency_from_containers':val,'mean_allocated_memory_lifecycle':val/10,
                           'lifecycle_window_s':10.,'container_count':2,'fallbacks':0})
        paired=calculate_pairs(sample)
        require(len(paired)==2 and math.isclose(paired[0]['saving_residency_percent'],25.),
                'self test matching NoTL/DIFF')
        require(math.isclose(paired[1]['saving_residency_percent'],50.),'self test ratio')
    print('PASS self-test: schema full-01, durata, totale, due architetture, memoria media')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['self-test','check','run'])
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args()
    if args.mode=='self-test':self_test();return
    src=args.source
    require(src.is_dir(),f'campagna non trovata: {src}')
    folders=sorted(x for x in (src/'runs').iterdir() if x.is_dir())
    require(len(folders)==180,f'attesi 180 run, presenti {len(folders)}')
    rows=[per_run(x) for x in folders]
    require(len({r['run_key'] for r in rows})==180,'run duplicati')
    require(len({r['function_name'] for r in rows})==10,'attese 10 funzioni')
    require(sum(x['container_count'] for x in rows)==1105,'attesi 1105 container')
    paired=calculate_pairs(rows)
    require(len(paired)==120,'attesi 120 confronti N/D e N/R')
    print('PASS memoria: 180 run, 1105 container, 10 funzioni, 120 confronti con No-TL')
    for c in (.2,.4,.8):
        for label in 'DR':
            r=[x for x in paired if x['c']==c and x['strategy']==label]
            print(f'c={c:.1f} {label}: '
                  f'risparmio MB.s medio su 20 coppie = {statistics.mean(x["saving_residency_percent"] for x in r):+.3f}% ; '
                  f'risparmio memoria-media MB = {statistics.mean(x["saving_mean_allocated_memory_percent"] for x in r):+.3f}%')
    if args.mode=='check':
        print('PASS CHECK: sola lettura; non creati file');return
    require(not args.out.exists(),f'output gia esistente, non sovrascrivo: {args.out}')
    args.out.mkdir(parents=True)
    save_csv(args.out/'memory_lifecycle_per_run.csv',rows)
    save_csv(args.out/'memory_lifecycle_paired.csv',paired)
    (args.out/'AUDIT_REPORT.txt').write_text(
        'Audit da container-residency.csv e memory-summary.json. Non e una rilettura indipendente dei log raw.\n'
        'Finestra per memoria media: prima creazione -> ultima distruzione dei container della singola esecuzione.\n'
        'Unita: si conserva la convenzione numerica originale memory_mb * residency_s (etichettata MB.s); verificata divisione per 1024 per GiB.s.\n'
        'Nessuna prova causale dai soli due replicati; separare run con fallback.\n',encoding='utf-8')
    print('PASS RUN: scritto in',args.out)

if __name__=='__main__':
    main()
