#!/usr/bin/env python3
"""Strict per-container GCP memory residency auditor (offline; no OPTIMAL_ARM).

Input: complete, individually downloaded worker logs from ONE condition/run.
Require a closed create/destroy pair for each target container. Never infer a
missing destroy from VM shutdown; that would fabricate its residency duration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shlex

PREFIX = '[CONTAINER_LIFECYCLE]'


def parse_event_lines(log_text: str, host: str, target: str) -> list[dict]:
    events = []
    for lineno, line in enumerate(log_text.splitlines(), 1):
        if PREFIX not in line:
            continue
        tokens = shlex.split(line.split(PREFIX, 1)[1])
        evt = {'log_host': host, 'log_line': lineno}
        for token in tokens:
            if '=' in token:
                k, v = token.split('=', 1)
                evt[k] = v
        if evt.get('function') != target:
            continue
        if evt.get('event') in {'container_created', 'container_destroyed', 'container_destroy_failed'}:
            events.append(evt)
    return events


def peak_resident(intervals: list[tuple[int, int]]) -> int:
    # For coinciding event timestamps, process destroys before creates.
    changes = []
    for start, end in intervals:
        changes.extend([(start, 1), (end, -1)])
    running = peak = 0
    for _, delta in sorted(changes):
        running += delta
        if running < 0:
            raise ValueError('negative concurrent residency')
        peak = max(peak, running)
    if running != 0:
        raise ValueError('unclosed residency intervals')
    return peak


def analyze(inputs: dict[str, str], target: str, *, allow_prewarm: bool = False,
            allowed_destroy_reasons: tuple[str, ...] = ('expired',)) -> tuple[dict, list[dict]]:
    if not inputs:
        raise ValueError('nessun log fornito')
    created = {}
    destroyed = {}
    failed = []
    counts = Counter()
    for host, text in inputs.items():
        for evt in parse_event_lines(text, host, target):
            cid = evt.get('container_id')
            if not cid:
                raise ValueError(f'{host} linea {evt["log_line"]}: container_id assente')
            key = (host, cid)
            event = evt['event']
            if event == 'container_destroy_failed':
                failed.append(key)
                continue
            store = created if event == 'container_created' else destroyed
            if key in store:
                raise ValueError(f'evento duplicato {event}: {key}')
            store[key] = evt
            counts[event] += 1
    if failed:
        raise ValueError(f'distruzioni fallite: {failed[:8]}')
    if not created:
        raise ValueError('nessun container creato per il target')
    if set(created) != set(destroyed):
        missing = sorted(set(created) - set(destroyed))
        extras = sorted(set(destroyed) - set(created))
        raise ValueError(f'lifecycle incompleto: mancanti={missing[:8]}, destroy senza create={extras[:8]}')
    by_arch_mem = defaultdict(float)
    by_arch_count = Counter()
    by_host_intervals = defaultdict(list)
    by_host_mem = defaultdict(float)
    reasons = Counter()
    rows = []
    for key in sorted(created):
        c, d = created[key], destroyed[key]
        host, cid = key
        if c.get('node') != d.get('node'):
            raise ValueError(f'node mismatch: {key}')
        if c.get('machine_tag') != d.get('machine_tag'):
            raise ValueError(f'arch mismatch: {key}')
        arm = c.get('machine_tag')
        if arm not in {'amd64', 'arm64'}:
            raise ValueError(f'arch sconosciuta: {arm} per {key}')
        if c.get('creation_kind') == 'prewarm' and not allow_prewarm:
            raise ValueError(f'prewarm artificiale del target: {key}')
        reason = d.get('reason', '')
        if reason not in allowed_destroy_reasons:
            raise ValueError(f'distruzione {reason!r} inattesa per {key}; la campagna richiede expired')
        start, end = int(c['ts']), int(d['ts'])
        if end <= start:
            raise ValueError(f'durata non positiva per {key}: {start}..{end}')
        memory_mb = int(c['memory_mb'])
        if memory_mb <= 0:
            raise ValueError(f'memoria configurata non valida {memory_mb} per {key}')
        if d.get('memory_lookup_ok') == 'true' and int(d['memory_mb']) != memory_mb:
            raise ValueError(f'memoria differente create/destroy per {key}')
        residency_s = (end - start) / 1000.0
        mb_s = memory_mb * residency_s
        by_arch_mem[arm] += mb_s
        by_arch_count[arm] += 1
        by_host_mem[host] += mb_s
        by_host_intervals[host].append((start, end))
        reasons[reason] += 1
        rows.append({'worker': host, 'node': c.get('node'), 'container_id': cid, 'function_name': target,
                     'machine_tag': arm, 'memory_mb': memory_mb, 'created_ts_ms': start,
                     'destroyed_ts_ms': end, 'residency_s': residency_s,
                     'configured_MB_s': mb_s, 'configured_GiB_s_if_MiB': mb_s/1024.0,
                     'creation_kind': c.get('creation_kind'), 'destroy_reason': reason})
    summary = {'schema_version': 1, 'target_function': target, 'pairing': 'PASS',
               'created_containers': len(created), 'destroyed_containers': len(destroyed),
               'containers_by_arm': dict(by_arch_count),
               'memory_MB_s_by_arm': dict(by_arch_mem),
               'memory_GiB_s_by_arm_if_MiB': {k:v/1024 for k,v in by_arch_mem.items()},
               'memory_MB_s_by_worker': dict(by_host_mem),
               'peak_resident_containers_by_worker': {k:peak_resident(v) for k,v in by_host_intervals.items()},
               'destroy_reasons': dict(reasons),
               'total_memory_MB_s': sum(by_arch_mem.values()),
               'total_memory_GiB_s_if_MiB': sum(by_arch_mem.values())/1024,
               'no_optimal_arm_input': True,
               'notes': ['Valore di memoria configurata, non RSS.',
                         'GiB.s assume memory_mb stored in MiB; verificare semantica del runtime.',
                         'Nessuna inferenza di destroy mancante; pairing completo obbligatorio.',
                         'Picchi per worker; picco globale non calcolato senza sincronizzazione clock.']}
    return summary, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--log', required=True, action='append', metavar='WORKER=PATH',
                   help='Ripetere per ogni worker, es. --log sl-x86-1=logs/x86.log')
    p.add_argument('--dest', required=True, type=Path)
    p.add_argument('--allow-prewarm', action='store_true', help='Solo analisi diagnostica, non campagna principale')
    args = p.parse_args()
    inputs = {}
    provenance = []
    for entry in args.log:
        host, sep, file_path = entry.partition('=')
        if not sep or not host or not file_path or host in inputs:
            p.error(f'--log errato/duplicato: {entry}')
        path = Path(file_path).resolve()
        blob = path.read_bytes()
        inputs[host] = blob.decode('utf-8', errors='replace')
        provenance.append({'worker': host, 'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest()})
    summary, rows = analyze(inputs, args.target, allow_prewarm=args.allow_prewarm)
    summary['sources'] = provenance
    if args.dest.exists() and any(args.dest.iterdir()):
        p.error(f'destinazione non vuota: {args.dest}')
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    with (args.dest/'container-residency.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'PASS: {len(rows)} container; {summary["total_memory_MB_s"]:.3f} MB.s; CSV e JSON: {args.dest}')


if __name__ == '__main__':
    main()
