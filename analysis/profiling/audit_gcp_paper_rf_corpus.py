#!/usr/bin/env python3
"""Audit statico dei 67 benchmark al commit Serverledge congelato.

Non effettua richieste a GCP, non crea VM e non modifica codice o manifest.
Verifica nomi/runtime/risorse, sorgenti nel commit pinned, archivio Go e
riferimenti agli handler; NON puo' garantire il successo delle invocazioni.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import gcp_paper_rf_campaign as collector

ALLOWED_RUNTIMES = frozenset({'go125', 'python314', 'python312ml', 'nodejs17ng'})
EXPERIMENTS = 'examples/experiments/'


def blob_at_pin(relative: str) -> bytes | None:
    """Legge i byte versionati al PIN, senza usare il working tree modificabile."""
    full = EXPERIMENTS + relative
    result = subprocess.run(['git', '-C', str(collector.ROOT), 'show',
                             f'{collector.PIN}:{full}'],
                            capture_output=True)
    return result.stdout if result.returncode == 0 else None


def inspect_source(entry: dict, blob: bytes | None) -> tuple[list[str], list[str]]:
    """Ritorna (ERROR, WARNING). Non interpreta dipendenze remote o behavior."""
    errors, warnings = [], []
    src, runtime, handler = entry['src'], entry['runtime'], entry['handler']
    if runtime not in ALLOWED_RUNTIMES:
        errors.append(f'runtime non previsto: {runtime}')
    if blob is None:
        return [f'sorgente assente da {collector.PIN[:12]}: {src}'], warnings
    if not blob:
        return [f'sorgente vuoto: {src}'], warnings
    if runtime == 'go125':
        if not src.endswith('.tar'):
            errors.append(f'Go richiede archivio .tar, trovato {src}')
        else:
            try:
                with tarfile.open(fileobj=io.BytesIO(blob), mode='r:*') as archive:
                    members = archive.getmembers()
                    files = [m for m in members if m.isfile()]
                    if not files:
                        errors.append('archivio Go senza file regolari')
                    for m in members:
                        components = Path(m.name).parts
                        if m.name.startswith('/') or '..' in components or m.issym() or m.islnk():
                            errors.append(f'archivio con membro potenzialmente non sicuro: {m.name}')
                            break
            except (tarfile.TarError, EOFError, OSError) as exc:
                errors.append(f'archivio Go non leggibile: {exc}')
        if handler:
            warnings.append(f'handler Go non vuoto: {handler}')
    elif runtime.startswith('python'):
        if not src.endswith('.py'):
            errors.append(f'Python richiede .py: {src}')
        try:
            tree = ast.parse(blob.decode('utf-8'), filename=src)
            if not handler or '.' not in handler:
                errors.append(f'handler Python non in formato modulo.funzione: {handler!r}')
            else:
                module, function = handler.rsplit('.', 1)
                if module != Path(src).stem:
                    errors.append(f'modulo handler {module} diverso da sorgente {Path(src).stem}')
                top_level = {
                    n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if function not in top_level:
                    warnings.append(f'handler Python {function} non definito top-level; verificare esportazione')
        except (UnicodeDecodeError, SyntaxError) as exc:
            errors.append(f'Python non parsabile staticamente: {exc}')
    elif runtime == 'nodejs17ng':
        if not src.endswith('.js'):
            errors.append(f'Node richiede .js: {src}')
        if not handler:
            errors.append('handler Node non dichiarato')
        else:
            try:
                source = blob.decode('utf-8')
                if not re.search(r'\b' + re.escape(handler) + r'\b', source):
                    warnings.append(f'handler Node {handler} non nominato nel sorgente; verificare export')
            except UnicodeDecodeError as exc:
                errors.append(f'JavaScript non UTF-8: {exc}')
    return errors, warnings


def audit(entries: list[dict], loader=blob_at_pin) -> dict:
    records = []
    for entry in entries:
        blob = loader(entry['src'])
        errors, warnings = inspect_source(entry, blob)
        records.append({
            'name': entry['name'], 'runtime': entry['runtime'],
            'src': entry['src'], 'handler': entry['handler'],
            'memory_mb': entry['memory_mb'],
            'source_sha256': hashlib.sha256(blob).hexdigest() if blob is not None else None,
            'errors': errors, 'warnings': warnings,
            'status': 'FAIL' if errors else ('REVIEW' if warnings else 'PASS')
        })
    return {
        'pin': collector.PIN,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'functions': len(records),
        'passed': sum(r['status'] == 'PASS' for r in records),
        'to_review': sum(r['status'] == 'REVIEW' for r in records),
        'failed': sum(r['status'] == 'FAIL' for r in records),
        'records': records,
        'limits': 'Audit statico: non valida input dinamici, librerie Docker, durata o correttezza output'
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=(collector.ROOT / 'data/profiling/gcp-paper-rf-schema6/audit/corpus-static.json'))
    args = parser.parse_args(argv)
    repo_pin = collector.git('rev-parse', 'HEAD')
    if repo_pin != collector.PIN:
        parser.error(f'HEAD={repo_pin} diverso dal PIN={collector.PIN}')
    entries = collector.read_catalog(collector.pinned_catalog().decode('utf-8'))
    report = audit(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f"AUDIT: {report['functions']} funzioni | PASS={report['passed']} | REVIEW={report['to_review']} | FAIL={report['failed']}")
    for item in report['records']:
        if item['status'] != 'PASS':
            print(f"{item['status']}: {item['name']} ({item['runtime']})")
            for message in item['errors'] + item['warnings']:
                print(f'  - {message}')
    print(f"RAPPORTO: {args.output}")
    print('Questo controllo NON garantisce che tutte le funzioni funzionino su entrambe le VM.')
    return 1 if report['failed'] else (2 if report['to_review'] else 0)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
