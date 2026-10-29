#!/usr/bin/env python3
"""Supervisore paper-RF senza scadenza: export o salvataggio di emergenza, poi DELETE.

Esegue tutte le fasi fino al target e verifica il dataset. In caso di errore
prova a finalizzare un checkpoint pending e scaricare raw non archiviati.
Infine richiama gcp_down.sh anche se il recupero fallisce: DELETE e' una
preferenza esplicita e distruttiva. Non installa timer di distruzione su GCP.

Richiede un PC operativo e raggiungibilita' delle API GCP durante il cleanup.
L'interruzione forzata del PC o della rete NON garantisce il DELETE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import gcp_paper_rf_campaign as collector

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).with_name('gcp_paper_rf_campaign.py')
VM_NAMES = ('sl-registry', 'sl-lb', 'sl-workload', 'sl-x86-1', 'sl-arm-1')
WORKERS = ('sl-x86-1', 'sl-arm-1')
PROJECT = 'serverledge-tesi-ale'
ZONE = 'europe-west4-a'
SHA256 = re.compile(r'^[a-f0-9]{64}$')
SAFE_NAME = re.compile(r'^[A-Za-z0-9_.-]+$')


class StopRequested(Exception):
    pass


def on_signal(signum, _frame):
    raise StopRequested(f'segnale ricevuto: {signum}')


def gc_list(kind: str) -> list:
    """Lettura GCP indipendente: nessun `|| true` che mascheri errori API."""
    if kind not in ('instances', 'disks', 'addresses', 'snapshots'):
        raise ValueError(f'Risorsa GCP inattesa: {kind}')
    cmd = ['gcloud', f'--project={PROJECT}', 'compute', kind, 'list', '--format=json', '--quiet']
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError(f'Impossibile controllare {kind} su GCP: {(proc.stderr or "")[-500:]}')
    result = json.loads(proc.stdout)
    if not isinstance(result, list):
        raise RuntimeError(f'Formato inatteso GCP {kind}')
    return result


def delete_cluster() -> bool:
    """Invoca lo script distruttivo gia' usato e verifica tutte le risorse residue."""
    script = ROOT / 'scripts/gcp_scripts/gcp_down.sh'
    if not script.is_file():
        print(f'ERROR: manca lo script di distruzione: {script}', file=sys.stderr)
        return False
    try:
        listed = gc_list('instances')
        labeled = []
        for item in listed:
            labels = item.get('labels') or {}
            zone = str(item.get('zone') or '').rsplit('/', 1)[-1]
            if labels.get('progetto') == 'serverledge-tesi' and zone == ZONE:
                labeled.append(item.get('name'))
        unexpected = set(labeled) - set(VM_NAMES)
        if unexpected:
            raise RuntimeError(f'VM con label inattese: {sorted(unexpected)}')
        env = os.environ.copy()
        env.update(PROJECT=PROJECT, ZONE=ZONE)
        print('INFO: DELETE tramite gcp_down.sh (senza --all)', flush=True)
        result = subprocess.run(['bash', str(script)], env=env, timeout=900)
        if result.returncode:
            raise RuntimeError(f'gcp_down.sh exit={result.returncode}')
        left = {k: gc_list(k) for k in ('instances', 'disks', 'addresses', 'snapshots')}
        found = {k: [x.get('name', '?') for x in v] for k, v in left.items() if v}
        if found:
            raise RuntimeError(f'Risorse residue rilevate: {found}')
        print('PASS: nessuna VM, disco, IP statico o snapshot nel progetto; immagini conservate', flush=True)
        return True
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f'ERROR: DELETE non verificato: {exc}', file=sys.stderr, flush=True)
        print('ATTENZIONE: controllare manualmente GCP; potrebbero esistere risorse fatturabili',
              file=sys.stderr, flush=True)
        return False


def phase_info(dest: Path):
    state = collector.load(dest)
    progress = collector.counts(dest, state)
    _, phase = collector.phases(dest, state)
    return state, sum(progress.values()), len(progress) * state['target'], phase


def print_local_plan(dest: Path):
    state, selected, target, phase = phase_info(dest)
    print(f'Runtime: {state["pin"]}; selezionati: {selected}/{target}')
    print(f'Pending: {(dest / "pending.json").exists()}')
    if phase is None:
        print('Campagna completata, pronto export locale')
    else:
        print(f'Prossima fase: round={phase[0]}, arch={phase[1]}, funzioni={len(phase[2])}')
    print('Nessuna VM interrogata o modificata')


def run_child(argv: list[str]):
    """Attende la conclusione del child, senza timeout di campagna.

    Se il supervisore riceve SIGINT/SIGTERM/SIGHUP, termina l'intero gruppo
    figlio PRIMA di tentare il recupero dei dati e la distruzione delle VM.
    """
    proc = subprocess.Popen(argv, start_new_session=True)
    try:
        code = proc.wait()
        if code:
            raise RuntimeError(f'Collector fallito (exit={code})')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=12)


def verify_export(dest: Path, target: int):
    """Ricontrolla manifest, dataset e provenienza prima di DELETE riuscito."""
    manifest_file = dest / 'dataset_manifest.json'
    data = json.loads(manifest_file.read_text())
    if data['rows'] != target or data['pin'] != collector.PIN:
        raise RuntimeError('Manifest finale non coerente con target/runtime')
    for filename, field in (
        ('paper-rf-dataset.csv', 'dataset_sha256'),
        ('paper-rf-provenance.csv', 'provenance_sha256'),
        ('campaign.json', 'campaign_manifest_sha256'),
    ):
        path = dest / filename
        calculated = hashlib.sha256(path.read_bytes()).hexdigest()
        if calculated != data[field]:
            raise RuntimeError(f'SHA256 finale inatteso: {filename}')
    if phase_info(dest)[3] is not None:
        raise RuntimeError('Campagna non completa dopo export')
    print(f'PASS: {target} campioni esportati, SHA256 finali verificati', flush=True)


def supervise(dest: Path):
    """Processa tutte le fasi, senza limite di ore, costo o numero di fasi."""
    while True:
        _, selected, target, phase = phase_info(dest)
        if (dest / 'pending.json').exists():
            raise RuntimeError('Presente pending.json: recupero obbligatorio prima della raccolta')
        if phase is None:
            print(f'INFO: target {selected}/{target}; esporto il dataset', flush=True)
            run_child([sys.executable, str(CAMPAIGN), 'export-dataset', '--dest', str(dest)])
            verify_export(dest, target)
            return
        rnd, arch, entries = phase
        print(f'FASE round={rnd} arch={arch} funzioni={len(entries)}; warm={selected}/{target}', flush=True)
        run_child([sys.executable, str(CAMPAIGN), 'run-next-phase',
                   '--dest', str(dest), '--confirm'])
        new_selected = phase_info(dest)[1]
        if new_selected <= selected:
            raise RuntimeError('Nessun nuovo campione selezionato: blocco anti-loop')
        print(f'PASS: fase elaborata; warm={new_selected}/{target}', flush=True)


def remote_ssh(host: str, script: str, arguments=(), *, timeout=50) -> str:
    """SSH a tempo limitato SOLO per la procedura di emergenza."""
    if host not in VM_NAMES:
        raise ValueError(f'Host inatteso: {host}')
    args = ' '.join(shlex.quote(str(x)) for x in arguments)
    payload = f"bash -se -- {args} <<'PAPER_RF_RESCUE'\n{script}\nPAPER_RF_RESCUE"
    cmd = ['gcloud', 'compute', 'ssh', host, f'--project={PROJECT}', f'--zone={ZONE}',
           '--quiet', '--command', payload]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'SSH {host} fallita: {(result.stderr or "")[-400:]}')
    return result.stdout.strip()


def remote_scp(host: str, source: str, target: Path):
    """SCP limitato alla home utente della VM e con timeout autonomo."""
    if host not in VM_NAMES or not source.startswith('/home/'):
        raise ValueError('Destinazione/source SSH inattesa')
    result = subprocess.run(
        ['gcloud', 'compute', 'scp', f'{host}:{source}', str(target),
         f'--project={PROJECT}', f'--zone={ZONE}', '--quiet'],
        text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'SCP {host} fallita: {(result.stderr or "")[-400:]}')


def snapshot_remote(host: str, source: str, dest: Path, basename: str) -> str:
    """Copia il file remoto in locale con SHA256. `ABSENT` non crea file."""
    if not SAFE_NAME.fullmatch(basename) or not source.startswith(('/var/lib/serverledge/', '/home/')):
        raise ValueError('Percorso di rescue inatteso')
    source_script = '''set -euo pipefail
SRC="$1"; NAME="$2"
if ! sudo test -s "$SRC"; then echo ABSENT; exit 0; fi
mkdir -p "$HOME/paper-rf-rescue"
sudo install -m 0600 -o "$(id -un)" -g "$(id -gn)" \
  "$SRC" "$HOME/paper-rf-rescue/$NAME"
sha256sum "$HOME/paper-rf-rescue/$NAME" | cut -d ' ' -f1'''
    checksum = remote_ssh(host, source_script, (source, basename))
    if checksum == 'ABSENT':
        return 'ABSENT'
    if not SHA256.fullmatch(checksum):
        raise RuntimeError(f'SHA256 snapshot remoto invalido su {host}: {checksum[-120:]}')
    home = remote_ssh(host, 'printf "%s" "$HOME"')
    if not home.startswith('/home/'):
        raise RuntimeError(f'Home inattesa su {host}: {home}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.download')
    remote_scp(host, f'{home}/paper-rf-rescue/{basename}', partial)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f'SHA256 non coincide nel salvataggio {dest}')
    os.replace(partial, dest)
    return checksum


def emergency_salvage(dest: Path) -> bool:
    """Prova a preservare i raw non completati PRIMA della cancellazione.

    Tenta, senza dare per scontata la presenza dei file: raw attivo di ENTRAMBI
    i worker, copia archiviata del pending e CSV Locust del pending.
    Non usa il salvataggio emergenza per conteggiare righe RF automaticamente.
    """
    base = dest / 'emergency'
    base.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='rescue_', dir=base))
    captured = []
    issues = []
    pending = None
    pending_file = dest / 'pending.json'
    if pending_file.is_file():
        try:
            pending = json.loads(pending_file.read_text())
            (folder / 'pending_original.json').write_bytes(pending_file.read_bytes())
        except (OSError, ValueError) as exc:
            issues.append(f'pending.json non leggibile: {exc}')

    def save(host: str, path: str, basename: str):
        local = folder / basename
        try:
            result = snapshot_remote(host, path, local, basename)
            if result == 'ABSENT':
                print(f'INFO: rescue file assente: {host} {path}', flush=True)
                return False
            captured.append({'vm': host, 'origin': path, 'local': local.name, 'sha256': result})
            print(f'PASS: raw/artefatto di emergenza salvato: {local}', flush=True)
            return True
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            issues.append(f'{host} {path}: {exc}')
            print(f'WARN: salvataggio non riuscito: {host} {path}: {exc}',
                  file=sys.stderr, flush=True)
            return False

    # File attivo: conserva anche raw non ancora associato a pending.json.
    recovered_workers = set()
    for worker in WORKERS:
        path = '/var/lib/serverledge/profiling-samples.jsonl'
        if save(worker, path, f'{worker}_active.jsonl'):
            recovered_workers.add(worker)

    if pending is not None:
        worker = collector.EXPECTED.get(pending.get('arch'))
        block_id = pending.get('id', '')
        if worker is None or not SAFE_NAME.fullmatch(block_id):
            issues.append('pending contiene worker o ID non validi')
        else:
            archive = f'/var/lib/serverledge/paper-rf-checkpoints/{block_id}.jsonl'
            archived = save(worker, archive, f'{worker}_{block_id}_archived.jsonl')
            if worker not in recovered_workers and not archived:
                issues.append(f'Raw pending {block_id} non recuperato dal worker {worker}')
            # Conserva i file Locust disponibili, se l'esecuzione li ha creati.
            try:
                home = remote_ssh('sl-workload', 'printf "%s" "$HOME"')
                if not home.startswith('/home/'):
                    raise RuntimeError('Home sl-workload inattesa')
                for name in ('experiment_results.csv', 'summary.csv', 'locust_stats.csv',
                             'locust_failures.csv', 'locust_exceptions.csv'):
                    save('sl-workload', f'{home}/paper-rf-campaign/{block_id}/{name}',
                         f'workload_{block_id}_{name}')
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                issues.append(f'Workload del pending non recuperabile: {exc}')

    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'SNAPSHOT_DI_EMERGENZA_NON_VALIDATO_RF',
        'pending_at_failure': pending,
        'files': captured,
        'issues': issues,
    }
    (folder / 'rescue_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'INFO: resoconto rescue: {folder / "rescue_manifest.json"}', flush=True)
    if issues:
        print(f'WARN: recupero incompleto; {len(issues)} problemi. Si procedera COMUNQUE al DELETE',
              file=sys.stderr, flush=True)
    elif pending is not None:
        print('INFO: snapshot recuperato; pending richiede verifica manuale prima del riutilizzo', flush=True)
    return not issues


def try_recover_checkpoint(dest: Path):
    """Se disponibile, usa prima la procedura ufficiale per chiudere il pending."""
    pending = dest / 'pending.json'
    if not pending.is_file():
        return
    print('INFO: errore con pending.json; provo --recover prima del DELETE', flush=True)
    try:
        run_child([sys.executable, str(CAMPAIGN), 'recover', '--dest', str(dest), '--confirm'])
        print('PASS: pending finalizzato come checkpoint locale', flush=True)
    except (OSError, RuntimeError, StopRequested) as exc:
        print(f'WARN: recovery automatico impossibile: {exc}; provo snapshot raw',
              file=sys.stderr, flush=True)


def validate_launch_manifest(dest: Path) -> dict:
    """Gate locale PRE-esecuzione: un typo sul path non deve toccare GCP.

    Verifica presenza e struttura minima del manifest senza creare cartelle,
    interrogare VM o modificare checkpoint. Una campagna validata puo' entrare
    nella gestione errori con rescue e DELETE; un comando non inizializzato NO.
    """
    path = dest / 'campaign.json'
    if not path.is_file():
        raise RuntimeError(f'Manca campagna: {path}')
    if not CAMPAIGN.is_file():
        raise RuntimeError(f'Manca collector: {CAMPAIGN}')
    manifest = collector.load(dest)  # Controlla anche PIN, schema e CPU.
    entries = manifest.get('entries')
    target = manifest.get('target')
    block = manifest.get('block')
    if (not isinstance(entries, list) or not entries or
            not isinstance(target, int) or isinstance(target, bool) or target < 1 or
            not isinstance(block, int) or isinstance(block, bool) or not 1 <= block <= target):
        raise RuntimeError('Manifest non inizializzato correttamente: entries/target/block')
    names = []
    for e in entries:
        if not isinstance(e, dict) or not isinstance(e.get('name'), str) or not e['name']:
            raise RuntimeError('Manifest: funzione senza nome')
        names.append(e['name'])
    if len(names) != len(set(names)):
        raise RuntimeError('Manifest: nomi funzione duplicati')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', required=True, type=Path)
    p.add_argument('--run', action='store_true', help='Esegue fino al target; poi DELETE VM e dischi')
    p.add_argument('--confirm-delete', action='store_true',
                   help='Consente DELETE anche in caso di errore, dopo tentativo di recupero dati')
    args = p.parse_args(argv)
    dest = args.dest.resolve()
    if not args.run:
        if not (dest / 'campaign.json').is_file():
            p.error(f'Manca campagna: {dest / "campaign.json"}')
        print_local_plan(dest)
        print('Modalita di lettura; nessuna cancellazione (richiede --run --confirm-delete)')
        return 0
    if not args.confirm_delete:
        p.error('Per eseguire e CANCELLARE VM e boot disk serve --confirm-delete')

    # Gate senza effetti: il manifest non deve essere creato dal supervisore.
    # Un errore qui non invoca rescue e soprattutto NON invoca DELETE.
    try:
        validate_launch_manifest(dest)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'ERROR: preflight campagna rifiutato: {exc}', file=sys.stderr, flush=True)
        print('INFO: nessuna operazione GCP tentata, nessun DELETE', flush=True)
        return 2

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for sig in previous:
        signal.signal(sig, on_signal)
    success = False
    deleted = False
    rescued = True
    try:
        supervise(dest)
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        try:
            try_recover_checkpoint(dest)
            rescued = emergency_salvage(dest)
        except Exception as rescue_exc:
            rescued = False
            print(f'ERROR: rescue fallito: {rescue_exc}', file=sys.stderr, flush=True)
    finally:
        # Durante DELETE non permettere a Ctrl+C/SIGTERM di interrompere cleanup.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            deleted = delete_cluster()
        except Exception as exc:
            print(f'ERROR: eccezione imprevista durante DELETE: {exc}', file=sys.stderr)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if not deleted:
        return 3
    if not success:
        if not rescued:
            print('WARN: raw non recuperati completamente; boot disk comunque eliminati',
                  file=sys.stderr, flush=True)
        print(f'ERROR: campagna incompleta; verificare checkpoint e cartella emergency di {dest}',
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
