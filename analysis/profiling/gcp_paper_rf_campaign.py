#!/usr/bin/env python3
"""Paper-RF orchestration/schema 6, with no changes to the frozen runtime.

The orchestrator saves checkpoints in pairs (function, architecture) and does not
count Locust requests as valid rows: it uses ONLY the CSV exported by the
pinned binary.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

PIN = '9f9e9cfc5631b41f525f6d91fadad4415b93a8c1'
PROJECT = 'serverledge-tesi-ale'
ZONE = 'europe-west4-a'
ARCHS = ('amd64', 'arm64')
EXPECTED = {'amd64': 'sl-x86-1', 'arm64': 'sl-arm-1'}
META_COLUMNS = {'request_id', 'function_name', 'machine_tag', 'configured_cpus'}
SCHEMA = 6
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = ROOT / 'data/profiling/gcp-paper-rf-schema6/campaign/paper_rf_schema6_final'
IDENT = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')


def err(message: str) -> None:
    raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha(path.read_bytes())


def command(args, *, capture=True, check=True, stdin=None, timeout=None):
    cmd = [str(x) for x in args]
    print('CMD:', ' '.join(shlex.quote(x) for x in cmd[:8]), flush=True)
    result = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, text=True, timeout=timeout)
    if result.returncode and check:
        err(f"Comando fallito ({result.returncode}): {' '.join(cmd[:5])}\n{(result.stderr or '')[-2500:]}")
    return (result.stdout or '').strip() if capture else result.returncode


def git(*args):
    return command(['git', '-C', ROOT, *args])


def pinned_catalog():
    result = subprocess.run(['git', '-C', str(ROOT), 'show',
                             f'{PIN}:scripts/gcp_scripts/functions_catalog.sh'],
                            capture_output=True, check=True)
    return result.stdout


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    os.replace(temp, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_catalog(raw: str):
    entries = []
    for match in re.finditer(r'^\s*"([^"\n]+)"\s*$', raw, flags=re.MULTILINE):
        fields = match.group(1).split('|')
        if len(fields) != 5:
            continue
        name, runtime, src, mem, handler = fields
        if not all(IDENT.fullmatch(x) for x in (name, runtime)):
            err(f'Catalogo non sicuro: {name}')
        if not re.fullmatch(r'[\w.\-/]+', src) or '..' in Path(src).parts:
            err(f'Path non sicuro: {src}')
        if handler and not re.fullmatch(r'[\w.]+', handler):
            err(f'Handler non sicuro: {handler}')
        if not mem.isdigit() or int(mem) <= 0:
            err(f'Memoria non valida: {mem}')
        entries.append(dict(name=name, runtime=runtime, src=src, memory_mb=int(mem), handler=handler))
    names = [e['name'] for e in entries]
    if len(entries) != 67 or len(set(names)) != len(names):
        err(f'Catalogo inatteso: {len(entries)} entries, {len(set(names))} nomi unici; attesi 67')
    return entries


def load(dest: Path):
    path = dest / 'campaign.json'
    if not path.is_file():
        err(f'Campagna non inizializzata: {path}')
    c = read_json(path)
    if c['pin'] != PIN or c['schema'] != SCHEMA or c['cpu'] != 1:
        err('Manifest non compatibile con il runtime congelato')
    return c


def completed_blocks(dest: Path, c: dict):
    records = []
    valid_names = {e['name'] for e in c['entries']}
    seen_ids = set()
    for path in sorted((dest / 'blocks').glob('*/*/*/checkpoint.json')):
        item = read_json(path)
        if item['pin'] != PIN or item['function'] not in valid_names or item['arch'] not in ARCHS:
            err(f'Checkpoint non compatibile: {path}')
        for rel, expected_sha in item['sha256'].items():
            f = path.parent / rel
            if not f.is_file() or digest(f) != expected_sha:
                err(f'Checkpoint corrotto / SHA errato: {f}')
        if len(item['selected_ids']) != item['selected_count']:
            err(f'Checkpoint con cardinalità errata: {path}')
        if len(set(item['selected_ids'])) != item['selected_count']:
            err(f'ID duplicati nel blocco {path}')
        for rid in item['selected_ids']:
            k = (item['arch'], item['function'], rid)
            if k in seen_ids:
                err(f'ID riutilizzato in più blocchi: {k}')
            seen_ids.add(k)
        records.append(item)
    return records


def counts(dest, c):
    ans = {(a, e['name']): 0 for a in ARCHS for e in c['entries']}
    for r in completed_blocks(dest, c):
        ans[r['arch'], r['function']] += r['selected_count']
    if any(v > c['target'] for v in ans.values()):
        err('Checkpoint oltre il target')
    return ans


def pending_path(dest):
    return dest / 'pending.json'


def phases(dest, c):
    state = counts(dest, c)
    # Una fase e' completa quando ogni funzione ha almeno un checkpoint del round,
    # oppure ha gia' raggiunto il proprio target.
    records = completed_blocks(dest, c)
    result = []
    round_no = 0
    while True:
        for arch in (ARCHS if round_no % 2 == 0 else tuple(reversed(ARCHS))):
            outstanding = [e for e in c['entries'] if
                state[arch, e['name']] < c['target'] and
                not any(r['arch'] == arch and r['function'] == e['name'] and r['round'] == round_no
                        for r in records)]
            if outstanding:
                return result, (round_no, arch, outstanding)
            result.append((round_no, arch))
        if all(state[arch, e['name']] >= c['target'] for arch in ARCHS for e in c['entries']):
            return result, None
        round_no += 1
        if round_no > c['target'] + 10:
            err('Troppi round: verificare checkpoint/validità')


def show_plan(dest, c):
    progress = counts(dest, c)
    _, next_phase = phases(dest, c)
    total = sum(progress.values())
    desired = len(progress) * c['target']
    print(f'Runtime {c["pin"]} | {len(c["entries"])} funzioni | {total}/{desired} warm RF selezionati')
    if pending_path(dest).exists():
        item = read_json(pending_path(dest))
        print(f'STOP: blocco pendente da recuperare: {item["id"]}')
    if next_phase is None:
        print('PASS: CAMPAGNA COMPLETATA')
    else:
        rnd, arch, queue = next_phase
        print(f'PROSSIMA FASE: round={rnd}, arch={arch}, funzioni da completare={len(queue)}')
        for entry in queue[:8]:
            print(f'  {entry["name"]}: {progress[arch,entry["name"]]}/{c["target"]}')
        if len(queue) > 8:
            print(f'  ... altre {len(queue)-8}')
    print('Nessuna risorsa GCP interrogata.')


def gcloud(*args, capture=True):
    return command(['gcloud', '--project', PROJECT, 'compute', *args, '--zone', ZONE, '--quiet'], capture=capture)


def remote(host: str, script: str, args=(), capture=True):
    if host not in ('sl-registry','sl-lb','sl-workload','sl-x86-1','sl-arm-1'):
        err('Host remoto inatteso')
    # Comandi parametrizzati: soltanto catalogo congelato e identificatori validati.
    argtext = ' '.join(shlex.quote(str(a)) for a in args)
    payload = "bash -se -- " + argtext + " <<'PAPER_RF_REMOTE'\n" + script + "\nPAPER_RF_REMOTE"
    return gcloud('ssh', host, '--command', payload, capture=capture)


def scp(host, remote_path, local_path):
    if not str(remote_path).startswith('/home/'):
        err('SCP limitato alle copie nella home remota')
    gcloud('scp', f'{host}:{remote_path}', str(local_path), capture=False)


def internal_ip(host):
    return gcloud('instances', 'describe', host, '--format=value(networkInterfaces[0].networkIP)')


def worker_status(host):
    raw = remote(host, 'curl -fsS --max-time 10 http://localhost:1323/status')
    data = json.loads(raw)
    warm = sum(int(v) for v in (data.get('AvailableWarmContainers') or {}).values())
    return (warm, int(data['AvailableMemory']), int(data['FreeMemory']), float(data['UsedCPU']))


def drain(host, baseline, *, seconds=150):
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        active = remote(host, 'sudo docker ps -q | wc -l').strip()
        status = worker_status(host)
        if active == '0' and status == baseline:
            print(f'PASS: {host} drain e contabilità tornata al baseline')
            return
        time.sleep(3)
    err(f'{host}: drain/accounting non torna al baseline, NON archiviare/avviare altri blocchi')


def raw_exists(host):
    s = remote(host, "sudo sh -c 'if test -s /var/lib/serverledge/profiling-samples.jsonl; then echo PRESENT; else echo EMPTY; fi'")
    return s == 'PRESENT'


def check_running(arch):
    worker = EXPECTED[arch]
    other = EXPECTED['arm64' if arch == 'amd64' else 'amd64']
    def count(h):
        return remote(h, 'pgrep -xc serverledge || true').strip()
    if count(worker) != '1' or count(other) != '0':
        err(f'Isolamento worker non valido per {arch}')
    reg = remote('sl-registry',
        'sudo docker exec Etcd-server etcdctl get registry/ --prefix --keys-only')
    lines = [line.strip() for line in reg.splitlines() if line.strip()]
    a = sum(bool(re.fullmatch(r'registry/cloud-region/amd64/[^/]+', line)) for line in lines)
    b = sum(bool(re.fullmatch(r'registry/cloud-region/arm64/[^/]+', line)) for line in lines)
    if (a,b) != ((1,0) if arch == 'amd64' else (0,1)):
        err(f'Registry errato: amd64={a}, arm64={b}')
    remote('sl-lb', 'pgrep -x lb >/dev/null')
    remote(worker, 'curl -fsS --max-time 5 http://localhost:1323/status >/dev/null')
    return worker


def start_arch(repo: Path, arch):
    path = repo / 'scripts/gcp_scripts/gcp_paper_rf_start.sh'
    if not path.is_file():
        err(f'Script avvio mancante: {path}')
    env = os.environ.copy()
    env['PAPER_RF_START_CONFIRM'] = 'YES'
    result = subprocess.run(['bash', str(path), '--start', arch], env=env)
    if result.returncode:
        err(f'Avvio cluster {arch} fallito')
    return check_running(arch)


def ensure_binaries(repo):
    if git('rev-parse', 'HEAD') != PIN:
        err('HEAD del repository principale non congelato')
    script = repo / 'scripts/gcp_scripts/gcp_paper_rf_deploy.sh'
    if subprocess.run(['bash', str(script), '--verify']).returncode:
        err('Verifica binari remoti fallita')
    exporter = repo / ('build/gcp-paper-rf-schema6-' + PIN[:12]) / 'bin/serverledge-profiling-amd64'
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        err(f'Exporter pinned assente: {exporter}')
    return exporter


def register(entry, lbip):
    shell_script = '''set -euo pipefail
LB="$1"; NAME="$2"; RUNTIME="$3"; SRC="$4"; MEM="$5"; HANDLER="$6"
cd /opt/serverledge/examples/experiments
export SERVERLEDGE_HOST="$LB" SERVERLEDGE_PORT=1323
args=(create -f "$NAME" --runtime "$RUNTIME" --src "$SRC" --memory "$MEM" --cpu 1 --update)
if [[ -n "$HANDLER" ]]; then args+=(--handler "$HANDLER"); fi
/opt/serverledge/bin/serverledge-cli "${args[@]}"
'''
    print(remote('sl-workload', shell_script, [lbip, entry['name'], entry['runtime'],
                                      entry['src'], entry['memory_mb'], entry['handler']]))


def invoke(entry, lbip, block_id, limit, target):
    script = '''set -euo pipefail
LB="$1"; NAME="$2"; ID="$3"; LIMIT="$4"; TARGET="$5"
BASE="$HOME/paper-rf-campaign/$ID"
mkdir -p "$BASE"
if [[ ! -x "$HOME/locust-venv/bin/locust" ]]; then
  echo 'ERRORE: Locust non installato; non installo dipendenze durante la campagna' >&2
  exit 4
fi
export FUNCTIONS_TO_RUN="$NAME" MAX_REQUESTS_PER_FUNCTION="$LIMIT"
export REQUESTS_PER_FUNCTION="$LIMIT" TARGET_WARM_VALID="$TARGET" LB_POLICY=RoundRobin
export COLLECTION_CSV="$BASE/experiment_results.csv" COLLECTION_SUMMARY="$BASE/summary.csv"
"$HOME/locust-venv/bin/locust" -f /opt/serverledge/examples/experiments/locustfile_collection.py \\
  --headless --users 1 --spawn-rate 1 --run-time 90m \\
  --host "http://${LB}:1323" --csv "$BASE/locust"
'''
    remote('sl-workload', script, [lbip, entry['name'], block_id, limit, target], capture=False)


def fetch_raw(worker, block_id, dest):
    # Copia immutabile del raw nella home VM con hash per controllare il trasporto.
    script = '''set -euo pipefail
ID="$1"; DIR="$HOME/paper-rf-campaign/$ID"; mkdir -p "$DIR"
sudo test -s /var/lib/serverledge/profiling-samples.jsonl
sudo install -m 0600 -o "$(id -un)" -g "$(id -gn)" \\
  /var/lib/serverledge/profiling-samples.jsonl "$DIR/profiling-samples.jsonl"
sha256sum "$DIR/profiling-samples.jsonl" | cut -d' ' -f1
'''
    remote_sha = remote(worker, script, [block_id]).strip()
    if not re.fullmatch('[a-f0-9]{64}', remote_sha):
        err(f'SHA remoto non valido: {remote_sha}')
    out = dest / 'raw.jsonl'
    partial = dest / 'raw.download'
    # La home sulle immagini custom e /home/<utente>; usa path risolto via SSH.
    home = remote(worker, 'printf "%s" "$HOME"').strip()
    if not home.startswith('/home/'):
        err('Home VM non attesa')
    scp(worker, f'{home}/paper-rf-campaign/{block_id}/profiling-samples.jsonl', partial)
    if digest(partial) != remote_sha:
        err('SHA diverso tra raw locale e remoto; copia parziale non promossa')
    os.replace(partial, out)
    return out


def fetch_locust(block_id, dest):
    home = remote('sl-workload', 'printf "%s" "$HOME"').strip()
    if not home.startswith('/home/'):
        err('Home workload inattesa')
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('experiment_results.csv', 'summary.csv', 'locust_stats.csv',
                 'locust_failures.csv', 'locust_exceptions.csv',
                 'locust_stats_history.csv'):
        try:
            scp('sl-workload', f'{home}/paper-rf-campaign/{block_id}/{name}', dest / name)
        except RuntimeError:
            print(f'WARN: {name} non disponibile; raw resta sorgente primaria')


def validate_raw(raw, *, name, arch):
    count = 0
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        if s.get('schema_version') != 6 or s.get('function_name') != name or s.get('machine_tag') != arch:
            err(f'Raw non coerente (schema/funzione/arch): {raw}')
        if float((s.get('function_configuration') or {}).get('configured_cpus', -1)) != 1:
            err('configured_cpus != 1')
        count += 1
    if count == 0:
        err('Raw senza righe')
    return count


def export_select(exporter, raw, csv_path, selected_path, experiment_id, needed):
    command([exporter, 'export-paper-rf-csv', '--input', raw,
             '--experiment-id', experiment_id, '--output', csv_path], capture=False)
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        if len(cols) != 22 or not META_COLUMNS.issubset(cols):
            err(f'CSV schema RF inatteso: {cols}')
        rows = list(reader)
    ids = [r['request_id'] for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        err('Exporter contiene ID vuoti/duplicati')
    chosen = rows[:needed]
    with selected_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, cols)
        writer.writeheader()
        writer.writerows(chosen)
    return rows, chosen


def rotate_remote(worker, block_id, expected_sha):
    # Non sovrascrivere mai un raw remoto. Backup unico e non distruttivo.
    script = '''set -euo pipefail
ID="$1"; SHA="$2"; SRC=/var/lib/serverledge/profiling-samples.jsonl
ARCHIVE=/var/lib/serverledge/paper-rf-checkpoints
sudo mkdir -p "$ARCHIVE"
if sudo test -s "$SRC"; then
   ACTUAL="$(sudo sha256sum "$SRC" | cut -d' ' -f1)"
   [[ "$ACTUAL" == "$SHA" ]] || { echo 'ERRORE: raw cambiato dopo snapshot' >&2; exit 5; }
   sudo test ! -e "$ARCHIVE/$ID.jsonl" || { echo 'ERRORE: archivio già esistente' >&2; exit 5; }
   sudo mv "$SRC" "$ARCHIVE/$ID.jsonl"
elif sudo test -s "$ARCHIVE/$ID.jsonl"; then
   ACTUAL="$(sudo sha256sum "$ARCHIVE/$ID.jsonl" | cut -d' ' -f1)"
   [[ "$ACTUAL" == "$SHA" ]] || exit 5
else
   echo 'ERRORE: raw e archivio non presenti' >&2; exit 5
fi
'''
    remote(worker, script, [block_id, expected_sha])


def drain_recovery(host, *, seconds=150):
    """Attende un worker completamente scarico senza fingere un baseline ignoto."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = worker_status(host)
        active = remote(host, 'sudo docker ps -q | wc -l').strip()
        warm, available, free, used_cpu = status
        if active == '0' and warm == 0 and abs(used_cpu) < 1e-6 and available == free:
            time.sleep(3)
            status2 = worker_status(host)
            active2 = remote(host, 'sudo docker ps -q | wc -l').strip()
            if status2 == status and active2 == '0':
                return
        time.sleep(3)
    err('Worker non completamente scarico: non procedo al recovery')


def commit_pending(dest, c, exporter, pending, *, recovery=False):
    arch, name, block_id = pending['arch'], pending['function'], pending['id']
    worker = EXPECTED[arch]
    target_path = dest / 'blocks' / arch / f'round_{pending["round"]:03d}' / name
    target_path.mkdir(parents=True, exist_ok=True)
    checkpoint = target_path / 'checkpoint.json'
    if checkpoint.is_file():
        record = read_json(checkpoint)
        rotate_remote(worker, block_id, record['sha256']['raw.jsonl'])
        pending_path(dest).unlink()
        print(f'PASS: completato archivio di checkpoint esistente {block_id}')
        return
    # Sia dopo Locust, sia durante recovery, nessun processo Locust può scrivere.
    if remote('sl-workload', 'pgrep -f "[l]ocust.*locustfile_collection.py" >/dev/null && echo ACTIVE || echo IDLE') != 'IDLE':
        err('Locust ancora attivo sul workload: impossibile chiudere il blocco')
    # Il worker deve avere concluso la raccolta e raggiunto il baseline.
    if recovery:
        # Il baseline originale non sopravvive a un crash: richiede stato completamente scarico.
        drain_recovery(worker)
    raw = target_path / 'raw.jsonl'
    if raw.is_file():
        # Una copia esistente non basta: risnapshot e confronto se remoto disponibile.
        previous = digest(raw)
        raw = fetch_raw(worker, block_id, target_path)
        if digest(raw) != previous:
            err('Raw remoto cambiato rispetto alla copia locale: blocco ambiguo')
    else:
        raw = fetch_raw(worker, block_id, target_path)
    nraw = validate_raw(raw, name=name, arch=arch)
    fetch_locust(block_id, target_path)
    full = target_path / 'paper-rf-features.csv'
    selected = target_path / 'selected.csv'
    counts_before = counts(dest, c)
    remaining = c['target'] - counts_before[arch, name]
    rows, chosen = export_select(exporter, raw, full, selected, block_id,
                                 min(remaining, c['block']))
    for row in rows:
        if row['function_name'] != name or row['machine_tag'] != arch or float(row['configured_cpus']) != 1:
            err('Metadati CSV exporter incoerenti')
    if len(chosen) == 0:
        err(f'{block_id}: exporter RF non ha prodotto alcun campione utilizzabile')
    previously_used = {rid for rec in completed_blocks(dest, c)
                       if rec['arch'] == arch and rec['function'] == name
                       for rid in rec['selected_ids']}
    if any(row['request_id'] in previously_used for row in chosen):
        err('Request ID riutilizzato in due blocchi della stessa funzione')
    item = dict(pin=PIN, schema=SCHEMA, id=block_id, arch=arch, function=name,
        round=pending['round'], recorded_at=now(), raw_count=nraw,
        rf_valid_count=len(rows), selected_count=len(chosen), selected_ids=[r['request_id'] for r in chosen],
        sha256={f: digest(target_path / f) for f in ('raw.jsonl','paper-rf-features.csv','selected.csv')})
    # Checkpoint atomico prima della rotazione: se crash, --recover lo completa.
    write_json(checkpoint, item)
    rotate_remote(worker, block_id, item['sha256']['raw.jsonl'])
    pending_path(dest).unlink()
    print(f'PASS: {block_id}: raw={nraw}, RF={len(rows)}, selezionati={len(chosen)}')


def run_next(dest, c, repo, max_new_blocks=0):
    if pending_path(dest).exists():
        err('Esiste pending.json: eseguire --recover prima di ripartire')
    _, phase = phases(dest, c)
    if phase is None:
        print('Campagna completa, nessuna invocazione richiesta')
        return
    exporter = ensure_binaries(repo)
    rnd, arch, queue = phase
    worker = start_arch(repo, arch)
    lbip = internal_ip('sl-lb')
    print(f'PHASE round={rnd} arch={arch} funzioni={len(queue)}', flush=True)
    baseline = worker_status(worker)
    if raw_exists(worker):
        err(f'{worker}: raw preesistente; non sovrascrivere. Investigare residui prima della raccolta')
    for i, entry in enumerate(queue):
        if max_new_blocks and i >= max_new_blocks:
            print(f'STOP volontario: limite {max_new_blocks} nuovi blocchi raggiunto, riprendibile')
            break
        if pending_path(dest).exists():
            err('Blocco pendente; interrompo')
        saved = counts(dest, c)[arch, entry['name']]
        missing = c['target'] - saved
        if missing <= 0:
            continue
        if raw_exists(worker):
            err(f'{worker}: raw inatteso prima di {entry["name"]}')
        if worker_status(worker) != baseline:
            err(f'{worker}: accounting deviato dal baseline, stop')
        needed = min(missing, c['block'])
        max_reqs = needed + max(3, needed // 4) + 1  # 1 cold + margine di retry
        block_id = f'r{rnd:03d}_{arch}_{entry["name"]}'
        pending = dict(id=block_id, function=entry['name'], arch=arch, round=rnd,
                       started_at=now(), planned_warm=needed, max_requests=max_reqs)
        write_json(pending_path(dest), pending)
        print(f'INIZIO: {block_id} ({saved}/{c["target"]}, max requests={max_reqs})', flush=True)
        register(entry, lbip)
        if worker_status(worker) != baseline or remote(worker, 'sudo docker ps -q | wc -l') != '0':
            err('Worker non al baseline dopo la registrazione')
        invoke(entry, lbip, block_id, max_reqs, needed)
        drain(worker, baseline)
        commit_pending(dest, c, exporter, pending)
    print(f'PASS: fase {rnd}/{arch} completata')


def abort_empty(dest, c, repo):
    pending_file = pending_path(dest)
    if not pending_file.exists():
        err('Nessun blocco pending da annullare')
    pending = read_json(pending_file)
    worker = EXPECTED[pending['arch']]
    # L'annullamento e' ammesso SOLO se si puo' dimostrare che non esistono dati.
    check_running(pending['arch'])
    if remote('sl-workload', 'pgrep -f "[l]ocust.*locustfile_collection.py" >/dev/null && echo ACTIVE || echo IDLE') != 'IDLE':
        err('Locust attivo: non posso annullare')
    if raw_exists(worker):
        err('Esiste raw sul worker: usare --recover, NON --abort-empty')
    block = dest / 'blocks' / pending['arch'] / f'round_{pending["round"]:03d}' / pending['function']
    if block.exists() and any(block.iterdir()):
        err('Esistono artefatti locali per il blocco: non annullo automaticamente')
    pending_file.unlink()
    print('PASS: blocco privo di raw annullato; nessun campione scartato')


def recover(dest, c, repo):
    if not pending_path(dest).exists():
        err('Nessun blocco pending')
    exporter = ensure_binaries(repo)
    pending = read_json(pending_path(dest))
    check_running(pending['arch'])
    print(f'RECOVERY EXPLICIT: {pending["id"]}', flush=True)
    commit_pending(dest, c, exporter, pending, recovery=True)


def report(dest, c):
    rec = completed_blocks(dest, c)
    progress = counts(dest, c)
    lines = ['arch,function,warm_rf_selected,target']
    for arch in ARCHS:
        for e in c['entries']:
            lines.append(f'{arch},{e["name"]},{progress[arch,e["name"]]},{c["target"]}')
    (dest / 'progress.csv').write_text('\n'.join(lines) + '\n')
    show_plan(dest, c)
    print(f'Checkpoint verificati: {len(rec)}; riepilogo: {dest / "progress.csv"}')


def export_dataset(dest, c):
    if pending_path(dest).exists():
        err("Un blocco e' ancora pendente")
    progress = counts(dest, c)
    if not all(v == c['target'] for v in progress.values()):
        err('Campagna incompleta; il dataset finale si esporta solo al target')
    records = sorted(completed_blocks(dest, c), key=lambda x: (x['arch'], x['function'], x['round']))
    out = dest / 'paper-rf-dataset.csv'
    provenance = dest / 'paper-rf-provenance.csv'
    headers = None
    count = 0
    with out.open('w', newline='') as d, provenance.open('w', newline='') as p:
        datawriter = None
        proof = csv.writer(p)
        proof.writerow(['arch','function','request_id','block_id','round'])
        for rec in records:
            source = dest/'blocks'/rec['arch']/f'round_{rec["round"]:03d}'/rec['function']/'selected.csv'
            with source.open(newline='') as f:
                reader = csv.DictReader(f)
                if headers is None:
                    headers = reader.fieldnames
                    if headers is None or len(headers) != 22 or not META_COLUMNS.issubset(headers):
                        err('Intestazione exporter inattesa nel CSV selezionato')
                    datawriter = csv.DictWriter(d, headers)
                    datawriter.writeheader()
                elif headers != reader.fieldnames:
                    err(f'CSV con intestazioni diverse: {source}')
                rows = list(reader)
                if len(rows) != rec['selected_count']:
                    err(f'Numero righe incoerente: {source}')
                for row in rows:
                    if row['machine_tag'] != rec['arch'] or row['function_name'] != rec['function']:
                        err(f'Metadati incoerenti nel dataset: {source}')
                    datawriter.writerow(row)
                    proof.writerow([rec['arch'], rec['function'], row['request_id'], rec['id'], rec['round']])
                    count += 1
    expected = len(c['entries'])*2*c['target']
    if count != expected:
        err(f'Dataset con {count} righe, attese {expected}')
    write_json(dest/'dataset_manifest.json', dict(pin=PIN, rows=count, exported_at=now(),
        dataset_sha256=digest(out), provenance_sha256=digest(provenance),
        campaign_manifest_sha256=digest(dest/'campaign.json')))
    print(f'PASS: dataset finale {out} ({count} righe); SHA256 salvati')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init','plan','report','run-next-phase','recover','abort-empty','export-dataset'))
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--target', type=int, default=100)
    parser.add_argument('--block', type=int, default=20)
    parser.add_argument('--only', default='', help='Sottoinsieme funzioni, SOLO con init')
    parser.add_argument('--confirm', action='store_true', help='Richiesto per operazioni su GCP')
    parser.add_argument('--max-new-blocks', type=int, default=0, help='Limite di sicurezza per --run-next-phase; 0=senza limite')
    args = parser.parse_args()
    if not 1 <= args.block <= args.target <= 1000 or args.max_new_blocks < 0:
        err('Target/block/limite non validi')
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / '.collector.lock'
    with lock.open('w') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            err('Altro collector attivo sulla stessa campagna')
        if args.action == 'init':
            # La directory ora contiene soltanto il file lock, che non e' un manifest.
            if (dest / 'campaign.json').exists() or (dest / 'blocks').exists():
                err('Campagna già inizializzata')
            initialize_existing(dest, args.target, args.block, args.only)
            return
        c = load(dest)
        if args.action == 'plan':
            show_plan(dest, c)
        elif args.action == 'report':
            report(dest, c)
        elif args.action == 'export-dataset':
            export_dataset(dest, c)
        else:
            if not args.confirm:
                err('Operazione GCP: aggiungere --confirm')
            if args.action == 'run-next-phase':
                run_next(dest, c, ROOT, args.max_new_blocks)
            elif args.action == 'recover':
                recover(dest, c, ROOT)
            else:
                abort_empty(dest, c, ROOT)


def initialize_existing(dest, target, block, only):
    if git('rev-parse', 'HEAD') != PIN:
        err('HEAD locale diverso da quello congelato')
    source = pinned_catalog()
    entries = read_catalog(source.decode())
    if only:
        chosen = only.split(',')
        if len(set(chosen)) != len(chosen):
            err('Nomi duplicati in --only')
        if not set(chosen).issubset({e['name'] for e in entries}):
            err('--only contiene funzione sconosciuta')
        entries = [e for e in entries if e['name'] in chosen]
    if not entries:
        err('Corpus vuoto')
    write_json(dest / 'campaign.json', dict(schema=SCHEMA, pin=PIN, project=PROJECT,
        zone=ZONE, cpu=1, policy='RoundRobin', target=target, block=block,
        worker_amd64='n2-standard-4', worker_arm64='t2a-standard-4',
        pool_memory_mb=12000, janitor_interval_s=5, container_expiration_s=10,
        catalog_sha256=sha(source), created_at=now(), entries=entries))
    print(f'PASS: manifest inizializzato con {len(entries)} funzioni, {len(entries)*2*target} warm validi RF target')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
