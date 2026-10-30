#!/usr/bin/env bash
# Serverledge memory campaign: resume a frozen V3 full-01 until 180/180.
# Does NOT provision/deploy; does NOT alter experiment, prior, or runtime.
# Locks the same campaign lock used by gcp_memory_auto_batches.sh, so it
# can safely wait for the existing short-run supervisor to finish.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "$ROOT_DIR"

WRAPPER="${SCRIPT_DIR}/gcp_memory.sh"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv-analysis/bin/python}"
DEST="data/profiling/gcp-memory/cloud-runs/actual-memory-01"
CONCURRENCY=3
BATCH_SIZE=9
MAX_HOURS=48
FINISH_ON_SUCCESS=0

usage() {
  cat <<'EOF'
Uso: ./scripts/gcp_scripts/gcp_memory_until_complete.sh [opzioni]
  --dest PATH           Campagna gia' deployata (default: full-01)
  --batch-size N        Condizioni nuove per riavvio, 1..20 (default: 9)
  --max-hours N         Limite di tempo globale, 1..72 ore (default: 48)
  --finish-on-success   Dopo audit 180/180, DELETE delle sole 5 VM possedute
  --help                Mostra questa guida

Se gcp_memory_auto_batches.sh e' in corso, attende il suo lock e poi
riprende dai checkpoint: mai due supervisori contemporaneamente.
Per ogni blocco: verifica checkpoint/VM -> resume -> run -> STOP verificato.
In caso di errore o budget di tempo esaurito: STOP, dati salvati, nessuna
ripetizione automatica della misura incompleta. Non crea ne' redeploya VM.
EOF
}

positive_integer() { [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]; }

while (( $# > 0 )); do
  case "$1" in
    --dest|--batch-size|--max-hours)
      (( $# >= 2 )) || { echo "Manca il valore per $1" >&2; exit 2; }
      case "$1" in
        --dest) DEST="$2" ;;
        --batch-size) BATCH_SIZE="$2" ;;
        --max-hours) MAX_HOURS="$2" ;;
      esac
      shift 2 ;;
    --finish-on-success) FINISH_ON_SUCCESS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opzione sconosciuta: $1" >&2; usage >&2; exit 2 ;;
  esac
done

positive_integer "$BATCH_SIZE" && (( BATCH_SIZE <= 20 )) || {
  echo "--batch-size deve essere tra 1 e 20" >&2; exit 2;
}
positive_integer "$MAX_HOURS" && (( MAX_HOURS <= 72 )) || {
  echo "--max-hours deve essere tra 1 e 72" >&2; exit 2;
}
[[ -x "$PYTHON" && -x "$WRAPPER" ]] || {
  echo "Python del virtualenv o wrapper gcp_memory.sh non disponibile" >&2; exit 1;
}
[[ -f "$DEST/campaign.json" ]] || {
  echo "Manifest campagna non trovato: $DEST" >&2; exit 1;
}
command -v flock >/dev/null || { echo "Richiesto flock (util-linux)" >&2; exit 1; }
DEST="$(realpath -e -- "$DEST")"

# NON chiamare GCP e NON installare un trap STOP prima di acquisire il lock:
# se il vecchio supervisore e' attivo, e' lui il responsabile delle VM.
exec 9>"${DEST}/.auto-batches.lock"
echo "Attendo uso esclusivo della campagna: $DEST"
flock -x 9
echo "PASS lock acquisito; nessun altro supervisore puo' avviare batch"

mkdir -p -- "$DEST/automation-logs"
LOG="${DEST}/automation-logs/$(date -u +%Y%m%dT%H%M%SZ)-until-complete-$$.log"
exec > >(tee -a -- "$LOG") 2>&1
printf 'Log completo: %s\n' "$LOG"

# Run from original verified source with read-only VM owner check. A surprising
# folder (e.g. incomplete run) blocks all further GCP activity.
read_state() {
  "$PYTHON" - "$DEST" "$CONCURRENCY" <<'PY'
from pathlib import Path
import sys
from analysis.profiling.gcp_memory_campaign import (
    VMS,
    ACTUAL_MEMORY_CONFIG_IDS,
    campaign,
    vm_guard,
    expected_keys,
    verified,
)
root = Path(sys.argv[1])
q = int(sys.argv[2])
doc = campaign(root)
assert doc['mode'] == 'full', 'Expected a full campaign'
assert doc.get('provisioned') is True and doc.get('deployed') is True, 'Not deployed'
assert doc.get('run_concurrency') == q, 'Frozen concurrency mismatch'
assert set(doc.get('vm_ids', {})) == set(VMS), 'VM ownership mismatch'
cloud = vm_guard(doc, require_running=False)
states = {name: cloud[name]['status'] for name in VMS}
assert all(s == 'TERMINATED' for s in states.values()), f'VM not stopped: {states}'
keys = expected_keys(doc, q)
known = set(keys)
runs = root / 'runs'
if runs.exists():
    for folder in runs.iterdir():
        if folder.is_dir():
            assert folder.name in known, f'Unexpected run folder: {folder.name}'
            assert verified(folder), f'Incomplete/tampered measurement: {folder.name}'
completed = sum(verified(runs / key) for key in keys)
assert doc["replicas"] == 2
assert len(doc["targets"]) == 10

assert doc["configs"] == list(ACTUAL_MEMORY_CONFIG_IDS), (
    "Unexpected experimental conditions"
)

assert len(keys) == 120, (
    f"Expected 120 measurements, found {len(keys)}"
)
print(completed, len(keys))
PY
}

CLOUD_MAY_BE_RUNNING=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  trap '' INT TERM
  if (( CLOUD_MAY_BE_RUNNING )); then
    echo '===== STOP EMERGENZA ====='
    if ! "$WRAPPER" stop --dest "$DEST" --confirm; then
      echo 'CRITICO: STOP non verificato; controllare GCP manualmente subito.' >&2
      (( status != 0 )) || status=1
    fi
  fi
  printf 'Log della sessione: %s\n' "$LOG"
  exit "$status"
}
trap cleanup EXIT
trap 'echo "Interruzione richiesta" >&2; exit 130' INT
trap 'echo "Terminazione richiesta" >&2; exit 143' TERM

state="$(read_state)"
read -r completed total <<< "$state"
echo "Stato iniziale: $completed/$total; concorrenza $CONCURRENCY; cinque VM TERMINATED"
echo "Batch da $BATCH_SIZE, tempo massimo $MAX_HOURS ore, finish_on_success=$FINISH_ON_SUCCESS"

started_at="$(date +%s)"
batch=0
while (( completed < total )); do
  elapsed_s=$(( $(date +%s) - started_at ))
  if (( elapsed_s >= MAX_HOURS * 3600 )); then
    echo "STOP: limite di ${MAX_HOURS} ore raggiunto; $completed/$total validati." >&2
    echo 'Nessun nuovo batch avviato. Le VM erano state verificate TERMINATED.' >&2
    exit 3
  fi
  batch=$((batch + 1))
  echo "===== BATCH $batch: $completed/$total verificati ====="

  # Set before resume: even if startup partially succeeds, the EXIT trap
  # attempts a guarded STOP. run --stop-after also does its own STOP.
  CLOUD_MAY_BE_RUNNING=1
  "$WRAPPER" resume --dest "$DEST" --confirm
  "$WRAPPER" run --dest "$DEST" --concurrency "$CONCURRENCY" \
    --max-runs "$BATCH_SIZE" --stop-after --confirm

  state="$(read_state)"
  read -r after total <<< "$state"
  if (( after <= completed || after - completed > BATCH_SIZE )); then
    echo "ERRORE: avanzamento checkpoint inatteso ($completed -> $after)" >&2
    exit 1
  fi
  CLOUD_MAY_BE_RUNNING=0
  echo "PASS: +$((after-completed)) misurazioni; $after/$total; VM TERMINATED"
  completed="$after"
done

echo '===== CAMPAGNA COMPLETA: audit rigoroso dei 180 risultati grezzi ====='
"$WRAPPER" audit-grid --dest "$DEST" --concurrency "$CONCURRENCY"
echo 'PASS AUDIT-GRID: 120/120; 6000 invocazioni dichiarate'

if (( FINISH_ON_SUCCESS )); then
  echo '===== FINISH SICURO: DELETE solo dopo AUDIT COMPLETO ====='
  "$WRAPPER" finish --dest "$DEST" --concurrency "$CONCURRENCY" --confirm
  echo 'PASS FINISH: campagna archiviata localmente; VM e boot disk cancellati'
else
  echo 'PASS: VM TERMINATED, NON eliminate. Eseguire finish manualmente quando opportuno.'
fi

echo "===== COMPLETATO; LOG: $LOG ====="
