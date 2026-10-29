#!/usr/bin/env bash
# Esegue piu' batch della campagna GCP memoria gia' provisioned/deployed.
# NON crea VM, NON cambia MAB/prior, NON cancella le VM.
# Ogni batch: resume -> run con checkpoint -> STOP -> verifica GCP.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

WRAPPER="${SCRIPT_DIR}/gcp_memory.sh"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv-analysis/bin/python}"
DEST="data/profiling/gcp-memory/cloud-runs/full-01"
CONCURRENCY=3
BATCH_SIZE=9
BATCHES=3
MAX_MINUTES=180
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Uso: ./scripts/gcp_scripts/gcp_memory_auto_batches.sh [opzioni]
  --dest PATH         Directory campagna (default: full-01)
  --batch-size N      Nuove misurazioni per batch, 1..20 (default: 9)
  --batches N         Numero massimo di batch, 1..6 (default: 3)
  --max-minutes N     Non iniziare nuovi batch dopo N minuti (default: 180)
  --check-only        Verifica manifest, checkpoint e VM; non avvia GCP
  --help              Mostra questa guida

Prerequisiti: full-01 gia' provisioned/deployed; 5 VM STOPPED.
Ogni batch lascia le VM STOPPED. Non esegue finish/DELETE.
In caso di errore: STOP di emergenza; il batch si interrompe.
EOF
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (( $# )); do
  case "$1" in
    --dest|--batch-size|--batches|--max-minutes)
      (( $# >= 2 )) || { echo "Valore mancante per $1" >&2; exit 2; }
      case "$1" in
        --dest) DEST="$2" ;;
        --batch-size) BATCH_SIZE="$2" ;;
        --batches) BATCHES="$2" ;;
        --max-minutes) MAX_MINUTES="$2" ;;
      esac
      shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opzione sconosciuta: $1" >&2; usage >&2; exit 2 ;;
  esac
done

positive_integer "$BATCH_SIZE" && (( BATCH_SIZE <= 20 )) || {
  echo "--batch-size deve essere tra 1 e 20" >&2; exit 2;
}
positive_integer "$BATCHES" && (( BATCHES <= 6 )) || {
  echo "--batches deve essere tra 1 e 6" >&2; exit 2;
}
positive_integer "$MAX_MINUTES" || { echo "--max-minutes deve essere positivo" >&2; exit 2; }

[[ -x "$PYTHON" && -x "$WRAPPER" ]] || {
  echo "Python virtualenv o wrapper gcp_memory.sh non disponibile" >&2; exit 1;
}
[[ -f "$DEST/campaign.json" ]] || { echo "Manifest non trovato: $DEST" >&2; exit 1; }
DEST="$(realpath -e -- "$DEST")"

# Un solo supervisore locale per campagna, anche fra diversi terminali.
command -v flock >/dev/null || { echo "Richiesto flock (util-linux)" >&2; exit 1; }
exec 9>"${DEST}/.auto-batches.lock"
flock -n 9 || { echo "Un altro supervisore ha gia' il lock di questa campagna" >&2; exit 1; }

# Tutti i controlli dello stato interpellano GCP in sola lettura e verificano
# ID, label di proprieta', stato STOPPED e checksum dei checkpoint.
read_state() {
  "$PYTHON" - "$DEST" "$CONCURRENCY" <<'PY'
import sys
from pathlib import Path
from analysis.profiling.gcp_memory_campaign import (
    VMS, campaign, vm_guard, expected_keys, verified,
)

root = Path(sys.argv[1])
concurrency = int(sys.argv[2])
doc = campaign(root)
assert doc['mode'] == 'full', 'Il manifest non e\' full'
assert doc.get('provisioned') is True and doc.get('deployed') is True, 'Provisioning/deploy non completati'
assert doc.get('run_concurrency') == concurrency, 'Concorrenza congelata differente'
assert set(doc.get('vm_ids', {})) == set(VMS), 'VM non coincidenti con le cinque previste'
cloud = vm_guard(doc, require_running=False)
states = {name: cloud[name]['status'] for name in VMS}
assert all(state == 'TERMINATED' for state in states.values()), f'VM non tutte ferme: {states}'
keys = expected_keys(doc, concurrency)
known = set(keys)
runs = root / 'runs'
if runs.exists():
    for folder in runs.iterdir():
        if folder.is_dir():
            assert folder.name in known, f'Directory di misura inattesa: {folder.name}'
            assert verified(folder), f'Misura incompleta/alterata: {folder.name}'
completed = sum(verified(runs / key) for key in keys)
assert completed <= len(keys)
print(completed, len(keys))
PY
}

# Valida la situazione PRIMA di accendere alcuna VM.
state="$(read_state)"
read -r completed total <<< "$state"
echo "Campagna: $DEST"
echo "Concorrenza: $CONCURRENCY; checkpoint: $completed/$total; cinque VM TERMINATED"

if (( CHECK_ONLY )); then
  echo "PASS CHECK-ONLY: nessuna VM avviata"
  exit 0
fi

mkdir -p -- "$DEST/automation-logs"
LOG="${DEST}/automation-logs/$(date -u +%Y%m%dT%H%M%SZ)-auto-batches.log"
exec > >(tee -a "$LOG") 2>&1
printf 'Log sessione: %s\n' "$LOG"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  trap '' INT TERM
  printf '\n===== STOP DI SICUREZZA =====\n'
  if ! "$WRAPPER" stop --dest "$DEST" --confirm; then
    echo "CRITICO: STOP non verificato. Controllare immediatamente GCP." >&2
    (( code != 0 )) || code=1
  fi
  printf 'Log: %s\n' "$LOG"
  exit "$code"
}
trap cleanup EXIT
trap 'echo "Interruzione richiesta" >&2; exit 130' INT
trap 'echo "Terminazione richiesta" >&2; exit 143' TERM

started_at="$(date +%s)"
for (( batch=1; batch<=BATCHES; batch++ )); do
  state="$(read_state)"
  read -r completed total <<< "$state"
  if (( completed == total )); then
    echo "Tutte le $total misurazioni sono gia' validate. Nessun nuovo avvio."
    break
  fi
  elapsed_minutes=$(( ($(date +%s) - started_at) / 60 ))
  if (( elapsed_minutes >= MAX_MINUTES )); then
    echo "Limite sessione raggiunto (${elapsed_minutes}m >= ${MAX_MINUTES}m). Nessun nuovo batch."
    break
  fi

  printf '\n===== BATCH %d/%d | %d/%d VALIDATI =====\n' \
    "$batch" "$BATCHES" "$completed" "$total"
  "$WRAPPER" resume --dest "$DEST" --confirm
  "$WRAPPER" run --dest "$DEST" --concurrency "$CONCURRENCY" \
    --max-runs "$BATCH_SIZE" --stop-after --confirm

  state="$(read_state)"
  read -r after total <<< "$state"
  if (( after <= completed )); then
    echo "ERRORE: il batch non ha prodotto nuovi checkpoint verificati" >&2
    exit 1
  fi
  echo "PASS BLOCCO: +$((after-completed)) misurazioni; $after/$total validate; VM TERMINATED"
done

echo "===== SESSIONE AUTOMATICA COMPLETATA; VM FERME ====="
echo "Le VM NON sono cancellate: terminata l'intera campagna usare audit-grid e finish."
