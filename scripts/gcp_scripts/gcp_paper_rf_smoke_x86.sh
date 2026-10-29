#!/usr/bin/env bash
# Smoke test paper-RF/schema 6 per il cluster GCP gia' avviato su amd64.
# Non modifica ne' ricompila il runtime; non cancella raw o container.
# Esegue al massimo 6 invocazioni di hashing (CPU=1), scarica il raw,
# lo esporta con il binario paper-RF compilato dal commit congelato.
#
# Uso:
#   bash scripts/gcp_scripts/gcp_paper_rf_smoke_x86.sh --check
#   PAPER_RF_SMOKE_CONFIRM=YES bash scripts/gcp_scripts/gcp_paper_rf_smoke_x86.sh --run
#
# ATTENZIONE: --run registra una funzione in etcd ed esegue workload su GCP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/gcp_config.sh"

PIN='9f9e9cfc5631b41f525f6d91fadad4415b93a8c1'
WORKER='sl-x86-1'
INACTIVE='sl-arm-1'
FUNCTION='hashing'
RUNTIME='go125'
SRC='functions/bundles/hashing.tar'
MEMORY_MB='1024'
CPU='1'
MAX_REQUESTS='6'
LB_POLICY='RoundRobin'
RAW_REMOTE='/var/lib/serverledge/profiling-samples.jsonl'
BUILD_DIR="${ROOT_DIR}/build/gcp-paper-rf-schema6-${PIN:0:12}"
EXPORTER="${BUILD_DIR}/bin/serverledge-profiling-amd64"
MODE="${1:-}"

info() { printf 'INFO: %s\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Uso: gcp_paper_rf_smoke_x86.sh --check
     PAPER_RF_SMOKE_CONFIRM=YES gcp_paper_rf_smoke_x86.sh --run
--check non crea funzioni e non esegue invocazioni; --run e' un test limitato
alla funzione hashing: 6 richieste massime, RAW conservato, CSV esportato.
EOF
}

case "$MODE" in
  --check|--run) [[ "$#" -eq 1 ]] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac

[[ "$PROJECT" == 'serverledge-tesi-ale' ]] || fail "Progetto diverso dall'atteso: $PROJECT"
[[ "$ZONE" == 'europe-west4-a' ]] || fail "Zona diversa dall'attesa: $ZONE"
[[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" == "$PIN" ]] || fail 'Il repository locale non e al commit congelato'
[[ -x "$EXPORTER" ]] || fail "Exporter locale non trovato: $EXPORTER"
[[ -f "${ROOT_DIR}/examples/experiments/locustfile_collection.py" ]] || fail 'Locustfile mancante nel repository'
[[ -f "${ROOT_DIR}/scripts/gcp_scripts/functions_catalog.sh" ]] || fail 'Catalogo mancante'
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/functions_catalog.sh"
expected="${FUNCTION}|${RUNTIME}|${SRC}|${MEMORY_MB}|"
found=0
for item in "${FUNCTIONS[@]}"; do
  [[ "$item" == "$expected" ]] && found=1
done
[[ "$found" == 1 ]] || fail "Specifiche catalogo non coerenti: $expected"

gcloud_ssh() {
  local host="$1"; shift
  gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

ip_of() {
  gc compute instances describe "$1" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

preflight() {
  info 'Verifico versioni e binari prima di toccare il workload'
  bash "${SCRIPT_DIR}/gcp_paper_rf_deploy.sh" --verify
  local host
  for host in "$NAME_REGISTRY" "$NAME_LB" "$NAME_WORKLOAD" "$WORKER" "$INACTIVE"; do
    [[ "$(gc compute instances describe "$host" --zone="$ZONE" --format='value(status)')" == RUNNING ]] \
      || fail "$host non RUNNING"
  done
  [[ "$(gcloud_ssh "$WORKER" 'uname -m' | tr -d '[:space:]')" == x86_64 ]] || fail 'Worker attivo non x86_64'
  [[ "$(gcloud_ssh "$INACTIVE" 'pgrep -xc serverledge || true' | tr -d '[:space:]')" == 0 ]] || fail 'Worker ARM e attivo: test non isolato'
  gcloud_ssh "$WORKER" 'pgrep -x serverledge >/dev/null && curl -fsS --max-time 5 http://localhost:1323/status >/dev/null' \
    || fail 'Worker x86 non sano'
  gcloud_ssh "$NAME_LB" 'pgrep -x lb >/dev/null && sudo ss -lnt | grep -q ":1323 "' \
    || fail 'Load balancer non sano'
  local registered_x86 registered_arm
  registered_x86="$(gcloud_ssh "$NAME_REGISTRY" \
    'sudo docker exec Etcd-server etcdctl get registry/ --prefix --keys-only | grep -Ec "^registry/cloud-region/amd64/[^/]+$" || true' \
    | tr -d '[:space:]')"
  registered_arm="$(gcloud_ssh "$NAME_REGISTRY" \
    'sudo docker exec Etcd-server etcdctl get registry/ --prefix --keys-only | grep -Ec "^registry/cloud-region/arm64/[^/]+$" || true' \
    | tr -d '[:space:]')"
  [[ "$registered_x86" == 1 && "$registered_arm" == 0 ]] || fail "Registry inatteso: amd64=$registered_x86 (atteso 1), arm64=$registered_arm (atteso 0)"
  pass 'Registry: un worker amd64, nessun worker ARM (LB escluso)'
  gcloud_ssh "$NAME_WORKLOAD" "test -f '/opt/serverledge/examples/experiments/${SRC}' && test -x /opt/serverledge/bin/serverledge-cli && test -f /opt/serverledge/examples/experiments/locustfile_collection.py" \
    || fail 'Source, CLI o locustfile mancante su sl-workload'
  pass 'Preflight smoke x86 completato: worker unico, LB, registry, sorgenti presenti'
}

preflight
if [[ "$MODE" == --check ]]; then
  info 'Nessuna risorsa GCP modificata'
  exit 0
fi

[[ "${PAPER_RF_SMOKE_CONFIRM:-}" == YES ]] || fail 'Aggiungere PAPER_RF_SMOKE_CONFIRM=YES per eseguire 6 invocazioni'
# Evita di mescolare questo smoke con campioni gia presenti in un altro run.
raw_bytes="$(gcloud_ssh "$WORKER" \
  "sudo sh -c 'if test -e ${RAW_REMOTE}; then wc -c < ${RAW_REMOTE}; else echo 0; fi'" \
  | tr -d '[:space:]')"
[[ "$raw_bytes" =~ ^[0-9]+$ ]] || fail "Dimensione raw non interpretabile: $raw_bytes"
[[ "$raw_bytes" == 0 ]] || fail "Raw preesistente di $raw_bytes byte: fermo il test per non mescolare esperimenti; salvare prima il file"

run_id="paper_rf_smoke_x86_$(date -u +%Y%m%dT%H%M%SZ)"
local_dir="${ROOT_DIR}/data/profiling/gcp-paper-rf-schema6/smoke/${run_id}"
mkdir -p "${local_dir}/raw" "${local_dir}/locust"
lb_ip="$(ip_of "$NAME_LB")"
[[ "$lb_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "IP LB interno non valido: $lb_ip"
info "Result directory: $local_dir"
info "Registrazione ${FUNCTION}: ${RUNTIME}, CPU=${CPU}, memory=${MEMORY_MB}MB"

# Tutti i parametri sono costanti letterali e non provengono da input libero.
# SSH con stdin preserva l'esatta sequenza delle operazioni remote.
gc compute ssh "$NAME_WORKLOAD" --zone="$ZONE" --quiet \
  --command="bash -se -- '$lb_ip'" <<'REMOTE_REGISTER'
set -euo pipefail
LB_IP="$1"
cd /opt/serverledge/examples/experiments
export SERVERLEDGE_HOST="$LB_IP" SERVERLEDGE_PORT=1323
/opt/serverledge/bin/serverledge-cli create -f hashing \
  --runtime go125 \
  --src functions/bundles/hashing.tar \
  --memory 1024 \
  --cpu 1 \
  --update
REMOTE_REGISTER

info 'Eseguo al massimo 6 invocazioni seriali con Locust, senza target warm anticipato'
# In caso di errore delle invocazioni si tenta comunque di salvare il raw.
collect_started=1
collect_artifacts() {
  local status="$1"
  [[ "${collect_started:-0}" == 1 ]] || return 0
  info 'Provo a recuperare i risultati parziali dello smoke test'
  # Il raw originale non viene ne eliminato ne alterato.
  gc compute ssh "$WORKER" --zone="$ZONE" --quiet \
    --command="bash -se -- '$run_id'" <<'REMOTE_SNAPSHOT' || true
set -euo pipefail
RUN_ID="$1"
mkdir -p "$HOME/paper-rf-smoke/$RUN_ID"
if sudo test -s /var/lib/serverledge/profiling-samples.jsonl; then
  sudo install -m 0600 -o "$(id -un)" -g "$(id -gn)" \
    /var/lib/serverledge/profiling-samples.jsonl \
    "$HOME/paper-rf-smoke/$RUN_ID/profiling-samples.jsonl"
fi
REMOTE_SNAPSHOT
  gc compute scp --zone="$ZONE" --quiet \
    "$WORKER:~/paper-rf-smoke/$run_id/profiling-samples.jsonl" \
    "${local_dir}/raw/sl-x86-1.jsonl" || true
  gc compute scp --zone="$ZONE" --quiet --recurse \
    "$NAME_WORKLOAD:~/paper-rf-smoke/$run_id" \
    "${local_dir}/locust/" || true
  if [[ "$status" -ne 0 ]]; then
    printf 'Smoke interrotto (exit %s); raw originali sul worker non modificati.\n' "$status" >&2
  fi
}
trap 'status=$?; trap - EXIT; collect_artifacts "$status"; exit "$status"' EXIT

gc compute ssh "$NAME_WORKLOAD" --zone="$ZONE" --quiet \
  --command="bash -se -- '$lb_ip' '$run_id'" <<'REMOTE_LOCUST'
set -euo pipefail
LB_IP="$1"
RUN_ID="$2"
OUT="$HOME/paper-rf-smoke/$RUN_ID"
mkdir -p "$OUT"
if [[ ! -x "$HOME/locust-venv/bin/locust" ]]; then
  python3 -m venv "$HOME/locust-venv"
  "$HOME/locust-venv/bin/pip" install --quiet locust
fi
export FUNCTIONS_TO_RUN=hashing
export MAX_REQUESTS_PER_FUNCTION=6
export REQUESTS_PER_FUNCTION=6
export TARGET_WARM_VALID=0
export LB_POLICY=RoundRobin
export COLLECTION_CSV="$OUT/experiment_results.csv"
export COLLECTION_SUMMARY="$OUT/summary.csv"
"$HOME/locust-venv/bin/locust" \
  -f /opt/serverledge/examples/experiments/locustfile_collection.py \
  --headless --users 1 --spawn-rate 1 --run-time 10m \
  --host "http://${LB_IP}:1323" --csv "$OUT/locust"
REMOTE_LOCUST

# Salva i dati prima di interpretare: lo smoke e' evidenza, non dataset definitivo.
trap - EXIT
collect_artifacts 0
collect_started=0

raw_file="${local_dir}/raw/sl-x86-1.jsonl"
[[ -s "$raw_file" ]] || fail "Raw non scaricato: $raw_file"
output_csv="${local_dir}/paper-rf-features.csv"
"$EXPORTER" export-paper-rf-csv \
  --input "$raw_file" \
  --experiment-id "$run_id" \
  --output "$output_csv"

python3 - "$raw_file" "$output_csv" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

raw_path, csv_path = (Path(p) for p in sys.argv[1:])
samples = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
if not samples:
    raise SystemExit('ERROR: raw vuoto')
if any(s.get('schema_version') != 6 for s in samples):
    raise SystemExit('ERROR: raw non tutto schema 6')
if any(s.get('function_name') != 'hashing' or s.get('machine_tag') != 'amd64' for s in samples):
    raise SystemExit('ERROR: funzione o architettura inattesa nel raw')
if any(float((s.get('function_configuration') or {}).get('configured_cpus', -1)) != 1 for s in samples):
    raise SystemExit('ERROR: configured_cpus diverso da 1')
with csv_path.open(newline='') as f:
    rows = list(csv.DictReader(f))
if not 1 <= len(rows) <= 6:
    raise SystemExit(f'ERROR: righe CSV inattese: {len(rows)}')
for row in rows:
    if (row.get('function_name') != 'hashing'
            or row.get('machine_tag') != 'amd64'
            or float(row.get('configured_cpus', -1)) != 1):
        raise SystemExit('ERROR: metadati RF CSV inattesi')
    if len(row) != 22:
        raise SystemExit(f'ERROR: {len(row)} colonne, attese 22 (8 metadata + 14 feature)')
req_ids = [r['request_id'] for r in rows]
if len(set(req_ids)) != len(req_ids):
    raise SystemExit('ERROR: request_id duplicati nel CSV')
counts = Counter(('warm' if s['warm_start'] else 'cold') if s['execution_succeeded'] else 'failed' for s in samples)
print(f"PASS: raw_schema6={len(samples)}, cold={counts['cold']}, warm={counts['warm']}, failed={counts['failed']}, rf_valid_rows={len(rows)}")
print('PASS: schema/architettura/funzione/CPU e 14 feature coerenti')
PY

{
  printf 'experiment_id=%s\n' "$run_id"
  printf 'runtime_commit=%s\n' "$PIN"
  printf 'project=%s\nzone=%s\nworker=%s\n' "$PROJECT" "$ZONE" "$WORKER"
  printf 'function=%s\nruntime=%s\nconfigured_cpus=%s\nconfigured_memory_mb=%s\n' "$FUNCTION" "$RUNTIME" "$CPU" "$MEMORY_MB"
  printf 'max_requests=%s\n' "$MAX_REQUESTS"
  printf 'raw_sha256=%s\n' "$(sha256sum "$raw_file" | awk '{print $1}')"
  printf 'csv_sha256=%s\n' "$(sha256sum "$output_csv" | awk '{print $1}')"
} > "${local_dir}/manifest.txt"
pass "Smoke test x86 completato. Evidenze in: $local_dir"
info 'Il JSONL originale resta sul worker. NON mescolarlo con la futura campagna definitiva.'
