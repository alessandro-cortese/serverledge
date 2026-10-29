#!/usr/bin/env bash
# Avvio isolato del cluster paper-RF/schema 6 per un solo worker attivo.
# Uso:
#   bash scripts/gcp_scripts/gcp_paper_rf_start.sh --check
#   PAPER_RF_START_CONFIRM=YES bash scripts/gcp_scripts/gcp_paper_rf_start.sh --start amd64
#   PAPER_RF_START_CONFIRM=YES bash scripts/gcp_scripts/gcp_paper_rf_start.sh --start arm64
#
# Richiede VM gia' create + deploy verificato. Non esegue git pull, checkout,
# compilazioni, registrazioni funzioni, invocazioni, eliminazioni VM o raw.
# --start ricrea esplicitamente il container Etcd-server (registry vuoto).
# Eseguire soltanto tra blocchi, mai con una raccolta in corso.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/gcp_config.sh"

PIN='9f9e9cfc5631b41f525f6d91fadad4415b93a8c1'
VM_X86='sl-x86-1'
VM_ARM='sl-arm-1'
JANITOR_INTERVAL="${PAPER_RF_JANITOR_INTERVAL:-5}"
CONTAINER_EXPIRATION="${PAPER_RF_CONTAINER_EXPIRATION:-10}"
CONTAINER_POOL_MEMORY="${PAPER_RF_CONTAINER_POOL_MEMORY:-12000}"
VM_LIST=("$NAME_REGISTRY" "$NAME_LB" "$NAME_WORKLOAD" "$VM_X86" "$VM_ARM")

info() { printf 'INFO: %s\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Uso: gcp_paper_rf_start.sh --check
     PAPER_RF_START_CONFIRM=YES gcp_paper_rf_start.sh --start {amd64|arm64}
--check: controlli remoti di sola lettura, compresi commit e binari.
--start: arresta LB/entrambi i worker, ricrea etcd vuoto, avvia un solo
         worker con profiling schema 6 e avvia LB RoundRobin.
         Non cancella i file JSONL sui worker.
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
MODE="$1"
ARCH="${2:-}"
case "$MODE" in
  --check) [[ $# -eq 1 ]] || { usage; exit 2; } ;;
  --start) [[ $# -eq 2 && ( "$ARCH" == amd64 || "$ARCH" == arm64 ) ]] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac

[[ "${PROJECT}" == serverledge-tesi-ale ]] || fail "Progetto inatteso: $PROJECT"
[[ "${ZONE}" == europe-west4-a ]] || fail "Zona inattesa: $ZONE"
[[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" == "$PIN" ]] || fail 'HEAD locale non congelato'
[[ -f "$SCRIPT_DIR/gcp_paper_rf_deploy.sh" ]] || fail 'Manca il deploy script paper-RF'
for value in "$JANITOR_INTERVAL" "$CONTAINER_EXPIRATION" "$CONTAINER_POOL_MEMORY"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'Parametri di container/janitor non validi'
done

remote() {
  local host="$1"; shift
  gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

vm_status() {
  gc compute instances describe "$1" --zone="$ZONE" --format='value(status)' 2>/dev/null
}

ip_of() {
  gc compute instances describe "$1" --zone="$ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

# Controlli invarianti: niente git pull automatici; verifica la versione prima
# di ogni avvio e impedisce di entrare in una configurazione VM differente.
preflight() {
  command -v gcloud >/dev/null || fail 'gcloud non trovato'
  local vm
  for vm in "${VM_LIST[@]}"; do
    [[ "$(vm_status "$vm")" == RUNNING ]] || fail "VM $vm non RUNNING"
  done
  bash "$SCRIPT_DIR/gcp_paper_rf_deploy.sh" --verify
  for vm in "$VM_X86" "$VM_ARM"; do
    local ram
    ram="$(remote "$vm" "awk '/MemTotal:/ {printf \"%d\", \$2/1024}' /proc/meminfo" | tr -d '\r[:space:]')"
    [[ "$ram" =~ ^[0-9]+$ ]] || fail "$vm: RAM non leggibile ($ram)"
    (( CONTAINER_POOL_MEMORY < ram )) || fail "$vm: pool ${CONTAINER_POOL_MEMORY}MB >= RAM ${ram}MB"
    info "$vm: RAM=${ram}MB; pool=${CONTAINER_POOL_MEMORY}MB"
  done
}

check_readonly() {
  preflight
  for vm in "$VM_X86" "$VM_ARM"; do
    info "$vm: processo serverledge=$(remote "$vm" 'pgrep -xc serverledge || true' | tr -d '\r[:space:]'); container Docker attivi=$(remote "$vm" 'sudo docker ps -q | wc -l' | tr -d '\r[:space:]')"
    info "$vm: raw=$(remote "$vm" 'sudo sh -c '\''if test -f /var/lib/serverledge/profiling-samples.jsonl; then wc -c < /var/lib/serverledge/profiling-samples.jsonl; else echo assente; fi'\''' | tr -d '\r[:space:]') byte"
  done
  pass 'Controllo read-only completato (cluster non modificato)'
}

# Richiede entrambi i worker senza container attivi PRIMA di fermare servizi.
# La presenza di container indica che il blocco precedente non e' stato
# svuotato: interrompersi protegge i campioni e la contabilita' dei nodi.
require_drained() {
  local vm active
  for vm in "$VM_X86" "$VM_ARM"; do
    active="$(remote "$vm" 'sudo docker ps -q | wc -l' | tr -d '\r[:space:]')"
    [[ "$active" == 0 ]] || fail "$vm: $active container in esecuzione; terminare/recuperare il blocco prima del cambio architettura"
  done
}

# Valida una proprieta' osservabile via SSH, con tentativi limitati.
wait_remote() {
  local vm="$1" command="$2" label="$3" attempt
  for attempt in {1..20}; do
    if remote "$vm" "$command" >/dev/null 2>&1; then
      pass "$vm: $label"
      return 0
    fi
    sleep 2
  done
  fail "$vm: timeout per $label"
}

start_cluster() {
  [[ "${PAPER_RF_START_CONFIRM:-}" == YES ]] || fail 'Usare PAPER_RF_START_CONFIRM=YES per avviare'
  preflight
  require_drained

  local worker other expected_arch registry_ip lb_ip registered
  if [[ "$ARCH" == amd64 ]]; then
    worker="$VM_X86"; other="$VM_ARM"; expected_arch='x86_64'
  else
    worker="$VM_ARM"; other="$VM_X86"; expected_arch='aarch64'
  fi
  registry_ip="$(ip_of "$NAME_REGISTRY")"
  lb_ip="$(ip_of "$NAME_LB")"
  [[ "$registry_ip" =~ ^[0-9.]+$ && "$lb_ip" =~ ^[0-9.]+$ ]] || fail 'Indirizzi interni non validi'

  info "ATTIVO=$worker ($expected_arch), INATTIVO=$other; Etcd sara' ricreato vuoto"
  info 'Arresto LB e entrambi i worker (senza eliminare i raw o i container)'
  remote "$NAME_LB" 'sudo pkill -x lb || true'
  remote "$VM_X86" 'sudo pkill -x serverledge || true'
  remote "$VM_ARM" 'sudo pkill -x serverledge || true'
  sleep 2
  for vm in "$VM_X86" "$VM_ARM"; do
    [[ "$(remote "$vm" 'pgrep -xc serverledge || true' | tr -d '\r[:space:]')" == 0 ]] \
      || fail "$vm: serverledge ancora attivo"
  done

  info 'Riavvio etcd vuoto (solo registry, NON i file raw)'
  remote "$NAME_REGISTRY" "
    set -e
    sudo docker rm -f Etcd-server >/dev/null 2>&1 || true
    sudo docker run -d \\
      -p 2379:2379 -p 2380:2380 --name Etcd-server \\
      gcr.io/etcd-development/etcd:v3.5.13 \\
      /usr/local/bin/etcd \\
      --name s1 --data-dir /etcd-data \\
      --listen-client-urls http://0.0.0.0:2379 \\
      --advertise-client-urls http://0.0.0.0:2379 \\
      --listen-peer-urls http://0.0.0.0:2380 \\
      --initial-advertise-peer-urls http://0.0.0.0:2380 \\
      --initial-cluster s1=http://0.0.0.0:2380 \\
      --initial-cluster-token tkn --initial-cluster-state new \\
      --max-request-bytes 33554432 --quota-backend-bytes 8589934592 >/dev/null
  "
  wait_remote "$NAME_REGISTRY" \
    'sudo docker exec Etcd-server etcdctl endpoint health' 'etcd sano'

  info "Scrivo worker.yaml e avvio il SOLO worker $worker"
  remote "$worker" "
    set -e
    sudo tee /opt/serverledge/worker.yaml >/dev/null <<'EOF_WORKER'
registry:
  area: \"cloud-region\"
  udp.port: 9877
etcd:
  address: \"${registry_ip}:2379\"
container:
  pool:
    memory: ${CONTAINER_POOL_MEMORY}
  expiration: ${CONTAINER_EXPIRATION}
janitor:
  interval: ${JANITOR_INTERVAL}
profiling:
  enabled: true
  export:
    enabled: true
    path: \"/var/lib/serverledge/profiling-samples.jsonl\"
EOF_WORKER
    sudo install -d -m 0755 /var/lib/serverledge
    sudo sh -c 'cd /opt/serverledge && nohup ./bin/serverledge worker.yaml > /var/log/serverledge-node.log 2>&1 </dev/null &'
  "
  wait_remote "$worker" 'curl -fsS --max-time 5 http://localhost:1323/status >/dev/null' 'endpoint worker /status'
  registered="$(remote "$NAME_REGISTRY" \
    'sudo docker exec Etcd-server etcdctl get registry/ --prefix --keys-only | grep -c cloud-region || true' \
    | tr -d '\r[:space:]')"
  [[ "$registered" == 1 ]] || fail "Nodi registrati in etcd: $registered (atteso esattamente 1)"
  pass "Un solo worker registrato: $worker"
  [[ "$(remote "$other" 'pgrep -xc serverledge || true' | tr -d '\r[:space:]')" == 0 ]] \
    || fail "$other: processo serverledge inaspettato"

  info 'Avvio LB RoundRobin architecture-unaware, transfer disabilitato'
  remote "$NAME_LB" "
    set -e
    sudo tee /opt/serverledge/lb.yaml >/dev/null <<'EOF_LB'
registry:
  area: \"cloud-region\"
  udp.port: 9877
lb:
  replicas: 128
  mode: \"RoundRobin\"
  arch_awareness: false
  refresh_interval: 30
mab.policy: \"RoundRobin\"
mab:
  transfer:
    control:
      enabled: false
etcd:
  address: \"${registry_ip}:2379\"
EOF_LB
    sudo sh -c 'cd /opt/serverledge && nohup ./bin/lb lb.yaml > /var/log/serverledge-lb.log 2>&1 </dev/null &'
  "
  wait_remote "$NAME_LB" 'pgrep -x lb >/dev/null && sudo ss -lnt | grep -q ":1323 "' 'processo LB e porta 1323'
  pass "Cluster avviato, active=$ARCH, worker=$worker, LB interno=http://${lb_ip}:1323"
  info 'Funzioni NON ancora registrate: prima del workload usare il collector paper-RF'
  info 'Raw dei worker preservati; nessuna invocazione o esportazione eseguita'
}

case "$MODE" in
  --check) check_readonly ;;
  --start) start_cluster ;;
esac
