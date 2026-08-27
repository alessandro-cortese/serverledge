#!/usr/bin/env bash
#
# Configura e avvia l'intero cluster Serverledge sulle VM già create.
#
#     ./gcp_start_cluster.sh RoundRobin
#     ./gcp_start_cluster.sh LinUCB
#     ./gcp_start_cluster.sh UCB1
#
# Va eseguito dopo gcp_up.sh. Legge da solo gli indirizzi interni delle VM,
# quindi non serve aggiornare nulla a mano quando il cluster viene ricreato e
# gli IP cambiano.
#
# Ferma sempre tutto prima di riavviare, così passare da una policy all'altra
# non lascia processi né stato del bandit della sessione precedente: due
# esperimenti consecutivi devono partire dalle stesse condizioni.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

POLICY="${1:-RoundRobin}"

# PROFILING=1 attiva la scrittura dei campioni di profiling sui worker.
# Serve solo per le sessioni di raccolta dati destinate al clustering e al
# transfer learning: negli esperimenti di confronto fra policy resta spento,
# perche' la scrittura del JSONL a ogni invocazione altera i tempi misurati.
PROFILING="${PROFILING:-0}"

case "$POLICY" in
    RoundRobin|LinUCB|UCB1) ;;
    *)
        echo "Policy non riconosciuta: $POLICY"
        echo "Valori ammessi: RoundRobin, LinUCB, UCB1"
        exit 1
        ;;
esac

# Con RoundRobin il load balancer resta architecture-unaware: è la baseline ad
# hash ring singolo. Con le due policy MAB serve lb.arch_awareness, che è la
# chiave che governa davvero la scelta del balancer.
if [[ "$POLICY" == "RoundRobin" ]]; then
    ARCH_AWARE="false"
    LB_MODE="RoundRobin"
else
    ARCH_AWARE="true"
    LB_MODE="MAB"
fi

banner "AVVIO CLUSTER — policy $POLICY"

# --- Indirizzi interni -------------------------------------------------------

internal_ip() {
    gc compute instances describe "$1" --zone="$ZONE" \
        --format="value(networkInterfaces[0].networkIP)"
}

REGISTRY_IP="$(internal_ip "$NAME_REGISTRY")"
LB_IP="$(internal_ip "$NAME_LB")"

echo "registry: $REGISTRY_IP"
echo "lb:       $LB_IP"

# Le VM spot possono essere preemptate in qualsiasi momento. Configurare solo
# quelle che esistono davvero evita che l'intero avvio si interrompa a meta'
# per una macchina sparita, e rende evidente quante ne mancano.
EXISTING="$(
    gc compute instances list --zones="$ZONE" --format="value(name)" || true
)"

WORKERS=()
MISSING=()

while read -r name; do
    [[ -z "$name" ]] && continue
    if grep -qx "$name" <<< "$EXISTING"; then
        WORKERS+=("$name")
    else
        MISSING+=("$name")
    fi
done < <(x86_node_names; arm_node_names)

echo "worker:   ${WORKERS[*]}"

if (( ${#MISSING[@]} > 0 )); then
    echo
    echo "ATTENZIONE — worker mancanti (probabile preemption): ${MISSING[*]}"
    echo "Ricreali prima di misurare, altrimenti il cluster non ha la"
    echo "configurazione dichiarata nel manifest:"
    echo "    N_X86=${N_X86} N_ARM=${N_ARM} SPOT=${SPOT} ./gcp_up.sh"
    echo
fi

if (( ${#WORKERS[@]} == 0 )); then
    echo "Nessun worker disponibile: impossibile avviare il cluster."
    exit 1
fi

remote() {
    local host="$1"
    shift
    gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

# --- 1. Stop di tutto --------------------------------------------------------

banner "ARRESTO PROCESSI PRECEDENTI"

# pkill -x confronta il nome del processo, non la riga di comando: con -f il
# comando ucciderebbe la shell remota che lo sta eseguendo.
for host in "$NAME_LB" "${WORKERS[@]}"; do
    remote "$host" "sudo pkill -x lb || true; sudo pkill -x serverledge || true" \
        >/dev/null 2>&1 || true
    echo "  $host fermato"
done

# --- 2. Registry ed etcd -----------------------------------------------------

banner "ETCD"

# etcd viene avviato qui invece che con scripts/start-etcd.sh perche' quello e'
# lo script upstream e va lasciato intatto. La differenza sta in due parametri:
#
#   --max-request-bytes    il default di etcd rifiuta messaggi oltre i 2 MB,
#                          mentre i bundle delle funzioni Go multi-architettura
#                          arrivano a 6-7 MB (uno zip con handler_amd64 e
#                          handler_arm64). Senza questo, la registrazione
#                          fallisce con ResourceExhausted e il load balancer
#                          risponde 503 a ogni invocazione.
#
#   --quota-backend-bytes  evita che il database si riempia registrando piu'
#                          volte funzioni di qualche megabyte.
#
# Il lato client era gia' configurato per 10 MB in utils/etcd.go; mancava
# soltanto la controparte sul server.
remote "$NAME_REGISTRY" "
    sudo docker rm -f Etcd-server >/dev/null 2>&1 || true

    sudo docker run -d \
      -p 2379:2379 -p 2380:2380 \
      --name Etcd-server \
      gcr.io/etcd-development/etcd:v3.5.13 \
      /usr/local/bin/etcd \
      --name s1 \
      --data-dir /etcd-data \
      --listen-client-urls http://0.0.0.0:2379 \
      --advertise-client-urls http://0.0.0.0:2379 \
      --listen-peer-urls http://0.0.0.0:2380 \
      --initial-advertise-peer-urls http://0.0.0.0:2380 \
      --initial-cluster s1=http://0.0.0.0:2380 \
      --initial-cluster-token tkn \
      --initial-cluster-state new \
      --max-request-bytes 33554432 \
      --quota-backend-bytes 8589934592 >/dev/null

    sleep 3
    sudo docker ps --filter name=Etcd-server --format '  {{.Names}} {{.Status}}'
"

# etcd riparte vuoto a ogni esperimento: le registrazioni dei nodi sono
# effimere e i bandit non devono ereditare stato dalla sessione precedente.

# --- 3. Worker ---------------------------------------------------------------

if [[ "$PROFILING" == "1" ]]; then
    PROFILING_BLOCK="
profiling:
  enabled: true
  export:
    enabled: true
    path: \"/var/lib/serverledge/profiling-samples.jsonl\""
    echo
    echo "Profiling ATTIVO: i campioni verranno scritti in"
    echo "  /var/lib/serverledge/profiling-samples.jsonl su ciascun worker"
else
    PROFILING_BLOCK=""
fi

banner "WORKER"

for host in "${WORKERS[@]}"; do

    remote "$host" "
        cd /opt/serverledge && sudo git pull --quiet || true

        sudo tee worker.yaml > /dev/null <<EOF
registry:
  area: \"cloud-region\"
  udp.port: 9877

etcd:
  address: \"${REGISTRY_IP}:2379\"

container:
  pool:
    memory: 12000

janitor:
  interval: 60
${PROFILING_BLOCK}
EOF

        sudo mkdir -p /var/lib/serverledge
        sudo chmod 777 /var/lib/serverledge

        sudo sh -c 'cd /opt/serverledge && nohup ./bin/serverledge worker.yaml > /var/log/serverledge-node.log 2>&1 &'
    " >/dev/null

    echo "  $host avviato"
done

echo
echo "Attendo la registrazione dei worker..."
sleep 10

REGISTERED="$(
    remote "$NAME_REGISTRY" \
        "sudo docker exec Etcd-server etcdctl get registry/ --prefix --keys-only | grep -c cloud-region || true"
)"

echo "Nodi registrati in etcd: $REGISTERED (attesi: ${#WORKERS[@]})"

if [[ "$REGISTERED" != "${#WORKERS[@]}" ]]; then
    echo "I numeri non tornano: controlla /var/log/serverledge-node.log sui worker."
fi

# --- 4. Load balancer --------------------------------------------------------

banner "LOAD BALANCER"

remote "$NAME_LB" "
    cd /opt/serverledge && sudo git pull --quiet || true

    sudo tee lb.yaml > /dev/null <<EOF
registry:
  area: \"cloud-region\"
  udp.port: 9877

lb:
  replicas: 128
  mode: \"${LB_MODE}\"
  arch_awareness: ${ARCH_AWARE}
  refresh_interval: 30

mab.policy: \"${POLICY}\"

etcd:
  address: \"${REGISTRY_IP}:2379\"
EOF

    sudo sh -c 'cd /opt/serverledge && nohup ./bin/lb lb.yaml > /var/log/serverledge-lb.log 2>&1 &'
    sleep 8
    sudo tail -15 /var/log/serverledge-lb.log
"

# --- 5. Riepilogo ------------------------------------------------------------

banner "CLUSTER PRONTO"

cat <<EOF
policy:       $POLICY
arch_aware:   $ARCH_AWARE
load balancer: http://${LB_IP}:1323   (interno)

Registra le funzioni e lancia l'esperimento dal generatore di workload:

    ./gcp_run_experiment.sh $POLICY

Al termine raccogli i risultati con:

    ./gcp_collect.sh $POLICY
EOF
