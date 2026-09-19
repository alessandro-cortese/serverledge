#!/usr/bin/env bash
#
# Avvio del cluster per la campagna GCP finale di transfer learning.
#
# NON crea VM: usa esclusivamente quelle create da gcp_up.sh.
# Di conseguenza gcp_down.sh continua a poter distruggere l'intero cluster.
#
# Uso:
#
#   N_X86=1 N_ARM=1 \
#   ./gcp_start_transfer_cluster.sh no-transfer
#
#   N_X86=1 N_ARM=1 \
#   ./gcp_start_transfer_cluster.sh coupled
#
#   N_X86=1 N_ARM=1 \
#   ./gcp_start_transfer_cluster.sh decoupled
#
# Variabili:
#
#   PROFILING=1|0
#   MAB_UCB1_C=0.8
#   CONTAINER_POOL_MEMORY=12000
#   JANITOR_INTERVAL=60
#
# Condizioni finali:
#
#   no-transfer:
#       policy = UCB1
#       prior  = disabled
#
#   coupled:
#       policy = UCB1
#       prior  = enabled
#       equivalent_observation_weight = 0.25
#       # stesso peso per reward ed exploration
#
#   decoupled:
#       policy = UCB1Decoupled
#       prior  = enabled
#       reward_observation_weight      = 0.25
#       exploration_observation_weight = 1.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

MODE="${1:-}"

PROFILING="${PROFILING:-1}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"

JANITOR_INTERVAL="${JANITOR_INTERVAL:-60}"
CONTAINER_POOL_MEMORY="${CONTAINER_POOL_MEMORY:-12000}"

case "$MODE" in
    no-transfer)
        POLICY="UCB1"
        TRANSFER_ENABLED="false"
        ;;
    coupled)
        POLICY="UCB1"
        TRANSFER_ENABLED="true"
        ;;
    decoupled)
        POLICY="UCB1Decoupled"
        TRANSFER_ENABLED="true"
        ;;
    *)
        echo "Uso:"
        echo "  $0 no-transfer"
        echo "  $0 coupled"
        echo "  $0 decoupled"
        exit 1
        ;;
esac

if [[ "$PROFILING" != "0" && "$PROFILING" != "1" ]]; then
    echo "PROFILING deve essere 0 oppure 1"
    exit 1
fi

if ! [[ "$JANITOR_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "JANITOR_INTERVAL deve essere un intero > 0"
    exit 1
fi

if ! [[ "$CONTAINER_POOL_MEMORY" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTAINER_POOL_MEMORY deve essere un intero > 0"
    exit 1
fi

banner "TRANSFER FINAL — RESET CLUSTER"

echo "mode:            $MODE"
echo "policy:          $POLICY"
echo "mab.ucb1.c:      $MAB_UCB1_C"
echo "profiling:       $PROFILING"
echo "transfer API:    $TRANSFER_ENABLED"
echo "x86 workers:     $N_X86"
echo "arm64 workers:   $N_ARM"

internal_ip() {
    gc compute instances describe "$1" \
        --zone="$ZONE" \
        --format="value(networkInterfaces[0].networkIP)"
}

remote() {
    local host="$1"
    shift

    gc compute ssh "$host" \
        --zone="$ZONE" \
        --quiet \
        --command="$*"
}

REGISTRY_IP="$(internal_ip "$NAME_REGISTRY")"
LB_IP="$(internal_ip "$NAME_LB")"

EXISTING="$(
    gc compute instances list \
        --zones="$ZONE" \
        --format="value(name)"
)"

WORKERS=()

while read -r name; do
    [[ -z "$name" ]] && continue

    if grep -qx "$name" <<< "$EXISTING"; then
        WORKERS+=("$name")
    else
        echo "Worker richiesto ma assente: $name"
        exit 1
    fi
done < <(x86_node_names; arm_node_names)

if (( ${#WORKERS[@]} == 0 )); then
    echo "Nessun worker disponibile."
    exit 1
fi

echo "registry IP:     $REGISTRY_IP"
echo "LB IP:           $LB_IP"
echo "workers:         ${WORKERS[*]}"

# ---------------------------------------------------------------------------
# Verifica commit congelato
# ---------------------------------------------------------------------------

EXPECTED_COMMIT="08600a177ce1d88c048251a4b6d52d31fe65a536"

banner "VERIFY FROZEN COMMIT"

for host in \
    "$NAME_REGISTRY" \
    "$NAME_LB" \
    "$NAME_WORKLOAD" \
    "${WORKERS[@]}"
do
    actual="$(
        remote "$host" \
            "sudo git -C /opt/serverledge rev-parse HEAD" \
            | tr -d '[:space:]'
    )"

    if [[ "$actual" != "$EXPECTED_COMMIT" ]]; then
        echo "FATAL: $host commit=$actual"
        echo "atteso: $EXPECTED_COMMIT"
        exit 1
    fi

    echo "  $host: $actual"
done

# ---------------------------------------------------------------------------
# Stop del runtime precedente
# ---------------------------------------------------------------------------

banner "STOP PREVIOUS RUNTIME"

for host in "$NAME_LB" "${WORKERS[@]}"; do
    remote "$host" \
        "sudo pkill -x lb >/dev/null 2>&1 || true;
         sudo pkill -x serverledge >/dev/null 2>&1 || true"

    echo "  $host fermato"
done

# ---------------------------------------------------------------------------
# etcd completamente nuovo
# ---------------------------------------------------------------------------

banner "ETCD RESET"

remote "$NAME_REGISTRY" "
    set -e

    sudo docker rm -f Etcd-server >/dev/null 2>&1 || true

    sudo docker run -d \
      -p 2379:2379 \
      -p 2380:2380 \
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
      --quota-backend-bytes 8589934592 \
      >/dev/null

    sleep 3

    sudo docker exec Etcd-server \
      etcdctl endpoint health
"

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

banner "WORKERS"

for host in "${WORKERS[@]}"; do

    case "$host" in
        sl-x86-*)
            MACHINE_TAG="amd64"
            ;;
        sl-arm-*)
            MACHINE_TAG="arm64"
            ;;
        *)
            echo "Impossibile determinare machine tag per $host"
            exit 1
            ;;
    esac

    if [[ "$PROFILING" == "1" ]]; then
        PROFILING_BLOCK="
profiling:
  enabled: true
  export:
    enabled: true
    path: \"/var/lib/serverledge/profiling-samples.jsonl\""
    else
        PROFILING_BLOCK="
profiling:
  enabled: false
  export:
    enabled: false"
    fi

    remote "$host" "
        set -e

        sudo rm -f /var/lib/serverledge/profiling-samples.jsonl
        sudo mkdir -p /var/lib/serverledge
        sudo chmod 777 /var/lib/serverledge

        sudo tee /opt/serverledge/worker-transfer-final.yaml \
          >/dev/null <<EOF
registry:
  area: \"cloud-region\"
  node:
    id: \"${host}\"
  udp:
    port: 9877

etcd:
  address: \"${REGISTRY_IP}:2379\"

node:
  machine_tag: \"${MACHINE_TAG}\"

container:
  pool:
    memory: ${CONTAINER_POOL_MEMORY}

janitor:
  interval: ${JANITOR_INTERVAL}

${PROFILING_BLOCK}
EOF

        sudo sh -c '
          cd /opt/serverledge &&
          nohup ./bin/serverledge \
            worker-transfer-final.yaml \
            > /var/log/serverledge-node.log 2>&1 &
        '
    "

    echo "  $host avviato machine_tag=$MACHINE_TAG"
done

echo
echo "Attendo registrazione worker..."
sleep 10

REGISTERED="$(
    remote "$NAME_REGISTRY" \
        "sudo docker exec Etcd-server \
         etcdctl get registry/ --prefix --keys-only |
         grep -c cloud-region || true"
)"

echo "worker registrati: $REGISTERED"
echo "worker attesi:     ${#WORKERS[@]}"

if [[ "$REGISTERED" != "${#WORKERS[@]}" ]]; then
    echo "FATAL: numero worker registrati inatteso."
    exit 1
fi

# ---------------------------------------------------------------------------
# Load balancer
# ---------------------------------------------------------------------------

banner "LOAD BALANCER"

remote "$NAME_LB" "
    set -e

    sudo tee /opt/serverledge/lb-transfer-final.yaml \
      >/dev/null <<EOF
registry:
  area: \"cloud-region\"
  udp:
    port: 9877

lb:
  replicas: 128
  mode: \"MAB\"
  arch_awareness: true
  refresh_interval: 5

mab:
  policy: \"${POLICY}\"
  ucb1:
    c: ${MAB_UCB1_C}
  reward:
    mode: latency
  transfer:
    control:
      enabled: ${TRANSFER_ENABLED}

etcd:
  address: \"${REGISTRY_IP}:2379\"
EOF

    sudo sh -c '
      cd /opt/serverledge &&
      nohup ./bin/lb \
        lb-transfer-final.yaml \
        > /var/log/serverledge-lb.log 2>&1 &
    '

    sleep 8

    sudo tail -30 /var/log/serverledge-lb.log
"

# ---------------------------------------------------------------------------
# Gate finale
# ---------------------------------------------------------------------------

banner "FINAL RUNTIME GATE"

echo "Verifico il load balancer..."

remote "$NAME_LB" "
    set -euo pipefail

    # Processo LB realmente attivo.
    pgrep -x lb >/dev/null

    # Il runtime deve essere quello architecture-aware in modalità MAB.
    grep -q 'Running ArchitectureAwareLB' \
        /var/log/serverledge-lb.log

    grep -q 'LB mode set to MAB' \
        /var/log/serverledge-lb.log

    # Entrambi i bracci architetturali devono essere stati scoperti.
    grep -q 'event=new_machine_tag_discovered tag=amd64 arch=amd64' \
        /var/log/serverledge-lb.log

    grep -q 'event=new_machine_tag_discovered tag=arm64 arch=arm64' \
        /var/log/serverledge-lb.log

    # Controllo della policy dalla configurazione effettivamente fornita al LB.
    grep -q 'policy: \"${POLICY}\"' \
        /opt/serverledge/lb-transfer-final.yaml

    # c deve essere quello congelato.
    grep -q 'c: ${MAB_UCB1_C}' \
        /opt/serverledge/lb-transfer-final.yaml

    # Verifica della modalità transfer attesa.
    grep -q 'enabled: ${TRANSFER_ENABLED}' \
        /opt/serverledge/lb-transfer-final.yaml

    echo '  LB: PASS'
"

echo
echo "Verifico i worker..."

for host in "${WORKERS[@]}"; do

    case "$host" in
        sl-x86-*)
            EXPECTED_TAG="amd64"
            ;;
        sl-arm-*)
            EXPECTED_TAG="arm64"
            ;;
        *)
            echo "FATAL: worker inatteso: $host"
            exit 1
            ;;
    esac

    remote "$host" "
        set -euo pipefail

        pgrep -x serverledge >/dev/null

        grep -q 'machine_tag: \"${EXPECTED_TAG}\"' \
            /opt/serverledge/worker-transfer-final.yaml

        grep -q 'Local node id:.*${EXPECTED_TAG}' \
            /var/log/serverledge-node.log

        grep -q 'etcd: connected and ready' \
            /var/log/serverledge-node.log

        echo '  ${host}: PASS (${EXPECTED_TAG})'
    "
done

echo
echo "Verifico etcd..."

REGISTERED_FINAL="$(
    remote "$NAME_REGISTRY" \
        "sudo docker exec Etcd-server \
         etcdctl get registry/cloud-region/ --prefix --keys-only |
         grep -E '^registry/cloud-region/(amd64|arm64)/' |
         wc -l"
)"

if [[ "$REGISTERED_FINAL" != "${#WORKERS[@]}" ]]; then
    echo "FATAL: etcd contiene $REGISTERED_FINAL worker reali; attesi ${#WORKERS[@]}."
    exit 1
fi

echo "  etcd: PASS (${REGISTERED_FINAL}/${#WORKERS[@]} worker reali)"

echo
echo "============================================================"
echo "TRANSFER FINAL CLUSTER READY"
echo "============================================================"
echo "mode:         $MODE"
echo "policy:       $POLICY"
echo "c:            $MAB_UCB1_C"
echo "profiling:    $PROFILING"
echo "transfer API: $TRANSFER_ENABLED"
echo "LB:           http://${LB_IP}:1323"
echo
echo "Nessuna VM è stata creata da questo script."
echo "Cleanup completo:"
echo "  ./gcp_down.sh"