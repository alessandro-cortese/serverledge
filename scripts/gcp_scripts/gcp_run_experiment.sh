#!/usr/bin/env bash
#
# Registra le funzioni di benchmark e lancia l'esperimento con locust.
#
#     ./gcp_run_experiment.sh RoundRobin
#     ./gcp_run_experiment.sh LinUCB 20m
#
# Da eseguire dopo gcp_start_cluster.sh con la stessa policy: il nome della
# policy finisce nel CSV prodotto da locust, quindi deve corrispondere a quella
# con cui il load balancer sta effettivamente girando.
#
# Locust gira sulla VM sl-workload, nella stessa zona del cluster. Lanciarlo dal
# portatile aggiungerebbe la latenza Italia-Olanda a ogni misura di tempo di
# risposta.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

POLICY="${1:-RoundRobin}"
DURATION="${2:-20m}"
USERS="${USERS:-9}"
SPAWN_RATE="${SPAWN_RATE:-9}"

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/opt/serverledge/examples/experiments}"

# Memoria e CPU per funzione. Filippo usava container.pool.memory 4096 per nodo;
# con 512 MB a funzione e nove funzioni il pool da 12000 MB regge senza che le
# richieste vengano rifiutate per memoria insufficiente.
FUNC_MEMORY="${FUNC_MEMORY:-512}"
FUNC_CPU="${FUNC_CPU:-1.0}"

banner "ESPERIMENTO — policy $POLICY, durata $DURATION"

internal_ip() {
    gc compute instances describe "$1" --zone="$ZONE" \
        --format="value(networkInterfaces[0].networkIP)"
}

LB_IP="$(internal_ip "$NAME_LB")"

echo "load balancer: $LB_IP"
echo "utenti:        $USERS (spawn rate $SPAWN_RATE)"

remote() {
    gc compute ssh "$NAME_WORKLOAD" --zone="$ZONE" --quiet --command="$*"
}

# --- 1. Preparazione del generatore di workload ------------------------------

banner "PREPARAZIONE"

remote "
    cd /opt/serverledge && sudo git pull --quiet || true

    # pip installa gli eseguibili in ~/.local/bin, che non e' nel PATH delle
    # sessioni SSH non interattive.
    export PATH=\$HOME/.local/bin:\$PATH

    if ! command -v locust >/dev/null 2>&1; then
        echo 'Installo locust...'
        pip3 install --quiet --break-system-packages locust 2>&1 | grep -v WARNING || true
    fi

    locust --version
    ls ${EXPERIMENTS_DIR}/locustfile.py
"

# --- 2. Registrazione delle funzioni -----------------------------------------
#
# Le funzioni vengono ricreate a ogni esperimento: una delete seguita da create
# garantisce che i due run confrontati partano dalla stessa identica
# definizione, senza residui di configurazioni precedenti.

banner "REGISTRAZIONE FUNZIONI"

remote "
    cd ${EXPERIMENTS_DIR}
    export SERVERLEDGE_HOST=${LB_IP}
    export SERVERLEDGE_PORT=1323
    CLI=/opt/serverledge/bin/serverledge-cli

    # I bundle di Filippo sono TAR contenenti uno ZIP con due binari Go,
    # handler_amd64 e handler_arm64. L'entrypoint del runtime go125 scompatta
    # lo zip e sceglie il binario in base a uname -m: e' cosi' che la stessa
    # funzione gira su entrambe le architetture.
    for tar in *.tar; do
        [ -e \"\$tar\" ] || continue
        name=\$(basename \"\$tar\" .tar)
        \$CLI delete -f \"\$name\" >/dev/null 2>&1 || true
        if out=\$(\$CLI create -f \"\$name\" --runtime go125 --src \"\$tar\" \
             --memory ${FUNC_MEMORY} --cpu ${FUNC_CPU} 2>&1); then
            echo \"  \$name registrata\"
        else
            echo \"  \$name NON registrata: \$out\"
        fi
    done

    echo
    echo 'Funzioni registrate:'
    \$CLI list
"

banner "CONTROLLO"

cat <<EOF
Verifica sopra che tutte le funzioni attese siano presenti prima di procedere.
Se qualcuna manca, il comando di registrazione va adattato al formato dei
bundle di Filippo: interrompi qui e sistemalo, perché un esperimento con
funzioni mancanti non è confrontabile con i suoi risultati.

Premi INVIO per lanciare locust per $DURATION, oppure Ctrl-C per fermarti.
EOF

read -r _

# --- 3. Esecuzione -----------------------------------------------------------

banner "LOCUST — $DURATION"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="/opt/serverledge/results/${POLICY}_${STAMP}"

remote "
    sudo mkdir -p ${RESULT_DIR}
    sudo chmod 777 ${RESULT_DIR}
    cd ${EXPERIMENTS_DIR}

    export PATH=\$HOME/.local/bin:\$PATH
    export LB_POLICY=${POLICY}

    rm -f experiment_results.csv

    locust -f locustfile.py \
        --headless \
        --users ${USERS} \
        --spawn-rate ${SPAWN_RATE} \
        --run-time ${DURATION} \
        --host http://${LB_IP}:1323 \
        --csv ${RESULT_DIR}/locust \
        2>&1 | tail -40

    cp experiment_results.csv ${RESULT_DIR}/ 2>/dev/null || true
    ls -la ${RESULT_DIR}
"

banner "FATTO"

echo "Risultati sul generatore di workload in: ${RESULT_DIR}"
echo
echo "Scaricali in locale con:"
echo "    ./gcp_collect.sh ${POLICY}"