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

# Sorgenti delle funzioni: nel repository, di proprieta' di root.
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/opt/serverledge/examples/experiments}"

# Directory di lavoro di locust. Deve essere scrivibile dall'utente SSH:
# il locustfile scrive experiment_results.csv nella directory corrente, e in
# /opt/serverledge l'open() fallisce dentro l'event listener senza che
# l'eccezione risalga — l'esperimento gira ma il CSV per-richiesta, che e'
# l'unico a contenere l'architettura di esecuzione, non viene mai creato.
WORK_DIR="${WORK_DIR:-\$HOME/experiments}"

# Memoria e CPU per funzione. Filippo usava container.pool.memory 4096 per nodo;
# con 512 MB a funzione e nove funzioni il pool da 12000 MB regge senza che le
# richieste vengano rifiutate per memoria insufficiente.
# CPUDemand deve essere piccolo ma STRETTAMENTE POSITIVO.
#
# Serverledge tiene impegnate le risorse dei container warm anche a funzione
# ferma: con nove funzioni distribuite sull'anello, una riserva di 1 CPU
# ciascuna esaurirebbe i quattro core di un nodo non appena quattro funzioni vi
# finissero sopra, e le invocazioni successive verrebbero rifiutate con
# "Node has not enough resources".
#
# Zero pero' non e' ammesso: la CPU configurata fa parte dell'identita' del
# FunctionProfile — e' uno dei quattro campi della chiave di raggruppamento — e
# la pipeline di profiling la valida con finitePositive in quattro punti. Con
# CPUDemand a zero l'aggregazione fallisce con "configured CPUs are invalid: 0"
# e i campioni raccolti risultano inutilizzabili.
#
# 0.25 soddisfa entrambi i vincoli: nel caso peggiore nove funzioni sullo stesso
# nodo impegnano 2,25 core su 4.
FUNC_MEMORY="${FUNC_MEMORY:-512}"
FUNC_CPU="${FUNC_CPU:-0.25}"

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

# locust va installato in un virtualenv dedicato.
#
# Con "pip3 install --break-system-packages" su Ubuntu 24.04 l'installazione
# riesce ma lascia le dipendenze di gevent incomplete, e locust muore all'avvio
# con "ModuleNotFoundError: No module named 'zope.event'". Il virtualenv le
# risolve correttamente ed e' comunque la forma corretta.
#
# LOCUST_BIN e' il percorso assoluto: affidarsi al PATH non basta, perche' una
# eventuale installazione rotta in ~/.local/bin verrebbe trovata per prima.
LOCUST_BIN="\$HOME/locust-venv/bin/locust"

remote "
    cd /opt/serverledge && sudo git pull --quiet || true

    if [ ! -x ${LOCUST_BIN} ]; then
        echo 'Creo il virtualenv e installo locust...'
        python3 -m venv \$HOME/locust-venv
        \$HOME/locust-venv/bin/pip install --quiet --upgrade pip
        \$HOME/locust-venv/bin/pip install --quiet locust
    fi

    # Se locust non parte, l'esperimento non puo' procedere: meglio fermarsi
    # qui che scoprirlo dopo aver registrato le funzioni.
    if ! ${LOCUST_BIN} --version; then
        echo 'FATAL: locust non funziona.'
        exit 1
    fi

    mkdir -p ${WORK_DIR}
    cp ${EXPERIMENTS_DIR}/locustfile.py ${WORK_DIR}/
    ls -la ${WORK_DIR}/locustfile.py
"

# --- 2. Registrazione delle funzioni -----------------------------------------
#
# Le funzioni vengono ricreate a ogni esperimento: una delete seguita da create
# garantisce che i due run confrontati partano dalla stessa identica
# definizione, senza residui di configurazioni precedenti.

banner "REGISTRAZIONE FUNZIONI"

# Tabella esplicita invece di un glob su *.tar, per tre ragioni:
#
#   - il nome di registrazione deve coincidere con quello invocato dal
#     locustfile, e non sempre corrisponde al nome del file: primenum.tar
#     va registrato come "primenumber";
#   - due funzioni sono Python e richiedono runtime e handler diversi;
#   - la memoria varia per funzione ed e' stata derivata dai requisiti reali
#     di ciascuna (vedi passo_12_setup_sperimentale_gcp.md, sezione 4.3).
#
# Le due funzioni sintetiche usano i bundle V2. Le versioni senza suffisso
# allocano 256 MB e vi eseguono cinque passate di AES-GCM: impiegano oltre
# dieci secondi e superano il timeout che il locustfile impone proprio a
# queste due, fallendo sistematicamente. Le V2 completano in 1-2 secondi, in
# linea con i risultati della tesi precedente, e sono quindi le versioni su
# cui il locustfile era stato tarato. Il nome di registrazione resta pero'
# amd_faster e arm_faster, perche' e' quello che il locustfile invoca.
#
# Formato: nome|runtime|sorgente|memoria|handler
FUNCTIONS=(
    "base64stream|go125|functions/bundles/base64stream.tar|1024|"
    "compression|go125|functions/bundles/compression.tar|1024|"
    "dna-visualisation|go125|functions/bundles/dna-visualisation.tar|1024|"
    "dynamichtml|go125|functions/bundles/dynamichtml.tar|1024|"
    "goroutines|go125|functions/bundles/goroutines.tar|1024|"
    "graph-bfs|go125|functions/bundles/graph-bfs.tar|1024|"
    "graph-mst|go125|functions/bundles/graph-mst.tar|1024|"
    "graph-pagerank|go125|functions/bundles/graph-pagerank.tar|1024|"
    "hashing|go125|functions/bundles/hashing.tar|1024|"
    "jsonparse|go125|functions/bundles/jsonparse.tar|1024|"
    "matmul|go125|functions/bundles/matmul.tar|1024|"
    "mutexcontention|go125|functions/bundles/mutexcontention.tar|1024|"
    "pointerchase|go125|functions/bundles/pointerchase.tar|1024|"
    "randomaccess|go125|functions/bundles/randomaccess.tar|1024|"
    "sorting|go125|functions/bundles/sorting.tar|1024|"
    "syscallstorm|go125|functions/bundles/syscallstorm.tar|1024|"
    "tempfileio|go125|functions/bundles/tempfileio.tar|1024|"
    "thumbnailer|go125|functions/bundles/thumbnailer.tar|1024|"
    "twin-chacha20|go125|functions/bundles/twin-chacha20.tar|1024|"
    "twin-primenumber|go125|functions/bundles/twin-primenumber.tar|1024|"
    "twin-readmemory|go125|functions/bundles/twin-readmemory.tar|1024|"
    "primenumber|go125|primenum.tar|1024|"
    "chacha20|go125|chacha20.tar|1024|"
    "readdisk|go125|readdisk.tar|1024|"
    "readmemory|go125|readmemory.tar|1024|"
    "thread|go125|thread.tar|1024|"
    "amd_faster|go125|amd_fasterV2.tar|1024|"
    "arm_faster|go125|arm_fasterV2.tar|1024|"
    "linpack|python312ml|linpack.py|2048|linpack.handler"
    "filehandle|python314|filehandle.py|1024|filehandle.handler"
)

REGISTER_SCRIPT="cd ${EXPERIMENTS_DIR}
export SERVERLEDGE_HOST=${LB_IP}
export SERVERLEDGE_PORT=1323
CLI=/opt/serverledge/bin/serverledge-cli
"

for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name runtime src memory handler <<< "$entry"

    handler_flag=""
    if [[ -n "$handler" ]]; then
        handler_flag="--handler \"${handler}\""
    fi

    # --update sovrascrive una funzione gia' presente: senza, create fallisce
    # con 409 Conflict e i parametri restano quelli della registrazione
    # precedente.
    REGISTER_SCRIPT+="
echo -n \"  ${name}: \"
\$CLI create -f ${name} --runtime ${runtime} --src ${src} \
    --memory ${memory} --cpu ${FUNC_CPU} ${handler_flag} --update 2>&1 | tr -d '\n '
echo
"
done

remote "$REGISTER_SCRIPT"

# --- Verifica: una invocazione per funzione ---------------------------------
#
# Registrare non basta. Una funzione che risponde in fase di registrazione puo'
# comunque fallire all'esecuzione — per un comando esterno mancante
# nell'immagine o per memoria insufficiente — e l'esperimento la conterebbe
# come fallimento per tutta la sua durata.

banner "VERIFICA — una invocazione per funzione"

VERIFY_SCRIPT="LB=${LB_IP}
"
for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$entry"
    VERIFY_SCRIPT+="
r=\$(curl -s -m 300 -X POST \"http://\${LB}:1323/invoke/${name}\" \
    -H 'Content-Type: application/json' -d '{\"params\":{}}')
case \"\$r\" in
    *'\"Success\":true'*) echo \"  ${name}: OK\" ;;
    *)                     echo \"  ${name}: FALLITA — \$(echo \$r | head -c 80)\" ;;
esac
"
done

remote "$VERIFY_SCRIPT"

banner "CONTROLLO"

cat <<EOF
Tutte le funzioni sopra devono riportare OK.

Una funzione FALLITA produrrebbe errori per l'intera durata dell'esperimento e
renderebbe i risultati non confrontabili: interrompi e sistemala prima di
proseguire. La diagnosi parte sempre dal log del container sul nodo che ha
eseguito, non dal messaggio restituito al chiamante.

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
    cd ${WORK_DIR}

    export LB_POLICY=${POLICY}

    rm -f experiment_results.csv

    \$HOME/locust-venv/bin/locust -f locustfile.py \
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