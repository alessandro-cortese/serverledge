#!/usr/bin/env bash
#
# Raccolta controllata dei profili Serverledge — variante veloce.
#
# Uso:
#   ./gcp_collect_profiles_fast.sh amd64
#   ./gcp_collect_profiles_fast.sh arm64
#
# Differenze rispetto a gcp_collect_profiles.sh (che resta l'implementazione di
# riferimento e NON va modificata):
#
#   - il cluster viene avviato UNA sola volta per l'intera campagna, non una
#     volta per funzione;
#   - tutte le funzioni vengono registrate UNA sola volta: registrare una
#     funzione scrive solo in etcd, non crea container;
#   - fra una funzione e l'altra non si riavvia nulla. Il container precedente
#     viene recuperato dal janitor di Serverledge (janitor.interval /
#     container.expiration), che e' il percorso che mantiene coerente la
#     contabilita' delle risorse del nodo;
#   - prima di passare alla funzione successiva si verifica DAVVERO che il
#     worker non abbia piu' container runtime, invece di fare uno sleep cieco:
#     free_memory_mb e' node-scoped, quindi un container residuo la falserebbe;
#   - ogni funzione si ferma appena raggiunge TARGET_WARM_VALID campioni warm
#     validi, con un tetto di MAX_REQUESTS_PER_FUNCTION invocazioni.
#
# Cosa NON cambia, perche' e' cio' che rende confrontabili x86 e ARM:
#
#   - stesso catalogo, stessa memoria, stesso runtime, stesso handler;
#   - CPU della funzione = 1;
#   - profiling attivo su entrambe le architetture;
#   - una sola funzione in esecuzione alla volta su un solo worker;
#   - cold start conservati.
#
# Precondizioni:
#   - le VM della sola architettura scelta sono gia' state create con gcp_up.sh;
#   - il repository remoto contiene locustfile_collection.py con il supporto a
#     TARGET_WARM_VALID.
#
# Variabili opzionali:
#   TARGET_WARM_VALID=12              # warm validi dopo i quali si smette
#   MAX_REQUESTS_PER_FUNCTION=16      # tetto invocazioni per funzione
#   JANITOR_INTERVAL=5                # secondi fra due passaggi del janitor
#   CONTAINER_EXPIRATION=10           # secondi di idle prima della scadenza
#   CONTAINER_POOL_MEMORY=12000       # MB del pool container sul worker
#   DRAIN_TIMEOUT=90                  # attesa massima svuotamento worker
#   DRAIN_POLL=2                      # intervallo di polling
#   MAX_RUNTIME=90m                   # tetto Locust per singola funzione
#   STRICT=1                          # 1 = interrompe alla prima incompleta
#   DEST_ROOT=./risultati
#   ONLY_FUNCTIONS=nome1,nome2,...
#   GCLOUD_RETRIES=8
#   GCLOUD_RETRY_DELAY=5
#   CLUSTER_START_RETRIES=5
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

ARCH="${1:-}"
POLICY="RoundRobin"
FUNC_CPU="1"

TARGET_WARM_VALID="${TARGET_WARM_VALID:-12}"
MAX_REQUESTS_PER_FUNCTION="${MAX_REQUESTS_PER_FUNCTION:-16}"
JANITOR_INTERVAL="${JANITOR_INTERVAL:-5}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-10}"
CONTAINER_POOL_MEMORY="${CONTAINER_POOL_MEMORY:-12000}"
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-90}"
DRAIN_POLL="${DRAIN_POLL:-2}"
MAX_RUNTIME="${MAX_RUNTIME:-90m}"
STRICT="${STRICT:-1}"

EXPERIMENTS_DIR="/opt/serverledge/examples/experiments"
WORK_DIR="\$HOME/profile-collection"
LOCUST_BIN="\$HOME/locust-venv/bin/locust"
DEST_ROOT="${DEST_ROOT:-${SCRIPT_DIR}/risultati}"
ONLY_FUNCTIONS="${ONLY_FUNCTIONS:-}"
GCLOUD_RETRIES="${GCLOUD_RETRIES:-8}"
GCLOUD_RETRY_DELAY="${GCLOUD_RETRY_DELAY:-5}"
CLUSTER_START_RETRIES="${CLUSTER_START_RETRIES:-5}"

case "$ARCH" in
    amd64|arm64) ;;
    *)
        echo "Uso: $0 {amd64|arm64}"
        exit 1
        ;;
esac

for var in TARGET_WARM_VALID MAX_REQUESTS_PER_FUNCTION JANITOR_INTERVAL \
           CONTAINER_EXPIRATION CONTAINER_POOL_MEMORY DRAIN_TIMEOUT \
           DRAIN_POLL GCLOUD_RETRIES CLUSTER_START_RETRIES; do
    if ! [[ "${!var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$var deve essere un intero > 0 (valore: ${!var})"
        exit 1
    fi
done

if ! [[ "$GCLOUD_RETRY_DELAY" =~ ^[0-9]+$ ]]; then
    echo "GCLOUD_RETRY_DELAY deve essere un intero >= 0"
    exit 1
fi

if (( TARGET_WARM_VALID + 1 > MAX_REQUESTS_PER_FUNCTION )); then
    echo "MAX_REQUESTS_PER_FUNCTION deve lasciare spazio ad almeno un cold start:"
    echo "target warm=${TARGET_WARM_VALID}, massimo=${MAX_REQUESTS_PER_FUNCTION}."
    exit 1
fi

if [[ "$STRICT" != "0" && "$STRICT" != "1" ]]; then
    echo "STRICT deve valere 0 oppure 1 (valore: ${STRICT})."
    exit 1
fi

# -----------------------------------------------------------------------------
# Catalogo condiviso e controllo di coerenza con lo script di riferimento.
#
# Il rischio peggiore di avere due script e' che i corpora divergano senza che
# nessuno se ne accorga: a quel punto i dataset amd64 e arm64 non sarebbero piu'
# confrontabili. Qui il confronto e' esplicito e blocca la campagna.
# -----------------------------------------------------------------------------

source "${SCRIPT_DIR}/functions_catalog.sh"

REFERENCE_SCRIPT="${SCRIPT_DIR}/gcp_collect_profiles.sh"

if [[ -f "$REFERENCE_SCRIPT" ]]; then
    # Confronta l'intera specifica del catalogo, non soltanto i nomi: runtime,
    # sorgente, memoria e handler devono essere identici allo script validato.
    catalog_entries="$(printf '%s\n' "${FUNCTIONS[@]}" | sort)"
    reference_entries="$(
        sed -n '/^FUNCTIONS=(/,/^)$/p' "$REFERENCE_SCRIPT" \
            | sed -n 's/^[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' \
            | sort
    )"

    if [[ "$catalog_entries" != "$reference_entries" ]]; then
        echo "ERRORE: functions_catalog.sh e gcp_collect_profiles.sh divergono."
        echo "Devono coincidere nome, runtime, sorgente, memoria e handler."
        diff <(echo "$reference_entries") <(echo "$catalog_entries") \
            --label riferimento --label catalogo || true
        exit 1
    fi
fi

if (( ${#FUNCTIONS[@]} != 53 )) && [[ -z "$ONLY_FUNCTIONS" ]]; then
    echo "ERRORE: il corpus congelato deve contenere 53 funzioni, trovate ${#FUNCTIONS[@]}."
    exit 1
fi

CATALOG_SHA256="$(printf '%s\n' "${FUNCTIONS[@]}" | sha256sum | awk '{print $1}')"

if [[ -n "$ONLY_FUNCTIONS" ]]; then
    IFS=',' read -r -a requested_names <<< "$ONLY_FUNCTIONS"
    filtered=()
    missing_requested=()

    for requested in "${requested_names[@]}"; do
        requested="${requested//[[:space:]]/}"
        [[ -z "$requested" ]] && continue

        found=0
        for entry in "${FUNCTIONS[@]}"; do
            IFS='|' read -r name _ _ _ _ <<< "$entry"
            if [[ "$name" == "$requested" ]]; then
                filtered+=("$entry")
                found=1
                break
            fi
        done

        if [[ "$found" -eq 0 ]]; then
            missing_requested+=("$requested")
        fi
    done

    if (( ${#missing_requested[@]} > 0 )); then
        echo "Funzioni richieste non presenti nel catalogo: ${missing_requested[*]}"
        exit 1
    fi

    FUNCTIONS=("${filtered[@]}")
fi

# -----------------------------------------------------------------------------
# Helper infrastrutturali (identici allo script di riferimento).
# -----------------------------------------------------------------------------

retry_cmd() {
    local max_attempts="$1"
    local delay="$2"
    shift 2

    local attempt=1
    local status=0

    while true; do
        if "$@"; then
            return 0
        else
            status=$?
        fi

        if (( attempt >= max_attempts )); then
            echo "ERRORE: comando fallito dopo ${attempt} tentativi: $*" >&2
            return "$status"
        fi

        echo "ATTENZIONE: comando fallito (tentativo ${attempt}/${max_attempts})." >&2
        echo "Riprovo tra ${delay}s: $*" >&2
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

# Nessun retry: ripetere una sequenza Locust gia' partita duplicherebbe campioni.
remote_once() {
    local host="$1"
    shift
    gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

remote_retry() {
    local host="$1"
    shift
    retry_cmd "$GCLOUD_RETRIES" "$GCLOUD_RETRY_DELAY" \
        gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

running_workers() {
    local prefix="$1"
    retry_cmd "$GCLOUD_RETRIES" "$GCLOUD_RETRY_DELAY" \
        gc compute instances list \
            --zones="$ZONE" \
            --filter="name~'^${prefix}-[0-9]+$' AND status=RUNNING" \
            --format="value(name)" | sort -V
}

mapfile -t X86_WORKERS < <(running_workers "sl-x86")
mapfile -t ARM_WORKERS < <(running_workers "sl-arm")

if [[ "$ARCH" == "amd64" ]]; then
    ACTIVE_WORKERS=("${X86_WORKERS[@]}")
    OTHER_WORKERS=("${ARM_WORKERS[@]}")
    ACTIVE_N_X86="${#X86_WORKERS[@]}"
    ACTIVE_N_ARM=0
else
    ACTIVE_WORKERS=("${ARM_WORKERS[@]}")
    OTHER_WORKERS=("${X86_WORKERS[@]}")
    ACTIVE_N_X86=0
    ACTIVE_N_ARM="${#ARM_WORKERS[@]}"
fi

if (( ${#ACTIVE_WORKERS[@]} == 0 )); then
    echo "Nessun worker $ARCH RUNNING trovato in $ZONE."
    exit 1
fi

if (( ${#OTHER_WORKERS[@]} > 0 )); then
    echo "ERRORE: sono attivi worker di entrambe le architetture."
    echo "Per questa raccolta deve essere attiva solo $ARCH."
    exit 1
fi

# La raccolta ha senso con un solo worker: una funzione alla volta su un nodo
# solo. Con piu' worker l'hash ring distribuirebbe le funzioni e free_memory_mb
# verrebbe da nodi diversi.
if (( ${#ACTIVE_WORKERS[@]} > 1 )); then
    echo "ATTENZIONE: sono attivi ${#ACTIVE_WORKERS[@]} worker $ARCH."
    echo "Questa raccolta e' pensata per UN solo worker per architettura."
    echo "Worker attivi: ${ACTIVE_WORKERS[*]}"
    exit 1
fi

WORKER="${ACTIVE_WORKERS[0]}"

EXPECTED_UNAME="x86_64"
[[ "$ARCH" == "arm64" ]] && EXPECTED_UNAME="aarch64"
ACTUAL_UNAME="$(remote_retry "$WORKER" "uname -m" </dev/null | tr -d '[:space:]')"
if [[ "$ACTUAL_UNAME" != "$EXPECTED_UNAME" ]]; then
    echo "ERRORE: $WORKER ha architettura ${ACTUAL_UNAME}, attesa ${EXPECTED_UNAME}."
    exit 1
fi

PHYSICAL_MEM_MB="$(
    remote_retry "$WORKER" "awk '/MemTotal:/ {printf \"%d\n\", \\$2/1024}' /proc/meminfo" \
        </dev/null | tr -d '[:space:]'
)"
if [[ ! "$PHYSICAL_MEM_MB" =~ ^[0-9]+$ ]]; then
    echo "ERRORE: impossibile determinare la RAM fisica del worker."
    exit 1
fi
if (( CONTAINER_POOL_MEMORY >= PHYSICAL_MEM_MB )); then
    echo "ERRORE: container.pool.memory=${CONTAINER_POOL_MEMORY} MB non puo' essere"
    echo ">= RAM fisica del worker (${PHYSICAL_MEM_MB} MB)."
    exit 1
fi

LB_IP="$(
    retry_cmd "$GCLOUD_RETRIES" "$GCLOUD_RETRY_DELAY" \
        gc compute instances describe "$NAME_LB" --zone="$ZONE" \
            --format="value(networkInterfaces[0].networkIP)"
)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="profiles_fast_${ARCH}_${STAMP}"
REMOTE_RESULT_ROOT="/opt/serverledge/results/${RUN_ID}"
LOCAL_RESULT_ROOT="${DEST_ROOT}/${RUN_ID}"
TOTAL_FUNCTIONS="${#FUNCTIONS[@]}"

banner "RACCOLTA PROFILI VELOCE — $ARCH"
echo "policy:                  $POLICY"
echo "worker:                  $WORKER (${ACTUAL_UNAME}, RAM ${PHYSICAL_MEM_MB} MB)"
echo "load balancer:           $LB_IP"
echo "funzioni:                $TOTAL_FUNCTIONS"
[[ -n "$ONLY_FUNCTIONS" ]] && echo "filtro funzioni:         $ONLY_FUNCTIONS"
echo "warm validi target:      $TARGET_WARM_VALID"
echo "tetto invocazioni:       $MAX_REQUESTS_PER_FUNCTION"
echo "CPU per funzione:        $FUNC_CPU"
echo "janitor:                 ogni ${JANITOR_INTERVAL}s"
echo "container expiration:    ${CONTAINER_EXPIRATION}s"
echo "container pool memory:   ${CONTAINER_POOL_MEMORY} MB"
echo "attesa svuotamento:      max ${DRAIN_TIMEOUT}s ogni ${DRAIN_POLL}s"
echo "cold start:              conservati"
echo "modalita' strict:        $STRICT"
echo "max runtime/funzione:    $MAX_RUNTIME"
echo "catalog sha256:          $CATALOG_SHA256"
echo "output locale:           $LOCAL_RESULT_ROOT"

# In caso di errore/interruzione prova a salvare localmente quanto gia'
# raccolto. Non viene eseguito nel percorso di successo e quindi non aggiunge
# overhead alla campagna normale.
RESCUE_READY=1
rescue_partial_results() {
    local exit_status="$1"
    [[ "$exit_status" -eq 0 || "${RESCUE_READY:-0}" -ne 1 ]] && return 0

    set +e
    echo
    echo "ATTENZIONE: campagna interrotta (exit=${exit_status})."
    echo "Provo a salvare localmente i dati parziali prima dell'uscita..."
    mkdir -p "${LOCAL_RESULT_ROOT}/partial/locust" "${LOCAL_RESULT_ROOT}/partial/raw"

    gc compute scp --zone="$ZONE" --quiet --recurse \
        "${NAME_WORKLOAD}:${REMOTE_RESULT_ROOT}" \
        "${LOCAL_RESULT_ROOT}/partial/locust/" >/dev/null 2>&1 || true

    remote_retry "$WORKER" \
        "sudo chmod a+r /var/lib/serverledge/profiling-samples.jsonl" \
        >/dev/null 2>&1 || true
    gc compute scp --zone="$ZONE" --quiet \
        "${WORKER}:/var/lib/serverledge/profiling-samples.jsonl" \
        "${LOCAL_RESULT_ROOT}/partial/raw/${WORKER}.jsonl" >/dev/null 2>&1 || true

    echo "Salvataggio best-effort completato in: ${LOCAL_RESULT_ROOT}/partial"
    set -e
}
trap 'status=$?; trap - EXIT; rescue_partial_results "$status"; exit "$status"' EXIT

# -----------------------------------------------------------------------------
# Preparazione generatore di workload.
# -----------------------------------------------------------------------------

banner "PREPARAZIONE WORKLOAD"

remote_retry "$NAME_WORKLOAD" "
    cd /opt/serverledge
    sudo git -c safe.directory=/opt/serverledge pull --ff-only --quiet

    if [ ! -x ${LOCUST_BIN} ]; then
        echo 'Creo virtualenv Locust...'
        python3 -m venv \$HOME/locust-venv
        \$HOME/locust-venv/bin/pip install --quiet --upgrade pip
        \$HOME/locust-venv/bin/pip install --quiet locust
    fi

    ${LOCUST_BIN} --version

    test -f ${EXPERIMENTS_DIR}/locustfile_collection.py || {
        echo 'FATAL: locustfile_collection.py non presente sul workload.'
        exit 1
    }

    grep -q 'TARGET_WARM_VALID' ${EXPERIMENTS_DIR}/locustfile_collection.py || {
        echo 'FATAL: il locustfile remoto non supporta TARGET_WARM_VALID.'
        echo 'Fai il push della versione aggiornata e riprova.'
        exit 1
    }

    mkdir -p ${WORK_DIR}
    cp ${EXPERIMENTS_DIR}/locustfile_collection.py ${WORK_DIR}/

    sudo mkdir -p ${REMOTE_RESULT_ROOT}
    sudo chmod 777 ${REMOTE_RESULT_ROOT}
"

# -----------------------------------------------------------------------------
# Controllo catalogo e azzeramento dei campioni precedenti.
# -----------------------------------------------------------------------------

banner "CONTROLLO CATALOGO"

CHECK_SCRIPT="cd ${EXPERIMENTS_DIR}
missing=0
"

for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name runtime src memory handler <<< "$entry"
    CHECK_SCRIPT+="
if [ ! -f '${src}' ]; then
    echo 'MANCANTE: ${name} -> ${src}'
    missing=1
fi
"
done

CHECK_SCRIPT+="
if [ \"\$missing\" -ne 0 ]; then
    exit 1
fi
echo 'Catalogo OK: ${TOTAL_FUNCTIONS} sorgenti presenti.'
"

remote_retry "$NAME_WORKLOAD" "$CHECK_SCRIPT"

banner "AZZERAMENTO PROFILI PRECEDENTI"

remote_retry "$WORKER" "
    sudo mkdir -p /var/lib/serverledge
    sudo rm -f /var/lib/serverledge/profiling-samples.jsonl
    sudo chmod 777 /var/lib/serverledge
" >/dev/null
echo "  $WORKER: JSONL azzerato"

# -----------------------------------------------------------------------------
# Avvio del cluster: UNA sola volta, con janitor aggressivo.
# -----------------------------------------------------------------------------

banner "AVVIO CLUSTER (una sola volta)"

remote_retry "$WORKER" "
    sudo pkill -x serverledge >/dev/null 2>&1 || true
    ids=\$(sudo docker ps -aq)
    if [ -n \"\$ids\" ]; then
        sudo docker rm -f \$ids >/dev/null
    fi
" >/dev/null

retry_cmd "$CLUSTER_START_RETRIES" "$GCLOUD_RETRY_DELAY" \
    env \
        N_X86="$ACTIVE_N_X86" \
        N_ARM="$ACTIVE_N_ARM" \
        PROFILING=1 \
        JANITOR_INTERVAL="$JANITOR_INTERVAL" \
        CONTAINER_EXPIRATION="$CONTAINER_EXPIRATION" \
        CONTAINER_POOL_MEMORY="$CONTAINER_POOL_MEMORY" \
        "${SCRIPT_DIR}/gcp_start_cluster.sh" "$POLICY"

# -----------------------------------------------------------------------------
# Stato di riferimento del worker.
#
# Togliendo il riavvio fra una funzione e l'altra, la contabilita' in memoria
# del nodo (usedCPUs, warmPoolUsedMem) non viene piu' azzerata: se il percorso
# del janitor perdesse risorse, il nodo si crederebbe pieno dopo qualche decina
# di funzioni. Registriamo il valore di partenza e lo confrontiamo a ogni giro.
# -----------------------------------------------------------------------------

worker_status() {
    remote_retry "$WORKER" \
        "curl -fsS --max-time 10 http://localhost:1323/status" </dev/null
}

parse_status() {
    python3 -c "
import json, sys
raw = sys.stdin.read().strip()
try:
    data = json.loads(raw)
    warm = data.get('AvailableWarmContainers') or {}
    print('%d %d %d %.6f' % (
        sum(int(v) for v in warm.values()),
        int(data['AvailableMemory']),
        int(data['FreeMemory']),
        float(data['UsedCPU']),
    ))
except Exception:
    print('ERRORE')
"
}

banner "STATO INIZIALE DEL WORKER"

BASELINE_RAW="$(worker_status)"
read -r BASE_WARM BASE_AVAILABLE_MEM BASE_FREE_MEM BASE_CPU <<< "$(echo "$BASELINE_RAW" | parse_status)"

if [[ "$BASE_WARM" == "ERRORE" || -z "${BASE_CPU:-}" ]]; then
    echo "ERRORE: /status non raggiungibile o formato inatteso sul worker."
    echo "Risposta: ${BASELINE_RAW}"
    exit 1
fi

BASE_CONTAINERS="$(
    remote_retry "$WORKER" "sudo docker ps -q | wc -l" </dev/null | tr -d '[:space:]'
)"

echo "  container runtime:     $BASE_CONTAINERS"
echo "  container warm:        $BASE_WARM"
echo "  memoria disponibile:   ${BASE_AVAILABLE_MEM} MB"
echo "  memoria realmente free:${BASE_FREE_MEM} MB"
echo "  CPU usate:             $BASE_CPU"

# -----------------------------------------------------------------------------
# Registrazione di TUTTE le funzioni, una sola volta.
# Registrare in etcd non crea container: il worker resta vuoto.
# -----------------------------------------------------------------------------

banner "REGISTRAZIONE ${TOTAL_FUNCTIONS} FUNZIONI (una sola volta)"

REGISTER_SCRIPT="cd ${EXPERIMENTS_DIR}
export SERVERLEDGE_HOST=${LB_IP}
export SERVERLEDGE_PORT=1323
CLI=/opt/serverledge/bin/serverledge-cli
failed=0
"

for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name runtime src memory handler <<< "$entry"

    handler_flag=""
    if [[ -n "$handler" ]]; then
        handler_flag="--handler '${handler}'"
    fi

    REGISTER_SCRIPT+="
echo -n '  ${name}: '
if output=\$(\$CLI create -f '${name}' \\
    --runtime '${runtime}' \\
    --src '${src}' \\
    --memory '${memory}' \\
    --cpu '${FUNC_CPU}' \\
    ${handler_flag} \\
    --update 2>&1); then
    printf '%s\n' "\$output" | tr -d '\\n '
    echo
else
    status=\$?
    printf '%s\n' "\$output"
    echo '  ${name}: FALLITA (exit='"\$status"')'
    failed=1
fi
"
done

REGISTER_SCRIPT+="
if [ \"\$failed\" -ne 0 ]; then
    echo 'FATAL: registrazione incompleta.'
    exit 1
fi
"

remote_retry "$NAME_WORKLOAD" "$REGISTER_SCRIPT"

# -----------------------------------------------------------------------------
# Attesa reale dello svuotamento del worker.
# -----------------------------------------------------------------------------

wait_worker_drained() {
    local label="$1"
    local waited=0
    local containers

    while (( waited < DRAIN_TIMEOUT )); do
        containers="$(
            remote_retry "$WORKER" "sudo docker ps -q | wc -l" </dev/null \
                | tr -d '[:space:]'
        )"

        if [[ "$containers" == "$BASE_CONTAINERS" ]]; then
            echo "  worker svuotato dopo ${waited}s"
            return 0
        fi

        sleep "$DRAIN_POLL"
        waited=$((waited + DRAIN_POLL))
    done

    echo "ERRORE: dopo ${DRAIN_TIMEOUT}s il worker ha ancora container attivi"
    echo "        (attesi ${BASE_CONTAINERS}, presenti ${containers}) — ${label}"
    return 1
}

check_accounting() {
    local label="$1"
    local warm available_mem free_mem cpu

    read -r warm available_mem free_mem cpu <<< "$(worker_status | parse_status)"

    if [[ "$warm" == "ERRORE" || -z "${cpu:-}" ]]; then
        echo "ERRORE: /status non leggibile dopo ${label}."
        return 1
    fi

    # AvailableMemory non sottrae il warm pool; FreeMemory invece sottrae sia
    # busyPoolUsedMem sia warmPoolUsedMem. Per rilevare una perdita contabile
    # del janitor dobbiamo quindi confrontare ANCHE FreeMemory.
    if [[ "$warm" != "$BASE_WARM" \
       || "$available_mem" != "$BASE_AVAILABLE_MEM" \
       || "$free_mem" != "$BASE_FREE_MEM" \
       || "$cpu" != "$BASE_CPU" ]]; then
        echo "ERRORE: contabilita' del nodo non tornata al baseline dopo ${label}."
        echo "        warm:              ${BASE_WARM} -> ${warm}"
        echo "        available memory:  ${BASE_AVAILABLE_MEM} -> ${available_mem} MB"
        echo "        free memory:       ${BASE_FREE_MEM} -> ${free_mem} MB"
        echo "        CPU usate:         ${BASE_CPU} -> ${cpu}"
        echo "        Interrompo per evitare drift cumulativo nella campagna."
        return 1
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Verifica che registrare le funzioni non abbia creato container né alterato
# la contabilità prima della prima invocazione.
# -----------------------------------------------------------------------------

wait_worker_drained "post-registrazione"
check_accounting "post-registrazione"

# -----------------------------------------------------------------------------
# Esecuzione: una funzione alla volta.
# -----------------------------------------------------------------------------

banner "ESECUZIONE — ${TOTAL_FUNCTIONS} FUNZIONI"

INCOMPLETE_FILE="${LOCAL_RESULT_ROOT}/incomplete_functions.txt"
mkdir -p "$LOCAL_RESULT_ROOT"
: > "$INCOMPLETE_FILE"

index=0
for entry in "${FUNCTIONS[@]}"; do
    index=$((index + 1))
    IFS='|' read -r name runtime src memory handler <<< "$entry"

    printf -v slot "func_%02d_%s" "$index" "$name"
    remote_dir="${REMOTE_RESULT_ROOT}/${slot}"

    banner "[${index}/${TOTAL_FUNCTIONS}] ${name}"

    remote_once "$NAME_WORKLOAD" "
        mkdir -p '${remote_dir}'
        cd ${WORK_DIR}

        export FUNCTIONS_TO_RUN='${name}'
        export TARGET_WARM_VALID='${TARGET_WARM_VALID}'
        export MAX_REQUESTS_PER_FUNCTION='${MAX_REQUESTS_PER_FUNCTION}'
        export REQUESTS_PER_FUNCTION='${MAX_REQUESTS_PER_FUNCTION}'
        export LB_POLICY='${POLICY}'
        export COLLECTION_CSV='${remote_dir}/experiment_results.csv'
        export COLLECTION_SUMMARY='${remote_dir}/summary.csv'

        ${LOCUST_BIN} -f locustfile_collection.py \\
            --headless \\
            --users 1 \\
            --spawn-rate 1 \\
            --run-time '${MAX_RUNTIME}' \\
            --host 'http://${LB_IP}:1323' \\
            --csv '${remote_dir}/locust'
    "

    summary_line="$(
        remote_retry "$NAME_WORKLOAD" \
            "tail -n 1 '${remote_dir}/summary.csv' 2>/dev/null || true" \
            </dev/null | tr -d '\r'
    )"

    warm_valid="$(echo "$summary_line" | cut -d',' -f3)"
    target_reached="$(echo "$summary_line" | cut -d',' -f4)"

    # Anche se la funzione e' incompleta, prima ripristiniamo il worker al
    # baseline: in STRICT=1 potremo uscire senza lasciare stato ambiguo.
    echo "  attendo il janitor..."
    wait_worker_drained "$name"
    check_accounting "$name"

    if [[ "$target_reached" != "1" ]]; then
        echo "  INCOMPLETA: ${warm_valid:-0}/${TARGET_WARM_VALID} warm validi"
        echo "$name ${warm_valid:-0}/${TARGET_WARM_VALID}" >> "$INCOMPLETE_FILE"

        if [[ "$STRICT" == "1" ]]; then
            echo
            echo "STRICT=1: interrompo la campagna."
            echo "I dati raccolti finora restano su ${REMOTE_RESULT_ROOT}."
            echo "Per proseguire ignorando le funzioni incomplete: STRICT=0"
            exit 1
        fi
    else
        echo "  ok: ${warm_valid}/${TARGET_WARM_VALID} warm validi"
    fi

done

# -----------------------------------------------------------------------------
# Download risultati.
# -----------------------------------------------------------------------------

banner "DOWNLOAD RISULTATI"

mkdir -p "${LOCAL_RESULT_ROOT}/locust" "${LOCAL_RESULT_ROOT}/raw"

retry_cmd "$GCLOUD_RETRIES" "$GCLOUD_RETRY_DELAY" \
    gc compute scp --zone="$ZONE" --quiet --recurse \
        "${NAME_WORKLOAD}:${REMOTE_RESULT_ROOT}" \
        "${LOCAL_RESULT_ROOT}/locust/"

remote_retry "$WORKER" \
    "sudo chmod a+r /var/lib/serverledge/profiling-samples.jsonl" \
    >/dev/null 2>&1 || true

if retry_cmd "$GCLOUD_RETRIES" "$GCLOUD_RETRY_DELAY" \
    gc compute scp --zone="$ZONE" --quiet \
        "${WORKER}:/var/lib/serverledge/profiling-samples.jsonl" \
        "${LOCAL_RESULT_ROOT}/raw/${WORKER}.jsonl"; then
    echo "  ${WORKER}: profiling JSONL scaricato"
else
    echo "ERRORE: profiling JSONL assente su ${WORKER}"
    exit 1
fi

# Un solo worker: il raw del worker coincide con il dataset cumulativo. Manteniamo
# anche all_samples.jsonl per compatibilita' con gli script di analisi esistenti.
cp "${LOCAL_RESULT_ROOT}/raw/${WORKER}.jsonl" "${LOCAL_RESULT_ROOT}/all_samples.jsonl"

REMOTE_COMMIT="$(
    remote_retry "$NAME_WORKLOAD" \
        "git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD" \
        </dev/null | tr -d '[:space:]'
)"

cat > "${LOCAL_RESULT_ROOT}/manifest.txt" <<EOF
raccolta:                profiling-clustering (variante veloce)
architettura:            ${ARCH}
policy:                  ${POLICY}
data:                    $(date -Iseconds)
progetto:                ${PROJECT}
zona:                    ${ZONE}
worker:                  ${WORKER}
numero worker:           1
CPU funzione:            ${FUNC_CPU}
warm validi target:      ${TARGET_WARM_VALID}
tetto invocazioni:       ${MAX_REQUESTS_PER_FUNCTION}
janitor interval:        ${JANITOR_INTERVAL}
container expiration:    ${CONTAINER_EXPIRATION}
container pool memory:   ${CONTAINER_POOL_MEMORY}
max runtime/funzione:     ${MAX_RUNTIME}
catalog sha256:           ${CATALOG_SHA256}
numero funzioni:         ${TOTAL_FUNCTIONS}
avvii cluster:           1
registrazioni:           1
commit remoto:           ${REMOTE_COMMIT}
cold start:              conservati nel raw dataset
warm clustering:         filtrare warm + success + valid + exclusive
EOF

: > "${LOCAL_RESULT_ROOT}/expected_functions.txt"
for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$entry"
    echo "$name" >> "${LOCAL_RESULT_ROOT}/expected_functions.txt"
done

# -----------------------------------------------------------------------------
# Riepilogo finale sul dataset raw.
# -----------------------------------------------------------------------------

banner "RIEPILOGO RAW"

set +e
python3 - "${LOCAL_RESULT_ROOT}" "${ARCH}" "${TARGET_WARM_VALID}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected_arch = sys.argv[2]
target_warm_valid = int(sys.argv[3])
expected = [
    line.strip()
    for line in (root / "expected_functions.txt").read_text().splitlines()
    if line.strip()
]
files = sorted((root / "raw").glob("sl-*.jsonl"))

counts = Counter()
unexpected_arch = Counter()
wrong_cpu = Counter()
invalid_json = 0

for path in files:
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                invalid_json += 1
                continue

            name = row.get("function_name", "unknown")
            counts[(name, "total")] += 1

            machine_tag = row.get("machine_tag")
            if machine_tag != expected_arch:
                unexpected_arch[(name, machine_tag)] += 1

            config = row.get("function_configuration") or {}
            cpu = config.get("configured_cpus")
            if cpu != 1 and cpu != 1.0:
                wrong_cpu[(name, cpu)] += 1

            if not row.get("execution_succeeded", False):
                counts[(name, "failed")] += 1
                continue

            if not row.get("warm_start", False):
                counts[(name, "cold")] += 1
                continue

            counts[(name, "warm")] += 1
            profile = row.get("profile") or {}
            eligibility = row.get("eligibility") or {}

            if (
                profile.get("valid") is True
                and profile.get("exclusive_container") is True
                and eligibility.get("resource_clustering") is True
                and eligibility.get("performance_analysis") is True
            ):
                counts[(name, "warm_valid")] += 1

for name in expected:
    print(
        f"{name:24s} "
        f"total={counts[(name, 'total')]:2d} "
        f"cold={counts[(name, 'cold')]:2d} "
        f"warm={counts[(name, 'warm')]:2d} "
        f"warm_valid={counts[(name, 'warm_valid')]:2d} "
        f"failed={counts[(name, 'failed')]:2d}"
    )

problems = False
missing = [name for name in expected if counts[(name, "total")] == 0]
insufficient = [
    name for name in expected
    if counts[(name, "warm_valid")] < target_warm_valid
]

print()
if missing:
    problems = True
    print("ERRORE: nessun campione per:")
    for name in missing:
        print(f"  - {name}")

if insufficient:
    problems = True
    print(f"ERRORE: meno di {target_warm_valid} warm validi per:")
    for name in insufficient:
        print(f"  - {name}: {counts[(name, 'warm_valid')]} warm validi")

if unexpected_arch:
    problems = True
    print("ERRORE: machine_tag inattesi:")
    for (name, tag), count in sorted(unexpected_arch.items(), key=lambda x: str(x[0])):
        print(f"  - {name}: {tag!r} x{count}")

if wrong_cpu:
    problems = True
    print("ERRORE: configured_cpus diverso da 1:")
    for (name, cpu), count in sorted(wrong_cpu.items(), key=lambda x: str(x[0])):
        print(f"  - {name}: {cpu!r} x{count}")

if invalid_json:
    problems = True
    print(f"ERRORE: {invalid_json} righe JSONL non parseabili.")

if problems:
    sys.exit(2)

print(
    f"OK: {len(expected)} funzioni presenti, architettura corretta, CPU=1 e "
    f"almeno {target_warm_valid} warm validi ciascuna."
)
PY
SUMMARY_STATUS=$?
set -e

if (( SUMMARY_STATUS == 0 )); then
    RESCUE_READY=0
fi

banner "FATTO"
echo "Risultati in: ${LOCAL_RESULT_ROOT}"

if [[ -s "$INCOMPLETE_FILE" ]]; then
    echo
    echo "Funzioni incomplete elencate in:"
    echo "  ${INCOMPLETE_FILE}"
fi

if (( SUMMARY_STATUS != 0 )); then
    echo
    echo "Il riepilogo ha segnalato funzioni assenti o sotto target."
    echo "Ripetile con ONLY_FUNCTIONS prima di considerare chiusa la campagna."
    exit "$SUMMARY_STATUS"
fi
