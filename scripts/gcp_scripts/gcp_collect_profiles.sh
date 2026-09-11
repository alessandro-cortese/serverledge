#!/usr/bin/env bash
#
# Raccolta controllata dei profili Serverledge su una sola architettura.
#
# Uso:
#   ./gcp_collect_profiles.sh amd64
#   ./gcp_collect_profiles.sh arm64
#
# Precondizioni:
#   - le VM della sola architettura scelta sono gia' state create con gcp_up.sh;
#   - il repository locale e origin/main contengono locustfile_collection.py;
#   - la raccolta usa RoundRobin e profiling attivo;
#   - CPU configurata = 1 per tutte le funzioni.
#
# Lo script:
#   1. verifica che sia attiva una sola architettura;
#   2. prepara Locust sul generatore di workload;
#   3. azzera i vecchi JSONL di profiling una sola volta;
#   4. divide le 53 implementazioni in batch;
#   5. prima di ogni batch elimina i container warm e riavvia il cluster;
#   6. registra solo le funzioni del batch;
#   7. esegue un numero finito di invocazioni per funzione con Locust;
#   8. conserva sia cold sia warm;
#   9. scarica CSV e profiling-samples.jsonl in locale.
#
# Variabili opzionali:
#   BATCH_SIZE=3
#   REQUESTS_PER_FUNCTION=16
#   MAX_RUNTIME=90m
#   DEST_ROOT=./risultati
#   ONLY_FUNCTIONS=nome1,nome2,...   # opzionale, utile per preflight/rerun
#   MIN_WARM_VALID=10                 # soglia controllo finale
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

ARCH="${1:-}"
POLICY="RoundRobin"
BATCH_SIZE="${BATCH_SIZE:-3}"
REQUESTS_PER_FUNCTION="${REQUESTS_PER_FUNCTION:-16}"
MAX_RUNTIME="${MAX_RUNTIME:-90m}"
FUNC_CPU="1"
EXPERIMENTS_DIR="/opt/serverledge/examples/experiments"
WORK_DIR="\$HOME/profile-collection"
LOCUST_BIN="\$HOME/locust-venv/bin/locust"
DEST_ROOT="${DEST_ROOT:-${SCRIPT_DIR}/risultati}"
ONLY_FUNCTIONS="${ONLY_FUNCTIONS:-}"
MIN_WARM_VALID="${MIN_WARM_VALID:-10}"

case "$ARCH" in
    amd64|arm64) ;;
    *)
        echo "Uso: $0 {amd64|arm64}"
        exit 1
        ;;
esac

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE deve essere un intero > 0"
    exit 1
fi

if ! [[ "$REQUESTS_PER_FUNCTION" =~ ^[1-9][0-9]*$ ]]; then
    echo "REQUESTS_PER_FUNCTION deve essere un intero > 0"
    exit 1
fi

if ! [[ "$MIN_WARM_VALID" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIN_WARM_VALID deve essere un intero > 0"
    exit 1
fi

# Formato:
#   nome|runtime|sorgente|memoria_MB|handler
#
# I primi 30 elementi sono il corpus gia' usato negli esperimenti precedenti.
# Seguono 15 implementazioni Python e 8 Node.js del corpus multilinguaggio.
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

    "base64stream-py|python314|multilang/python/base64stream.py|1024|base64stream.handler"
    "compression-py|python314|multilang/python/compression_bench.py|1024|compression_bench.handler"
    "dna-visualisation-py|python314|multilang/python/dna_visualisation.py|1024|dna_visualisation.handler"
    "dynamic-html-py|python314|multilang/python/dynamic_html.py|1024|dynamic_html.handler"
    "float-ops-py|python314|multilang/python/float_ops.py|1024|float_ops.handler"
    "graph-bfs-py|python314|multilang/python/graph_bfs.py|1024|graph_bfs.handler"
    "graph-mst-py|python314|multilang/python/graph_mst.py|1024|graph_mst.handler"
    "graph-pagerank-py|python314|multilang/python/graph_pagerank.py|1024|graph_pagerank.handler"
    "hashing-py|python314|multilang/python/hashing.py|1024|hashing.handler"
    "json-dumps-py|python314|multilang/python/json_dumps.py|1024|json_dumps.handler"
    "jsonparse-py|python314|multilang/python/jsonparse.py|1024|jsonparse.handler"
    "memory-rw-py|python314|multilang/python/memory_rw.py|1024|memory_rw.handler"
    "pointerchase-py|python314|multilang/python/pointerchase.py|1024|pointerchase.handler"
    "randomaccess-py|python314|multilang/python/randomaccess.py|1024|randomaccess.handler"
    "thumbnailer-py|python314|multilang/python/thumbnailer.py|1024|thumbnailer.handler"

    "compression-node|nodejs17ng|multilang/nodejs/compression.js|1024|compression"
    "dynamic-html-node|nodejs17ng|multilang/nodejs/dynamic_html.js|1024|dynamic_html"
    "float-ops-node|nodejs17ng|multilang/nodejs/float_ops.js|1024|float_ops"
    "graph-bfs-node|nodejs17ng|multilang/nodejs/graph_bfs.js|1024|graph_bfs"
    "graph-pagerank-node|nodejs17ng|multilang/nodejs/graph_pagerank.js|1024|graph_pagerank"
    "json-dumps-node|nodejs17ng|multilang/nodejs/json_dumps.js|1024|json_dumps"
    "memory-rw-node|nodejs17ng|multilang/nodejs/memory_rw.js|1024|memory_rw"
    "thumbnailer-node|nodejs17ng|multilang/nodejs/thumbnailer.js|1024|thumbnailer"
)

# Se richiesto, limita il catalogo a un sottoinsieme esplicito. È utile per
# validare lo script con un preflight economico o per ripetere solo funzioni
# che non hanno raggiunto il numero minimo di campioni validi.
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

remote() {
    local host="$1"
    shift
    gc compute ssh "$host" --zone="$ZONE" --quiet --command="$*"
}

running_workers() {
    local prefix="$1"
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
    echo "Worker attivi richiesti: ${ACTIVE_WORKERS[*]}"
    echo "Worker dell'altra architettura: ${OTHER_WORKERS[*]}"
    exit 1
fi

LB_IP="$(
    gc compute instances describe "$NAME_LB" --zone="$ZONE" \
        --format="value(networkInterfaces[0].networkIP)"
)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="profiles_${ARCH}_${STAMP}"
REMOTE_RESULT_ROOT="/opt/serverledge/results/${RUN_ID}"
LOCAL_RESULT_ROOT="${DEST_ROOT}/${RUN_ID}"

banner "RACCOLTA PROFILI — $ARCH"
echo "policy:                 $POLICY"
echo "worker:                 ${ACTIVE_WORKERS[*]}"
echo "load balancer:          $LB_IP"
echo "funzioni:               ${#FUNCTIONS[@]}"
[[ -n "$ONLY_FUNCTIONS" ]] && echo "filtro funzioni:         $ONLY_FUNCTIONS"
echo "batch size:             $BATCH_SIZE"
echo "richieste per funzione: $REQUESTS_PER_FUNCTION"
echo "warm validi minimi:     $MIN_WARM_VALID"
echo "CPU per funzione:       $FUNC_CPU"
echo "cold start:             conservati"
echo "output locale:          $LOCAL_RESULT_ROOT"

# -----------------------------------------------------------------------------
# Preparazione generatore di workload
# -----------------------------------------------------------------------------

banner "PREPARAZIONE WORKLOAD"

remote "$NAME_WORKLOAD" "
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

    mkdir -p ${WORK_DIR}
    cp ${EXPERIMENTS_DIR}/locustfile_collection.py ${WORK_DIR}/

    sudo mkdir -p ${REMOTE_RESULT_ROOT}
    sudo chmod 777 ${REMOTE_RESULT_ROOT}
"

# -----------------------------------------------------------------------------
# Azzera solo i campioni precedenti. Durante i reset fra batch il JSONL resta.
# -----------------------------------------------------------------------------

banner "AZZERAMENTO PROFILI PRECEDENTI"

for host in "${ACTIVE_WORKERS[@]}"; do
    remote "$host" "
        sudo mkdir -p /var/lib/serverledge
        sudo rm -f /var/lib/serverledge/profiling-samples.jsonl
        sudo chmod 777 /var/lib/serverledge
    " >/dev/null
    echo "  $host: JSONL azzerato"
done

# -----------------------------------------------------------------------------
# Controllo che tutti i sorgenti del catalogo esistano sul workload.
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
echo 'Catalogo OK: ${#FUNCTIONS[@]} sorgenti presenti.'
"

remote "$NAME_WORKLOAD" "$CHECK_SCRIPT"

# -----------------------------------------------------------------------------
# Reset del cluster prima di ogni batch.
# I container vengono rimossi a processo worker fermo; poi gcp_start_cluster.sh
# riparte con etcd vuoto, profiling attivo e senza stato warm precedente.
# Il profiling-samples.jsonl NON viene cancellato qui.
# -----------------------------------------------------------------------------

reset_cluster_for_batch() {
    banner "RESET CONTAINER"

    for host in "${ACTIVE_WORKERS[@]}"; do
        remote "$host" "
            sudo pkill -x serverledge >/dev/null 2>&1 || true
            ids=\$(sudo docker ps -aq)
            if [ -n \"\$ids\" ]; then
                sudo docker rm -f \$ids >/dev/null
            fi
        " >/dev/null
        echo "  $host: container rimossi"
    done

    N_X86="$ACTIVE_N_X86" \
    N_ARM="$ACTIVE_N_ARM" \
    PROFILING=1 \
        "${SCRIPT_DIR}/gcp_start_cluster.sh" "$POLICY"
}

# -----------------------------------------------------------------------------
# Registrazione di un singolo batch.
# -----------------------------------------------------------------------------

register_batch() {
    local -n batch_ref=$1
    local register_script

    register_script="cd ${EXPERIMENTS_DIR}
export SERVERLEDGE_HOST=${LB_IP}
export SERVERLEDGE_PORT=1323
CLI=/opt/serverledge/bin/serverledge-cli
"

    for entry in "${batch_ref[@]}"; do
        IFS='|' read -r name runtime src memory handler <<< "$entry"

        handler_flag=""
        if [[ -n "$handler" ]]; then
            handler_flag="--handler '${handler}'"
        fi

        register_script+="
echo -n '  ${name}: '
\$CLI create -f '${name}' \\
    --runtime '${runtime}' \\
    --src '${src}' \\
    --memory '${memory}' \\
    --cpu '${FUNC_CPU}' \\
    ${handler_flag} \\
    --update 2>&1 | tr -d '\\n '
echo
"
    done

    remote "$NAME_WORKLOAD" "$register_script"
}

# -----------------------------------------------------------------------------
# Locust batch.
# Nessuna invocazione preliminare: la prima richiesta Locust resta il cold start.
# -----------------------------------------------------------------------------

run_batch() {
    local batch_number="$1"
    shift
    local batch_entries=("$@")
    local function_names=()
    local entry name runtime src memory handler

    for entry in "${batch_entries[@]}"; do
        IFS='|' read -r name runtime src memory handler <<< "$entry"
        function_names+=("$name")
    done

    local functions_csv
    functions_csv="$(IFS=,; echo "${function_names[*]}")"

    local batch_tag
    printf -v batch_tag "batch_%02d" "$batch_number"

    local remote_batch_dir="${REMOTE_RESULT_ROOT}/${batch_tag}"
    local users="${#function_names[@]}"

    banner "${batch_tag} — ${functions_csv}"

    reset_cluster_for_batch

    banner "REGISTRAZIONE ${batch_tag}"
    register_batch batch_entries

    banner "LOCUST ${batch_tag}"

    remote "$NAME_WORKLOAD" "
        mkdir -p '${remote_batch_dir}'
        cd ${WORK_DIR}

        export FUNCTIONS_TO_RUN='${functions_csv}'
        export REQUESTS_PER_FUNCTION='${REQUESTS_PER_FUNCTION}'
        export LB_POLICY='${POLICY}'
        export COLLECTION_CSV='${remote_batch_dir}/experiment_results.csv'

        ${LOCUST_BIN} -f locustfile_collection.py \\
            --headless \\
            --users '${users}' \\
            --spawn-rate '${users}' \\
            --run-time '${MAX_RUNTIME}' \\
            --host 'http://${LB_IP}:1323' \\
            --csv '${remote_batch_dir}/locust'
    "

    banner "CONTROLLO ${batch_tag}"

    remote "$NAME_WORKLOAD" "python3 - '${remote_batch_dir}/experiment_results.csv' <<'PY'
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
counts = defaultdict(lambda: {
    'total': 0,
    'cold': 0,
    'warm': 0,
    'warm_valid': 0,
    'failed': 0,
})

with open(path, newline='') as f:
    for row in csv.DictReader(f):
        name = row['function']
        c = counts[name]
        c['total'] += 1

        success = row['execution_succeeded'].lower() == 'true'
        warm = row['is_warm_start'].lower() == 'true'
        valid = row['profile_valid'].lower() == 'true'
        exclusive = row['exclusive_container'].lower() == 'true'

        if not success:
            c['failed'] += 1
        elif warm:
            c['warm'] += 1
            if valid and exclusive:
                c['warm_valid'] += 1
        else:
            c['cold'] += 1

for name in sorted(counts):
    c = counts[name]
    print(
        f\"{name}: total={c['total']} cold={c['cold']} warm={c['warm']} \"
        f\"warm_valid={c['warm_valid']} failed={c['failed']}\"
    )
PY"
}

# -----------------------------------------------------------------------------
# Esecuzione di tutti i batch.
# -----------------------------------------------------------------------------

TOTAL_FUNCTIONS="${#FUNCTIONS[@]}"
TOTAL_BATCHES=$(( (TOTAL_FUNCTIONS + BATCH_SIZE - 1) / BATCH_SIZE ))

banner "ESECUZIONE — ${TOTAL_BATCHES} BATCH"

batch_number=0
for ((start = 0; start < TOTAL_FUNCTIONS; start += BATCH_SIZE)); do
    batch_number=$((batch_number + 1))
    batch=("${FUNCTIONS[@]:start:BATCH_SIZE}")
    run_batch "$batch_number" "${batch[@]}"
done

# -----------------------------------------------------------------------------
# Raccolta locale prima che le VM vengano distrutte.
# -----------------------------------------------------------------------------

banner "DOWNLOAD RISULTATI"

mkdir -p "${LOCAL_RESULT_ROOT}/locust" "${LOCAL_RESULT_ROOT}/raw"

# Risultati Locust di tutti i batch.
gc compute scp --zone="$ZONE" --quiet --recurse \
    "${NAME_WORKLOAD}:${REMOTE_RESULT_ROOT}" \
    "${LOCAL_RESULT_ROOT}/locust/"

# Profiling JSONL cumulativo di ogni worker.
for host in "${ACTIVE_WORKERS[@]}"; do
    remote "$host" "sudo chmod a+r /var/lib/serverledge/profiling-samples.jsonl" \
        >/dev/null 2>&1 || true

    if gc compute scp --zone="$ZONE" --quiet \
        "${host}:/var/lib/serverledge/profiling-samples.jsonl" \
        "${LOCAL_RESULT_ROOT}/raw/${host}.jsonl"; then
        echo "  ${host}: profiling JSONL scaricato"
    else
        echo "ERRORE: profiling JSONL assente su ${host}"
        exit 1
    fi
done

REMOTE_COMMIT="$(
    remote "$NAME_WORKLOAD" \
        "git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD" \
        </dev/null | tr -d '[:space:]'
)"

cat > "${LOCAL_RESULT_ROOT}/manifest.txt" <<EOF
raccolta:               profiling-clustering
architettura:           ${ARCH}
policy:                  ${POLICY}
data:                    $(date -Iseconds)
progetto:                ${PROJECT}
zona:                    ${ZONE}
worker:                  ${ACTIVE_WORKERS[*]}
numero worker:           ${#ACTIVE_WORKERS[@]}
CPU funzione:            ${FUNC_CPU}
batch size:              ${BATCH_SIZE}
richieste per funzione:  ${REQUESTS_PER_FUNCTION}
warm validi minimi:      ${MIN_WARM_VALID}
numero funzioni:         ${TOTAL_FUNCTIONS}
numero batch:            ${TOTAL_BATCHES}
commit remoto:           ${REMOTE_COMMIT}
cold start:              conservati nel raw dataset
warm clustering:         filtrare warm + success + valid + exclusive
EOF

# Elenco atteso usato anche dal controllo finale.
: > "${LOCAL_RESULT_ROOT}/expected_functions.txt"
for entry in "${FUNCTIONS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$entry"
    echo "$name" >> "${LOCAL_RESULT_ROOT}/expected_functions.txt"
done

banner "RIEPILOGO RAW"

set +e
python3 - "${LOCAL_RESULT_ROOT}" "${ARCH}" "${MIN_WARM_VALID}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected_arch = sys.argv[2]
min_warm_valid = int(sys.argv[3])
expected = [
    line.strip()
    for line in (root / "expected_functions.txt").read_text().splitlines()
    if line.strip()
]
files = sorted((root / "raw").glob("sl-*.jsonl"))

counts = Counter()
unexpected_arch = Counter()
wrong_cpu = Counter()

for path in files:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
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
    if counts[(name, "warm_valid")] < min_warm_valid
]

print()
if missing:
    problems = True
    print("ERRORE: nessun campione per:")
    for name in missing:
        print(f"  - {name}")

if insufficient:
    problems = True
    print(f"ATTENZIONE: meno di {min_warm_valid} warm validi per:")
    for name in insufficient:
        print(f"  - {name}: {counts[(name, 'warm_valid')]} warm validi")

if unexpected_arch:
    problems = True
    print("ERRORE: machine_tag inattesi:")
    for (name, tag), count in sorted(unexpected_arch.items()):
        print(f"  - {name}: {tag!r} x{count}")

if wrong_cpu:
    problems = True
    print("ERRORE: configured_cpus diverso da 1:")
    for (name, cpu), count in sorted(wrong_cpu.items(), key=lambda item: str(item[0])):
        print(f"  - {name}: {cpu!r} x{count}")

if problems:
    sys.exit(2)

print(
    f"OK: {len(expected)} funzioni presenti, architettura corretta, "
    f"CPU=1 e almeno {min_warm_valid} warm validi ciascuna."
)
PY

STATUS=$?
set -e

cat "${LOCAL_RESULT_ROOT}"/raw/sl-*.jsonl > "${LOCAL_RESULT_ROOT}/all_samples.jsonl"

banner "FATTO"
echo "Dati salvati in: ${LOCAL_RESULT_ROOT}"
echo "Manifest:        ${LOCAL_RESULT_ROOT}/manifest.txt"
echo
echo "I cold start sono stati conservati e NON entrano automaticamente nel clustering."
echo "Non eseguire gcp_down.sh prima di aver verificato che questa cartella contenga i dati attesi."

exit "$STATUS"