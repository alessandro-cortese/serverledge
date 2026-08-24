#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Serverledge - Pre-cloud regression gate
#
# Script locale di validazione prima degli esperimenti GCP.
# Non modifica il codice di produzione.
#
# Usa:
#   .venv-analysis/bin/python
#
# Produce:
#   logs/precloud_regression_<timestamp>.txt
#
# IMPORTANTE:
#   prima della full Go suite rimuove SOLO l'eventuale
#   container Docker del progetto chiamato "Etcd-server",
#   perché internal/test avvia un proprio etcd embedded.
# ============================================================


# ------------------------------------------------------------
# ROOT DEL PROGETTO
# ------------------------------------------------------------

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" \
    && pwd
)"

ROOT="$(
    cd -- "${SCRIPT_DIR}/.." \
    && pwd
)"

cd "$ROOT"


# ------------------------------------------------------------
# LOG
# ------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="${ROOT}/logs"

mkdir -p "$LOG_DIR"

TEST_LOG="${TEST_LOG:-${LOG_DIR}/precloud_regression_${STAMP}.txt}"


# Tutto quello che segue viene mostrato a schermo
# e contemporaneamente scritto nel file.
exec > >(
    tee -a "$TEST_LOG"
) 2>&1


# ------------------------------------------------------------
# STATO DEL TEST
# ------------------------------------------------------------

CURRENT_STAGE="initialization"

TMP_DIR=""


cleanup() {
    local status=$?

    if [[ -n "${TMP_DIR:-}" ]] && [[ -d "$TMP_DIR" ]]; then
        rm -rf "$TMP_DIR"
    fi

    echo
    echo "============================================================"

    if [[ "$status" -eq 0 ]]; then
        echo "PRECLOUD REGRESSION GATE: PASS"
    else
        echo "PRECLOUD REGRESSION GATE: FAIL"
        echo "Failed stage: ${CURRENT_STAGE}"
        echo "Exit status: ${status}"
    fi

    echo "Log:"
    echo "$TEST_LOG"

    echo "============================================================"

    exit "$status"
}

trap cleanup EXIT


stage() {
    CURRENT_STAGE="$1"

    echo
    echo "============================================================"
    echo "$CURRENT_STAGE"
    echo "============================================================"
}


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

require_command() {
    local command_name="$1"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "FAIL: comando non trovato: $command_name"
        exit 1
    fi
}


port_is_busy() {
    local port="$1"

    ss -H -ltn "sport = :${port}" 2>/dev/null \
        | grep -q .
}


show_relevant_ports() {
    echo
    echo "Listener rilevati sulle porte rilevanti:"

    ss -ltnp 2>/dev/null \
        | grep -E ':(1323|2379|2380)[[:space:]]' \
        || true
}


cleanup_repository_etcd() {
    echo "Controllo eventuale Etcd-server Docker..."

    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker non disponibile: impossibile controllare Etcd-server."
        return
    fi

    if docker ps -a \
        --format '{{.Names}}' 2>/dev/null \
        | grep -Fxq 'Etcd-server'; then

        echo "Trovato container Docker Etcd-server."
        echo "Lo rimuovo prima dei test con etcd embedded."

        docker rm -f Etcd-server >/dev/null

        echo "Etcd-server rimosso."
    else
        echo "Nessun container Docker Etcd-server presente."
    fi

    # Residuo previsto anche dallo script stop-etcd.sh originale.
    rm -rf "${ROOT}/default.etcd"
}


assert_test_ports_free() {
    local busy=0

    for port in 1323 2379 2380; do

        if port_is_busy "$port"; then
            echo "FAIL: porta ${port} già occupata."
            busy=1
        else
            echo "PASS: porta ${port} libera."
        fi

    done

    if [[ "$busy" -ne 0 ]]; then

        show_relevant_ports

        echo
        echo "Una porta necessaria ai test è occupata."
        echo "Non modifico o termino processi sconosciuti automaticamente."
        echo "Liberare la porta indicata e rilanciare lo script."

        exit 1
    fi
}


# ------------------------------------------------------------
# VERIFICA ROOT
# ------------------------------------------------------------

stage "PROJECT ROOT CHECK"

if [[ ! -f "${ROOT}/go.mod" ]]; then
    echo "FAIL: go.mod non trovato."
    exit 1
fi

if [[ ! -d "${ROOT}/internal" ]]; then
    echo "FAIL: directory internal non trovata."
    exit 1
fi

if [[ ! -d "${ROOT}/scripts" ]]; then
    echo "FAIL: directory scripts non trovata."
    exit 1
fi

echo "PASS: root Serverledge"
echo "$ROOT"


# ------------------------------------------------------------
# TOOLCHAIN
# ------------------------------------------------------------

stage "TOOLCHAIN CHECK"

require_command git
require_command go
require_command docker
require_command gofmt
require_command make
require_command ss

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-analysis/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FAIL: ambiente Python non trovato:"
    echo "$PYTHON_BIN"
    echo
    echo "Atteso:"
    echo "${ROOT}/.venv-analysis/bin/python"
    exit 1
fi

echo "Python utilizzato:"
echo "$PYTHON_BIN"

"$PYTHON_BIN" --version

echo
echo "Verifica NumPy e scikit-learn..."

"$PYTHON_BIN" - <<'PY'
import numpy
import sklearn

print("numpy =", numpy.__version__)
print("sklearn =", sklearn.__version__)
PY

echo "PASS: ambiente Python analysis"


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

stage "ENVIRONMENT"

date

echo
echo "Repository:"
pwd

echo
echo "Git commit:"
git rev-parse HEAD

echo
echo "Git status:"
git status --short

echo
echo "Go:"
go version

echo
echo "Python:"
"$PYTHON_BIN" --version

echo
echo "Docker:"
docker --version


# ------------------------------------------------------------
# GIT DIFF CHECK
# ------------------------------------------------------------

stage "GIT DIFF CHECK"

git diff --check

echo "PASS: git diff --check"


# ------------------------------------------------------------
# GOFMT
# ------------------------------------------------------------

stage "GOFMT CHECK"

UNFORMATTED="$(
    find \
        internal \
        cmd \
        serverledge \
        -type f \
        -name '*.go' \
        -exec gofmt -l {} +
)"

if [[ -n "$UNFORMATTED" ]]; then

    echo "FAIL: file Go non formattati:"
    echo "$UNFORMATTED"

    exit 1
fi

echo "PASS: tutti i file Go sono formattati"


# ------------------------------------------------------------
# BASH SYNTAX
# ------------------------------------------------------------

stage "BASH SYNTAX CHECK"

while IFS= read -r -d '' file; do

    echo "[bash -n] ${file#${ROOT}/}"

    bash -n "$file"

done < <(
    find "${ROOT}/scripts" \
        -type f \
        -name '*.sh' \
        -print0
)

echo "PASS: tutti gli script Bash"


# ------------------------------------------------------------
# PYTHON COMPILE
# ------------------------------------------------------------

stage "PYTHON COMPILE"

"$PYTHON_BIN" \
    -m compileall \
    -q \
    analysis/profiling

echo "PASS: Python compileall"


# ------------------------------------------------------------
# PYTHON UNIT TESTS
# ------------------------------------------------------------

stage "PYTHON UNIT TESTS"

"$PYTHON_BIN" \
    -m unittest discover \
    -s analysis/profiling \
    -p 'test_*.py'

echo "PASS: Python unit tests"


# ------------------------------------------------------------
# REAL SCHEMA 4 DATASET
# ------------------------------------------------------------

stage "SCHEMA 4 DATASET CHECK"

TMP_DIR="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/serverledge-precloud.XXXXXX"
)"

if [[ -n "${SCHEMA4_INPUT:-}" ]]; then

    if [[ ! -f "$SCHEMA4_INPUT" ]]; then
        echo "FAIL: dataset schema 4 indicato ma non trovato:"
        echo "$SCHEMA4_INPUT"
        exit 1
    fi

    echo "Dataset reale indicato: $SCHEMA4_INPUT"

else

    SCHEMA4_INPUT="${TMP_DIR}/synthetic-schema4.jsonl"

    LATEST_REAL="$(
        ls -1dt "${ROOT}"/logs/profiling_aggregation_local_*/profiling-samples.jsonl \
            2>/dev/null \
        | head -1 \
        || true
    )"

    if [[ -n "$LATEST_REAL" && -f "$LATEST_REAL" ]]; then

        SCHEMA4_INPUT="$LATEST_REAL"

        echo "Dataset reale piu' recente: $SCHEMA4_INPUT"

    else

        echo "Nessun dataset reale disponibile: uso una fixture sintetica."

        SCHEMA4_INPUT="$SCHEMA4_INPUT" \
        "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["SCHEMA4_INPUT"]

sample = {
    "schema_version": 4,
    "timestamp_ms": 1_700_000_000_000,
    "request_id": "precloud-synthetic-0",
    "function_name": "precloud-synthetic",
    "machine_tag": "precloud-x86",
    "node_name": "precloud-node",
    "container_id": "precloud-container",
    "function_configuration": {
        "configured_cpus": 1.0,
        "configured_memory_mb": 128,
    },
    "warm_start": True,
    "execution_succeeded": True,
    "timing": {
        "duration_ms": 100.0,
        "response_time_ms": 110.0,
        "init_time_ms": 0.0,
        "queueing_time_ms": 0.0,
        "offload_latency_ms": 0.0,
        "invocation_wait_ms": 0.0,
        "execution_wall_time_ms": 105.0,
    },
    "eligibility": {
        "resource_clustering": True,
        "cold_start_analysis": False,
        "performance_analysis": True,
        "exclusion_reasons": [],
    },
}

with open(path, "w", encoding="utf-8") as handle:
    for index in range(10):
        record = json.loads(json.dumps(sample))
        record["request_id"] = f"precloud-synthetic-{index}"
        record["timestamp_ms"] += index
        record["timing"]["duration_ms"] = 100.0 + index
        handle.write(json.dumps(record) + "\n")
PY

    fi
fi

SCHEMA4_OUTPUT="${TMP_DIR}/performance-schema4.csv"

"$PYTHON_BIN" \
    -m analysis.profiling.preference \
    profiles \
    --input "$SCHEMA4_INPUT" \
    --run-id precloud-schema4-check \
    --samples 10 \
    --output "$SCHEMA4_OUTPUT"

if [[ ! -s "$SCHEMA4_OUTPUT" ]]; then
    echo "FAIL: CSV schema 4 non prodotto."
    exit 1
fi

echo "PASS: preference.py legge InvocationSample schema 4"


# ------------------------------------------------------------
# GO MODULES
# ------------------------------------------------------------

stage "GO MODULE VERIFY"

go mod verify

echo "PASS: go mod verify"


# ------------------------------------------------------------
# GO BUILD
# ------------------------------------------------------------

stage "GO BUILD"

go build ./...

echo "PASS: go build ./..."


# ------------------------------------------------------------
# GO VET
#
# Volutamente limitato ai package interessati dalla tesi.
# Il repository upstream contiene warning in codice
# infrastrutturale non modificato.
# ------------------------------------------------------------

stage "GO VET - THESIS RELEVANT PACKAGES"

go vet \
    ./internal/mab \
    ./internal/profiling \
    ./internal/lb \
    ./internal/container \
    ./internal/function \
    ./internal/scheduling

echo "PASS: go vet sui package rilevanti"


# ------------------------------------------------------------
# MAB + LB
# ------------------------------------------------------------

stage "MAB AND LOAD BALANCER TESTS"

go test \
    -vet=off \
    ./internal/mab \
    ./internal/lb \
    -count=1

echo "PASS: MAB + LB"


# ------------------------------------------------------------
# RACE DETECTOR
# ------------------------------------------------------------

stage "MAB AND LOAD BALANCER RACE DETECTOR"

go test \
    -vet=off \
    -race \
    ./internal/mab \
    ./internal/lb \
    -count=1

echo "PASS: race detector MAB + LB"


# ------------------------------------------------------------
# PROFILING / CONTAINER / FUNCTION / SCHEDULING
# ------------------------------------------------------------

stage "PROFILING / CONTAINER / FUNCTION / SCHEDULING"

go test \
    -vet=off \
    ./internal/profiling \
    ./internal/container \
    ./internal/function \
    ./internal/scheduling \
    -count=1

echo "PASS: profiling/container/function/scheduling"


# ------------------------------------------------------------
# PREPARAZIONE FULL SUITE
#
# internal/test usa un etcd embedded.
# Il container Etcd-server usato dagli smoke locali occupa
# proprio 2379/2380 e deve essere spento prima.
# ------------------------------------------------------------

stage "PREPARE EMBEDDED ETCD TESTS"

cleanup_repository_etcd

sleep 1

assert_test_ports_free

echo "PASS: ambiente pronto per etcd embedded"


# ------------------------------------------------------------
# FULL GO TEST SUITE
#
# -p 1:
#   esecuzione seriale dei package, più stabile per i test
#   di integrazione legacy.
#
# -vet=off:
#   il vet rilevante è già stato eseguito esplicitamente
#   sui package della tesi.
#
# -count=1:
#   nessun risultato dalla Go test cache.
# ------------------------------------------------------------

stage "FULL GO TEST SUITE"

go test \
    -vet=off \
    -p 1 \
    ./... \
    -count=1 \
    -timeout=30m

echo "PASS: full Go test suite"


# ------------------------------------------------------------
# TRANSFER BOOTSTRAP
# ------------------------------------------------------------

stage "TRANSFER BOOTSTRAP VALIDATION"

scripts/test-transfer-bootstrap.sh

echo "PASS: transfer bootstrap"


# ------------------------------------------------------------
# SINGLE ARCH TRANSFER
# ------------------------------------------------------------

stage "SINGLE-ARCH TRANSFER BOOTSTRAP VALIDATION"

scripts/test-transfer-bootstrap-single-arch.sh

echo "PASS: single-architecture transfer bootstrap"


# ------------------------------------------------------------
# COMPLETATO
# ------------------------------------------------------------

stage "STATIC / UNIT REGRESSION GATE COMPLETE"

echo "Tutti i controlli previsti sono passati."
echo
echo "STATIC / UNIT REGRESSION GATE: PASS"