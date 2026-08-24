#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Serverledge - Pre-cloud local runtime smoke gate
#
# Valida end-to-end:
#
#   1. profiling/export
#   2. profiling aggregation schema 4
#   3. MAB baseline UCB1
#   4. MAB baseline LinUCB
#   5. transfer bootstrap target x86
#   6. transfer bootstrap target arm64
#
# Questa è una validazione FUNZIONALE locale.
# I nodi x86/arm64 del transfer girano sullo stesso host fisico.
# Non valida quindi prestazioni reali ARM-vs-x86.
#
# Kepler NON è incluso qui: viene validato separatamente
# con l'exporter ufficiale attivo.
# ============================================================


# ------------------------------------------------------------
# ROOT
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

TEST_LOG="${TEST_LOG:-${LOG_DIR}/precloud_runtime_smoke_${STAMP}.txt}"

exec > >(
    tee -a "$TEST_LOG"
) 2>&1


# ------------------------------------------------------------
# STATO
# ------------------------------------------------------------

CURRENT_STAGE="initialization"

cleanup() {
    local status=$?

    set +e

    echo
    echo "============================================================"
    echo "FINAL CLEANUP"
    echo "============================================================"

    if command -v docker >/dev/null 2>&1; then
        docker rm \
            -f \
            Etcd-server \
            >/dev/null \
            2>&1 \
            || true
    fi

    if [[ "$status" -eq 0 ]]; then
        echo
        echo "PRECLOUD RUNTIME SMOKE GATE: PASS"
    else
        echo
        echo "PRECLOUD RUNTIME SMOKE GATE: FAIL"
        echo "Failed stage: ${CURRENT_STAGE}"
        echo "Exit status: ${status}"
    fi

    echo
    echo "Combined log:"
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


fail() {
    echo "FAIL: $*" >&2
    exit 1
}


require_command() {
    local cmd="$1"

    command -v "$cmd" >/dev/null 2>&1 \
        || fail "comando richiesto non trovato: $cmd"
}


# ------------------------------------------------------------
# RESET ETCD TRA UNO SMOKE E L'ALTRO
# ------------------------------------------------------------

reset_etcd() {

    echo
    echo "[cleanup] Reset eventuale Etcd-server..."

    docker rm \
        -f \
        Etcd-server \
        >/dev/null \
        2>&1 \
        || true

    rm -rf "${ROOT}/default.etcd"

    sleep 1
}


# ------------------------------------------------------------
# CONTROLLO PORTE
# ------------------------------------------------------------

port_is_busy() {
    local port="$1"

    ss -H -ltn "sport = :${port}" 2>/dev/null \
        | grep -q .
}


assert_ports_free() {

    local busy=0

    # Porte usate dagli smoke locali Serverledge.
    for port in \
        1323 \
        1333 \
        1334 \
        2379 \
        2380 \
        8080; do

        if port_is_busy "$port"; then

            echo "FAIL: porta ${port} occupata."
            busy=1

        else

            echo "PASS: porta ${port} libera."

        fi

    done

    if [[ "$busy" -ne 0 ]]; then

        echo
        echo "Listener rilevanti:"

        ss -ltnp 2>/dev/null \
            | grep -E ':(1323|1333|1334|2379|2380|8080)[[:space:]]' \
            || true

        echo
        echo "Una o più porte richieste dagli smoke sono occupate."
        echo "Lo script non termina automaticamente processi sconosciuti."

        exit 1
    fi
}


prepare_stage() {

    reset_etcd

    assert_ports_free
}


# ------------------------------------------------------------
# ROOT CHECK
# ------------------------------------------------------------

stage "PROJECT ROOT CHECK"

[[ -f "${ROOT}/go.mod" ]] \
    || fail "go.mod non trovato"

[[ -d "${ROOT}/scripts" ]] \
    || fail "directory scripts non trovata"

echo "PASS: root Serverledge"
echo "$ROOT"


# ------------------------------------------------------------
# TOOLCHAIN
# ------------------------------------------------------------

stage "TOOLCHAIN CHECK"

require_command docker
require_command curl
require_command make
require_command ss

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-analysis/bin/python}"

[[ -x "$PYTHON_BIN" ]] \
    || fail "ambiente Python analysis non trovato: $PYTHON_BIN"

echo "Python analysis:"
echo "$PYTHON_BIN"

"$PYTHON_BIN" --version

docker info >/dev/null 2>&1 \
    || fail "Docker daemon non disponibile"

echo "PASS: toolchain runtime disponibile"


# ------------------------------------------------------------
# SCRIPT CHECK
# ------------------------------------------------------------

stage "SMOKE SCRIPT CHECK"

REQUIRED_SCRIPTS=(
    "scripts/validate.sh"
    "scripts/validate-profiling-aggregation.sh"
    "scripts/mab_baseline_local_smoke.sh"
    "scripts/transfer_local_smoke.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do

    [[ -f "$script" ]] \
        || fail "script mancante: $script"

    bash -n "$script"

    echo "PASS: sintassi $script"

done


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

stage "RUNTIME ENVIRONMENT"

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
echo "Docker:"
docker --version

echo
echo "Host architecture:"
uname -m


# ============================================================
# 1. PROFILING EXPORT
# ============================================================

stage "1/6 - PROFILING EXPORT RUNTIME"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
    scripts/validate.sh

echo
echo "PASS: profiling export runtime"


# ============================================================
# 2. PROFILING AGGREGATION
# ============================================================

stage "2/6 - PROFILING AGGREGATION SCHEMA 4"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
WARM_SAMPLES=10 \
    scripts/validate-profiling-aggregation.sh

echo
echo "PASS: profiling aggregation runtime"


# ============================================================
# 3. BASELINE UCB1
# ============================================================

stage "3/6 - BASELINE UCB1"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
POLICY=UCB1 \
WARM_REQUESTS=4 \
    scripts/mab_baseline_local_smoke.sh

echo
echo "PASS: baseline UCB1 runtime"


# ============================================================
# 4. BASELINE LINUCB
# ============================================================

stage "4/6 - BASELINE LINUCB"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
POLICY=LinUCB \
WARM_REQUESTS=4 \
    scripts/mab_baseline_local_smoke.sh

echo
echo "PASS: baseline LinUCB runtime"


# ============================================================
# 5. TRANSFER - TARGET BOOTSTRAP X86
# ============================================================

stage "5/6 - TRANSFER TARGET BOOTSTRAP X86"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
TARGET_BOOTSTRAP_TAG=x86 \
SAMPLES_PER_ARCH=10 \
    scripts/transfer_local_smoke.sh

echo
echo "PASS: transfer target bootstrap x86"


# ============================================================
# 6. TRANSFER - TARGET BOOTSTRAP ARM64
# ============================================================

stage "6/6 - TRANSFER TARGET BOOTSTRAP ARM64"

prepare_stage

PYTHON_BIN="$PYTHON_BIN" \
TARGET_BOOTSTRAP_TAG=arm64 \
SAMPLES_PER_ARCH=10 \
    scripts/transfer_local_smoke.sh

echo
echo "PASS: transfer target bootstrap arm64"


# ============================================================
# COMPLETATO
# ============================================================

stage "LOCAL RUNTIME SMOKE COMPLETE"

echo "Tutti gli smoke runtime locali previsti sono passati."

echo
echo "Validated:"
echo "  profiling export"
echo "  profiling aggregation schema 4"
echo "  MAB baseline UCB1"
echo "  MAB baseline LinUCB"
echo "  transfer bootstrap x86"
echo "  transfer bootstrap arm64"

echo
echo "IMPORTANT:"
echo "  x86/arm64 transfer nodes are logical nodes on the same host."
echo "  This validates software integration, not hardware performance."

echo
echo "PRECLOUD RUNTIME SMOKE GATE: PASS"

