#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" &&
  pwd
)"

ROOT_DIR="$(
  cd "$SCRIPT_DIR/.." &&
  pwd
)"

cd "$ROOT_DIR"

# ============================================================
# Configurazione
# ============================================================

X86_PORT="${X86_PORT:-1333}"
ARM_PORT="${ARM_PORT:-1334}"
LB_PORT="${LB_PORT:-8080}"

X86_UDP_PORT="${X86_UDP_PORT:-9899}"
ARM_UDP_PORT="${ARM_UDP_PORT:-9900}"

POLICY="${POLICY:-UCB1}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"

WARM_REQUESTS="${WARM_REQUESTS:-4}"

STAMP="$(
  date +%Y%m%d_%H%M%S
)"

RUN_ID="mab-baseline-local-smoke-${STAMP}"

LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"

mkdir -p \
  "$LOG_DIR"

X86_CONF="$LOG_DIR/node-x86.yaml"
ARM_CONF="$LOG_DIR/node-arm64.yaml"
LB_CONF="$LOG_DIR/lb.yaml"

X86_LOG="$LOG_DIR/node-x86.log"
ARM_LOG="$LOG_DIR/node-arm64.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"

FUNCTION_NAME="baseline_smoke_${STAMP}"

PIDS=()
ETCD_STARTED=0

# ============================================================
# Utility
# ============================================================

fail() {
  echo \
    "[FAIL] $*" \
    >&2

  exit 1
}

cleanup() {
  status=$?

  set +e

  echo
  echo \
    "[cleanup] Arresto processi locali..."

  for pid in "${PIDS[@]:-}"; do

    if [[ -n "$pid" ]] &&
      kill -0 "$pid" 2>/dev/null; then

      kill \
        -INT \
        "$pid" \
        2>/dev/null ||
        true
    fi

  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do

    if [[ -n "$pid" ]] &&
      kill -0 "$pid" 2>/dev/null; then

      kill \
        -KILL \
        "$pid" \
        2>/dev/null ||
        true
    fi

  done

  if [[ "$ETCD_STARTED" -eq 1 ]]; then

    docker rm \
      -f \
      Etcd-server \
      >/dev/null \
      2>&1 ||
      true

  fi

  echo \
    "[cleanup] Artefatti conservati in: $LOG_DIR"

  exit "$status"
}

trap \
  cleanup \
  EXIT \
  INT \
  TERM

wait_for_api() {
  local label="$1"
  local pid="$2"
  local port="$3"
  local log_file="$4"

  for _ in $(
    seq \
      1 \
      60
  ); do

    if ! kill \
      -0 \
      "$pid" \
      2>/dev/null; then

      echo
      echo \
        "===== $label LOG ====="

      tail \
        -100 \
        "$log_file" \
        || true

      fail \
        "$label terminato durante l'avvio"
    fi

    if curl \
      -sS \
      --max-time 1 \
      -o /dev/null \
      "http://127.0.0.1:${port}/status" \
      2>/dev/null; then

      echo \
        "[setup] $label pronto"

      return
    fi

    sleep 0.5
  done

  fail \
    "$label non disponibile sulla porta $port"
}

# ============================================================
# Pre-check
# ============================================================

for cmd in \
  docker \
  curl \
  make \
  grep \
  python3; do

  command \
    -v \
    "$cmd" \
    >/dev/null \
    2>&1 ||
    fail \
      "Comando richiesto non trovato: $cmd"

done

[[ -f Makefile ]] ||
  fail \
    "Eseguire lo script dalla root di Serverledge"

[[ "$WARM_REQUESTS" =~ ^[0-9]+$ ]] ||
  fail \
    "WARM_REQUESTS deve essere numerico"

if (( WARM_REQUESTS < 1 )); then

  fail \
    "WARM_REQUESTS deve essere almeno 1"

fi

for port in \
  "$X86_PORT" \
  "$ARM_PORT" \
  "$LB_PORT"; do

  if curl \
    -sS \
    --max-time 1 \
    -o /dev/null \
    "http://127.0.0.1:${port}/status" \
    2>/dev/null; then

    fail \
      "La porta API $port è già occupata da Serverledge"
  fi

done

# ============================================================
# Piano
# ============================================================

echo \
  "============================================================"

echo \
  "MAB BASELINE — LOCAL END-TO-END SMOKE TEST"

echo \
  "============================================================"

echo \
  "run_id:             $RUN_ID"

echo \
  "function:           $FUNCTION_NAME"

echo \
  "policy:             $POLICY"

echo \
  "reward:             latency"

echo \
  "transfer learning:  DISABLED"

echo \
  "profiling:          DISABLED"

echo \
  "x86 logical node:   127.0.0.1:$X86_PORT tag=x86"

echo \
  "ARM logical node:   127.0.0.1:$ARM_PORT tag=arm64"

echo \
  "LB:                 127.0.0.1:$LB_PORT"

echo

echo \
  "Workflow:"

echo \
  "  1. create a completely new function"

echo \
  "  2. first application invocation goes directly through MAB"

echo \
  "  3. no profiling bootstrap is executed"

echo \
  "  4. no donor is selected"

echo \
  "  5. no transfer prior is applied"

echo \
  "  6. subsequent requests continue through the same MAB agent"

echo

echo \
  "ATTENZIONE: x86 e arm64 sono due nodi LOGICI."

echo \
  "Entrambi vengono eseguiti sullo stesso host fisico."

echo \
  "Il test valida la baseline funzionale, non prestazioni ARM-vs-x86."

echo

# ============================================================
# Build
# ============================================================

echo \
  "[setup] Build..."

make

docker info \
  >/dev/null \
  2>&1 ||
  fail \
    "Docker non disponibile"

docker rm \
  -f \
  Etcd-server \
  >/dev/null \
  2>&1 ||
  true

# ============================================================
# Config nodi
# ============================================================

cat > "$X86_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${X86_PORT}

registry.area: BASELINE_LOCAL_SMOKE
registry.node.id: baseline-x86-${STAMP}
registry.udp.port: ${X86_UDP_PORT}

node.machine_tag: x86

container.pool.memory: 512
container.expiration: 600
factory.images.refresh: false

profiling.enabled: false
profiling.export.enabled: false
profiling.kepler.enabled: false
YAML

cat > "$ARM_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${ARM_PORT}

registry.area: BASELINE_LOCAL_SMOKE
registry.node.id: baseline-arm64-${STAMP}
registry.udp.port: ${ARM_UDP_PORT}

node.machine_tag: arm64

container.pool.memory: 512
container.expiration: 600
factory.images.refresh: false

profiling.enabled: false
profiling.export.enabled: false
profiling.kepler.enabled: false
YAML

# ============================================================
# Config LB
# ============================================================

cat > "$LB_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${LB_PORT}

registry.area: BASELINE_LOCAL_SMOKE

lb.arch_awareness: true
lb.mode: MAB
lb.replicas: 128
lb.refresh_interval: 1

mab.policy: ${POLICY}
mab.ucb1.c: ${MAB_UCB1_C}

mab.cold_start.mode: skip

mab.reward.mode: latency

# Baseline esplicita: nessuna inizializzazione tramite transfer.
mab.transfer.control.enabled: false
YAML

# ============================================================
# etcd
# ============================================================

echo \
  "[setup] Avvio etcd..."

bash \
  scripts/start-etcd.sh \
  > "$ETCD_LOG" \
  2>&1

ETCD_STARTED=1

for _ in $(
  seq \
    1 \
    30
); do

  if docker exec \
    Etcd-server \
    etcdctl endpoint health \
    >/dev/null \
    2>&1; then

    break
  fi

  sleep 0.5
done

docker exec \
  Etcd-server \
  etcdctl endpoint health \
  >/dev/null \
  2>&1 ||
  fail \
    "etcd non disponibile"

echo \
  "[setup] etcd pronto"

# ============================================================
# Nodi
# ============================================================

echo \
  "[setup] Avvio nodo logico x86..."

bin/serverledge \
  "$X86_CONF" \
  > "$X86_LOG" \
  2>&1 &

X86_PID=$!

PIDS+=(
  "$X86_PID"
)

wait_for_api \
  "nodo x86" \
  "$X86_PID" \
  "$X86_PORT" \
  "$X86_LOG"

echo \
  "[setup] Avvio nodo logico arm64..."

bin/serverledge \
  "$ARM_CONF" \
  > "$ARM_LOG" \
  2>&1 &

ARM_PID=$!

PIDS+=(
  "$ARM_PID"
)

wait_for_api \
  "nodo arm64" \
  "$ARM_PID" \
  "$ARM_PORT" \
  "$ARM_LOG"

# ============================================================
# Load Balancer
# ============================================================

echo \
  "[setup] Avvio load balancer MAB..."

bin/lb \
  "$LB_CONF" \
  > "$LB_LOG" \
  2>&1 &

LB_PID=$!

PIDS+=(
  "$LB_PID"
)

wait_for_api \
  "load balancer" \
  "$LB_PID" \
  "$LB_PORT" \
  "$LB_LOG"

# Aspettiamo che entrambi i machine tag siano scoperti.
for _ in $(
  seq \
    1 \
    40
); do

  x86_found=0
  arm_found=0

  grep \
    -q \
    'event=new_machine_tag_discovered tag=x86 ' \
    "$LB_LOG" \
    && x86_found=1

  grep \
    -q \
    'event=new_machine_tag_discovered tag=arm64 ' \
    "$LB_LOG" \
    && arm_found=1

  if (( x86_found == 1 &&
        arm_found == 1 )); then

    break
  fi

  sleep 0.25
done

grep \
  -q \
  'event=new_machine_tag_discovered tag=x86 ' \
  "$LB_LOG" ||
  fail \
    "LB non ha scoperto il machine tag x86"

grep \
  -q \
  'event=new_machine_tag_discovered tag=arm64 ' \
  "$LB_LOG" ||
  fail \
    "LB non ha scoperto il machine tag arm64"

echo \
  "[setup] entrambi gli arm MAB disponibili: x86, arm64"

# ============================================================
# Creazione funzione NUOVA
# ============================================================

echo \
  "[setup] Creazione nuova funzione $FUNCTION_NAME..."

bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory 128 \
  --cpu 1 \
  --max_concurrency 1 \
  --src examples/hello.py \
  --runtime python314 \
  --handler "hello.handler" \
  -H 127.0.0.1 \
  -P "$X86_PORT" \
  > "$LOG_DIR/create.json"

# ============================================================
# Prima della prima invocation:
# il MAB target non deve ancora avere effettuato selezioni.
# ============================================================

PRE_SELECTIONS="$(
  grep \
    -c \
    "event=before_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if [[ "$PRE_SELECTIONS" -ne 0 ]]; then

  fail \
    "Il target presenta selezioni MAB prima della prima invocation: $PRE_SELECTIONS"
fi

PRE_TRANSFER_EVENTS="$(
  grep \
    -Ec \
    "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if [[ "$PRE_TRANSFER_EVENTS" -ne 0 ]]; then

  fail \
    "Il target presenta transfer event prima della prima invocation"
fi

echo \
  "[baseline-check] prima della request: selections=0 transfer_events=0"

# ============================================================
# PRIMA INVOCATION: deve passare SUBITO dal MAB
# ============================================================

echo \
  "[baseline] PRIMA invocation della nuova funzione attraverso il MAB..."

bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "name:first-request" \
  --return_output \
  -H 127.0.0.1 \
  -P "$LB_PORT" \
  > "$LOG_DIR/invoke-first.json"

sleep 1

FIRST_BEFORE_COUNT="$(
  grep \
    -c \
    "event=before_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

FIRST_AFTER_COUNT="$(
  grep \
    -c \
    "event=after_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if (( FIRST_BEFORE_COUNT < 1 )); then

  fail \
    "La prima invocation non è entrata nel MAB"
fi

if (( FIRST_AFTER_COUNT < 1 )); then

  fail \
    "La prima invocation non ha prodotto una selezione MAB"
fi

FIRST_SELECTION="$(
  grep \
    "event=after_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    | head \
      -1
)"

echo \
  "[baseline-check] first_request_selected_by_mab=true"

echo \
  "[baseline-check] $FIRST_SELECTION"

# ============================================================
# Assenza transfer
# ============================================================

TRANSFER_EVENTS="$(
  grep \
    -Ec \
    "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if [[ "$TRANSFER_EVENTS" -ne 0 ]]; then

  echo
  grep \
    -E \
    "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
    "$LB_LOG" \
    || true

  fail \
    "La baseline ha applicato transfer learning"
fi

echo \
  "[baseline-check] transfer_events=0"

# ============================================================
# Prewarm SOLO dopo la prima decisione.
#
# Serve esclusivamente per rendere le successive richieste warm
# e verificare il normale aggiornamento del MAB.
# Non è profiling e non è transfer learning.
# ============================================================

echo \
  "[baseline] Prewarm post-prima-request sui due nodi..."

bin/serverledge-cli prewarm \
  -f "$FUNCTION_NAME" \
  -c 1 \
  -H 127.0.0.1 \
  -P "$X86_PORT" \
  > "$LOG_DIR/prewarm-x86.json"

bin/serverledge-cli prewarm \
  -f "$FUNCTION_NAME" \
  -c 1 \
  -H 127.0.0.1 \
  -P "$ARM_PORT" \
  > "$LOG_DIR/prewarm-arm64.json"

sleep 1

# ============================================================
# Richieste successive: stesso MAB, nessun transfer
# ============================================================

echo \
  "[baseline] $WARM_REQUESTS richieste successive attraverso il MAB..."

for i in $(
  seq \
    1 \
    "$WARM_REQUESTS"
); do

  bin/serverledge-cli invoke \
    -f "$FUNCTION_NAME" \
    -p "name:warm-${i}" \
    --return_output \
    -H 127.0.0.1 \
    -P "$LB_PORT" \
    > "$LOG_DIR/invoke-warm-${i}.json"

  sleep 0.2
done

sleep 1

# ============================================================
# Verifiche finali
# ============================================================

EXPECTED_SELECTIONS=$((1 + WARM_REQUESTS))

BEFORE_COUNT="$(
  grep \
    -c \
    "event=before_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

AFTER_COUNT="$(
  grep \
    -c \
    "event=after_select function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if (( BEFORE_COUNT < EXPECTED_SELECTIONS )); then

  fail \
    "Numero before_select insufficiente: $BEFORE_COUNT < $EXPECTED_SELECTIONS"
fi

if (( AFTER_COUNT < EXPECTED_SELECTIONS )); then

  fail \
    "Numero after_select insufficiente: $AFTER_COUNT < $EXPECTED_SELECTIONS"
fi

TRANSFER_EVENTS="$(
  grep \
    -Ec \
    "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if [[ "$TRANSFER_EVENTS" -ne 0 ]]; then

  fail \
    "Sono comparsi transfer event nella baseline: $TRANSFER_EVENTS"
fi

UPDATE_COUNT="$(
  grep \
    -c \
    "event=update_reward .*function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    2>/dev/null ||
    true
)"

if (( UPDATE_COUNT < 1 )); then

  echo
  echo \
    "===== MAB TARGET LOG ====="

  grep \
    "function=${FUNCTION_NAME} " \
    "$LB_LOG" \
    || true

  fail \
    "Nessun reward reale ha aggiornato il MAB"
fi

# Non deve essere mai stato chiamato il workflow di transfer.
if grep \
  -Eq \
  "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
  "$LB_LOG"; then

  fail \
    "Transfer rilevato per la funzione baseline"
fi

echo
echo \
  "===== FIRST MAB SELECTION ====="

grep \
  "event=after_select function=${FUNCTION_NAME} " \
  "$LB_LOG" \
  | head \
    -1

echo
echo \
  "===== ALL MAB SELECTIONS ====="

grep \
  -E \
  "event=(before_select|after_select).*function=${FUNCTION_NAME}" \
  "$LB_LOG" \
  || true

echo
echo \
  "===== REWARD UPDATES ====="

grep \
  "event=update_reward .*function=${FUNCTION_NAME} " \
  "$LB_LOG" \
  || true

echo
echo \
  "===== TRANSFER EVENTS ====="

grep \
  -E \
  "event=(runtime_transfer|selection_runtime_transfer).*target_function=${FUNCTION_NAME}" \
  "$LB_LOG" \
  || echo \
    "none"

# ============================================================
# PASS
# ============================================================

echo
echo \
  "============================================================"

echo \
  "MAB BASELINE LOCAL SMOKE TEST: PASS"

echo \
  "============================================================"

echo \
  "✓ nuova funzione creata senza donor"

echo \
  "✓ zero selezioni MAB prima della prima richiesta"

echo \
  "✓ prima invocation gestita immediatamente dal MAB"

echo \
  "✓ nessuna fase di profiling bootstrap"

echo \
  "✓ nessun donor selezionato"

echo \
  "✓ nessun transfer prior applicato"

echo \
  "✓ richieste successive gestite dallo stesso MAB"

echo \
  "✓ almeno un feedback warm ha aggiornato il reward"

echo

echo \
  "NON VALIDATO: prestazioni ARM-vs-x86."

echo \
  "Entrambi i nodi logici girano sullo stesso host fisico."

echo

echo \
  "Artefatti: $LOG_DIR"