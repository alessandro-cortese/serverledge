#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
  pwd
)"

cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_ID="container-lifecycle-disabled-${STAMP}"
LOG_DIR="logs/${RUN_ID}"

NODE_CONF="${LOG_DIR}/node.yaml"
NODE_LOG="${LOG_DIR}/node.log"
ETCD_LOG="${LOG_DIR}/etcd.log"

API_PORT=1324
UDP_PORT=9877

FUNCTION_NAME="lifecycle_disabled_${STAMP}"

NODE_PID=""
ETCD_STARTED=0

mkdir -p "$LOG_DIR"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

cleanup() {
  status=$?

  set +e

  echo
  echo "[cleanup] Arresto processi locali..."

  if [[ -n "$NODE_PID" ]] &&
     kill -0 "$NODE_PID" 2>/dev/null; then

    kill -INT "$NODE_PID" 2>/dev/null || true

    for _ in $(seq 1 20); do
      kill -0 "$NODE_PID" 2>/dev/null || break
      sleep 0.25
    done

    if kill -0 "$NODE_PID" 2>/dev/null; then
      kill -KILL "$NODE_PID" 2>/dev/null || true
    fi
  fi

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1 || true
  fi

  echo "[cleanup] Artefatti: $LOG_DIR"

  exit "$status"
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "CONTAINER LIFECYCLE DISABLED — REGRESSION SMOKE"
echo "============================================================"
echo "run:                 $RUN_ID"
echo "function:            $FUNCTION_NAME"
echo "lifecycle logging:   DISABLED"
echo "max concurrency:     1"
echo "container memory:    128 MiB"
echo "expiration:          4 s"
echo "janitor interval:    1 s"
echo

docker info >/dev/null 2>&1 ||
  fail "Docker non disponibile"

echo "[setup] Build..."
make >/dev/null

docker rm -f Etcd-server >/dev/null 2>&1 || true

cat > "$NODE_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: LIFECYCLE_DISABLED_TEST
registry.node.id: lifecycle-disabled-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: x86

container.pool.memory: 512
container.pool.cpus: 4

container.expiration: 4
janitor.interval: 1

factory.images.refresh: false

profiling.enabled: false
profiling.export.enabled: false

scheduler.queue.capacity: 0
YAML

echo "[setup] Avvio etcd..."

bash scripts/start-etcd.sh \
  >"$ETCD_LOG" \
  2>&1

ETCD_STARTED=1

for _ in $(seq 1 30); do
  if docker exec Etcd-server \
    etcdctl endpoint health \
    >/dev/null 2>&1; then
    break
  fi

  sleep 0.5
done

docker exec Etcd-server \
  etcdctl endpoint health \
  >/dev/null 2>&1 ||
  fail "etcd non disponibile"

echo "[setup] Avvio Serverledge SENZA lifecycle logging..."

env -u SERVERLEDGE_CONTAINER_LIFECYCLE_LOG \
  bin/serverledge "$NODE_CONF" \
  >"$NODE_LOG" \
  2>&1 &

NODE_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "$NODE_PID" 2>/dev/null; then
    tail -100 "$NODE_LOG" || true
    fail "Serverledge terminato durante l'avvio"
  fi

  if curl \
    -sS \
    --max-time 1 \
    -o /dev/null \
    "http://127.0.0.1:${API_PORT}/status" \
    2>/dev/null; then
    break
  fi

  sleep 0.5
done

curl \
  -sS \
  --max-time 2 \
  -o /dev/null \
  "http://127.0.0.1:${API_PORT}/status" ||
  fail "API Serverledge non disponibile"

echo "[setup] Creo funzione..."

bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory 128 \
  --src examples/sleeper.py \
  --runtime python314 \
  --handler "sleeper.handler" \
  --max_concurrency 1 \
  --cpu 1 \
  --input "n:Int" \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"${LOG_DIR}/create.json"

echo "[test] Invocazione cold..."

bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:1" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"${LOG_DIR}/invoke.json" \
  2>&1

grep -q '"Success": true' \
  "${LOG_DIR}/invoke.json" ||
  fail "invocazione non riuscita"

CONTAINER_ID="$(
  grep -m1 \
    'Executing on container:' \
    "$NODE_LOG" |
  sed -E \
    's/.*Executing on container: ([^[:space:]]+).*/\1/'
)"

[[ -n "$CONTAINER_ID" ]] ||
  fail "container ID non trovato nel log legacy"

echo "[test] container creato normalmente: $CONTAINER_ID"
echo "[test] Attendo expiration + janitor..."

sleep 8

LIFECYCLE_COUNT="$(
  grep -c \
    '\[CONTAINER_LIFECYCLE\]' \
    "$NODE_LOG" ||
  true
)"

[[ "$LIFECYCLE_COUNT" -eq 0 ]] ||
  fail \
    "logging disabilitato ma trovati $LIFECYCLE_COUNT eventi lifecycle"

if docker inspect \
  "$CONTAINER_ID" \
  >/dev/null \
  2>&1; then

  fail \
    "container ancora presente dopo expiration: $CONTAINER_ID"
fi

echo
echo "============================================================"
echo "REGRESSION RESULTS"
echo "============================================================"
echo "invocation:                 PASS"
echo "legacy container creation:  PASS"
echo "lifecycle events emitted:   0"
echo "container expired/destroyed: PASS"
echo
echo "CONTAINER LIFECYCLE DISABLED REGRESSION: PASS"
echo "============================================================"