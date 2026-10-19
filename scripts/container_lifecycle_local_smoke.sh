#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
  pwd
)"

cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_ID="container-lifecycle-local-${STAMP}"
LOG_DIR="logs/${RUN_ID}"

NODE_CONF="${LOG_DIR}/node.yaml"
NODE_LOG="${LOG_DIR}/node.log"
ETCD_LOG="${LOG_DIR}/etcd.log"

API_PORT=1324
UDP_PORT=9877

FUNCTION_NAME="lifecycle_smoke_${STAMP}"

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

    kill -0 "$NODE_PID" 2>/dev/null &&
      kill -KILL "$NODE_PID" 2>/dev/null || true
  fi

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1 || true
  fi

  echo "[cleanup] Artefatti: $LOG_DIR"

  exit "$status"
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "CONTAINER LIFECYCLE LOCAL SMOKE"
echo "============================================================"
echo "run:              $RUN_ID"
echo "function:         $FUNCTION_NAME"
echo "expiration:       4 s"
echo "janitor interval: 1 s"
echo "max concurrency:  1"
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

registry.area: LIFECYCLE_LOCAL_TEST
registry.node.id: lifecycle-node-${STAMP}
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

echo "[setup] Avvio Serverledge con lifecycle logging..."

SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1 \
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

echo "[test] Attendo expiration + janitor..."

sleep 7

echo
echo "============================================================"
echo "LIFECYCLE EVENTS"
echo "============================================================"

grep '\[CONTAINER_LIFECYCLE\]' \
  "$NODE_LOG" || true

CREATED_COUNT="$(
  grep -c \
    '\[CONTAINER_LIFECYCLE\] event=container_created' \
    "$NODE_LOG" ||
  true
)"

DESTROYED_COUNT="$(
  grep -c \
    '\[CONTAINER_LIFECYCLE\] event=container_destroyed' \
    "$NODE_LOG" ||
  true
)"

EXPIRED_COUNT="$(
  grep \
    '\[CONTAINER_LIFECYCLE\] event=container_destroyed' \
    "$NODE_LOG" |
  grep -c \
    'reason="expired"' ||
  true
)"

[[ "$CREATED_COUNT" -eq 1 ]] ||
  fail \
    "atteso 1 container_created, trovato $CREATED_COUNT"

[[ "$DESTROYED_COUNT" -eq 1 ]] ||
  fail \
    "atteso 1 container_destroyed, trovato $DESTROYED_COUNT"

[[ "$EXPIRED_COUNT" -eq 1 ]] ||
  fail \
    "il container non risulta distrutto per expiration"

python3 - "$NODE_LOG" <<'PY'
import re
import sys

path = sys.argv[1]

created = None
destroyed = None

pattern = re.compile(
    r'event=(container_created|container_destroyed).*?'
    r'ts=(\d+).*?'
    r'container_id="([^"]+)".*?'
    r'memory_mb=(-?\d+)'
)

with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        if "[CONTAINER_LIFECYCLE]" not in line:
            continue

        match = pattern.search(line)

        if not match:
            continue

        event, ts, cid, memory = match.groups()

        item = {
            "ts": int(ts),
            "container_id": cid,
            "memory_mb": int(memory),
        }

        if event == "container_created":
            created = item
        elif event == "container_destroyed":
            destroyed = item

if created is None:
    raise SystemExit("FAIL: container_created non parsabile")

if destroyed is None:
    raise SystemExit("FAIL: container_destroyed non parsabile")

if created["container_id"] != destroyed["container_id"]:
    raise SystemExit(
        "FAIL: container_id creation/destruction differente"
    )

if created["memory_mb"] != 128:
    raise SystemExit(
        f'FAIL: memoria creation inattesa: '
        f'{created["memory_mb"]}'
    )

if destroyed["memory_mb"] != 128:
    raise SystemExit(
        f'FAIL: memoria destruction inattesa: '
        f'{destroyed["memory_mb"]}'
    )

lifetime_ms = destroyed["ts"] - created["ts"]

if lifetime_ms <= 0:
    raise SystemExit("FAIL: lifetime non positivo")

memory_mib_s = (
    created["memory_mb"] *
    lifetime_ms /
    1000.0
)

memory_gib_s = memory_mib_s / 1024.0

print()
print("PAIRING: PASS")
print("container_id:", created["container_id"])
print("memory_mb:", created["memory_mb"])
print("lifetime_ms:", lifetime_ms)
print("memory_MiB_s:", f"{memory_mib_s:.6f}")
print("memory_GiB_s:", f"{memory_gib_s:.6f}")
PY

echo
echo "============================================================"
echo "CONTAINER LIFECYCLE LOCAL SMOKE: PASS"
echo "============================================================"