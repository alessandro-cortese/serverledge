#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
  pwd
)"

cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_ID="container-lifecycle-concurrency-${STAMP}"
LOG_DIR="logs/${RUN_ID}"

NODE_CONF="${LOG_DIR}/node.yaml"
NODE_LOG="${LOG_DIR}/node.log"
ETCD_LOG="${LOG_DIR}/etcd.log"

API_PORT=1324
UDP_PORT=9877

FUNCTION_NAME="lifecycle_concurrency_${STAMP}"

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
echo "CONTAINER LIFECYCLE CONCURRENCY LOCAL SMOKE"
echo "============================================================"
echo "run:                 $RUN_ID"
echo "function:            $FUNCTION_NAME"
echo "parallel requests:   3"
echo "max concurrency:     1"
echo "container memory:    128 MiB"
echo "pool memory:         512 MiB"
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

registry.area: LIFECYCLE_CONCURRENCY_TEST
registry.node.id: lifecycle-concurrency-${STAMP}
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

echo "[setup] Avvio Serverledge..."

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

echo "[test] Lancio 3 richieste contemporanee..."

REQUEST_PIDS=()

for i in 1 2 3; do
  bin/serverledge-cli invoke \
    -f "$FUNCTION_NAME" \
    -p "n:2" \
    --return_output \
    -H 127.0.0.1 \
    -P "$API_PORT" \
    >"${LOG_DIR}/invoke-${i}.json" \
    2>&1 &

  REQUEST_PIDS+=("$!")
done

for pid in "${REQUEST_PIDS[@]}"; do
  wait "$pid" ||
    fail "una richiesta concorrente è fallita"
done

echo "[test] Tutte le richieste completate"

echo "[test] Attendo expiration + janitor..."

sleep 8

echo
echo "============================================================"
echo "LIFECYCLE EVENTS"
echo "============================================================"

grep '\[CONTAINER_LIFECYCLE\]' \
  "$NODE_LOG" || true

python3 - "$NODE_LOG" <<'PY'
import re
import sys
from collections import defaultdict

path = sys.argv[1]

event_re = re.compile(
    r'event=(container_created|container_destroyed).*?'
    r'ts=(\d+).*?'
    r'container_id="([^"]+)".*?'
    r'memory_mb=(-?\d+)'
)

reason_re = re.compile(
    r'reason="([^"]+)"'
)

events = defaultdict(dict)

with open(
    path,
    encoding="utf-8",
    errors="replace",
) as f:
    for line in f:

        if "[CONTAINER_LIFECYCLE]" not in line:
            continue

        match = event_re.search(line)

        if not match:
            continue

        event, ts, cid, memory = match.groups()

        item = {
            "ts": int(ts),
            "memory_mb": int(memory),
            "line": line.rstrip(),
        }

        if event == "container_destroyed":
            reason_match = reason_re.search(line)

            item["reason"] = (
                reason_match.group(1)
                if reason_match
                else ""
            )

        events[cid][event] = item

created_ids = {
    cid
    for cid, values in events.items()
    if "container_created" in values
}

destroyed_ids = {
    cid
    for cid, values in events.items()
    if "container_destroyed" in values
}

if len(created_ids) != 3:
    raise SystemExit(
        f"FAIL: attesi 3 container creati, "
        f"trovati {len(created_ids)}"
    )

if destroyed_ids != created_ids:
    raise SystemExit(
        "FAIL: pairing create/destroy incompleto\n"
        f"created={sorted(created_ids)}\n"
        f"destroyed={sorted(destroyed_ids)}"
    )

total_mib_s = 0.0

print()
print("PER-CONTAINER RESULTS")

for cid in sorted(created_ids):

    created = events[cid]["container_created"]
    destroyed = events[cid]["container_destroyed"]

    if created["memory_mb"] != 128:
        raise SystemExit(
            f"FAIL: memoria creation inattesa "
            f"per {cid}: {created['memory_mb']}"
        )

    if destroyed["memory_mb"] != 128:
        raise SystemExit(
            f"FAIL: memoria destruction inattesa "
            f"per {cid}: {destroyed['memory_mb']}"
        )

    if destroyed.get("reason") != "expired":
        raise SystemExit(
            f"FAIL: reason inatteso per {cid}: "
            f"{destroyed.get('reason')}"
        )

    lifetime_ms = (
        destroyed["ts"] -
        created["ts"]
    )

    if lifetime_ms <= 0:
        raise SystemExit(
            f"FAIL: lifetime non positivo per {cid}"
        )

    memory_mib_s = (
        created["memory_mb"] *
        lifetime_ms /
        1000.0
    )

    total_mib_s += memory_mib_s

    print()
    print("container_id:", cid)
    print("memory_mb:", created["memory_mb"])
    print("lifetime_ms:", lifetime_ms)
    print(
        "memory_MiB_s:",
        f"{memory_mib_s:.6f}",
    )
    print(
        "memory_GiB_s:",
        f"{memory_mib_s / 1024.0:.6f}",
    )

print()
print("SUMMARY")
print("created_containers:", len(created_ids))
print("destroyed_containers:", len(destroyed_ids))
print("pairing:", "PASS")
print(
    "total_memory_MiB_s:",
    f"{total_mib_s:.6f}",
)
print(
    "total_memory_GiB_s:",
    f"{total_mib_s / 1024.0:.6f}",
)

print()
print("CONCURRENCY LIFECYCLE VALIDATION: PASS")
PY

echo
echo "============================================================"
echo "CONTAINER LIFECYCLE CONCURRENCY LOCAL SMOKE: PASS"
echo "============================================================"