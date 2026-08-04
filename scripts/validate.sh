#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

API_PORT="${API_PORT:-1333}"
UDP_PORT="${UDP_PORT:-9899}"
MEMORY_MB="${MEMORY_MB:-128}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/profiling_export_local_${STAMP}"
NODE_LOG="$LOG_DIR/node.log"
ETCD_LOG="$LOG_DIR/etcd.log"
CONF_FILE="$LOG_DIR/node-profiling-export-conf.yaml"
DATASET_FILE="$ROOT_DIR/$LOG_DIR/profiling-samples.jsonl"
FUNCTION_NAME="profiling_export_${STAMP}"
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
  echo "[cleanup] Arresto dei processi..."

  if [[ -n "$NODE_PID" ]] && kill -0 "$NODE_PID" 2>/dev/null; then
    kill -INT "$NODE_PID" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$NODE_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$NODE_PID" 2>/dev/null && kill -KILL "$NODE_PID" 2>/dev/null
    wait "$NODE_PID" 2>/dev/null
  fi

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1
  fi

  echo "[cleanup] Log conservati in: $LOG_DIR"
  exit "$status"
}

trap cleanup EXIT INT TERM

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 || fail "Comando richiesto non trovato: $cmd"
done

[[ -f Makefile ]] || fail "Eseguire lo script dalla root di Serverledge."

echo "[setup] Build..."
make

docker info >/dev/null 2>&1 || fail "Docker non è disponibile."

docker rm -f Etcd-server >/dev/null 2>&1 || true
rm -f "$DATASET_FILE"

cat > "$CONF_FILE" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: PROFILE_EXPORT_TEST
registry.node.id: profiling-export-node-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: profiling-test

container.pool.memory: 768
container.pool.cpus: 8
container.expiration: 600
factory.images.refresh: false

profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${DATASET_FILE}"

scheduler.queue.capacity: 0
YAML

echo "[setup] Avvio etcd..."
bash scripts/start-etcd.sh >"$ETCD_LOG" 2>&1
ETCD_STARTED=1

for _ in $(seq 1 30); do
  docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 && break
  sleep 0.5
done

docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 || fail "etcd non disponibile."

echo "[setup] Avvio Serverledge..."
bin/serverledge "$CONF_FILE" >"$NODE_LOG" 2>&1 &
NODE_PID=$!

for _ in $(seq 1 60); do
  kill -0 "$NODE_PID" 2>/dev/null || {
    tail -100 "$NODE_LOG" || true
    fail "Serverledge è terminato durante l'avvio."
  }

  if curl -sS --max-time 1 -o /dev/null "http://127.0.0.1:${API_PORT}/status" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

curl -sS --max-time 2 -o /dev/null "http://127.0.0.1:${API_PORT}/status" || fail "API non disponibile."

echo "[setup] Creazione funzione $FUNCTION_NAME..."
bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory "$MEMORY_MB" \
  --src examples/sleeper.py \
  --runtime python314 \
  --handler "sleeper.handler" \
  --max_concurrency 1 \
  --input "n:Int" \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  | tee "$LOG_DIR/create.json"

echo "[test] Invocazione cold riuscita..."
bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:${SLEEP_SECONDS}" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/cold.log" 2>&1

echo "[test] Invocazione warm riuscita..."
bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:${SLEEP_SECONDS}" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/warm.log" 2>&1

echo "[test] Invocazione warm fallita (errore atteso)..."
set +e
bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:not-a-number" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/failed.log" 2>&1
FAILED_STATUS=$?
set -e

if [[ "$FAILED_STATUS" -eq 0 ]]; then
  fail "L'invocazione volutamente errata è risultata riuscita."
fi

# L'export è eseguito in defer dopo la completion notification: attendiamo
# esplicitamente che le tre righe siano visibili.
for _ in $(seq 1 50); do
  if [[ -f "$DATASET_FILE" ]] && [[ "$(wc -l < "$DATASET_FILE")" -ge 3 ]]; then
    break
  fi
  sleep 0.2
done

[[ -f "$DATASET_FILE" ]] || fail "Dataset JSONL non creato."

python3 - "$DATASET_FILE" "$FUNCTION_NAME" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
function_name = sys.argv[2]
raw = path.read_bytes()

failures = []

if not raw.endswith(b"\n"):
    failures.append("il file JSONL non termina con newline")

lines = [line for line in raw.splitlines() if line.strip()]
samples = []

for index, line in enumerate(lines, start=1):
    try:
        sample = json.loads(line)
    except json.JSONDecodeError as exc:
        failures.append(f"riga {index}: JSON non valido: {exc}")
        continue

    samples.append(sample)

if len(samples) != 3:
    failures.append(f"attesi 3 campioni, trovati {len(samples)}")

for index, sample in enumerate(samples, start=1):
    if sample.get("schema_version") != 1:
        failures.append(f"campione {index}: schema_version non valido")
    if sample.get("function_name") != function_name:
        failures.append(f"campione {index}: function_name non valido")
    if not sample.get("request_id"):
        failures.append(f"campione {index}: request_id mancante")
    if not sample.get("machine_tag"):
        failures.append(f"campione {index}: machine_tag mancante")
    if not sample.get("node_name"):
        failures.append(f"campione {index}: node_name mancante")
    if "eligibility" not in sample:
        failures.append(f"campione {index}: eligibility mancante")

request_ids = [sample.get("request_id") for sample in samples]
if len(set(request_ids)) != len(request_ids):
    failures.append("request_id duplicati nel dataset")

cold_success = [
    sample for sample in samples
    if sample.get("warm_start") is False
    and sample.get("execution_succeeded") is True
]
warm_success = [
    sample for sample in samples
    if sample.get("warm_start") is True
    and sample.get("execution_succeeded") is True
]
failed = [
    sample for sample in samples
    if sample.get("execution_succeeded") is False
]

if len(cold_success) != 1:
    failures.append(f"atteso 1 campione cold riuscito, trovati {len(cold_success)}")
else:
    sample = cold_success[0]
    eligibility = sample.get("eligibility", {})
    timing = sample.get("timing", {})

    if sample.get("profile") not in (None, {}):
        failures.append("cold: il profilo risorse deve essere assente")
    if eligibility.get("resource_clustering") is not False:
        failures.append("cold: resource_clustering deve essere false")
    if eligibility.get("cold_start_analysis") is not True:
        failures.append("cold: cold_start_analysis deve essere true")
    if eligibility.get("performance_analysis") is not False:
        failures.append("cold: performance_analysis deve essere false")
    if "cold_start" not in eligibility.get("exclusion_reasons", []):
        failures.append("cold: exclusion_reasons non contiene cold_start")

    init_time = timing.get("init_time_ms")
    if not isinstance(init_time, (int, float)) or not math.isfinite(init_time) or init_time < 0:
        failures.append("cold: init_time_ms non valido")

if len(warm_success) != 1:
    failures.append(f"atteso 1 campione warm riuscito, trovati {len(warm_success)}")
else:
    sample = warm_success[0]
    eligibility = sample.get("eligibility", {})
    profile = sample.get("profile")

    if not isinstance(profile, dict):
        failures.append("warm: profilo risorse assente")
    else:
        # InvocationResourceProfile al momento usa i nomi Go esportati.
        for key, expected in (
            ("Enabled", True),
            ("Collected", True),
            ("Valid", True),
            ("ExclusiveContainer", True),
        ):
            if profile.get(key) is not expected:
                failures.append(f"warm: {key}={profile.get(key)!r}, atteso {expected!r}")

    if eligibility.get("resource_clustering") is not True:
        failures.append("warm: resource_clustering deve essere true")
    if eligibility.get("cold_start_analysis") is not False:
        failures.append("warm: cold_start_analysis deve essere false")
    if eligibility.get("performance_analysis") is not True:
        failures.append("warm: performance_analysis deve essere true")

if len(failed) != 1:
    failures.append(f"atteso 1 campione fallito, trovati {len(failed)}")
else:
    sample = failed[0]
    eligibility = sample.get("eligibility", {})

    if not sample.get("execution_error"):
        failures.append("failed: execution_error mancante")
    if eligibility.get("resource_clustering") is not False:
        failures.append("failed: resource_clustering deve essere false")
    if eligibility.get("performance_analysis") is not False:
        failures.append("failed: performance_analysis deve essere false")
    if "execution_failed" not in eligibility.get("exclusion_reasons", []):
        failures.append("failed: exclusion_reasons non contiene execution_failed")

print("============================================================")
print("VALIDAZIONE MODIFICA 07")
print("============================================================")
print(f"file:                   {path}")
print(f"righe JSON valide:      {len(samples)}")
print(f"cold riusciti:          {len(cold_success)}")
print(f"warm riusciti:          {len(warm_success)}")
print(f"falliti:                {len(failed)}")
print(f"request ID unici:       {len(set(request_ids))}")

if failures:
    print()
    for failure in failures:
        print(f"[FAIL] {failure}")
    sys.exit(1)

print()
print("[PASS] Ogni invocazione ha prodotto una sola riga JSON valida.")
print("[PASS] Il cold start conserva InitTime ma non il profilo risorse.")
print("[PASS] Il warm riuscito è eleggibile per il clustering.")
print("[PASS] L'esecuzione fallita è conservata ma non è eleggibile.")
print("[PASS] Metadati, schema ed eligibility sono presenti.")
PY

echo
echo "[done] Dataset: $DATASET_FILE"
echo "[done] Log:     $LOG_DIR"
