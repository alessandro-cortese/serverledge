#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICY="${POLICY:-UCB1}"
INVOCATIONS="${INVOCATIONS:-3}"
CPU_SECONDS="${CPU_SECONDS:-2.5}"
REWARD_ZONE="${REWARD_ZONE:-package}"
KEPLER_URL="${KEPLER_URL:-http://127.0.0.1:28282/metrics}"

UCB1_C="${UCB1_C:-0.8}"
LINUCB_ALPHA="${LINUCB_ALPHA:-0.1}"

API_PORT="${API_PORT:-1323}"
LB_PORT="${LB_PORT:-8080}"
UDP_PORT="${UDP_PORT:-9876}"

POOL_MEMORY_MB="${POOL_MEMORY_MB:-512}"
POOL_CPUS="${POOL_CPUS:-4}"
FUNCTION_MEMORY_MB="${FUNCTION_MEMORY_MB:-128}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-60}"

KEPLER_TIMEOUT_MS="${KEPLER_TIMEOUT_MS:-3000}"
KEPLER_POLL_INTERVAL_MS="${KEPLER_POLL_INTERVAL_MS:-100}"
KEPLER_REFRESH_TIMEOUT_MS="${KEPLER_REFRESH_TIMEOUT_MS:-5000}"
KEPLER_DISCOVERY_TIMEOUT="${KEPLER_DISCOVERY_TIMEOUT:-30}"

SLEEP_BETWEEN_INVOCATIONS="${SLEEP_BETWEEN_INVOCATIONS:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
POLICY_LOWER="$(printf '%s' "$POLICY" | tr '[:upper:]' '[:lower:]')"
RUN_ID="${RUN_ID:-kepler-mab-${POLICY_LOWER}-local-${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
FUNCTION_NAME="${FUNCTION_NAME:-kepler_energy_${POLICY_LOWER}_${STAMP}}"

NODE_CONF="$LOG_DIR/node.yaml"
LB_CONF="$LOG_DIR/lb.yaml"
NODE_LOG="$LOG_DIR/node.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"
SUMMARY_JSON="$LOG_DIR/summary.json"
INVOCATIONS_JSONL="$LOG_DIR/invocations.jsonl"

mkdir -p "$LOG_DIR/requests"

PIDS=()
ETCD_STARTED=0
PREWARM_CONTAINER_ID=""

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

cleanup() {
  status=$?
  set +e

  echo
  echo "[cleanup] Arresto processi..."

  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" >/dev/null 2>&1 || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  done

  if [[ -n "$PREWARM_CONTAINER_ID" ]]; then
    docker rm -f "$PREWARM_CONTAINER_ID" >/dev/null 2>&1 || true
  fi

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1 || true
  fi

  echo "[cleanup] Artefatti: $LOG_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1"
  local pid="$2"
  local logfile="$3"

  for _ in $(seq 1 100); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$logfile" || true
      fail "processo terminato durante l'avvio: $logfile"
    fi

    if curl -fsS --max-time 1 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi

    sleep 0.25
  done

  tail -100 "$logfile" || true
  fail "timeout aspettando $url"
}

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 || fail "comando richiesto non trovato: $cmd"
done

case "$POLICY" in
  UCB1|LinUCB) ;;
  *) fail "POLICY deve essere UCB1 oppure LinUCB" ;;
esac

[[ "$INVOCATIONS" =~ ^[1-9][0-9]*$ ]] || fail "INVOCATIONS deve essere > 0"

python3 - "$CPU_SECONDS" "$UCB1_C" "$LINUCB_ALPHA" <<'PY'
import math
import sys

for name, raw, positive in (
    ("CPU_SECONDS", sys.argv[1], True),
    ("UCB1_C", sys.argv[2], False),
    ("LINUCB_ALPHA", sys.argv[3], False),
):
    value = float(raw)
    if not math.isfinite(value):
        raise SystemExit(f"{name} deve essere finito")
    if positive and value <= 0:
        raise SystemExit(f"{name} deve essere > 0")
    if not positive and value < 0:
        raise SystemExit(f"{name} deve essere >= 0")
PY

[[ -f examples/kepler_cpu_smoke.py ]] \
  || fail "manca examples/kepler_cpu_smoke.py"

echo "============================================================"
echo "KEPLER -> MAB ENERGY LOCAL — $POLICY"
echo "============================================================"
echo "run_id:                 $RUN_ID"
echo "function:               $FUNCTION_NAME"
echo "policy:                 $POLICY"
echo "invocations:            $INVOCATIONS"
echo "CPU seconds/invocation: $CPU_SECONDS"
echo "reward mode:            kepler_energy"
echo "reward zone:            $REWARD_ZONE"
echo "Kepler URL:             $KEPLER_URL"
echo

echo "[preflight] Kepler endpoint..."
curl -fsS --max-time 5 "$KEPLER_URL" >"$LOG_DIR/kepler-preflight.prom" \
  || fail "Kepler non raggiungibile su $KEPLER_URL"

grep -q '^kepler_build_info' "$LOG_DIR/kepler-preflight.prom" \
  || fail "endpoint raggiungibile ma kepler_build_info assente"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[setup] Build..."
  make >/dev/null
fi

docker info >/dev/null 2>&1 || fail "Docker non disponibile"
docker rm -f Etcd-server >/dev/null 2>&1 || true

REGISTRY_AREA="KEPLER_MAB_${POLICY}_${STAMP}"

cat >"$NODE_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: ${REGISTRY_AREA}
registry.node.id: kepler-node-${POLICY_LOWER}-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: x86

container.pool.memory: ${POOL_MEMORY_MB}
container.pool.cpus: ${POOL_CPUS}
container.expiration: ${CONTAINER_EXPIRATION}

profiling.enabled: false
profiling.export.enabled: false

profiling.kepler.enabled: true
profiling.kepler.url: "${KEPLER_URL}"
profiling.kepler.timeout_ms: ${KEPLER_TIMEOUT_MS}
profiling.kepler.poll_interval_ms: ${KEPLER_POLL_INTERVAL_MS}
profiling.kepler.refresh_timeout_ms: ${KEPLER_REFRESH_TIMEOUT_MS}

scheduler.queue.capacity: 0
YAML

cat >"$LB_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${LB_PORT}

registry.area: ${REGISTRY_AREA}

lb.arch_awareness: true
lb.mode: MAB

mab.policy: ${POLICY}
mab.ucb1.c: ${UCB1_C}
mab.linucb.alpha: ${LINUCB_ALPHA}
mab.cold_start.mode: skip

mab.reward.mode: kepler_energy
mab.reward.kepler.zone: ${REWARD_ZONE}

lb.replicas: 128
lb.refresh_interval: 1
YAML

echo "[setup] Avvio etcd..."
bash scripts/start-etcd.sh >"$ETCD_LOG" 2>&1
ETCD_STARTED=1

for _ in $(seq 1 30); do
  docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 && break
  sleep 0.5
done
docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 \
  || fail "etcd non disponibile"

echo "[setup] Avvio nodo Serverledge..."
bin/serverledge "$NODE_CONF" >"$NODE_LOG" 2>&1 &
node_pid="$!"
PIDS+=("$node_pid")
wait_http "http://127.0.0.1:${API_PORT}/status" "$node_pid" "$NODE_LOG"

echo "[setup] Avvio LB $POLICY..."
bin/lb "$LB_CONF" >"$LB_LOG" 2>&1 &
lb_pid="$!"
PIDS+=("$lb_pid")
wait_http "http://127.0.0.1:${LB_PORT}/status" "$lb_pid" "$LB_LOG"

sleep 2

echo "[setup] Creo funzione CPU-bound..."
bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory "$FUNCTION_MEMORY_MB" \
  --cpu 1 \
  --max_concurrency 1 \
  --src examples/kepler_cpu_smoke.py \
  --runtime python314 \
  --handler "kepler_cpu_smoke.handler" \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/create.json"

BEFORE_IDS="$LOG_DIR/runtime-containers-before.txt"
AFTER_IDS="$LOG_DIR/runtime-containers-after.txt"

docker ps \
  --filter ancestor=grussorusso/serverledge-python314 \
  --format '{{.ID}}' \
  | sort >"$BEFORE_IDS"

echo "[setup] Prewarm di un container..."
bin/serverledge-cli prewarm \
  -f "$FUNCTION_NAME" \
  -c 1 \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/prewarm.json"

sleep 0.5

docker ps \
  --filter ancestor=grussorusso/serverledge-python314 \
  --format '{{.ID}}' \
  | sort >"$AFTER_IDS"

NEW_IDS="$(comm -13 "$BEFORE_IDS" "$AFTER_IDS" || true)"
NEW_COUNT="$(printf '%s\n' "$NEW_IDS" | sed '/^$/d' | wc -l)"

[[ "$NEW_COUNT" == "1" ]] \
  || {
    echo "Container runtime prima:"
    cat "$BEFORE_IDS" || true
    echo "Container runtime dopo:"
    cat "$AFTER_IDS" || true
    fail "atteso esattamente 1 nuovo runtime container dopo prewarm, trovati $NEW_COUNT"
  }

PREWARM_CONTAINER_ID_SHORT="$(printf '%s\n' "$NEW_IDS" | sed '/^$/d' | head -1)"
PREWARM_CONTAINER_ID="$(docker inspect --format '{{.Id}}' "$PREWARM_CONTAINER_ID_SHORT")"

echo "[setup] prewarm container_id=$PREWARM_CONTAINER_ID"

echo "[setup] Attendo che Kepler esponga il container prewarmed..."
deadline="$(( $(date +%s) + KEPLER_DISCOVERY_TIMEOUT ))"
while true; do
  curl -fsS --max-time 3 "$KEPLER_URL" >"$LOG_DIR/kepler-discovery.prom"

  if grep '^kepler_container_cpu_joules_total' "$LOG_DIR/kepler-discovery.prom" \
      | grep -Fq "container_id=\"${PREWARM_CONTAINER_ID}\""; then
    break
  fi

  (( $(date +%s) < deadline )) \
    || fail "Kepler non ha esposto il container prewarmed entro ${KEPLER_DISCOVERY_TIMEOUT}s"

  sleep 0.5
done

echo "[setup] Kepler vede il container prewarmed."
grep '^kepler_container_cpu_joules_total' "$LOG_DIR/kepler-discovery.prom" \
  | grep -F "container_id=\"${PREWARM_CONTAINER_ID}\"" \
  >"$LOG_DIR/prewarm-kepler-metrics.txt"

grep -q "zone=\"${REWARD_ZONE}\"" "$LOG_DIR/prewarm-kepler-metrics.txt" \
  || {
    echo "Zone disponibili:"
    cat "$LOG_DIR/prewarm-kepler-metrics.txt"
    fail "reward zone '$REWARD_ZONE' non disponibile per il container"
  }

sleep 1

echo "[test] Invio $INVOCATIONS invocation warm tramite LB..."
: >"$INVOCATIONS_JSONL"

for i in $(seq 1 "$INVOCATIONS"); do
  request_file="$LOG_DIR/requests/invoke-$(printf '%02d' "$i").json"

  bin/serverledge-cli invoke \
    -f "$FUNCTION_NAME" \
    -p "seconds:${CPU_SECONDS}" \
    --return_output \
    -H 127.0.0.1 \
    -P "$LB_PORT" \
    >"$request_file" \
    2>&1

  python3 - "$request_file" "$i" "$PREWARM_CONTAINER_ID" "$REWARD_ZONE" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
index = int(sys.argv[2])
expected_container = sys.argv[3]
zone = sys.argv[4]

data = json.loads(path.read_text(encoding="utf-8"))

if data.get("Success") is not True:
    raise SystemExit(f"invocation {index}: Success != true")

if data.get("IsWarmStart") is not True:
    raise SystemExit(f"invocation {index}: non è warm")

energy = data.get("KeplerEnergy")
if not isinstance(energy, dict):
    raise SystemExit(f"invocation {index}: KeplerEnergy assente")

if energy.get("available") is not True:
    raise SystemExit(
        f"invocation {index}: KeplerEnergy unavailable: "
        f"{energy.get('invalid_reason')}"
    )

container_id = energy.get("container_id", "")
if container_id != expected_container:
    raise SystemExit(
        f"invocation {index}: container mismatch "
        f"{container_id!r} != {expected_container!r}"
    )

zones = energy.get("cpu_joules_by_zone") or {}
if zone not in zones:
    raise SystemExit(
        f"invocation {index}: zona {zone!r} assente; "
        f"zone={sorted(zones)}"
    )

joules = float(zones[zone])
if not math.isfinite(joules) or joules <= 0:
    raise SystemExit(
        f"invocation {index}: energia {zone} non valida: {joules}"
    )

print(
    f"[invoke {index}] PASS warm=true "
    f"{zone}={joules:.9f} J "
    f"container={container_id[:12]}"
)
PY

  python3 - "$request_file" >>"$INVOCATIONS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(data, sort_keys=True))
PY

  sleep "$SLEEP_BETWEEN_INVOCATIONS"
done

sleep 1

NODE_TRUE_COUNT="$(
  grep -c "event=kepler_invocation_energy available=true.*container_id=${PREWARM_CONTAINER_ID}" \
    "$NODE_LOG" 2>/dev/null || true
)"
NODE_FALSE_COUNT="$(
  grep -c "event=kepler_invocation_energy available=false.*container_id=${PREWARM_CONTAINER_ID}" \
    "$NODE_LOG" 2>/dev/null || true
)"
LB_TRUE_COUNT="$(
  grep -c "event=kepler_energy_feedback.*function=${FUNCTION_NAME}.*available=true" \
    "$LB_LOG" 2>/dev/null || true
)"
LB_FALSE_COUNT="$(
  grep -c "event=kepler_energy_feedback.*function=${FUNCTION_NAME}.*available=false" \
    "$LB_LOG" 2>/dev/null || true
)"
REWARD_COUNT="$(
  grep -c "event=reward_breakdown.*function=${FUNCTION_NAME}.*reward_mode=kepler_energy" \
    "$LB_LOG" 2>/dev/null || true
)"
UPDATE_COUNT="$(
  grep -c "event=update_reward.*policy=${POLICY}.*function=${FUNCTION_NAME}" \
    "$LB_LOG" 2>/dev/null || true
)"
REWARD_FAIL_COUNT="$(
  grep -c "event=reward_calculation_failed.*function=${FUNCTION_NAME}" \
    "$LB_LOG" 2>/dev/null || true
)"
RESOLVED_COUNT="$(
  grep -c "event=decision_resolved.*function=${FUNCTION_NAME}" \
    "$LB_LOG" 2>/dev/null || true
)"
FALLBACK_COUNT="$(
  grep "event=decision_resolved.*function=${FUNCTION_NAME}" "$LB_LOG" 2>/dev/null \
    | grep -c 'fallback=true' || true
)"

[[ "$NODE_TRUE_COUNT" == "$INVOCATIONS" ]] \
  || fail "node Kepler available=true=$NODE_TRUE_COUNT, atteso=$INVOCATIONS"
[[ "$NODE_FALSE_COUNT" == "0" ]] \
  || fail "node Kepler available=false inattesi=$NODE_FALSE_COUNT"
[[ "$LB_TRUE_COUNT" == "$INVOCATIONS" ]] \
  || fail "LB Kepler available=true=$LB_TRUE_COUNT, atteso=$INVOCATIONS"
[[ "$LB_FALSE_COUNT" == "0" ]] \
  || fail "LB Kepler available=false inattesi=$LB_FALSE_COUNT"
[[ "$REWARD_COUNT" == "$INVOCATIONS" ]] \
  || fail "reward_breakdown=$REWARD_COUNT, atteso=$INVOCATIONS"
[[ "$UPDATE_COUNT" == "$INVOCATIONS" ]] \
  || fail "update_reward=$UPDATE_COUNT, atteso=$INVOCATIONS"
[[ "$REWARD_FAIL_COUNT" == "0" ]] \
  || fail "reward_calculation_failed inattesi=$REWARD_FAIL_COUNT"
[[ "$RESOLVED_COUNT" == "$INVOCATIONS" ]] \
  || fail "decision_resolved=$RESOLVED_COUNT, atteso=$INVOCATIONS"
[[ "$FALLBACK_COUNT" == "0" ]] \
  || fail "fallback inattesi=$FALLBACK_COUNT"

python3 - \
  "$INVOCATIONS_JSONL" \
  "$LB_LOG" \
  "$FUNCTION_NAME" \
  "$POLICY" \
  "$REWARD_ZONE" \
  "$INVOCATIONS" \
  "$PREWARM_CONTAINER_ID" \
  "$SUMMARY_JSON" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

(
    invocations_path,
    lb_log_path,
    function_name,
    policy,
    zone,
    expected_count,
    container_id,
    summary_path,
) = sys.argv[1:]

expected_count = int(expected_count)

responses = [
    json.loads(line)
    for line in Path(invocations_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(responses) != expected_count:
    raise SystemExit(
        f"responses={len(responses)}, atteso={expected_count}"
    )

response_energies = []
for index, response in enumerate(responses, start=1):
    energy = response["KeplerEnergy"]
    response_energies.append(float(energy["cpu_joules_by_zone"][zone]))

reward_rows = []
for line in Path(lb_log_path).read_text(encoding="utf-8", errors="replace").splitlines():
    if "event=reward_breakdown" not in line:
        continue
    if f"function={function_name}" not in line:
        continue
    if f"policy={policy}" not in line:
        continue

    fields = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value.strip('"')

    reward_rows.append({
        "energy": float(fields["reward_input_value"]),
        "reward": float(fields["final_reward"]),
        "mode": fields["reward_mode"],
        "input": fields["reward_input"],
        "unit": fields["reward_input_unit"],
    })

if len(reward_rows) != expected_count:
    raise SystemExit(
        f"reward rows={len(reward_rows)}, atteso={expected_count}"
    )

samples = []
for index, (response_energy, row) in enumerate(
    zip(response_energies, reward_rows),
    start=1,
):
    if row["mode"] != "kepler_energy":
        raise SystemExit(f"sample {index}: reward_mode errato")
    if row["input"] != f"kepler_energy_{zone}":
        raise SystemExit(
            f"sample {index}: reward_input={row['input']!r}"
        )
    if row["unit"] != "J":
        raise SystemExit(f"sample {index}: unità={row['unit']!r}")

    if abs(response_energy - row["energy"]) > 1e-5:
        raise SystemExit(
            f"sample {index}: response/log energy mismatch "
            f"{response_energy} vs {row['energy']}"
        )

    expected_reward = -math.log(row["energy"] / 1.0)
    error = abs(row["reward"] - expected_reward)
    if error > 1e-5:
        raise SystemExit(
            f"sample {index}: reward formula mismatch "
            f"reward={row['reward']} expected={expected_reward} "
            f"error={error}"
        )

    samples.append({
        "index": index,
        "energy_j": row["energy"],
        "reward": row["reward"],
        "expected_reward": expected_reward,
        "absolute_error": error,
    })

summary = {
    "schema_version": 1,
    "status": "PASS",
    "policy": policy,
    "function": function_name,
    "reward_mode": "kepler_energy",
    "reward_zone": zone,
    "invocations": expected_count,
    "warm_invocations": expected_count,
    "valid_kepler_measurements": expected_count,
    "energy_coverage": 1.0,
    "container_id": container_id,
    "samples": samples,
    "mean_energy_j": sum(s["energy_j"] for s in samples) / len(samples),
}

Path(summary_path).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print()
print("NUMERICAL REWARD CHECK: PASS")
for sample in samples:
    print(
        f"sample {sample['index']}: "
        f"E={sample['energy_j']:.6f} J "
        f"reward={sample['reward']:.6f} "
        f"expected={sample['expected_reward']:.6f} "
        f"error={sample['absolute_error']:.3e}"
    )
print("energy coverage=100%")
PY

echo
echo "===== NODE KEPLER ====="
grep "event=kepler_invocation_energy.*container_id=${PREWARM_CONTAINER_ID}" \
  "$NODE_LOG" || true

echo
echo "===== LB KEPLER FEEDBACK ====="
grep "event=kepler_energy_feedback.*function=${FUNCTION_NAME}" \
  "$LB_LOG" || true

echo
echo "===== ENERGY REWARD ====="
grep "event=reward_breakdown.*function=${FUNCTION_NAME}" \
  "$LB_LOG" || true

echo
echo "===== MAB UPDATE ====="
grep "event=update_reward.*policy=${POLICY}.*function=${FUNCTION_NAME}" \
  "$LB_LOG" || true

echo
echo "===== SUMMARY ====="
cat "$SUMMARY_JSON"

echo
echo "============================================================"
echo "KEPLER -> MAB ENERGY LOCAL $POLICY: PASS"
echo "============================================================"
echo "summary: $SUMMARY_JSON"
echo
echo "Interpretazione: pipeline locale end-to-end validata."
echo "NON è un confronto prestazionale/energetico fra architetture."
