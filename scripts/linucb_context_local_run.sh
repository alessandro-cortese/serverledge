#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO="${SCENARIO:-A}"
TARGET_REQUESTS="${TARGET_REQUESTS:-8}"
PHASE1_REQUESTS="${PHASE1_REQUESTS:-4}"
MAB_LINUCB_ALPHA="${MAB_LINUCB_ALPHA:-0.1}"

TARGET_MEMORY_MB="${TARGET_MEMORY_MB:-128}"
BACKGROUND_MEMORY_MB="${BACKGROUND_MEMORY_MB:-128}"
BACKGROUND_FUNCTION_COUNT="${BACKGROUND_FUNCTION_COUNT:-2}"
BACKGROUND_HOLD_SECONDS="${BACKGROUND_HOLD_SECONDS:-12}"

POOL_MEMORY_MB="${POOL_MEMORY_MB:-512}"
POOL_CPUS="${POOL_CPUS:-4}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-60}"
JANITOR_INTERVAL="${JANITOR_INTERVAL:-1}"

SLEEP_BETWEEN_TARGET_REQUESTS="${SLEEP_BETWEEN_TARGET_REQUESTS:-0.25}"
CONTEXT_SETTLE_SECONDS="${CONTEXT_SETTLE_SECONDS:-2}"
SKIP_BUILD="${SKIP_BUILD:-0}"

AMD64_PORT="${AMD64_PORT:-1333}"
ARM64_PORT="${ARM64_PORT:-1334}"
LB_PORT="${LB_PORT:-8080}"
AMD64_UDP_PORT="${AMD64_UDP_PORT:-9899}"
ARM64_UDP_PORT="${ARM64_UDP_PORT:-9900}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCENARIO_LOWER="$(printf '%s' "$SCENARIO" | tr '[:upper:]' '[:lower:]')"
RUN_ID="${RUN_ID:-linucb-context-${SCENARIO_LOWER}-local-${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"

TARGET_FUNCTION="${TARGET_FUNCTION:-linucb_context_target_${STAMP}}"
BACKGROUND_FUNCTION_BASE="${BACKGROUND_FUNCTION_BASE:-linucb_context_background_${STAMP}}"

mkdir -p "$LOG_DIR/requests"

AMD64_CONF="$LOG_DIR/node-amd64.yaml"
ARM64_CONF="$LOG_DIR/node-arm64.yaml"
LB_CONF="$LOG_DIR/lb.yaml"

AMD64_LOG="$LOG_DIR/node-amd64.log"
ARM64_LOG="$LOG_DIR/node-arm64.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"

OBS_CSV="$LOG_DIR/observations.csv"
SUMMARY_JSON="$LOG_DIR/summary.json"

PIDS=()
BACKGROUND_PIDS=()
ETCD_STARTED=0

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

cleanup() {
  status=$?
  set +e

  echo
  echo "[cleanup] Arresto processi locali..."

  for pid in "${BACKGROUND_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" >/dev/null 2>&1 || true
    fi
  done

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

    if curl -sS --max-time 1 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi

    sleep 0.25
  done

  tail -100 "$logfile" || true
  fail "timeout aspettando $url"
}

read_free_memory() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}/status" \
    | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["FreeMemory"]))'
}

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 || fail "comando richiesto non trovato: $cmd"
done

case "$SCENARIO" in
  A|B) ;;
  *) fail "SCENARIO deve essere A oppure B" ;;
esac

[[ "$TARGET_REQUESTS" =~ ^[1-9][0-9]*$ ]] || fail "TARGET_REQUESTS deve essere > 0"
[[ "$PHASE1_REQUESTS" =~ ^[1-9][0-9]*$ ]] || fail "PHASE1_REQUESTS deve essere > 0"
(( PHASE1_REQUESTS < TARGET_REQUESTS )) || fail "PHASE1_REQUESTS deve essere < TARGET_REQUESTS"

for value_name in TARGET_MEMORY_MB BACKGROUND_MEMORY_MB BACKGROUND_FUNCTION_COUNT BACKGROUND_HOLD_SECONDS POOL_MEMORY_MB POOL_CPUS CONTAINER_EXPIRATION JANITOR_INTERVAL; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$value_name deve essere > 0"
done

background_total_mb=$((BACKGROUND_MEMORY_MB * BACKGROUND_FUNCTION_COUNT))
(( background_total_mb + TARGET_MEMORY_MB <= POOL_MEMORY_MB )) \
  || fail "background + target superano il pool arm64"

python3 - "$MAB_LINUCB_ALPHA" <<'PY'
import math
import sys

alpha = float(sys.argv[1])
if not math.isfinite(alpha) or alpha < 0:
    raise SystemExit("MAB_LINUCB_ALPHA deve essere finito e >= 0")
PY

for port in "$AMD64_PORT" "$ARM64_PORT" "$LB_PORT"; do
  if curl -sS --max-time 1 -o /dev/null "http://127.0.0.1:${port}/status" 2>/dev/null; then
    fail "porta API già occupata: $port"
  fi
done

echo "============================================================"
echo "LINUCB CONTEXT LOCAL — SCENARIO $SCENARIO"
echo "============================================================"
echo "run_id:                    $RUN_ID"
echo "target:                    $TARGET_FUNCTION"
echo "scenario:                  $SCENARIO"
echo "target requests:           $TARGET_REQUESTS"
echo "phase-1 requests:          $PHASE1_REQUESTS"
echo "phase-2 requests:          $((TARGET_REQUESTS - PHASE1_REQUESTS))"
echo "LinUCB alpha:              $MAB_LINUCB_ALPHA"
echo "target memory:             ${TARGET_MEMORY_MB} MiB"
echo "background memory each:    ${BACKGROUND_MEMORY_MB} MiB"
echo "background function count: ${BACKGROUND_FUNCTION_COUNT}"
echo "background hold:           ${BACKGROUND_HOLD_SECONDS}s"
echo "pool memory/ring:          ${POOL_MEMORY_MB} MiB"
echo "container expiration:      ${CONTAINER_EXPIRATION}s"
if [[ "$SCENARIO" == "A" ]]; then
  echo "background context load:   NONE"
else
  echo "background context load:   ${BACKGROUND_FUNCTION_COUNT} concurrent ACTIVE FaaS invocations on arm64"
fi
echo
echo "ATTENZIONE: amd64 e arm64 sono due ring LOGICI sullo stesso host fisico."
echo "Il test valida context, request pairing e pipeline LinUCB; NON prestazioni ARM-vs-amd64."
echo

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[setup] Build..."
  make >/dev/null
fi

docker info >/dev/null 2>&1 || fail "Docker non disponibile"
docker rm -f Etcd-server >/dev/null 2>&1 || true

cat > "$AMD64_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${AMD64_PORT}
registry.area: LINUCB_CONTEXT_LOCAL
registry.node.id: linucb-amd64-${STAMP}
registry.udp.port: ${AMD64_UDP_PORT}
node.machine_tag: amd64
container.pool.memory: ${POOL_MEMORY_MB}
container.pool.cpus: ${POOL_CPUS}
container.expiration: ${CONTAINER_EXPIRATION}
janitor.interval: ${JANITOR_INTERVAL}
factory.images.refresh: false
profiling.enabled: false
profiling.export.enabled: false
scheduler.queue.capacity: 0
YAML

cat > "$ARM64_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${ARM64_PORT}
registry.area: LINUCB_CONTEXT_LOCAL
registry.node.id: linucb-arm64-${STAMP}
registry.udp.port: ${ARM64_UDP_PORT}
node.machine_tag: arm64
container.pool.memory: ${POOL_MEMORY_MB}
container.pool.cpus: ${POOL_CPUS}
container.expiration: ${CONTAINER_EXPIRATION}
janitor.interval: ${JANITOR_INTERVAL}
factory.images.refresh: false
profiling.enabled: false
profiling.export.enabled: false
scheduler.queue.capacity: 0
YAML

cat > "$LB_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${LB_PORT}
registry.area: LINUCB_CONTEXT_LOCAL
lb.arch_awareness: true
lb.mode: MAB
mab.policy: LinUCB
mab.linucb.alpha: ${MAB_LINUCB_ALPHA}
mab.reward.mode: latency
mab.cold_start.mode: skip
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
docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 || fail "etcd non disponibile"

echo "[setup] Avvio nodo amd64..."
bin/serverledge "$AMD64_CONF" >"$AMD64_LOG" 2>&1 &
amd64_pid="$!"
PIDS+=("$amd64_pid")
wait_http "http://127.0.0.1:${AMD64_PORT}/status" "$amd64_pid" "$AMD64_LOG"

echo "[setup] Avvio nodo arm64..."
bin/serverledge "$ARM64_CONF" >"$ARM64_LOG" 2>&1 &
arm64_pid="$!"
PIDS+=("$arm64_pid")
wait_http "http://127.0.0.1:${ARM64_PORT}/status" "$arm64_pid" "$ARM64_LOG"

echo "[setup] Avvio LB LinUCB + observability..."
SERVERLEDGE_LINUCB_OBSERVABILITY=1 \
  bin/lb "$LB_CONF" >"$LB_LOG" 2>&1 &
lb_pid="$!"
PIDS+=("$lb_pid")
wait_http "http://127.0.0.1:${LB_PORT}/status" "$lb_pid" "$LB_LOG"

sleep 2

echo "[setup] Registro target senza tag/prewarm..."
bin/serverledge-cli create \
  -f "$TARGET_FUNCTION" \
  --memory "$TARGET_MEMORY_MB" \
  --src examples/hello.py \
  --runtime python314 \
  --handler "hello.handler" \
  --max_concurrency 1 \
  --cpu 1 \
  -H 127.0.0.1 \
  -P "$AMD64_PORT" \
  >"$LOG_DIR/create-target.json"

if [[ "$SCENARIO" == "B" ]]; then
  echo "[setup] Registro ${BACKGROUND_FUNCTION_COUNT} sleeper background functions..."
  for index in $(seq 1 "$BACKGROUND_FUNCTION_COUNT"); do
    background_function="${BACKGROUND_FUNCTION_BASE}_${index}"
    bin/serverledge-cli create \
      -f "$background_function" \
      --memory "$BACKGROUND_MEMORY_MB" \
      --src examples/sleeper.py \
      --runtime python314 \
      --handler "sleeper.handler" \
      --max_concurrency 1 \
      --cpu 1 \
      --input "n:Int" \
      -H 127.0.0.1 \
      -P "$AMD64_PORT" \
      >"$LOG_DIR/create-background-${index}.json"
  done
fi

sleep 1

OBS_BEFORE="$(grep -c "\[LINUCB_OBSERVABILITY\].*function=\"${TARGET_FUNCTION}\"" "$LB_LOG" 2>/dev/null || true)"
[[ "$OBS_BEFORE" == "0" ]] || fail "target ha observation LinUCB prima della misura"

invoke_target() {
  local index="$1"
  local phase="$2"

  bin/serverledge-cli invoke \
    -f "$TARGET_FUNCTION" \
    -p "name:World" \
    --return_output \
    -H 127.0.0.1 \
    -P "$LB_PORT" \
    >"$LOG_DIR/requests/target-$(printf '%03d' "$index")-phase${phase}.json" \
    2>&1

  grep -q '"Success"[[:space:]]*:[[:space:]]*true' \
    "$LOG_DIR/requests/target-$(printf '%03d' "$index")-phase${phase}.json" \
    || fail "target request $index fallita"

  sleep "$SLEEP_BETWEEN_TARGET_REQUESTS"
}

echo "[phase 1] Invio $PHASE1_REQUESTS target requests pulite..."
for index in $(seq 1 "$PHASE1_REQUESTS"); do
  invoke_target "$index" 1
done

ARM64_FREE_BEFORE="$(read_free_memory "$ARM64_PORT")"
echo "[phase boundary] arm64 FreeMemory before background=${ARM64_FREE_BEFORE} MiB"
curl -fsS "http://127.0.0.1:${ARM64_PORT}/status" >"$LOG_DIR/arm64-status-before-background.json"

if [[ "$SCENARIO" == "B" ]]; then
  echo "[phase boundary] Avvio ${BACKGROUND_FUNCTION_COUNT} invocazioni sleeper CONCORRENTI su arm64..."
  for index in $(seq 1 "$BACKGROUND_FUNCTION_COUNT"); do
    background_function="${BACKGROUND_FUNCTION_BASE}_${index}"
    (
      bin/serverledge-cli invoke \
        -f "$background_function" \
        -p "n:${BACKGROUND_HOLD_SECONDS}" \
        --return_output \
        -H 127.0.0.1 \
        -P "$ARM64_PORT" \
        >"$LOG_DIR/background-arm64-${index}.json" \
        2>&1
    ) &
    bg_pid="$!"
    BACKGROUND_PIDS+=("$bg_pid")
  done

  expected_free_max=$((POOL_MEMORY_MB - background_total_mb))
  echo "[phase boundary] Attendo che il worker esponga FreeMemory <= ${expected_free_max} MiB..."

  busy_seen=0
  for _ in $(seq 1 40); do
    current_free="$(read_free_memory "$ARM64_PORT")"
    if (( current_free <= expected_free_max )); then
      busy_seen=1
      break
    fi
    sleep 0.25
  done

  [[ "$busy_seen" -eq 1 ]] \
    || fail "il worker arm64 non ha esposto la memoria busy attesa"

  ARM64_FREE_BUSY="$(read_free_memory "$ARM64_PORT")"
  echo "[phase boundary] arm64 FreeMemory with active background=${ARM64_FREE_BUSY} MiB"
  curl -fsS "http://127.0.0.1:${ARM64_PORT}/status" >"$LOG_DIR/arm64-status-active-background.json"

  echo "[phase boundary] Attendo ${CONTEXT_SETTLE_SECONDS}s per refresh LB mentre il background è ancora attivo..."
  sleep "$CONTEXT_SETTLE_SECONDS"

  for pid in "${BACKGROUND_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null \
      || fail "background terminato prima della fase 2"
  done
else
  echo "[phase boundary] Scenario A: nessun background; stessa pausa di stabilizzazione."
  sleep "$CONTEXT_SETTLE_SECONDS"
  ARM64_FREE_BUSY="$ARM64_FREE_BEFORE"
fi

echo "[phase 2] Invio $((TARGET_REQUESTS - PHASE1_REQUESTS)) target requests..."
for index in $(seq $((PHASE1_REQUESTS + 1)) "$TARGET_REQUESTS"); do
  invoke_target "$index" 2
done

if [[ "$SCENARIO" == "B" ]]; then
  echo "[phase 2] Attendo completion delle invocazioni background..."
  for index in $(seq 1 "$BACKGROUND_FUNCTION_COUNT"); do
    pid="${BACKGROUND_PIDS[$((index - 1))]}"
    wait "$pid"
    grep -q '"Success"[[:space:]]*:[[:space:]]*true' \
      "$LOG_DIR/background-arm64-${index}.json" \
      || fail "background invocation #${index} fallita"
  done
  BACKGROUND_PIDS=()
fi

sleep 1

OBS_COUNT="$(grep -c "\[LINUCB_OBSERVABILITY\].*function=\"${TARGET_FUNCTION}\"" "$LB_LOG" 2>/dev/null || true)"
RESOLVED_COUNT="$(grep -c "event=decision_resolved.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null || true)"
FALLBACK_COUNT="$(grep "event=decision_resolved.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null | grep -c 'fallback=true' || true)"
UPDATE_COUNT="$(grep -c "event=update_reward.*policy=LinUCB.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null || true)"

[[ "$OBS_COUNT" == "$TARGET_REQUESTS" ]] \
  || fail "observations=$OBS_COUNT, atteso=$TARGET_REQUESTS"
[[ "$RESOLVED_COUNT" == "$TARGET_REQUESTS" ]] \
  || fail "decision_resolved=$RESOLVED_COUNT, atteso=$TARGET_REQUESTS"
[[ "$FALLBACK_COUNT" == "0" ]] \
  || fail "fallback inattesi=$FALLBACK_COUNT"

python3 - \
  "$LB_LOG" \
  "$TARGET_FUNCTION" \
  "$SCENARIO" \
  "$TARGET_REQUESTS" \
  "$PHASE1_REQUESTS" \
  "$MAB_LINUCB_ALPHA" \
  "$BACKGROUND_MEMORY_MB" \
  "$BACKGROUND_FUNCTION_COUNT" \
  "$BACKGROUND_HOLD_SECONDS" \
  "$ARM64_FREE_BEFORE" \
  "$ARM64_FREE_BUSY" \
  "$UPDATE_COUNT" \
  "$OBS_CSV" \
  "$SUMMARY_JSON" <<'PY'
import csv
import json
import re
import shlex
import sys
from pathlib import Path

(
    log_path,
    target,
    scenario,
    target_requests,
    phase1_requests,
    alpha,
    background_memory_mb,
    background_function_count,
    background_hold_seconds,
    arm64_free_before,
    arm64_free_busy,
    update_count,
    csv_path,
    summary_path,
) = sys.argv[1:]

target_requests = int(target_requests)
phase1_requests = int(phase1_requests)
alpha = float(alpha)
background_memory_mb = int(background_memory_mb)
background_function_count = int(background_function_count)
background_hold_seconds = int(background_hold_seconds)
arm64_free_before = int(arm64_free_before)
arm64_free_busy = int(arm64_free_busy)
update_count = int(update_count)

obs = []
decision_resolved_ids = set()

for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
    if "[LINUCB_OBSERVABILITY]" in line and f'function="{target}"' in line:
        payload = line.split("[LINUCB_OBSERVABILITY]", 1)[1].strip()
        fields = {}
        for token in shlex.split(payload):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value

        context = {}
        raw_context = fields.get("context", "")
        if raw_context:
            for item in raw_context.split(","):
                if not item:
                    continue
                arm, value = item.split("=", 1)
                context[arm] = float(value)

        obs.append({
            "request_id": fields.get("request_id", ""),
            "selected_arm": fields.get("selected_arm", ""),
            "execution_arm": fields.get("execution_arm", ""),
            "fallback": fields.get("fallback", ""),
            "context_present": fields.get("context_present", ""),
            "amd64_utilization": context.get("amd64"),
            "arm64_utilization": context.get("arm64"),
            "selected_utilization": float(fields.get("selected_utilization", "nan")),
            "execution_utilization": float(fields.get("execution_utilization", "nan")),
            "duration_ms": float(fields.get("duration_ms", "nan")),
            "warm_start": fields.get("warm_start", "").lower() == "true",
            "node_name": fields.get("node_name", ""),
            "execution_node": fields.get("execution_node", ""),
        })

    if "event=decision_resolved" in line and f"function={target}" in line:
        match = re.search(r"\brequest_id=([^ ]+)", line)
        if match:
            decision_resolved_ids.add(match.group(1).strip('"'))

if len(obs) != target_requests:
    raise SystemExit(f"osservazioni parse={len(obs)}, atteso={target_requests}")

if len({o["request_id"] for o in obs}) != target_requests:
    raise SystemExit("request_id LinUCB non univoci")

for o in obs:
    if not o["request_id"]:
        raise SystemExit("request_id vuoto")
    if o["request_id"] not in decision_resolved_ids:
        raise SystemExit(
            f"request_id senza decision_resolved: {o['request_id']}"
        )
    if o["context_present"].lower() != "true":
        raise SystemExit(
            f"context assente per request_id={o['request_id']}"
        )
    if o["amd64_utilization"] is None or o["arm64_utilization"] is None:
        raise SystemExit(
            f"context non contiene entrambi gli arm per {o['request_id']}"
        )
    if o["fallback"].lower() != "false":
        raise SystemExit(
            f"fallback inatteso per {o['request_id']}"
        )
    if o["selected_arm"] != o["execution_arm"]:
        raise SystemExit(
            f"selected/execution mismatch per {o['request_id']}"
        )

warm_count = sum(1 for o in obs if o["warm_start"])
cold_count = len(obs) - warm_count

if update_count != warm_count:
    raise SystemExit(
        f"update_reward={update_count}, warm observations={warm_count}"
    )

for idx, o in enumerate(obs, start=1):
    o["request_index"] = idx
    o["phase"] = 1 if idx <= phase1_requests else 2

phase1 = obs[:phase1_requests]
phase2 = obs[phase1_requests:]

def values(rows, key):
    return [float(row[key]) for row in rows if row[key] is not None]

summary = {
    "schema_version": 3,
    "scenario": scenario,
    "target_function": target,
    "target_requests": target_requests,
    "phase1_requests": phase1_requests,
    "phase2_requests": target_requests - phase1_requests,
    "linucb_alpha": alpha,
    "observation_count": len(obs),
    "decision_resolved_count": len(decision_resolved_ids),
    "warm_observation_count": warm_count,
    "cold_observation_count": cold_count,
    "update_reward_count": update_count,
    "fallback_count": 0,
    "selected_arm_counts": {
        arm: sum(1 for o in obs if o["selected_arm"] == arm)
        for arm in ("amd64", "arm64")
    },
    "phase1_context": {
        "amd64_min": min(values(phase1, "amd64_utilization")),
        "amd64_max": max(values(phase1, "amd64_utilization")),
        "arm64_min": min(values(phase1, "arm64_utilization")),
        "arm64_max": max(values(phase1, "arm64_utilization")),
    },
    "phase2_context": {
        "amd64_min": min(values(phase2, "amd64_utilization")),
        "amd64_max": max(values(phase2, "amd64_utilization")),
        "arm64_min": min(values(phase2, "arm64_utilization")),
        "arm64_max": max(values(phase2, "arm64_utilization")),
    },
    "background": {
        "memory_mb_each": background_memory_mb,
        "function_count": background_function_count,
        "hold_seconds": background_hold_seconds,
        "arm64_free_memory_before_mb": arm64_free_before,
        "arm64_free_memory_active_mb": arm64_free_busy,
        "worker_busy_memory_delta_mb": arm64_free_before - arm64_free_busy,
    },
}

if scenario == "B":
    pre_arm64_max = summary["phase1_context"]["arm64_max"]
    post_arm64_max = summary["phase2_context"]["arm64_max"]
    observed_shift = post_arm64_max - pre_arm64_max
    summary["arm64_context_shift_max"] = observed_shift

    if arm64_free_before - arm64_free_busy < background_memory_mb:
        raise SystemExit(
            "worker status non mostra nemmeno una background allocation: "
            f"before={arm64_free_before} busy={arm64_free_busy}"
        )

    if observed_shift < 0.20:
        raise SystemExit(
            "Scenario B non mostra uno shift sufficiente del context arm64 "
            "mentre il worker risulta busy: "
            f"pre_max={pre_arm64_max:.6f} "
            f"post_max={post_arm64_max:.6f} "
            f"shift={observed_shift:.6f}"
        )

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "request_index",
        "phase",
        "request_id",
        "selected_arm",
        "execution_arm",
        "warm_start",
        "amd64_utilization",
        "arm64_utilization",
        "selected_utilization",
        "execution_utilization",
        "duration_ms",
        "node_name",
        "execution_node",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for o in obs:
        writer.writerow({k: o.get(k) for k in fieldnames})

Path(summary_path).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("REQUEST/CONTEXT PAIRING: PASS")
print(f"observations={len(obs)}")
print(f"warm={warm_count} cold={cold_count} update_reward={update_count}")
print(
    "phase1 arm64 range="
    f"[{summary['phase1_context']['arm64_min']:.6f}, "
    f"{summary['phase1_context']['arm64_max']:.6f}]"
)
print(
    "phase2 arm64 range="
    f"[{summary['phase2_context']['arm64_min']:.6f}, "
    f"{summary['phase2_context']['arm64_max']:.6f}]"
)
print(
    "worker arm64 FreeMemory before/active="
    f"{arm64_free_before}/{arm64_free_busy} MiB"
)
if scenario == "B":
    print(
        "arm64 context shift(max)="
        f"{summary['arm64_context_shift_max']:.6f}"
    )
PY

echo
echo "===== LINUCB OBSERVATIONS ====="
grep "\[LINUCB_OBSERVABILITY\].*function=\"${TARGET_FUNCTION}\"" "$LB_LOG" || true

echo
echo "===== SUMMARY ====="
cat "$SUMMARY_JSON"

echo
echo "============================================================"
echo "LINUCB CONTEXT LOCAL SCENARIO $SCENARIO: PASS"
echo "============================================================"
echo "observations CSV: $OBS_CSV"
echo "summary JSON:     $SUMMARY_JSON"
