#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_ID="${CONFIG_ID:-N08}"
TARGET_FUNCTION="${TARGET_FUNCTION:-billibench-knn}"
FORMULA="${FORMULA:-none}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"
REWARD_PRIOR_WEIGHT="${REWARD_PRIOR_WEIGHT:-}"
EXPLORATION_PRIOR_WEIGHT="${EXPLORATION_PRIOR_WEIGHT:-}"
PRIOR_REQUEST="${PRIOR_REQUEST:-}"
OPTIMAL_ARM="${OPTIMAL_ARM:-arm64}"

REQUESTS="${REQUESTS:-5}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
CONTAINER_MEMORY_MB="${CONTAINER_MEMORY_MB:-128}"
POOL_MEMORY_MB="${POOL_MEMORY_MB:-512}"
POOL_CPUS="${POOL_CPUS:-4}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-4}"
JANITOR_INTERVAL="${JANITOR_INTERVAL:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"

AMD64_PORT="${AMD64_PORT:-1333}"
ARM64_PORT="${ARM64_PORT:-1334}"
LB_PORT="${LB_PORT:-8080}"
AMD64_UDP_PORT="${AMD64_UDP_PORT:-9899}"
ARM64_UDP_PORT="${ARM64_UDP_PORT:-9900}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-ucb1-memory-local-${CONFIG_ID}-${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
mkdir -p "$LOG_DIR/requests"

AMD64_CONF="$LOG_DIR/node-amd64.yaml"
ARM64_CONF="$LOG_DIR/node-arm64.yaml"
LB_CONF="$LOG_DIR/lb.yaml"
AMD64_LOG="$LOG_DIR/node-amd64.log"
ARM64_LOG="$LOG_DIR/node-arm64.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"
CONTROL_RESPONSE="$LOG_DIR/materialized-response.json"
SUMMARY_JSON="$LOG_DIR/summary.json"
SUMMARY_CSV="$LOG_DIR/container-residency.csv"

PIDS=()
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

  for _ in $(seq 1 80); do
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

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 || fail "comando richiesto non trovato: $cmd"
done

[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || fail "REQUESTS deve essere > 0"
[[ "$REQUEST_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || fail "REQUEST_CONCURRENCY deve essere > 0"
(( REQUEST_CONCURRENCY <= REQUESTS )) || fail "REQUEST_CONCURRENCY non può superare REQUESTS"
[[ "$CONTAINER_MEMORY_MB" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_MEMORY_MB deve essere > 0"
[[ "$POOL_MEMORY_MB" =~ ^[1-9][0-9]*$ ]] || fail "POOL_MEMORY_MB deve essere > 0"
[[ "$POOL_CPUS" =~ ^[1-9][0-9]*$ ]] || fail "POOL_CPUS deve essere > 0"
[[ "$CONTAINER_EXPIRATION" =~ ^[1-9][0-9]*$ ]] || fail "CONTAINER_EXPIRATION deve essere > 0"
[[ "$JANITOR_INTERVAL" =~ ^[1-9][0-9]*$ ]] || fail "JANITOR_INTERVAL deve essere > 0"

python3 - "$MAB_UCB1_C" <<'PY'
import math
import sys
c = float(sys.argv[1])
if not math.isfinite(c) or c < 0:
    raise SystemExit("MAB_UCB1_C deve essere finito e >= 0")
PY

case "$FORMULA" in
  none)
    [[ -z "$PRIOR_REQUEST" ]] || fail "FORMULA=none non deve avere PRIOR_REQUEST"
    ;;
  difference|ratio)
    [[ -n "$PRIOR_REQUEST" ]] || fail "FORMULA=$FORMULA richiede PRIOR_REQUEST"
    [[ -f "$PRIOR_REQUEST" ]] || fail "PRIOR_REQUEST non trovato: $PRIOR_REQUEST"
    [[ -n "$REWARD_PRIOR_WEIGHT" ]] || fail "REWARD_PRIOR_WEIGHT mancante"
    [[ -n "$EXPLORATION_PRIOR_WEIGHT" ]] || fail "EXPLORATION_PRIOR_WEIGHT mancante"

    python3 - \
      "$PRIOR_REQUEST" \
      "$TARGET_FUNCTION" \
      "$FORMULA" \
      "$REWARD_PRIOR_WEIGHT" \
      "$EXPLORATION_PRIOR_WEIGHT" <<'PY'
import json
import math
import sys

path, target, mode = sys.argv[1:4]
wr = float(sys.argv[4])
we = float(sys.argv[5])

if not math.isfinite(wr) or not (0 < wr <= 1):
    raise SystemExit("wR deve essere in (0,1]")
if not math.isfinite(we) or we <= 0:
    raise SystemExit("wE deve essere finito e > 0")

doc = json.load(open(path, encoding="utf-8"))
if doc.get("target_function_name") != target:
    raise SystemExit(
        f"target nel prior inatteso: {doc.get('target_function_name')} != {target}"
    )

prior = doc.get("prior") or {}
if prior.get("policy") != "UCB1Decoupled":
    raise SystemExit(f"policy prior inattesa: {prior.get('policy')}")

cfg = prior.get("config") or {}
actual_wr = float(cfg.get("reward_observation_weight", -1))
actual_we = float(cfg.get("exploration_observation_weight", -1))
if not math.isclose(actual_wr, wr, rel_tol=0, abs_tol=1e-12):
    raise SystemExit(f"wR prior={actual_wr}, atteso={wr}")
if not math.isclose(actual_we, we, rel_tol=0, abs_tol=1e-12):
    raise SystemExit(f"wE prior={actual_we}, atteso={we}")

anchor = cfg.get("ucb1_reference_anchor") or {}
if anchor.get("mode", "difference") != mode:
    raise SystemExit(
        f"formula prior={anchor.get('mode')}, attesa={mode}"
    )
if anchor.get("reference_arm") != "amd64":
    raise SystemExit(
        f"reference arm inatteso: {anchor.get('reference_arm')}"
    )

arms = set((prior.get("arms") or {}).keys())
if arms != {"amd64", "arm64"}:
    raise SystemExit(f"arms prior inattesi: {sorted(arms)}")

print(
    f"PRIOR PREFLIGHT PASS mode={mode} wR={actual_wr} wE={actual_we}"
)
PY
    ;;
  *)
    fail "FORMULA deve essere none, difference oppure ratio"
    ;;
esac

case "$OPTIMAL_ARM" in
  amd64|arm64) ;;
  *) fail "OPTIMAL_ARM deve essere amd64 oppure arm64" ;;
esac

for port in "$AMD64_PORT" "$ARM64_PORT" "$LB_PORT"; do
  if curl -sS --max-time 1 -o /dev/null "http://127.0.0.1:${port}/status" 2>/dev/null; then
    fail "porta API già occupata: $port"
  fi
done

echo "============================================================"
echo "UCB1 MEMORY LOCAL SINGLE-RUN"
echo "============================================================"
echo "run_id:               $RUN_ID"
echo "config_id:            $CONFIG_ID"
echo "target:               $TARGET_FUNCTION"
echo "formula:              $FORMULA"
echo "c:                    $MAB_UCB1_C"
if [[ "$FORMULA" != "none" ]]; then
  echo "wR:                   $REWARD_PRIOR_WEIGHT"
  echo "wE:                   $EXPLORATION_PRIOR_WEIGHT"
  echo "prior:                $PRIOR_REQUEST"
fi
echo "requests:             $REQUESTS"
echo "request concurrency:  $REQUEST_CONCURRENCY"
echo "MaxConcurrency:       1"
echo "container memory:     ${CONTAINER_MEMORY_MB} MiB"
echo "expiration:           ${CONTAINER_EXPIRATION}s"
echo "janitor interval:     ${JANITOR_INTERVAL}s"
echo "optimal arm metadata: $OPTIMAL_ARM"
echo "prewarm target:       DISABLED"
echo

echo "ATTENZIONE: amd64 e arm64 sono ring LOGICI sullo stesso host fisico."
echo "Questo test valida pipeline e accounting, NON confronti prestazionali."
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
registry.area: UCB1_MEMORY_LOCAL
registry.node.id: memory-amd64-${STAMP}
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
registry.area: UCB1_MEMORY_LOCAL
registry.node.id: memory-arm64-${STAMP}
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
registry.area: UCB1_MEMORY_LOCAL
lb.arch_awareness: true
lb.mode: MAB
mab.policy: UCB1Decoupled
mab.ucb1.c: ${MAB_UCB1_C}
mab.cold_start.mode: skip
mab.transfer.control.enabled: true
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

echo "[setup] Avvio nodo amd64 logico..."
SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1 \
  bin/serverledge "$AMD64_CONF" >"$AMD64_LOG" 2>&1 &
amd64_pid="$!"
PIDS+=("$amd64_pid")
wait_http "http://127.0.0.1:${AMD64_PORT}/status" "$amd64_pid" "$AMD64_LOG"

echo "[setup] Avvio nodo arm64 logico..."
SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1 \
  bin/serverledge "$ARM64_CONF" >"$ARM64_LOG" 2>&1 &
arm64_pid="$!"
PIDS+=("$arm64_pid")
wait_http "http://127.0.0.1:${ARM64_PORT}/status" "$arm64_pid" "$ARM64_LOG"

echo "[setup] Avvio LB UCB1Decoupled..."
bin/lb "$LB_CONF" >"$LB_LOG" 2>&1 &
lb_pid="$!"
PIDS+=("$lb_pid")
wait_http "http://127.0.0.1:${LB_PORT}/status" "$lb_pid" "$LB_LOG"

sleep 2

echo "[setup] Registro target $TARGET_FUNCTION senza prewarm..."
bin/serverledge-cli create \
  -f "$TARGET_FUNCTION" \
  --memory "$CONTAINER_MEMORY_MB" \
  --src examples/sleeper.py \
  --runtime python314 \
  --handler "sleeper.handler" \
  --max_concurrency 1 \
  --cpu 1 \
  --input "n:Int" \
  -H 127.0.0.1 \
  -P "$AMD64_PORT" \
  >"$LOG_DIR/create.json"

sleep 1

if grep -h \
  '\[CONTAINER_LIFECYCLE\].*event=container_created' \
  "$AMD64_LOG" "$ARM64_LOG" \
  | grep -F "function=\"${TARGET_FUNCTION}\"" \
  >/dev/null 2>&1; then
  fail "container target creato prima della misura: prewarm/side effect inatteso"
fi

echo "[pre-measure] zero target container: PASS"

if [[ "$FORMULA" != "none" ]]; then
  echo "[transfer] Applico prior materializzato..."

  HTTP_CODE="$(curl -sS --max-time 30 \
    -o "$CONTROL_RESPONSE" \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data-binary "@$PRIOR_REQUEST" \
    "http://127.0.0.1:${LB_PORT}/mab/transfer/initialize-materialized")"

  [[ "$HTTP_CODE" == "200" ]] || {
    cat "$CONTROL_RESPONSE" >&2 || true
    fail "materialized transfer HTTP $HTTP_CODE"
  }

  python3 - "$PRIOR_REQUEST" "$CONTROL_RESPONSE" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
response = json.load(open(sys.argv[2], encoding="utf-8"))

if response.get("transfer_attempted") is not True:
    raise SystemExit("transfer_attempted != true")
if response.get("transfer_applied") is not True:
    raise SystemExit("transfer_applied != true")
if response.get("initialization_source") != "materialized_prior":
    raise SystemExit("initialization_source != materialized_prior")
if response.get("target_function_name") != request.get("target_function_name"):
    raise SystemExit("target mismatch")
if response.get("prior") != request.get("prior"):
    raise SystemExit("returned prior differs from requested prior")
print("MATERIALIZED PRIOR APPLY: PASS")
PY
else
  cat > "$LOG_DIR/no-transfer.json" <<EOF2
{
  "config_id": "${CONFIG_ID}",
  "target_function": "${TARGET_FUNCTION}",
  "policy": "UCB1Decoupled",
  "c": ${MAB_UCB1_C},
  "prior_applied": false
}
EOF2
fi

if grep -h \
  '\[CONTAINER_LIFECYCLE\].*event=container_created' \
  "$AMD64_LOG" "$ARM64_LOG" \
  | grep -F "function=\"${TARGET_FUNCTION}\"" \
  >/dev/null 2>&1; then
  fail "applicare il prior ha creato un container target"
fi

echo "[pre-measure] prior/noTL non ha creato target container: PASS"

echo "[measure] Lancio $REQUESTS richieste con concorrenza massima $REQUEST_CONCURRENCY..."

invoke_one() {
  local index="$1"
  bin/serverledge-cli invoke \
    -f "$TARGET_FUNCTION" \
    -p "n:${SLEEP_SECONDS}" \
    --return_output \
    -H 127.0.0.1 \
    -P "$LB_PORT" \
    >"$LOG_DIR/requests/request-$(printf '%03d' "$index").json" \
    2>&1
}

running=0
pids=()
indices=()

for index in $(seq 1 "$REQUESTS"); do
  invoke_one "$index" &
  pids+=("$!")
  indices+=("$index")
  running=$((running + 1))

  if (( running == REQUEST_CONCURRENCY || index == REQUESTS )); then
    batch_failed=0
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then
        echo "[FAIL] request ${indices[$j]} fallita" >&2
        batch_failed=1
      fi
    done
    (( batch_failed == 0 )) || fail "una o più richieste sono fallite"
    pids=()
    indices=()
    running=0
  fi
done

for index in $(seq 1 "$REQUESTS"); do
  request_file="$LOG_DIR/requests/request-$(printf '%03d' "$index").json"
  grep -q '"Success"[[:space:]]*:[[:space:]]*true' "$request_file" \
    || fail "request $index non contiene Success=true"
done

echo "[measure] tutte le richieste: PASS"

RESOLVED_COUNT="$(grep -c "event=decision_resolved.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null || true)"
FALLBACK_COUNT="$(grep "event=decision_resolved.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null | grep -c 'fallback=true' || true)"

[[ "$RESOLVED_COUNT" == "$REQUESTS" ]] \
  || fail "decision_resolved=$RESOLVED_COUNT, atteso=$REQUESTS"
[[ "$FALLBACK_COUNT" == "0" ]] \
  || fail "fallback inattesi nel local smoke: $FALLBACK_COUNT"

echo "[measure] decision_resolved=$RESOLVED_COUNT fallback=0: PASS"

echo "[measure] Attendo distruzione naturale dei container..."
WAIT_SECONDS=$((CONTAINER_EXPIRATION + JANITOR_INTERVAL + 4))
sleep "$WAIT_SECONDS"

cat "$AMD64_LOG" "$ARM64_LOG" > "$LOG_DIR/node-lifecycle-combined.log"

python3 - \
  "$LOG_DIR/node-lifecycle-combined.log" \
  "$TARGET_FUNCTION" \
  "$CONFIG_ID" \
  "$FORMULA" \
  "$MAB_UCB1_C" \
  "${REWARD_PRIOR_WEIGHT:-}" \
  "${EXPLORATION_PRIOR_WEIGHT:-}" \
  "$REQUESTS" \
  "$REQUEST_CONCURRENCY" \
  "$RESOLVED_COUNT" \
  "$FALLBACK_COUNT" \
  "$OPTIMAL_ARM" \
  "$SUMMARY_JSON" \
  "$SUMMARY_CSV" <<'PY'
import csv
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

(
    log_path,
    target,
    config_id,
    formula,
    c,
    wr,
    we,
    requests,
    concurrency,
    resolved_count,
    fallback_count,
    optimal_arm,
    summary_path,
    csv_path,
) = sys.argv[1:]

prefix = "[CONTAINER_LIFECYCLE]"
created = {}
destroyed = {}
failed_destroy = []
duplicates = []

for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
    if prefix not in line:
        continue
    payload = line.split(prefix, 1)[1].strip()
    fields = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    if fields.get("function") != target:
        continue

    event = fields.get("event")
    cid = fields.get("container_id")
    if not cid:
        continue

    if event == "container_created":
        if cid in created:
            duplicates.append(f"duplicate create {cid}")
        created[cid] = fields
    elif event == "container_destroyed":
        if cid in destroyed:
            duplicates.append(f"duplicate destroy {cid}")
        destroyed[cid] = fields
    elif event == "container_destroy_failed":
        failed_destroy.append(fields)

if not created:
    raise SystemExit("nessun container_created target trovato")
if failed_destroy:
    raise SystemExit(f"container_destroy_failed presenti: {len(failed_destroy)}")
if duplicates:
    raise SystemExit("; ".join(duplicates))

created_ids = set(created)
destroyed_ids = set(destroyed)
missing_destroy = sorted(created_ids - destroyed_ids)
unmatched_destroy = sorted(destroyed_ids - created_ids)
if missing_destroy or unmatched_destroy:
    raise SystemExit(
        f"pairing fallito missing_destroy={missing_destroy} unmatched_destroy={unmatched_destroy}"
    )

rows = []
by_arm_mib_s = defaultdict(float)
by_arm_containers = Counter()
reason_counts = Counter()
prewarm_count = 0

for cid in sorted(created_ids):
    c_evt = created[cid]
    d_evt = destroyed[cid]

    if c_evt.get("creation_kind") == "prewarm":
        prewarm_count += 1

    c_ts = int(c_evt["ts"])
    d_ts = int(d_evt["ts"])
    lifetime_ms = d_ts - c_ts
    if lifetime_ms <= 0:
        raise SystemExit(f"lifetime non positivo per {cid}: {lifetime_ms}")

    memory_mb = int(c_evt["memory_mb"])
    arm = c_evt.get("machine_tag", "")
    if arm not in {"amd64", "arm64"}:
        raise SystemExit(f"machine_tag inatteso per {cid}: {arm}")
    if d_evt.get("machine_tag") != arm:
        raise SystemExit(f"machine_tag create/destroy mismatch per {cid}")
    if d_evt.get("function") != target:
        raise SystemExit(f"function create/destroy mismatch per {cid}")

    reason = d_evt.get("reason", "")
    reason_counts[reason] += 1

    memory_mib_s = memory_mb * lifetime_ms / 1000.0
    memory_gib_s = memory_mib_s / 1024.0

    by_arm_mib_s[arm] += memory_mib_s
    by_arm_containers[arm] += 1

    rows.append({
        "container_id": cid,
        "machine_tag": arm,
        "memory_mb": memory_mb,
        "created_ts_ms": c_ts,
        "destroyed_ts_ms": d_ts,
        "lifetime_ms": lifetime_ms,
        "memory_MiB_s": memory_mib_s,
        "memory_GiB_s": memory_gib_s,
        "creation_kind": c_evt.get("creation_kind", ""),
        "destroy_reason": reason,
    })

if prewarm_count != 0:
    raise SystemExit(f"trovati {prewarm_count} container target prewarm")

if any(reason != "expired" for reason in reason_counts):
    raise SystemExit(f"reason di distruzione inattese: {dict(reason_counts)}")

total_mib_s = sum(row["memory_MiB_s"] for row in rows)
wrong_arm = "arm64" if optimal_arm == "amd64" else "amd64"
wrong_mib_s = by_arm_mib_s.get(wrong_arm, 0.0)
optimal_mib_s = by_arm_mib_s.get(optimal_arm, 0.0)

with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "schema_version": 1,
    "config_id": config_id,
    "target_function": target,
    "formula": formula,
    "c": float(c),
    "reward_prior_weight": None if wr == "" else float(wr),
    "exploration_prior_weight": None if we == "" else float(we),
    "requests": int(requests),
    "request_concurrency": int(concurrency),
    "decision_resolved_count": int(resolved_count),
    "fallback_count": int(fallback_count),
    "optimal_arm_metadata": optimal_arm,
    "wrong_arm_metadata": wrong_arm,
    "created_containers": len(created_ids),
    "destroyed_containers": len(destroyed_ids),
    "pairing": "PASS",
    "prewarm_target_containers": prewarm_count,
    "destroy_reasons": dict(reason_counts),
    "containers_by_arm": dict(by_arm_containers),
    "memory_MiB_s_by_arm": dict(by_arm_mib_s),
    "memory_GiB_s_by_arm": {
        arm: value / 1024.0 for arm, value in by_arm_mib_s.items()
    },
    "total_memory_MiB_s": total_mib_s,
    "total_memory_GiB_s": total_mib_s / 1024.0,
    "optimal_arm_memory_GiB_s": optimal_mib_s / 1024.0,
    "wrong_arm_memory_GiB_s": wrong_mib_s / 1024.0,
}

Path(summary_path).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAIRING PASS")
print(f"created_containers={len(created_ids)}")
print(f"destroyed_containers={len(destroyed_ids)}")
print(f"containers_by_arm={dict(by_arm_containers)}")
print(f"memory_GiB_s_by_arm={summary['memory_GiB_s_by_arm']}")
print(f"total_memory_GiB_s={summary['total_memory_GiB_s']:.6f}")
print(f"wrong_arm_memory_GiB_s={summary['wrong_arm_memory_GiB_s']:.6f}")
PY

echo
echo "============================================================"
echo "UCB1 MEMORY LOCAL SINGLE-RUN: PASS"
echo "============================================================"
cat "$SUMMARY_JSON"
