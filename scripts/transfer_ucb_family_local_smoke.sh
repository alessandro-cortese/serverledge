#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-analysis/bin/python}"
SAMPLES_PER_ARCH="${SAMPLES_PER_ARCH:-10}"
DONOR_MAX_REQUESTS="${DONOR_MAX_REQUESTS:-12}"
MIN_DONOR_OBSERVATIONS="${MIN_DONOR_OBSERVATIONS:-1}"
TRANSFER_MODE="${TRANSFER_MODE:-decoupled}"
REWARD_PRIOR_WEIGHT="${REWARD_PRIOR_WEIGHT:-0.25}"
EXPLORATION_PRIOR_WEIGHT="${EXPLORATION_PRIOR_WEIGHT:-1.0}"
EQUIVALENT_PRIOR_WEIGHT="${EQUIVALENT_PRIOR_WEIGHT:-0.25}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"
MAX_DISTANCE="${MAX_DISTANCE:-1000000000000}"

X86_PORT="${X86_PORT:-1333}"
ARM_PORT="${ARM_PORT:-1334}"
LB_PORT="${LB_PORT:-8080}"
X86_UDP_PORT="${X86_UDP_PORT:-9899}"
ARM_UDP_PORT="${ARM_UDP_PORT:-9900}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="transfer-ucb-family-local-smoke-${TRANSFER_MODE}-${STAMP}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
mkdir -p "$LOG_DIR"

X86_CONF="$LOG_DIR/node-x86.yaml"
ARM_CONF="$LOG_DIR/node-arm64.yaml"
LB_CONF="$LOG_DIR/lb.yaml"
X86_LOG="$LOG_DIR/node-x86.log"
ARM_LOG="$LOG_DIR/node-arm64.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"
RAW_X86="$LOG_DIR/profiling-x86.jsonl"
RAW_ARM="$LOG_DIR/profiling-arm64.jsonl"
DONOR_FUNCTION="donor_decoupled_smoke_${STAMP}"
TARGET_FUNCTION="target_decoupled_smoke_${STAMP}"
DONOR_DIR="$LOG_DIR/donor"
TARGET_DIR="$LOG_DIR/target"
mkdir -p "$DONOR_DIR" "$TARGET_DIR"

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
      kill -INT "$pid" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1 || true
  fi

  echo "[cleanup] Artefatti conservati in: $LOG_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

for cmd in docker curl make python3; do
  command -v "$cmd" >/dev/null 2>&1 || fail "Comando richiesto non trovato: $cmd"
done

[[ -x "$PYTHON_BIN" ]] || fail "Python analysis non trovato: $PYTHON_BIN"
[[ -f analysis/profiling/preprocess.py ]] || fail "preprocess.py non trovato"
[[ -f analysis/profiling/transfer_query.py ]] || fail "transfer_query.py non trovato"
[[ -f analysis/profiling/similarity_selection.py ]] || fail "similarity_selection.py non trovato"

[[ "$SAMPLES_PER_ARCH" =~ ^[0-9]+$ ]] || fail "SAMPLES_PER_ARCH deve essere numerico"
[[ "$DONOR_MAX_REQUESTS" =~ ^[0-9]+$ ]] || fail "DONOR_MAX_REQUESTS deve essere numerico"
[[ "$MIN_DONOR_OBSERVATIONS" =~ ^[0-9]+$ ]] || fail "MIN_DONOR_OBSERVATIONS deve essere numerico"
(( SAMPLES_PER_ARCH >= 10 && SAMPLES_PER_ARCH <= 20 )) || fail "SAMPLES_PER_ARCH deve essere tra 10 e 20"
(( DONOR_MAX_REQUESTS >= 2 )) || fail "DONOR_MAX_REQUESTS deve essere almeno 2"
(( MIN_DONOR_OBSERVATIONS >= 1 )) || fail "MIN_DONOR_OBSERVATIONS deve essere almeno 1"

case "$TRANSFER_MODE" in
  decoupled)
    POLICY="UCB1Decoupled"
    ;;
  coupled)
    POLICY="UCB1"
    ;;
  *)
    fail "TRANSFER_MODE deve essere decoupled oppure coupled"
    ;;
esac

"$PYTHON_BIN" - \
  "$TRANSFER_MODE" \
  "$REWARD_PRIOR_WEIGHT" \
  "$EXPLORATION_PRIOR_WEIGHT" \
  "$EQUIVALENT_PRIOR_WEIGHT" \
  "$MAB_UCB1_C" <<'PY'
import math
import sys

mode = sys.argv[1]
wr, we, w, c = map(float, sys.argv[2:])
if mode == "decoupled":
    for name, value in (("wR", wr), ("wE", we)):
        if not math.isfinite(value) or not (0.0 < value <= 1.0):
            raise SystemExit(f"{name} deve essere in (0,1]")
elif mode == "coupled":
    if not math.isfinite(w) or not (0.0 < w <= 1.0):
        raise SystemExit("Equivalent weight deve essere in (0,1]")
else:
    raise SystemExit("mode non valido")
if not math.isfinite(c) or c < 0.0:
    raise SystemExit("MAB_UCB1_C deve essere finito e >= 0")
PY

for port in "$X86_PORT" "$ARM_PORT" "$LB_PORT"; do
  if curl -sS --max-time 1 -o /dev/null "http://127.0.0.1:${port}/status" 2>/dev/null; then
    fail "La porta API $port sembra già occupata da Serverledge"
  fi
done

echo "============================================================"
echo "PRE-GCP LOCAL END-TO-END — UCB1 FAMILY + ANCHORED PRIOR"
echo "============================================================"
echo "run_id:       $RUN_ID"
echo "donor:        $DONOR_FUNCTION"
echo "target:       $TARGET_FUNCTION"
echo "target ref:   x86 ONLY"
echo "distance:     Manhattan within same cluster"
echo "mode:         $TRANSFER_MODE"
echo "policy:       $POLICY"
if [[ "$TRANSFER_MODE" == "decoupled" ]]; then
  echo "wR:           $REWARD_PRIOR_WEIGHT"
  echo "wE:           $EXPLORATION_PRIOR_WEIGHT"
else
  echo "w:            $EQUIVALENT_PRIOR_WEIGHT"
fi
echo "c:            $MAB_UCB1_C"
echo "x86 logical:  127.0.0.1:$X86_PORT"
echo "arm logical:  127.0.0.1:$ARM_PORT"
echo "LB:           127.0.0.1:$LB_PORT"
echo
echo "ATTENZIONE: i due ring sono logici e girano sulla stessa CPU fisica."
echo "Questo smoke valida integrazione e semantica, NON prestazioni x86-vs-ARM."
echo

echo "[setup] Build..."
make

docker info >/dev/null 2>&1 || fail "Docker non disponibile"
docker rm -f Etcd-server >/dev/null 2>&1 || true
rm -f "$RAW_X86" "$RAW_ARM"

cat > "$X86_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${X86_PORT}
registry.area: TRANSFER_UCB_FAMILY_LOCAL_SMOKE
registry.node.id: smoke-x86-${STAMP}
registry.udp.port: ${X86_UDP_PORT}
node.machine_tag: x86
container.pool.memory: 1024
container.pool.cpus: 4
container.expiration: 600
factory.images.refresh: false
profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${RAW_X86}"
scheduler.queue.capacity: 0
YAML

cat > "$ARM_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${ARM_PORT}
registry.area: TRANSFER_UCB_FAMILY_LOCAL_SMOKE
registry.node.id: smoke-arm64-${STAMP}
registry.udp.port: ${ARM_UDP_PORT}
node.machine_tag: arm64
container.pool.memory: 1024
container.pool.cpus: 4
container.expiration: 600
factory.images.refresh: false
profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${RAW_ARM}"
scheduler.queue.capacity: 0
YAML

cat > "$LB_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${LB_PORT}
registry.area: TRANSFER_UCB_FAMILY_LOCAL_SMOKE
lb.arch_awareness: true
lb.mode: MAB
mab.policy: ${POLICY}
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

wait_http() {
  local url="$1"
  local pid="$2"
  local logfile="$3"
  for _ in $(seq 1 80); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -100 "$logfile" || true
      fail "Processo terminato durante l'avvio: $logfile"
    fi
    if curl -sS --max-time 1 -o /dev/null "$url" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  tail -100 "$logfile" || true
  fail "Timeout attendendo $url"
}

echo "[setup] Avvio nodo x86..."
bin/serverledge "$X86_CONF" >"$X86_LOG" 2>&1 &
X86_PID=$!
PIDS+=("$X86_PID")
wait_http "http://127.0.0.1:${X86_PORT}/status" "$X86_PID" "$X86_LOG"

echo "[setup] Avvio nodo arm64 logico..."
bin/serverledge "$ARM_CONF" >"$ARM_LOG" 2>&1 &
ARM_PID=$!
PIDS+=("$ARM_PID")
wait_http "http://127.0.0.1:${ARM_PORT}/status" "$ARM_PID" "$ARM_LOG"

echo "[setup] Avvio LB..."
bin/lb "$LB_CONF" >"$LB_LOG" 2>&1 &
LB_PID=$!
PIDS+=("$LB_PID")
wait_http "http://127.0.0.1:${LB_PORT}/status" "$LB_PID" "$LB_LOG"

for _ in $(seq 1 30); do
  if grep -q "event=new_machine_tag_discovered tag=x86" "$LB_LOG" 2>/dev/null && \
     grep -q "event=new_machine_tag_discovered tag=arm64" "$LB_LOG" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
grep -q "event=new_machine_tag_discovered tag=x86" "$LB_LOG" || fail "LB non ha scoperto x86"
grep -q "event=new_machine_tag_discovered tag=arm64" "$LB_LOG" || fail "LB non ha scoperto arm64"

echo "[setup] Creo donor e target..."
for function_name in "$DONOR_FUNCTION" "$TARGET_FUNCTION"; do
  bin/serverledge-cli create \
    -f "$function_name" \
    --memory 128 \
    --cpu 1 \
    --max_concurrency 1 \
    --src examples/hello.py \
    --runtime python314 \
    --handler "hello.handler" \
    -H 127.0.0.1 \
    -P "$X86_PORT" \
    >"$LOG_DIR/create-${function_name}.json"
done
sleep 1

DIRECT_BODY="$LOG_DIR/direct-invoke.json"
cat > "$DIRECT_BODY" <<'JSON'
{
  "Params": {"name": "World"},
  "QoSClass": 0,
  "QoSMaxRespT": -1.0,
  "CanDoOffloading": false,
  "Async": false,
  "ReturnOutput": true
}
JSON

direct_invoke() {
  local function_name="$1"
  local port="$2"
  local output="$3"
  curl -fsS --max-time 120 \
    -H 'Content-Type: application/json' \
    --data-binary "@$DIRECT_BODY" \
    "http://127.0.0.1:${port}/invoke/${function_name}" >"$output"
}

prewarm_one() {
  local function_name="$1"
  local port="$2"
  local label="$3"
  bin/serverledge-cli prewarm \
    -f "$function_name" -c 1 \
    -H 127.0.0.1 -P "$port" \
    >"$LOG_DIR/prewarm-${function_name}-${label}.json"
}

filter_samples() {
  local source="$1"
  local function_name="$2"
  local machine_tag="$3"
  local required="$4"
  local output="$5"

  "$PYTHON_BIN" - "$source" "$function_name" "$machine_tag" "$required" "$output" <<'PY'
import json
import sys
from pathlib import Path

source, function_name, machine_tag, required_raw, output = sys.argv[1:]
required = int(required_raw)
source = Path(source)
if not source.is_file():
    raise SystemExit(f"raw profiling dataset mancante: {source}")

selected = []
eligible = 0
for raw in source.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    sample = json.loads(raw)
    if sample.get("function_name") != function_name:
        continue
    if sample.get("machine_tag") != machine_tag:
        continue
    selected.append(sample)
    if (sample.get("eligibility") or {}).get("resource_clustering") is True:
        eligible += 1

if eligible < required:
    raise SystemExit(
        f"{function_name}/{machine_tag}: eligible insufficienti: {eligible} < {required}"
    )

output = Path(output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for sample in selected:
        handle.write(json.dumps(sample, sort_keys=True) + "\n")

print(
    f"[profile] function={function_name} tag={machine_tag} "
    f"selected={len(selected)} eligible={eligible}"
)
PY
}

aggregate_function() {
  local function_name="$1"
  local filtered_dir="$2"
  local output_dir="$3"
  bin/serverledge-profiling aggregate \
    --input-dir "$filtered_dir" \
    --samples "$SAMPLES_PER_ARCH" \
    --output "$output_dir/function-profiles.jsonl" \
    >"$output_dir/aggregate.log"

  bin/serverledge-profiling export-csv \
    --input "$output_dir/function-profiles.jsonl" \
    --experiment-id "$RUN_ID-$function_name" \
    --mean-output "$output_dir/function-profiles-mean.csv" \
    --median-output "$output_dir/function-profiles-median.csv" \
    >"$output_dir/export-csv.log"
}

# ------------------------------------------------------------------
# 1. Donor resource profile: both logical architectures.
# ------------------------------------------------------------------
echo "[donor-profile] Prewarm e profiling su x86 + arm64..."
prewarm_one "$DONOR_FUNCTION" "$X86_PORT" x86
prewarm_one "$DONOR_FUNCTION" "$ARM_PORT" arm64
sleep 1
for i in $(seq 1 "$SAMPLES_PER_ARCH"); do
  direct_invoke "$DONOR_FUNCTION" "$X86_PORT" "$DONOR_DIR/direct-x86-${i}.json"
  direct_invoke "$DONOR_FUNCTION" "$ARM_PORT" "$DONOR_DIR/direct-arm64-${i}.json"
done

mkdir -p "$DONOR_DIR/filtered/x86" "$DONOR_DIR/filtered/arm64"
filter_samples "$RAW_X86" "$DONOR_FUNCTION" x86 "$SAMPLES_PER_ARCH" "$DONOR_DIR/filtered/x86/profiling-samples.jsonl"
filter_samples "$RAW_ARM" "$DONOR_FUNCTION" arm64 "$SAMPLES_PER_ARCH" "$DONOR_DIR/filtered/arm64/profiling-samples.jsonl"
aggregate_function "$DONOR_FUNCTION" "$DONOR_DIR/filtered" "$DONOR_DIR"

# ------------------------------------------------------------------
# 2. Preprocess donor reference profile and build a one-donor fixture.
#    This fixture validates runtime wiring. Actual KMeans quality was already
#    evaluated offline; a separate local smoke below validates Manhattan rank.
# ------------------------------------------------------------------
MODEL_JSON="$DONOR_DIR/preprocess-model.json"
DONOR_PREPROCESSED="$DONOR_DIR/donor-preprocessed.csv"
CATALOG_JSON="$DONOR_DIR/smoke-donor-catalog.json"

"$PYTHON_BIN" analysis/profiling/preprocess.py fit-transform \
  --input "$DONOR_DIR/function-profiles-median.csv" \
  --scaler none \
  --model "$MODEL_JSON" \
  --output "$DONOR_PREPROCESSED" \
  >"$DONOR_DIR/preprocess.log"

"$PYTHON_BIN" - "$DONOR_PREPROCESSED" "$CATALOG_JSON" "$DONOR_FUNCTION" "$RUN_ID" <<'PY'
import csv
import json
import sys
from pathlib import Path
from analysis.profiling import preprocess, transfer_catalog

source_raw, output_raw, donor_name, run_id = sys.argv[1:]
with Path(source_raw).open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

matches = [
    row for row in rows
    if row["function_name"] == donor_name and row["machine_tag"] == "x86"
]
if len(matches) != 1:
    raise SystemExit(f"attesa una sola riga donor x86, trovate {len(matches)}")
row = matches[0]
vector = [float(row[name]) for name in preprocess.FEATURE_NAMES]

catalog = {
    "schema_version": transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION,
    "catalog_run_id": f"smoke-catalog-{run_id}",
    "feature_names": list(preprocess.FEATURE_NAMES),
    "feature_space": {
        "representation": "preprocessed",
        "scaler": "none",
        "distance_candidate": "manhattan",
        "distance_policy_selected": True,
    },
    "clustering": {
        "clustering_run_id": f"smoke-cluster-{run_id}",
        "algorithm": "local-smoke-fixture",
        "aggregation": "median",
        "profile_machine_tag": "x86",
    },
    "donor_policy": {
        "readiness_required": False,
        "noise_is_eligible": False,
        "bandit_prior_materialized": False,
        "architecture_preference_is_bandit_reward": False,
    },
    "summary": {
        "donor_count": 1,
        "eligible_donor_count": 1,
        "ineligible_donor_count": 0,
        "noise_donor_count": 0,
    },
    "donors": [{
        "function_name": donor_name,
        "configured_cpus": float(row["configured_cpus"]),
        "configured_memory_mb": int(row["configured_memory_mb"]),
        "profile_machine_tag": "x86",
        "aggregation": "median",
        "scaler": "none",
        "algorithm": "local-smoke-fixture",
        "cluster_label": 0,
        "is_noise": False,
        "donor_eligible": True,
        "donor_ineligibility_reason": "",
        "architecture_preference": "architecture_independent",
        "arm_vs_x86_delta_percent": 0.0,
        "threshold_percent": 15.0,
        "x86_duration_ms": 0.0,
        "arm_duration_ms": 0.0,
        "feature_vector": vector,
        "bandit_prior": None,
    }],
}
Path(output_raw).write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"catalog={output_raw}")
PY

# ------------------------------------------------------------------
# 3. Donor accumulates REAL reward feedback in the live LB MAB.
# ------------------------------------------------------------------
echo "[donor-mab] Accumulo osservazioni reali del donor..."
x86_updates=0
arm_updates=0
for i in $(seq 1 "$DONOR_MAX_REQUESTS"); do
  bin/serverledge-cli invoke \
    -f "$DONOR_FUNCTION" \
    -p "name:World" \
    --return_output \
    -H 127.0.0.1 -P "$LB_PORT" \
    >"$DONOR_DIR/lb-invoke-${i}.json"
  sleep 0.15

  x86_updates="$(grep -c "event=update_reward.*function=${DONOR_FUNCTION}.*arm=x86 " "$LB_LOG" 2>/dev/null || true)"
  arm_updates="$(grep -c "event=update_reward.*function=${DONOR_FUNCTION}.*arm=arm64 " "$LB_LOG" 2>/dev/null || true)"

  if (( x86_updates >= MIN_DONOR_OBSERVATIONS && arm_updates >= MIN_DONOR_OBSERVATIONS )); then
    break
  fi
done

if (( x86_updates < MIN_DONOR_OBSERVATIONS || arm_updates < MIN_DONOR_OBSERVATIONS )); then
  grep "event=update_reward.*function=${DONOR_FUNCTION}" "$LB_LOG" || true
  fail "Donor real observations insufficienti: x86=$x86_updates arm64=$arm_updates"
fi
echo "[donor-mab] x86=$x86_updates arm64=$arm_updates"

# ------------------------------------------------------------------
# 4. New target profile: x86 ONLY before transfer.
# ------------------------------------------------------------------
echo "[target-profile] Profilazione SOLO x86..."
prewarm_one "$TARGET_FUNCTION" "$X86_PORT" x86
sleep 1
for i in $(seq 1 "$SAMPLES_PER_ARCH"); do
  direct_invoke "$TARGET_FUNCTION" "$X86_PORT" "$TARGET_DIR/direct-x86-${i}.json"
done

mkdir -p "$TARGET_DIR/filtered/x86"
filter_samples "$RAW_X86" "$TARGET_FUNCTION" x86 "$SAMPLES_PER_ARCH" "$TARGET_DIR/filtered/x86/profiling-samples.jsonl"
aggregate_function "$TARGET_FUNCTION" "$TARGET_DIR/filtered" "$TARGET_DIR"

# No target ARM profile is permitted before transfer.
"$PYTHON_BIN" - "$RAW_ARM" "$TARGET_FUNCTION" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
function_name = sys.argv[2]
count = 0
if source.is_file():
    for raw in source.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        sample = json.loads(raw)
        if sample.get("function_name") == function_name and sample.get("machine_tag") == "arm64":
            count += 1
if count != 0:
    raise SystemExit(f"target ha {count} sample ARM inattesi prima del transfer")
print("[target-check] arm64 samples before transfer = 0")
PY

if grep -q "event=select_arm.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null; then
  fail "Target MAB usato prima del transfer"
fi

# Mean of the individual target x86 rewards, NOT -ln(mean duration).
TARGET_REF_MEAN_REWARD="$($PYTHON_BIN - "$TARGET_DIR/filtered/x86/profiling-samples.jsonl" "$TARGET_FUNCTION" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
function_name = sys.argv[2]
rewards = []
for raw in path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    sample = json.loads(raw)
    if sample.get("function_name") != function_name or sample.get("machine_tag") != "x86":
        continue
    if sample.get("warm_start") is not True:
        continue
    if sample.get("execution_succeeded") is not True:
        continue
    if (sample.get("eligibility") or {}).get("performance_analysis") is not True:
        continue
    profile = sample.get("profile") or {}
    if profile.get("valid") is not True or profile.get("exclusive_container") is not True:
        continue
    duration = float((sample.get("timing") or {}).get("duration_ms", 0.0))
    if not math.isfinite(duration) or duration <= 0.0:
        continue
    rewards.append(-math.log(duration))

if not rewards:
    raise SystemExit("nessun reward x86 target valido per reference anchoring")
print(format(statistics.fmean(rewards), ".17g"))
PY
)"
echo "[anchor] target x86 mean reward = $TARGET_REF_MEAN_REWARD"

# ------------------------------------------------------------------
# 5. Query + same-cluster Manhattan donor selection.
# ------------------------------------------------------------------
QUERY_JSON="$TARGET_DIR/transfer-query.json"
SELECTION_JSON="$TARGET_DIR/selection.json"
SELECTION_CSV="$TARGET_DIR/selection.csv"
CONTROL_REQUEST="$TARGET_DIR/control-request.json"
CONTROL_RESPONSE="$TARGET_DIR/control-response.json"

"$PYTHON_BIN" analysis/profiling/transfer_query.py \
  --input "$TARGET_DIR/function-profiles-median.csv" \
  --catalog "$CATALOG_JSON" \
  --model "$MODEL_JSON" \
  --function "$TARGET_FUNCTION" \
  --query-id "query-$RUN_ID" \
  --cluster-label 0 \
  --output "$QUERY_JSON" \
  >"$TARGET_DIR/transfer-query.log"

"$PYTHON_BIN" analysis/profiling/similarity_selection.py \
  --catalog "$CATALOG_JSON" \
  --query "$QUERY_JSON" \
  --run-id "selection-$RUN_ID" \
  --max-distance "$MAX_DISTANCE" \
  --distance manhattan \
  --require-same-cluster \
  --output-json "$SELECTION_JSON" \
  --output-csv "$SELECTION_CSV" \
  >"$TARGET_DIR/similarity.log"

"$PYTHON_BIN" - "$SELECTION_JSON" "$DONOR_FUNCTION" <<'PY'
import json
import sys
from pathlib import Path

doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
if doc.get("status") != "selected":
    raise SystemExit(f"selection status={doc.get('status')} reason={doc.get('reason')}")
if (doc.get("selected_donor") or {}).get("function_name") != expected:
    raise SystemExit("donor selezionato inatteso")
policy = doc.get("selection_policy") or {}
if policy.get("distance") != "manhattan":
    raise SystemExit("selection artifact non usa Manhattan")
if policy.get("require_same_cluster") is not True:
    raise SystemExit("selection artifact non richiede same-cluster")
print(f"[selection] donor={expected} metric=manhattan same_cluster=true")
PY

# ------------------------------------------------------------------
# 6. Build decoupled + reference-anchored control request.
# ------------------------------------------------------------------
"$PYTHON_BIN" - \
  "$SELECTION_JSON" \
  "$TARGET_FUNCTION" \
  "$TRANSFER_MODE" \
  "$REWARD_PRIOR_WEIGHT" \
  "$EXPLORATION_PRIOR_WEIGHT" \
  "$EQUIVALENT_PRIOR_WEIGHT" \
  "$MIN_DONOR_OBSERVATIONS" \
  "$TARGET_REF_MEAN_REWARD" \
  >"$CONTROL_REQUEST" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = sys.argv[2]
mode = sys.argv[3]
wr = float(sys.argv[4])
we = float(sys.argv[5])
w = float(sys.argv[6])
minimum = int(sys.argv[7])
target_ref = float(sys.argv[8])

prior_config = {
    "min_real_observations_per_arm": minimum,
    "ucb1_reference_anchor": {
        "enabled": True,
        "reference_arm": "x86",
        "target_reference_mean_reward": target_ref,
    },
}
if mode == "decoupled":
    prior_config["reward_observation_weight"] = wr
    prior_config["exploration_observation_weight"] = we
elif mode == "coupled":
    prior_config["equivalent_observation_weight"] = w
else:
    raise SystemExit("mode non valido")

json.dump(
    {
        "target_function_name": target,
        "selection_artifact": selection,
        "prior_config": prior_config,
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)
print()
PY

HTTP_CODE="$(curl -sS --max-time 30 \
  -o "$CONTROL_RESPONSE" \
  -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  --data-binary "@$CONTROL_REQUEST" \
  "http://127.0.0.1:${LB_PORT}/mab/transfer/initialize")"

if [[ "$HTTP_CODE" != "200" ]]; then
  cat "$CONTROL_RESPONSE" >&2 || true
  fail "Transfer control API HTTP $HTTP_CODE"
fi

"$PYTHON_BIN" - \
  "$CONTROL_RESPONSE" \
  "$DONOR_FUNCTION" \
  "$TARGET_FUNCTION" \
  "$TRANSFER_MODE" \
  "$REWARD_PRIOR_WEIGHT" \
  "$EXPLORATION_PRIOR_WEIGHT" \
  "$EQUIVALENT_PRIOR_WEIGHT" \
  "$TARGET_REF_MEAN_REWARD" <<'PY'
import json
import math
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_donor = sys.argv[2]
expected_target = sys.argv[3]
mode = sys.argv[4]
wr = float(sys.argv[5])
we = float(sys.argv[6])
w = float(sys.argv[7])
target_ref = float(sys.argv[8])
expected_policy = "UCB1Decoupled" if mode == "decoupled" else "UCB1"

if response.get("target_function_name") != expected_target:
    raise SystemExit("target inatteso nella control response")
if response.get("selected_donor_function_name") != expected_donor:
    raise SystemExit("donor inatteso nella control response")
if response.get("transfer_attempted") is not True or response.get("transfer_applied") is not True:
    raise SystemExit(f"transfer non applicato: {response.get('runtime_reason')}")

prior = response.get("prior") or {}
if prior.get("has_prior") is not True:
    raise SystemExit("weak prior assente")
if prior.get("policy") != expected_policy:
    raise SystemExit(f"prior policy inattesa: {prior.get('policy')}")
config = prior.get("config") or {}
if mode == "decoupled":
    if not math.isclose(float(config.get("reward_observation_weight", -1)), wr, abs_tol=1e-12):
        raise SystemExit("wR inatteso nella prior response")
    if not math.isclose(float(config.get("exploration_observation_weight", -1)), we, abs_tol=1e-12):
        raise SystemExit("wE inatteso nella prior response")
else:
    if not math.isclose(float(config.get("equivalent_observation_weight", -1)), w, abs_tol=1e-12):
        raise SystemExit("equivalent weight inatteso nella prior response")

anchor = config.get("ucb1_reference_anchor") or {}
if anchor.get("enabled") is not True or anchor.get("reference_arm") != "x86":
    raise SystemExit("reference anchor non attivo su x86")
if not math.isclose(float(anchor.get("target_reference_mean_reward")), target_ref, rel_tol=1e-12, abs_tol=1e-12):
    raise SystemExit("target reference mean reward inatteso")

arms = prior.get("arms") or {}
for arm in ("x86", "arm64"):
    arm_prior = arms.get(arm) or {}
    if arm_prior.get("transferred") is not True:
        raise SystemExit(f"prior non trasferito per arm {arm}")
    ucb = arm_prior.get("ucb1") or {}
    if mode == "decoupled":
        if not math.isclose(float(ucb.get("observation_weight", -1)), wr, abs_tol=1e-12):
            raise SystemExit(f"wR inatteso su {arm}")
        if not math.isclose(float(ucb.get("exploration_observation_weight", -1)), we, abs_tol=1e-12):
            raise SystemExit(f"wE inatteso su {arm}")
    else:
        if not math.isclose(float(ucb.get("observation_weight", -1)), w, abs_tol=1e-12):
            raise SystemExit(f"w inatteso su {arm}")
        if not math.isclose(float(ucb.get("exploration_observation_weight", -1)), w, abs_tol=1e-12):
            raise SystemExit(f"exploration coupled weight inatteso su {arm}")

x86_mean = float((arms["x86"].get("ucb1") or {}).get("mean_reward"))
if not math.isclose(x86_mean, target_ref, rel_tol=1e-12, abs_tol=1e-12):
    raise SystemExit(
        f"reference anchoring errato: prior x86={x86_mean}, target_ref={target_ref}"
    )

print(
    f"[transfer] applied=true policy={expected_policy} donor={expected_donor} "
    f"mode={mode} anchor=x86"
)
PY

# ------------------------------------------------------------------
# 7. Normal target request through LB, then verify decision + feedback.
# ------------------------------------------------------------------
# The target was profiled only on x86 before transfer. Now that the transfer
# has already been applied, prewarm the logical ARM ring as well so the first
# post-transfer MAB request is warm whichever arm is selected. This does not
# add target ARM profiling evidence to the prior and does not pass through MAB.
echo "[target-mab] Prewarm arm64 DOPO il transfer (runtime-only, no prior evidence)..."
prewarm_one "$TARGET_FUNCTION" "$ARM_PORT" arm64-after-transfer
sleep 1

echo "[target-mab] Prima richiesta target attraverso LB dopo il transfer..."
PRE_UPDATES="$(grep -c "event=update_reward.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null || true)"

bin/serverledge-cli invoke \
  -f "$TARGET_FUNCTION" \
  -p "name:World" \
  --return_output \
  -H 127.0.0.1 -P "$LB_PORT" \
  >"$TARGET_DIR/lb-invoke-after-transfer.json"

sleep 0.3

POST_UPDATES="$(grep -c "event=update_reward.*function=${TARGET_FUNCTION}" "$LB_LOG" 2>/dev/null || true)"
(( POST_UPDATES > PRE_UPDATES )) || fail "Nessun real update_reward target dopo la richiesta LB"

if [[ "$TRANSFER_MODE" == "decoupled" ]]; then
  grep -q "event=runtime_transfer.*target_function=${TARGET_FUNCTION}.*policy=UCB1Decoupled.*applied=true" "$LB_LOG" || \
    fail "runtime_transfer UCB1Decoupled applied=true non trovato"

  grep -q "event=select_arm.*policy=UCB1Decoupled.*function=${TARGET_FUNCTION}" "$LB_LOG" || \
    fail "select_arm UCB1Decoupled target non trovato"

  WR_FMT="$($PYTHON_BIN - "$REWARD_PRIOR_WEIGHT" <<'PY'
import sys
print(f"{float(sys.argv[1]):.6f}")
PY
)"
  WE_FMT="$($PYTHON_BIN - "$EXPLORATION_PRIOR_WEIGHT" <<'PY'
import sys
print(f"{float(sys.argv[1]):.6f}")
PY
)"

  grep -q "event=arm_score.*policy=UCB1Decoupled.*function=${TARGET_FUNCTION}.*reward_prior_weight=${WR_FMT}.*exploration_prior_weight=${WE_FMT}" "$LB_LOG" || \
    fail "arm_score target non espone wR=$WR_FMT e wE=$WE_FMT"

  grep -q "event=update_reward.*policy=UCB1Decoupled.*function=${TARGET_FUNCTION}" "$LB_LOG" || \
    fail "update_reward target UCB1Decoupled non trovato"
else
  grep -q "event=runtime_transfer.*target_function=${TARGET_FUNCTION}.*policy=UCB1.*applied=true" "$LB_LOG" || \
    fail "runtime_transfer UCB1 applied=true non trovato"

  grep -q "event=select_arm.*policy=UCB1.*function=${TARGET_FUNCTION}" "$LB_LOG" || \
    fail "select_arm UCB1 target non trovato"

  W_FMT="$($PYTHON_BIN - "$EQUIVALENT_PRIOR_WEIGHT" <<'PY'
import sys
print(f"{float(sys.argv[1]):.6f}")
PY
)"
  grep -q "event=arm_score.*policy=UCB1.*function=${TARGET_FUNCTION}.*prior_observation_weight=${W_FMT}" "$LB_LOG" || \
    fail "arm_score coupled target non espone w=$W_FMT"

  grep -q "event=update_reward.*policy=UCB1.*function=${TARGET_FUNCTION}" "$LB_LOG" || \
    fail "update_reward target UCB1 non trovato"
fi

echo
echo "============================================================"
echo "PRE-GCP LOCAL END-TO-END SMOKE: PASS"
echo "============================================================"
echo "✓ build + due ring logici + LB policy=$POLICY"
echo "✓ donor profilato su entrambi i ring e con real MAB feedback"
echo "✓ target profilato SOLO su x86 prima del transfer"
echo "✓ clustering-feature aggregation: median"
echo "✓ query marcata cluster=0 e donor selection same-cluster"
echo "✓ donor ranking artifact: Manhattan"
echo "✓ target reference mean reward = mean[-ln(duration_ms)]"
echo "✓ reference-anchored prior applicato via control API"
if [[ "$TRANSFER_MODE" == "decoupled" ]]; then
  echo "✓ wR=$REWARD_PRIOR_WEIGHT e wE=$EXPLORATION_PRIOR_WEIGHT"
else
  echo "✓ coupled equivalent weight=$EQUIVALENT_PRIOR_WEIGHT"
fi
echo "✓ c=$MAB_UCB1_C"
echo "✓ normale richiesta target -> arm_score -> select_arm -> update_reward"
echo
echo "NON VALIDATO localmente: vantaggio prestazionale reale x86-vs-ARM."
echo "Artefatti: $LOG_DIR"
