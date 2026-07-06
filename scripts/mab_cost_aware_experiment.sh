#!/usr/bin/env bash
set -euo pipefail

# Cost-aware MAB experiment for Serverledge.
#
# Goal:
#   Test whether the MAB reward is affected by node-declared cost_factor
#   and energy_factor at machine_tag/ring granularity.
#
# This script is intentionally separate from mab_dynamic_nodes_experiment.sh.
# The dynamic scripts validate runtime discovery of new arms.
# This script validates cost-aware reward and tag/capability selection.
#
# Usage examples:
#
#   # UCB1, no cost pressure
#   MAB_POLICY=UCB1 MAB_COST_WEIGHT=0.0 MAB_ENERGY_WEIGHT=0.0 \
#     bash scripts/mab_cost_aware_experiment.sh
#
#   # UCB1, cost-aware
#   MAB_POLICY=UCB1 MAB_COST_WEIGHT=0.6 MAB_ENERGY_WEIGHT=0.2 \
#     bash scripts/mab_cost_aware_experiment.sh
#
#   # LinUCB, cost-aware
#   MAB_POLICY=LinUCB MAB_COST_WEIGHT=0.6 MAB_ENERGY_WEIGHT=0.2 \
#     MAB_LINUCB_ALPHA=0.1 MAB_LINUCB_LAMBDA=0.7 \
#     bash scripts/mab_cost_aware_experiment.sh
#
#   # GPU-only function: only nodes exposing gpu/nvidia should be compatible
#   FUNCTION_TAG_PATTERN="gpu,nvidia" MAB_COST_WEIGHT=0.6 MAB_ENERGY_WEIGHT=0.2 \
#     bash scripts/mab_cost_aware_experiment.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REQUESTS="${REQUESTS:-80}"
SLEEP_BETWEEN_REQUESTS="${SLEEP_BETWEEN_REQUESTS:-0.2}"
LB_REFRESH_INTERVAL="${LB_REFRESH_INTERVAL:-3}"

MAB_POLICY="${MAB_POLICY:-UCB1}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"
MAB_LINUCB_ALPHA="${MAB_LINUCB_ALPHA:-0.1}"
MAB_LINUCB_LAMBDA="${MAB_LINUCB_LAMBDA:-0.7}"

MAB_COST_WEIGHT="${MAB_COST_WEIGHT:-0.0}"
MAB_ENERGY_WEIGHT="${MAB_ENERGY_WEIGHT:-0.0}"

FUNCTION_TAG_PATTERN="${FUNCTION_TAG_PATTERN:-}"
FUNCTION_NAME="${FUNCTION_NAME:-mab_cost_aware_$(date +%s)}"

CREATE_PORT="${CREATE_PORT:-1324}"
LB_PORT="${LB_PORT:-8080}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SAFE_POLICY="$(echo "$MAB_POLICY" | tr '[:upper:]' '[:lower:]')"
LOG_DIR="${LOG_DIR:-logs/mab_cost_aware_${SAFE_POLICY}_${RUN_ID}}"
mkdir -p "$LOG_DIR"

LB_CONF="$LOG_DIR/lb-cost-aware-conf.yaml"

cat > "$LB_CONF" <<EOF
etcd.address: 127.0.0.1:2379
api.port: ${LB_PORT}
registry.area: ROME
lb.arch_awareness: true
lb.mode: MAB
mab.policy: ${MAB_POLICY}
mab.ucb1.c: ${MAB_UCB1_C}
mab.linucb.alpha: ${MAB_LINUCB_ALPHA}
mab.linucb.lambda: ${MAB_LINUCB_LAMBDA}
mab.cost.weight: ${MAB_COST_WEIGHT}
mab.energy.weight: ${MAB_ENERGY_WEIGHT}
lb.replicas: 128
lb.refresh_interval: ${LB_REFRESH_INTERVAL}
EOF

# Generate temporary node configs for the experiment.
# All nodes run locally, so physical architecture is still runtime.GOARCH.
# machine_tag expresses the logical class/ring used by the MAB.
NODE_CONFS=(
  "$LOG_DIR/node-x86-tiny.yaml"
  "$LOG_DIR/node-x86-large.yaml"
  "$LOG_DIR/node-x86-tiny-gpu-nvidia.yaml"
  "$LOG_DIR/node-arm-tiny.yaml"
)

cat > "${NODE_CONFS[0]}" <<EOF
etcd.address: 127.0.0.1:2379
api.port: 1324
registry.area: ROME
registry.udp.port: 9877
node.machine_tag: x86-tiny
node.cost_factor: 0.4
node.energy_factor: 1.0
container.pool.memory: 512
EOF

cat > "${NODE_CONFS[1]}" <<EOF
etcd.address: 127.0.0.1:2379
api.port: 1325
registry.area: ROME
registry.udp.port: 9878
node.machine_tag: x86-large
node.cost_factor: 1.2
node.energy_factor: 1.2
container.pool.memory: 512
EOF

cat > "${NODE_CONFS[2]}" <<EOF
etcd.address: 127.0.0.1:2379
api.port: 1326
registry.area: ROME
registry.udp.port: 9879
node.machine_tag: x86-tiny/gpu/nvidia
node.cost_factor: 1.6
node.energy_factor: 1.8
container.pool.memory: 512
EOF

cat > "${NODE_CONFS[3]}" <<EOF
etcd.address: 127.0.0.1:2379
api.port: 1327
registry.area: ROME
registry.udp.port: 9880
node.machine_tag: arm-tiny
node.cost_factor: 0.4
node.energy_factor: 0.6
container.pool.memory: 512
EOF

PIDS=()

cleanup() {
  echo
  echo "[cleanup] Stopping background processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done

  bash scripts/stop-etcd.sh >/dev/null 2>&1 || true

  echo "[cleanup] Logs saved in: $LOG_DIR"
}
trap cleanup EXIT INT TERM

echo "[setup] Building project..."
make

echo "[setup] Starting etcd..."
bash scripts/start-etcd.sh | tee "$LOG_DIR/etcd.log"

echo "[setup] Starting cost-profiled nodes:"
for conf in "${NODE_CONFS[@]}"; do
  log_name="$(basename "$conf" .yaml)"
  echo "  - $conf"
  bin/serverledge "$conf" > "$LOG_DIR/${log_name}.log" 2>&1 &
  PIDS+=("$!")
done

sleep 5

echo "[setup] Starting LB in MAB cost-aware mode with config $LB_CONF"
bin/lb "$LB_CONF" > "$LOG_DIR/lb.log" 2>&1 &
PIDS+=("$!")

sleep 5

CREATE_ARGS=(
  create
  -f "$FUNCTION_NAME"
  --memory 128
  --src examples/hello.py
  --runtime python314
  --handler "hello.handler"
  -H localhost
  -P "$CREATE_PORT"
)

if [[ -n "$FUNCTION_TAG_PATTERN" ]]; then
  CREATE_ARGS+=(--tag_pattern "$FUNCTION_TAG_PATTERN")
fi

echo "[setup] Creating function $FUNCTION_NAME on port $CREATE_PORT"
echo "[setup] Function tag_pattern: ${FUNCTION_TAG_PATTERN:-<none>}"
bin/serverledge-cli "${CREATE_ARGS[@]}" | tee "$LOG_DIR/create.json"

echo
echo "[run] Sending $REQUESTS requests..."
for i in $(seq 1 "$REQUESTS"); do
  echo "[request][$i/$REQUESTS]" | tee -a "$LOG_DIR/invocations.jsonl"
  bin/serverledge-cli invoke -f "$FUNCTION_NAME" \
    -p "name:World" \
    --return_output \
    -H localhost -P "$LB_PORT" | tee -a "$LOG_DIR/invocations.jsonl" || true
  echo >> "$LOG_DIR/invocations.jsonl"
  sleep "$SLEEP_BETWEEN_REQUESTS"
done

echo
echo "[summary] Selected tag occurrences:"
grep -o "selected_tag=[^ ]*" "$LOG_DIR/lb.log" | sort | uniq -c | tee "$LOG_DIR/selected_tags_summary.txt" || true

echo
echo "[summary] Reward breakdown sample:"
grep "event=reward_breakdown" "$LOG_DIR/lb.log" | head -40 | tee "$LOG_DIR/reward_breakdown_sample.log" || true

echo
echo "[summary] Compatibility / MAB relevant log lines:"
grep -E "\[LB\]\[MAB\]|\[MAB\]|compatible_tags|reward_breakdown|BanditManager" "$LOG_DIR/lb.log" | tee "$LOG_DIR/lb_relevant.log" || true

echo
echo "[summary] ExecutionNode occurrences:"
grep -h '"ExecutionNode"' "$LOG_DIR"/invocations.jsonl | sort | uniq -c | tee "$LOG_DIR/execution_nodes_summary.txt" || true

echo
echo "[done] Experiment completed."
echo "[done] Policy: $MAB_POLICY"
echo "[done] Function name: $FUNCTION_NAME"
echo "[done] Function tag_pattern: ${FUNCTION_TAG_PATTERN:-<none>}"
echo "[done] Cost weight: $MAB_COST_WEIGHT"
echo "[done] Energy weight: $MAB_ENERGY_WEIGHT"
echo "[done] Logs: $LOG_DIR"