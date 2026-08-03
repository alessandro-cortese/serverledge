#!/usr/bin/env bash
set -euo pipefail

# Dynamic LinUCB experiment for Serverledge.
#
# Scenario:
#   1. start etcd
#   2. start only first-architecture nodes, e.g. x86
#   3. start LB in MAB mode with LinUCB policy
#   4. create a function WITHOUT tag_pattern, so LinUCB can choose the ring
#   5. send requests in phase 1
#   6. start a second-architecture node, e.g. arm
#   7. wait for LB refresh so AddTarget() discovers it and AddArmToAll() adds the new MAB arm
#   8. send requests in phase 2
#   9. keep logs for analysis
#
# Usage examples:
#   bash scripts/mab_dynamic_nodes_experiment_linucb.sh
#   PRE_REQUESTS=10 POST_REQUESTS=60 SECOND_NODE_CONF=node1-conf.yaml bash scripts/mab_dynamic_nodes_experiment_linucb.sh
#   MAB_LINUCB_ALPHA=0.2 bash scripts/mab_dynamic_nodes_experiment_linucb.sh
#
# Notes:
#   - This script demonstrates dynamic discovery and LinUCB arm activation.
#   - On a single local machine with simulated tags, the script proves that
#     the new arm is discovered and becomes selectable.
#   - It does not prove a real hardware advantage between x86 and arm, because
#     both tags run locally on the same physical architecture.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRE_REQUESTS="${PRE_REQUESTS:-15}"
POST_REQUESTS="${POST_REQUESTS:-50}"
SLEEP_BETWEEN_REQUESTS="${SLEEP_BETWEEN_REQUESTS:-0.2}"
LB_REFRESH_INTERVAL="${LB_REFRESH_INTERVAL:-3}"

# LinUCB parameters.
MAB_LINUCB_ALPHA="${MAB_LINUCB_ALPHA:-0.1}"

MAB_FALLBACK_PENALTY="${MAB_FALLBACK_PENALTY:--12.0}"

# First phase: start only these nodes.
FIRST_NODE_CONFS=(${FIRST_NODE_CONFS:-node2-conf.yaml node6-conf.yaml})

# Second phase: add this node later.
SECOND_NODE_CONF="${SECOND_NODE_CONF:-node1-conf.yaml}"

# Function used for the experiment.
FUNCTION_NAME="${FUNCTION_NAME:-mab_dynamic_linucb_$(date +%s)}"
CREATE_PORT="${CREATE_PORT:-1324}"
LB_PORT="${LB_PORT:-8080}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/mab_dynamic_linucb_${RUN_ID}}"
mkdir -p "$LOG_DIR"

LB_CONF="$LOG_DIR/lb-mab-linucb-conf.yaml"

cat > "$LB_CONF" <<EOF
etcd.address: 127.0.0.1:2379
api.port: ${LB_PORT}
registry.area: ROME
lb.arch_awareness: true
lb.mode: MAB
mab.policy: LinUCB
mab.fallback.penalty: ${MAB_FALLBACK_PENALTY}
mab.linucb.alpha: ${MAB_LINUCB_ALPHA}
lb.replicas: 128
lb.refresh_interval: ${LB_REFRESH_INTERVAL}
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

echo "[phase 1] Starting initial nodes only: ${FIRST_NODE_CONFS[*]}"
for conf in "${FIRST_NODE_CONFS[@]}"; do
  log_name="$(basename "$conf" .yaml)"
  bin/serverledge "$conf" > "$LOG_DIR/${log_name}.log" 2>&1 &
  PIDS+=("$!")
done

sleep 5

echo "[setup] Starting LB in MAB LinUCB mode with config $LB_CONF"
bin/lb "$LB_CONF" > "$LOG_DIR/lb.log" 2>&1 &
PIDS+=("$!")

sleep 5

echo "[setup] Creating function $FUNCTION_NAME on port $CREATE_PORT"
bin/serverledge-cli create -f "$FUNCTION_NAME" \
  --memory 128 \
  --src examples/hello.py \
  --runtime python314 \
  --handler "hello.handler" \
  -H localhost -P "$CREATE_PORT" | tee "$LOG_DIR/create.json"

echo
echo "[phase 1] Sending $PRE_REQUESTS requests BEFORE adding second architecture..."
for i in $(seq 1 "$PRE_REQUESTS"); do
  echo "[phase 1][$i/$PRE_REQUESTS]" | tee -a "$LOG_DIR/invocations_pre.jsonl"
  bin/serverledge-cli invoke -f "$FUNCTION_NAME" \
    -p "name:World" \
    --return_output \
    -H localhost -P "$LB_PORT" | tee -a "$LOG_DIR/invocations_pre.jsonl" || true
  echo >> "$LOG_DIR/invocations_pre.jsonl"
  sleep "$SLEEP_BETWEEN_REQUESTS"
done

echo
echo "[phase 2] Starting second-architecture node: $SECOND_NODE_CONF"
second_log_name="$(basename "$SECOND_NODE_CONF" .yaml)"
bin/serverledge "$SECOND_NODE_CONF" > "$LOG_DIR/${second_log_name}.log" 2>&1 &
PIDS+=("$!")

WAIT_SECONDS=$((LB_REFRESH_INTERVAL + 5))
echo "[phase 2] Waiting ${WAIT_SECONDS}s for LB periodic refresh and dynamic AddTarget/AddArmToAll..."
sleep "$WAIT_SECONDS"

echo
echo "[phase 2] Sending $POST_REQUESTS requests AFTER adding second architecture..."
for i in $(seq 1 "$POST_REQUESTS"); do
  echo "[phase 2][$i/$POST_REQUESTS]" | tee -a "$LOG_DIR/invocations_post.jsonl"
  bin/serverledge-cli invoke -f "$FUNCTION_NAME" \
    -p "name:World" \
    --return_output \
    -H localhost -P "$LB_PORT" | tee -a "$LOG_DIR/invocations_post.jsonl" || true
  echo >> "$LOG_DIR/invocations_post.jsonl"
  sleep "$SLEEP_BETWEEN_REQUESTS"
done

echo
echo "[summary] Relevant LB and MAB log lines:"
grep -E "\[LB\]|\[MAB\]|\[LinUCB\]|BanditManager|Selected target|Adding " "$LOG_DIR/lb.log" | tee "$LOG_DIR/lb_relevant.log" || true

echo
echo "[summary] MAB selected_tag occurrences:"
grep -o "selected_tag=[^ ]*" "$LOG_DIR/lb.log" | sort | uniq -c | tee "$LOG_DIR/selected_tags_summary.txt" || true

echo
echo "[summary] ExecutionNode occurrences:"
grep -h '"ExecutionNode"' "$LOG_DIR"/invocations_*.jsonl | sort | uniq -c | tee "$LOG_DIR/execution_nodes_summary.txt" || true

echo
echo "[done] Experiment completed."
echo "[done] Function name: $FUNCTION_NAME"
echo "[done] Logs: $LOG_DIR"