#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_REQUESTS="${TARGET_REQUESTS:-8}"
PHASE1_REQUESTS="${PHASE1_REQUESTS:-4}"
MAB_LINUCB_ALPHA="${MAB_LINUCB_ALPHA:-0.1}"
TARGET_MEMORY_MB="${TARGET_MEMORY_MB:-128}"
BACKGROUND_MEMORY_MB="${BACKGROUND_MEMORY_MB:-128}"
BACKGROUND_FUNCTION_COUNT="${BACKGROUND_FUNCTION_COUNT:-2}"
BACKGROUND_HOLD_SECONDS="${BACKGROUND_HOLD_SECONDS:-12}"
POOL_MEMORY_MB="${POOL_MEMORY_MB:-512}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-60}"
JANITOR_INTERVAL="${JANITOR_INTERVAL:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PAIR_ID="${PAIR_ID:-linucb-context-pair-local-${STAMP}}"
PAIR_DIR="${PAIR_DIR:-logs/${PAIR_ID}}"
mkdir -p "$PAIR_DIR"

RUNNER="scripts/linucb_context_local_run.sh"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

[[ -x "$RUNNER" ]] || fail "runner non eseguibile: $RUNNER"

echo "============================================================"
echo "LINUCB CONTEXT LOCAL — A/B PAIR"
echo "============================================================"
echo "pair_id:               $PAIR_ID"
echo "target requests:       $TARGET_REQUESTS"
echo "phase1 requests:       $PHASE1_REQUESTS"
echo "alpha:                 $MAB_LINUCB_ALPHA"
echo "target memory:         ${TARGET_MEMORY_MB} MiB"
echo "background funcs B:    ${BACKGROUND_FUNCTION_COUNT}"
echo "background memory/B:   ${BACKGROUND_MEMORY_MB} MiB each"
echo "background hold B:     ${BACKGROUND_HOLD_SECONDS}s"
echo "pool memory/ring:      ${POOL_MEMORY_MB} MiB"
echo

echo "[setup] Build una sola volta..."
make >/dev/null

COMMON=(
  TARGET_REQUESTS="$TARGET_REQUESTS"
  PHASE1_REQUESTS="$PHASE1_REQUESTS"
  MAB_LINUCB_ALPHA="$MAB_LINUCB_ALPHA"
  TARGET_MEMORY_MB="$TARGET_MEMORY_MB"
  BACKGROUND_MEMORY_MB="$BACKGROUND_MEMORY_MB"
  BACKGROUND_FUNCTION_COUNT="$BACKGROUND_FUNCTION_COUNT"
  BACKGROUND_HOLD_SECONDS="$BACKGROUND_HOLD_SECONDS"
  POOL_MEMORY_MB="$POOL_MEMORY_MB"
  CONTAINER_EXPIRATION="$CONTAINER_EXPIRATION"
  JANITOR_INTERVAL="$JANITOR_INTERVAL"
  SKIP_BUILD=1
)

echo
echo "------------------------------------------------------------"
echo "SCENARIO A — CLEAN"
echo "------------------------------------------------------------"
env \
  "${COMMON[@]}" \
  SCENARIO=A \
  RUN_ID="${PAIR_ID}-A" \
  LOG_DIR="$PAIR_DIR/A" \
  "$RUNNER"

echo
echo "------------------------------------------------------------"
echo "SCENARIO B — ACTIVE CONTEXT VARIATION"
echo "------------------------------------------------------------"
env \
  "${COMMON[@]}" \
  SCENARIO=B \
  RUN_ID="${PAIR_ID}-B" \
  LOG_DIR="$PAIR_DIR/B" \
  "$RUNNER"

python3 - \
  "$PAIR_DIR/A/summary.json" \
  "$PAIR_DIR/B/summary.json" \
  "$PAIR_DIR/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

for key in (
    "target_requests",
    "phase1_requests",
    "phase2_requests",
    "linucb_alpha",
):
    if a.get(key) != b.get(key):
        raise SystemExit(
            f"A/B mismatch per {key}: A={a.get(key)} B={b.get(key)}"
        )

a_post_max = float(a["phase2_context"]["arm64_max"])
b_post_max = float(b["phase2_context"]["arm64_max"])
increment_over_clean = b_post_max - a_post_max

if increment_over_clean < 0.20:
    raise SystemExit(
        "B non amplia abbastanza il supporto del context arm64 rispetto ad A: "
        f"A_phase2_max={a_post_max:.6f} "
        f"B_phase2_max={b_post_max:.6f} "
        f"increment={increment_over_clean:.6f}"
    )

comparison = {
    "schema_version": 3,
    "status": "PASS",
    "shared_target_requests": a["target_requests"],
    "shared_phase1_requests": a["phase1_requests"],
    "shared_linucb_alpha": a["linucb_alpha"],
    "scenario_A": {
        "phase1_context": a["phase1_context"],
        "phase2_context": a["phase2_context"],
        "warm_observation_count": a["warm_observation_count"],
        "selected_arm_counts": a["selected_arm_counts"],
    },
    "scenario_B": {
        "phase1_context": b["phase1_context"],
        "phase2_context": b["phase2_context"],
        "warm_observation_count": b["warm_observation_count"],
        "selected_arm_counts": b["selected_arm_counts"],
        "arm64_context_shift_max": b.get("arm64_context_shift_max"),
        "background": b.get("background"),
    },
    "B_minus_A_phase2_arm64_max": increment_over_clean,
}

Path(sys.argv[3]).write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("A/B SHARED-PROTOCOL CHECK: PASS")
print(f"A phase2 arm64 max={a_post_max:.6f}")
print(f"B phase2 arm64 max={b_post_max:.6f}")
print(f"B-A phase2 arm64 max={increment_over_clean:.6f}")
PY

echo
echo "============================================================"
echo "LINUCB CONTEXT LOCAL A/B PAIR: PASS"
echo "============================================================"
echo "A summary:      $PAIR_DIR/A/summary.json"
echo "B summary:      $PAIR_DIR/B/summary.json"
echo "comparison:     $PAIR_DIR/comparison.json"
echo
echo "Questo PASS valida il meccanismo locale."
echo "Le conclusioni su hardware/beneficio LinUCB richiederanno GCP reale."
