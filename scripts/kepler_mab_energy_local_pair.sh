#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INVOCATIONS="${INVOCATIONS:-3}"
CPU_SECONDS="${CPU_SECONDS:-2.5}"
REWARD_ZONE="${REWARD_ZONE:-package}"
KEPLER_URL="${KEPLER_URL:-http://127.0.0.1:28282/metrics}"
UCB1_C="${UCB1_C:-0.8}"
LINUCB_ALPHA="${LINUCB_ALPHA:-0.1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PAIR_ID="${PAIR_ID:-kepler-mab-energy-pair-local-${STAMP}}"
PAIR_DIR="${PAIR_DIR:-logs/${PAIR_ID}}"
RUNNER="scripts/kepler_mab_energy_local_run.sh"

mkdir -p "$PAIR_DIR"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

[[ -x "$RUNNER" ]] || fail "runner non eseguibile: $RUNNER"

curl -fsS --max-time 5 "$KEPLER_URL" >/dev/null \
  || fail "Kepler non raggiungibile: avvialo prima del test"

echo "============================================================"
echo "KEPLER -> MAB ENERGY LOCAL — UCB1 + LinUCB"
echo "============================================================"
echo "pair_id:      $PAIR_ID"
echo "invocations:  $INVOCATIONS per policy"
echo "CPU seconds:  $CPU_SECONDS"
echo "reward zone:  $REWARD_ZONE"
echo "UCB1 c:       $UCB1_C"
echo "LinUCB alpha: $LINUCB_ALPHA"
echo

echo "[setup] Build una sola volta..."
make >/dev/null

COMMON=(
  INVOCATIONS="$INVOCATIONS"
  CPU_SECONDS="$CPU_SECONDS"
  REWARD_ZONE="$REWARD_ZONE"
  KEPLER_URL="$KEPLER_URL"
  UCB1_C="$UCB1_C"
  LINUCB_ALPHA="$LINUCB_ALPHA"
  SKIP_BUILD=1
)

echo
echo "------------------------------------------------------------"
echo "UCB1 + KEPLER ENERGY"
echo "------------------------------------------------------------"
env \
  "${COMMON[@]}" \
  POLICY=UCB1 \
  RUN_ID="${PAIR_ID}-UCB1" \
  LOG_DIR="$PAIR_DIR/UCB1" \
  "$RUNNER"

echo
echo "------------------------------------------------------------"
echo "LINUCB + KEPLER ENERGY"
echo "------------------------------------------------------------"
env \
  "${COMMON[@]}" \
  POLICY=LinUCB \
  RUN_ID="${PAIR_ID}-LinUCB" \
  LOG_DIR="$PAIR_DIR/LinUCB" \
  "$RUNNER"

python3 - \
  "$PAIR_DIR/UCB1/summary.json" \
  "$PAIR_DIR/LinUCB/summary.json" \
  "$PAIR_DIR/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

ucb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lin = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

for summary, policy in ((ucb, "UCB1"), (lin, "LinUCB")):
    if summary.get("status") != "PASS":
        raise SystemExit(f"{policy}: status non PASS")
    if summary.get("energy_coverage") != 1.0:
        raise SystemExit(
            f"{policy}: energy coverage != 100%"
        )
    if summary.get("reward_mode") != "kepler_energy":
        raise SystemExit(
            f"{policy}: reward mode non energetico"
        )

for key in ("reward_zone", "invocations"):
    if ucb.get(key) != lin.get(key):
        raise SystemExit(
            f"A/B mismatch {key}: "
            f"UCB1={ucb.get(key)!r} LinUCB={lin.get(key)!r}"
        )

comparison = {
    "schema_version": 1,
    "status": "PASS",
    "reward_zone": ucb["reward_zone"],
    "invocations_per_policy": ucb["invocations"],
    "UCB1": {
        "energy_coverage": ucb["energy_coverage"],
        "mean_energy_j": ucb["mean_energy_j"],
    },
    "LinUCB": {
        "energy_coverage": lin["energy_coverage"],
        "mean_energy_j": lin["mean_energy_j"],
    },
    "note": (
        "Local mechanics validation only; mean energy values are not "
        "a UCB1-vs-LinUCB performance comparison."
    ),
}

Path(sys.argv[3]).write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("PAIR CONSISTENCY CHECK: PASS")
print("UCB1 coverage=100%")
print("LinUCB coverage=100%")
PY

echo
echo "============================================================"
echo "KEPLER -> MAB ENERGY LOCAL PAIR: PASS"
echo "============================================================"
echo "UCB1 summary:   $PAIR_DIR/UCB1/summary.json"
echo "LinUCB summary: $PAIR_DIR/LinUCB/summary.json"
echo "comparison:     $PAIR_DIR/comparison.json"
