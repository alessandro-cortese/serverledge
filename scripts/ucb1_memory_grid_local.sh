#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-analysis/bin/python}"
TARGET_FUNCTION="${TARGET_FUNCTION:-billibench-knn}"
REPLICA="${REPLICA:-${TARGET_FUNCTION}-r01}"
REQUESTS="${REQUESTS:-5}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-3}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-4}"
JANITOR_INTERVAL="${JANITOR_INTERVAL:-1}"
OPTIMAL_ARM="${OPTIMAL_ARM:-arm64}"

STAMP="$(date +%Y%m%d_%H%M%S)"
GRID_ID="${GRID_ID:-ucb1-memory-grid-local-${TARGET_FUNCTION}-${STAMP}}"
GRID_DIR="${GRID_DIR:-logs/${GRID_ID}}"
RUNS_DIR="$GRID_DIR/runs"
PRIORS_DIR="$GRID_DIR/priors"
mkdir -p "$RUNS_DIR" "$PRIORS_DIR"

RUNNER="scripts/ucb1_memory_local_run.sh"
EVIDENCE_BASE="${EVIDENCE_BASE:-data/profiling/gcp-ucb1-final/replicas/${REPLICA}/evidence}"
SOURCE_BUNDLE="${SOURCE_BUNDLE:-${EVIDENCE_BASE}/source}"
EVIDENCE_JSON="${EVIDENCE_JSON:-${EVIDENCE_BASE}/evidence.json}"
MATERIALIZER_MODULE="analysis.profiling.transfer_materialized_tuned"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

[[ -x "$PYTHON_BIN" ]] || fail "Python analysis non trovato: $PYTHON_BIN"
[[ -x "$RUNNER" ]] || fail "runner non eseguibile: $RUNNER"
[[ -f "$SOURCE_BUNDLE/frozen-prior.json" ]] || fail "source frozen prior mancante: $SOURCE_BUNDLE"
[[ -f "$SOURCE_BUNDLE/manifest.json" ]] || fail "source manifest mancante: $SOURCE_BUNDLE"
[[ -f "$EVIDENCE_JSON" ]] || fail "evidence.json mancante: $EVIDENCE_JSON"

case "$OPTIMAL_ARM" in
  amd64|arm64) ;;
  *) fail "OPTIMAL_ARM deve essere amd64 oppure arm64" ;;
esac

# The source bundle fixes target anchor + donor selected by the frozen final
# clustering/donor-selection pipeline. Formula and UCB parameters are the only
# TL dimensions varied by this launcher.
read -r DONOR_REWARD_AMD64 DONOR_REWARD_ARM64 < <(
  "$PYTHON_BIN" - "$EVIDENCE_JSON" <<'PY'
import json
import sys

doc = json.load(open(sys.argv[1], encoding="utf-8"))
donor = doc.get("donor") or {}
try:
    amd64 = float((donor.get("amd64") or {})["mean_reward"])
    arm64 = float((donor.get("arm64") or {})["mean_reward"])
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"donor mean rewards non leggibili: {exc}")
print(f"{amd64:.17g} {arm64:.17g}")
PY
)

"$PYTHON_BIN" - "$SOURCE_BUNDLE" "$EVIDENCE_JSON" "$TARGET_FUNCTION" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
evidence_path = Path(sys.argv[2])
target = sys.argv[3]

manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

if manifest.get("target_function") != target:
    raise SystemExit(
        f"source target={manifest.get('target_function')} != requested target={target}"
    )

prior = json.loads((source / "frozen-prior.json").read_text(encoding="utf-8"))
if prior.get("policy") != "UCB1":
    raise SystemExit("source bundle non è coupled UCB1")

anchor = (prior.get("config") or {}).get("ucb1_reference_anchor") or {}
if anchor.get("reference_arm") != "amd64":
    raise SystemExit("source reference arm != amd64")

print("SOURCE PROVENANCE PREFLIGHT: PASS")
print(f"target={target}")
print(f"donor={prior.get('donor_function_name')}")
print(f"anchor={anchor.get('target_reference_mean_reward')}")
print(f"evidence_donor={(evidence.get('donor') or {}).get('function_name', evidence.get('donor_function'))}")
PY

echo "============================================================"
echo "UCB1 MEMORY LOCAL GRID"
echo "============================================================"
echo "grid_id:              $GRID_ID"
echo "target:               $TARGET_FUNCTION"
echo "source bundle:        $SOURCE_BUNDLE"
echo "evidence:             $EVIDENCE_JSON"
echo "donor reward amd64:   $DONOR_REWARD_AMD64"
echo "donor reward arm64:   $DONOR_REWARD_ARM64"
echo "requests/config:      $REQUESTS"
echo "concurrency:          $REQUEST_CONCURRENCY"
echo "expiration:           $CONTAINER_EXPIRATION"
echo "janitor:              $JANITOR_INTERVAL"
echo "optimal arm metadata: $OPTIMAL_ARM"
echo

echo "[setup] Build una sola volta..."
make >/dev/null

# id|formula|wR|wE|c
CONFIGS=(
  "N08|none|||0.8"
  "D08|difference|0.25|1|0.8"
  "R08|ratio|0.25|1|0.8"
  "N02|none|||0.2"
  "D02|difference|0.05|64|0.2"
  "R02|ratio|0.05|64|0.2"
  "N04|none|||0.4"
  "D04|difference|0.01|64|0.4"
  "R04|ratio|0.01|64|0.4"
)

prepare_prior() {
  local id="$1"
  local formula="$2"
  local wr="$3"
  local we="$4"
  local c="$5"
  local out="$PRIORS_DIR/$id"

  rm -rf "$out"
  mkdir -p "$out"
  rmdir "$out"

  "$PYTHON_BIN" -m "$MATERIALIZER_MODULE" \
    --source-bundle "$SOURCE_BUNDLE" \
    --output-bundle "$out" \
    --mode "$formula" \
    --reward-weight "$wr" \
    --exploration-weight "$we" \
    --donor-reward-ref "$DONOR_REWARD_AMD64" \
    --donor-reward-other "$DONOR_REWARD_ARM64" \
    >"$PRIORS_DIR/${id}-materialize.log"

  "$PYTHON_BIN" - \
    "$out/manifest.json" \
    "$id" \
    "$c" \
    "$formula" \
    "$wr" \
    "$we" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
id_ = sys.argv[2]
c = float(sys.argv[3])
formula = sys.argv[4]
wr = float(sys.argv[5])
we = float(sys.argv[6])

doc = json.loads(path.read_text(encoding="utf-8"))
# c is a bandit runtime parameter, not part of the prior itself. The generic
# tuned materializer historically writes formula-specific target_c metadata;
# for this controlled memory grid we record the actual matched runtime c.
doc["target_c"] = c
doc["memory_grid_config_id"] = id_
doc["memory_grid_formula"] = formula
doc["memory_grid_runtime_c"] = c
doc["memory_grid_reward_observation_weight"] = wr
doc["memory_grid_exploration_observation_weight"] = we
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "[prepare] Materializzo offline le 6 condizioni TL dalla STESSA evidenza sorgente..."
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r id formula wr we c <<<"$row"
  if [[ "$formula" != "none" ]]; then
    echo "  - $id formula=$formula wR=$wr wE=$we c=$c"
    prepare_prior "$id" "$formula" "$wr" "$we" "$c"
  fi
done

cat > "$GRID_DIR/grid-config.csv" <<'CSV'
config_id,formula,w_R,w_E,c
N08,none,,,0.8
D08,difference,0.25,1,0.8
R08,ratio,0.25,1,0.8
N02,none,,,0.2
D02,difference,0.05,64,0.2
R02,ratio,0.05,64,0.2
N04,none,,,0.4
D04,difference,0.01,64,0.4
R04,ratio,0.01,64,0.4
CSV

echo
echo "[grid] Avvio le 9 condizioni..."

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r id formula wr we c <<<"$row"
  run_dir="$RUNS_DIR/$id"
  prior_request=""

  if [[ "$formula" != "none" ]]; then
    prior_request="$PRIORS_DIR/$id/materialized-request.json"
  fi

  echo
  echo "------------------------------------------------------------"
  echo "RUN $id formula=$formula wR=${wr:--} wE=${we:--} c=$c"
  echo "------------------------------------------------------------"

  CONFIG_ID="$id" \
  TARGET_FUNCTION="$TARGET_FUNCTION" \
  FORMULA="$formula" \
  MAB_UCB1_C="$c" \
  REWARD_PRIOR_WEIGHT="$wr" \
  EXPLORATION_PRIOR_WEIGHT="$we" \
  PRIOR_REQUEST="$prior_request" \
  OPTIMAL_ARM="$OPTIMAL_ARM" \
  REQUESTS="$REQUESTS" \
  REQUEST_CONCURRENCY="$REQUEST_CONCURRENCY" \
  CONTAINER_EXPIRATION="$CONTAINER_EXPIRATION" \
  JANITOR_INTERVAL="$JANITOR_INTERVAL" \
  SKIP_BUILD=1 \
  RUN_ID="${GRID_ID}-${id}" \
  LOG_DIR="$run_dir" \
    "$RUNNER"
done

echo
echo "[summary] Aggrego i 9 summary.json..."

"$PYTHON_BIN" - "$RUNS_DIR" "$GRID_DIR/grid-summary.csv" "$GRID_DIR/grid-summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
csv_out = Path(sys.argv[2])
json_out = Path(sys.argv[3])

order = ["N08", "D08", "R08", "N02", "D02", "R02", "N04", "D04", "R04"]
rows = []

for config_id in order:
    path = runs_dir / config_id / "summary.json"
    if not path.is_file():
        raise SystemExit(f"summary mancante: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("pairing") != "PASS":
        raise SystemExit(f"pairing non PASS per {config_id}")
    rows.append({
        "config_id": config_id,
        "formula": doc["formula"],
        "c": doc["c"],
        "w_R": doc["reward_prior_weight"],
        "w_E": doc["exploration_prior_weight"],
        "requests": doc["requests"],
        "request_concurrency": doc["request_concurrency"],
        "created_containers": doc["created_containers"],
        "amd64_containers": doc.get("containers_by_arm", {}).get("amd64", 0),
        "arm64_containers": doc.get("containers_by_arm", {}).get("arm64", 0),
        "amd64_memory_GiB_s": doc.get("memory_GiB_s_by_arm", {}).get("amd64", 0.0),
        "arm64_memory_GiB_s": doc.get("memory_GiB_s_by_arm", {}).get("arm64", 0.0),
        "total_memory_GiB_s": doc["total_memory_GiB_s"],
        "wrong_arm_memory_GiB_s": doc["wrong_arm_memory_GiB_s"],
        "prewarm_target_containers": doc["prewarm_target_containers"],
        "pairing": doc["pairing"],
    })

with csv_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

print("config  formula      c    wR    wE   containers   total_GiB_s   wrong_GiB_s")
for row in rows:
    wr = "-" if row["w_R"] is None else f"{row['w_R']:g}"
    we = "-" if row["w_E"] is None else f"{row['w_E']:g}"
    print(
        f"{row['config_id']:<6} "
        f"{row['formula']:<11} "
        f"{row['c']:<4g} "
        f"{wr:<5} "
        f"{we:<5} "
        f"{row['created_containers']:<12} "
        f"{row['total_memory_GiB_s']:<13.6f} "
        f"{row['wrong_arm_memory_GiB_s']:.6f}"
    )
PY

echo
echo "============================================================"
echo "UCB1 MEMORY LOCAL GRID: PASS"
echo "============================================================"
echo "summary CSV:  $GRID_DIR/grid-summary.csv"
echo "summary JSON: $GRID_DIR/grid-summary.json"
echo "runs:         $RUNS_DIR"
echo "priors:       $PRIORS_DIR"
echo
echo "NOTA: i valori locali NON sono evidenza prestazionale x86-vs-ARM."
echo "Questo PASS valida la pipeline sperimentale delle 9 condizioni."
