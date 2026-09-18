#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-analysis/bin/python}"
TMP_DIR="${TMP_DIR:-$(mktemp -d /tmp/serverledge-manhattan-smoke.XXXXXX)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

cleanup() {
  if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "[smoke] artifacts: $TMP_DIR"
  fi
}
trap cleanup EXIT

[[ -x "$PYTHON_BIN" ]] || {
  echo "[FAIL] Python analysis non trovato: $PYTHON_BIN" >&2
  exit 1
}

[[ -f analysis/profiling/similarity_selection.py ]] || {
  echo "[FAIL] similarity_selection.py non trovato" >&2
  exit 1
}

CATALOG="$TMP_DIR/catalog.json"
QUERY="$TMP_DIR/query.json"
EUCLIDEAN_JSON="$TMP_DIR/euclidean.json"
EUCLIDEAN_CSV="$TMP_DIR/euclidean.csv"
MANHATTAN_JSON="$TMP_DIR/manhattan.json"
MANHATTAN_CSV="$TMP_DIR/manhattan.csv"

"$PYTHON_BIN" - "$CATALOG" "$QUERY" <<'PY'
import json
import sys
from pathlib import Path

from analysis.profiling import preprocess, transfer_catalog

catalog_path = Path(sys.argv[1])
query_path = Path(sys.argv[2])

feature_names = list(preprocess.FEATURE_NAMES)
dim = len(feature_names)
if dim < 2:
    raise SystemExit("servono almeno due feature per lo smoke Manhattan")

zero = [0.0] * dim

donor_a = zero.copy()
donor_a[0] = 2.0

donor_b = zero.copy()
donor_b[0] = 1.1
donor_b[1] = 1.1

# Query all'origine:
#   Euclidean(A)=2.0
#   Euclidean(B)=sqrt(1.1^2+1.1^2)=~1.556 -> B
#   Manhattan(A)=2.0
#   Manhattan(B)=2.2 -> A
query_vector = zero.copy()

common = {
    "configured_cpus": 1.0,
    "configured_memory_mb": 128,
    "profile_machine_tag": "x86",
    "aggregation": "median",
    "scaler": "none",
    "algorithm": "local-manhattan-smoke",
    "cluster_label": 7,
    "is_noise": False,
    "donor_eligible": True,
    "donor_ineligibility_reason": "",
    "architecture_preference": "architecture_independent",
    "arm_vs_x86_delta_percent": 0.0,
    "threshold_percent": 15.0,
    "x86_duration_ms": 100.0,
    "arm_duration_ms": 100.0,
    "bandit_prior": None,
}

catalog = {
    "schema_version": transfer_catalog.TRANSFER_CATALOG_SCHEMA_VERSION,
    "catalog_run_id": "local-manhattan-smoke-catalog",
    "feature_names": feature_names,
    "feature_space": {
        "representation": "preprocessed",
        "scaler": "none",
        "distance_candidate": "manhattan",
        "distance_policy_selected": True,
    },
    "clustering": {
        "clustering_run_id": "local-manhattan-smoke-cluster",
        "algorithm": "fixture",
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
        "donor_count": 2,
        "eligible_donor_count": 2,
        "ineligible_donor_count": 0,
        "noise_donor_count": 0,
    },
    "donors": [
        {**common, "function_name": "donor_manhattan", "feature_vector": donor_a},
        {**common, "function_name": "donor_euclidean", "feature_vector": donor_b},
    ],
}

query = {
    "schema_version": 1,
    "query_id": "local-manhattan-smoke-query",
    "function_name": "target_smoke",
    "configured_cpus": 1.0,
    "configured_memory_mb": 128,
    "sample_count": 10,
    "profile_machine_tag": "x86",
    "aggregation": "median",
    "scaler": "none",
    "feature_names": feature_names,
    "feature_vector": query_vector,
    "cluster_label": 7,
}

catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
query_path.write_text(json.dumps(query, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" analysis/profiling/similarity_selection.py \
  --catalog "$CATALOG" \
  --query "$QUERY" \
  --run-id smoke-euclidean \
  --max-distance 100 \
  --distance euclidean \
  --require-same-cluster \
  --output-json "$EUCLIDEAN_JSON" \
  --output-csv "$EUCLIDEAN_CSV" >/dev/null

"$PYTHON_BIN" analysis/profiling/similarity_selection.py \
  --catalog "$CATALOG" \
  --query "$QUERY" \
  --run-id smoke-manhattan \
  --max-distance 100 \
  --distance manhattan \
  --require-same-cluster \
  --output-json "$MANHATTAN_JSON" \
  --output-csv "$MANHATTAN_CSV" >/dev/null

"$PYTHON_BIN" - "$EUCLIDEAN_JSON" "$MANHATTAN_JSON" <<'PY'
import json
import math
import sys
from pathlib import Path

euclidean = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manhattan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

if euclidean["selection_policy"]["distance"] != "euclidean":
    raise SystemExit("artifact euclidean non dichiara distance=euclidean")
if manhattan["selection_policy"]["distance"] != "manhattan":
    raise SystemExit("artifact Manhattan non dichiara distance=manhattan")
if not euclidean["selection_policy"]["require_same_cluster"]:
    raise SystemExit("same-cluster non attivo nell'artifact euclidean")
if not manhattan["selection_policy"]["require_same_cluster"]:
    raise SystemExit("same-cluster non attivo nell'artifact Manhattan")

ed = euclidean["selected_donor"]
md = manhattan["selected_donor"]

if ed["function_name"] != "donor_euclidean":
    raise SystemExit(f"donor Euclidean inatteso: {ed['function_name']}")
if md["function_name"] != "donor_manhattan":
    raise SystemExit(f"donor Manhattan inatteso: {md['function_name']}")

if not math.isclose(ed["distance"], math.sqrt(2.42), rel_tol=1e-12, abs_tol=1e-12):
    raise SystemExit(f"distanza Euclidean inattesa: {ed['distance']}")
if not math.isclose(md["distance"], 2.0, rel_tol=1e-12, abs_tol=1e-12):
    raise SystemExit(f"distanza Manhattan inattesa: {md['distance']}")

print("============================================================")
print("LOCAL MANHATTAN DONOR-SELECTION SMOKE: PASS")
print("============================================================")
print(f"euclidean -> {ed['function_name']} distance={ed['distance']:.6f}")
print(f"manhattan -> {md['function_name']} distance={md['distance']:.6f}")
print("same-cluster filtering: active")
print("metric metadata in runtime artifact: correct")
PY
