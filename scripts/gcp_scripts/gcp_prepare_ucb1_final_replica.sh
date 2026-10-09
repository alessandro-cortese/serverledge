#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/gcp_config.sh"
source "${SCRIPT_DIR}/functions_catalog.sh"

if [[ $# -ne 2 ]]; then
    echo "Uso: $0 <target-function> <replica-label>" >&2
    echo "Esempio: $0 graph-bfs r01" >&2
    exit 2
fi

TARGET_FUNCTION="$1"
REPLICA_LABEL="$2"
REPLICA_ID="${TARGET_FUNCTION}-${REPLICA_LABEL}"

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT mancante}"

TARGET_PROFILE_SAMPLES="${TARGET_PROFILE_SAMPLES:-10}"

# c is irrelevant for direct target profiling, but we make the
# profiling runtime deterministic and consistent with Difference.
MAB_UCB1_C="${MAB_UCB1_C:-0.2}"

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

FINAL_SELECTOR="${ROOT_DIR}/analysis/profiling/gcp_ucb1_final_donor.py"
LEGACY_HELPER="${ROOT_DIR}/analysis/profiling/gcp_transfer_final_analysis.py"

CORPUS_ROOT="${CORPUS_ROOT:-${ROOT_DIR}/data/profiling/final-corpus64-20261005-analysis-01}"

HISTORICAL_PROFILES="${HISTORICAL_PROFILES:-${CORPUS_ROOT}/resource/x86/function-profiles-median.csv}"
STATIC_METRICS="${STATIC_METRICS:-${CORPUS_ROOT}/static-code-metrics.csv}"
PREFERENCES="${PREFERENCES:-${CORPUS_ROOT}/ground_truth/preferences-2p5.csv}"

RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/data/profiling/gcp-ucb1-final}"

REPLICA_DIR="${RESULT_ROOT}/replicas/${REPLICA_ID}"
PROFILE_DIR="${REPLICA_DIR}/profile"
SELECTION_DIR="${REPLICA_DIR}/selection"

NAME_WORKLOAD="sl-workload"
X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"

REFERENCE_ARM="amd64"


banner() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}


fail() {
    echo "FATAL: $*" >&2
    exit 1
}


remote() {
    local host="$1"
    shift

    gcloud compute ssh "$host" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        --command="$*"
}


scp_from() {
    local host="$1"
    local remote_path="$2"
    local local_path="$3"

    gcloud compute scp \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        "${host}:${remote_path}" \
        "$local_path"
}


internal_ip() {
    local host="$1"

    gcloud compute instances describe "$host" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --format='value(networkInterfaces[0].networkIP)'
}


lookup_function() {
    local wanted="$1"
    local entry

    for entry in "${FUNCTIONS[@]}"; do
        if [[ "${entry%%|*}" == "$wanted" ]]; then
            printf '%s\n' "$entry"
            return 0
        fi
    done

    return 1
}


register_function() {
    local function_name="$1"
    local metadata

    metadata="$(lookup_function "$function_name")" \
        || fail "funzione non presente nel catalogo: $function_name"

    local name runtime src memory handler

    IFS='|' read -r \
        name runtime src memory handler \
        <<< "$metadata"

    local handler_flag=""

    if [[ -n "$handler" ]]; then
        handler_flag="--handler ${handler}"
    fi

    echo \
        "  register ${name} runtime=${runtime} memory=${memory}MB cpu=1"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        export SERVERLEDGE_HOST='${LB_IP}'
        export SERVERLEDGE_PORT=1323

        /opt/serverledge/bin/serverledge-cli create \
            -f '${name}' \
            --runtime '${runtime}' \
            --src '${src}' \
            --memory '${memory}' \
            --cpu 1 \
            ${handler_flag} \
            --update
    " >/dev/null
}


direct_invoke_x86() {
    local label="$1"
    local remote_output="/tmp/${REPLICA_ID}-${label}.json"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        export SERVERLEDGE_HOST='${X86_IP}'
        export SERVERLEDGE_PORT=1323

        /opt/serverledge/bin/serverledge-cli invoke \
            -f '${TARGET_FUNCTION}' \
            >'${remote_output}'

        python3 - '${remote_output}' <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    r = json.load(f)

if r.get('Success') is not True:
    raise SystemExit(f'invocazione fallita: {r}')
PY
    " >/dev/null
}


find_profile_file() {
    local worker="$1"

    remote "$worker" "
        if sudo test -f /var/lib/serverledge/profiling-samples.jsonl; then
            echo /var/lib/serverledge/profiling-samples.jsonl
        fi
    "
}


# ==============================================================================
# Local preflight
# ==============================================================================

[[ -x "$PY" ]] \
    || fail "Python venv non trovato: $PY"

[[ -f "$FINAL_SELECTOR" ]] \
    || fail "selector finale non trovato: $FINAL_SELECTOR"

[[ -f "$LEGACY_HELPER" ]] \
    || fail "helper profiling non trovato: $LEGACY_HELPER"

[[ -f "$HISTORICAL_PROFILES" ]] \
    || fail "historical profiles non trovati"

[[ -f "$STATIC_METRICS" ]] \
    || fail "static metrics non trovate"

[[ -f "$PREFERENCES" ]] \
    || fail "preferences non trovate"

lookup_function "$TARGET_FUNCTION" >/dev/null \
    || fail "target non presente nel catalogo: $TARGET_FUNCTION"

if [[ -e "${REPLICA_DIR}/bootstrap.json" ]]; then
    fail "replica già esistente: $REPLICA_DIR"
fi

mkdir -p \
    "$PROFILE_DIR" \
    "$SELECTION_DIR"


# ==============================================================================
# Fresh profiling runtime
# ==============================================================================

banner "PREPARE UCB1 FINAL REPLICA — ${REPLICA_ID}"

echo "target=$TARGET_FUNCTION"
echo "samples=$TARGET_PROFILE_SAMPLES"
echo "expected_commit=$EXPECTED_COMMIT"
echo "replica_dir=$REPLICA_DIR"

banner "RESET RUNTIME WITH PROFILING"

(
    cd "$SCRIPT_DIR"

    EXPECTED_COMMIT="$EXPECTED_COMMIT" \
    N_X86=1 \
    N_ARM=1 \
    MAB_UCB1_C="$MAB_UCB1_C" \
    ./gcp_start_ucb1_final_cluster.sh profiling
)


# ==============================================================================
# IP
# ==============================================================================

LB_IP="$(internal_ip sl-lb)"
X86_IP="$(internal_ip "$X86_WORKER")"


# ==============================================================================
# Register target
# ==============================================================================

banner "REGISTER TARGET"

register_function "$TARGET_FUNCTION"


# ==============================================================================
# Clean old profiling files
# ==============================================================================

banner "RESET PROFILING FILES"

for worker in "$X86_WORKER" "$ARM_WORKER"; do
    remote "$worker" "
        sudo mkdir -p /var/lib/serverledge
        sudo rm -f /var/lib/serverledge/profiling-samples.jsonl
        sudo chmod 777 /var/lib/serverledge
    " >/dev/null
done


# ==============================================================================
# Profile target DIRECTLY on x86 only
# ==============================================================================

banner "TARGET PROFILE — AMD64 ONLY"

echo "  prewarm x86"
direct_invoke_x86 "prewarm"

for index in $(seq 1 "$TARGET_PROFILE_SAMPLES"); do
    echo "  sample ${index}/${TARGET_PROFILE_SAMPLES}"
    direct_invoke_x86 "sample-${index}"
done


# ==============================================================================
# Recover raw x86 profile
# ==============================================================================

X86_PROFILE_SOURCE="$(find_profile_file "$X86_WORKER")"

[[ -n "$X86_PROFILE_SOURCE" ]] \
    || fail "profiling-samples.jsonl x86 non trovato"

REMOTE_COPY="/tmp/${REPLICA_ID}-profiling-samples.jsonl"

remote "$X86_WORKER" "
    sudo cp '${X86_PROFILE_SOURCE}' '${REMOTE_COPY}'
    sudo chmod 644 '${REMOTE_COPY}'
"

scp_from \
    "$X86_WORKER" \
    "$REMOTE_COPY" \
    "${PROFILE_DIR}/raw-x86-all.jsonl"


# ==============================================================================
# Zero ARM target samples gate
# ==============================================================================

ARM_PROFILE_SOURCE="$(
    find_profile_file "$ARM_WORKER" || true
)"

if [[ -n "$ARM_PROFILE_SOURCE" ]]; then

    ARM_TARGET_COUNT="$(
        remote "$ARM_WORKER" "
            sudo grep -c \
                '\"function_name\":\"${TARGET_FUNCTION}\"' \
                '${ARM_PROFILE_SOURCE}' \
            || true
        " | tr -d '[:space:]'
    )"

else
    ARM_TARGET_COUNT=0
fi

[[ "${ARM_TARGET_COUNT:-0}" == "0" ]] \
    || fail \
        "trovati ${ARM_TARGET_COUNT} sample target ARM prima del transfer"

echo "  target ARM samples=0: PASS"


# ==============================================================================
# Select exactly N warm eligible samples and compute x86 anchor
# ==============================================================================

banner "FILTER TARGET PROFILE"

FILTER_RESULT="$(
    "$PY" \
        "$LEGACY_HELPER" \
        filter-profile \
        --raw "${PROFILE_DIR}/raw-x86-all.jsonl" \
        --target "$TARGET_FUNCTION" \
        --machine-tag amd64 \
        --samples "$TARGET_PROFILE_SAMPLES" \
        --output "${PROFILE_DIR}/profiling-samples.jsonl"
)"

echo "$FILTER_RESULT"

TARGET_REFERENCE_MEAN_REWARD="$(
    "$PY" - "$FILTER_RESULT" <<'PY'
import json
import sys

doc = json.loads(sys.argv[1])

print(
    doc["target_reference_mean_reward"]
)
PY
)"

echo \
    "  target_reference_mean_reward=${TARGET_REFERENCE_MEAN_REWARD}"


# ==============================================================================
# Aggregate target profile
# ==============================================================================

banner "AGGREGATE TARGET PROFILE"

AGGREGATE_INPUT="${PROFILE_DIR}/aggregate-input/sl-x86-1"

mkdir -p "$AGGREGATE_INPUT"

cp \
    "${PROFILE_DIR}/profiling-samples.jsonl" \
    "${AGGREGATE_INPUT}/profiling-samples.jsonl"

"${ROOT_DIR}/bin/serverledge-profiling" \
    aggregate \
    --input-dir "${PROFILE_DIR}/aggregate-input" \
    --output "${PROFILE_DIR}/function-profiles.jsonl" \
    --samples "$TARGET_PROFILE_SAMPLES" \
    >"${PROFILE_DIR}/aggregate.log"

"${ROOT_DIR}/bin/serverledge-profiling" \
    export-csv \
    --input "${PROFILE_DIR}/function-profiles.jsonl" \
    --experiment-id "${REPLICA_ID}" \
    >"${PROFILE_DIR}/export-csv.log"

TARGET_PROFILE_CSV="${PROFILE_DIR}/function-profiles-median.csv"

[[ -f "$TARGET_PROFILE_CSV" ]] \
    || fail "function-profiles-median.csv non generato"


# ==============================================================================
# Final donor selection
# ==============================================================================

banner "FINAL DONOR SELECTION"

"$PY" \
    "$FINAL_SELECTOR" \
    --historical-profiles "$HISTORICAL_PROFILES" \
    --target-profiles "$TARGET_PROFILE_CSV" \
    --static-metrics "$STATIC_METRICS" \
    --preferences "$PREFERENCES" \
    --target "$TARGET_FUNCTION" \
    --run-id "$REPLICA_ID" \
    --output-dir "$SELECTION_DIR"

SELECTION_JSON="${SELECTION_DIR}/selection.json"

read -r \
    SELECTED_DONOR \
    TARGET_CLUSTER \
    PREDICTION \
    < <(
        "$PY" - "$SELECTION_JSON" <<'PY'
import json
import sys

doc = json.load(
    open(sys.argv[1], encoding="utf-8")
)

if doc.get("status") != "selected":
    raise SystemExit(
        "selection status != selected"
    )

print(
    doc["selected_donor"]["function_name"],
    doc["target_cluster"],
    doc["prediction"],
)
PY
    )

lookup_function "$SELECTED_DONOR" >/dev/null \
    || fail \
        "donor live non presente nel functions catalog: ${SELECTED_DONOR}"

echo "  cluster=${TARGET_CLUSTER}"
echo "  prediction=${PREDICTION}"
echo "  donor=${SELECTED_DONOR}"


# ==============================================================================
# Freeze bootstrap
# ==============================================================================

banner "FREEZE FINAL REPLICA BOOTSTRAP"

"$PY" - \
    "$REPLICA_DIR" \
    "$SELECTION_JSON" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$TARGET_CLUSTER" \
    "$PREDICTION" \
    "$TARGET_REFERENCE_MEAN_REWARD" \
    "$TARGET_PROFILE_SAMPLES" \
    "$EXPECTED_COMMIT" <<'PY'

import hashlib
import json
import sys
from pathlib import Path

(
    replica_dir_raw,
    selection_raw,
    target,
    donor,
    cluster_raw,
    prediction,
    anchor_raw,
    samples_raw,
    expected_commit,
) = sys.argv[1:]

replica_dir = Path(
    replica_dir_raw
).resolve()

selection_path = Path(
    selection_raw
).resolve()

selection_sha = hashlib.sha256(
    selection_path.read_bytes()
).hexdigest()

doc = {
    "schema_version": 1,
    "experiment": "gcp-ucb1-final",
    "replica_id": replica_dir.name,
    "serverledge_commit": expected_commit,

    "target_function": target,

    "target_profile": {
        "reference_arm": "amd64",
        "sample_count": int(samples_raw),
        "aggregation": "median",
        "target_reference_mean_reward":
            float(anchor_raw),
        "reward_definition":
            "-ln(duration_ms)",
        "arm_ground_truth_used": False,
    },

    "donor_selection": {
        "selected_donor": donor,
        "target_cluster": int(cluster_raw),
        "prediction": prediction,

        "selection_artifact":
            str(selection_path),

        "selection_artifact_sha256":
            selection_sha,

        "features": [
            "page_faults_delta",
            "utilized_cpus",
            "free_memory_mb",
            "cpu_user_delta_ms",
            "cpu_kernel_delta_ms",
            "static_token_count_mean",
            "static_function_count",
        ],

        "scaler": {
            "type": "minmax",
            "fit": "donor_catalog_only",
            "target_excluded": True,
        },

        "clustering": {
            "algorithm": "kmeans",
            "k": 6,
            "n_init": 50,
            "random_state": 11,
            "dynamic_divisor": "sqrt(5)",
            "static_divisor": "sqrt(2)",
        },

        "ranking": {
            "candidate_restriction":
                "same_cluster+predicted_direction",
            "distance":
                "plain_manhattan_on_7_minmax_features",
            "directional_vote":
                "1/(d+1e-9)^2",
        },
    },

    "experimental_block": {
        "policy": "UCB1Decoupled",

        "conditions": {
            "notl-difference": {
                "prior": False,
                "c": 0.2,
            },

            "difference": {
                "prior": True,
                "mode": "difference",
                "c": 0.2,
                "reward_observation_weight":
                    0.05,
                "exploration_observation_weight":
                    64,
            },

            "notl-ratio": {
                "prior": False,
                "c": 0.4,
            },

            "ratio": {
                "prior": True,
                "mode": "ratio",
                "c": 0.4,
                "reward_observation_weight":
                    0.01,
                "exploration_observation_weight":
                    64,
            },
        },

        "primary_horizon": 10,
        "secondary_horizons":
            [5, 20, 50],
    },
}

output = replica_dir / "bootstrap.json"

output.write_text(
    json.dumps(
        doc,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"bootstrap={output}"
)

print(
    "selection_sha256="
    + selection_sha
)
PY


# ==============================================================================
# Final summary
# ==============================================================================

banner "UCB1 FINAL REPLICA READY"

echo "replica=$REPLICA_ID"
echo "target=$TARGET_FUNCTION"
echo "cluster=$TARGET_CLUSTER"
echo "prediction=$PREDICTION"
echo "donor=$SELECTED_DONOR"
echo "anchor=$TARGET_REFERENCE_MEAN_REWARD"
echo "bootstrap=${REPLICA_DIR}/bootstrap.json"

echo
echo "PASS — target ARM ground truth never used"
echo "PASS — target excluded from donor fit"
echo "PASS — donor selected with final K6/plain-Manhattan pipeline"
echo
echo "Nessuna misura online target è stata eseguita."