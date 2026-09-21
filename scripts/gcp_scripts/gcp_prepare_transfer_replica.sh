#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/gcp_config.sh"
source "${SCRIPT_DIR}/functions_catalog.sh"

if [[ $# -ne 2 ]]; then
    echo "Uso: $0 <target-function> <replica-label>"
    echo "Esempio: $0 graph-pagerank-node r02"
    exit 2
fi

TARGET_FUNCTION="$1"
REPLICA_LABEL="$2"
REPLICA_ID="${TARGET_FUNCTION}-${REPLICA_LABEL}"

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

TARGET_PROFILE_SAMPLES="${TARGET_PROFILE_SAMPLES:-10}"
MAB_UCB1_C="${MAB_UCB1_C:-0.8}"

REFERENCE_ARM="amd64"

NAME_LB="sl-lb"
NAME_WORKLOAD="sl-workload"
X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"

EXPECTED_COMMIT="${EXPECTED_COMMIT:-08600a177ce1d88c048251a4b6d52d31fe65a536}"

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

ANALYSIS_HELPER="${ROOT_DIR}/analysis/profiling/gcp_transfer_final_analysis.py"

HISTORICAL_PROFILES="${HISTORICAL_PROFILES:-${ROOT_DIR}/data/profiling/final-20260913-analysis-01/resource/x86/function-profiles-median.csv}"

REPLICA_DIR="${ROOT_DIR}/data/profiling/gcp-transfer-final/replicas/${REPLICA_ID}"
PROFILE_DIR="${REPLICA_DIR}/profile"
SELECTION_DIR="${REPLICA_DIR}/selection"


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

    if ! metadata="$(lookup_function "$function_name")"; then
        fail "funzione non presente nel catalogo: $function_name"
    fi

    local name runtime src memory handler
    IFS='|' read -r name runtime src memory handler <<< "$metadata"

    local handler_flag=""

    if [[ -n "$handler" ]]; then
        handler_flag="--handler ${handler}"
    fi

    echo "  register ${name} runtime=${runtime} memory=${memory}MB cpu=1"

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

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        export SERVERLEDGE_HOST='${X86_IP}'
        export SERVERLEDGE_PORT=1323

        /opt/serverledge/bin/serverledge-cli invoke \
            -f '${TARGET_FUNCTION}' \
            >'/tmp/${REPLICA_ID}-${label}.json'

        python3 - '/tmp/${REPLICA_ID}-${label}.json' <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    response = json.load(handle)

if response.get('Success') is not True:
    raise SystemExit(f'invocazione fallita: {response}')
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

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------

[[ -x "$PY" ]] || fail "Python venv non trovato: $PY"
[[ -f "$ANALYSIS_HELPER" ]] || fail "helper non trovato: $ANALYSIS_HELPER"
[[ -f "$HISTORICAL_PROFILES" ]] || fail "profili storici non trovati"

lookup_function "$TARGET_FUNCTION" >/dev/null \
    || fail "target non presente nel catalogo: $TARGET_FUNCTION"

if [[ -e "${REPLICA_DIR}/bootstrap.json" ]]; then
    fail "replica già esistente: ${REPLICA_DIR}"
fi

mkdir -p "$PROFILE_DIR" "$SELECTION_DIR"


# ------------------------------------------------------------------------------
# Fresh profiling runtime
# ------------------------------------------------------------------------------

banner "PREPARE REPLICA — ${REPLICA_ID}"

echo "target=${TARGET_FUNCTION}"
echo "samples=${TARGET_PROFILE_SAMPLES}"
echo "reference_arm=${REFERENCE_ARM}"
echo "replica_dir=${REPLICA_DIR}"

banner "RESET RUNTIME WITH PROFILING"

(
    cd "$SCRIPT_DIR"

    N_X86=1 \
    N_ARM=1 \
    START_PROFILING=1 \
    MAB_UCB1_C="$MAB_UCB1_C" \
    ./gcp_start_transfer_cluster.sh no-transfer
)


# ------------------------------------------------------------------------------
# IP
# ------------------------------------------------------------------------------

LB_IP="$(internal_ip "$NAME_LB")"
X86_IP="$(internal_ip "$X86_WORKER")"


# ------------------------------------------------------------------------------
# Register target
# ------------------------------------------------------------------------------

banner "REGISTER TARGET"

register_function "$TARGET_FUNCTION"


# ------------------------------------------------------------------------------
# Remove eventual old profiling files
# ------------------------------------------------------------------------------

banner "RESET PROFILING FILES"

for worker in "$X86_WORKER" "$ARM_WORKER"; do
    remote "$worker" "
        sudo mkdir -p /var/lib/serverledge
        sudo rm -f /var/lib/serverledge/profiling-samples.jsonl
        sudo chmod 777 /var/lib/serverledge
    " >/dev/null
done


# ------------------------------------------------------------------------------
# Target profiling on amd64 only
# ------------------------------------------------------------------------------

banner "TARGET PROFILE — AMD64 ONLY"

echo "  prewarm x86"
direct_invoke_x86 "prewarm"

for index in $(seq 1 "$TARGET_PROFILE_SAMPLES"); do
    echo "  sample ${index}/${TARGET_PROFILE_SAMPLES}"
    direct_invoke_x86 "sample-${index}"
done


# ------------------------------------------------------------------------------
# Recover x86 raw profile
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Verify zero target ARM profiling samples
# ------------------------------------------------------------------------------

ARM_PROFILE_SOURCE="$(find_profile_file "$ARM_WORKER" || true)"

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
    || fail "trovati ${ARM_TARGET_COUNT} sample target su ARM prima del transfer"

echo "  target ARM samples=0: PASS"


# ------------------------------------------------------------------------------
# Select exactly N eligible samples + compute anchor
# ------------------------------------------------------------------------------

banner "FILTER TARGET PROFILE"

FILTER_RESULT="$(
    "$PY" \
        "$ANALYSIS_HELPER" \
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

document = json.loads(sys.argv[1])
print(document["target_reference_mean_reward"])
PY
)"

echo "  anchor=${TARGET_REFERENCE_MEAN_REWARD}"


# ------------------------------------------------------------------------------
# Aggregate target profile
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Donor selection
# ------------------------------------------------------------------------------

banner "DONOR SELECTION"

"$PY" \
    "$ANALYSIS_HELPER" \
    select-donor \
    --historical-profiles "$HISTORICAL_PROFILES" \
    --target-profiles "$TARGET_PROFILE_CSV" \
    --target "$TARGET_FUNCTION" \
    --reference-tag amd64 \
    --configured-memory-mb 1024 \
    --run-id "$REPLICA_ID" \
    --output-dir "$SELECTION_DIR"

SELECTION_JSON="${SELECTION_DIR}/selection.json"

read -r SELECTED_DONOR TARGET_CLUSTER < <(
    "$PY" - "$SELECTION_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)

if document.get("status") != "selected":
    raise SystemExit(
        f"donor selection fallita: "
        f"{document.get('reason')}"
    )

print(
    document["selected_donor"]["function_name"],
    document["query"]["cluster_label"],
)
PY
)

lookup_function "$SELECTED_DONOR" >/dev/null \
    || fail "donor selezionato non disponibile nel catalogo: ${SELECTED_DONOR}"

echo "  target cluster=${TARGET_CLUSTER}"
echo "  selected donor=${SELECTED_DONOR}"


# ------------------------------------------------------------------------------
# Freeze bootstrap
# ------------------------------------------------------------------------------

banner "FREEZE REPLICA BOOTSTRAP"

"$PY" - \
    "$REPLICA_DIR" \
    "$SELECTION_JSON" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$TARGET_REFERENCE_MEAN_REWARD" \
    "$TARGET_PROFILE_SAMPLES" <<'PY'

import hashlib
import json
import sys
from pathlib import Path

(
    replica_dir_raw,
    selection_raw,
    target,
    donor,
    anchor_raw,
    samples_raw,
) = sys.argv[1:]

replica_dir = Path(replica_dir_raw).resolve()
selection_path = Path(selection_raw).resolve()

selection_sha = hashlib.sha256(
    selection_path.read_bytes()
).hexdigest()

document = {
    "schema_version": 1,
    "replica_id": replica_dir.name,
    "target_function": target,

    "target_profile": {
        "reference_arm": "amd64",
        "sample_count": int(samples_raw),
        "target_reference_mean_reward": float(anchor_raw),
        "reward_definition": "-ln(duration_ms)",
    },

    "donor_selection": {
        "selected_donor": donor,
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": selection_sha,

        "features": [
            "page_faults_delta",
            "utilized_cpus",
            "free_memory_mb",
            "cpu_user_delta_ms",
            "cpu_kernel_delta_ms",
        ],

        "aggregation": "median",
        "scaler": "minmax",

        "clustering": {
            "algorithm": "kmeans",
            "k": 5,
            "n_init": 50,
            "random_state": 42,
            "target_assignment": "euclidean_to_centroid",
        },

        "ranking": {
            "candidate_restriction": "same_cluster",
            "distance": "manhattan",
            "special_donors_excluded": True,
        },
    },

    "experimental_block": {
        "policies": {
            "no-transfer": {
                "policy": "UCB1",
                "c": 0.8,
            },
            "coupled": {
                "policy": "UCB1",
                "c": 0.8,
                "equivalent_observation_weight": 0.25,
            },
            "decoupled": {
                "policy": "UCB1Decoupled",
                "c": 0.8,
                "reward_observation_weight": 0.25,
                "exploration_observation_weight": 1.0,
            },
        }
    },
}

output = replica_dir / "bootstrap.json"

output.write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"bootstrap={output}")
print(f"selection_sha256={selection_sha}")
PY


banner "REPLICA BOOTSTRAP READY"

echo "replica=${REPLICA_ID}"
echo "target=${TARGET_FUNCTION}"
echo "donor=${SELECTED_DONOR}"
echo "cluster=${TARGET_CLUSTER}"
echo "anchor=${TARGET_REFERENCE_MEAN_REWARD}"
echo "bootstrap=${REPLICA_DIR}/bootstrap.json"

echo
echo "Nessuna policy target è stata misurata."
echo "Il runtime può ora essere resettato per no-transfer/coupled/decoupled."