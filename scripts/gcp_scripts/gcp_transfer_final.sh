#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/gcp_config.sh"
source "${SCRIPT_DIR}/functions_catalog.sh"

# ==============================================================================
# Modalità
# ==============================================================================

MODE="${1:-}"

case "$MODE" in
    no-transfer|coupled|decoupled)
        ;;
    *)
        echo "Uso: $0 {no-transfer|coupled|decoupled}" >&2
        exit 2
        ;;
esac

# ==============================================================================
# Configurazione generale
# ==============================================================================

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

N_X86="${N_X86:-1}"
N_ARM="${N_ARM:-1}"

TARGET_FUNCTION="${TARGET_FUNCTION:-graph-pagerank-node}"

# Solo controllo opzionale.
# NON viene più popolato automaticamente dalla shortlist offline.
EXPECTED_DONOR="${EXPECTED_DONOR:-}"

# Directory della replica congelata.
#
# Esempio:
#
# data/profiling/gcp-transfer-final/replicas/graph-pagerank-node-r01
#
REPLICA_BOOTSTRAP_DIR="${REPLICA_BOOTSTRAP_DIR:-}"

MAB_UCB1_C="${MAB_UCB1_C:-0.8}"

EQUIVALENT_PRIOR_WEIGHT="${EQUIVALENT_PRIOR_WEIGHT:-0.25}"

REWARD_PRIOR_WEIGHT="${REWARD_PRIOR_WEIGHT:-0.25}"
EXPLORATION_PRIOR_WEIGHT="${EXPLORATION_PRIOR_WEIGHT:-1.0}"

MIN_DONOR_OBSERVATIONS="${MIN_DONOR_OBSERVATIONS:-10}"

DONOR_MAX_REQUESTS="${DONOR_MAX_REQUESTS:-200}"
DONOR_BATCH_SIZE="${DONOR_BATCH_SIZE:-10}"

MEASURE_REQUESTS="${MEASURE_REQUESTS:-50}"

REFERENCE_ARM="amd64"
OTHER_ARM="arm64"

EXPECTED_COMMIT="${EXPECTED_COMMIT:-08600a177ce1d88c048251a4b6d52d31fe65a536}"

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

ANALYSIS_HELPER="${ROOT_DIR}/analysis/profiling/gcp_transfer_final_analysis.py"

HISTORICAL_PROFILES="${HISTORICAL_PROFILES:-${ROOT_DIR}/data/profiling/final-20260913-analysis-01/resource/x86/function-profiles-median.csv}"

# ==============================================================================
# Nomi VM
# ==============================================================================

NAME_REGISTRY="sl-registry"
NAME_LB="sl-lb"
NAME_WORKLOAD="sl-workload"

X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"

# ==============================================================================
# Directory risultato
# ==============================================================================

RUN_ID="gcp-transfer-final-${TARGET_FUNCTION}-${MODE}-$(date +%Y%m%d_%H%M%S)"

RESULT_DIR="${ROOT_DIR}/data/profiling/gcp-transfer-final/${RUN_ID}"

TARGET_DIR="${RESULT_DIR}/target"
DONOR_DIR="${RESULT_DIR}/donor"
MEASURE_DIR="${RESULT_DIR}/measure"
TRANSFER_DIR="${RESULT_DIR}/transfer"

mkdir -p \
    "$TARGET_DIR" \
    "$DONOR_DIR" \
    "$MEASURE_DIR" \
    "$TRANSFER_DIR"

# ==============================================================================
# Policy runtime
# ==============================================================================

POLICY="UCB1"
START_PROFILING=0
TRANSFER_ENABLED=false

case "$MODE" in

    no-transfer)
        POLICY="UCB1"
        START_PROFILING=0
        TRANSFER_ENABLED=false
        ;;

    coupled)
        POLICY="UCB1"
        START_PROFILING=0
        TRANSFER_ENABLED=true
        ;;

    decoupled)
        POLICY="UCB1Decoupled"
        START_PROFILING=0
        TRANSFER_ENABLED=true
        ;;

esac

# ==============================================================================
# Utility
# ==============================================================================

banner() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}

fail() {

    echo "FATAL: $*" >&2

    echo "Artefatti: $RESULT_DIR" >&2

    echo "Le VM restano intatte; cleanup:" >&2
    echo "  cd ${SCRIPT_DIR} && ./gcp_down.sh" >&2

    exit 1
}

on_error() {

    local status=$?

    echo >&2
    echo "ERRORE: gcp_transfer_final.sh interrotto (exit=${status})." >&2
    echo "Artefatti parziali: ${RESULT_DIR}" >&2

    echo "Le VM NON vengono distrutte automaticamente." >&2

    exit "$status"
}

trap on_error ERR

require_file() {

    local path="$1"

    [[ -f "$path" ]] || fail "file non trovato: $path"
}

require_positive_int() {

    local name="$1"
    local value="$2"

    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        fail "$name deve essere un intero > 0 (ricevuto: $value)"
    fi
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

scp_to() {

    local local_path="$1"
    local host="$2"
    local remote_path="$3"

    gcloud compute scp \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        "$local_path" \
        "${host}:${remote_path}"
}

internal_ip() {

    local host="$1"

    gcloud compute instances describe "$host" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --format='value(networkInterfaces[0].networkIP)'
}

# ==============================================================================
# Catalogo funzioni
# ==============================================================================

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
        fail "funzione non presente in functions_catalog.sh: $function_name"
    fi

    local name
    local runtime
    local src
    local memory
    local handler

    IFS='|' read -r \
        name \
        runtime \
        src \
        memory \
        handler \
        <<< "$metadata"

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
            --update \
            >/tmp/register-${name}.log 2>&1 || {

                cat /tmp/register-${name}.log >&2
                exit 1
            }

        cat /tmp/register-${name}.log
    " >/dev/null
}

# ==============================================================================
# Invocazioni
# ==============================================================================

direct_invoke() {

    local function_name="$1"
    local destination_ip="$2"
    local label="$3"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        export SERVERLEDGE_HOST='${destination_ip}'
        export SERVERLEDGE_PORT=1323

        /opt/serverledge/bin/serverledge-cli invoke \
            -f '${function_name}' \
            >'/tmp/${RUN_ID}-${label}.json'

        python3 - '/tmp/${RUN_ID}-${label}.json' <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    response = json.load(handle)

if response.get('Success') is not True:
    raise SystemExit(f'invocazione fallita: {response}')
PY
    " >/dev/null
}

lb_invoke_batch() {

    local function_name="$1"
    local count="$2"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        export SERVERLEDGE_HOST='${LB_IP}'
        export SERVERLEDGE_PORT=1323

        for i in \$(seq 1 '${count}'); do

            /opt/serverledge/bin/serverledge-cli invoke \
                -f '${function_name}' \
                >/tmp/${RUN_ID}-donor-invoke.json

            python3 - '/tmp/${RUN_ID}-donor-invoke.json' <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    response = json.load(handle)

if response.get('Success') is not True:
    raise SystemExit(f'invocazione donor fallita: {response}')
PY

        done
    " >/dev/null
}

# ==============================================================================
# Conteggio feedback MAB
# ==============================================================================

warm_updates() {

    local function_name="$1"
    local arm="$2"

    remote "$NAME_LB" "
        sudo grep 'event=update_reward' \
            /var/log/serverledge-lb.log 2>/dev/null \
        | grep 'function=${function_name}' \
        | grep 'arm=${arm}' \
        | grep -c 'warm_start=true' \
        || true
    " | tr -d '[:space:]'
}

target_update_count() {

    remote "$NAME_LB" "
        sudo grep 'event=update_reward' \
            /var/log/serverledge-lb.log 2>/dev/null \
        | grep -F -c 'function=${TARGET_FUNCTION} ' \
        || true
    " | tr -d '[:space:]'
}

# ==============================================================================
# Bootstrap congelato
# ==============================================================================

load_replica_bootstrap() {

    local bootstrap_dir="$1"

    local bootstrap_json="${bootstrap_dir}/bootstrap.json"
    local bootstrap_selection="${bootstrap_dir}/selection/selection.json"

    require_file "$bootstrap_json"
    require_file "$bootstrap_selection"

    local values

    if ! values="$(
        "$PY" - \
            "$bootstrap_json" \
            "$bootstrap_selection" \
            "$TARGET_FUNCTION" \
            "$MODE" \
            "$MAB_UCB1_C" \
            "$EQUIVALENT_PRIOR_WEIGHT" \
            "$REWARD_PRIOR_WEIGHT" \
            "$EXPLORATION_PRIOR_WEIGHT" <<'PY'

import hashlib
import json
import math
import sys


(
    bootstrap_path,
    selection_path,
    expected_target,
    mode,
    c_raw,
    w_raw,
    wr_raw,
    we_raw,
) = sys.argv[1:]


with open(bootstrap_path, encoding="utf-8") as handle:
    bootstrap = json.load(handle)


with open(selection_path, encoding="utf-8") as handle:
    selection = json.load(handle)


if bootstrap.get("schema_version") != 1:
    raise SystemExit("bootstrap schema_version inattesa")


replica_id = str(
    bootstrap.get("replica_id", "")
).strip()

if not replica_id:
    raise SystemExit("replica_id assente")


target = str(
    bootstrap.get("target_function", "")
).strip()

if target != expected_target:
    raise SystemExit(
        f"bootstrap target={target!r}, "
        f"requested target={expected_target!r}"
    )


if selection.get("schema_version") != 1:
    raise SystemExit(
        "selection schema_version inattesa"
    )


if selection.get("status") != "selected":
    raise SystemExit(
        "selection non selezionata: "
        f"status={selection.get('status')!r}, "
        f"reason={selection.get('reason')!r}"
    )


query = selection.get("query") or {}

if query.get("function_name") != target:
    raise SystemExit(
        "selection target diverso dal bootstrap target"
    )


expected_features = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]


if query.get("feature_names") != expected_features:
    raise SystemExit(
        "selection non usa PAPER-5 esatte"
    )


if query.get("aggregation") != "median":
    raise SystemExit(
        "selection aggregation non median"
    )


paper5 = selection.get("paper5") or {}


if paper5.get("feature_names") != expected_features:
    raise SystemExit(
        "paper5.feature_names inattese"
    )


if paper5.get("scaler") != "minmax":
    raise SystemExit(
        "selection scaler non minmax"
    )


if paper5.get(
    "target_assignment"
) != "euclidean_to_centroid":

    raise SystemExit(
        "target assignment inatteso"
    )


if paper5.get("donor_ranking") != "manhattan":
    raise SystemExit(
        "donor ranking non Manhattan"
    )


kmeans = paper5.get("kmeans") or {}

if (
    kmeans.get("k") != 5
    or kmeans.get("n_init") != 50
    or kmeans.get("random_state") != 42
):
    raise SystemExit(
        "configurazione KMeans inattesa"
    )


selection_policy = (
    selection.get("selection_policy") or {}
)


if (
    selection_policy.get("distance")
    != "manhattan"
):
    raise SystemExit(
        "selection_policy.distance non Manhattan"
    )


if (
    selection_policy.get(
        "require_same_cluster"
    )
    is not True
):
    raise SystemExit(
        "selection non richiede same cluster"
    )


if (
    selection_policy.get(
        "configuration_match_required"
    )
    is not True
):
    raise SystemExit(
        "selection non richiede configuration match"
    )


if (
    selection_policy.get(
        "bandit_prior_materialized"
    )
    is not False
):
    raise SystemExit(
        "selection contiene prior materializzato"
    )


if (
    selection.get(
        "bandit_prior",
        "missing",
    )
    is not None
):
    raise SystemExit(
        "selection.bandit_prior deve essere null"
    )


donor = str(
    (
        selection.get(
            "selected_donor"
        )
        or {}
    ).get(
        "function_name",
        "",
    )
).strip()


if not donor:
    raise SystemExit(
        "selected donor assente"
    )


lower = donor.lower()

if (
    lower.startswith("twin-")
    or lower
    in {
        "amd_faster",
        "arm_faster",
        "amd-faster",
        "arm-faster",
    }
):
    raise SystemExit(
        f"donor speciale non ammesso: {donor}"
    )


donor_selection = (
    bootstrap.get(
        "donor_selection"
    )
    or {}
)


bootstrap_donor = str(
    donor_selection.get(
        "selected_donor",
        "",
    )
).strip()


if donor != bootstrap_donor:

    raise SystemExit(
        f"selection donor={donor!r}, "
        f"bootstrap donor={bootstrap_donor!r}"
    )


expected_sha = str(
    donor_selection.get(
        "selection_artifact_sha256",
        "",
    )
).strip()


with open(
    selection_path,
    "rb",
) as handle:

    actual_sha = hashlib.sha256(
        handle.read()
    ).hexdigest()


if actual_sha != expected_sha:

    raise SystemExit(
        "selection.json SHA-256 "
        "non coincide con bootstrap.json"
    )


target_profile = (
    bootstrap.get(
        "target_profile"
    )
    or {}
)


if (
    target_profile.get(
        "reference_arm"
    )
    != "amd64"
):
    raise SystemExit(
        "reference arm del bootstrap non amd64"
    )


sample_count = int(
    target_profile.get(
        "sample_count",
        0,
    )
)

if sample_count <= 0:
    raise SystemExit(
        "sample_count bootstrap non valido"
    )


anchor = float(
    target_profile[
        "target_reference_mean_reward"
    ]
)


if not math.isfinite(anchor):
    raise SystemExit(
        "target reference mean reward non finito"
    )


if (
    target_profile.get(
        "reward_definition"
    )
    != "-ln(duration_ms)"
):
    raise SystemExit(
        "reward definition bootstrap inattesa"
    )


experimental_block = (
    bootstrap.get(
        "experimental_block"
    )
    or {}
)


policies = (
    experimental_block.get(
        "policies"
    )
    or {}
)


current_policy = (
    policies.get(mode)
    or {}
)


runtime_c = float(c_raw)


if current_policy:

    bootstrap_c = float(
        current_policy.get(
            "c",
            float("nan"),
        )
    )

    if not math.isclose(
        bootstrap_c,
        runtime_c,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"c runtime={runtime_c} "
            f"diverso dal bootstrap={bootstrap_c}"
        )


if mode == "coupled":

    runtime_w = float(w_raw)

    bootstrap_w = float(
        current_policy.get(
            "equivalent_observation_weight",
            float("nan"),
        )
    )

    if not math.isclose(
        runtime_w,
        bootstrap_w,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"coupled w runtime={runtime_w} "
            f"diverso dal bootstrap={bootstrap_w}"
        )


if mode == "decoupled":

    runtime_wr = float(wr_raw)
    runtime_we = float(we_raw)

    bootstrap_wr = float(
        current_policy.get(
            "reward_observation_weight",
            float("nan"),
        )
    )

    bootstrap_we = float(
        current_policy.get(
            "exploration_observation_weight",
            float("nan"),
        )
    )

    if not math.isclose(
        runtime_wr,
        bootstrap_wr,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"decoupled wR runtime={runtime_wr} "
            f"diverso dal bootstrap={bootstrap_wr}"
        )

    if not math.isclose(
        runtime_we,
        bootstrap_we,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            f"decoupled wE runtime={runtime_we} "
            f"diverso dal bootstrap={bootstrap_we}"
        )


cluster = int(
    query["cluster_label"]
)


print(
    "\t".join(
        [
            replica_id,
            target,
            donor,
            repr(anchor),
            str(cluster),
            str(sample_count),
            actual_sha,
        ]
    )
)

PY
    )"; then

        fail \
            "bootstrap replica non valido: $bootstrap_dir"
    fi

    IFS=$'\t' read -r \
        BOOTSTRAP_REPLICA_ID \
        BOOTSTRAP_TARGET \
        BLOCK_SELECTED_DONOR \
        TARGET_REFERENCE_MEAN_REWARD \
        TARGET_CLUSTER \
        BOOTSTRAP_PROFILE_SAMPLES \
        BOOTSTRAP_SELECTION_SHA \
        <<< "$values"

    mkdir -p \
        "$TARGET_DIR/selection-work"

    cp \
        "$bootstrap_selection" \
        "$TARGET_DIR/selection-work/selection.json"

    local artifact

    for artifact in \
        selection.csv \
        transfer-query.json \
        paper5-model.json
    do

        if [[ -f "${bootstrap_dir}/selection/${artifact}" ]]; then

            cp \
                "${bootstrap_dir}/selection/${artifact}" \
                "$TARGET_DIR/selection-work/${artifact}"
        fi

    done

    SELECTION_JSON="$TARGET_DIR/selection-work/selection.json"

    echo "  replica bootstrap: PASS"
    echo "  replica id:        $BOOTSTRAP_REPLICA_ID"
    echo "  target:            $BOOTSTRAP_TARGET"
    echo "  donor block:       $BLOCK_SELECTED_DONOR"
    echo "  target cluster:    $TARGET_CLUSTER"
    echo "  reference arm:     $REFERENCE_ARM"
    echo "  profile samples:   $BOOTSTRAP_PROFILE_SAMPLES"
    echo "  anchor:            $TARGET_REFERENCE_MEAN_REWARD"
    echo "  selection sha256:  $BOOTSTRAP_SELECTION_SHA"
}

# ==============================================================================
# Reset worker prima della misura
#
# IMPORTANTE:
# - NON si riavvia il load balancer;
# - quindi donor knowledge + weak prior restano nel BanditManager;
# - si eliminano solo i container runtime;
# - profiling viene disabilitato;
# - poi target viene prewarmato direttamente su entrambi i worker.
# ==============================================================================

restart_workers_measurement() {

    echo \
        "  riavvio worker con profiling OFF e container vuoti"

    local worker
    local tag

    for worker in \
        "$X86_WORKER" \
        "$ARM_WORKER"
    do

        if [[ "$worker" == "$X86_WORKER" ]]; then
            tag="amd64"
        else
            tag="arm64"
        fi

        remote "$worker" "
            set -euo pipefail

            sudo pkill -x serverledge \
                >/dev/null 2>&1 \
                || true

            sleep 1

            ids=\$(sudo docker ps -aq)

            if [[ -n \"\$ids\" ]]; then
                sudo docker rm -f \$ids >/dev/null
            fi

            sudo tee \
                /opt/serverledge/worker-transfer-final.yaml \
                >/dev/null <<EOF_WORKER

registry:
  area: \"cloud-region\"
  node:
    id: \"${worker}\"
  udp:
    port: 9877

etcd:
  address: \"${REGISTRY_IP}:2379\"

node:
  machine_tag: \"${tag}\"

container:
  pool:
    memory: 12000

janitor:
  interval: 60

EOF_WORKER

            sudo sh -c '
                cd /opt/serverledge &&
                nohup ./bin/serverledge \
                    worker-transfer-final.yaml \
                    > /var/log/serverledge-node.log \
                    2>&1 &
            '
        " >/dev/null
    done

    sleep 8

    for worker in \
        "$X86_WORKER" \
        "$ARM_WORKER"
    do

        remote "$worker" \
            "pgrep -x serverledge >/dev/null" \
            >/dev/null \
            || fail \
                "worker non ripartito: $worker"
    done
}

# ==============================================================================
# Manifest
# ==============================================================================

write_manifest() {

    local selected_donor_for_manifest="${SELECTED_DONOR:-}"
    local block_donor_for_manifest="${BLOCK_SELECTED_DONOR:-}"

    local bootstrap_reused=false

    if [[ -n "$REPLICA_BOOTSTRAP_DIR" ]]; then
        bootstrap_reused=true
    fi

    cat >"${RESULT_DIR}/manifest.txt" <<EOF_MANIFEST
run_id=${RUN_ID}
mode=${MODE}
policy=${POLICY}
target=${TARGET_FUNCTION}
selected_donor=${selected_donor_for_manifest}
block_selected_donor=${block_donor_for_manifest}
expected_donor=${EXPECTED_DONOR}
target_cluster=${TARGET_CLUSTER:-}
target_reference_mean_reward=${TARGET_REFERENCE_MEAN_REWARD:-}
c=${MAB_UCB1_C}
coupled_w=${EQUIVALENT_PRIOR_WEIGHT}
decoupled_wR=${REWARD_PRIOR_WEIGHT}
decoupled_wE=${EXPLORATION_PRIOR_WEIGHT}
min_donor_observations_per_arm=${MIN_DONOR_OBSERVATIONS}
donor_warm_updates_amd64=${DONOR_AMD64_UPDATES:-0}
donor_warm_updates_arm64=${DONOR_ARM64_UPDATES:-0}
target_profile_samples_amd64=${BOOTSTRAP_PROFILE_SAMPLES:-0}
measured_requests=${MEASURE_REQUESTS}
serverledge_commit=${EXPECTED_COMMIT}
historical_profiles=${HISTORICAL_PROFILES}
features=page_faults_delta,utilized_cpus,free_memory_mb,cpu_user_delta_ms,cpu_kernel_delta_ms
clustering=minmax+kmeans_k5_ninit50_seed42
target_assignment=euclidean_to_centroid
donor_ranking=same_cluster_manhattan
special_donors_excluded=true
replica_bootstrap_dir=${REPLICA_BOOTSTRAP_DIR}
bootstrap_reused=${bootstrap_reused}
bootstrap_replica_id=${BOOTSTRAP_REPLICA_ID:-}
bootstrap_selection_sha256=${BOOTSTRAP_SELECTION_SHA:-}
selection_artifact=${SELECTION_JSON:-}
EOF_MANIFEST
}

# ==============================================================================
# Preflight locale
# ==============================================================================

require_positive_int \
    "N_X86" \
    "$N_X86"

require_positive_int \
    "N_ARM" \
    "$N_ARM"

require_positive_int \
    "MIN_DONOR_OBSERVATIONS" \
    "$MIN_DONOR_OBSERVATIONS"

require_positive_int \
    "DONOR_MAX_REQUESTS" \
    "$DONOR_MAX_REQUESTS"

require_positive_int \
    "DONOR_BATCH_SIZE" \
    "$DONOR_BATCH_SIZE"

require_positive_int \
    "MEASURE_REQUESTS" \
    "$MEASURE_REQUESTS"

[[ "$N_X86" == "1" ]] \
    || fail \
        "campagna finale attesa con N_X86=1"

[[ "$N_ARM" == "1" ]] \
    || fail \
        "campagna finale attesa con N_ARM=1"

require_file \
    "$ANALYSIS_HELPER"

[[ -x "$PY" ]] \
    || fail \
        "Python venv non trovato/eseguibile: $PY"

require_file \
    "${SCRIPT_DIR}/gcp_start_transfer_cluster.sh"

# ==============================================================================
# Vincolo metodologico
#
# coupled e decoupled NON possono più creare una donor selection propria.
# Devono consumare il bootstrap congelato della replica.
# ==============================================================================

if [[ "$MODE" != "no-transfer" && -z "$REPLICA_BOOTSTRAP_DIR" ]]; then

    fail \
        "${MODE} richiede REPLICA_BOOTSTRAP_DIR: donor e anchor devono essere congelati prima del confronto"
fi

# ==============================================================================
# Caricamento bootstrap
#
# Può essere passato anche alla baseline, solo per associare formalmente
# la baseline allo stesso experimental block.
# ==============================================================================

if [[ -n "$REPLICA_BOOTSTRAP_DIR" ]]; then

    REPLICA_BOOTSTRAP_DIR="$(
        cd "$REPLICA_BOOTSTRAP_DIR"
        pwd
    )"

    banner \
        "LOAD FROZEN REPLICA BOOTSTRAP"

    load_replica_bootstrap \
        "$REPLICA_BOOTSTRAP_DIR"
fi

# ==============================================================================
# Donor della policy
# ==============================================================================

SELECTED_DONOR=""

if [[ "$MODE" != "no-transfer" ]]; then

    SELECTED_DONOR="$BLOCK_SELECTED_DONOR"

    if [[ -n "$EXPECTED_DONOR" && "$SELECTED_DONOR" != "$EXPECTED_DONOR" ]]; then

        fail \
            "donor bootstrap=${SELECTED_DONOR}, expected=${EXPECTED_DONOR}"
    fi

    lookup_function \
        "$SELECTED_DONOR" \
        >/dev/null \
        || fail \
            "donor bootstrap non presente in functions_catalog.sh: $SELECTED_DONOR"
fi

lookup_function \
    "$TARGET_FUNCTION" \
    >/dev/null \
    || fail \
        "target non presente in functions_catalog.sh: $TARGET_FUNCTION"

# ==============================================================================
# Header
# ==============================================================================

banner \
    "GCP TRANSFER FINAL — SINGLE REPLICA"

echo "run=$RUN_ID"
echo "mode=$MODE"
echo "policy=$POLICY"
echo "target=$TARGET_FUNCTION"
echo "block_donor=${BLOCK_SELECTED_DONOR:-none}"
echo "applied_donor=${SELECTED_DONOR:-none}"
echo "c=$MAB_UCB1_C"
echo "bootstrap=${REPLICA_BOOTSTRAP_DIR:-none}"
echo "results=$RESULT_DIR"

# ==============================================================================
# Le VM devono già esistere.
#
# Questo script NON crea e NON distrugge VM.
# ==============================================================================

for vm in \
    "$NAME_REGISTRY" \
    "$NAME_LB" \
    "$NAME_WORKLOAD" \
    "$X86_WORKER" \
    "$ARM_WORKER"
do

    status="$(
        gcloud compute instances describe \
            "$vm" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --format='value(status)' \
            2>/dev/null \
            || true
    )"

    [[ "$status" == "RUNNING" ]] \
        || fail \
            "VM $vm non RUNNING (status=${status:-missing})"
done

# ==============================================================================
# Fresh runtime
#
# gcp_start_transfer_cluster.sh:
# - resetta etcd;
# - crea fresh BanditManager;
# - registra i due worker reali;
# - usa UCB1 o UCB1Decoupled in base a MODE;
# - abilita transfer API solo per coupled/decoupled.
#
# Profiling è OFF qui:
# il target profile appartiene già al bootstrap della replica.
# ==============================================================================

banner \
    "RESET RUNTIME"

(
    cd "$SCRIPT_DIR"

    N_X86="$N_X86" \
    N_ARM="$N_ARM" \
    PROFILING="$START_PROFILING" \
    MAB_UCB1_C="$MAB_UCB1_C" \
    ./gcp_start_transfer_cluster.sh \
        "$MODE"
)

# ==============================================================================
# IP
# ==============================================================================

REGISTRY_IP="$(
    internal_ip \
        "$NAME_REGISTRY"
)"

LB_IP="$(
    internal_ip \
        "$NAME_LB"
)"

X86_IP="$(
    internal_ip \
        "$X86_WORKER"
)"

ARM_IP="$(
    internal_ip \
        "$ARM_WORKER"
)"

# ==============================================================================
# Target registration
# ==============================================================================

register_function \
    "$TARGET_FUNCTION"

# ==============================================================================
# Transfer conditions
# ==============================================================================

DONOR_AMD64_UPDATES=0
DONOR_ARM64_UPDATES=0

CONTROL_REQUEST=""
CONTROL_RESPONSE=""

SELECTION_JSON="${SELECTION_JSON:-}"

if [[ "$MODE" != "no-transfer" ]]; then

    # --------------------------------------------------------------------------
    # Donor live knowledge
    #
    # L'identità del donor è congelata dal bootstrap.
    #
    # La conoscenza MAB del donor viene invece ricostruita REALMENTE su GCP
    # nel BanditManager fresco di questa policy.
    # --------------------------------------------------------------------------

    banner \
        "DONOR LIVE KNOWLEDGE — ${SELECTED_DONOR}"

    register_function \
        "$SELECTED_DONOR"

    # Cold start diretto per entrambi gli executor.
    # Queste chiamate NON attraversano MAB.
    direct_invoke \
        "$SELECTED_DONOR" \
        "$X86_IP" \
        "donor-prewarm-amd64"

    direct_invoke \
        "$SELECTED_DONOR" \
        "$ARM_IP" \
        "donor-prewarm-arm64"

    total_donor_requests=0

    while (( total_donor_requests < DONOR_MAX_REQUESTS )); do

        lb_invoke_batch \
            "$SELECTED_DONOR" \
            "$DONOR_BATCH_SIZE"

        total_donor_requests=$((total_donor_requests + DONOR_BATCH_SIZE))

        DONOR_AMD64_UPDATES="$(
            warm_updates \
                "$SELECTED_DONOR" \
                "amd64"
        )"

        DONOR_ARM64_UPDATES="$(
            warm_updates \
                "$SELECTED_DONOR" \
                "arm64"
        )"

        echo \
            "  requests=${total_donor_requests} amd64=${DONOR_AMD64_UPDATES} arm64=${DONOR_ARM64_UPDATES}"

        if (( \
            DONOR_AMD64_UPDATES >= MIN_DONOR_OBSERVATIONS \
            && DONOR_ARM64_UPDATES >= MIN_DONOR_OBSERVATIONS \
        )); then

            break
        fi
    done

    if (( \
        DONOR_AMD64_UPDATES < MIN_DONOR_OBSERVATIONS \
        || DONOR_ARM64_UPDATES < MIN_DONOR_OBSERVATIONS \
    )); then

        remote "$NAME_LB" "
            sudo grep \
                'event=update_reward.*function=${SELECTED_DONOR}' \
                /var/log/serverledge-lb.log \
                || true
        " >"${DONOR_DIR}/insufficient-feedback.log" \
            || true

        printf '%s\n' \
            '{' \
            '  "schema_version": 1,' \
            '  "status": "insufficient-real-observations",' \
            "  \"target_function\": \"${TARGET_FUNCTION}\"," \
            "  \"donor_function\": \"${SELECTED_DONOR}\"," \
            "  \"mode\": \"${MODE}\"," \
            "  \"minimum_observations_per_arm\": ${MIN_DONOR_OBSERVATIONS}," \
            "  \"maximum_donor_requests\": ${DONOR_MAX_REQUESTS}," \
            '  "observations": {' \
            "    \"amd64\": ${DONOR_AMD64_UPDATES}," \
            "    \"arm64\": ${DONOR_ARM64_UPDATES}" \
            '  },' \
            '  "transfer_eligible": false' \
            '}' \
            >"${DONOR_DIR}/donor-readiness.json"

        fail \
            "donor real observations insufficienti: amd64=${DONOR_AMD64_UPDATES} arm64=${DONOR_ARM64_UPDATES}"
    fi

    printf '%s\n' \
        '{' \
        '  "schema_version": 1,' \
        '  "status": "ready",' \
        "  \"target_function\": \"${TARGET_FUNCTION}\"," \
        "  \"donor_function\": \"${SELECTED_DONOR}\"," \
        "  \"mode\": \"${MODE}\"," \
        "  \"minimum_observations_per_arm\": ${MIN_DONOR_OBSERVATIONS}," \
        "  \"maximum_donor_requests\": ${DONOR_MAX_REQUESTS}," \
        '  "observations": {' \
        "    \"amd64\": ${DONOR_AMD64_UPDATES}," \
        "    \"arm64\": ${DONOR_ARM64_UPDATES}" \
        '  },' \
        '  "transfer_eligible": true' \
        '}' \
        >"${DONOR_DIR}/donor-readiness.json"

    echo \
        "  donor real feedback: PASS (amd64=${DONOR_AMD64_UPDATES}, arm64=${DONOR_ARM64_UPDATES})"

    # Conserviamo anche le linee di feedback donor per audit.
    remote "$NAME_LB" "
        sudo grep \
            'event=update_reward' \
            /var/log/serverledge-lb.log \
        | grep \
            'function=${SELECTED_DONOR}' \
        || true
    " >"${DONOR_DIR}/update-reward.log"

    # --------------------------------------------------------------------------
    # Costruzione weak prior
    #
    # selection.json = IDENTICO tra coupled e decoupled della replica.
    #
    # anchor = IDENTICO tra coupled e decoupled.
    #
    # Ciò che cambia:
    #
    # coupled:
    #   equivalent_observation_weight=.25
    #
    # decoupled:
    #   reward_observation_weight=.25
    #   exploration_observation_weight=1.0
    # --------------------------------------------------------------------------

    banner \
        "APPLY FROZEN-SELECTION WEAK PRIOR"

    CONTROL_REQUEST="${TRANSFER_DIR}/control-request.json"
    CONTROL_RESPONSE="${TRANSFER_DIR}/control-response.json"

    "$PY" \
        "$ANALYSIS_HELPER" \
        build-prior \
        --selection "$SELECTION_JSON" \
        --target "$TARGET_FUNCTION" \
        --mode "$MODE" \
        --weight "$EQUIVALENT_PRIOR_WEIGHT" \
        --reward-weight "$REWARD_PRIOR_WEIGHT" \
        --exploration-weight "$EXPLORATION_PRIOR_WEIGHT" \
        --minimum "$MIN_DONOR_OBSERVATIONS" \
        --target-reference-mean-reward \
            "$TARGET_REFERENCE_MEAN_REWARD" \
        --reference-arm "$REFERENCE_ARM" \
        --output "$CONTROL_REQUEST"

    remote_request="/tmp/${RUN_ID}-transfer-request.json"
    remote_response="/tmp/${RUN_ID}-transfer-response.json"

    scp_to \
        "$CONTROL_REQUEST" \
        "$NAME_WORKLOAD" \
        "$remote_request"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        HTTP_CODE=\$(

            curl \
                -sS \
                --max-time 30 \
                -o '${remote_response}' \
                -w '%{http_code}' \
                -H 'Content-Type: application/json' \
                --data-binary '@${remote_request}' \
                'http://${LB_IP}:1323/mab/transfer/initialize'
        )

        echo \"HTTP_CODE=\$HTTP_CODE\"

        [[ \"\$HTTP_CODE\" == '200' ]]
    "

    scp_from \
        "$NAME_WORKLOAD" \
        "$remote_response" \
        "$CONTROL_RESPONSE"

    "$PY" \
        "$ANALYSIS_HELPER" \
        verify-prior \
        --response "$CONTROL_RESPONSE" \
        --target "$TARGET_FUNCTION" \
        --donor "$SELECTED_DONOR" \
        --mode "$MODE" \
        --weight "$EQUIVALENT_PRIOR_WEIGHT" \
        --reward-weight "$REWARD_PRIOR_WEIGHT" \
        --exploration-weight "$EXPLORATION_PRIOR_WEIGHT" \
        --target-reference-mean-reward \
            "$TARGET_REFERENCE_MEAN_REWARD" \
        --reference-arm "$REFERENCE_ARM" \
        --other-arm "$OTHER_ARM"

    # --------------------------------------------------------------------------
    # Nessuna richiesta target deve aver attraversato il MAB.
    # --------------------------------------------------------------------------

    TARGET_UPDATES_AFTER_PRIOR="$(
        target_update_count
    )"

    [[ "$TARGET_UPDATES_AFTER_PRIOR" == "0" ]] \
        || fail \
            "target ha real feedback prima della misura: ${TARGET_UPDATES_AFTER_PRIOR}"
fi

# ==============================================================================
# Normalizzazione misura
#
# - LB NON viene riavviato;
# - target prior resta intatto;
# - worker vengono riavviati;
# - container cancellati;
# - profiling OFF;
# - target prewarmato direttamente su entrambi.
# ==============================================================================

banner \
    "NORMALIZE MEASUREMENT STATE"

restart_workers_measurement

# Prewarm diretto amd64.
direct_invoke \
    "$TARGET_FUNCTION" \
    "$X86_IP" \
    "target-prewarm-amd64"

# Prewarm diretto arm64.
direct_invoke \
    "$TARGET_FUNCTION" \
    "$ARM_IP" \
    "target-prewarm-arm64"

# Nessuna delle due richieste deve essere passata attraverso MAB.
TARGET_UPDATES_PRE_MEASURE="$(
    target_update_count
)"

[[ "$TARGET_UPDATES_PRE_MEASURE" == "0" ]] \
    || fail \
        "target real feedback pre-measure=${TARGET_UPDATES_PRE_MEASURE}, atteso 0"

echo \
    "  target real feedback pre-measure=0: PASS"

# ==============================================================================
# Punto del log da cui inizia la misura
# ==============================================================================

LB_LINES_BEFORE="$(
    remote \
        "$NAME_LB" \
        "sudo wc -l < /var/log/serverledge-lb.log" \
    | tr -d '[:space:]'
)"

[[ "$LB_LINES_BEFORE" =~ ^[0-9]+$ ]] \
    || fail \
        "line count LB non valido: $LB_LINES_BEFORE"

# ==============================================================================
# Misura
# ==============================================================================

banner \
    "MEASURE ${MEASURE_REQUESTS} TARGET REQUESTS"

REMOTE_MEASURE_SCRIPT="/tmp/${RUN_ID}-measure.py"
REMOTE_REQUEST_CSV="/tmp/${RUN_ID}-requests.csv"

LOCAL_MEASURE_SCRIPT="${MEASURE_DIR}/measure.py"

cat >"$LOCAL_MEASURE_SCRIPT" <<'PY'

#!/usr/bin/env python3

import csv
import json
import sys
import time

from datetime import datetime, timezone
from urllib.request import Request, urlopen


lb_ip = sys.argv[1]
function_name = sys.argv[2]
request_count = int(sys.argv[3])
output_csv = sys.argv[4]


url = (
    f"http://{lb_ip}:1323/"
    f"invoke/{function_name}"
)


payload = json.dumps(
    {
        "params": {}
    }
).encode("utf-8")


with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(handle)

    writer.writerow(
        [
            "request_index",
            "timestamp_utc",
            "client_elapsed_ms",
            "node_arch",
            "success",
        ]
    )

    for index in range(
        1,
        request_count + 1,
    ):

        request = Request(
            url,
            data=payload,
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST",
        )

        started = time.perf_counter()

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        with urlopen(
            request,
            timeout=180,
        ) as response:

            body = response.read()

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            node_arch = (
                response.headers.get("Serverledge-Node-Arch")
                or response.headers.get("Serverledge-Node-Arch")
                or ""
            ).strip()

        document = json.loads(
            body.decode("utf-8")
        )

        success = (
            document.get("Success")
            is True
        )

        if not success:
            raise SystemExit(
                f"request {index} "
                f"failed: {document}"
            )

        if not node_arch:
            raise SystemExit(
                f"request {index}: "
                "Node-Arch header missing"
            )

        writer.writerow(
            [
                index,
                timestamp,
                f"{elapsed_ms:.6f}",
                node_arch,
                "true",
            ]
        )

        handle.flush()

        print(
            f"request={index}/"
            f"{request_count} "
            f"arch={node_arch}",
            flush=True,
        )

PY

scp_to \
    "$LOCAL_MEASURE_SCRIPT" \
    "$NAME_WORKLOAD" \
    "$REMOTE_MEASURE_SCRIPT"

remote "$NAME_WORKLOAD" "
    set -euo pipefail

    python3 \
        '${REMOTE_MEASURE_SCRIPT}' \
        '${LB_IP}' \
        '${TARGET_FUNCTION}' \
        '${MEASURE_REQUESTS}' \
        '${REMOTE_REQUEST_CSV}'
"

scp_from \
    "$NAME_WORKLOAD" \
    "$REMOTE_REQUEST_CSV" \
    "${MEASURE_DIR}/requests.csv"

# ==============================================================================
# Solo porzione LB relativa alla misura
# ==============================================================================

LB_START_LINE=$((LB_LINES_BEFORE + 1))

REMOTE_LB_MEASURE="/tmp/${RUN_ID}-lb-measure.log"
REMOTE_LB_FULL="/tmp/${RUN_ID}-lb-full.log"

remote "$NAME_LB" "
    set -euo pipefail

    sudo tail \
        -n +${LB_START_LINE} \
        /var/log/serverledge-lb.log \
        > '${REMOTE_LB_MEASURE}'

    sudo cp \
        /var/log/serverledge-lb.log \
        '${REMOTE_LB_FULL}'

    sudo chmod \
        644 \
        '${REMOTE_LB_MEASURE}' \
        '${REMOTE_LB_FULL}'
"

scp_from \
    "$NAME_LB" \
    "$REMOTE_LB_MEASURE" \
    "${MEASURE_DIR}/lb-measure.log"

scp_from \
    "$NAME_LB" \
    "$REMOTE_LB_FULL" \
    "${RESULT_DIR}/lb-full.log"

# ==============================================================================
# Summary
# ==============================================================================

"$PY" \
    "$ANALYSIS_HELPER" \
    summarize \
    --lb-log \
        "${MEASURE_DIR}/lb-measure.log" \
    --request-csv \
        "${MEASURE_DIR}/requests.csv" \
    --target \
        "$TARGET_FUNCTION" \
    --requests \
        "$MEASURE_REQUESTS" \
    --mode \
        "$MODE" \
    --policy \
        "$POLICY" \
    --output-csv \
        "${MEASURE_DIR}/mab-requests.csv" \
    --output-json \
        "${MEASURE_DIR}/summary.json"

# ==============================================================================
# Manifest
# ==============================================================================

write_manifest

trap - ERR

# ==============================================================================
# Fine
# ==============================================================================

banner \
    "GCP TRANSFER FINAL SINGLE REPLICA: PASS"

echo "mode=$MODE"
echo "target=$TARGET_FUNCTION"
echo "donor=${SELECTED_DONOR:-none}"
echo "block_donor=${BLOCK_SELECTED_DONOR:-none}"
echo "replica=${BOOTSTRAP_REPLICA_ID:-none}"
echo "artifacts=$RESULT_DIR"

echo
echo "Le VM restano accese."

echo "Cleanup completo:"
echo "  cd ${SCRIPT_DIR} && ./gcp_down.sh"