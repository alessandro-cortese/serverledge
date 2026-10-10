#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/gcp_config.sh"
source "${SCRIPT_DIR}/functions_catalog.sh"

if [[ $# -ne 3 ]]; then
    echo \
        "Uso: $0 <target-function> <replica-label> <condition>" \
        >&2

    echo \
        "condition = notl-difference | difference | notl-ratio | ratio" \
        >&2

    exit 2
fi

TARGET_FUNCTION="$1"
REPLICA_LABEL="$2"
CONDITION="$3"

case "$CONDITION" in
    notl-difference|difference|notl-ratio|ratio)
        ;;
    *)
        echo "FATAL: condition non valida: $CONDITION" >&2
        exit 2
        ;;
esac


PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

ANALYSIS_HELPER="${ROOT_DIR}/analysis/profiling/gcp_transfer_final_analysis.py"

RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/data/profiling/gcp-ucb1-final}"

REPLICA_ID="${TARGET_FUNCTION}-${REPLICA_LABEL}"
REPLICA_DIR="${RESULT_ROOT}/replicas/${REPLICA_ID}"

BOOTSTRAP="${REPLICA_DIR}/bootstrap.json"
EVIDENCE_DIR="${REPLICA_DIR}/evidence"

MEASURE_REQUESTS="${MEASURE_REQUESTS:-50}"

EXPERIMENT_BLOCK="${EXPERIMENT_BLOCK:-unpaired}"
ORDER_POSITION="${ORDER_POSITION:-0}"

NAME_LB="sl-lb"
NAME_WORKLOAD="sl-workload"

X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"

POLICY="UCB1Decoupled"

RUN_ID="gcp-ucb1-final-${TARGET_FUNCTION}-${REPLICA_LABEL}-${CONDITION}-$(date +%Y%m%d_%H%M%S)"

RESULT_DIR="${RESULT_ROOT}/runs/${TARGET_FUNCTION}/${REPLICA_LABEL}/${RUN_ID}"

MEASURE_DIR="${RESULT_DIR}/measure"
TRANSFER_DIR="${RESULT_DIR}/transfer"
PREWARM_DIR="${RESULT_DIR}/prewarm"

mkdir -p \
    "$MEASURE_DIR" \
    "$TRANSFER_DIR" \
    "$PREWARM_DIR"


banner() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}


fail() {
    echo "FATAL: $*" >&2
    echo "Artefatti: $RESULT_DIR" >&2
    echo "Le VM restano intatte." >&2
    echo "Cleanup: cd ${SCRIPT_DIR} && ./gcp_down.sh" >&2
    exit 1
}


on_error() {
    local status=$?

    echo >&2
    echo \
        "ERRORE: gcp_measure_ucb1_final.sh exit=${status}" \
        >&2

    echo \
        "Artefatti parziali: ${RESULT_DIR}" \
        >&2

    exit "$status"
}

trap on_error ERR


remote() {
    local host="$1"
    shift

    gcloud compute ssh "$host" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        --command="$*"
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


direct_prewarm() {
    local arm="$1"
    local ip="$2"

    local remote_json="/tmp/${RUN_ID}-prewarm-${arm}.json"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        cd /opt/serverledge/examples/experiments

        /opt/serverledge/bin/serverledge-cli prewarm \
            -f '${TARGET_FUNCTION}' \
            -c 1 \
            -H '${ip}' \
            -P 1323 \
            > '${remote_json}'

        python3 - '${remote_json}' <<'PY'
import json
import sys

doc = json.load(
    open(sys.argv[1], encoding='utf-8')
)

prewarmed = doc.get('Prewarmed')

if (
    not isinstance(prewarmed, int)
    or prewarmed < 1
):
    raise SystemExit(
        f'prewarm failed: {doc}'
    )
PY
    " >/dev/null

    scp_from \
        "$NAME_WORKLOAD" \
        "$remote_json" \
        "${PREWARM_DIR}/${arm}.json"
}

wait_for_target_warm() {
    local arm="$1"
    local worker="$2"

    local max_attempts=60
    local sleep_seconds=1
    local attempt
    local status_json
    local warm_count

    echo \
        "  attendo warm readiness ${arm} target=${TARGET_FUNCTION}"

    for attempt in $(seq 1 "$max_attempts"); do

        status_json="$(
            remote "$worker" \
                "curl -fsS \
                    --max-time 5 \
                    http://127.0.0.1:1323/status" \
                2>/dev/null \
                || true
        )"

        warm_count="$(
            printf '%s' "$status_json" \
            | "$PY" -c '
import json
import sys

raw = sys.stdin.read().strip()

try:
    doc = json.loads(raw)
except Exception:
    print(-1)
    raise SystemExit(0)

warm = doc.get("AvailableWarmContainers") or {}

value = warm.get(sys.argv[1], 0)

try:
    print(int(value))
except Exception:
    print(-1)
' "$TARGET_FUNCTION"
        )"

        if [[ "$warm_count" =~ ^[0-9]+$ ]] \
            && (( warm_count >= 1 )); then

            echo \
                "  ${arm}: PASS target warm containers=${warm_count}"

            return 0
        fi

        sleep "$sleep_seconds"
    done

    fail \
        "target warm readiness timeout: arm=${arm} function=${TARGET_FUNCTION}"
}


# ==============================================================================
# Local preflight
# ==============================================================================

[[ -x "$PY" ]] \
    || fail "Python venv non trovato"

[[ -f "$ANALYSIS_HELPER" ]] \
    || fail "analysis helper non trovato"

[[ -f "$BOOTSTRAP" ]] \
    || fail "bootstrap non trovato"

[[ -f "${EVIDENCE_DIR}/evidence.json" ]] \
    || fail \
        "donor evidence non presente: eseguire prima gcp_collect_ucb1_final_evidence.sh"

[[ "$MEASURE_REQUESTS" =~ ^[1-9][0-9]*$ ]] \
    || fail "MEASURE_REQUESTS deve essere > 0"

(( MEASURE_REQUESTS >= 50 )) \
    || fail \
        "la campagna finale richiede almeno 50 richieste per preservare H=50"


# ==============================================================================
# Load frozen configuration
# ==============================================================================

read -r \
    BOOT_TARGET \
    SELECTED_DONOR \
    BOOT_COMMIT \
    MAB_UCB1_C \
    REQUIRES_PRIOR \
    PRIOR_MODE \
    REWARD_WEIGHT \
    EXPLORATION_WEIGHT \
    SELECTION_SHA \
    < <(
        "$PY" - \
            "$BOOTSTRAP" \
            "$TARGET_FUNCTION" \
            "$CONDITION" <<'PY'

import json
import sys

path, expected_target, condition = sys.argv[1:]

doc = json.load(
    open(path, encoding="utf-8")
)

target = doc["target_function"]

if target != expected_target:
    raise SystemExit(
        f"target mismatch: {target} != {expected_target}"
    )

conditions = (
    doc["experimental_block"]["conditions"]
)

if condition not in conditions:
    raise SystemExit(
        f"condition assente dal bootstrap: {condition}"
    )

cfg = conditions[condition]

prior = bool(
    cfg.get("prior", False)
)

print(
    target,
    doc["donor_selection"]["selected_donor"],
    doc["serverledge_commit"],
    cfg["c"],
    "true" if prior else "false",
    cfg.get("mode", "none"),
    cfg.get(
        "reward_observation_weight",
        0.0,
    ),
    cfg.get(
        "exploration_observation_weight",
        0.0,
    ),
    doc["donor_selection"][
        "selection_artifact_sha256"
    ],
)
PY
    )


EXPECTED_COMMIT="${EXPECTED_COMMIT:-$BOOT_COMMIT}"

# graph-bfs-r01 was prepared/evidenced on the original frozen runtime.
# A technical prewarm lifecycle bug was discovered before any accepted
# measurement. Permit only this exact frozen runtime transition.
TECHNICAL_FIX_BASE_COMMIT="4a7359156af6ad3d83841c2935efebd67905cb21"
TECHNICAL_FIX_RUNTIME_COMMIT="4c9f427a76e188c9cb026d0b65ccf44b4d116d6f"

TECHNICAL_RUNTIME_FIX="false"
TECHNICAL_FIX_PATH=""
TECHNICAL_FIX_REASON=""

if [[ "$EXPECTED_COMMIT" != "$BOOT_COMMIT" ]]; then
    [[ "$BOOT_COMMIT" == "$TECHNICAL_FIX_BASE_COMMIT" ]] \
        || fail "bootstrap commit non ammesso per technical runtime fix"

    [[ "$EXPECTED_COMMIT" == "$TECHNICAL_FIX_RUNTIME_COMMIT" ]] \
        || fail "measurement runtime non ammesso per technical runtime fix"

    TECHNICAL_RUNTIME_FIX="true"
    TECHNICAL_FIX_PATH="internal/node/pool.go"
    TECHNICAL_FIX_REASON="initialize_expiration_for_prewarmed_containers"
fi

cat >"${RESULT_DIR}/runtime-provenance.txt" <<EOF_RUNTIME_PROVENANCE
bootstrap_serverledge_commit=${BOOT_COMMIT}
measurement_serverledge_commit=${EXPECTED_COMMIT}
technical_runtime_fix=${TECHNICAL_RUNTIME_FIX}
technical_fix_path=${TECHNICAL_FIX_PATH}
technical_fix_reason=${TECHNICAL_FIX_REASON}
EOF_RUNTIME_PROVENANCE


PRIOR_SHA=""

if [[ "$REQUIRES_PRIOR" == "true" ]]; then

    BUNDLE_DIR="${EVIDENCE_DIR}/${PRIOR_MODE}"

    [[ -f "${BUNDLE_DIR}/frozen-prior.json" ]] \
        || fail "prior bundle non trovato: $BUNDLE_DIR"

    [[ -f "${BUNDLE_DIR}/materialized-request.json" ]] \
        || fail "materialized-request.json assente"

    PRIOR_SHA="$(
        sha256sum \
            "${BUNDLE_DIR}/frozen-prior.json" \
        | awk '{print $1}'
    )"

else
    BUNDLE_DIR=""
fi


# ==============================================================================
# Verify bundle/condition before touching GCP
# ==============================================================================

"$PY" - \
    "$BOOTSTRAP" \
    "${EVIDENCE_DIR}/evidence.json" \
    "$CONDITION" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$MAB_UCB1_C" \
    "$REQUIRES_PRIOR" \
    "${BUNDLE_DIR:-}" <<'PY'

import hashlib
import json
import math
import sys

(
    bootstrap_path,
    evidence_path,
    condition,
    target,
    donor,
    c_raw,
    requires_prior_raw,
    bundle_raw,
) = sys.argv[1:]

bootstrap = json.load(
    open(bootstrap_path, encoding="utf-8")
)

evidence = json.load(
    open(evidence_path, encoding="utf-8")
)

cfg = (
    bootstrap["experimental_block"]
    ["conditions"][condition]
)

if evidence["target_function"] != target:
    raise SystemExit("evidence target mismatch")

if evidence["donor_function"] != donor:
    raise SystemExit("evidence donor mismatch")

if not math.isclose(
    float(cfg["c"]),
    float(c_raw),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("c mismatch")

requires_prior = (
    requires_prior_raw == "true"
)

if bool(cfg["prior"]) != requires_prior:
    raise SystemExit("prior flag mismatch")

if requires_prior:
    bundle = __import__(
        "pathlib"
    ).Path(bundle_raw)

    manifest = json.load(
        open(
            bundle / "manifest.json",
            encoding="utf-8",
        )
    )

    prior_path = (
        bundle / "frozen-prior.json"
    )

    actual_sha = hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()

    if manifest["target_function"] != target:
        raise SystemExit(
            "prior target mismatch"
        )

    if manifest["donor_function"] != donor:
        raise SystemExit(
            "prior donor mismatch"
        )

    if manifest["mode"] != cfg["mode"]:
        raise SystemExit(
            "prior formula mismatch"
        )

    if not math.isclose(
        float(manifest["target_c"]),
        float(cfg["c"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            "prior target_c mismatch"
        )

    if actual_sha != manifest["prior_sha256"]:
        raise SystemExit(
            "prior SHA mismatch"
        )

print("FINAL CONDITION PREFLIGHT: PASS")
PY


# ==============================================================================
# Fresh runtime
# ==============================================================================

banner "UCB1 FINAL MEASUREMENT"

echo "run=$RUN_ID"
echo "target=$TARGET_FUNCTION"
echo "replica=$REPLICA_ID"
echo "condition=$CONDITION"
echo "policy=$POLICY"
echo "c=$MAB_UCB1_C"
echo "prior=$REQUIRES_PRIOR"
echo "formula=$PRIOR_MODE"
echo "block=$EXPERIMENT_BLOCK"
echo "order_position=$ORDER_POSITION"

banner "RESET FRESH UCB1DECOUPLED RUNTIME"

(
    cd "$SCRIPT_DIR"

    EXPECTED_COMMIT="$EXPECTED_COMMIT" \
    N_X86=1 \
    N_ARM=1 \
    MAB_UCB1_C="$MAB_UCB1_C" \
    ./gcp_start_ucb1_final_cluster.sh measurement
)


LB_IP="$(internal_ip "$NAME_LB")"
X86_IP="$(internal_ip "$X86_WORKER")"
ARM_IP="$(internal_ip "$ARM_WORKER")"


# ==============================================================================
# Register target
# ==============================================================================

banner "REGISTER TARGET"

register_function "$TARGET_FUNCTION"


# ==============================================================================
# Apply materialized prior only in TL conditions
# ==============================================================================

if [[ "$REQUIRES_PRIOR" == "true" ]]; then

    banner "APPLY ${PRIOR_MODE^^} MATERIALIZED PRIOR"

    CONTROL_REQUEST="${BUNDLE_DIR}/materialized-request.json"
    CONTROL_RESPONSE="${TRANSFER_DIR}/materialized-response.json"

    cp \
        "${BUNDLE_DIR}/frozen-prior.json" \
        "${TRANSFER_DIR}/frozen-prior.json"

    cp \
        "${BUNDLE_DIR}/prior.sha256" \
        "${TRANSFER_DIR}/prior.sha256"

    cp \
        "$CONTROL_REQUEST" \
        "${TRANSFER_DIR}/materialized-request.json"

    REMOTE_REQUEST="/tmp/${RUN_ID}-materialized-request.json"
    REMOTE_RESPONSE="/tmp/${RUN_ID}-materialized-response.json"

    scp_to \
        "$CONTROL_REQUEST" \
        "$NAME_WORKLOAD" \
        "$REMOTE_REQUEST"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        HTTP_CODE=\$(
            curl \
                -sS \
                --max-time 30 \
                -o '${REMOTE_RESPONSE}' \
                -w '%{http_code}' \
                -H 'Content-Type: application/json' \
                --data-binary '@${REMOTE_REQUEST}' \
                'http://${LB_IP}:1323/mab/transfer/initialize-materialized'
        )

        echo \"HTTP_CODE=\$HTTP_CODE\"

        [[ \"\$HTTP_CODE\" == '200' ]]
    "

    scp_from \
        "$NAME_WORKLOAD" \
        "$REMOTE_RESPONSE" \
        "$CONTROL_RESPONSE"

    "$PY" - \
        "$CONTROL_REQUEST" \
        "$CONTROL_RESPONSE" \
        "$TARGET_FUNCTION" \
        "$SELECTED_DONOR" <<'PY'

import json
import sys

(
    request_path,
    response_path,
    target,
    donor,
) = sys.argv[1:]

request = json.load(
    open(request_path, encoding="utf-8")
)

response = json.load(
    open(response_path, encoding="utf-8")
)

if response.get(
    "transfer_attempted"
) is not True:
    raise SystemExit(
        "transfer_attempted != true"
    )

if response.get(
    "transfer_applied"
) is not True:
    raise SystemExit(
        "transfer_applied != true"
    )

if response.get(
    "initialization_source"
) != "materialized_prior":
    raise SystemExit(
        "initialization_source mismatch"
    )

if response.get(
    "target_function_name"
) != target:
    raise SystemExit(
        "target response mismatch"
    )

if response.get(
    "donor_function_name"
) != donor:
    raise SystemExit(
        "donor response mismatch"
    )

# Semantic comparison, intentionally not raw-byte SHA:
# Go may serialize JSON numbers differently.
if response.get("prior") != request.get("prior"):
    raise SystemExit(
        "returned prior differs from requested prior"
    )

with open(
    response_path + ".returned-prior.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        response["prior"],
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print(
    "MATERIALIZED PRIOR APPLY: PASS"
)
PY

else

    cat >"${TRANSFER_DIR}/status.json" <<EOF
{
  "schema_version": 1,
  "condition": "${CONDITION}",
  "prior_applied": false,
  "policy": "UCB1Decoupled",
  "c": ${MAB_UCB1_C}
}
EOF

fi


# ==============================================================================
# Normalize warm state
#
# Direct calls do NOT cross the MAB.
# ==============================================================================

banner "PREWARM TARGET DIRECTLY ON BOTH ARMS"

direct_prewarm \
    amd64 \
    "$X86_IP"

direct_prewarm \
    arm64 \
    "$ARM_IP"

banner "WAIT TARGET WARM READINESS"

wait_for_target_warm \
    amd64 \
    "$X86_WORKER"

wait_for_target_warm \
    arm64 \
    "$ARM_WORKER"

echo \
    "  warm containers visibili; attendo 5s di stabilizzazione"

sleep 5

wait_for_target_warm \
    amd64 \
    "$X86_WORKER"

wait_for_target_warm \
    arm64 \
    "$ARM_WORKER"

TARGET_UPDATES_PRE="$(
    remote "$NAME_LB" "
        sudo grep -c \
            'event=update_reward.*function=${TARGET_FUNCTION}' \
            /var/log/serverledge-lb.log \
        || true
    " | tr -d '[:space:]'
)"

[[ "${TARGET_UPDATES_PRE:-0}" == "0" ]] \
    || fail \
        "target ha real feedback prima della misura: ${TARGET_UPDATES_PRE}"

echo "target feedback pre-measure=0: PASS"


# ==============================================================================
# Mark beginning of measured LB log
# ==============================================================================

LB_LINES_BEFORE="$(
    remote "$NAME_LB" \
        "sudo wc -l < /var/log/serverledge-lb.log" \
    | tr -d '[:space:]'
)"

[[ "$LB_LINES_BEFORE" =~ ^[0-9]+$ ]] \
    || fail "LB line count non valido"


# ==============================================================================
# Measurement client
# ==============================================================================

banner "MEASURE ${MEASURE_REQUESTS} WARM TARGET REQUESTS"

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
    {"params": {}}
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
                    "application/json",
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
            timeout=300,
        ) as response:

            body = response.read()

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            node_arch = (
                response.headers.get(
                    "Serverledge-Node-Arch"
                )
                or ""
            ).strip()

        document = json.loads(
            body.decode("utf-8")
        )

        if document.get("Success") is not True:
            raise SystemExit(
                f"request {index} failed: "
                f"{document}"
            )

        if document.get(
            "IsWarmStart"
        ) is not True:
            raise SystemExit(
                f"request {index}: "
                "cold invocation inside "
                "measured horizon"
            )

        if not node_arch:
            raise SystemExit(
                f"request {index}: "
                "Serverledge-Node-Arch "
                "header missing"
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


REMOTE_MEASURE_SCRIPT="/tmp/${RUN_ID}-measure.py"
REMOTE_REQUEST_CSV="/tmp/${RUN_ID}-requests.csv"

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
# Extract exactly the measured LB log portion
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
# Reuse validated MAB log summarizer
# ==============================================================================

banner "SUMMARIZE"

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
        decoupled \
    --policy \
        UCB1Decoupled \
    --output-csv \
        "${MEASURE_DIR}/mab-requests.csv" \
    --output-json \
        "${MEASURE_DIR}/summary.json"


EVIDENCE_SOURCE_SHA="$(
    awk '{print $1}' \
        "${EVIDENCE_DIR}/source/prior.sha256"
)"


# ==============================================================================
# Enrich summary + write machine-readable manifest
# ==============================================================================

"$PY" - \
    "${MEASURE_DIR}/summary.json" \
    "${RESULT_DIR}/manifest.json" \
    "$RUN_ID" \
    "$REPLICA_ID" \
    "$CONDITION" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$MAB_UCB1_C" \
    "$REQUIRES_PRIOR" \
    "$PRIOR_MODE" \
    "$REWARD_WEIGHT" \
    "$EXPLORATION_WEIGHT" \
    "$PRIOR_SHA" \
    "$EVIDENCE_SOURCE_SHA" \
    "$SELECTION_SHA" \
    "$EXPECTED_COMMIT" \
    "$EXPERIMENT_BLOCK" \
    "$ORDER_POSITION" <<'PY'

import json
import sys

(
    summary_path,
    manifest_path,
    run_id,
    replica_id,
    condition,
    target,
    donor,
    c_raw,
    prior_raw,
    formula,
    wr_raw,
    we_raw,
    prior_sha,
    evidence_sha,
    selection_sha,
    commit,
    experiment_block,
    order_position_raw,
) = sys.argv[1:]


with open(
    summary_path,
    encoding="utf-8",
) as f:
    summary = json.load(f)


prior_applied = (
    prior_raw == "true"
)


summary.update(
    {
        "experiment":
            "gcp-ucb1-final",

        "condition":
            condition,

        "replica_id":
            replica_id,

        "donor_function":
            donor,

        "c":
            float(c_raw),

        "prior_applied":
            prior_applied,

        "transfer_formula":
            None
            if formula == "none"
            else formula,

        "reward_observation_weight":
            (
                float(wr_raw)
                if prior_applied
                else None
            ),

        "exploration_observation_weight":
            (
                float(we_raw)
                if prior_applied
                else None
            ),

        "prior_sha256":
            prior_sha or None,

        "evidence_source_sha256":
            evidence_sha,

        "bootstrap_selection_sha256":
            selection_sha,

        "serverledge_commit":
            commit,

        "experiment_block":
            experiment_block,

        "order_position":
            int(order_position_raw),
    }
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


manifest = {
    "schema_version": 1,

    "experiment":
        "gcp-ucb1-final",

    "run_id":
        run_id,

    "replica_id":
        replica_id,

    "condition":
        condition,

    "policy":
        "UCB1Decoupled",

    "target_function":
        target,

    "donor_function":
        donor,

    "c":
        float(c_raw),

    "prior_applied":
        prior_applied,

    "transfer_formula":
        None
        if formula == "none"
        else formula,

    "reward_observation_weight":
        (
            float(wr_raw)
            if prior_applied
            else None
        ),

    "exploration_observation_weight":
        (
            float(we_raw)
            if prior_applied
            else None
        ),

    "prior_sha256":
        prior_sha or None,

    "evidence_source_sha256":
        evidence_sha,

    "bootstrap_selection_sha256":
        selection_sha,

    "serverledge_commit":
        commit,

    "experiment_block":
        experiment_block,

    "order_position":
        int(order_position_raw),

    "request_count":
        summary["request_count"],

    "primary_horizon":
        10,

    "secondary_horizons":
        [5, 20, 50],

    "all_warm":
        summary["all_warm"],

    "fallback_count":
        summary["fallback_count"],
}


with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


h10 = summary["horizons"]["10"]

print()
print("===== PRIMARY H=10 =====")
print(
    "cumulative_duration_ms=",
    h10["cumulative_duration_ms"],
)
print(
    "amd64_executions=",
    h10["amd64_executions"],
)
print(
    "arm64_executions=",
    h10["arm64_executions"],
)
print(
    "fallbacks=",
    h10["fallbacks"],
)
PY


trap - ERR


banner "UCB1 FINAL MEASUREMENT: PASS"

echo "condition=$CONDITION"
echo "target=$TARGET_FUNCTION"
echo "donor=$SELECTED_DONOR"
echo "c=$MAB_UCB1_C"
echo "prior=$REQUIRES_PRIOR"
echo "formula=$PRIOR_MODE"
echo "requests=$MEASURE_REQUESTS"
echo "results=$RESULT_DIR"

echo
echo "Le VM restano accese."
echo "Cleanup: cd ${SCRIPT_DIR} && ./gcp_down.sh"