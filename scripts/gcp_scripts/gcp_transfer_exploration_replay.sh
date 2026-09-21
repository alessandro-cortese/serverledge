#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"

TARGET_FUNCTION="${TARGET_FUNCTION:-}"
REPLICA_BOOTSTRAP_DIR="${REPLICA_BOOTSTRAP_DIR:-}"
SOURCE_RESULT_DIR="${SOURCE_RESULT_DIR:-}"

[[ -n "$TARGET_FUNCTION" ]] || {
    echo "FATAL: TARGET_FUNCTION mancante" >&2
    exit 1
}

[[ -d "$REPLICA_BOOTSTRAP_DIR" ]] || {
    echo "FATAL: REPLICA_BOOTSTRAP_DIR non valido: $REPLICA_BOOTSTRAP_DIR" >&2
    exit 1
}

[[ -d "$SOURCE_RESULT_DIR" ]] || {
    echo "FATAL: SOURCE_RESULT_DIR non valido: $SOURCE_RESULT_DIR" >&2
    exit 1
}

SOURCE_RESPONSE="${SOURCE_RESULT_DIR}/transfer/control-response.json"
SOURCE_READINESS="${SOURCE_RESULT_DIR}/donor/donor-readiness.json"
SOURCE_SUMMARY="${SOURCE_RESULT_DIR}/measure/summary.json"

[[ -f "$SOURCE_RESPONSE" ]] || {
    echo "FATAL: control-response sorgente assente: $SOURCE_RESPONSE" >&2
    exit 1
}

[[ -f "$SOURCE_READINESS" ]] || {
    echo "FATAL: donor-readiness sorgente assente: $SOURCE_READINESS" >&2
    exit 1
}

[[ -f "$SOURCE_SUMMARY" ]] || {
    echo "FATAL: source coupled run non completo: $SOURCE_SUMMARY assente" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Verifica source run prima di costruire il replay.
# ---------------------------------------------------------------------------

python3 - \
    "$SOURCE_RESPONSE" \
    "$SOURCE_READINESS" \
    "$TARGET_FUNCTION" <<'PY'
import json
import math
import sys

response_path, readiness_path, target = sys.argv[1:]

with open(response_path, encoding="utf-8") as f:
    response = json.load(f)

with open(readiness_path, encoding="utf-8") as f:
    readiness = json.load(f)

if response.get("transfer_applied") is not True:
    raise SystemExit(
        "source run non ha transfer_applied=true"
    )

prior = response.get("prior")

if not isinstance(prior, dict):
    raise SystemExit(
        "source response non contiene prior materializzato"
    )

if prior.get("has_prior") is not True:
    raise SystemExit(
        "source prior has_prior != true"
    )

if prior.get("policy") != "UCB1":
    raise SystemExit(
        f"source prior policy={prior.get('policy')}, atteso UCB1"
    )

config = prior.get("config") or {}

weight = float(
    config.get("equivalent_observation_weight", 0.0)
)

if not math.isclose(
    weight,
    0.25,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        f"source prior weight={weight}, atteso 0.25"
    )

arms = prior.get("arms") or {}

for arm in ("amd64", "arm64"):
    payload = arms.get(arm)

    if not isinstance(payload, dict):
        raise SystemExit(
            f"source prior manca arm {arm}"
        )

    if payload.get("transferred") is not True:
        raise SystemExit(
            f"source prior arm {arm} non transferred"
        )

    ucb = payload.get("ucb1")

    if not isinstance(ucb, dict):
        raise SystemExit(
            f"source prior arm {arm} manca payload UCB1"
        )

status = readiness.get("status")

if status != "ready":
    raise SystemExit(
        f"source donor readiness={status}, atteso ready"
    )

obs = readiness.get("observations") or {}

for arm in ("amd64", "arm64"):
    value = int(obs.get(arm, 0))

    if value < 10:
        raise SystemExit(
            f"source donor {arm} observations={value} < 10"
        )

print("PASS — source c=.8 valido per materialized replay")
print(
    "donor=",
    prior.get("donor_function_name"),
)
print(
    "source_observations=",
    prior.get("source_real_observation_count"),
)
PY

# ---------------------------------------------------------------------------
# Bootstrap temporaneo per l'ablation.
#
# NON modifichiamo il bootstrap congelato.
# Cambiamo soltanto coupled.c da 0.8 a 0.0 nella copia usata dal runner,
# perché c è precisamente la variabile indipendente dell'ablation.
# ---------------------------------------------------------------------------

TMP_BOOTSTRAP="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/serverledge-ablation-bootstrap.XXXXXX"
)"

TMP_SCRIPT="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-ablation-c00.XXXXXX.sh"
)"

cleanup() {
    rm -rf "$TMP_BOOTSTRAP"
    rm -f "$TMP_SCRIPT"
}

trap cleanup EXIT

cp -a \
    "${REPLICA_BOOTSTRAP_DIR}/." \
    "${TMP_BOOTSTRAP}/"

python3 - \
    "${TMP_BOOTSTRAP}/bootstrap.json" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, encoding="utf-8") as f:
    doc = json.load(f)

try:
    coupled = (
        doc["experimental_block"]
        ["policies"]
        ["coupled"]
    )
except KeyError as exc:
    raise SystemExit(
        f"bootstrap coupled block mancante: {exc}"
    )

original_c = float(coupled["c"])

if abs(original_c - 0.8) > 1e-12:
    raise SystemExit(
        f"bootstrap originale coupled.c={original_c}, atteso 0.8"
    )

coupled["c"] = 0.0

with open(path, "w", encoding="utf-8") as f:
    json.dump(
        doc,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print(
    "PASS — ablation bootstrap copy coupled.c=0"
)
PY

# ---------------------------------------------------------------------------
# Generiamo una copia temporanea del runner già validato.
# ---------------------------------------------------------------------------

cp \
    "${SCRIPT_DIR}/gcp_transfer_final.sh" \
    "$TMP_SCRIPT"

python3 - \
    "$TMP_SCRIPT" \
    "$ABLATION_RUNTIME_COMMIT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
commit = sys.argv[2]

text = path.read_text(encoding="utf-8")

# Commit dedicato all'instrumentation dell'ablation.
old_commit = (
    'EXPECTED_COMMIT="${EXPECTED_COMMIT:-'
    '08600a177ce1d88c048251a4b6d52d31fe65a536}"'
)

new_commit = (
    'EXPECTED_COMMIT="${EXPECTED_COMMIT:-'
    + commit
    + '}"'
)

if text.count(old_commit) != 1:
    raise SystemExit(
        "EXPECTED_COMMIT replacement count inatteso"
    )

text = text.replace(
    old_commit,
    new_commit,
    1,
)

# Starter dedicato: stessa logica originale, commit atteso diverso.
old_start = "./gcp_start_transfer_cluster.sh"
new_start = "./gcp_start_exploration_ablation_cluster.sh"

if text.count(old_start) != 1:
    raise SystemExit(
        "starter replacement count inatteso: "
        + str(text.count(old_start))
    )

text = text.replace(
    old_start,
    new_start,
    1,
)

# Nome artefatti inequivocabile.
old_run_id = (
    'RUN_ID="gcp-transfer-final-${TARGET_FUNCTION}-'
    '${MODE}-$(date +%Y%m%d_%H%M%S)"'
)

new_run_id = (
    'RUN_ID="gcp-transfer-ablation-${TARGET_FUNCTION}-'
    'c00-$(date +%Y%m%d_%H%M%S)"'
)

if text.count(old_run_id) != 1:
    raise SystemExit(
        "RUN_ID replacement count inatteso"
    )

text = text.replace(
    old_run_id,
    new_run_id,
    1,
)

start_marker = """# ==============================================================================
# Transfer conditions
# ==============================================================================
"""

end_marker = """# ==============================================================================
# Normalizzazione misura
"""

start = text.find(start_marker)

if start < 0:
    raise SystemExit(
        "inizio Transfer conditions non trovato"
    )

end = text.find(
    end_marker,
    start,
)

if end < 0:
    raise SystemExit(
        "inizio Normalizzazione misura non trovato"
    )

replacement = r'''# ==============================================================================
# Transfer conditions — MATERIALIZED PRIOR REPLAY
# ==============================================================================

DONOR_AMD64_UPDATES=0
DONOR_ARM64_UPDATES=0

CONTROL_REQUEST="${TRANSFER_DIR}/materialized-request.json"
CONTROL_RESPONSE="${TRANSFER_DIR}/materialized-response.json"

FROZEN_PRIOR="${TRANSFER_DIR}/frozen-prior.json"
RETURNED_PRIOR="${TRANSFER_DIR}/returned-prior.json"

SOURCE_RESPONSE_COPY="${TRANSFER_DIR}/source-control-response.json"
SOURCE_READINESS_COPY="${DONOR_DIR}/source-donor-readiness.json"

cp \
    "$MATERIALIZED_SOURCE_RESPONSE" \
    "$SOURCE_RESPONSE_COPY"

cp \
    "$MATERIALIZED_SOURCE_READINESS" \
    "$SOURCE_READINESS_COPY"

read -r \
    DONOR_AMD64_UPDATES \
    DONOR_ARM64_UPDATES \
    < <(
        python3 - \
            "$MATERIALIZED_SOURCE_READINESS" <<'PYREADINESS'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    doc = json.load(f)

obs = doc.get("observations") or {}

print(
    int(obs.get("amd64", 0)),
    int(obs.get("arm64", 0)),
)
PYREADINESS
    )

banner \
    "FREEZE MATERIALIZED PRIOR"

python3 - \
    "$MATERIALIZED_SOURCE_RESPONSE" \
    "$FROZEN_PRIOR" \
    "$CONTROL_REQUEST" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" <<'PYPRIOR'
import json
import math
import sys

(
    source_path,
    prior_path,
    request_path,
    target,
    expected_donor,
) = sys.argv[1:]

with open(source_path, encoding="utf-8") as f:
    response = json.load(f)

if response.get("transfer_applied") is not True:
    raise SystemExit(
        "source response transfer_applied != true"
    )

prior = response.get("prior")

if not isinstance(prior, dict):
    raise SystemExit(
        "source response non contiene prior"
    )

if prior.get("policy") != "UCB1":
    raise SystemExit(
        f"prior policy={prior.get('policy')}, atteso UCB1"
    )

if prior.get("has_prior") is not True:
    raise SystemExit(
        "prior has_prior != true"
    )

donor = str(
    prior.get("donor_function_name", "")
).strip()

if donor != expected_donor:
    raise SystemExit(
        f"prior donor={donor}, bootstrap donor={expected_donor}"
    )

config = prior.get("config") or {}

weight = float(
    config.get(
        "equivalent_observation_weight",
        0.0,
    )
)

if not math.isclose(
    weight,
    0.25,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        f"prior equivalent weight={weight}, atteso 0.25"
    )

arms = prior.get("arms") or {}

for arm in ("amd64", "arm64"):
    arm_prior = arms.get(arm)

    if not isinstance(arm_prior, dict):
        raise SystemExit(
            f"prior arm {arm} assente"
        )

    if arm_prior.get("transferred") is not True:
        raise SystemExit(
            f"prior arm {arm} non transferred"
        )

    payload = arm_prior.get("ucb1")

    if not isinstance(payload, dict):
        raise SystemExit(
            f"prior arm {arm} manca payload UCB1"
        )

with open(
    prior_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        prior,
        f,
        sort_keys=True,
        separators=(",", ":"),
    )

request = {
    "target_function_name": target,
    "prior": prior,
}

with open(
    request_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        request,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print("materialized prior: PASS")
print(f"donor={donor}")
print(
    "source_real_observations="
    + str(
        prior.get(
            "source_real_observation_count",
            0,
        )
    )
)
PYPRIOR

SOURCE_PRIOR_SHA="$(
    sha256sum \
        "$FROZEN_PRIOR" \
    | awk '{print $1}'
)"

printf '%s  %s\n' \
    "$SOURCE_PRIOR_SHA" \
    "frozen-prior.json" \
    >"${TRANSFER_DIR}/frozen-prior.sha256"

echo \
    "  frozen prior sha256=${SOURCE_PRIOR_SHA}"

banner \
    "APPLY MATERIALIZED PRIOR — TARGET c=0"

remote_request="/tmp/${RUN_ID}-materialized-request.json"
remote_response="/tmp/${RUN_ID}-materialized-response.json"

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
            'http://${LB_IP}:1323/mab/transfer/initialize-materialized'
    )

    echo \"HTTP_CODE=\$HTTP_CODE\"

    [[ \"\$HTTP_CODE\" == '200' ]]
"

scp_from \
    "$NAME_WORKLOAD" \
    "$remote_response" \
    "$CONTROL_RESPONSE"

python3 - \
    "$MATERIALIZED_SOURCE_RESPONSE" \
    "$CONTROL_RESPONSE" \
    "$RETURNED_PRIOR" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" <<'PYVERIFY'
import json
import sys

(
    source_path,
    replay_path,
    returned_prior_path,
    target,
    expected_donor,
) = sys.argv[1:]

with open(source_path, encoding="utf-8") as f:
    source = json.load(f)

with open(replay_path, encoding="utf-8") as f:
    replay = json.load(f)

if replay.get("transfer_attempted") is not True:
    raise SystemExit(
        "materialized response transfer_attempted != true"
    )

if replay.get("transfer_applied") is not True:
    raise SystemExit(
        "materialized response transfer_applied != true"
    )

if replay.get("initialization_source") != "materialized_prior":
    raise SystemExit(
        "initialization_source inattesa: "
        + str(
            replay.get("initialization_source")
        )
    )

if replay.get("target_function_name") != target:
    raise SystemExit(
        "materialized response target mismatch"
    )

if replay.get("donor_function_name") != expected_donor:
    raise SystemExit(
        "materialized response donor mismatch"
    )

if replay.get("policy") != "UCB1":
    raise SystemExit(
        "materialized response policy != UCB1"
    )

source_prior = source.get("prior")
replayed_prior = replay.get("prior")

if source_prior != replayed_prior:
    raise SystemExit(
        "FATAL: prior materializzato diverso dal prior sorgente"
    )

with open(
    returned_prior_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        replayed_prior,
        f,
        sort_keys=True,
        separators=(",", ":"),
    )

print(
    "PASS — response prior identico "
    "al prior sorgente"
)
PYVERIFY

RETURNED_PRIOR_SHA="$(
    sha256sum \
        "$RETURNED_PRIOR" \
    | awk '{print $1}'
)"

printf '%s  %s\n' \
    "$RETURNED_PRIOR_SHA" \
    "returned-prior.json" \
    >"${TRANSFER_DIR}/returned-prior.sha256"

[[ "$SOURCE_PRIOR_SHA" == "$RETURNED_PRIOR_SHA" ]] \
    || fail \
        "prior SHA mismatch: source=${SOURCE_PRIOR_SHA} replay=${RETURNED_PRIOR_SHA}"

echo \
    "  exact prior SHA equality: PASS"

printf '%s\n' \
    '{' \
    '  "schema_version": 1,' \
    '  "status": "materialized-prior-replay",' \
    "  \"target_function\": \"${TARGET_FUNCTION}\"," \
    "  \"donor_function\": \"${SELECTED_DONOR}\"," \
    '  "donor_recollected": false,' \
    "  \"source_observations_amd64\": ${DONOR_AMD64_UPDATES}," \
    "  \"source_observations_arm64\": ${DONOR_ARM64_UPDATES}," \
    "  \"prior_sha256\": \"${SOURCE_PRIOR_SHA}\"" \
    '}' \
    >"${DONOR_DIR}/donor-readiness.json"

TARGET_UPDATES_AFTER_PRIOR="$(
    target_update_count
)"

[[ "$TARGET_UPDATES_AFTER_PRIOR" == "0" ]] \
    || fail \
        "target ha real feedback prima della misura: ${TARGET_UPDATES_AFTER_PRIOR}"

echo \
    "  target real feedback after materialized prior=0: PASS"

'''

text = (
    text[:start]
    + replacement
    + "\n"
    + text[end:]
)

path.write_text(
    text,
    encoding="utf-8",
)
PY

chmod +x "$TMP_SCRIPT"

# ---------------------------------------------------------------------------
# Esecuzione.
#
# coupled rimane la policy UCB1 classica.
# La sola variabile MAB modificata è c=0.
# ---------------------------------------------------------------------------

MATERIALIZED_SOURCE_RESPONSE="$SOURCE_RESPONSE" \
MATERIALIZED_SOURCE_READINESS="$SOURCE_READINESS" \
TARGET_FUNCTION="$TARGET_FUNCTION" \
REPLICA_BOOTSTRAP_DIR="$TMP_BOOTSTRAP" \
N_X86="${N_X86:-1}" \
N_ARM="${N_ARM:-1}" \
MAB_UCB1_C=0.0 \
EQUIVALENT_PRIOR_WEIGHT=0.25 \
MIN_DONOR_OBSERVATIONS=10 \
"$TMP_SCRIPT" coupled
