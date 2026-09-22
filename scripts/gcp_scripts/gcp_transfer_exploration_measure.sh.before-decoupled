#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="${SCRIPT_DIR}/gcp_transfer_final.sh"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"

TARGET_FUNCTION="${TARGET_FUNCTION:-}"
MATERIALIZED_DIR="${MATERIALIZED_DIR:-}"
TARGET_C="${TARGET_C:-}"
PATCH_ONLY="${PATCH_ONLY:-0}"

# ==============================================================================
# c ammessi
# ==============================================================================

case "$TARGET_C" in

    0|0.0|0.00)

        TARGET_C="0.0"
        C_TAG="c00"
        ;;

    .8|0.8|0.80)

        TARGET_C="0.8"
        C_TAG="c08"
        ;;

    "")

        if [[ "$PATCH_ONLY" == "1" ]]; then

            TARGET_C="0.0"
            C_TAG="c00"

        else

            echo \
                "FATAL: TARGET_C mancante (ammessi 0.0 o 0.8)" >&2

            exit 1
        fi
        ;;

    *)

        echo \
            "FATAL: TARGET_C=$TARGET_C non valido (ammessi 0.0 o 0.8)" >&2

        exit 1
        ;;
esac

TMP_BOOTSTRAP="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/serverledge-materialized-bootstrap.XXXXXX"
)"

TMP_SCRIPT="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-measure-${C_TAG}.XXXXXX.sh"
)"

cleanup() {
    rm -rf \
        "$TMP_BOOTSTRAP"

    rm -f \
        "$TMP_SCRIPT"
}

trap cleanup EXIT

[[ -f "$SOURCE_SCRIPT" ]] || {
    echo \
        "FATAL: runner assente: $SOURCE_SCRIPT" >&2
    exit 1
}

cp \
    "$SOURCE_SCRIPT" \
    "$TMP_SCRIPT"

# ==============================================================================
# Generazione runner materialized
# ==============================================================================

python3 - \
    "$TMP_SCRIPT" \
    "$ABLATION_RUNTIME_COMMIT" \
    "$C_TAG" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
commit = sys.argv[2]
c_tag = sys.argv[3]

text = path.read_text(
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Runtime commit.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Starter ablation.
# ---------------------------------------------------------------------------

old_start = (
    "./gcp_start_transfer_cluster.sh"
)

new_start = (
    "./gcp_start_exploration_ablation_cluster.sh"
)

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

# ---------------------------------------------------------------------------
# Run ID.
# ---------------------------------------------------------------------------

old_run_id = (
    'RUN_ID="gcp-transfer-final-${TARGET_FUNCTION}-'
    '${MODE}-$(date +%Y%m%d_%H%M%S)"'
)

new_run_id = (
    'RUN_ID="gcp-transfer-materialized-${TARGET_FUNCTION}-'
    + c_tag
    + '-$(date +%Y%m%d_%H%M%S)"'
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

# ---------------------------------------------------------------------------
# Sostituiamo COMPLETAMENTE:
#
#   DONOR LIVE KNOWLEDGE
#   build-prior
#   /mab/transfer/initialize
#
# con il replay del prior congelato.
# ---------------------------------------------------------------------------

start_marker = """# ==============================================================================
# Transfer conditions
# ==============================================================================
"""

end_marker = """# ==============================================================================
# Normalizzazione misura
"""

start = text.find(
    start_marker,
)

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
# Transfer conditions — REPLAY EXACT MATERIALIZED PRIOR
# ==============================================================================

[[ "$MODE" == "coupled" ]] \
    || fail \
        "materialized measurement richiede MODE=coupled"

DONOR_AMD64_UPDATES=0
DONOR_ARM64_UPDATES=0

CONTROL_REQUEST="${TRANSFER_DIR}/materialized-request.json"

CONTROL_RESPONSE="${TRANSFER_DIR}/materialized-response.json"

FROZEN_PRIOR_COPY="${TRANSFER_DIR}/frozen-prior.json"

RETURNED_PRIOR="${TRANSFER_DIR}/returned-prior.json"

SOURCE_READINESS_COPY="${DONOR_DIR}/source-donor-readiness.json"

# ---------------------------------------------------------------------------
# Bundle.
# ---------------------------------------------------------------------------

[[ -f "$MATERIALIZED_PRIOR_FILE" ]] \
    || fail \
        "materialized prior assente: $MATERIALIZED_PRIOR_FILE"

[[ -f "$MATERIALIZED_READINESS_FILE" ]] \
    || fail \
        "materialized readiness assente: $MATERIALIZED_READINESS_FILE"

[[ -f "$MATERIALIZED_PRIOR_SHA_FILE" ]] \
    || fail \
        "materialized prior sha assente: $MATERIALIZED_PRIOR_SHA_FILE"

cp \
    "$MATERIALIZED_PRIOR_FILE" \
    "$FROZEN_PRIOR_COPY"

cp \
    "$MATERIALIZED_READINESS_FILE" \
    "$SOURCE_READINESS_COPY"

# ---------------------------------------------------------------------------
# Recuperiamo soltanto i CONTATORI donor per audit.
#
# Non viene effettuata alcuna richiesta donor.
# ---------------------------------------------------------------------------

read -r \
    DONOR_AMD64_UPDATES \
    DONOR_ARM64_UPDATES \
    < <(
        python3 - \
            "$MATERIALIZED_READINESS_FILE" <<'PYREADINESS'
import json
import sys

with open(
    sys.argv[1],
    encoding="utf-8",
) as f:
    doc = json.load(f)

if doc.get("status") != "ready":
    raise SystemExit(
        "materialized donor readiness="
        + str(doc.get("status"))
        + ", atteso ready"
    )

obs = (
    doc.get("observations")
    or {}
)

print(
    int(
        obs.get(
            "amd64",
            0,
        )
    ),
    int(
        obs.get(
            "arm64",
            0,
        )
    ),
)
PYREADINESS
    )

(( DONOR_AMD64_UPDATES >= 10 )) \
    || fail \
        "materialized donor amd64 observations < 10"

(( DONOR_ARM64_UPDATES >= 10 )) \
    || fail \
        "materialized donor arm64 observations < 10"

banner \
    "APPLY EXACT MATERIALIZED PRIOR — TARGET c=${TARGET_C}"

# ---------------------------------------------------------------------------
# Ricostruiamo SOLTANTO la request HTTP usando il prior congelato.
# Non viene ricalcolato nessun reward.
# ---------------------------------------------------------------------------

python3 - \
    "$MATERIALIZED_PRIOR_FILE" \
    "$CONTROL_REQUEST" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" <<'PYREQUEST'
import json
import math
import sys

(
    prior_path,
    request_path,
    target,
    expected_donor,
) = sys.argv[1:]

with open(
    prior_path,
    encoding="utf-8",
) as f:
    prior = json.load(f)

if prior.get(
    "has_prior"
) is not True:
    raise SystemExit(
        "materialized prior has_prior != true"
    )

if prior.get(
    "policy"
) != "UCB1":
    raise SystemExit(
        "materialized prior policy="
        + str(prior.get("policy"))
        + ", atteso UCB1"
    )

donor = str(
    prior.get(
        "donor_function_name",
        "",
    )
).strip()

if donor != expected_donor:
    raise SystemExit(
        f"materialized prior donor={donor}, "
        f"bootstrap donor={expected_donor}"
    )

config = (
    prior.get("config")
    or {}
)

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
        f"materialized prior weight={weight}, "
        "atteso 0.25"
    )

arms = (
    prior.get("arms")
    or {}
)

for arm in (
    "amd64",
    "arm64",
):
    payload = arms.get(arm)

    if not isinstance(
        payload,
        dict,
    ):
        raise SystemExit(
            f"materialized prior manca arm {arm}"
        )

    if payload.get(
        "transferred"
    ) is not True:
        raise SystemExit(
            f"materialized prior arm {arm} "
            "non transferred"
        )

    if not isinstance(
        payload.get("ucb1"),
        dict,
    ):
        raise SystemExit(
            f"materialized prior arm {arm} "
            "manca payload UCB1"
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
PYREQUEST

# ---------------------------------------------------------------------------
# SHA del bundle PRIMA di inviarlo.
# ---------------------------------------------------------------------------

EXPECTED_PRIOR_SHA="$(
    awk \
        'NF {print $1; exit}' \
        "$MATERIALIZED_PRIOR_SHA_FILE"
)"

SOURCE_PRIOR_SHA="$(
    sha256sum \
        "$MATERIALIZED_PRIOR_FILE" \
    | awk '{print $1}'
)"

[[ -n "$EXPECTED_PRIOR_SHA" ]] \
    || fail \
        "prior.sha256 vuoto"

[[ "$SOURCE_PRIOR_SHA" == "$EXPECTED_PRIOR_SHA" ]] \
    || fail \
        "bundle prior SHA mismatch: expected=${EXPECTED_PRIOR_SHA} actual=${SOURCE_PRIOR_SHA}"

echo \
    "  frozen prior sha256=${SOURCE_PRIOR_SHA}: PASS"

# ---------------------------------------------------------------------------
# Replay nel BanditManager fresco.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Il runtime deve restituire ESATTAMENTE lo stesso prior.
# ---------------------------------------------------------------------------

python3 - \
    "$MATERIALIZED_PRIOR_FILE" \
    "$CONTROL_RESPONSE" \
    "$RETURNED_PRIOR" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" <<'PYVERIFY'
import json
import sys

(
    prior_path,
    response_path,
    returned_path,
    target,
    expected_donor,
) = sys.argv[1:]

with open(
    prior_path,
    encoding="utf-8",
) as f:
    frozen = json.load(f)

with open(
    response_path,
    encoding="utf-8",
) as f:
    response = json.load(f)

if response.get(
    "transfer_attempted"
) is not True:
    raise SystemExit(
        "materialized response "
        "transfer_attempted != true"
    )

if response.get(
    "transfer_applied"
) is not True:
    raise SystemExit(
        "materialized response "
        "transfer_applied != true"
    )

if response.get(
    "initialization_source"
) != "materialized_prior":
    raise SystemExit(
        "initialization_source inattesa: "
        + str(
            response.get(
                "initialization_source"
            )
        )
    )

if response.get(
    "target_function_name"
) != target:
    raise SystemExit(
        "materialized response "
        "target mismatch"
    )

if response.get(
    "donor_function_name"
) != expected_donor:
    raise SystemExit(
        "materialized response "
        "donor mismatch"
    )

if response.get(
    "policy"
) != "UCB1":
    raise SystemExit(
        "materialized response "
        "policy != UCB1"
    )

returned = response.get(
    "prior"
)

if frozen != returned:
    raise SystemExit(
        "FATAL: returned prior "
        "diverso dal frozen prior"
    )

with open(
    returned_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        returned,
        f,
        sort_keys=True,
        separators=(",", ":"),
    )

print(
    "PASS — response prior "
    "identico al frozen prior"
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
        "prior SHA mismatch: source=${SOURCE_PRIOR_SHA} returned=${RETURNED_PRIOR_SHA}"

echo \
    "  exact prior SHA equality: PASS"

# ---------------------------------------------------------------------------
# Audit:
#
# donor_recollected=false è una proprietà fondamentale di R02/R03.
# ---------------------------------------------------------------------------

printf '%s\n' \
    '{' \
    '  "schema_version": 1,' \
    '  "status": "materialized-prior-replay",' \
    "  \"target_function\": \"${TARGET_FUNCTION}\"," \
    "  \"donor_function\": \"${SELECTED_DONOR}\"," \
    "  \"target_c\": ${TARGET_C}," \
    '  "donor_recollected": false,' \
    "  \"source_observations_amd64\": ${DONOR_AMD64_UPDATES}," \
    "  \"source_observations_arm64\": ${DONOR_ARM64_UPDATES}," \
    "  \"prior_sha256\": \"${SOURCE_PRIOR_SHA}\"" \
    '}' \
    >"${DONOR_DIR}/donor-readiness.json"

# ---------------------------------------------------------------------------
# Nessun feedback target prima della prima richiesta sperimentale.
# ---------------------------------------------------------------------------

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

chmod +x \
    "$TMP_SCRIPT"

bash -n \
    "$TMP_SCRIPT"

echo \
    "PASS — generated measurement runner (${TARGET_C}): bash -n"

# ==============================================================================
# PATCH ONLY
# ==============================================================================

if [[ "$PATCH_ONLY" == "1" ]]; then

    echo \
        "PASS — PATCH_ONLY=1: nessuna VM avviata"

    exit 0
fi

# ==============================================================================
# Bundle materializzato
# ==============================================================================

[[ -n "$TARGET_FUNCTION" ]] || {
    echo \
        "FATAL: TARGET_FUNCTION mancante" >&2
    exit 1
}

[[ -d "$MATERIALIZED_DIR" ]] || {
    echo \
        "FATAL: MATERIALIZED_DIR non valido: $MATERIALIZED_DIR" >&2
    exit 1
}

MATERIALIZED_DIR="$(
    cd "$MATERIALIZED_DIR"
    pwd
)"

MATERIALIZED_PRIOR_FILE="${MATERIALIZED_DIR}/frozen-prior.json"

MATERIALIZED_READINESS_FILE="${MATERIALIZED_DIR}/donor-readiness.json"

MATERIALIZED_PRIOR_SHA_FILE="${MATERIALIZED_DIR}/prior.sha256"

MATERIALIZED_MANIFEST_FILE="${MATERIALIZED_DIR}/manifest.json"

BOOTSTRAP_SNAPSHOT_DIR="${MATERIALIZED_DIR}/bootstrap"

for f in \
    "$MATERIALIZED_PRIOR_FILE" \
    "$MATERIALIZED_READINESS_FILE" \
    "$MATERIALIZED_PRIOR_SHA_FILE" \
    "$MATERIALIZED_MANIFEST_FILE" \
    "${BOOTSTRAP_SNAPSHOT_DIR}/bootstrap.json"
do

    [[ -f "$f" ]] || {
        echo \
            "FATAL: file bundle assente: $f" >&2
        exit 1
    }
done

# ==============================================================================
# Preflight locale
# ==============================================================================

python3 - \
    "$MATERIALIZED_MANIFEST_FILE" \
    "$MATERIALIZED_PRIOR_FILE" \
    "$MATERIALIZED_PRIOR_SHA_FILE" \
    "$TARGET_FUNCTION" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

(
    manifest_path,
    prior_path,
    sha_path,
    target,
) = sys.argv[1:]

with open(
    manifest_path,
    encoding="utf-8",
) as f:
    manifest = json.load(f)

with open(
    prior_path,
    encoding="utf-8",
) as f:
    prior = json.load(f)

if manifest.get(
    "status"
) != "materialized":
    raise SystemExit(
        "manifest status="
        + str(
            manifest.get("status")
        )
        + ", atteso materialized"
    )

if manifest.get(
    "target_function"
) != target:
    raise SystemExit(
        "manifest target="
        + str(
            manifest.get(
                "target_function"
            )
        )
        + f", atteso {target}"
    )

if manifest.get(
    "policy"
) != "UCB1":
    raise SystemExit(
        "manifest policy != UCB1"
    )

if not math.isclose(
    float(
        manifest.get(
            "source_c",
            -1.0,
        )
    ),
    0.8,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        "manifest source_c != 0.8"
    )

prior_donor = str(
    prior.get(
        "donor_function_name",
        "",
    )
).strip()

if prior_donor != manifest.get(
    "donor_function"
):
    raise SystemExit(
        "manifest/prior donor mismatch"
    )

actual = hashlib.sha256(
    pathlib.Path(
        prior_path
    ).read_bytes()
).hexdigest()

expected = (
    pathlib.Path(
        sha_path
    )
    .read_text(
        encoding="utf-8"
    )
    .split()[0]
)

if actual != expected:
    raise SystemExit(
        "prior SHA mismatch prima di GCP: "
        f"expected={expected} "
        f"actual={actual}"
    )

if actual != manifest.get(
    "prior_sha256"
):
    raise SystemExit(
        "manifest/prior SHA mismatch"
    )

print(
    "PASS — materialized bundle preflight"
)

print(
    f"target={target}"
)

print(
    f"donor={prior_donor}"
)

print(
    f"prior_sha256={actual}"
)
PY

# ==============================================================================
# Bootstrap
#
# La fonte NON è più il bootstrap esterno:
# usiamo la fotografia congelata durante la materializzazione.
#
# Modifichiamo esclusivamente coupled.c.
# ==============================================================================

cp -a \
    "${BOOTSTRAP_SNAPSHOT_DIR}/." \
    "$TMP_BOOTSTRAP/"

python3 - \
    "${TMP_BOOTSTRAP}/bootstrap.json" \
    "$TARGET_C" <<'PY'
import json
import math
import sys

path = sys.argv[1]
target_c = float(
    sys.argv[2]
)

with open(
    path,
    encoding="utf-8",
) as f:
    doc = json.load(f)

try:
    coupled = (
        doc[
            "experimental_block"
        ][
            "policies"
        ][
            "coupled"
        ]
    )
except KeyError as exc:
    raise SystemExit(
        f"bootstrap coupled block mancante: {exc}"
    )

original_c = float(
    coupled["c"]
)

if not math.isclose(
    original_c,
    0.8,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        f"bootstrap snapshot coupled.c={original_c}, "
        "atteso 0.8"
    )

coupled["c"] = target_c

with open(
    path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        doc,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print(
    f"PASS — bootstrap measurement coupled.c={target_c}"
)
PY

# ==============================================================================
# Esecuzione
# ==============================================================================

TARGET_FUNCTION="$TARGET_FUNCTION" \
REPLICA_BOOTSTRAP_DIR="$TMP_BOOTSTRAP" \
MATERIALIZED_PRIOR_FILE="$MATERIALIZED_PRIOR_FILE" \
MATERIALIZED_READINESS_FILE="$MATERIALIZED_READINESS_FILE" \
MATERIALIZED_PRIOR_SHA_FILE="$MATERIALIZED_PRIOR_SHA_FILE" \
TARGET_C="$TARGET_C" \
MAB_UCB1_C="$TARGET_C" \
EQUIVALENT_PRIOR_WEIGHT=0.25 \
MIN_DONOR_OBSERVATIONS=10 \
"$TMP_SCRIPT" coupled