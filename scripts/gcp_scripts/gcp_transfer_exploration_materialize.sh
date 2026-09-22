#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="${SCRIPT_DIR}/gcp_transfer_final.sh"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"
TARGET_FUNCTION="${TARGET_FUNCTION:-}"
REPLICA_BOOTSTRAP_DIR="${REPLICA_BOOTSTRAP_DIR:-}"
MATERIALIZED_OUTPUT_DIR="${MATERIALIZED_OUTPUT_DIR:-}"
PATCH_ONLY="${PATCH_ONLY:-0}"

TMP_SCRIPT="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-materialize.XXXXXX.sh"
)"

cleanup() {
    rm -f "$TMP_SCRIPT"
}

trap cleanup EXIT

[[ -f "$SOURCE_SCRIPT" ]] || {
    echo "FATAL: runner assente: $SOURCE_SCRIPT" >&2
    exit 1
}

cp \
    "$SOURCE_SCRIPT" \
    "$TMP_SCRIPT"

python3 - \
    "$TMP_SCRIPT" \
    "$ABLATION_RUNTIME_COMMIT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
commit = sys.argv[2]

text = path.read_text(
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Commit dell'instrumentation già validata nell'R01.
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
# Starter dedicato all'ablation.
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
# Nome run inequivocabile.
# ---------------------------------------------------------------------------

old_run_id = (
    'RUN_ID="gcp-transfer-final-${TARGET_FUNCTION}-'
    '${MODE}-$(date +%Y%m%d_%H%M%S)"'
)

new_run_id = (
    'RUN_ID="gcp-transfer-materialize-${TARGET_FUNCTION}-'
    '$(date +%Y%m%d_%H%M%S)"'
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
# Interrompiamo il runner PRIMA della normalizzazione/misura target.
#
# Il gcp_transfer_final originale avrà già:
#
#   1. avviato il cluster;
#   2. registrato target;
#   3. raccolto donor live knowledge;
#   4. verificato readiness;
#   5. costruito il weak prior;
#   6. applicato il prior al target;
#   7. verificato target_update_count == 0.
#
# A questo punto congeliamo tutto ed usciamo.
# ---------------------------------------------------------------------------

marker = """# ==============================================================================
# Normalizzazione misura
"""

pos = text.find(marker)

if pos < 0:
    raise SystemExit(
        "marker Normalizzazione misura non trovato"
    )

export_block = r'''# ==============================================================================
# EXPORT MATERIALIZED PRIOR — stop before target measurement
# ==============================================================================

[[ "$MODE" == "coupled" ]] \
    || fail \
        "materialization richiede MODE=coupled"

[[ -n "${MATERIALIZED_OUTPUT_DIR:-}" ]] \
    || fail \
        "MATERIALIZED_OUTPUT_DIR mancante"

[[ -n "${REPLICA_BOOTSTRAP_DIR:-}" ]] \
    || fail \
        "REPLICA_BOOTSTRAP_DIR mancante"

banner \
    "EXPORT MATERIALIZED PRIOR"

mkdir -p \
    "$MATERIALIZED_OUTPUT_DIR"

FROZEN_PRIOR="${MATERIALIZED_OUTPUT_DIR}/frozen-prior.json"

PRIOR_SHA_FILE="${MATERIALIZED_OUTPUT_DIR}/prior.sha256"

SOURCE_RESPONSE_COPY="${MATERIALIZED_OUTPUT_DIR}/source-control-response.json"

SOURCE_READINESS_COPY="${MATERIALIZED_OUTPUT_DIR}/donor-readiness.json"

MATERIALIZED_REQUEST="${MATERIALIZED_OUTPUT_DIR}/materialized-request.json"

MANIFEST="${MATERIALIZED_OUTPUT_DIR}/manifest.json"

BOOTSTRAP_SNAPSHOT="${MATERIALIZED_OUTPUT_DIR}/bootstrap"

[[ ! -e "$FROZEN_PRIOR" ]] \
    || fail \
        "bundle già materializzato: $FROZEN_PRIOR"

# ---------------------------------------------------------------------------
# Estraggo dal control-response il prior ESATTO che il runtime ha applicato.
# ---------------------------------------------------------------------------

python3 - \
    "$CONTROL_RESPONSE" \
    "$FROZEN_PRIOR" \
    "$MATERIALIZED_REQUEST" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" <<'PYMATERIALIZE'
import json
import math
import sys

(
    response_path,
    prior_path,
    request_path,
    target,
    expected_donor,
) = sys.argv[1:]

with open(
    response_path,
    encoding="utf-8",
) as f:
    response = json.load(f)

if response.get("transfer_applied") is not True:
    raise SystemExit(
        "transfer_applied != true"
    )

prior = response.get("prior")

if not isinstance(prior, dict):
    raise SystemExit(
        "control response non contiene prior"
    )

if prior.get("has_prior") is not True:
    raise SystemExit(
        "prior has_prior != true"
    )

if prior.get("policy") != "UCB1":
    raise SystemExit(
        f"prior policy={prior.get('policy')}, "
        "atteso UCB1"
    )

donor = str(
    prior.get(
        "donor_function_name",
        "",
    )
).strip()

if donor != expected_donor:
    raise SystemExit(
        f"prior donor={donor}, "
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
        f"prior weight={weight}, "
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
            f"prior manca arm {arm}"
        )

    if payload.get(
        "transferred"
    ) is not True:
        raise SystemExit(
            f"prior arm {arm} "
            "non transferred"
        )

    if not isinstance(
        payload.get("ucb1"),
        dict,
    ):
        raise SystemExit(
            f"prior arm {arm} "
            "manca payload UCB1"
        )

# Canonical JSON:
# in questo modo lo SHA dipende solo dal prior,
# non dall'indentazione.
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

print(
    "materialized prior: PASS"
)

print(
    f"target={target}"
)

print(
    f"donor={donor}"
)

print(
    "source_real_observations="
    + str(
        prior.get(
            "source_real_observation_count",
            0,
        )
    )
)
PYMATERIALIZE

# ---------------------------------------------------------------------------
# Conserviamo anche gli artefatti sorgente.
# ---------------------------------------------------------------------------

cp \
    "$CONTROL_RESPONSE" \
    "$SOURCE_RESPONSE_COPY"

cp \
    "${DONOR_DIR}/donor-readiness.json" \
    "$SOURCE_READINESS_COPY"

# ---------------------------------------------------------------------------
# Congeliamo l'INTERO bootstrap della replica.
#
# I successivi c=0 e c=.8 partiranno da questa stessa fotografia;
# measure.sh modificherà soltanto coupled.c nella propria copia temporanea.
# ---------------------------------------------------------------------------

rm -rf \
    "$BOOTSTRAP_SNAPSHOT"

mkdir -p \
    "$BOOTSTRAP_SNAPSHOT"

cp -a \
    "${REPLICA_BOOTSTRAP_DIR}/." \
    "$BOOTSTRAP_SNAPSHOT/"

# ---------------------------------------------------------------------------
# SHA.
# ---------------------------------------------------------------------------

PRIOR_SHA="$(
    sha256sum \
        "$FROZEN_PRIOR" \
    | awk '{print $1}'
)"

printf '%s  %s\n' \
    "$PRIOR_SHA" \
    "frozen-prior.json" \
    >"$PRIOR_SHA_FILE"

BOOTSTRAP_SHA="$(
    sha256sum \
        "${BOOTSTRAP_SNAPSHOT}/bootstrap.json" \
    | awk '{print $1}'
)"

# ---------------------------------------------------------------------------
# Manifest del bundle.
# ---------------------------------------------------------------------------

python3 - \
    "$MANIFEST" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$PRIOR_SHA" \
    "$BOOTSTRAP_SHA" \
    "$DONOR_AMD64_UPDATES" \
    "$DONOR_ARM64_UPDATES" <<'PYMANIFEST'
import json
import sys

(
    path,
    target,
    donor,
    prior_sha,
    bootstrap_sha,
    amd64_obs,
    arm64_obs,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "status": "materialized",
    "target_function": target,
    "donor_function": donor,
    "policy": "UCB1",
    "source_c": 0.8,
    "equivalent_observation_weight": 0.25,
    "donor_observations": {
        "amd64": int(amd64_obs),
        "arm64": int(arm64_obs),
    },
    "prior_sha256": prior_sha,
    "bootstrap_sha256": bootstrap_sha,
    "target_real_feedback_before_export": 0,
}

with open(
    path,
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
PYMANIFEST

# ---------------------------------------------------------------------------
# Gate finale.
# ---------------------------------------------------------------------------

TARGET_UPDATES_EXPORT="$(
    target_update_count
)"

[[ "$TARGET_UPDATES_EXPORT" == "0" ]] \
    || fail \
        "target ha real feedback prima dell'export: ${TARGET_UPDATES_EXPORT}"

echo \
    "  target real feedback before export=0: PASS"

echo \
    "  prior sha256=${PRIOR_SHA}"

echo \
    "  bootstrap sha256=${BOOTSTRAP_SHA}"

echo \
    "  materialized bundle=${MATERIALIZED_OUTPUT_DIR}"

echo \
    "  STOP: nessuna misura target eseguita"

exit 0

'''

text = (
    text[:pos]
    + export_block
    + text[pos:]
)

path.write_text(
    text,
    encoding="utf-8",
)
PY

chmod +x \
    "$TMP_SCRIPT"

# Anche il runner generato deve essere Bash valido.
bash -n \
    "$TMP_SCRIPT"

echo \
    "PASS — generated materialization runner: bash -n"

# ---------------------------------------------------------------------------
# Modalità esclusivamente locale.
# ---------------------------------------------------------------------------

if [[ "$PATCH_ONLY" == "1" ]]; then

    echo \
        "PASS — PATCH_ONLY=1: nessuna VM avviata"

    exit 0
fi

# ---------------------------------------------------------------------------
# Validazioni runtime.
# ---------------------------------------------------------------------------

[[ -n "$TARGET_FUNCTION" ]] || {
    echo \
        "FATAL: TARGET_FUNCTION mancante" >&2
    exit 1
}

[[ -d "$REPLICA_BOOTSTRAP_DIR" ]] || {
    echo \
        "FATAL: REPLICA_BOOTSTRAP_DIR non valido: $REPLICA_BOOTSTRAP_DIR" >&2
    exit 1
}

[[ -f "${REPLICA_BOOTSTRAP_DIR}/bootstrap.json" ]] || {
    echo \
        "FATAL: bootstrap.json assente in $REPLICA_BOOTSTRAP_DIR" >&2
    exit 1
}

[[ -n "$MATERIALIZED_OUTPUT_DIR" ]] || {
    echo \
        "FATAL: MATERIALIZED_OUTPUT_DIR mancante" >&2
    exit 1
}

# Il bootstrap congelato deve essere quello standard c=.8.
python3 - \
    "${REPLICA_BOOTSTRAP_DIR}/bootstrap.json" <<'PY'
import json
import math
import sys

with open(
    sys.argv[1],
    encoding="utf-8",
) as f:
    doc = json.load(f)

try:
    c = float(
        doc[
            "experimental_block"
        ][
            "policies"
        ][
            "coupled"
        ][
            "c"
        ]
    )
except KeyError as exc:
    raise SystemExit(
        f"bootstrap coupled block mancante: {exc}"
    )

if not math.isclose(
    c,
    0.8,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        f"bootstrap coupled.c={c}, "
        "atteso 0.8"
    )

print(
    "PASS — bootstrap sorgente coupled.c=0.8"
)
PY

mkdir -p \
    "$MATERIALIZED_OUTPUT_DIR"

MATERIALIZED_OUTPUT_DIR="$(
    cd "$MATERIALIZED_OUTPUT_DIR"
    pwd
)"

REPLICA_BOOTSTRAP_DIR="$(
    cd "$REPLICA_BOOTSTRAP_DIR"
    pwd
)"

if [[ -e "${MATERIALIZED_OUTPUT_DIR}/frozen-prior.json" ]]; then

    echo \
        "FATAL: bundle già presente in $MATERIALIZED_OUTPUT_DIR" >&2

    exit 1
fi

# ---------------------------------------------------------------------------
# MATERIALIZATION.
# ---------------------------------------------------------------------------

TARGET_FUNCTION="$TARGET_FUNCTION" \
REPLICA_BOOTSTRAP_DIR="$REPLICA_BOOTSTRAP_DIR" \
MATERIALIZED_OUTPUT_DIR="$MATERIALIZED_OUTPUT_DIR" \
MAB_UCB1_C=0.8 \
EQUIVALENT_PRIOR_WEIGHT=0.25 \
MIN_DONOR_OBSERVATIONS=10 \
"$TMP_SCRIPT" coupled