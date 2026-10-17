#!/usr/bin/env bash

set -euo pipefail


# ==============================================================================
# Paths
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


# ==============================================================================
# Frozen runtime
# ==============================================================================

# Runtime usato dalla campagna GCP finale accettata.
FROZEN_RUNTIME_COMMIT="4c9f427a76e188c9cb026d0b65ccf44b4d116d6f"

# graph-bfs è stato preparato prima del fix tecnico del prewarm.
# Il runner finale originale consente esclusivamente questa transizione:
#
#   4a735... -> 4c9f427...
#
TECHNICAL_FIX_BASE_COMMIT="4a7359156af6ad3d83841c2935efebd67905cb21"


# ==============================================================================
# Arguments
# ==============================================================================

if [[ $# -ne 4 ]]; then
    echo \
        "Uso: $0 <B|C> <target> <replica> <condition>" \
        >&2

    echo \
        "condition = notl-difference | difference | notl-ratio | ratio" \
        >&2

    exit 2
fi


FAMILY="$1"
TARGET_FUNCTION="$2"
REPLICA_LABEL="$3"
CONDITION="$4"


# ==============================================================================
# Formula-matched family
# ==============================================================================

case "$FAMILY" in
    B)
        EXPECTED_C="0.2"
        EXPECTED_WR="0.05"
        EXPECTED_WE="64"
        ;;

    C)
        EXPECTED_C="0.4"
        EXPECTED_WR="0.01"
        EXPECTED_WE="64"
        ;;

    *)
        echo \
            "FATAL: family non valida: $FAMILY" \
            >&2

        exit 2
        ;;
esac


case "$CONDITION" in
    notl-difference|difference|notl-ratio|ratio)
        ;;

    *)
        echo \
            "FATAL: condition non valida: $CONDITION" \
            >&2

        exit 2
        ;;
esac


# ==============================================================================
# Experiment paths
# ==============================================================================

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

MATCHED_BASE="${MATCHED_BASE:-${ROOT_DIR}/data/profiling/gcp-ucb1-formula-matched}"
RESULT_ROOT="${MATCHED_BASE}/${FAMILY}"

BASE_RUNNER="${SCRIPT_DIR}/gcp_measure_ucb1_final.sh"

REPLICA_DIR="${RESULT_ROOT}/replicas/${TARGET_FUNCTION}-${REPLICA_LABEL}"

BOOTSTRAP="${REPLICA_DIR}/bootstrap.json"


# ==============================================================================
# Required local artifacts
# ==============================================================================

[[ -x "$PY" ]] || {
    echo \
        "FATAL: Python non trovato: $PY" \
        >&2

    exit 1
}


[[ -x "$BASE_RUNNER" ]] || {
    echo \
        "FATAL: runner base non eseguibile: $BASE_RUNNER" \
        >&2

    exit 1
}


[[ -f "$BOOTSTRAP" ]] || {
    echo \
        "FATAL: bootstrap non trovato: $BOOTSTRAP" \
        >&2

    exit 1
}


# ==============================================================================
# Frozen runtime gate
#
# Tutte le nuove misure devono essere eseguite con il runtime finale congelato
# 4c9f427...
#
# billibench-knn, dynamic-html-py e functionbench-gzip hanno già questo commit
# nel bootstrap.
#
# graph-bfs conserva invece il bootstrap 4a735..., ma il runner finale originale
# permette esplicitamente la transizione tecnica 4a735... -> 4c9f427....
# ==============================================================================

BOOT_COMMIT="$(
    "$PY" - "$BOOTSTRAP" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    doc = json.load(handle)

print(doc["serverledge_commit"])
PY
)"


if [[ "$BOOT_COMMIT" == "$FROZEN_RUNTIME_COMMIT" ]]; then

    echo "FROZEN RUNTIME GATE: PASS"
    echo "bootstrap_commit=$BOOT_COMMIT"
    echo "measurement_commit=$FROZEN_RUNTIME_COMMIT"

elif [[ \
    "$TARGET_FUNCTION" == "graph-bfs" \
    && "$BOOT_COMMIT" == "$TECHNICAL_FIX_BASE_COMMIT" \
]]; then

    echo "FROZEN RUNTIME GATE: PASS (technical transition)"
    echo "bootstrap_commit=$BOOT_COMMIT"
    echo "measurement_commit=$FROZEN_RUNTIME_COMMIT"

else

    echo \
        "FATAL: bootstrap/runtime non ammessi per la campagna congelata" \
        >&2

    echo \
        "bootstrap_commit=$BOOT_COMMIT" \
        >&2

    echo \
        "measurement_commit=$FROZEN_RUNTIME_COMMIT" \
        >&2

    exit 1
fi


# ==============================================================================
# Local formula-matched preflight
#
# Nessuna chiamata GCP viene effettuata in questa sezione.
#
# Verifica:
#
#   - target;
#   - c;
#   - formula;
#   - w_R;
#   - w_E;
#   - manifest;
#   - prior materializzato;
#   - SHA256 del prior.
# ==============================================================================

"$PY" - \
    "$BOOTSTRAP" \
    "$TARGET_FUNCTION" \
    "$CONDITION" \
    "$EXPECTED_C" \
    "$EXPECTED_WR" \
    "$EXPECTED_WE" \
    "$REPLICA_DIR" <<'PY'

import hashlib
import json
import math
import pathlib
import sys


(
    bootstrap_path,
    expected_target,
    condition,
    expected_c,
    expected_wr,
    expected_we,
    replica_dir_raw,
) = sys.argv[1:]


def close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

with open(
    bootstrap_path,
    encoding="utf-8",
) as handle:
    bootstrap = json.load(handle)


if bootstrap.get("target_function") != expected_target:
    raise SystemExit(
        "bootstrap target mismatch"
    )


conditions = (
    bootstrap[
        "experimental_block"
    ]["conditions"]
)


if condition not in conditions:
    raise SystemExit(
        f"condition missing: {condition}"
    )


cfg = conditions[condition]


if not close(
    cfg["c"],
    expected_c,
):
    raise SystemExit(
        "bootstrap c mismatch"
    )


requires_prior = bool(
    cfg.get("prior", False)
)


# ------------------------------------------------------------------------------
# Transfer Learning condition
# ------------------------------------------------------------------------------

if requires_prior:

    mode = cfg.get("mode")

    if mode not in (
        "difference",
        "ratio",
    ):
        raise SystemExit(
            "invalid TL formula"
        )


    if not close(
        cfg[
            "reward_observation_weight"
        ],
        expected_wr,
    ):
        raise SystemExit(
            "bootstrap w_R mismatch"
        )


    if not close(
        cfg[
            "exploration_observation_weight"
        ],
        expected_we,
    ):
        raise SystemExit(
            "bootstrap w_E mismatch"
        )


    replica_dir = pathlib.Path(
        replica_dir_raw
    )


    bundle = (
        replica_dir
        / "evidence"
        / mode
    )


    manifest_path = (
        bundle
        / "manifest.json"
    )


    prior_path = (
        bundle
        / "frozen-prior.json"
    )


    sha_path = (
        bundle
        / "prior.sha256"
    )


    request_path = (
        bundle
        / "materialized-request.json"
    )


    for path in (
        manifest_path,
        prior_path,
        sha_path,
        request_path,
    ):
        if not path.is_file():
            raise SystemExit(
                f"missing bundle artifact: {path}"
            )


    # --------------------------------------------------------------------------
    # Manifest
    # --------------------------------------------------------------------------

    with open(
        manifest_path,
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)


    if (
        manifest["target_function"]
        != expected_target
    ):
        raise SystemExit(
            "manifest target mismatch"
        )


    if manifest["mode"] != mode:
        raise SystemExit(
            "manifest mode mismatch"
        )


    if not close(
        manifest["target_c"],
        expected_c,
    ):
        raise SystemExit(
            "manifest c mismatch"
        )


    if not close(
        manifest[
            "reward_observation_weight"
        ],
        expected_wr,
    ):
        raise SystemExit(
            "manifest w_R mismatch"
        )


    if not close(
        manifest[
            "exploration_observation_weight"
        ],
        expected_we,
    ):
        raise SystemExit(
            "manifest w_E mismatch"
        )


    # --------------------------------------------------------------------------
    # Frozen prior
    # --------------------------------------------------------------------------

    with open(
        prior_path,
        encoding="utf-8",
    ) as handle:
        prior = json.load(handle)


    config = prior["config"]


    if (
        config[
            "ucb1_reference_anchor"
        ]["mode"]
        != mode
    ):
        raise SystemExit(
            "prior mode mismatch"
        )


    if not close(
        config[
            "reward_observation_weight"
        ],
        expected_wr,
    ):
        raise SystemExit(
            "prior w_R mismatch"
        )


    if not close(
        config[
            "exploration_observation_weight"
        ],
        expected_we,
    ):
        raise SystemExit(
            "prior w_E mismatch"
        )


    # --------------------------------------------------------------------------
    # SHA256
    # --------------------------------------------------------------------------

    actual_sha = hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()


    recorded_sha = (
        sha_path
        .read_text(
            encoding="utf-8"
        )
        .split()[0]
    )


    if actual_sha != recorded_sha:
        raise SystemExit(
            "prior.sha256 mismatch"
        )


    if (
        actual_sha
        != manifest["prior_sha256"]
    ):
        raise SystemExit(
            "manifest SHA mismatch"
        )


    print(
        "FORMULA-MATCHED LOCAL PREFLIGHT: PASS"
    )

    print(
        f"target={expected_target}"
    )

    print(
        f"condition={condition}"
    )

    print(
        f"formula={mode}"
    )

    print(
        f"c={float(expected_c):g}"
    )

    print(
        f"w_R={float(expected_wr):g}"
    )

    print(
        f"w_E={float(expected_we):g}"
    )


# ------------------------------------------------------------------------------
# No-TL condition
# ------------------------------------------------------------------------------

else:

    print(
        "FORMULA-MATCHED NO-TL PREFLIGHT: PASS"
    )

    print(
        f"target={expected_target}"
    )

    print(
        f"condition={condition}"
    )

    print(
        f"c={float(expected_c):g}"
    )

PY


# ==============================================================================
# Preflight-only mode
# ==============================================================================

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then

    echo
    echo "PREFLIGHT_ONLY: PASS"

    exit 0
fi


# ==============================================================================
# GCP measurement
# ==============================================================================

echo
echo "============================================================"
echo "GCP FORMULA-MATCHED MEASUREMENT"
echo "============================================================"
echo "family=$FAMILY"
echo "target=$TARGET_FUNCTION"
echo "replica=$REPLICA_LABEL"
echo "condition=$CONDITION"
echo "c=$EXPECTED_C"
echo "w_R=$EXPECTED_WR"
echo "w_E=$EXPECTED_WE"
echo "bootstrap_commit=$BOOT_COMMIT"
echo "measurement_commit=$FROZEN_RUNTIME_COMMIT"
echo "result_root=$RESULT_ROOT"


BLOCK_NAME="${EXPERIMENT_BLOCK:-formula-matched-${FAMILY}}"
REQUESTS="${MEASURE_REQUESTS:-50}"


# ==============================================================================
# Delegate to the accepted final GCP runner
#
# EXPECTED_COMMIT is deliberately forced to the frozen final runtime.
#
# In particular, graph-bfs must NOT run with its older bootstrap runtime
# 4a735...; the accepted final runner handles the allowed technical transition
# to 4c9f427....
# ==============================================================================

EXPECTED_COMMIT="$FROZEN_RUNTIME_COMMIT" \
RESULT_ROOT="$RESULT_ROOT" \
EXPERIMENT_BLOCK="$BLOCK_NAME" \
MEASURE_REQUESTS="$REQUESTS" \
"$BASE_RUNNER" \
    "$TARGET_FUNCTION" \
    "$REPLICA_LABEL" \
    "$CONDITION"