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

PY="${PY:-${ROOT_DIR}/.venv-analysis/bin/python}"

RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/data/profiling/gcp-ucb1-final}"

REPLICA_DIR="${RESULT_ROOT}/replicas/${REPLICA_ID}"
BOOTSTRAP="${REPLICA_DIR}/bootstrap.json"
EVIDENCE_DIR="${REPLICA_DIR}/evidence"

DONOR_SAMPLES_PER_ARM="${DONOR_SAMPLES_PER_ARM:-10}"

NAME_WORKLOAD="sl-workload"
X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"


banner() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}


fail() {
    echo "FATAL: $*" >&2
    echo "Le VM restano intatte." >&2
    echo "Cleanup: cd ${SCRIPT_DIR} && ./gcp_down.sh" >&2
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


# ==============================================================================
# Preflight
# ==============================================================================

[[ -x "$PY" ]] \
    || fail "Python venv non trovato: $PY"

[[ -f "$BOOTSTRAP" ]] \
    || fail "bootstrap non trovato: $BOOTSTRAP"

[[ "$DONOR_SAMPLES_PER_ARM" =~ ^[1-9][0-9]*$ ]] \
    || fail "DONOR_SAMPLES_PER_ARM deve essere > 0"

if [[ -d "$EVIDENCE_DIR" ]] \
    && find "$EVIDENCE_DIR" -mindepth 1 -print -quit | grep -q .; then

    fail "evidence già presente: $EVIDENCE_DIR"
fi

mkdir -p "$EVIDENCE_DIR"


# ==============================================================================
# Read frozen replica
# ==============================================================================

read -r \
    BOOT_TARGET \
    SELECTED_DONOR \
    TARGET_ANCHOR \
    BOOT_COMMIT \
    BOOT_SELECTION_SHA \
    < <(
        "$PY" - \
            "$BOOTSTRAP" \
            "$TARGET_FUNCTION" <<'PY'
import json
import sys

path, expected_target = sys.argv[1:]

doc = json.load(open(path, encoding="utf-8"))

if doc.get("experiment") != "gcp-ucb1-final":
    raise SystemExit("experiment bootstrap inatteso")

target = doc["target_function"]

if target != expected_target:
    raise SystemExit(
        f"target bootstrap={target} requested={expected_target}"
    )

sel = doc["donor_selection"]

print(
    target,
    sel["selected_donor"],
    doc["target_profile"]["target_reference_mean_reward"],
    doc["serverledge_commit"],
    sel["selection_artifact_sha256"],
)
PY
    )

EXPECTED_COMMIT="${EXPECTED_COMMIT:-$BOOT_COMMIT}"

[[ "$EXPECTED_COMMIT" == "$BOOT_COMMIT" ]] \
    || fail \
        "EXPECTED_COMMIT=$EXPECTED_COMMIT diverso dal bootstrap=$BOOT_COMMIT"

lookup_function "$SELECTED_DONOR" >/dev/null \
    || fail \
        "donor non presente nel functions catalog: $SELECTED_DONOR"


# ==============================================================================
# Start clean runtime
#
# c is irrelevant here because donor observations are collected DIRECTLY
# on each worker, never through the MAB.
# ==============================================================================

banner "DONOR EVIDENCE — ${REPLICA_ID}"

echo "target=$TARGET_FUNCTION"
echo "donor=$SELECTED_DONOR"
echo "samples_per_arm=$DONOR_SAMPLES_PER_ARM"
echo "anchor=$TARGET_ANCHOR"
echo "commit=$EXPECTED_COMMIT"

banner "RESET RUNTIME"

(
    cd "$SCRIPT_DIR"

    EXPECTED_COMMIT="$EXPECTED_COMMIT" \
    N_X86=1 \
    N_ARM=1 \
    MAB_UCB1_C=0.2 \
    ./gcp_start_ucb1_final_cluster.sh measurement
)


LB_IP="$(internal_ip sl-lb)"
X86_IP="$(internal_ip "$X86_WORKER")"
ARM_IP="$(internal_ip "$ARM_WORKER")"


# ==============================================================================
# Register donor
# ==============================================================================

banner "REGISTER DONOR"

register_function "$SELECTED_DONOR"


# ==============================================================================
# Direct balanced donor collector
# ==============================================================================

COLLECTOR="${EVIDENCE_DIR}/collect_direct_donor.py"

cat >"$COLLECTOR" <<'PY'
#!/usr/bin/env python3

import csv
import json
import math
import sys
import time

from urllib.request import Request, urlopen


host = sys.argv[1]
function_name = sys.argv[2]
sample_count = int(sys.argv[3])
output_csv = sys.argv[4]
output_jsonl = sys.argv[5]
arm = sys.argv[6]

url = f"http://{host}:1323/invoke/{function_name}"

payload = json.dumps(
    {"params": {}}
).encode("utf-8")


def invoke():
    request = Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=300,
    ) as response:
        body = response.read()

    document = json.loads(
        body.decode("utf-8")
    )

    if document.get("Success") is not True:
        raise SystemExit(
            f"donor invocation failed: {document}"
        )

    return document


# Cold/warm-up invocation excluded from evidence.
prewarm = invoke()

print(
    f"{arm}: prewarm "
    f"warm={prewarm.get('IsWarmStart')}"
)


with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8",
) as csv_handle, open(
    output_jsonl,
    "w",
    encoding="utf-8",
) as raw_handle:

    writer = csv.writer(csv_handle)

    writer.writerow(
        [
            "sample_index",
            "arm",
            "duration_ms",
            "response_time_ms",
            "reward",
            "warm_start",
        ]
    )

    for index in range(
        1,
        sample_count + 1,
    ):
        document = invoke()

        if document.get("IsWarmStart") is not True:
            raise SystemExit(
                f"{arm} sample {index}: "
                "expected warm invocation"
            )

        duration_seconds = float(
            document["Duration"]
        )

        if (
            not math.isfinite(duration_seconds)
            or duration_seconds <= 0
        ):
            raise SystemExit(
                f"{arm} sample {index}: "
                f"invalid Duration={duration_seconds}"
            )

        duration_ms = (
            duration_seconds * 1000.0
        )

        response_time_ms = (
            float(
                document.get(
                    "ResponseTime",
                    duration_seconds,
                )
            )
            * 1000.0
        )

        reward = -math.log(
            duration_ms
        )

        writer.writerow(
            [
                index,
                arm,
                f"{duration_ms:.12f}",
                f"{response_time_ms:.12f}",
                f"{reward:.15f}",
                "true",
            ]
        )

        raw_handle.write(
            json.dumps(
                document,
                sort_keys=True,
            )
            + "\n"
        )

        csv_handle.flush()
        raw_handle.flush()

        print(
            f"{arm}: "
            f"{index}/{sample_count} "
            f"duration_ms={duration_ms:.3f} "
            f"reward={reward:.6f}",
            flush=True,
        )

        time.sleep(0.10)
PY


REMOTE_COLLECTOR="/tmp/${REPLICA_ID}-collect-donor.py"

scp_to \
    "$COLLECTOR" \
    "$NAME_WORKLOAD" \
    "$REMOTE_COLLECTOR"


collect_arm() {
    local arm="$1"
    local ip="$2"

    local remote_csv="/tmp/${REPLICA_ID}-${arm}.csv"
    local remote_jsonl="/tmp/${REPLICA_ID}-${arm}.jsonl"

    remote "$NAME_WORKLOAD" "
        set -euo pipefail

        python3 \
            '${REMOTE_COLLECTOR}' \
            '${ip}' \
            '${SELECTED_DONOR}' \
            '${DONOR_SAMPLES_PER_ARM}' \
            '${remote_csv}' \
            '${remote_jsonl}' \
            '${arm}'
    "

    scp_from \
        "$NAME_WORKLOAD" \
        "$remote_csv" \
        "${EVIDENCE_DIR}/${arm}.csv"

    scp_from \
        "$NAME_WORKLOAD" \
        "$remote_jsonl" \
        "${EVIDENCE_DIR}/${arm}.jsonl"
}


banner "COLLECT DONOR AMD64"

collect_arm \
    amd64 \
    "$X86_IP"


banner "COLLECT DONOR ARM64"

collect_arm \
    arm64 \
    "$ARM_IP"


# ==============================================================================
# Build one common donor evidence + coupled Difference source bundle
# ==============================================================================

banner "BUILD COMMON DONOR EVIDENCE"

SOURCE_BUNDLE="${EVIDENCE_DIR}/source"

"$PY" - \
    "$REPLICA_DIR" \
    "$BOOTSTRAP" \
    "${EVIDENCE_DIR}/amd64.csv" \
    "${EVIDENCE_DIR}/arm64.csv" \
    "$SOURCE_BUNDLE" \
    "${EVIDENCE_DIR}/evidence.json" \
    "$TARGET_FUNCTION" \
    "$SELECTED_DONOR" \
    "$TARGET_ANCHOR" \
    "$DONOR_SAMPLES_PER_ARM" <<'PY'

import csv
import json
import math
import shutil
import statistics
import sys

from pathlib import Path

from analysis.profiling.transfer_materialized_prior import (
    sha256_file,
    write_json,
)


(
    replica_raw,
    bootstrap_raw,
    amd_raw,
    arm_raw,
    source_raw,
    evidence_raw,
    target,
    donor,
    anchor_raw,
    expected_count_raw,
) = sys.argv[1:]


replica = Path(replica_raw).resolve()
bootstrap = Path(bootstrap_raw).resolve()
amd_path = Path(amd_raw).resolve()
arm_path = Path(arm_raw).resolve()
source = Path(source_raw).resolve()
evidence_path = Path(evidence_raw).resolve()

anchor = float(anchor_raw)
expected_count = int(expected_count_raw)


def load_arm(path, expected_arm):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != expected_count:
        raise SystemExit(
            f"{expected_arm}: "
            f"rows={len(rows)}, "
            f"expected={expected_count}"
        )

    durations = []
    rewards = []

    for row in rows:
        if row["arm"] != expected_arm:
            raise SystemExit(
                f"arm mismatch in {path}"
            )

        if row["warm_start"].lower() != "true":
            raise SystemExit(
                f"non-warm evidence in {path}"
            )

        duration = float(
            row["duration_ms"]
        )

        reward = float(
            row["reward"]
        )

        if (
            not math.isfinite(duration)
            or duration <= 0
            or not math.isfinite(reward)
        ):
            raise SystemExit(
                f"invalid row in {path}: {row}"
            )

        expected_reward = -math.log(
            duration
        )

        if not math.isclose(
            reward,
            expected_reward,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise SystemExit(
                "reward != -ln(duration_ms)"
            )

        durations.append(duration)
        rewards.append(reward)

    return durations, rewards


amd_durations, amd_rewards = load_arm(
    amd_path,
    "amd64",
)

arm_durations, arm_rewards = load_arm(
    arm_path,
    "arm64",
)

donor_ref = statistics.fmean(
    amd_rewards
)

donor_other = statistics.fmean(
    arm_rewards
)

gap = donor_other - donor_ref

if abs(donor_ref) <= 1e-12:
    raise SystemExit(
        "donor reference mean reward ~0: "
        "Ratio undefined"
    )

ratio = donor_other / donor_ref


evidence = {
    "schema_version": 1,
    "status": "ready",
    "target_function": target,
    "donor_function": donor,

    "reward_definition":
        "-ln(duration_ms)",

    "target_reference_mean_reward":
        anchor,

    "observations": {
        "amd64": len(amd_rewards),
        "arm64": len(arm_rewards),
    },

    "donor": {
        "amd64": {
            "mean_duration_ms":
                statistics.fmean(
                    amd_durations
                ),
            "median_duration_ms":
                statistics.median(
                    amd_durations
                ),
            "mean_reward":
                donor_ref,
        },

        "arm64": {
            "mean_duration_ms":
                statistics.fmean(
                    arm_durations
                ),
            "median_duration_ms":
                statistics.median(
                    arm_durations
                ),
            "mean_reward":
                donor_other,
        },

        "reward_gap_arm64_minus_amd64":
            gap,

        "reward_ratio_arm64_over_amd64":
            ratio,
    },

    "provenance": {
        "collection":
            "direct-balanced-warm-invocations",
        "amd64_csv":
            str(amd_path),
        "arm64_csv":
            str(arm_path),
        "bootstrap":
            str(bootstrap),
        "bootstrap_sha256":
            sha256_file(bootstrap),
    },
}

write_json(
    evidence_path,
    evidence,
)


# ------------------------------------------------------------------
# Coupled Difference source used ONLY as an immutable provenance
# representation for transfer_materialized_tuned.
# ------------------------------------------------------------------

source.mkdir(
    parents=True,
    exist_ok=False,
)

source_bootstrap = (
    source / "bootstrap"
)

source_bootstrap.mkdir()

shutil.copy2(
    bootstrap,
    source_bootstrap
    / "bootstrap.json",
)

for name in (
    "selection",
    "profile",
):
    src = replica / name

    if src.is_dir():
        shutil.copytree(
            src,
            source_bootstrap / name,
        )


source_weight = 0.25

source_ref = anchor
source_other = anchor + gap

prior = {
    "schema_version": 1,
    "policy": "UCB1",
    "has_prior": True,

    "donor_function_name":
        donor,

    "source_real_observation_count":
        len(amd_rewards)
        + len(arm_rewards),

    "source_excluded_synthetic_observation_count":
        0,

    "arm_count": 2,
    "transferred_arm_count": 2,
    "skipped_arm_count": 0,

    "config": {
        "equivalent_observation_weight":
            source_weight,

        "min_real_observations_per_arm":
            expected_count,

        "ucb1_reference_anchor": {
            "enabled": True,
            "reference_arm": "amd64",
            "target_reference_mean_reward":
                anchor,
            "mode": "difference",
        },
    },

    "arms": {},
}


def source_arm(
    count,
    mean_reward,
):
    return {
        "transferred": True,

        "source_real_observation_count":
            count,

        "source_excluded_synthetic_observation_count":
            0,

        "applied_equivalent_observation_weight":
            source_weight,

        "applied_exploration_observation_weight":
            source_weight,

        "attenuation_scale":
            source_weight / count,

        "ucb1": {
            "observation_weight":
                source_weight,

            "exploration_observation_weight":
                source_weight,

            "mean_reward":
                mean_reward,

            "reward_sum":
                mean_reward
                * source_weight,
        },
    }


prior["arms"]["amd64"] = source_arm(
    len(amd_rewards),
    source_ref,
)

prior["arms"]["arm64"] = source_arm(
    len(arm_rewards),
    source_other,
)


prior_path = (
    source
    / "frozen-prior.json"
)

write_json(
    prior_path,
    prior,
    compact=True,
)

prior_sha = sha256_file(
    prior_path
)

(
    source
    / "prior.sha256"
).write_text(
    f"{prior_sha}  frozen-prior.json\n",
    encoding="utf-8",
)


readiness = {
    "schema_version": 1,
    "status": "ready",
    "target_function": target,
    "donor_function": donor,
    "collection":
        "direct-balanced-warm-invocations",
    "observations": {
        "amd64": len(amd_rewards),
        "arm64": len(arm_rewards),
    },
    "transfer_eligible": True,
}

write_json(
    source
    / "donor-readiness.json",
    readiness,
)


manifest = {
    "schema_version": 1,
    "status": "materialized-source",

    "target_function":
        target,

    "donor_function":
        donor,

    # Donor evidence was gathered directly,
    # therefore no source-side UCB exploration
    # coefficient generated these samples.
    "source_c": None,

    "prior_sha256":
        prior_sha,

    "donor_observations": {
        "amd64": len(amd_rewards),
        "arm64": len(arm_rewards),
    },

    "donor_reward_ref":
        donor_ref,

    "donor_reward_other":
        donor_other,

    "donor_reward_gap":
        gap,

    "target_real_feedback_before_export":
        0,
}

write_json(
    source / "manifest.json",
    manifest,
)


print(
    "donor_reward_ref="
    f"{donor_ref:.15f}"
)

print(
    "donor_reward_other="
    f"{donor_other:.15f}"
)

print(
    "donor_gap="
    f"{gap:.15f}"
)

print(
    "donor_ratio="
    f"{ratio:.15f}"
)

print(
    "source_prior_sha256="
    + prior_sha
)
PY


read -r \
    DONOR_REWARD_REF \
    DONOR_REWARD_OTHER \
    < <(
        "$PY" - \
            "${EVIDENCE_DIR}/evidence.json" <<'PY'
import json
import sys

doc = json.load(
    open(sys.argv[1], encoding="utf-8")
)

print(
    doc["donor"]["amd64"]["mean_reward"],
    doc["donor"]["arm64"]["mean_reward"],
)
PY
    )


# ==============================================================================
# Materialize BOTH final strategies from exact same evidence
# ==============================================================================

banner "MATERIALIZE DIFFERENCE"

"$PY" \
    -m analysis.profiling.transfer_materialized_tuned \
    --source-bundle "$SOURCE_BUNDLE" \
    --output-bundle "${EVIDENCE_DIR}/difference" \
    --mode difference \
    --reward-weight 0.05 \
    --exploration-weight 64 \
    --donor-reward-ref "$DONOR_REWARD_REF" \
    --donor-reward-other "$DONOR_REWARD_OTHER"


banner "MATERIALIZE RATIO"

"$PY" \
    -m analysis.profiling.transfer_materialized_tuned \
    --source-bundle "$SOURCE_BUNDLE" \
    --output-bundle "${EVIDENCE_DIR}/ratio" \
    --mode ratio \
    --reward-weight 0.01 \
    --exploration-weight 64 \
    --donor-reward-ref "$DONOR_REWARD_REF" \
    --donor-reward-other "$DONOR_REWARD_OTHER"


# ==============================================================================
# Cross-bundle verification
# ==============================================================================

banner "VERIFY COMMON PROVENANCE"

"$PY" - "$EVIDENCE_DIR" <<'PY'
import hashlib
import json
import math
import sys

from pathlib import Path


root = Path(sys.argv[1])

source_prior = (
    root / "source/frozen-prior.json"
)

source_sha = hashlib.sha256(
    source_prior.read_bytes()
).hexdigest()

evidence = json.loads(
    (root / "evidence.json").read_text()
)

ref = float(
    evidence["donor"]["amd64"][
        "mean_reward"
    ]
)

other = float(
    evidence["donor"]["arm64"][
        "mean_reward"
    ]
)

anchor = float(
    evidence[
        "target_reference_mean_reward"
    ]
)


expected = {
    "difference": {
        "c": 0.2,
        "wr": 0.05,
        "we": 64.0,
        "other":
            anchor + (other - ref),
    },

    "ratio": {
        "c": 0.4,
        "wr": 0.01,
        "we": 64.0,
        "other":
            anchor * other / ref,
    },
}


def close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=1e-9,
        abs_tol=1e-10,
    )


prior_shas = []


for mode in ("difference", "ratio"):
    bundle = root / mode

    manifest = json.loads(
        (
            bundle
            / "manifest.json"
        ).read_text()
    )

    prior = json.loads(
        (
            bundle
            / "frozen-prior.json"
        ).read_text()
    )

    exp = expected[mode]

    assert (
        manifest[
            "source_coupled_prior_sha256"
        ]
        == source_sha
    )

    assert manifest["mode"] == mode
    assert close(
        manifest["target_c"],
        exp["c"],
    )

    assert close(
        manifest[
            "reward_observation_weight"
        ],
        exp["wr"],
    )

    assert close(
        manifest[
            "exploration_observation_weight"
        ],
        exp["we"],
    )

    assert close(
        manifest["donor_reward_ref"],
        ref,
    )

    assert close(
        manifest["donor_reward_other"],
        other,
    )

    assert close(
        prior["arms"]["amd64"][
            "ucb1"
        ]["mean_reward"],
        anchor,
    )

    assert close(
        prior["arms"]["arm64"][
            "ucb1"
        ]["mean_reward"],
        exp["other"],
    )

    prior_shas.append(
        manifest["prior_sha256"]
    )


assert (
    prior_shas[0]
    != prior_shas[1]
)

print(
    "source_sha256=",
    source_sha,
)

print(
    "difference_prior_sha256=",
    prior_shas[0],
)

print(
    "ratio_prior_sha256=",
    prior_shas[1],
)

print(
    "COMMON DONOR EVIDENCE: PASS"
)

print(
    "DIFFERENCE + RATIO MATERIALIZED: PASS"
)
PY


banner "UCB1 FINAL DONOR EVIDENCE READY"

echo "target=$TARGET_FUNCTION"
echo "donor=$SELECTED_DONOR"
echo "evidence=${EVIDENCE_DIR}/evidence.json"
echo "difference=${EVIDENCE_DIR}/difference"
echo "ratio=${EVIDENCE_DIR}/ratio"

echo
echo "PASS — donor sampled directly 10+10 warm"
echo "PASS — same donor evidence used by Difference and Ratio"
echo "PASS — no target ARM ground truth used"