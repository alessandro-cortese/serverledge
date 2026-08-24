#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" &&
  pwd
)"

ROOT_DIR="$(
  cd "$SCRIPT_DIR/.." &&
  pwd
)"

cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-analysis/bin/python}"

POLICY="${POLICY:-UCB1}"

SAMPLES_PER_ARCH="${SAMPLES_PER_ARCH:-10}"

# Architecture used for the initial profiling phase of the target function.
#
# The donor is still profiled / learned on both MAB arms. Only the new target
# function is bootstrapped on one predetermined architecture before transfer.
TARGET_BOOTSTRAP_TAG="${TARGET_BOOTSTRAP_TAG:-x86}"

DONOR_MAX_REQUESTS="${DONOR_MAX_REQUESTS:-12}"

PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.5}"

MIN_DONOR_OBSERVATIONS="${MIN_DONOR_OBSERVATIONS:-1}"

MAX_DISTANCE="${MAX_DISTANCE:-1000000000000}"

X86_PORT="${X86_PORT:-1333}"
ARM_PORT="${ARM_PORT:-1334}"
LB_PORT="${LB_PORT:-8080}"

X86_UDP_PORT="${X86_UDP_PORT:-9899}"
ARM_UDP_PORT="${ARM_UDP_PORT:-9900}"

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_ID="transfer-local-smoke-${STAMP}"

LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"

mkdir -p "$LOG_DIR"

X86_CONF="$LOG_DIR/node-x86.yaml"
ARM_CONF="$LOG_DIR/node-arm64.yaml"
LB_CONF="$LOG_DIR/lb.yaml"

X86_LOG="$LOG_DIR/node-x86.log"
ARM_LOG="$LOG_DIR/node-arm64.log"
LB_LOG="$LOG_DIR/lb.log"
ETCD_LOG="$LOG_DIR/etcd.log"

RAW_X86="$LOG_DIR/profiling-x86.jsonl"
RAW_ARM="$LOG_DIR/profiling-arm64.jsonl"

case "$TARGET_BOOTSTRAP_TAG" in

x86)
  TARGET_BOOTSTRAP_LABEL="x86"
  TARGET_BOOTSTRAP_PORT="$X86_PORT"
  TARGET_BOOTSTRAP_RAW="$RAW_X86"

  TARGET_NON_BOOTSTRAP_TAG="arm64"
  TARGET_NON_BOOTSTRAP_RAW="$RAW_ARM"
  ;;

arm64)
  TARGET_BOOTSTRAP_LABEL="ARM"
  TARGET_BOOTSTRAP_PORT="$ARM_PORT"
  TARGET_BOOTSTRAP_RAW="$RAW_ARM"

  TARGET_NON_BOOTSTRAP_TAG="x86"
  TARGET_NON_BOOTSTRAP_RAW="$RAW_X86"
  ;;

*)
  echo \
    "[FAIL] TARGET_BOOTSTRAP_TAG deve essere x86 oppure arm64, ricevuto: $TARGET_BOOTSTRAP_TAG" \
    >&2

  exit 1
  ;;

esac

DONOR_FUNCTION="donor_smoke_${STAMP}"
TARGET_FUNCTION="target_smoke_${STAMP}"

DONOR_DIR="$LOG_DIR/donor"
TARGET_DIR="$LOG_DIR/target"

mkdir -p \
  "$DONOR_DIR" \
  "$TARGET_DIR"

PIDS=()

ETCD_STARTED=0

fail() {
  echo \
    "[FAIL] $*" \
    >&2

  exit 1
}

case "$POLICY" in
  UCB1|LinUCB)
    ;;
  *)
    fail "POLICY deve essere UCB1 oppure LinUCB, ricevuto: $POLICY"
    ;;
esac

cleanup() {
  status=$?

  set +e

  echo
  echo \
    "[cleanup] Arresto processi locali..."

  for pid in "${PIDS[@]:-}"; do

    if [[ -n "$pid" ]] &&
      kill -0 "$pid" 2>/dev/null; then

      kill \
        -INT \
        "$pid" \
        2>/dev/null ||
        true
    fi

  done

  sleep 1

  for pid in "${PIDS[@]:-}"; do

    if [[ -n "$pid" ]] &&
      kill -0 "$pid" 2>/dev/null; then

      kill \
        -KILL \
        "$pid" \
        2>/dev/null ||
        true
    fi

  done

  if [[ "$ETCD_STARTED" -eq 1 ]]; then

    docker rm \
      -f \
      Etcd-server \
      >/dev/null \
      2>&1 ||
      true

  fi

  echo \
    "[cleanup] Artefatti conservati in: $LOG_DIR"

  exit "$status"
}

trap \
  cleanup \
  EXIT \
  INT \
  TERM

for cmd in \
  docker \
  curl \
  make \
  python3; do

  command -v \
    "$cmd" \
    >/dev/null \
    2>&1 ||
    fail \
      "Comando richiesto non trovato: $cmd"

done

[[ -x "$PYTHON_BIN" ]] ||
  fail \
    "Python analysis non trovato: $PYTHON_BIN"

[[ -f analysis/profiling/transfer_query.py ]] ||
  fail \
    "transfer_query.py non trovato."

[[ -f analysis/profiling/similarity_selection.py ]] ||
  fail \
    "similarity_selection.py non trovato."

if (( SAMPLES_PER_ARCH < 10 ||
      SAMPLES_PER_ARCH > 20 )); then

  fail \
    "SAMPLES_PER_ARCH deve essere tra 10 e 20 con l'aggregatore FunctionProfile attuale."

fi

if (( DONOR_MAX_REQUESTS < 2 )); then

  fail \
    "DONOR_MAX_REQUESTS deve essere almeno 2."

fi

for port in \
  "$X86_PORT" \
  "$ARM_PORT" \
  "$LB_PORT"; do

  if curl \
    -sS \
    --max-time 1 \
    -o /dev/null \
    "http://127.0.0.1:${port}/status" \
    2>/dev/null; then

    fail \
      "La porta API $port sembra già occupata da un servizio Serverledge."

  fi

done

echo \
  "============================================================"

echo \
  "10C.3B.3 — LOCAL CONTAINER END-TO-END SMOKE TEST"

echo \
  "============================================================"

echo \
  "run_id:             $RUN_ID"

echo \
  "donor:              $DONOR_FUNCTION"

echo \
  "target:             $TARGET_FUNCTION"

echo \
  "donor samples/tag:  $SAMPLES_PER_ARCH"

echo \
  "target bootstrap N: $SAMPLES_PER_ARCH"

echo \
  "target bootstrap:   $TARGET_BOOTSTRAP_TAG"

echo \
  "x86 logical node:   127.0.0.1:$X86_PORT tag=x86"

echo \
  "ARM logical node:   127.0.0.1:$ARM_PORT tag=arm64"

echo \
  "LB:                  127.0.0.1:$LB_PORT policy=$POLICY"

echo

echo \
  "ATTENZIONE: entrambi i nodi sono fisicamente sulla stessa architettura host."

echo \
  "Il test valida solo l'integrazione funzionale, non le prestazioni ARM-vs-x86."

echo

# ============================================================
# Build
# ============================================================

echo \
  "[setup] Build..."

make

docker info \
  >/dev/null \
  2>&1 ||
  fail \
    "Docker non è disponibile."

docker rm \
  -f \
  Etcd-server \
  >/dev/null \
  2>&1 ||
  true

rm -f \
  "$RAW_X86" \
  "$RAW_ARM"

# ============================================================
# Configurazioni locali
# ============================================================

cat > "$X86_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${X86_PORT}

registry.area: TRANSFER_LOCAL_SMOKE
registry.node.id: smoke-x86-${STAMP}
registry.udp.port: ${X86_UDP_PORT}

node.machine_tag: x86

container.pool.memory: 1024
container.pool.cpus: 4
container.expiration: 600

factory.images.refresh: false

profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${RAW_X86}"

scheduler.queue.capacity: 0
YAML

cat > "$ARM_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${ARM_PORT}

registry.area: TRANSFER_LOCAL_SMOKE
registry.node.id: smoke-arm64-${STAMP}
registry.udp.port: ${ARM_UDP_PORT}

node.machine_tag: arm64

container.pool.memory: 1024
container.pool.cpus: 4
container.expiration: 600

factory.images.refresh: false

profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${RAW_ARM}"

scheduler.queue.capacity: 0
YAML

cat > "$LB_CONF" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${LB_PORT}

registry.area: TRANSFER_LOCAL_SMOKE

lb.arch_awareness: true
lb.mode: MAB

mab.policy: ${POLICY}
mab.ucb1.c: 0.8
mab.linucb.alpha: 0.1
mab.cold_start.mode: skip

mab.transfer.control.enabled: true

lb.replicas: 128
lb.refresh_interval: 1
YAML

# ============================================================
# Etcd
# ============================================================

echo \
  "[setup] Avvio etcd..."

bash scripts/start-etcd.sh \
  > "$ETCD_LOG" \
  2>&1

ETCD_STARTED=1

for _ in $(seq 1 30); do

  if docker exec \
    Etcd-server \
    etcdctl endpoint health \
    >/dev/null \
    2>&1; then

    break
  fi

  sleep 0.5

done

docker exec \
  Etcd-server \
  etcdctl endpoint health \
  >/dev/null \
  2>&1 ||
  fail \
    "etcd non disponibile."

# ============================================================
# Helper attesa HTTP
# ============================================================

wait_http() {
  local url="$1"
  local pid="$2"
  local logfile="$3"

  for _ in $(seq 1 80); do

    if ! kill -0 \
      "$pid" \
      2>/dev/null; then

      tail \
        -100 \
        "$logfile" ||
        true

      fail \
        "Processo terminato durante l'avvio: $logfile"

    fi

    if curl \
      -sS \
      --max-time 1 \
      -o /dev/null \
      "$url" \
      2>/dev/null; then

      return 0

    fi

    sleep 0.5

  done

  tail \
    -100 \
    "$logfile" ||
    true

  fail \
    "Timeout attendendo $url"
}

# ============================================================
# Nodi Serverledge
# ============================================================

echo \
  "[setup] Avvio nodo logico x86..."

bin/serverledge \
  "$X86_CONF" \
  > "$X86_LOG" \
  2>&1 &

X86_PID=$!

PIDS+=(
  "$X86_PID"
)

wait_http \
  "http://127.0.0.1:${X86_PORT}/status" \
  "$X86_PID" \
  "$X86_LOG"

echo \
  "[setup] Avvio nodo logico arm64..."

bin/serverledge \
  "$ARM_CONF" \
  > "$ARM_LOG" \
  2>&1 &

ARM_PID=$!

PIDS+=(
  "$ARM_PID"
)

wait_http \
  "http://127.0.0.1:${ARM_PORT}/status" \
  "$ARM_PID" \
  "$ARM_LOG"

# ============================================================
# Load balancer
# ============================================================

echo \
  "[setup] Avvio load balancer MAB..."

bin/lb \
  "$LB_CONF" \
  > "$LB_LOG" \
  2>&1 &

LB_PID=$!

PIDS+=(
  "$LB_PID"
)

wait_http \
  "http://127.0.0.1:${LB_PORT}/status" \
  "$LB_PID" \
  "$LB_LOG"

# Attendiamo discovery di entrambi i machine_tag.

for _ in $(seq 1 30); do

  if grep \
    -q \
    "event=new_machine_tag_discovered tag=x86" \
    "$LB_LOG" \
    2>/dev/null &&
    grep \
      -q \
      "event=new_machine_tag_discovered tag=arm64" \
      "$LB_LOG" \
      2>/dev/null; then

    break

  fi

  sleep 0.5

done

grep \
  -q \
  "event=new_machine_tag_discovered tag=x86" \
  "$LB_LOG" ||
  fail \
    "Il LB non ha scoperto il machine_tag x86."

grep \
  -q \
  "event=new_machine_tag_discovered tag=arm64" \
  "$LB_LOG" ||
  fail \
    "Il LB non ha scoperto il machine_tag arm64."

# ============================================================
# Creazione donor + target
# ============================================================

echo \
  "[setup] Creo donor e target..."

for function_name in \
  "$DONOR_FUNCTION" \
  "$TARGET_FUNCTION"; do

  bin/serverledge-cli create \
    -f "$function_name" \
    --memory 128 \
    --cpu 1 \
    --max_concurrency 1 \
    --src examples/hello.py \
    --runtime python314 \
    --handler "hello.handler" \
    -H 127.0.0.1 \
    -P "$X86_PORT" \
    > "$LOG_DIR/create-${function_name}.json"

done

sleep 1

# ============================================================
# Prewarm
# ============================================================

prewarm_on_both() {
  local function_name="$1"

  bin/serverledge-cli prewarm \
    -f "$function_name" \
    -c 1 \
    -H 127.0.0.1 \
    -P "$X86_PORT" \
    > "$LOG_DIR/prewarm-${function_name}-x86.json"

  bin/serverledge-cli prewarm \
    -f "$function_name" \
    -c 1 \
    -H 127.0.0.1 \
    -P "$ARM_PORT" \
    > "$LOG_DIR/prewarm-${function_name}-arm64.json"
}

# ============================================================
# Invocation body diretto
# ============================================================

DIRECT_BODY="$LOG_DIR/direct-invoke.json"

cat > "$DIRECT_BODY" <<'JSON'
{
  "Params": {
    "name": "World"
  },
  "QoSClass": 0,
  "QoSMaxRespT": -1.0,
  "CanDoOffloading": false,
  "Async": false,
  "ReturnOutput": true
}
JSON

direct_invoke() {
  local function_name="$1"
  local port="$2"
  local output="$3"

  curl \
    -fsS \
    --max-time 120 \
    -H \
      'Content-Type: application/json' \
    --data-binary \
      "@$DIRECT_BODY" \
    "http://127.0.0.1:${port}/invoke/${function_name}" \
    > "$output"
}

# ============================================================
# Profilazione controllata
# ============================================================

profile_function() {
  local function_name="$1"
  local prefix="$2"

  echo \
    "[profile] Prewarm $function_name sui due nodi..."

  prewarm_on_both \
    "$function_name"

  sleep 1

  echo \
    "[profile] $SAMPLES_PER_ARCH invocazioni dirette x86 + arm64 per $function_name..."

  for i in $(seq 1 "$SAMPLES_PER_ARCH"); do

    direct_invoke \
      "$function_name" \
      "$X86_PORT" \
      "$LOG_DIR/${prefix}-x86-${i}.json"

    direct_invoke \
      "$function_name" \
      "$ARM_PORT" \
      "$LOG_DIR/${prefix}-arm64-${i}.json"

  done

  docker ps \
    --format \
    '{{.ID}}\t{{.Image}}\t{{.Names}}' \
    > "$LOG_DIR/docker-after-${prefix}.txt"
}

# ============================================================
# Profilazione target su una sola architettura prestabilita
# ============================================================

profile_function_on_bootstrap_arch() {
  local function_name="$1"
  local prefix="$2"

  echo \
    "[bootstrap] Prewarm $function_name solo su $TARGET_BOOTSTRAP_TAG..."

  bin/serverledge-cli prewarm \
    -f "$function_name" \
    -c 1 \
    -H 127.0.0.1 \
    -P "$TARGET_BOOTSTRAP_PORT" \
    > "$LOG_DIR/prewarm-${function_name}-${TARGET_BOOTSTRAP_TAG}.json"

  sleep 1

  echo \
    "[bootstrap] $SAMPLES_PER_ARCH invocazioni dirette solo su $TARGET_BOOTSTRAP_TAG per $function_name..."

  for i in $(
    seq \
      1 \
      "$SAMPLES_PER_ARCH"
  ); do

    direct_invoke \
      "$function_name" \
      "$TARGET_BOOTSTRAP_PORT" \
      "$LOG_DIR/${prefix}-${TARGET_BOOTSTRAP_TAG}-${i}.json"

  done

  docker ps \
    --format \
      '{{.ID}}\t{{.Image}}\t{{.Names}}' \
    > "$LOG_DIR/docker-after-${prefix}.txt"
}

# ============================================================
# Filtro dataset della funzione
# ============================================================

filter_function_samples() {
  local function_name="$1"
  local output_dir="$2"

  mkdir -p \
    "$output_dir/x86" \
    "$output_dir/arm64"

  "$PYTHON_BIN" \
    - \
    "$RAW_X86" \
    "$RAW_ARM" \
    "$function_name" \
    "$SAMPLES_PER_ARCH" \
    "$output_dir/x86/profiling-samples.jsonl" \
    "$output_dir/arm64/profiling-samples.jsonl" <<'PY'

import json
import sys

from pathlib import Path


(
    raw_x86,
    raw_arm,
    function_name,
    required_raw,
    out_x86,
    out_arm,
) = sys.argv[1:]

required = int(
    required_raw
)


for (
    source_raw,
    expected_tag,
    output_raw,
) in (
    (
        raw_x86,
        "x86",
        out_x86,
    ),
    (
        raw_arm,
        "arm64",
        out_arm,
    ),
):

    source = Path(
        source_raw
    )

    if not source.is_file():

        raise SystemExit(
            "raw profiling dataset "
            f"mancante: {source}"
        )

    selected = []

    eligible = 0

    warm = 0

    cold = 0

    for (
        line_number,
        raw,
    ) in enumerate(
        source.read_text(
            encoding="utf-8"
        ).splitlines(),
        1,
    ):

        if not raw.strip():
            continue

        sample = json.loads(
            raw
        )

        if (
            sample.get(
                "function_name"
            )
            != function_name
        ):
            continue

        if (
            sample.get(
                "machine_tag"
            )
            != expected_tag
        ):
            continue

        selected.append(
            sample
        )

        if sample.get(
            "warm_start"
        ):

            warm += 1

        else:

            cold += 1

        if (
            (
                sample.get(
                    "eligibility"
                )
                or {}
            ).get(
                "resource_clustering"
            )
            is True
        ):

            eligible += 1

    if eligible < required:

        raise SystemExit(
            f"{function_name}/{expected_tag}: "
            "campioni eligible insufficienti: "
            f"{eligible} < {required}"
        )

    output = Path(
        output_raw
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for sample in selected:

            handle.write(
                json.dumps(
                    sample,
                    sort_keys=True,
                )
                + "\n"
            )

    print(
        "[profile] "
        f"function={function_name} "
        f"tag={expected_tag} "
        f"selected={len(selected)} "
        f"warm={warm} "
        f"cold={cold} "
        f"eligible={eligible}"
    )

PY
}

# ============================================================
# Filtro target: una sola architettura di bootstrap
# ============================================================

filter_function_samples_on_bootstrap_arch() {
  local function_name="$1"
  local output_dir="$2"

  local output_file=
  output_file="$output_dir/${TARGET_BOOTSTRAP_TAG}/profiling-samples.jsonl"

  mkdir -p \
    "$output_dir/${TARGET_BOOTSTRAP_TAG}"

  "$PYTHON_BIN" \
    - \
    "$TARGET_BOOTSTRAP_RAW" \
    "$function_name" \
    "$TARGET_BOOTSTRAP_TAG" \
    "$SAMPLES_PER_ARCH" \
    "$output_file" <<'PY'

import json
import sys

from pathlib import Path


(
    source_raw,
    function_name,
    expected_tag,
    required_raw,
    output_raw,
) = sys.argv[1:]


required = int(
    required_raw
)


source = Path(
    source_raw
)


if not source.is_file():

    raise SystemExit(
        "raw profiling dataset "
        f"mancante: {source}"
    )


selected = []

eligible = 0
warm = 0
cold = 0


for (
    line_number,
    raw,
) in enumerate(
    source.read_text(
        encoding="utf-8"
    ).splitlines(),
    1,
):

    if not raw.strip():
        continue

    sample = json.loads(
        raw
    )

    if (
        sample.get(
            "function_name"
        )
        != function_name
    ):
        continue

    if (
        sample.get(
            "machine_tag"
        )
        != expected_tag
    ):
        continue

    selected.append(
        sample
    )

    if sample.get(
        "warm_start"
    ):

        warm += 1

    else:

        cold += 1

    if (
        (
            sample.get(
                "eligibility"
            )
            or {}
        ).get(
            "resource_clustering"
        )
        is True
    ):

        eligible += 1


if eligible < required:

    raise SystemExit(
        f"{function_name}/{expected_tag}: "
        "campioni eligible insufficienti: "
        f"{eligible} < {required}"
    )


output = Path(
    output_raw
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with output.open(
    "w",
    encoding="utf-8",
) as handle:

    for sample in selected:

        handle.write(
            json.dumps(
                sample,
                sort_keys=True,
            )
            + "\n"
        )


print(
    "[bootstrap] "
    f"function={function_name} "
    f"tag={expected_tag} "
    f"selected={len(selected)} "
    f"warm={warm} "
    f"cold={cold} "
    f"eligible={eligible}"
)

PY
}

# ============================================================
# Aggregazione
# ============================================================

aggregate_function() {
  local function_name="$1"
  local filtered_dir="$2"
  local output_dir="$3"

  local profiles=
  local mean_csv=
  local median_csv=

  profiles="$output_dir/function-profiles.jsonl"

  mean_csv="$output_dir/function-profiles-mean.csv"

  median_csv="$output_dir/function-profiles-median.csv"

  bin/serverledge-profiling aggregate \
    --input-dir \
      "$filtered_dir" \
    --samples \
      "$SAMPLES_PER_ARCH" \
    --output \
      "$profiles" \
    > "$output_dir/aggregate.log"

  bin/serverledge-profiling export-csv \
    --input \
      "$profiles" \
    --experiment-id \
      "$RUN_ID-$function_name" \
    --mean-output \
      "$mean_csv" \
    --median-output \
      "$median_csv" \
    > "$output_dir/export-csv.log"
}

# ============================================================
# 1. Profilazione donor
# ============================================================

profile_function \
  "$DONOR_FUNCTION" \
  "donor-profile"

filter_function_samples \
  "$DONOR_FUNCTION" \
  "$DONOR_DIR/filtered"

aggregate_function \
  "$DONOR_FUNCTION" \
  "$DONOR_DIR/filtered" \
  "$DONOR_DIR"

# ============================================================
# 2. Preprocessing donor
# ============================================================

MODEL_JSON="$DONOR_DIR/preprocess-model.json"

DONOR_PREPROCESSED="$DONOR_DIR/donor-preprocessed.csv"

CATALOG_JSON="$DONOR_DIR/smoke-donor-catalog.json"

"$PYTHON_BIN" \
  analysis/profiling/preprocess.py \
  fit-transform \
  --input \
    "$DONOR_DIR/function-profiles-mean.csv" \
  --scaler \
    none \
  --model \
    "$MODEL_JSON" \
  --output \
    "$DONOR_PREPROCESSED" \
  > "$DONOR_DIR/preprocess.log"

# ============================================================
# 3. Catalogo donor SOLO PER LO SMOKE TEST
# ============================================================

"$PYTHON_BIN" \
  - \
  "$DONOR_PREPROCESSED" \
  "$CATALOG_JSON" \
  "$DONOR_FUNCTION" \
  "$RUN_ID" \
  "$TARGET_BOOTSTRAP_TAG" <<'PY'

import csv
import json
import sys

from pathlib import Path

from analysis.profiling import preprocess


(
    source_raw,
    output_raw,
    donor_name,
    run_id,
    profile_machine_tag,
) = sys.argv[1:]

with Path(
    source_raw
).open(
    newline="",
    encoding="utf-8",
) as handle:

    rows = list(
        csv.DictReader(
            handle
        )
    )


matches = [
    row
    for row
    in rows
    if (
        row[
            "function_name"
        ]
        == donor_name
        and
        row[
            "machine_tag"
        ]
        == profile_machine_tag
    )
]


if len(
    matches
) != 1:

    raise SystemExit(
        "attesa una sola riga "
        f"donor {profile_machine_tag}, trovate "
        f"{len(matches)}"
    )


row = matches[
    0
]


vector = [
    float(
        row[
            name
        ]
    )
    for name
    in preprocess.FEATURE_NAMES
]


catalog = {
    "schema_version":
        1,

    "catalog_run_id":
        f"smoke-catalog-{run_id}",

    "feature_names":
        list(
            preprocess.FEATURE_NAMES
        ),

    "feature_space": {
        "representation":
            "preprocessed",

        "scaler":
            "none",

        "distance_candidate":
            "euclidean",

        "distance_policy_selected":
            False,
    },

    "clustering": {
        "clustering_run_id":
            f"smoke-clustering-{run_id}",

        "algorithm":
            "local-smoke-fixture",

        "aggregation":
            "mean",

        "profile_machine_tag":
            profile_machine_tag,
    },

    "donor_policy": {
        "readiness_required":
            False,

        "noise_is_eligible":
            False,

        "bandit_prior_materialized":
            False,

        "architecture_preference_is_bandit_reward":
            False,
    },

    "summary": {
        "donor_count":
            1,

        "eligible_donor_count":
            1,

        "ineligible_donor_count":
            0,

        "noise_donor_count":
            0,
    },

    "donors": [
        {
            "function_name":
                donor_name,

            "configured_cpus":
                float(
                    row[
                        "configured_cpus"
                    ]
                ),

            "configured_memory_mb":
                int(
                    row[
                        "configured_memory_mb"
                    ]
                ),

            "profile_machine_tag":
                profile_machine_tag,

            "aggregation":
                "mean",

            "scaler":
                "none",

            "algorithm":
                "local-smoke-fixture",

            "cluster_label":
                0,

            "is_noise":
                False,

            "donor_eligible":
                True,

            "donor_ineligibility_reason":
                "",

            # Metadata non scientifici:
            # vengono usati soltanto per
            # completare lo schema dello smoke test.
            "architecture_preference":
                "architecture_independent",

            "arm_vs_x86_delta_percent":
                0.0,

            "threshold_percent":
                0.0,

            "x86_duration_ms":
                0.0,

            "arm_duration_ms":
                0.0,

            "feature_vector":
                vector,

            "bandit_prior":
                None,
        }
    ],
}


Path(
    output_raw
).write_text(
    json.dumps(
        catalog,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    f"catalog={output_raw}"
)

PY

# ============================================================
# 4. Donor accumula REAL observations nel MAB live
# ============================================================

echo \
  "[mab] Accumulo conoscenza reale del donor nel LB..."

prewarm_on_both \
  "$DONOR_FUNCTION"

x86_updates=0
arm_updates=0

for i in $(seq 1 "$DONOR_MAX_REQUESTS"); do

  bin/serverledge-cli invoke \
    -f "$DONOR_FUNCTION" \
    -p "name:World" \
    --return_output \
    -H 127.0.0.1 \
    -P "$LB_PORT" \
    > "$DONOR_DIR/lb-invoke-${i}.json"

  sleep 0.15

  x86_updates="$(
    grep \
      -c \
      "event=update_reward.*function=${DONOR_FUNCTION}.*arm=x86 " \
      "$LB_LOG" \
      2>/dev/null ||
      true
  )"

  arm_updates="$(
    grep \
      -c \
      "event=update_reward.*function=${DONOR_FUNCTION}.*arm=arm64 " \
      "$LB_LOG" \
      2>/dev/null ||
      true
  )"

  if (( x86_updates >= MIN_DONOR_OBSERVATIONS &&
        arm_updates >= MIN_DONOR_OBSERVATIONS )); then

    break

  fi

done

if (( x86_updates < MIN_DONOR_OBSERVATIONS ||
      arm_updates < MIN_DONOR_OBSERVATIONS )); then

  grep \
    "event=update_reward.*function=${DONOR_FUNCTION}" \
    "$LB_LOG" ||
    true

  fail \
    "Il donor non ha accumulato abbastanza real observations su entrambi gli arm: x86=$x86_updates arm64=$arm_updates"

fi

echo \
  "[mab] donor real updates: x86=$x86_updates arm64=$arm_updates"

# ============================================================
# 5. Bootstrap target DIRETTO su architettura prestabilita
# ============================================================

echo \
  "[bootstrap] target=$TARGET_FUNCTION tag=$TARGET_BOOTSTRAP_TAG samples=$SAMPLES_PER_ARCH"

profile_function_on_bootstrap_arch \
  "$TARGET_FUNCTION" \
  "target-profile"

filter_function_samples_on_bootstrap_arch \
  "$TARGET_FUNCTION" \
  "$TARGET_DIR/filtered"

aggregate_function \
  "$TARGET_FUNCTION" \
  "$TARGET_DIR/filtered" \
  "$TARGET_DIR"

# Prima del transfer il target non deve essere mai passato dal MAB.

# ============================================================
# Verifica: zero profiling target sull'altra architettura
# ============================================================

"$PYTHON_BIN" \
  - \
  "$TARGET_NON_BOOTSTRAP_RAW" \
  "$TARGET_FUNCTION" \
  "$TARGET_NON_BOOTSTRAP_TAG" <<'PY'

import json
import sys

from pathlib import Path


source = Path(
    sys.argv[
        1
    ]
)

function_name = sys.argv[
    2
]

unexpected_tag = sys.argv[
    3
]


count = 0


if source.is_file():

    for raw in source.read_text(
        encoding="utf-8"
    ).splitlines():

        if not raw.strip():
            continue

        sample = json.loads(
            raw
        )

        if (
            sample.get(
                "function_name"
            )
            == function_name
            and
            sample.get(
                "machine_tag"
            )
            == unexpected_tag
        ):

            count += 1


if count != 0:

    raise SystemExit(
        f"{function_name}: trovati "
        f"{count} profiling sample inattesi "
        f"su {unexpected_tag}"
    )


print(
    "[bootstrap-check] "
    f"function={function_name} "
    f"non_bootstrap_tag={unexpected_tag} "
    "samples=0"
)

PY

if grep \
  -q \
  "event=select_arm.*function=${TARGET_FUNCTION}" \
  "$LB_LOG" \
  2>/dev/null; then

  fail \
    "Il target MAB è stato usato prima del transfer."

fi

# ============================================================
# 6. Transfer query
# ============================================================

QUERY_JSON="$TARGET_DIR/transfer-query.json"

SELECTION_JSON="$TARGET_DIR/selection.json"

SELECTION_CSV="$TARGET_DIR/selection.csv"

CONTROL_REQUEST="$TARGET_DIR/control-request.json"

CONTROL_RESPONSE="$TARGET_DIR/control-response.json"

"$PYTHON_BIN" \
  analysis/profiling/transfer_query.py \
  --input \
    "$TARGET_DIR/function-profiles-mean.csv" \
  --catalog \
    "$CATALOG_JSON" \
  --model \
    "$MODEL_JSON" \
  --function \
    "$TARGET_FUNCTION" \
  --query-id \
    "query-$RUN_ID" \
  --output \
    "$QUERY_JSON" \
  > "$TARGET_DIR/transfer-query.log"

# ============================================================
# 7. Similarity
# ============================================================

"$PYTHON_BIN" \
  analysis/profiling/similarity_selection.py \
  --catalog \
    "$CATALOG_JSON" \
  --query \
    "$QUERY_JSON" \
  --run-id \
    "selection-$RUN_ID" \
  --max-distance \
    "$MAX_DISTANCE" \
  --output-json \
    "$SELECTION_JSON" \
  --output-csv \
    "$SELECTION_CSV" \
  > "$TARGET_DIR/similarity.log"

# ============================================================
# 8. Costruzione richiesta Control API
# ============================================================

"$PYTHON_BIN" \
  - \
  "$SELECTION_JSON" \
  "$TARGET_FUNCTION" \
  "$PRIOR_WEIGHT" \
  "$MIN_DONOR_OBSERVATIONS" \
  > "$CONTROL_REQUEST" <<'PY'

import json
import sys

from pathlib import Path


selection = json.loads(
    Path(
        sys.argv[
            1
        ]
    ).read_text(
        encoding="utf-8"
    )
)


json.dump(
    {
        "target_function_name":
            sys.argv[
                2
            ],

        "selection_artifact":
            selection,

        "prior_config": {
            "equivalent_observation_weight":
                float(
                    sys.argv[
                        3
                    ]
                ),

            "min_real_observations_per_arm":
                int(
                    sys.argv[
                        4
                    ]
                ),
        },
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)

print()

PY

# ============================================================
# 9. Applicazione transfer al MAB live
# ============================================================

HTTP_CODE="$(
  curl \
    -sS \
    --max-time 30 \
    -o "$CONTROL_RESPONSE" \
    -w '%{http_code}' \
    -H \
      'Content-Type: application/json' \
    --data-binary \
      "@$CONTROL_REQUEST" \
    "http://127.0.0.1:${LB_PORT}/mab/transfer/initialize"
)"

if [[ "$HTTP_CODE" != "200" ]]; then

  cat \
    "$CONTROL_RESPONSE" \
    >&2 ||
    true

  fail \
    "Transfer control API HTTP $HTTP_CODE"

fi

# ============================================================
# 10. Verifica response
# ============================================================

"$PYTHON_BIN" \
  - \
  "$SELECTION_JSON" \
  "$CONTROL_RESPONSE" \
  "$DONOR_FUNCTION" \
  "$TARGET_FUNCTION" <<'PY'

import json
import sys

from pathlib import Path


selection = json.loads(
    Path(
        sys.argv[
            1
        ]
    ).read_text(
        encoding="utf-8"
    )
)


response = json.loads(
    Path(
        sys.argv[
            2
        ]
    ).read_text(
        encoding="utf-8"
    )
)


expected_donor = sys.argv[
    3
]

expected_target = sys.argv[
    4
]


selected = (
    selection.get(
        "selected_donor"
    )
    or {}
)


if (
    selection.get(
        "status"
    )
    != "selected"
):

    raise SystemExit(
        "similarity non ha selezionato "
        "un donor: "
        f"{selection.get('reason')}"
    )


if (
    selected.get(
        "function_name"
    )
    != expected_donor
):

    raise SystemExit(
        "donor inatteso: "
        f"{selected.get('function_name')!r}, "
        f"atteso {expected_donor!r}"
    )


if (
    response.get(
        "target_function_name"
    )
    != expected_target
):

    raise SystemExit(
        "target_function_name inatteso "
        "nella control response"
    )


if (
    response.get(
        "selected_donor_function_name"
    )
    != expected_donor
):

    raise SystemExit(
        "selected donor inatteso "
        "nella control response"
    )


if (
    response.get(
        "transfer_attempted"
    )
    is not True
):

    raise SystemExit(
        "transfer_attempted != true"
    )


if (
    response.get(
        "transfer_applied"
    )
    is not True
):

    raise SystemExit(
        "transfer non applicato: "
        "runtime_reason="
        f"{response.get('runtime_reason')!r}"
    )


prior = (
    response.get(
        "prior"
    )
    or {}
)


if (
    prior.get(
        "has_prior"
    )
    is not True
):

    raise SystemExit(
        "control response senza weak prior"
    )


print(
    "[transfer] "
    "selected_donor="
    f"{expected_donor} "
    "transfer_applied=true "
    "source_real_observations="
    f"{prior.get('source_real_observation_count')} "
    "transferred_arms="
    f"{prior.get('transferred_arm_count')}"
)

PY

# ============================================================
# 11. Richiesta target NORMALE dopo transfer
# ============================================================

echo \
  "[mab] Invocazione target normale attraverso il LB..."

bin/serverledge-cli invoke \
  -f "$TARGET_FUNCTION" \
  -p "name:World" \
  --return_output \
  -H 127.0.0.1 \
  -P "$LB_PORT" \
  > "$TARGET_DIR/lb-invoke-after-transfer.json"

sleep 0.2

grep \
  -q \
  "event=select_arm.*function=${TARGET_FUNCTION}" \
  "$LB_LOG" ||
  fail \
    "Nessuna decisione MAB trovata per il target dopo il transfer."

case "$POLICY" in

  UCB1)

    grep \
      -q \
      "event=arm_score.*function=${TARGET_FUNCTION}.*prior_observation_weight=" \
      "$LB_LOG" ||
      fail \
        "Nessun arm_score UCB1 con prior trovato per il target."

    ;;


  LinUCB)

    LINUCB_TRANSFER_LINE="$(
      grep \
        "event=runtime_transfer" \
        "$LB_LOG" |
        grep \
          "target_function=${TARGET_FUNCTION}" |
        tail -n 1 ||
        true
    )"

    [[ -n "$LINUCB_TRANSFER_LINE" ]] ||
      fail \
        "Nessun runtime_transfer LinUCB trovato per il target."

    [[ "$LINUCB_TRANSFER_LINE" == *"policy=LinUCB"* ]] ||
      fail \
        "Il runtime_transfer del target non usa LinUCB."

    [[ "$LINUCB_TRANSFER_LINE" == *"applied=true"* ]] ||
      fail \
        "Il transfer LinUCB non risulta applicato al target."

    [[ "$LINUCB_TRANSFER_LINE" == *"prior_has_prior=true"* ]] ||
      fail \
        "Il transfer LinUCB non contiene un weak prior valido."

    grep \
      -q \
      "event=arm_score.*policy=LinUCB.*function=${TARGET_FUNCTION}" \
      "$LB_LOG" ||
      fail \
        "Nessun arm_score LinUCB trovato per il target."

    ;;

esac

# ============================================================
# PASS
# ============================================================

echo
echo \
  "============================================================"

echo \
  "LOCAL END-TO-END SMOKE TEST: PASS"

echo \
  "============================================================"

echo \
  "✓ due nodi Serverledge locali avviati"

echo \
  "✓ container Docker realmente prewarmed"

echo \
  "✓ donor profilato sui due machine_tag logici"

echo \
  "✓ target bootstrap solo su $TARGET_BOOTSTRAP_TAG"

echo \
  "✓ zero profiling target su $TARGET_NON_BOOTSTRAP_TAG prima del transfer"

echo \
  "✓ FunctionProfile aggregato"

echo \
  "✓ transfer_query.py eseguito"

echo \
  "✓ similarity_selection.py ha selezionato il donor"

echo \
  "✓ donor con real observations nel MAB live"

echo \
  "✓ weak prior applicato al target via control API"

echo \
  "✓ successiva richiesta target gestita dal normale $POLICY"

echo

echo \
  "NON VALIDATO: prestazioni ARM-vs-x86."

echo \
  "Entrambi i nodi vengono eseguiti sulla stessa architettura fisica."

echo

echo \
  "Artefatti: $LOG_DIR"