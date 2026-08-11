#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

API_PORT="${API_PORT:-1333}"
UDP_PORT="${UDP_PORT:-9899}"
MEMORY_MB="${MEMORY_MB:-128}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
WARM_SAMPLES="${WARM_SAMPLES:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/profiling_aggregation_local_${STAMP}"
NODE_LOG="$LOG_DIR/node.log"
ETCD_LOG="$LOG_DIR/etcd.log"
CONF_FILE="$LOG_DIR/node-profiling-aggregation-conf.yaml"

RAW_DATASET_FILE="$ROOT_DIR/$LOG_DIR/profiling-samples.jsonl"
PROFILE_DATASET_FILE="$ROOT_DIR/$LOG_DIR/function-profiles.jsonl"
AGGREGATE_LOG="$LOG_DIR/aggregate.log"

FUNCTION_NAME="profiling_aggregation_${STAMP}"

NODE_PID=""
ETCD_STARTED=0

mkdir -p "$LOG_DIR"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

cleanup() {
  status=$?
  set +e

  echo
  echo "[cleanup] Arresto dei processi..."

  if [[ -n "$NODE_PID" ]] && kill -0 "$NODE_PID" 2>/dev/null; then
    kill -INT "$NODE_PID" 2>/dev/null

    for _ in $(seq 1 20); do
      kill -0 "$NODE_PID" 2>/dev/null || break
      sleep 0.5
    done

    kill -0 "$NODE_PID" 2>/dev/null &&
      kill -KILL "$NODE_PID" 2>/dev/null

    wait "$NODE_PID" 2>/dev/null
  fi

  if [[ "$ETCD_STARTED" -eq 1 ]]; then
    docker rm -f Etcd-server >/dev/null 2>&1
  fi

  echo "[cleanup] Log conservati in: $LOG_DIR"

  exit "$status"
}

trap cleanup EXIT INT TERM

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 ||
    fail "Comando richiesto non trovato: $cmd"
done

[[ -f Makefile ]] ||
  fail "Eseguire lo script dalla root di Serverledge."

[[ "$WARM_SAMPLES" =~ ^[0-9]+$ ]] ||
  fail "WARM_SAMPLES deve essere un intero."

(( WARM_SAMPLES >= 10 && WARM_SAMPLES <= 20 )) ||
  fail "WARM_SAMPLES deve essere compreso tra 10 e 20."

[[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] ||
  fail "SLEEP_SECONDS deve essere un intero non negativo."

EXPECTED_RAW_SAMPLES=$((WARM_SAMPLES + 2))

# ============================================================
# Build
# ============================================================

echo "[setup] Build..."
make

[[ -x bin/serverledge-profiling ]] ||
  fail "bin/serverledge-profiling non è stato generato."

docker info >/dev/null 2>&1 ||
  fail "Docker non è disponibile."

docker rm -f Etcd-server >/dev/null 2>&1 || true

rm -f \
  "$RAW_DATASET_FILE" \
  "$PROFILE_DATASET_FILE"

# ============================================================
# Configurazione
# ============================================================

cat >"$CONF_FILE" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: PROFILE_AGGREGATION_TEST
registry.node.id: profiling-aggregation-node-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: profiling-test

container.pool.memory: 768
container.pool.cpus: 8
container.expiration: 600
factory.images.refresh: false

profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${RAW_DATASET_FILE}"

scheduler.queue.capacity: 0
YAML

# ============================================================
# etcd
# ============================================================

echo "[setup] Avvio etcd..."

bash scripts/start-etcd.sh >"$ETCD_LOG" 2>&1
ETCD_STARTED=1

for _ in $(seq 1 30); do
  docker exec Etcd-server \
    etcdctl endpoint health \
    >/dev/null 2>&1 &&
    break

  sleep 0.5
done

docker exec Etcd-server \
  etcdctl endpoint health \
  >/dev/null 2>&1 ||
  fail "etcd non disponibile."

# ============================================================
# Serverledge
# ============================================================

echo "[setup] Avvio Serverledge..."

bin/serverledge "$CONF_FILE" >"$NODE_LOG" 2>&1 &
NODE_PID=$!

for _ in $(seq 1 60); do
  kill -0 "$NODE_PID" 2>/dev/null || {
    tail -100 "$NODE_LOG" || true
    fail "Serverledge è terminato durante l'avvio."
  }

  if curl \
    -sS \
    --max-time 1 \
    -o /dev/null \
    "http://127.0.0.1:${API_PORT}/status" \
    2>/dev/null; then
    break
  fi

  sleep 0.5
done

curl \
  -sS \
  --max-time 2 \
  -o /dev/null \
  "http://127.0.0.1:${API_PORT}/status" ||
  fail "API non disponibile."

# ============================================================
# Funzione
# ============================================================

echo "[setup] Creazione funzione $FUNCTION_NAME..."

bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory "$MEMORY_MB" \
  --src examples/sleeper.py \
  --runtime python314 \
  --handler "sleeper.handler" \
  --max_concurrency 1 \
  --cpu 1 \
  --input "n:Int" \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  | tee "$LOG_DIR/create.json"

# ============================================================
# Cold
# ============================================================

echo "[test] Invocazione cold riuscita..."

bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:${SLEEP_SECONDS}" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/cold.log" 2>&1

# ============================================================
# Warm
# ============================================================

echo "[test] ${WARM_SAMPLES} invocazioni warm riuscite..."

for index in $(seq 1 "$WARM_SAMPLES"); do
  printf -v warm_log \
    "%s/warm_%02d.log" \
    "$LOG_DIR" \
    "$index"

  echo "[test] Warm ${index}/${WARM_SAMPLES}..."

  bin/serverledge-cli invoke \
    -f "$FUNCTION_NAME" \
    -p "n:${SLEEP_SECONDS}" \
    --return_output \
    -H 127.0.0.1 \
    -P "$API_PORT" \
    >"$warm_log" 2>&1
done

# ============================================================
# Failed
# ============================================================

echo "[test] Invocazione warm fallita (errore atteso)..."

set +e

bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:not-a-number" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/failed.log" 2>&1

FAILED_STATUS=$?

set -e

[[ "$FAILED_STATUS" -ne 0 ]] ||
  fail "L'invocazione volutamente errata è risultata riuscita."

# ============================================================
# Attesa raw dataset
# ============================================================

echo \
  "[test] Attesa di ${EXPECTED_RAW_SAMPLES} righe nel raw dataset..."

for _ in $(seq 1 100); do
  if [[ -f "$RAW_DATASET_FILE" ]] &&
    [[ "$(wc -l <"$RAW_DATASET_FILE")" -ge "$EXPECTED_RAW_SAMPLES" ]]; then
    break
  fi

  sleep 0.2
done

[[ -f "$RAW_DATASET_FILE" ]] ||
  fail "Raw dataset JSONL non creato: $RAW_DATASET_FILE"

RAW_LINES="$(wc -l <"$RAW_DATASET_FILE")"

[[ "$RAW_LINES" -eq "$EXPECTED_RAW_SAMPLES" ]] ||
  fail \
    "Numero righe raw inatteso: trovate $RAW_LINES, attese $EXPECTED_RAW_SAMPLES."

# ============================================================
# Aggregazione reale
# ============================================================

echo "[test] Aggregazione reale..."

bin/serverledge-profiling aggregate \
  --input "$RAW_DATASET_FILE" \
  --output "$PROFILE_DATASET_FILE" \
  --samples "$WARM_SAMPLES" \
  | tee "$AGGREGATE_LOG"

[[ -f "$PROFILE_DATASET_FILE" ]] ||
  fail \
    "Dataset FunctionProfile non creato: $PROFILE_DATASET_FILE"

# ============================================================
# Validazione indipendente
# ============================================================

python3 - \
  "$RAW_DATASET_FILE" \
  "$PROFILE_DATASET_FILE" \
  "$FUNCTION_NAME" \
  "$WARM_SAMPLES" \
  "$MEMORY_MB" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path


raw_path = Path(sys.argv[1])
profile_path = Path(sys.argv[2])
function_name = sys.argv[3]
warm_expected = int(sys.argv[4])
memory_expected = int(sys.argv[5])

features = (
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
)

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def load_jsonl(path):
    raw = path.read_bytes()

    check(
        raw.endswith(b"\n"),
        f"{path.name}: newline finale mancante",
    )

    records = []

    for number, line in enumerate(
        raw.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(
                json.loads(line)
            )
        except json.JSONDecodeError as exc:
            failures.append(
                f"{path.name}, riga {number}: {exc}"
            )

    return records


def finite(value):
    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
        and math.isfinite(
            value
        )
    )


def mean_like_go(values):
    result = 0.0

    for index, value in enumerate(
        values,
        start=1,
    ):
        result += (
            value - result
        ) / index

    return result


def extract(sample):
    profile = sample["profile"]
    node = sample["node_environment"]
    duration = sample["timing"]["duration_ms"]

    page_faults = profile[
        "PageFaultsDelta"
    ]

    user_ns = profile[
        "CPUUsageUserDeltaNs"
    ]

    kernel_ns = profile[
        "CPUUsageKernelDeltaNs"
    ]

    free_bytes = node[
        "FreeMemoryBeforeBytes"
    ]

    start_container = profile[
        "ProfilingStartOverheadMs"
    ]

    start_node = node[
        "SnapshotStartOverheadMs"
    ]

    values = (
        duration,
        page_faults,
        user_ns,
        kernel_ns,
        free_bytes,
        start_container,
        start_node,
    )

    if not all(
        finite(value)
        for value in values
    ):
        raise ValueError(
            "input numerico mancante/non finito"
        )

    if (
        duration <= 0
        or any(
            value < 0
            for value in values[1:]
        )
    ):
        raise ValueError(
            "input numerico fuori dominio"
        )

    return {
        "page_faults_delta":
            float(
                page_faults
            ),

        "utilized_cpus":
            (
                user_ns
                + kernel_ns
            )
            /
            (
                duration
                * 1_000_000.0
            ),

        "free_memory_mb":
            free_bytes
            /
            (
                1024.0
                * 1024.0
            ),

        "cpu_user_delta_ms":
            user_ns
            /
            1_000_000.0,

        "cpu_kernel_delta_ms":
            kernel_ns
            /
            1_000_000.0,

        "framework_runtime_ms":
            start_container
            + start_node,
    }


raw = load_jsonl(
    raw_path
)

aggregated = load_jsonl(
    profile_path
)

check(
    len(raw)
    ==
    warm_expected + 2,
    (
        f"raw: attesi {warm_expected + 2} campioni, "
        f"trovati {len(raw)}"
    ),
)

# ============================================================
# Metadati raw
# ============================================================

for index, sample in enumerate(
    raw,
    start=1,
):
    check(
        sample.get(
            "schema_version"
        ) == 3,
        f"raw {index}: schema_version non valido",
    )

    check(
        sample.get(
            "function_name"
        ) == function_name,
        f"raw {index}: function_name non valido",
    )

    check(
        sample.get(
            "machine_tag"
        ) == "profiling-test",
        f"raw {index}: machine_tag non valido",
    )

    config = sample.get(
        "function_configuration",
        {},
    )

    check(
        math.isclose(
            float(
                config.get(
                    "configured_cpus",
                    -1,
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"raw {index}: configured_cpus non valido",
    )

    check(
        config.get(
            "configured_memory_mb"
        )
        ==
        memory_expected,
        f"raw {index}: configured_memory_mb non valido",
    )


ids = [
    sample.get(
        "request_id"
    )
    for sample in raw
]

check(
    len(
        set(
            ids
        )
    )
    ==
    len(
        ids
    ),
    "raw: request_id duplicati",
)

# ============================================================
# Classificazione raw
# ============================================================

cold = [
    sample
    for sample in raw
    if sample.get(
        "warm_start"
    ) is False
    and sample.get(
        "execution_succeeded"
    ) is True
]

warm = [
    sample
    for sample in raw
    if sample.get(
        "warm_start"
    ) is True
    and sample.get(
        "execution_succeeded"
    ) is True
]

failed = [
    sample
    for sample in raw
    if sample.get(
        "execution_succeeded"
    ) is False
]

eligible = [
    (
        order,
        sample,
    )
    for order, sample in enumerate(
        raw
    )
    if sample.get(
        "eligibility",
        {},
    ).get(
        "resource_clustering"
    ) is True
]

check(
    len(cold) == 1,
    f"raw: cold riusciti={len(cold)}, atteso 1",
)

check(
    len(warm)
    ==
    warm_expected,
    (
        f"raw: warm riusciti={len(warm)}, "
        f"attesi {warm_expected}"
    ),
)

check(
    len(failed) == 1,
    f"raw: falliti={len(failed)}, atteso 1",
)

check(
    len(eligible)
    ==
    warm_expected,
    (
        f"raw: eleggibili={len(eligible)}, "
        f"attesi {warm_expected}"
    ),
)

check(
    all(
        sample.get(
            "eligibility",
            {},
        ).get(
            "resource_clustering"
        ) is False
        for sample in cold + failed
    ),
    "raw: cold o failed risulta eleggibile",
)

# ============================================================
# Stessa selezione dell'aggregatore Go
# ============================================================

eligible.sort(
    key=lambda item: (
        item[1].get(
            "timestamp_ms",
            0,
        ),
        item[0],
    )
)

selected = [
    sample
    for _, sample in
    eligible[
        -warm_expected:
    ]
]

expected_ids = [
    sample[
        "request_id"
    ]
    for sample in selected
]

# ============================================================
# Ricostruzione indipendente feature
# ============================================================

rows = []

for sample in selected:
    try:
        rows.append(
            extract(
                sample
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        failures.append(
            f"raw {sample.get('request_id')}: "
            f"feature non derivabili: {exc}"
        )

expected_mean = {}
expected_median = {}

if len(
    rows
) == warm_expected:

    for name in features:
        column = [
            row[
                name
            ]
            for row in rows
        ]

        expected_mean[
            name
        ] = mean_like_go(
            column
        )

        expected_median[
            name
        ] = statistics.median(
            column
        )

# ============================================================
# FunctionProfile
# ============================================================

check(
    len(
        aggregated
    ) == 1,
    (
        "aggregated: "
        f"FunctionProfile trovati={len(aggregated)}, "
        "atteso 1"
    ),
)

if len(
    aggregated
) == 1:

    result = aggregated[0]

    check(
        result.get(
            "schema_version"
        ) == 1,
        "aggregated: schema_version non valido",
    )

    check(
        result.get(
            "function_name"
        ) == function_name,
        "aggregated: function_name non valido",
    )

    check(
        result.get(
            "machine_tag"
        ) == "profiling-test",
        "aggregated: machine_tag non valido",
    )

    check(
        result.get(
            "sample_count"
        )
        ==
        warm_expected,
        "aggregated: sample_count non valido",
    )

    config = result.get(
        "function_configuration",
        {},
    )

    check(
        math.isclose(
            float(
                config.get(
                    "configured_cpus",
                    -1,
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "aggregated: configured_cpus non valido",
    )

    check(
        config.get(
            "configured_memory_mb"
        )
        ==
        memory_expected,
        "aggregated: configured_memory_mb non valido",
    )

    source_ids = result.get(
        "source_request_ids"
    )

    check(
        source_ids
        ==
        expected_ids,
        (
            "aggregated: source_request_ids "
            "non corrispondenti"
        ),
    )

    excluded_ids = {
        sample[
            "request_id"
        ]
        for sample in cold + failed
    }

    check(
        isinstance(
            source_ids,
            list,
        )
        and not excluded_ids.intersection(
            source_ids
        ),
        (
            "aggregated: cold o failed presente "
            "nei source_request_ids"
        ),
    )

    mean = result.get(
        "mean",
        {},
    )

    median = result.get(
        "median",
        {},
    )

    for name in features:
        check(
            finite(
                mean.get(
                    name
                )
            ),
            f"mean: {name} non valido",
        )

        check(
            finite(
                median.get(
                    name
                )
            ),
            f"median: {name} non valido",
        )

        if (
            name in expected_mean
            and finite(
                mean.get(
                    name
                )
            )
            and finite(
                median.get(
                    name
                )
            )
        ):
            check(
                math.isclose(
                    mean[
                        name
                    ],
                    expected_mean[
                        name
                    ],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                (
                    f"mean: {name} non coincide "
                    "con il raw dataset"
                ),
            )

            check(
                math.isclose(
                    median[
                        name
                    ],
                    expected_median[
                        name
                    ],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                (
                    f"median: {name} non coincide "
                    "con il raw dataset"
                ),
            )

# ============================================================
# Risultato
# ============================================================

print(
    "============================================================"
)
print(
    "VALIDAZIONE MODIFICA 08C.2"
)
print(
    "============================================================"
)

print(
    f"raw dataset:            {raw_path}"
)
print(
    f"function profile:       {profile_path}"
)
print(
    f"righe raw valide:       {len(raw)}"
)
print(
    f"cold riusciti:          {len(cold)}"
)
print(
    f"warm riusciti:          {len(warm)}"
)
print(
    f"falliti:                {len(failed)}"
)
print(
    f"warm eleggibili:        {len(eligible)}"
)
print(
    f"FunctionProfile:        {len(aggregated)}"
)

if failures:
    print()

    for failure in failures:
        print(
            f"[FAIL] {failure}"
        )

    sys.exit(
        1
    )

result = aggregated[0]

print()

print(
    f"[PASS] Raw dataset corretto: "
    f"1 cold, {warm_expected} warm, 1 failed."
)

print(
    "[PASS] Solo i warm validi sono "
    "eleggibili per il clustering."
)

print(
    "[PASS] serverledge-profiling ha prodotto "
    "un FunctionProfile reale."
)

print(
    "[PASS] sample_count e source_request_ids "
    "sono corretti."
)

print(
    "[PASS] Cold e failed sono esclusi "
    "dal FunctionProfile."
)

print(
    "[PASS] Mean e Median coincidono con "
    "il ricalcolo dal raw dataset."
)

print(
    "[PASS] FunctionName, MachineTag e "
    "configurazione sono preservati."
)

print()
print(
    "Mean:"
)

for name in features:
    print(
        f"  {name}: "
        f"{result['mean'][name]}"
    )

print(
    "Median:"
)

for name in features:
    print(
        f"  {name}: "
        f"{result['median'][name]}"
    )
PY

echo
echo "[done] Raw dataset:       $RAW_DATASET_FILE"
echo "[done] Function profiles: $PROFILE_DATASET_FILE"
echo "[done] Aggregate log:     $AGGREGATE_LOG"
echo "[done] Log directory:     $LOG_DIR"