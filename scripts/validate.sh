#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

API_PORT="${API_PORT:-1333}"
UDP_PORT="${UDP_PORT:-9899}"
MEMORY_MB="${MEMORY_MB:-128}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="logs/profiling_export_local_${STAMP}"
NODE_LOG="$LOG_DIR/node.log"
ETCD_LOG="$LOG_DIR/etcd.log"
CONF_FILE="$LOG_DIR/node-profiling-export-conf.yaml"
DATASET_FILE="$ROOT_DIR/$LOG_DIR/profiling-samples.jsonl"

FUNCTION_NAME="profiling_export_${STAMP}"

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

# ============================================================
# Build
# ============================================================

echo "[setup] Build..."
make

docker info >/dev/null 2>&1 ||
  fail "Docker non è disponibile."

docker rm -f Etcd-server >/dev/null 2>&1 || true

rm -f "$DATASET_FILE"

# ============================================================
# Configurazione nodo
# ============================================================

cat >"$CONF_FILE" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: PROFILE_EXPORT_TEST
registry.node.id: profiling-export-node-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: profiling-test

container.pool.memory: 768
container.pool.cpus: 8
container.expiration: 600
factory.images.refresh: false

profiling.enabled: true
profiling.export.enabled: true
profiling.export.path: "${DATASET_FILE}"

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
# Creazione funzione
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
# Cold riuscito
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
# Warm riuscito
# ============================================================

echo "[test] Invocazione warm riuscita..."

bin/serverledge-cli invoke \
  -f "$FUNCTION_NAME" \
  -p "n:${SLEEP_SECONDS}" \
  --return_output \
  -H 127.0.0.1 \
  -P "$API_PORT" \
  >"$LOG_DIR/warm.log" 2>&1

# ============================================================
# Warm fallito intenzionalmente
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

if [[ "$FAILED_STATUS" -eq 0 ]]; then
  fail "L'invocazione volutamente errata è risultata riuscita."
fi

# ============================================================
# Attesa esportazione JSONL
# ============================================================

# L'export è eseguito in defer dopo la completion notification.
# Attendiamo esplicitamente che le tre righe siano visibili.

for _ in $(seq 1 50); do
  if [[ -f "$DATASET_FILE" ]] &&
    [[ "$(wc -l <"$DATASET_FILE")" -ge 3 ]]; then
    break
  fi

  sleep 0.2
done

[[ -n "$DATASET_FILE" ]] ||
  fail "DATASET_FILE è vuoto."

[[ "$DATASET_FILE" != "." ]] ||
  fail "DATASET_FILE punta alla directory corrente."

[[ -f "$DATASET_FILE" ]] ||
  fail "Dataset JSONL non creato: $DATASET_FILE"

echo "[setup] Dataset da validare: $DATASET_FILE"

# ============================================================
# Validazione JSONL
# ============================================================

python3 - "$DATASET_FILE" "$FUNCTION_NAME" <<'PY'
import json
import math
import sys
from pathlib import Path


# ============================================================
# Argomenti
# ============================================================

if len(sys.argv) != 3:
    print(
        "[FAIL] Argomenti Python non validi: "
        f"ricevuti {sys.argv!r}"
    )
    sys.exit(1)

dataset_argument = sys.argv[1]
function_name = sys.argv[2]

if not dataset_argument:
    print(
        "[FAIL] Il percorso del dataset passato "
        "al validatore Python è vuoto."
    )
    sys.exit(1)

path = Path(dataset_argument)

if not path.is_file():
    print(
        "[FAIL] Il dataset non è un file regolare: "
        f"{path}"
    )
    sys.exit(1)

raw = path.read_bytes()

failures = []

# ============================================================
# Struttura JSONL
# ============================================================

if not raw.endswith(b"\n"):
    failures.append(
        "il file JSONL non termina con newline"
    )

lines = [
    line
    for line in raw.splitlines()
    if line.strip()
]

samples = []

for index, line in enumerate(lines, start=1):
    try:
        sample = json.loads(line)
    except json.JSONDecodeError as exc:
        failures.append(
            f"riga {index}: JSON non valido: {exc}"
        )
        continue

    samples.append(sample)

if len(samples) != 3:
    failures.append(
        f"attesi 3 campioni, trovati {len(samples)}"
    )

# ============================================================
# Controlli comuni
# ============================================================

for index, sample in enumerate(samples, start=1):
    if sample.get("schema_version") != 3:
        failures.append(
            f"campione {index}: "
            "schema_version non valido"
        )

    if sample.get("function_name") != function_name:
        failures.append(
            f"campione {index}: "
            "function_name non valido"
        )

    if not sample.get("request_id"):
        failures.append(
            f"campione {index}: "
            "request_id mancante"
        )

    if not sample.get("machine_tag"):
        failures.append(
            f"campione {index}: "
            "machine_tag mancante"
        )

    if not sample.get("node_name"):
        failures.append(
            f"campione {index}: "
            "node_name mancante"
        )

    if "eligibility" not in sample:
        failures.append(
            f"campione {index}: "
            "eligibility mancante"
        )

    # --------------------------------------------------------
    # Configurazione funzione
    # --------------------------------------------------------

    function_configuration = sample.get(
        "function_configuration"
    )

    if not isinstance(
        function_configuration,
        dict,
    ):
        failures.append(
            f"campione {index}: "
            "function_configuration mancante"
        )
    else:
        configured_cpus = (
            function_configuration.get(
                "configured_cpus"
            )
        )

        configured_memory_mb = (
            function_configuration.get(
                "configured_memory_mb"
            )
        )

        if (
            not isinstance(
                configured_cpus,
                (int, float),
            )
            or not math.isfinite(
                configured_cpus
            )
            or configured_cpus <= 0
        ):
            failures.append(
                f"campione {index}: "
                "configured_cpus non valido"
            )

        if (
            not isinstance(
                configured_memory_mb,
                int,
            )
            or configured_memory_mb <= 0
        ):
            failures.append(
                f"campione {index}: "
                "configured_memory_mb non valido"
            )

# ============================================================
# Request ID unici
# ============================================================

request_ids = [
    sample.get("request_id")
    for sample in samples
]

if len(set(request_ids)) != len(request_ids):
    failures.append(
        "request_id duplicati nel dataset"
    )

# ============================================================
# Classificazione campioni
# ============================================================

cold_success = [
    sample
    for sample in samples
    if sample.get("warm_start") is False
    and sample.get("execution_succeeded") is True
]

warm_success = [
    sample
    for sample in samples
    if sample.get("warm_start") is True
    and sample.get("execution_succeeded") is True
]

failed = [
    sample
    for sample in samples
    if sample.get("execution_succeeded") is False
]

# ============================================================
# Cold riuscito
# ============================================================

if len(cold_success) != 1:
    failures.append(
        f"atteso 1 campione cold riuscito, "
        f"trovati {len(cold_success)}"
    )
else:
    sample = cold_success[0]

    eligibility = sample.get(
        "eligibility",
        {},
    )

    timing = sample.get(
        "timing",
        {},
    )

    if sample.get("profile") not in (
        None,
        {},
    ):
        failures.append(
            "cold: il profilo risorse "
            "deve essere assente"
        )

    if sample.get("node_environment") not in (
        None,
        {},
    ):
        failures.append(
            "cold: il profilo del nodo "
            "deve essere assente"
        )

    if (
        eligibility.get(
            "resource_clustering"
        )
        is not False
    ):
        failures.append(
            "cold: resource_clustering "
            "deve essere false"
        )

    if (
        eligibility.get(
            "cold_start_analysis"
        )
        is not True
    ):
        failures.append(
            "cold: cold_start_analysis "
            "deve essere true"
        )

    if (
        eligibility.get(
            "performance_analysis"
        )
        is not False
    ):
        failures.append(
            "cold: performance_analysis "
            "deve essere false"
        )

    if (
        "cold_start"
        not in eligibility.get(
            "exclusion_reasons",
            [],
        )
    ):
        failures.append(
            "cold: exclusion_reasons "
            "non contiene cold_start"
        )

    init_time = timing.get(
        "init_time_ms"
    )

    if (
        not isinstance(
            init_time,
            (int, float),
        )
        or not math.isfinite(
            init_time
        )
        or init_time < 0
    ):
        failures.append(
            "cold: init_time_ms non valido"
        )

# ============================================================
# Warm riuscito
# ============================================================

if len(warm_success) != 1:
    failures.append(
        f"atteso 1 campione warm riuscito, "
        f"trovati {len(warm_success)}"
    )
else:
    sample = warm_success[0]

    eligibility = sample.get(
        "eligibility",
        {},
    )

    profile = sample.get(
        "profile"
    )

    # --------------------------------------------------------
    # Profilo container
    # --------------------------------------------------------

    if not isinstance(
        profile,
        dict,
    ):
        failures.append(
            "warm: profilo risorse assente"
        )
    else:
        for key, expected in (
            ("Enabled", True),
            ("Collected", True),
            ("Valid", True),
            ("ExclusiveContainer", True),
            ("PageFaultsAvailable", True),
        ):
            if profile.get(key) is not expected:
                failures.append(
                    f"warm: "
                    f"{key}={profile.get(key)!r}, "
                    f"atteso {expected!r}"
                )

        for key in (
            "PageFaultsDelta",
            "CPUUsageUserDeltaNs",
            "CPUUsageKernelDeltaNs",
            "ProfilingStartOverheadMs",
        ):
            value = profile.get(
                key
            )

            if (
                not isinstance(
                    value,
                    (int, float),
                )
                or not math.isfinite(
                    value
                )
                or value < 0
            ):
                failures.append(
                    f"warm: {key} non valido"
                )

    # --------------------------------------------------------
    # Profilo nodo
    # --------------------------------------------------------

    node_environment = sample.get(
        "node_environment"
    )

    if not isinstance(
        node_environment,
        dict,
    ):
        failures.append(
            "warm: node_environment assente"
        )
    else:
        for key, expected in (
            ("Collected", True),
            ("CPUAvailable", True),
            ("MemoryAvailable", True),
            ("VMStatAvailable", True),
        ):
            if (
                node_environment.get(key)
                is not expected
            ):
                failures.append(
                    f"warm node_environment: "
                    f"{key}="
                    f"{node_environment.get(key)!r}, "
                    f"atteso {expected!r}"
                )

        available_cpus = (
            node_environment.get(
                "AvailableCPUs"
            )
        )

        if (
            not isinstance(
                available_cpus,
                int,
            )
            or available_cpus <= 0
        ):
            failures.append(
                "warm node_environment: "
                "AvailableCPUs non valido"
            )

        cpu_total_ticks = (
            node_environment.get(
                "CPUTotalDeltaTicks"
            )
        )

        if (
            not isinstance(
                cpu_total_ticks,
                int,
            )
            or cpu_total_ticks <= 0
        ):
            failures.append(
                "warm node_environment: "
                "CPUTotalDeltaTicks non valido"
            )

        total_memory_after = (
            node_environment.get(
                "TotalMemoryAfterBytes"
            )
        )

        if (
            not isinstance(
                total_memory_after,
                int,
            )
            or total_memory_after <= 0
        ):
            failures.append(
                "warm node_environment: "
                "TotalMemoryAfterBytes non valido"
            )

        # freeMemory utilizzata nel vettore RF.
        free_memory_before = (
            node_environment.get(
                "FreeMemoryBeforeBytes"
            )
        )

        if (
            not isinstance(
                free_memory_before,
                int,
            )
            or free_memory_before < 0
        ):
            failures.append(
                "warm node_environment: "
                "FreeMemoryBeforeBytes non valido"
            )

        # Page fault assoluti.
        page_faults_before = (
            node_environment.get(
                "PageFaultsBefore"
            )
        )

        page_faults_after = (
            node_environment.get(
                "PageFaultsAfter"
            )
        )

        if (
            not isinstance(
                page_faults_before,
                int,
            )
            or page_faults_before < 0
        ):
            failures.append(
                "warm node_environment: "
                "PageFaultsBefore non valido"
            )

        if (
            not isinstance(
                page_faults_after,
                int,
            )
            or page_faults_after
            < page_faults_before
        ):
            failures.append(
                "warm node_environment: "
                "PageFaultsAfter non valido"
            )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    timing = sample.get(
        "timing",
        {},
    )

    duration_ms = timing.get(
        "duration_ms"
    )

    if (
        not isinstance(
            duration_ms,
            (int, float),
        )
        or not math.isfinite(
            duration_ms
        )
        or duration_ms <= 0
    ):
        failures.append(
            "warm: duration_ms non valido "
            "per il feature vector"
        )

    # --------------------------------------------------------
    # Verifica delle 6 feature Random Forest
    # --------------------------------------------------------

    if (
        isinstance(
            profile,
            dict,
        )
        and isinstance(
            node_environment,
            dict,
        )
        and isinstance(
            duration_ms,
            (int, float),
        )
        and math.isfinite(
            duration_ms
        )
        and duration_ms > 0
    ):
        page_faults_delta = (
            profile.get(
                "PageFaultsDelta"
            )
        )

        cpu_user_ns = (
            profile.get(
                "CPUUsageUserDeltaNs"
            )
        )

        cpu_kernel_ns = (
            profile.get(
                "CPUUsageKernelDeltaNs"
            )
        )

        free_memory_bytes = (
            node_environment.get(
                "FreeMemoryBeforeBytes"
            )
        )

        container_start_ms = (
            profile.get(
                "ProfilingStartOverheadMs"
            )
        )

        node_start_ms = (
            node_environment.get(
                "SnapshotStartOverheadMs"
            )
        )

        raw_feature_inputs = (
            page_faults_delta,
            cpu_user_ns,
            cpu_kernel_ns,
            free_memory_bytes,
            container_start_ms,
            node_start_ms,
        )

        if all(
            isinstance(
                value,
                (int, float),
            )
            and math.isfinite(
                value
            )
            and value >= 0
            for value in raw_feature_inputs
        ):
            # 1. pageFaultsDelta
            rf_page_faults_delta = float(
                page_faults_delta
            )

            # 2. utilizedCPUs
            utilized_cpus = (
                (
                    cpu_user_ns
                    + cpu_kernel_ns
                )
                /
                (
                    duration_ms
                    * 1_000_000.0
                )
            )

            # 3. freeMemory
            free_memory_mb = (
                free_memory_bytes
                /
                (1024.0 * 1024.0)
            )

            # 4. cpuUserDelta
            cpu_user_delta_ms = (
                cpu_user_ns
                /
                1_000_000.0
            )

            # 5. cpuKernelDelta
            cpu_kernel_delta_ms = (
                cpu_kernel_ns
                /
                1_000_000.0
            )

            # 6. frameworkRuntime
            framework_runtime_ms = (
                container_start_ms
                + node_start_ms
            )

            rf_features = (
                rf_page_faults_delta,
                utilized_cpus,
                free_memory_mb,
                cpu_user_delta_ms,
                cpu_kernel_delta_ms,
                framework_runtime_ms,
            )

            if not all(
                math.isfinite(
                    value
                )
                and value >= 0
                for value in rf_features
            ):
                failures.append(
                    "warm: le 6 feature "
                    "Random Forest non sono "
                    "tutte valide"
                )

        else:
            failures.append(
                "warm: dati insufficienti "
                "per costruire le 6 feature "
                "Random Forest"
            )

    # --------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------

    if (
        eligibility.get(
            "resource_clustering"
        )
        is not True
    ):
        failures.append(
            "warm: resource_clustering "
            "deve essere true"
        )

    if (
        eligibility.get(
            "cold_start_analysis"
        )
        is not False
    ):
        failures.append(
            "warm: cold_start_analysis "
            "deve essere false"
        )

    if (
        eligibility.get(
            "performance_analysis"
        )
        is not True
    ):
        failures.append(
            "warm: performance_analysis "
            "deve essere true"
        )

# ============================================================
# Esecuzione fallita
# ============================================================

if len(failed) != 1:
    failures.append(
        f"atteso 1 campione fallito, "
        f"trovati {len(failed)}"
    )
else:
    sample = failed[0]

    eligibility = sample.get(
        "eligibility",
        {},
    )

    if not sample.get(
        "execution_error"
    ):
        failures.append(
            "failed: execution_error mancante"
        )

    if (
        eligibility.get(
            "resource_clustering"
        )
        is not False
    ):
        failures.append(
            "failed: resource_clustering "
            "deve essere false"
        )

    if (
        eligibility.get(
            "performance_analysis"
        )
        is not False
    ):
        failures.append(
            "failed: performance_analysis "
            "deve essere false"
        )

    if (
        "execution_failed"
        not in eligibility.get(
            "exclusion_reasons",
            [],
        )
    ):
        failures.append(
            "failed: exclusion_reasons "
            "non contiene execution_failed"
        )

# ============================================================
# Risultato
# ============================================================

print(
    "============================================================"
)
print(
    "VALIDAZIONE MODIFICA 08A"
)
print(
    "============================================================"
)

print(f"file:                   {path}")
print(f"righe JSON valide:      {len(samples)}")
print(f"cold riusciti:          {len(cold_success)}")
print(f"warm riusciti:          {len(warm_success)}")
print(f"falliti:                {len(failed)}")
print(
    f"request ID unici:       "
    f"{len(set(request_ids))}"
)

if failures:
    print()

    for failure in failures:
        print(
            f"[FAIL] {failure}"
        )

    sys.exit(1)

print()

print(
    "[PASS] Ogni invocazione ha prodotto "
    "una sola riga JSON valida."
)

print(
    "[PASS] Il cold start conserva InitTime "
    "ma non i profili di risorse."
)

print(
    "[PASS] Il warm riuscito è eleggibile "
    "per il clustering."
)

print(
    "[PASS] Il warm contiene metriche "
    "node-scoped valide da /proc."
)

print(
    "[PASS] ConfiguredCPUs e ConfiguredMemoryMB "
    "sono presenti nel campione."
)

print(
    "[PASS] PageFaultsBefore, PageFaultsAfter "
    "e PageFaultsDelta sono disponibili."
)

print(
    "[PASS] Le 6 feature principali del Random Forest "
    "sono disponibili e derivabili dal campione warm."
)

print(
    "[PASS] L'esecuzione fallita è conservata "
    "ma non è eleggibile."
)

print(
    "[PASS] Metadati, schema ed eligibility "
    "sono presenti."
)
PY

echo
echo "[done] Dataset: $DATASET_FILE"
echo "[done] Log:     $LOG_DIR"