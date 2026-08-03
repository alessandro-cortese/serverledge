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

SAME_REQUESTS="${SAME_REQUESTS:-4}"
SAME_MAX_CONCURRENCY="${SAME_MAX_CONCURRENCY:-4}"
DIFFERENT_REQUESTS="${DIFFERENT_REQUESTS:-2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
API_PORT="${API_PORT:-1333}"
UDP_PORT="${UDP_PORT:-9899}"
MEMORY_MB="${MEMORY_MB:-128}"

STAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="logs/profiling_exclusive_local_${STAMP}"
NODE_LOG="$LOG_DIR/node.log"
ETCD_LOG="$LOG_DIR/etcd.log"
CONF_FILE="$LOG_DIR/node-profiling-conf.yaml"

FUNCTION_SAME="profiling_same_container_${STAMP}"
FUNCTION_DIFF="profiling_different_containers_${STAMP}"

NODE_PID=""
ETCD_STARTED=0

mkdir -p "$LOG_DIR"

fail() {
  echo "[FAIL] $*"
  exit 1
}

cleanup() {
  status=$?

  set +e

  echo
  echo "[cleanup] Arresto dei processi..."

  if [[ -n "$NODE_PID" ]] &&
    kill -0 "$NODE_PID" 2>/dev/null; then

    kill -INT "$NODE_PID" 2>/dev/null

    for _ in $(seq 1 20); do
      if ! kill -0 "$NODE_PID" 2>/dev/null; then
        break
      fi

      sleep 0.5
    done

    if kill -0 "$NODE_PID" 2>/dev/null; then
      kill -KILL "$NODE_PID" 2>/dev/null
    fi

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
  fail "Non trovo il Makefile nella root di Serverledge."

[[ "$SAME_REQUESTS" -ge 2 ]] ||
  fail "SAME_REQUESTS deve essere almeno 2."

[[ "$SAME_MAX_CONCURRENCY" -ge "$SAME_REQUESTS" ]] ||
  fail "SAME_MAX_CONCURRENCY deve essere >= SAME_REQUESTS."

[[ "$DIFFERENT_REQUESTS" -eq 2 ]] ||
  fail "DIFFERENT_REQUESTS deve essere 2."

echo "[setup] Parametri:"
echo "  richieste stesso container:       $SAME_REQUESTS"
echo "  MaxConcurrency stesso container:  $SAME_MAX_CONCURRENCY"
echo "  richieste container differenti:   $DIFFERENT_REQUESTS"
echo "  durata sleeper:                   ${SLEEP_SECONDS}s"
echo "  API port:                          $API_PORT"
echo "  directory log:                    $LOG_DIR"

echo
echo "[setup] Build..."

make

docker info >/dev/null 2>&1 ||
  fail "Docker non è disponibile."

if curl \
  -sS \
  --max-time 1 \
  -o /dev/null \
  "http://127.0.0.1:${API_PORT}/status" \
  2>/dev/null; then

  fail "La porta API ${API_PORT} è già occupata."
fi

docker rm \
  -f \
  Etcd-server \
  >/dev/null 2>&1 ||
  true

cat > "$CONF_FILE" <<YAML
etcd.address: "127.0.0.1:2379"

api.ip: "127.0.0.1"
api.port: ${API_PORT}

registry.area: PROFILE_TEST
registry.node.id: profiling-node-${STAMP}
registry.udp.port: ${UDP_PORT}

node.machine_tag: profiling-test

container.pool.memory: 768
container.pool.cpus: 8
container.expiration: 600

factory.images.refresh: false

profiling.enabled: true

scheduler.queue.capacity: 0
YAML

echo
echo "[setup] Avvio etcd..."

bash scripts/start-etcd.sh \
  >"$ETCD_LOG" \
  2>&1

ETCD_STARTED=1

for _ in $(seq 1 30); do
  if docker exec \
    Etcd-server \
    etcdctl endpoint health \
    >/dev/null 2>&1; then

    break
  fi

  sleep 0.5
done

docker exec \
  Etcd-server \
  etcdctl endpoint health \
  >/dev/null 2>&1 ||
  fail "etcd non è diventato disponibile."

echo "[setup] Avvio Serverledge con profiling.enabled=true..."

bin/serverledge "$CONF_FILE" \
  >"$NODE_LOG" \
  2>&1 &

NODE_PID=$!

for _ in $(seq 1 60); do
  if ! kill -0 "$NODE_PID" 2>/dev/null; then
    tail -80 "$NODE_LOG" || true

    fail "Serverledge è terminato durante l'avvio."
  fi

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
  fail "L'API Serverledge non è diventata disponibile."

create_function() {
  local name="$1"
  local max_concurrency="$2"
  local output="$3"

  bin/serverledge-cli create \
    -f "$name" \
    --memory "$MEMORY_MB" \
    --src examples/sleeper.py \
    --runtime python314 \
    --handler "sleeper.handler" \
    --max_concurrency "$max_concurrency" \
    --input "n:Int" \
    -H 127.0.0.1 \
    -P "$API_PORT" |
    tee "$output"
}

prewarm_function() {
  local name="$1"
  local count="$2"
  local output="$3"

  bin/serverledge-cli prewarm \
    -f "$name" \
    -c "$count" \
    -H 127.0.0.1 \
    -P "$API_PORT" |
    tee "$output"
}

invoke_batch() {
  local name="$1"
  local count="$2"
  local prefix="$3"

  local start_ns
  local end_ns
  local failures=0

  local -a pids=()

  start_ns="$(date +%s%N)"

  for i in $(seq 1 "$count"); do
    (
      bin/serverledge-cli invoke \
        -f "$name" \
        -p "n:${SLEEP_SECONDS}" \
        --return_output \
        -H 127.0.0.1 \
        -P "$API_PORT" \
        >"${prefix}_${i}.log" \
        2>&1
    ) &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done

  end_ns="$(date +%s%N)"

  if [[ "$failures" -ne 0 ]]; then
    echo "[FAIL] ${failures} invocazioni sono fallite."

    for file in "${prefix}"_*.log; do
      echo
      echo "--- $file ---"

      cat "$file"
    done

    return 1
  fi

  python3 \
    - \
    "$start_ns" \
    "$end_ns" <<'PY'
import sys

start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])

elapsed_seconds = (
    end_ns - start_ns
) / 1_000_000_000

print(
    f"{elapsed_seconds:.6f}"
)
PY
}

extract_profiles() {
  local from_line="$1"
  local function_name="$2"
  local output="$3"

  tail \
    -n "+$((from_line + 1))" \
    "$NODE_LOG" |
    grep \
      "event=invocation_resource_profile" |
    grep \
      "function=${function_name}" \
      >"$output" ||
    true
}

echo
echo "============================================================"
echo "FASE A — PIÙ RICHIESTE SULLO STESSO CONTAINER"
echo "============================================================"

echo "[fase A] Creazione funzione: $FUNCTION_SAME"

create_function \
  "$FUNCTION_SAME" \
  "$SAME_MAX_CONCURRENCY" \
  "$LOG_DIR/create_same.json"

echo "[fase A] Prewarm di un solo container..."

prewarm_function \
  "$FUNCTION_SAME" \
  1 \
  "$LOG_DIR/prewarm_same.json"

sleep 1

START_LINE_A="$(
  wc -l < "$NODE_LOG"
)"

echo \
  "[fase A] Invio di ${SAME_REQUESTS} richieste concorrenti da ${SLEEP_SECONDS}s..."

ELAPSED_A="$(
  invoke_batch \
    "$FUNCTION_SAME" \
    "$SAME_REQUESTS" \
    "$LOG_DIR/same_request"
)"

sleep 1

extract_profiles \
  "$START_LINE_A" \
  "$FUNCTION_SAME" \
  "$LOG_DIR/same_profiles.log"

echo "[fase A] Tempo totale: ${ELAPSED_A}s"

echo
echo "============================================================"
echo "FASE B — RICHIESTE SU CONTAINER DIFFERENTI"
echo "============================================================"

echo "[fase B] Creazione funzione: $FUNCTION_DIFF"

create_function \
  "$FUNCTION_DIFF" \
  1 \
  "$LOG_DIR/create_diff.json"

echo "[fase B] Prewarm di due container..."

prewarm_function \
  "$FUNCTION_DIFF" \
  2 \
  "$LOG_DIR/prewarm_diff.json"

sleep 1

START_LINE_B="$(
  wc -l < "$NODE_LOG"
)"

echo \
  "[fase B] Invio di due richieste concorrenti da ${SLEEP_SECONDS}s..."

ELAPSED_B="$(
  invoke_batch \
    "$FUNCTION_DIFF" \
    "$DIFFERENT_REQUESTS" \
    "$LOG_DIR/diff_request"
)"

sleep 1

extract_profiles \
  "$START_LINE_B" \
  "$FUNCTION_DIFF" \
  "$LOG_DIR/diff_profiles.log"

echo "[fase B] Tempo totale: ${ELAPSED_B}s"

echo
echo "============================================================"
echo "VALIDAZIONE AUTOMATICA"
echo "============================================================"

python3 \
  - \
  "$LOG_DIR/same_profiles.log" \
  "$LOG_DIR/diff_profiles.log" \
  "$SAME_REQUESTS" \
  "$SAME_MAX_CONCURRENCY" \
  "$SLEEP_SECONDS" \
  "$ELAPSED_A" \
  "$ELAPSED_B" <<'PY'
import math
import re
import sys
from pathlib import Path

(
    same_file,
    diff_file,
    same_requests_raw,
    same_max_raw,
    sleep_seconds_raw,
    elapsed_a_raw,
    elapsed_b_raw,
) = sys.argv[1:]

same_requests = int(
    same_requests_raw
)

same_max = int(
    same_max_raw
)

sleep_seconds = float(
    sleep_seconds_raw
)

elapsed_a = float(
    elapsed_a_raw
)

elapsed_b = float(
    elapsed_b_raw
)

field_pattern = re.compile(
    r'([A-Za-z_]+)=(?:"([^"]*)"|([^\s]+))'
)


def parse_profiles(path):
    events = []

    content = Path(path).read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in content.splitlines():
        event = {}

        for match in field_pattern.finditer(
            line
        ):
            key = match.group(1)

            value = (
                match.group(2)
                if match.group(2) is not None
                else match.group(3)
            )

            event[key] = value

        if (
            event.get("event")
            == "invocation_resource_profile"
        ):
            events.append(
                event
            )

    return events


def number(
    event,
    key,
):
    try:
        return float(
            event[key]
        )
    except (
        KeyError,
        ValueError,
    ):
        return math.nan


same_profiles = parse_profiles(
    same_file
)

diff_profiles = parse_profiles(
    diff_file
)

failures = []

if len(same_profiles) != same_requests:
    failures.append(
        "fase A: attesi "
        f"{same_requests} profili, "
        f"trovati {len(same_profiles)}"
    )

if len(diff_profiles) != 2:
    failures.append(
        "fase B: attesi 2 profili, "
        f"trovati {len(diff_profiles)}"
    )

same_container_ids = {
    profile.get(
        "container_id",
        "",
    )
    for profile in same_profiles
    if profile.get(
        "container_id"
    )
}

diff_container_ids = {
    profile.get(
        "container_id",
        "",
    )
    for profile in diff_profiles
    if profile.get(
        "container_id"
    )
}

if len(same_container_ids) != 1:
    failures.append(
        "fase A: atteso un solo container, "
        f"osservati {sorted(same_container_ids)}"
    )

if len(diff_container_ids) != 2:
    failures.append(
        "fase B: attesi due container differenti, "
        f"osservati {sorted(diff_container_ids)}"
    )

for (
    label,
    profiles,
    expected_max,
) in (
    (
        "fase A",
        same_profiles,
        str(same_max),
    ),
    (
        "fase B",
        diff_profiles,
        "1",
    ),
):
    for (
        index,
        profile,
    ) in enumerate(
        profiles,
        start=1,
    ):
        if (
            profile.get(
                "collected"
            )
            != "true"
        ):
            failures.append(
                f"{label}, profilo {index}: "
                "collected="
                f"{profile.get('collected')}"
            )

        if (
            profile.get(
                "valid"
            )
            != "true"
        ):
            failures.append(
                f"{label}, profilo {index}: "
                "valid="
                f"{profile.get('valid')} "
                "reason="
                f"{profile.get('invalid_reason')}"
            )

        if (
            profile.get(
                "exclusive_container"
            )
            != "true"
        ):
            failures.append(
                f"{label}, profilo {index}: "
                "exclusive_container="
                f"{profile.get('exclusive_container')}"
            )

        if (
            profile.get(
                "max_concurrency"
            )
            != expected_max
        ):
            failures.append(
                f"{label}, profilo {index}: "
                "max_concurrency="
                f"{profile.get('max_concurrency')}, "
                f"atteso={expected_max}"
            )

same_waits = [
    number(
        profile,
        "profiling_lock_wait_ms",
    )
    for profile in same_profiles
]

diff_waits = [
    number(
        profile,
        "profiling_lock_wait_ms",
    )
    for profile in diff_profiles
]

if any(
    not math.isfinite(value)
    or value < 0
    for value in (
        same_waits
        + diff_waits
    )
):
    failures.append(
        "profiling_lock_wait_ms "
        "assente o non valido"
    )

wait_threshold_ms = (
    sleep_seconds
    * 1000
    * 0.50
)

if (
    same_waits
    and max(same_waits)
    < wait_threshold_ms
):
    failures.append(
        "fase A: attesa massima "
        "troppo bassa "
        f"({max(same_waits):.3f} ms; "
        f"soglia {wait_threshold_ms:.3f} ms)"
    )

serial_threshold_seconds = (
    same_requests
    * sleep_seconds
    * 0.75
)

if elapsed_a < serial_threshold_seconds:
    failures.append(
        "fase A: tempo troppo breve "
        "per la serializzazione "
        f"({elapsed_a:.3f}s; "
        "soglia "
        f"{serial_threshold_seconds:.3f}s)"
    )

if elapsed_b >= elapsed_a:
    failures.append(
        "fase B non risulta più rapida "
        "della fase A "
        f"(A={elapsed_a:.3f}s, "
        f"B={elapsed_b:.3f}s)"
    )

print(
    "profili fase A:              "
    f"{len(same_profiles)}"
)

print(
    "container unici fase A:     "
    f"{len(same_container_ids)}"
)

print(
    "lock wait fase A (ms):      "
    f"{same_waits}"
)

print(
    "tempo totale fase A (s):    "
    f"{elapsed_a:.6f}"
)

print()

print(
    "profili fase B:              "
    f"{len(diff_profiles)}"
)

print(
    "container unici fase B:     "
    f"{len(diff_container_ids)}"
)

print(
    "lock wait fase B (ms):      "
    f"{diff_waits}"
)

print(
    "tempo totale fase B (s):    "
    f"{elapsed_b:.6f}"
)

print()

print(
    "TRACE FASE A"
)

for profile in sorted(
    same_profiles,
    key=lambda item: number(
        item,
        "profiling_lock_wait_ms",
    ),
):
    print(
        "  "
        "request_id="
        f"{profile.get('request_id')} "
        "container_id="
        f"{profile.get('container_id')} "
        "valid="
        f"{profile.get('valid')} "
        "exclusive="
        f"{profile.get('exclusive_container')} "
        "max_concurrency="
        f"{profile.get('max_concurrency')} "
        "lock_wait_ms="
        f"{profile.get('profiling_lock_wait_ms')} "
        "execution_ms="
        f"{profile.get('execution_wall_time_ms')}"
    )

print()
print(
    "TRACE FASE B"
)

for profile in sorted(
    diff_profiles,
    key=lambda item: item.get(
        "container_id",
        "",
    ),
):
    print(
        "  "
        "request_id="
        f"{profile.get('request_id')} "
        "container_id="
        f"{profile.get('container_id')} "
        "valid="
        f"{profile.get('valid')} "
        "exclusive="
        f"{profile.get('exclusive_container')} "
        "max_concurrency="
        f"{profile.get('max_concurrency')} "
        "lock_wait_ms="
        f"{profile.get('profiling_lock_wait_ms')} "
        "execution_ms="
        f"{profile.get('execution_wall_time_ms')}"
    )

print()

if failures:
    for failure in failures:
        print(
            f"[FAIL] {failure}"
        )

    sys.exit(1)

print(
    "[PASS] Le richieste della fase A "
    "usano un solo container."
)

print(
    "[PASS] La finestra profilata sullo "
    "stesso container è serializzata."
)

print(
    "[PASS] I profili con MaxConcurrency > 1 "
    "sono validi ed esclusivi."
)

print(
    "[PASS] profiling_lock_wait_ms evidenzia "
    "l'attesa sullo stesso container."
)

print(
    "[PASS] Le richieste della fase B "
    "usano container differenti."
)

print(
    "[PASS] Container differenti non "
    "condividono il lock di profilazione."
)

print(
    "[PASS] Tutti i profili osservati sono "
    "collected=true e valid=true."
)
PY

echo
echo "[done] Funzione fase A: $FUNCTION_SAME"
echo "[done] Funzione fase B: $FUNCTION_DIFF"
echo "[done] Log completi: $LOG_DIR"