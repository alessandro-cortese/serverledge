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

KEPLER_URL="${KEPLER_URL:-http://127.0.0.1:28282/metrics}"

KEPLER_SMOKE_IMAGE="${KEPLER_SMOKE_IMAGE:-busybox:1.36}"

SAMPLE_WAIT_SECONDS="${SAMPLE_WAIT_SECONDS:-7}"

CONTAINER_DISCOVERY_TIMEOUT="${CONTAINER_DISCOVERY_TIMEOUT:-30}"

STAMP="$(date +%Y%m%d_%H%M%S)"

CONTAINER_NAME="serverledge-kepler-smoke-${STAMP}"

OUTPUT_DIR="logs/kepler-local-smoke-${STAMP}"

BEFORE_JSON="$OUTPUT_DIR/before.json"

AFTER_JSON="$OUTPUT_DIR/after.json"

PROBE_ERROR="$OUTPUT_DIR/probe-error.log"

CONTAINER_ID=""

cleanup() {
  status=$?

  set +e

  if [[ -n "$CONTAINER_ID" ]]; then

    echo \
      "[cleanup] Arresto container smoke..."

    docker rm \
      -f \
      "$CONTAINER_ID" \
      >/dev/null \
      2>&1 ||
      true

  fi

  echo \
    "[cleanup] Artefatti: $OUTPUT_DIR"

  exit "$status"
}

trap \
  cleanup \
  EXIT \
  INT \
  TERM

fail() {
  echo \
    "[FAIL] $*" \
    >&2

  exit 1
}

for command in \
  docker \
  curl \
  go \
  python3; do

  command -v \
    "$command" \
    >/dev/null \
    2>&1 ||
    fail \
      "Comando richiesto non trovato: $command"

done

docker info \
  >/dev/null \
  2>&1 ||
  fail \
    "Docker daemon non disponibile."

mkdir -p \
  "$OUTPUT_DIR"

echo \
  "============================================================"

echo \
  "REAL KEPLER LOCAL SMOKE TEST"

echo \
  "============================================================"

echo \
  "Kepler URL:       $KEPLER_URL"

echo \
  "Docker image:     $KEPLER_SMOKE_IMAGE"

echo \
  "Sample wait:      ${SAMPLE_WAIT_SECONDS}s"

echo \
  "Output:           $OUTPUT_DIR"

echo

# ============================================================
# 1. Verifica endpoint Kepler
# ============================================================

echo \
  "[kepler] Verifica endpoint..."

METRICS_FILE="$OUTPUT_DIR/kepler-initial.prom"

curl \
  -fsS \
  --max-time 5 \
  "$KEPLER_URL" \
  > "$METRICS_FILE" ||
  fail \
    "Kepler non raggiungibile su $KEPLER_URL"

grep \
  -q \
  '^kepler_build_info' \
  "$METRICS_FILE" ||
  fail \
    "Endpoint raggiungibile ma kepler_build_info non presente."

echo \
  "[kepler] Exporter Kepler disponibile."

echo \
  "[kepler] Le metriche container verranno verificate dopo la creazione del container."

# ============================================================
# 2. Build del probe Serverledge
# ============================================================

echo \
  "[build] Compilo kepler-probe..."


go build \
  -o bin/kepler-probe \
  ./cmd/kepler

# ============================================================
# 3. Crea container CPU-bound reale
# ============================================================

echo \
  "[docker] Avvio workload CPU-bound reale..."

docker rm \
  -f \
  "$CONTAINER_NAME" \
  >/dev/null \
  2>&1 ||
  true

CONTAINER_ID="$(
  docker run \
    -d \
    --name "$CONTAINER_NAME" \
    "$KEPLER_SMOKE_IMAGE" \
    sh \
    -c \
    'yes > /dev/null'
)"

[[ -n "$CONTAINER_ID" ]] ||
  fail \
    "Docker non ha restituito un container ID."

FULL_CONTAINER_ID="$(
  docker inspect \
    --format '{{.Id}}' \
    "$CONTAINER_ID"
)"

[[ -n "$FULL_CONTAINER_ID" ]] ||
  fail \
    "Impossibile ottenere il container ID completo."

echo \
  "[docker] container_name=$CONTAINER_NAME"

echo \
  "[docker] container_id=$FULL_CONTAINER_ID"

docker ps \
  --filter \
    "id=$FULL_CONTAINER_ID" \
  --format \
    '{{.ID}} {{.Image}} {{.Names}}' \
  > "$OUTPUT_DIR/docker-container.txt"

# ============================================================
# 4. Attendi che Kepler scopra il container
# ============================================================

echo \
  "[kepler] Attendo che il container compaia nelle metriche..."

deadline="$(
  python3 \
    - \
    "$CONTAINER_DISCOVERY_TIMEOUT" <<'PY'

import sys
import time

print(
    time.time()
    + float(
        sys.argv[1]
    )
)

PY
)"

while true; do

  if bin/kepler-probe \
    -url "$KEPLER_URL" \
    -container-id "$FULL_CONTAINER_ID" \
    -timeout 3s \
    > "$BEFORE_JSON" \
    2> "$PROBE_ERROR"; then

    break

  fi

  now="$(
    python3 \
      - <<'PY'

import time

print(
    time.time()
)

PY
  )"

  if python3 \
    - \
    "$now" \
    "$deadline" <<'PY'
import sys

raise SystemExit(
    0
    if float(
        sys.argv[1]
    ) < float(
        sys.argv[2]
    )
    else 1
)
PY
  then

    sleep 0.5

  else

    echo
    echo \
      "[diagnostic] Ultimo errore kepler-probe:" \
      >&2

    cat \
      "$PROBE_ERROR" \
      >&2 ||
      true

    echo
    echo \
      "[diagnostic] Container Docker:" \
      >&2

    docker inspect \
      "$FULL_CONTAINER_ID" \
      > "$OUTPUT_DIR/docker-inspect.json" \
      2>/dev/null ||
      true

    fail \
      "Kepler non ha esposto il container entro ${CONTAINER_DISCOVERY_TIMEOUT}s."
  fi

done

echo \
  "[kepler] Container trovato."

echo
echo \
  "Snapshot BEFORE:"

cat \
  "$BEFORE_JSON"

# ============================================================
# 5. Lascia il workload attivo
# ============================================================

echo
echo \
  "[workload] Carico CPU per ${SAMPLE_WAIT_SECONDS}s..."

sleep \
  "$SAMPLE_WAIT_SECONDS"

# ============================================================
# 6. Secondo snapshot tramite lo stesso KeplerClient
# ============================================================

bin/kepler-probe \
  -url "$KEPLER_URL" \
  -container-id "$FULL_CONTAINER_ID" \
  -timeout 3s \
  > "$AFTER_JSON"

echo
echo \
  "Snapshot AFTER:"

cat \
  "$AFTER_JSON"

# ============================================================
# 7. Verifica avanzamento counter
# ============================================================

python3 \
  - \
  "$BEFORE_JSON" \
  "$AFTER_JSON" \
  "$FULL_CONTAINER_ID" <<'PY'

import json
import math
import sys

from pathlib import Path


before = json.loads(
    Path(
        sys.argv[1]
    ).read_text(
        encoding="utf-8"
    )
)

after = json.loads(
    Path(
        sys.argv[2]
    ).read_text(
        encoding="utf-8"
    )
)

expected_id = sys.argv[3]


before_id = before.get(
    "container_id"
)

after_id = after.get(
    "container_id"
)


if before_id != expected_id:

    raise SystemExit(
        "BEFORE container ID mismatch: "
        f"{before_id!r} != {expected_id!r}"
    )


if after_id != expected_id:

    raise SystemExit(
        "AFTER container ID mismatch: "
        f"{after_id!r} != {expected_id!r}"
    )




before_zones = (
        before.get(
            "cpu_joules_by_zone"
        )
        or {}
)

after_zones = (
        after.get(
            "cpu_joules_by_zone"
        )
        or {}
)


if not before_zones:

    raise SystemExit(
        "Snapshot BEFORE senza zone energetiche"
    )


if not after_zones:

    raise SystemExit(
        "Snapshot AFTER senza zone energetiche"
    )


all_zones = sorted(
    set(
        before_zones
    )
    | set(
        after_zones
    )
)


positive_energy_delta = False

energy_delta_total = 0.0


print()

print(
    "Counter deltas:"
)


for zone in all_zones:

    if (
            zone not in before_zones
            or
            zone not in after_zones
    ):

        raise SystemExit(
            f"zona {zone!r} "
            "non presente in entrambi gli snapshot"
        )

    before_value = float(
        before_zones[
            zone
        ]
    )

    after_value = float(
        after_zones[
            zone
        ]
    )

    delta = (
        after_value
        - before_value
    )

    if delta < 0:

        raise SystemExit(
            "counter energetico regredito "
            f"per zona {zone!r}: "
            f"{before_value} -> {after_value}"
        )

    if delta > 0:

        positive_energy_delta = True

    energy_delta_total += delta

    print(
        f"  energy_delta[{zone}]="
        f"{delta:.9f} J"
    )


if not positive_energy_delta:

    raise SystemExit(
        "nessun counter energetico Kepler "
        "è avanzato durante il workload"
    )


print(
    f"  diagnostic_sum_zone_deltas="
    f"{energy_delta_total:.9f} J"
)

print()

print(
    "[PASS] Kepler counters advanced "
    "for the real Docker container."
)

PY

# ============================================================
# 8. Snapshot raw finale
# ============================================================

curl \
  -fsS \
  --max-time 5 \
  "$KEPLER_URL" \
  > "$OUTPUT_DIR/kepler-final.prom"

echo
echo \
  "============================================================"

echo \
  "REAL KEPLER LOCAL SMOKE TEST: PASS"

echo \
  "============================================================"

echo \
  "✓ Kepler exporter raggiungibile"

echo \
  "✓ metriche container disponibili"

echo \
  "✓ container Docker reale creato"

echo \
  "✓ container_id riconosciuto da Kepler"

echo \
  "✓ KeplerClient Serverledge legge il container"

echo \
  "✓ kepler_container_cpu_joules_total disponibile"

echo \
  "✓ almeno un counter energetico per zone avanza"

echo
echo \
  "Artefatti: $OUTPUT_DIR"