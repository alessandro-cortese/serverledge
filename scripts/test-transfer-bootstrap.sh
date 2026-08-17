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

fail() {
  echo \
    "[FAIL] $*" \
    >&2

  exit 1
}

TMP_DIR="$(
  mktemp -d
)"

trap \
  'rm -rf "$TMP_DIR"' \
  EXIT \
  INT \
  TERM

VALID_MANIFEST="$TMP_DIR/valid.conf"

INVALID_MANIFEST="$TMP_DIR/missing-arm.conf"

CATALOG="$TMP_DIR/catalog.json"

BAD_CATALOG="$TMP_DIR/bad-catalog.json"

MODEL="$TMP_DIR/model.json"

cat > "$VALID_MANIFEST" <<'MANIFEST'

x86-node|x86|user@x86|/tmp/x86/profiling-samples.jsonl|22
arm-node|arm64|user@arm|/tmp/arm/profiling-samples.jsonl|22

MANIFEST

cat > "$INVALID_MANIFEST" <<'MANIFEST'

x86-node|x86|user@x86|/tmp/x86/profiling-samples.jsonl|22

MANIFEST

cat > "$CATALOG" <<'JSON'

{
  "clustering": {
    "aggregation": "mean",
    "profile_machine_tag": "x86"
  }
}

JSON

cat > "$BAD_CATALOG" <<'JSON'

{
  "clustering": {
    "aggregation": "mean",
    "profile_machine_tag": "gpu"
  }
}

JSON

echo '{}' \
  > "$MODEL"

COMMON_ARGS=(
  --function
  target-function

  --x86-node-name
  x86-node

  --x86-host
  192.0.2.10

  --x86-port
  1323

  --arm-node-name
  arm-node

  --arm-host
  192.0.2.20

  --arm-port
  1323

  --manifest
  "$VALID_MANIFEST"

  --catalog
  "$CATALOG"

  --model
  "$MODEL"

  --lb-url
  http://192.0.2.30:8080

  --max-distance
  1.0

  --samples-per-arch
  2

  --experiment
  bootstrap-script-test

  --plan-only
)

bash -n \
  scripts/transfer_bootstrap.sh

PLAN_OUTPUT="$TMP_DIR/plan.txt"

PYTHON_BIN=python3 \
  scripts/transfer_bootstrap.sh \
  "${COMMON_ARGS[@]}" \
  > "$PLAN_OUTPUT"

grep \
  -q \
  'PLAN ONLY' \
  "$PLAN_OUTPUT" ||
  fail \
    "plan-only non rilevato"

grep \
  -q \
  'CanDoOffloading=false' \
  "$PLAN_OUTPUT" ||
  fail \
    "manca il vincolo no-offloading"

grep \
  -q \
  'samples_per_arch:     10' \
  "$PLAN_OUTPUT" ||
  fail \
    "samples-per-arch non propagato"

grep \
  -q \
  'catalog aggregation:  mean' \
  "$PLAN_OUTPUT" ||
  fail \
    "aggregation catalogo non letta"

grep \
  -q \
  'catalog machine tag:  x86' \
  "$PLAN_OUTPUT" ||
  fail \
    "machine tag catalogo non letto"

echo \
  '[PASS] valid plan-only workflow'

MISSING_ARM_ARGS=(
  "${COMMON_ARGS[@]}"
)

for i in "${!MISSING_ARM_ARGS[@]}"; do

  if [[ "${MISSING_ARM_ARGS[$i]}" == "$VALID_MANIFEST" ]]; then

    MISSING_ARM_ARGS["$i"]="$INVALID_MANIFEST"

  fi

done

if PYTHON_BIN=python3 \
  scripts/transfer_bootstrap.sh \
  "${MISSING_ARM_ARGS[@]}" \
  >/dev/null \
  2>&1; then

  fail \
    "manifest senza nodo ARM accettato"

fi

echo \
  '[PASS] missing ARM node rejected'

if PYTHON_BIN=python3 \
  scripts/transfer_bootstrap.sh \
  "${COMMON_ARGS[@]}" \
  --prior-weight \
  2 \
  >/dev/null \
  2>&1; then

  fail \
    "prior weight > 1 accettato"

fi

echo \
  '[PASS] invalid prior weight rejected'

BAD_CATALOG_ARGS=(
  "${COMMON_ARGS[@]}"
)

for i in "${!BAD_CATALOG_ARGS[@]}"; do

  if [[ "${BAD_CATALOG_ARGS[$i]}" == "$CATALOG" ]]; then

    BAD_CATALOG_ARGS["$i"]="$BAD_CATALOG"

  fi

done

if PYTHON_BIN=python3 \
  scripts/transfer_bootstrap.sh \
  "${BAD_CATALOG_ARGS[@]}" \
  >/dev/null \
  2>&1; then

  fail \
    "catalog profile_machine_tag estraneo accettato"

fi

echo \
  '[PASS] incompatible catalog machine tag rejected'

echo

echo \
  '[PASS] transfer bootstrap shell tests completed'