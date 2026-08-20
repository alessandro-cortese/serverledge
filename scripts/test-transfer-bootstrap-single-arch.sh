#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(
  cd \
    "$(dirname "${BASH_SOURCE[0]}")/.." \
    && pwd
)"

cd "$ROOT_DIR"

BOOTSTRAP_SCRIPT="scripts/transfer_bootstrap.sh"

fail() {
  echo \
    "[FAIL] $*" \
    >&2

  exit 1
}

pass() {
  echo \
    "[PASS] $*"
}

[[ -x "$BOOTSTRAP_SCRIPT" ]] ||
  fail \
    "Script bootstrap non eseguibile: $BOOTSTRAP_SCRIPT"

TMP_DIR="$(
  mktemp -d
)"

cleanup() {
  rm -rf \
    "$TMP_DIR"
}

trap \
  cleanup \
  EXIT \
  INT \
  TERM

# ============================================================
# Fixture
# ============================================================

MANIFEST="$TMP_DIR/manifest.conf"

CATALOG_X86="$TMP_DIR/catalog-x86.json"
CATALOG_ARM="$TMP_DIR/catalog-arm64.json"

MODEL="$TMP_DIR/model.json"

PLAN_X86="$TMP_DIR/plan-x86.txt"
PLAN_ARM="$TMP_DIR/plan-arm64.txt"

cat > "$MANIFEST" <<'EOF'
x86-node|x86|user@x86|/tmp/x86/profiling-samples.jsonl|22
arm-node|arm64|user@arm|/tmp/arm/profiling-samples.jsonl|22
EOF

cat > "$CATALOG_X86" <<'EOF'
{
  "clustering": {
    "aggregation": "mean",
    "profile_machine_tag": "x86"
  }
}
EOF

cat > "$CATALOG_ARM" <<'EOF'
{
  "clustering": {
    "aggregation": "mean",
    "profile_machine_tag": "arm64"
  }
}
EOF

echo '{}' \
  > "$MODEL"

# ============================================================
# Helper
# ============================================================

run_plan() {
  local catalog="$1"
  local experiment="$2"
  local output="$3"

  PYTHON_BIN=python3 \
  "$BOOTSTRAP_SCRIPT" \
    --function target-function \
    --x86-node-name x86-node \
    --x86-host 192.0.2.10 \
    --x86-port 1323 \
    --x86-tag x86 \
    --arm-node-name arm-node \
    --arm-host 192.0.2.20 \
    --arm-port 1323 \
    --arm-tag arm64 \
    --manifest "$MANIFEST" \
    --catalog "$catalog" \
    --model "$MODEL" \
    --lb-url http://192.0.2.30:8080 \
    --max-distance 1.0 \
    --samples-per-arch 10 \
    --experiment "$experiment" \
    --prior-weight 0.5 \
    --min-donor-observations 2 \
    --plan-only \
    > "$output"
}

assert_contains() {
  local file="$1"
  local pattern="$2"
  local description="$3"

  if grep \
    -Eq \
    "$pattern" \
    "$file"; then

    pass \
      "$description"

    return
  fi

  echo
  echo \
    "===== OUTPUT ====="

  cat \
    "$file"

  echo

  fail \
    "$description"
}

assert_not_contains() {
  local file="$1"
  local pattern="$2"
  local description="$3"

  if grep \
    -Eiq \
    "$pattern" \
    "$file"; then

    echo
    echo \
      "===== MATCH NON ATTESO ====="

    grep \
      -Ein \
      "$pattern" \
      "$file" \
      || true

    echo

    fail \
      "$description"
  fi

  pass \
    "$description"
}

# ============================================================
# Syntax
# ============================================================

bash -n \
  "$BOOTSTRAP_SCRIPT"

pass \
  "transfer_bootstrap.sh sintatticamente valido"

# ============================================================
# x86 bootstrap
# ============================================================

echo
echo \
  "============================================================"

echo \
  "TEST: catalog x86 -> bootstrap solo x86"

echo \
  "============================================================"

run_plan \
  "$CATALOG_X86" \
  "test-single-x86" \
  "$PLAN_X86"

assert_contains \
  "$PLAN_X86" \
  '^bootstrap_samples:[[:space:]]+10$' \
  "x86: numero bootstrap sample corretto"

assert_contains \
  "$PLAN_X86" \
  '^bootstrap_arch:[[:space:]]+x86$' \
  "x86: bootstrap_arch corretto"

assert_contains \
  "$PLAN_X86" \
  '^bootstrap_machine_tag:[[:space:]]+x86$' \
  "x86: catalog machine tag seleziona x86"

assert_contains \
  "$PLAN_X86" \
  '^bootstrap_node:[[:space:]]+x86-node$' \
  "x86: nodo bootstrap corretto"

assert_contains \
  "$PLAN_X86" \
  '^bootstrap_api:[[:space:]]+http://192\.0\.2\.10:1323$' \
  "x86: API bootstrap corretta"

assert_contains \
  "$PLAN_X86" \
  'invoke the target 10 times on that node' \
  "x86: le N probe sono eseguite su un solo nodo"

assert_contains \
  "$PLAN_X86" \
  '\[PLAN ONLY\]' \
  "x86: nessuna modifica runtime in plan-only"

assert_not_contains \
  "$PLAN_X86" \
  'invoke target sequentially on both nodes|prewarm.*x86 and ARM|eligible samples per architecture|on both nodes' \
  "x86: vecchia semantica dual-architecture assente"

# ============================================================
# ARM bootstrap
# ============================================================

echo
echo \
  "============================================================"

echo \
  "TEST: catalog arm64 -> bootstrap solo ARM"

echo \
  "============================================================"

run_plan \
  "$CATALOG_ARM" \
  "test-single-arm64" \
  "$PLAN_ARM"

assert_contains \
  "$PLAN_ARM" \
  '^bootstrap_samples:[[:space:]]+10$' \
  "ARM: numero bootstrap sample corretto"

assert_contains \
  "$PLAN_ARM" \
  '^bootstrap_arch:[[:space:]]+ARM$' \
  "ARM: bootstrap_arch corretto"

assert_contains \
  "$PLAN_ARM" \
  '^bootstrap_machine_tag:[[:space:]]+arm64$' \
  "ARM: catalog machine tag seleziona arm64"

assert_contains \
  "$PLAN_ARM" \
  '^bootstrap_node:[[:space:]]+arm-node$' \
  "ARM: nodo bootstrap corretto"

assert_contains \
  "$PLAN_ARM" \
  '^bootstrap_api:[[:space:]]+http://192\.0\.2\.20:1323$' \
  "ARM: API bootstrap corretta"

assert_contains \
  "$PLAN_ARM" \
  'invoke the target 10 times on that node' \
  "ARM: le N probe sono eseguite su un solo nodo"

assert_contains \
  "$PLAN_ARM" \
  '\[PLAN ONLY\]' \
  "ARM: nessuna modifica runtime in plan-only"

assert_not_contains \
  "$PLAN_ARM" \
  'invoke target sequentially on both nodes|prewarm.*x86 and ARM|eligible samples per architecture|on both nodes' \
  "ARM: vecchia semantica dual-architecture assente"

# ============================================================
# Cross-check
# ============================================================

echo
echo \
  "============================================================"

echo \
  "CROSS-CHECK"

echo \
  "============================================================"

assert_not_contains \
  "$PLAN_X86" \
  '^bootstrap_machine_tag:[[:space:]]+arm64$' \
  "catalog x86 non seleziona arm64"

assert_not_contains \
  "$PLAN_X86" \
  '^bootstrap_node:[[:space:]]+arm-node$' \
  "catalog x86 non seleziona arm-node"

assert_not_contains \
  "$PLAN_ARM" \
  '^bootstrap_machine_tag:[[:space:]]+x86$' \
  "catalog ARM non seleziona x86"

assert_not_contains \
  "$PLAN_ARM" \
  '^bootstrap_node:[[:space:]]+x86-node$' \
  "catalog ARM non seleziona x86-node"

echo
echo \
  "============================================================"

echo \
  "TRANSFER SINGLE-ARCH BOOTSTRAP TEST: PASS"

echo \
  "============================================================"

echo
echo \
  "Validated workflow:"

echo

echo \
  "  catalog profile_machine_tag=x86"

echo \
  "      -> N profiling probes only on x86-node"

echo

echo \
  "  catalog profile_machine_tag=arm64"

echo \
  "      -> N profiling probes only on arm-node"

echo

echo \
  "  no MAB/runtime activity is performed by --plan-only"