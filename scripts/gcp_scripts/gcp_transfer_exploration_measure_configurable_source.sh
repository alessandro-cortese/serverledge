#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_MEASURE="${SCRIPT_DIR}/gcp_transfer_exploration_measure.sh"

EXPECTED_SOURCE_C="${EXPECTED_SOURCE_C:-0.8}"
PATCH_ONLY="${PATCH_ONLY:-0}"

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ -f "$SOURCE_MEASURE" ]] \
    || fail "measurement script sorgente mancante: $SOURCE_MEASURE"

python3 - "$EXPECTED_SOURCE_C" <<'PY'
import math
import sys

value = float(sys.argv[1])

if not math.isfinite(value):
    raise SystemExit("FATAL: EXPECTED_SOURCE_C non finito")

if value < 0:
    raise SystemExit("FATAL: EXPECTED_SOURCE_C negativo")

print(f"PASS — expected source c valido: {value}")
PY

TMP_MEASURE="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-measure-source-c.XXXXXX.sh"
)"

cleanup() {
    rm -f -- "$TMP_MEASURE"
}

trap cleanup EXIT

cp \
    "$SOURCE_MEASURE" \
    "$TMP_MEASURE"

python3 - \
    "$TMP_MEASURE" \
    "$EXPECTED_SOURCE_C" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_source_c = float(sys.argv[2])

text = path.read_text(encoding="utf-8")

error_literal = '"manifest source_c != 0.8"'

if text.count(error_literal) != 1:
    raise SystemExit(
        "FATAL: controllo source_c inatteso: "
        f"{text.count(error_literal)} occorrenze del messaggio"
    )

error_pos = text.index(error_literal)

# Cerchiamo soltanto nel blocco immediatamente precedente al messaggio
# di errore. In questo modo NON tocchiamo TARGET_C=0.8 né gli altri
# controlli dell'ablation.
window_start = max(0, error_pos - 1000)

window = text[window_start:error_pos]

numeric_pos = window.rfind("0.8")

if numeric_pos < 0:
    raise SystemExit(
        "FATAL: valore numerico source_c=0.8 "
        "non trovato nel preflight"
    )

numeric_pos += window_start

replacement = format(expected_source_c, "g")

text = (
    text[:numeric_pos]
    + replacement
    + text[numeric_pos + len("0.8"):]
)

text = text.replace(
    error_literal,
    f'"manifest source_c != {replacement}"',
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "PASS — temporary measurement preflight "
    f"expects source_c={replacement}"
)
PY

chmod +x "$TMP_MEASURE"

bash -n "$TMP_MEASURE"

echo "PASS — generated configurable measurement runner: bash -n"

if [[ "$PATCH_ONLY" == "1" ]]; then
    echo "PASS — PATCH_ONLY, nessuna VM contattata"
    exit 0
fi

EXPECTED_SOURCE_C="$EXPECTED_SOURCE_C" \
"$TMP_MEASURE" "$@"