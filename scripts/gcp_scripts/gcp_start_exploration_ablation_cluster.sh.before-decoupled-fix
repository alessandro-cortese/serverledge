#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"

SOURCE_SCRIPT="${SCRIPT_DIR}/gcp_start_transfer_cluster.sh"

TMP_SCRIPT="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-start-ablation.XXXXXX.sh"
)"

cleanup() {
    rm -f "$TMP_SCRIPT"
}

trap cleanup EXIT

cp "$SOURCE_SCRIPT" "$TMP_SCRIPT"

python3 - \
    "$TMP_SCRIPT" \
    "$ABLATION_RUNTIME_COMMIT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
commit = sys.argv[2]

text = path.read_text(encoding="utf-8")

old = (
    'EXPECTED_COMMIT='
    '"08600a177ce1d88c048251a4b6d52d31fe65a536"'
)

new = (
    f'EXPECTED_COMMIT="{commit}"'
)

count = text.count(old)

if count != 1:
    raise SystemExit(
        f"EXPECTED_COMMIT replacement count={count}, atteso 1"
    )

text = text.replace(
    old,
    new,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)
PY

chmod +x "$TMP_SCRIPT"

# L'ablation usa sempre UCB1 classico con transfer API attiva.
# Varia soltanto mab.ucb1.c.
PROFILING="${PROFILING:-0}" \
MAB_UCB1_C="${MAB_UCB1_C:-0.8}" \
"$TMP_SCRIPT" coupled
