#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"

SOURCE_SCRIPT="${SCRIPT_DIR}/gcp_transfer_final.sh"

TMP_SCRIPT="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-ablation-source.XXXXXX.sh"
)"

cleanup() {
    rm -f "$TMP_SCRIPT"
}

trap cleanup EXIT

# Questa è la condizione sorgente del pair:
# UCB1 classico + stesso prior + exploration standard c=.8.
if [[ "${MAB_UCB1_C:-0.8}" != "0.8" ]]; then
    echo "FATAL: source run richiede MAB_UCB1_C=0.8" >&2
    exit 1
fi

if [[ "${EQUIVALENT_PRIOR_WEIGHT:-0.25}" != "0.25" ]]; then
    echo "FATAL: source run richiede EQUIVALENT_PRIOR_WEIGHT=0.25" >&2
    exit 1
fi

if [[ "${MIN_DONOR_OBSERVATIONS:-10}" != "10" ]]; then
    echo "FATAL: source run richiede MIN_DONOR_OBSERVATIONS=10" >&2
    exit 1
fi

cp "$SOURCE_SCRIPT" "$TMP_SCRIPT"

python3 - \
    "$TMP_SCRIPT" \
    "$ABLATION_RUNTIME_COMMIT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
commit = sys.argv[2]

text = path.read_text(encoding="utf-8")

old_commit = (
    'EXPECTED_COMMIT="${EXPECTED_COMMIT:-'
    '08600a177ce1d88c048251a4b6d52d31fe65a536}"'
)

new_commit = (
    'EXPECTED_COMMIT="${EXPECTED_COMMIT:-'
    + commit
    + '}"'
)

if text.count(old_commit) != 1:
    raise SystemExit(
        "impossibile sostituire EXPECTED_COMMIT nel runner"
    )

text = text.replace(
    old_commit,
    new_commit,
    1,
)

old_start = "./gcp_start_transfer_cluster.sh"
new_start = "./gcp_start_exploration_ablation_cluster.sh"

if text.count(old_start) != 1:
    raise SystemExit(
        "numero inatteso di chiamate allo starter originale: "
        + str(text.count(old_start))
    )

text = text.replace(
    old_start,
    new_start,
    1,
)

path.write_text(
    text,
    encoding="utf-8",
)
PY

chmod +x "$TMP_SCRIPT"

MAB_UCB1_C=0.8 \
EQUIVALENT_PRIOR_WEIGHT=0.25 \
MIN_DONOR_OBSERVATIONS=10 \
"$TMP_SCRIPT" coupled
