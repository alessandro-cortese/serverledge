#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_MATERIALIZER="${SCRIPT_DIR}/gcp_transfer_exploration_materialize.sh"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-}"
TARGET_FUNCTION="${TARGET_FUNCTION:-}"
REPLICA_BOOTSTRAP_DIR="${REPLICA_BOOTSTRAP_DIR:-}"
MATERIALIZED_OUTPUT_DIR="${MATERIALIZED_OUTPUT_DIR:-}"

DONOR_SOURCE_C="${DONOR_SOURCE_C:-0.8}"

N_X86="${N_X86:-1}"
N_ARM="${N_ARM:-1}"

MIN_DONOR_OBSERVATIONS="${MIN_DONOR_OBSERVATIONS:-10}"
DONOR_MAX_REQUESTS="${DONOR_MAX_REQUESTS:-200}"

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ -n "$ABLATION_RUNTIME_COMMIT" ]] \
    || fail "ABLATION_RUNTIME_COMMIT mancante"

[[ -n "$TARGET_FUNCTION" ]] \
    || fail "TARGET_FUNCTION mancante"

[[ -d "$REPLICA_BOOTSTRAP_DIR" ]] \
    || fail "REPLICA_BOOTSTRAP_DIR non valido: $REPLICA_BOOTSTRAP_DIR"

[[ -f "$REPLICA_BOOTSTRAP_DIR/bootstrap.json" ]] \
    || fail "bootstrap.json mancante"

[[ -n "$MATERIALIZED_OUTPUT_DIR" ]] \
    || fail "MATERIALIZED_OUTPUT_DIR mancante"

[[ -f "$SOURCE_MATERIALIZER" ]] \
    || fail "materializer sorgente mancante: $SOURCE_MATERIALIZER"

python3 - "$DONOR_SOURCE_C" <<'PY'
import math
import sys

c = float(sys.argv[1])

if not math.isfinite(c):
    raise SystemExit("FATAL: DONOR_SOURCE_C non finito")

if c < 0:
    raise SystemExit("FATAL: DONOR_SOURCE_C negativo")

print(f"PASS — donor source c valido: {c}")
PY

if [[ -e "$MATERIALIZED_OUTPUT_DIR/frozen-prior.json" ]]; then
    fail "esiste già frozen-prior.json in $MATERIALIZED_OUTPUT_DIR"
fi

rm -rf -- "$MATERIALIZED_OUTPUT_DIR"

TMP_BOOTSTRAP="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/serverledge-source-bootstrap.XXXXXX"
)"

TMP_MATERIALIZER="$(
    mktemp \
        "${SCRIPT_DIR}/.gcp-transfer-materialize-configurable.XXXXXX.sh"
)"

cleanup() {
    rm -rf -- "$TMP_BOOTSTRAP"
    rm -f -- "$TMP_MATERIALIZER"
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Copia temporanea del bootstrap.
# L'artefatto congelato originale NON viene modificato.
# ---------------------------------------------------------------------------

cp -a \
    "${REPLICA_BOOTSTRAP_DIR}/." \
    "${TMP_BOOTSTRAP}/"

python3 - \
    "${TMP_BOOTSTRAP}/bootstrap.json" \
    "$DONOR_SOURCE_C" <<'PY'
import json
import math
import sys

path = sys.argv[1]
source_c = float(sys.argv[2])

with open(path, encoding="utf-8") as f:
    doc = json.load(f)

try:
    coupled = (
        doc["experimental_block"]
           ["policies"]
           ["coupled"]
    )
except KeyError as exc:
    raise SystemExit(
        f"FATAL: bootstrap coupled block mancante: {exc}"
    )

original_c = float(coupled["c"])

if not math.isclose(
    original_c,
    0.8,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        "FATAL: bootstrap originale coupled.c="
        f"{original_c}, atteso 0.8"
    )

coupled["c"] = source_c

with open(path, "w", encoding="utf-8") as f:
    json.dump(
        doc,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print(
    "PASS — temporary bootstrap "
    f"coupled.c: {original_c} -> {source_c}"
)
PY

# ---------------------------------------------------------------------------
# Copia temporanea del materializer.
#
# Il materializer originale contiene source_c=0.8 in cinque punti:
# - manifest
# - controllo bootstrap
# - messaggi diagnostici
# - MAB_UCB1_C passato al runner
#
# Modifichiamo SOLO la copia temporanea.
# ---------------------------------------------------------------------------

cp \
    "$SOURCE_MATERIALIZER" \
    "$TMP_MATERIALIZER"

python3 - \
    "$TMP_MATERIALIZER" \
    "$DONOR_SOURCE_C" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source_c = float(sys.argv[2])

text = path.read_text(encoding="utf-8")

count = text.count("0.8")

if count != 5:
    raise SystemExit(
        "FATAL: materializer inatteso: "
        f"trovate {count} occorrenze di 0.8, attese 5"
    )

replacement = format(source_c, "g")

text = text.replace(
    "0.8",
    replacement,
)

path.write_text(
    text,
    encoding="utf-8",
)

print(
    "PASS — temporary materializer "
    f"source_c={replacement}"
)
PY

chmod +x "$TMP_MATERIALIZER"

bash -n "$TMP_MATERIALIZER"

echo "PASS — generated configurable materializer: bash -n"

echo
echo "============================================================"
echo "SOURCE ACQUISITION CONFIGURATION"
echo "============================================================"
echo "target:                $TARGET_FUNCTION"
echo "donor source c:        $DONOR_SOURCE_C"
echo "min observations/arm:  $MIN_DONOR_OBSERVATIONS"
echo "max donor requests:    $DONOR_MAX_REQUESTS"
echo "original bootstrap:    $REPLICA_BOOTSTRAP_DIR"
echo "temporary bootstrap:   $TMP_BOOTSTRAP"
echo

# ---------------------------------------------------------------------------
# Materializzazione vera.
# ---------------------------------------------------------------------------

ABLATION_RUNTIME_COMMIT="$ABLATION_RUNTIME_COMMIT" \
TARGET_FUNCTION="$TARGET_FUNCTION" \
REPLICA_BOOTSTRAP_DIR="$TMP_BOOTSTRAP" \
MATERIALIZED_OUTPUT_DIR="$MATERIALIZED_OUTPUT_DIR" \
N_X86="$N_X86" \
N_ARM="$N_ARM" \
MIN_DONOR_OBSERVATIONS="$MIN_DONOR_OBSERVATIONS" \
DONOR_MAX_REQUESTS="$DONOR_MAX_REQUESTS" \
"$TMP_MATERIALIZER"

# ---------------------------------------------------------------------------
# Ripristiniamo nel bundle il bootstrap CANONICO.
#
# DONOR_SOURCE_C riguarda soltanto l'acquisizione del donor.
# Il bootstrap dell'esperimento target deve continuare a essere quello
# congelato originariamente, con coupled.c = 0.8.
# ---------------------------------------------------------------------------

[[ -f "$MATERIALIZED_OUTPUT_DIR/frozen-prior.json" ]] \
    || fail "materializzazione terminata senza frozen-prior.json"

[[ -f "$MATERIALIZED_OUTPUT_DIR/manifest.json" ]] \
    || fail "materializzazione terminata senza manifest.json"

rm -rf \
    "$MATERIALIZED_OUTPUT_DIR/bootstrap"

mkdir -p \
    "$MATERIALIZED_OUTPUT_DIR/bootstrap"

cp -a \
    "${REPLICA_BOOTSTRAP_DIR}/." \
    "$MATERIALIZED_OUTPUT_DIR/bootstrap/"

python3 - \
    "$MATERIALIZED_OUTPUT_DIR" \
    "$DONOR_SOURCE_C" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
source_c = float(sys.argv[2])

manifest_path = bundle / "manifest.json"
bootstrap_path = bundle / "bootstrap" / "bootstrap.json"
readiness_path = bundle / "donor-readiness.json"
prior_path = bundle / "frozen-prior.json"

for path in (
    manifest_path,
    bootstrap_path,
    readiness_path,
    prior_path,
):
    if not path.is_file():
        raise SystemExit(
            f"FATAL: artefatto mancante: {path}"
        )

with bootstrap_path.open(encoding="utf-8") as f:
    bootstrap = json.load(f)

canonical_c = float(
    bootstrap["experimental_block"]
             ["policies"]
             ["coupled"]
             ["c"]
)

if not math.isclose(
    canonical_c,
    0.8,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        "FATAL: bootstrap canonico coupled.c="
        f"{canonical_c}, atteso 0.8"
    )

with manifest_path.open(encoding="utf-8") as f:
    manifest = json.load(f)

manifest["source_c"] = source_c

bootstrap_sha = hashlib.sha256(
    bootstrap_path.read_bytes()
).hexdigest()

manifest["bootstrap_sha256"] = bootstrap_sha

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

with readiness_path.open(encoding="utf-8") as f:
    readiness = json.load(f)

obs = readiness.get("observations") or {}

amd = int(obs.get("amd64", 0))
arm = int(obs.get("arm64", 0))

minimum = int(
    readiness.get(
        "minimum_observations_per_arm",
        0,
    )
)

if amd < minimum or arm < minimum:
    raise SystemExit(
        "FATAL: donor non ready: "
        f"amd64={amd} arm64={arm} minimum={minimum}"
    )

if manifest.get(
    "target_real_feedback_before_export"
) != 0:
    raise SystemExit(
        "FATAL: target feedback presente "
        "prima dell'export"
    )

print("PASS — canonical bootstrap restored")
print(f"bootstrap.c={canonical_c}")
print(f"source_c={source_c}")
print(
    "donor observations: "
    f"amd64={amd} arm64={arm}"
)
print(
    "target real feedback before export=0"
)
print(
    "bootstrap_sha256="
    + bootstrap_sha
)
PY

(
    cd "$MATERIALIZED_OUTPUT_DIR"
    sha256sum -c prior.sha256
)

echo
echo "============================================================"
echo "CONFIGURABLE MATERIALIZATION: PASS"
echo "============================================================"
echo "bundle: $MATERIALIZED_OUTPUT_DIR"
echo "source_c: $DONOR_SOURCE_C"
echo