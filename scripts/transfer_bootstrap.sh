#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-analysis/bin/python}"
CLI_BIN="${CLI_BIN:-bin/serverledge-cli}"
PROFILING_BIN="${PROFILING_BIN:-bin/serverledge-profiling}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
PREWARM_SETTLE_SECONDS="${PREWARM_SETTLE_SECONDS:-1}"

FUNCTION_NAME=""
X86_NODE_NAME=""
X86_HOST=""
X86_PORT=""
X86_TAG="x86"

ARM_NODE_NAME=""
ARM_HOST=""
ARM_PORT=""
ARM_TAG="arm64"

COLLECTION_MANIFEST=""
CATALOG=""
MODEL=""
LB_URL=""
MAX_DISTANCE=""

SAMPLES_PER_ARCH=10
EXPERIMENT_ID=""

PRIOR_WEIGHT="0.5"
MIN_DONOR_OBSERVATIONS=2

PARAMS_FILE=""
CLUSTER_LABEL=""
REQUIRE_SAME_CLUSTER=0
PLAN_ONLY=0

PARAMS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/transfer_bootstrap.sh [options]

Required:
  --function <name>

  --x86-node-name <registry.node.id>
  --x86-host <host>
  --x86-port <port>

  --arm-node-name <registry.node.id>
  --arm-host <host>
  --arm-port <port>

  --manifest <profiling collection manifest>
  --catalog <transfer donor catalog JSON>
  --model <preprocessing model JSON>

  --lb-url <http://host:port>
  --max-distance <float>

Optional:
  --x86-tag <tag>
      Default: x86

  --arm-tag <tag>
      Default: arm64

  --samples-per-arch <N>
      Default: 2

  --experiment <id>
      Default: timestamp

  --prior-weight <0,1]
      Default: 0.5

  --min-donor-observations <N>
      Default: 2

  --params-file <JSON>

  --param <name:value>
      May be repeated.
      Values are interpreted as strings.

  --cluster-label <N>

  --require-same-cluster

  --plan-only
      Validate configuration and print the workflow
      without executing invocations or changing runtime state.

  -h, --help

Prerequisites:

  * the target function already exists and supports both
    profiling nodes;

  * both nodes have:

        profiling.enabled=true
        profiling.export.enabled=true

  * the manifest contains the two profiling nodes and
    their profiling-samples.jsonl paths;

  * the load balancer runs with:

        lb.mode=MAB
        mab.transfer.control.enabled=true

  * if similarity selects a donor, that donor must already
    exist in the live BanditManager with enough real
    observations to build a transferable prior.

Bootstrap invocations are sent directly to the selected
nodes with:

    CanDoOffloading=false

They therefore do not pass through the load balancer and do
not create/update the target MAB before transfer
initialization.
USAGE
}

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

trim_slash() {
  local value="$1"

  while [[ "$value" == */ ]]; do
    value="${value%/}"
  done

  printf '%s' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in

  --function)
    FUNCTION_NAME="${2:-}"
    shift 2
    ;;

  --x86-node-name)
    X86_NODE_NAME="${2:-}"
    shift 2
    ;;

  --x86-host)
    X86_HOST="${2:-}"
    shift 2
    ;;

  --x86-port)
    X86_PORT="${2:-}"
    shift 2
    ;;

  --x86-tag)
    X86_TAG="${2:-}"
    shift 2
    ;;

  --arm-node-name)
    ARM_NODE_NAME="${2:-}"
    shift 2
    ;;

  --arm-host)
    ARM_HOST="${2:-}"
    shift 2
    ;;

  --arm-port)
    ARM_PORT="${2:-}"
    shift 2
    ;;

  --arm-tag)
    ARM_TAG="${2:-}"
    shift 2
    ;;

  --manifest)
    COLLECTION_MANIFEST="${2:-}"
    shift 2
    ;;

  --catalog)
    CATALOG="${2:-}"
    shift 2
    ;;

  --model)
    MODEL="${2:-}"
    shift 2
    ;;

  --lb-url)
    LB_URL="${2:-}"
    shift 2
    ;;

  --max-distance)
    MAX_DISTANCE="${2:-}"
    shift 2
    ;;

  --samples-per-arch)
    SAMPLES_PER_ARCH="${2:-}"
    shift 2
    ;;

  --experiment)
    EXPERIMENT_ID="${2:-}"
    shift 2
    ;;

  --prior-weight)
    PRIOR_WEIGHT="${2:-}"
    shift 2
    ;;

  --min-donor-observations)
    MIN_DONOR_OBSERVATIONS="${2:-}"
    shift 2
    ;;

  --params-file)
    PARAMS_FILE="${2:-}"
    shift 2
    ;;

  --param)
    PARAMS+=("${2:-}")
    shift 2
    ;;

  --cluster-label)
    CLUSTER_LABEL="${2:-}"
    shift 2
    ;;

  --require-same-cluster)
    REQUIRE_SAME_CLUSTER=1
    shift
    ;;

  --plan-only)
    PLAN_ONLY=1
    shift
    ;;

  -h | --help)
    usage
    exit 0
    ;;

  *)
    fail "Argomento sconosciuto: $1"
    ;;

  esac
done

[[ -n "$FUNCTION_NAME" ]] ||
  fail "Specificare --function."

[[ -n "$X86_NODE_NAME" ]] ||
  fail "Specificare --x86-node-name."

[[ -n "$X86_HOST" ]] ||
  fail "Specificare --x86-host."

[[ -n "$X86_PORT" ]] ||
  fail "Specificare --x86-port."

[[ -n "$ARM_NODE_NAME" ]] ||
  fail "Specificare --arm-node-name."

[[ -n "$ARM_HOST" ]] ||
  fail "Specificare --arm-host."

[[ -n "$ARM_PORT" ]] ||
  fail "Specificare --arm-port."

[[ -n "$COLLECTION_MANIFEST" ]] ||
  fail "Specificare --manifest."

[[ -n "$CATALOG" ]] ||
  fail "Specificare --catalog."

[[ -n "$MODEL" ]] ||
  fail "Specificare --model."

[[ -n "$LB_URL" ]] ||
  fail "Specificare --lb-url."

[[ -n "$MAX_DISTANCE" ]] ||
  fail "Specificare --max-distance."

[[ "$X86_NODE_NAME" != "$ARM_NODE_NAME" ]] ||
  fail "I nodi x86 e ARM devono essere distinti."

[[ "$X86_TAG" != "$ARM_TAG" ]] ||
  fail "I machine tag x86 e ARM devono essere distinti."

[[ "$X86_PORT" =~ ^[0-9]+$ ]] ||
  fail "--x86-port deve essere intero."

[[ "$ARM_PORT" =~ ^[0-9]+$ ]] ||
  fail "--arm-port deve essere intero."

[[ "$SAMPLES_PER_ARCH" =~ ^[1-9][0-9]*$ ]] ||
  fail "--samples-per-arch deve essere positivo."

if (( SAMPLES_PER_ARCH < 10 || SAMPLES_PER_ARCH > 20 )); then
  fail "--samples-per-arch deve essere tra 10 e 20 con l'aggregatore FunctionProfile attuale."
fi

[[ "$MIN_DONOR_OBSERVATIONS" =~ ^[1-9][0-9]*$ ]] ||
  fail "--min-donor-observations deve essere positivo."

if [[ -n "$PARAMS_FILE" &&
      ${#PARAMS[@]} -gt 0 ]]; then

  fail "Usare --params-file oppure --param, non entrambi."
fi

[[ -f "$COLLECTION_MANIFEST" ]] ||
  fail "Manifest non trovato: $COLLECTION_MANIFEST"

[[ -f "$CATALOG" ]] ||
  fail "Donor catalog non trovato: $CATALOG"

[[ -f "$MODEL" ]] ||
  fail "Preprocessing model non trovato: $MODEL"

if [[ -n "$PARAMS_FILE" ]]; then
  [[ -f "$PARAMS_FILE" ]] ||
    fail "Params file non trovato: $PARAMS_FILE"
fi

if [[ -z "$EXPERIMENT_ID" ]]; then
  EXPERIMENT_ID="transfer-bootstrap-$(date +%Y%m%d_%H%M%S)"
fi

[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "Experiment ID non valido."

LB_URL="$(trim_slash "$LB_URL")"

"$PYTHON_BIN" \
  - \
  "$MAX_DISTANCE" \
  "$PRIOR_WEIGHT" \
  "$CLUSTER_LABEL" <<'PY_VALIDATE'

import math
import sys

max_distance = float(
    sys.argv[1]
)

prior_weight = float(
    sys.argv[2]
)

cluster_raw = sys.argv[3]

if (
        not math.isfinite(
            max_distance
        )
        or max_distance <= 0
):
    raise SystemExit(
        "--max-distance deve essere finito e positivo"
    )

if (
        not math.isfinite(
            prior_weight
        )
        or not (
            0.0
            < prior_weight
            <= 1.0
        )
):
    raise SystemExit(
        "--prior-weight deve essere in (0, 1]"
    )

if (
        cluster_raw
        and int(
            cluster_raw
        ) < 0
):
    raise SystemExit(
        "--cluster-label non può essere negativo"
    )

PY_VALIDATE

MANIFEST_METADATA="$(
  "$PYTHON_BIN" \
    - \
    "$COLLECTION_MANIFEST" \
    "$X86_NODE_NAME" \
    "$X86_TAG" \
    "$ARM_NODE_NAME" \
    "$ARM_TAG" <<'PY_MANIFEST'

import re
import sys

from pathlib import Path


path = Path(
    sys.argv[1]
)

expected = {
    sys.argv[2]: (
        "x86",
        sys.argv[3],
    ),

    sys.argv[4]: (
        "arm",
        sys.argv[5],
    ),
}

found = {}

safe_path = re.compile(
    r"^/[A-Za-z0-9._/@+\-]+$"
)

for (
        line_number,
        raw,
) in enumerate(
    path
    .read_text(
        encoding="utf-8"
    )
    .splitlines(),
    1,
):

    line = raw.strip()

    if (
            not line
            or line.startswith(
                "#"
            )
    ):
        continue

    parts = [
        part.strip()

        for part
        in line.split(
            "|"
        )
    ]

    if len(
        parts
    ) != 5:

        raise SystemExit(
            "manifest riga "
            f"{line_number}: "
            "formato non valido"
        )

    (
        node_name,
        machine_tag,
        ssh_target,
        remote_path,
        ssh_port,
    ) = parts

    if node_name not in expected:
        continue

    (
        label,
        expected_tag,
    ) = expected[
        node_name
    ]

    if (
            machine_tag
            != expected_tag
    ):

        raise SystemExit(
            f"manifest: {node_name} "
            f"ha machine_tag="
            f"{machine_tag!r}, "
            f"atteso "
            f"{expected_tag!r}"
        )

    if not safe_path.fullmatch(
        remote_path
    ):
        raise SystemExit(
            "manifest: remote_path "
            "non supportato dal bootstrap "
            f"per {node_name}: "
            f"{remote_path!r}"
        )

    if (
            ssh_port != "-"
            and not ssh_port.isdigit()
    ):

        raise SystemExit(
            "manifest: porta SSH "
            f"non valida per {node_name}"
        )

    found[
        node_name
    ] = (
        label,
        ssh_target,
        remote_path,
        ssh_port,
    )

missing = [
    name

    for name
    in expected

    if name not in found
]

if missing:
    raise SystemExit(
        "manifest: nodi richiesti "
        "mancanti: "
        + ", ".join(
            missing
        )
    )

for node_name in (
        sys.argv[2],
        sys.argv[4],
):

    print(
        "\t".join(
            found[
                node_name
            ]
        )
    )

PY_MANIFEST
)"

while IFS=$'\t' read -r \
  label \
  ssh_target \
  remote_path \
  ssh_port; do

  case "$label" in

  x86)
    X86_SSH_TARGET="$ssh_target"
    X86_REMOTE_PATH="$remote_path"
    X86_SSH_PORT="$ssh_port"
    ;;

  arm)
    ARM_SSH_TARGET="$ssh_target"
    ARM_REMOTE_PATH="$remote_path"
    ARM_SSH_PORT="$ssh_port"
    ;;

  *)
    fail \
      "Metadata manifest inattesi: $label"
    ;;

  esac

done <<< "$MANIFEST_METADATA"

read -r \
  CATALOG_AGGREGATION \
  CATALOG_MACHINE_TAG < <(
    "$PYTHON_BIN" \
      - \
      "$CATALOG" <<'PY_CATALOG'

import json
import sys

from pathlib import Path


document = json.loads(
    Path(
        sys.argv[1]
    ).read_text(
        encoding="utf-8"
    )
)

clustering = (
        document.get(
            "clustering"
        )
        or {}
)

aggregation = str(
    clustering.get(
        "aggregation",
        "",
    )
).strip()

machine_tag = str(
    clustering.get(
        "profile_machine_tag",
        "",
    )
).strip()

if aggregation not in {
        "mean",
        "median",
}:

    raise SystemExit(
        "donor catalog: aggregation "
        "deve essere mean o median"
    )

if not machine_tag:
    raise SystemExit(
        "donor catalog: "
        "profile_machine_tag mancante"
    )

print(
    aggregation,
    machine_tag,
)

PY_CATALOG
)

if [[ "$CATALOG_MACHINE_TAG" != "$X86_TAG" &&
      "$CATALOG_MACHINE_TAG" != "$ARM_TAG" ]]; then

  fail \
    "Il profile_machine_tag del catalogo ($CATALOG_MACHINE_TAG) non coincide con $X86_TAG o $ARM_TAG."
fi

# The transfer bootstrap must profile the new function on one predetermined
# architecture only.
#
# The donor catalog already defines the reference architecture through
# profile_machine_tag. Using the same machine tag for the target profiling
# guarantees that donor and target feature vectors belong to the same
# profiling space.
case "$CATALOG_MACHINE_TAG" in

"$X86_TAG")
  BOOTSTRAP_LABEL="x86"
  BOOTSTRAP_NODE_NAME="$X86_NODE_NAME"
  BOOTSTRAP_HOST="$X86_HOST"
  BOOTSTRAP_PORT="$X86_PORT"
  BOOTSTRAP_TAG="$X86_TAG"

  BOOTSTRAP_SSH_TARGET="$X86_SSH_TARGET"
  BOOTSTRAP_REMOTE_PATH="$X86_REMOTE_PATH"
  BOOTSTRAP_SSH_PORT="$X86_SSH_PORT"
  ;;

"$ARM_TAG")
  BOOTSTRAP_LABEL="ARM"
  BOOTSTRAP_NODE_NAME="$ARM_NODE_NAME"
  BOOTSTRAP_HOST="$ARM_HOST"
  BOOTSTRAP_PORT="$ARM_PORT"
  BOOTSTRAP_TAG="$ARM_TAG"

  BOOTSTRAP_SSH_TARGET="$ARM_SSH_TARGET"
  BOOTSTRAP_REMOTE_PATH="$ARM_REMOTE_PATH"
  BOOTSTRAP_SSH_PORT="$ARM_SSH_PORT"
  ;;

*)
  fail \
    "Machine tag bootstrap non supportato: $CATALOG_MACHINE_TAG"
  ;;

esac

[[ "$BOOTSTRAP_TAG" == "$CATALOG_MACHINE_TAG" ]] ||
  fail \
    "Bootstrap tag incoerente: catalog=$CATALOG_MACHINE_TAG selected=$BOOTSTRAP_TAG"

OUTPUT_DIR="data/profiling/transfer-bootstrap/${EXPERIMENT_ID}"

COLLECTION_ROOT="data/profiling/raw"

COLLECTION_DIR="${COLLECTION_ROOT}/${EXPERIMENT_ID}"

FILTERED_DIR="${OUTPUT_DIR}/filtered-raw"

FUNCTION_PROFILES="${OUTPUT_DIR}/function-profiles.jsonl"

MEAN_CSV="${OUTPUT_DIR}/function-profiles-mean.csv"

MEDIAN_CSV="${OUTPUT_DIR}/function-profiles-median.csv"

QUERY_JSON="${OUTPUT_DIR}/transfer-query.json"

SELECTION_JSON="${OUTPUT_DIR}/selection.json"

SELECTION_CSV="${OUTPUT_DIR}/selection.csv"

CONTROL_REQUEST="${OUTPUT_DIR}/transfer-control-request.json"

CONTROL_RESPONSE="${OUTPUT_DIR}/transfer-control-response.json"

INVOCATION_BODY="${OUTPUT_DIR}/invocation-request.json"

QUERY_ID="query-${EXPERIMENT_ID}"

SELECTION_RUN_ID="selection-${EXPERIMENT_ID}"

cat <<PLAN

============================================================
10C.3B.2 — EXPERIMENTAL TRANSFER BOOTSTRAP
============================================================

experiment_id:         $EXPERIMENT_ID
function:              $FUNCTION_NAME

bootstrap_samples:     $SAMPLES_PER_ARCH
bootstrap_arch:        $BOOTSTRAP_LABEL
bootstrap_machine_tag: $BOOTSTRAP_TAG
bootstrap_node:        $BOOTSTRAP_NODE_NAME
bootstrap_api:         http://$BOOTSTRAP_HOST:$BOOTSTRAP_PORT

x86:
  node:               $X86_NODE_NAME
  tag:                $X86_TAG
  API:                http://$X86_HOST:$X86_PORT

ARM:
  node:               $ARM_NODE_NAME
  tag:                $ARM_TAG
  API:                http://$ARM_HOST:$ARM_PORT

catalog aggregation:  $CATALOG_AGGREGATION
catalog machine tag:  $CATALOG_MACHINE_TAG

max distance:         $MAX_DISTANCE

prior weight:         $PRIOR_WEIGHT
min donor obs/arm:    $MIN_DONOR_OBSERVATIONS

LB control URL:
  $LB_URL/mab/transfer/initialize

output:
  $OUTPUT_DIR

Workflow:

  1. select the predetermined bootstrap architecture from
   donor catalog profile_machine_tag

  2. record the current profiling dataset line count only
   for the selected bootstrap node

  3. prewarm one target container directly on that node

  4. invoke the target $SAMPLES_PER_ARCH times on that node with:
     CanDoOffloading=false

  5. collect profiling JSONL datasets

  6. isolate only samples appended during this bootstrap
   for the target function and selected machine tag

  7. require >= $SAMPLES_PER_ARCH warm/exclusive
   eligible target samples

  8. aggregate FunctionProfile and export mean/median CSV

  9. build transfer-query.json with donor preprocessing model

  10. run similarity selection

  11. POST selection.json to the live LB transfer-control API

  12. subsequent normal requests go through UCB1/LinUCB

PLAN

if [[ "$PLAN_ONLY" -eq 1 ]]; then

  echo
  echo \
    "[PLAN ONLY] Nessuna invocazione, raccolta o modifica runtime eseguita."

  exit 0
fi

for command in \
  curl \
  ssh \
  "$PYTHON_BIN" \
  "$CLI_BIN" \
  "$PROFILING_BIN"; do

  if [[ "$command" == */* ]]; then

    [[ -x "$command" ]] ||
      fail \
        "Eseguibile non trovato: $command"

  else

    command -v \
      "$command" \
      >/dev/null \
      2>&1 ||
      fail \
        "Comando non trovato: $command"

  fi

done

[[ -f scripts/collect-profiling-datasets.sh ]] ||
  fail \
    "scripts/collect-profiling-datasets.sh non trovato."

[[ -f analysis/profiling/transfer_query.py ]] ||
  fail \
    "analysis/profiling/transfer_query.py non trovato."

[[ -f analysis/profiling/similarity_selection.py ]] ||
  fail \
    "analysis/profiling/similarity_selection.py non trovato."

[[ ! -e "$OUTPUT_DIR" ]] ||
  fail \
    "Output già esistente: $OUTPUT_DIR"

[[ ! -e "$COLLECTION_DIR" ]] ||
  fail \
    "Collection già esistente: $COLLECTION_DIR"

mkdir -p \
  "$OUTPUT_DIR"

"$PYTHON_BIN" \
  - \
  "$PARAMS_FILE" \
  "${PARAMS[@]}" \
  > "$INVOCATION_BODY" <<'PY_BODY'

import json
import sys


params_file = sys.argv[1]

raw_params = sys.argv[
    2:
]

params = {}

if params_file:

    with open(
        params_file,
        "r",
        encoding="utf-8",
    ) as handle:

        params = json.load(
            handle
        )

    if not isinstance(
        params,
        dict,
    ):
        raise SystemExit(
            "--params-file deve "
            "contenere un oggetto JSON"
        )

else:

    for raw in raw_params:

        if ":" not in raw:

            raise SystemExit(
                "parametro non valido: "
                f"{raw!r}; usare name:value"
            )

        (
            name,
            value,
        ) = raw.split(
            ":",
            1,
        )

        name = name.strip()

        if not name:

            raise SystemExit(
                "nome parametro vuoto: "
                f"{raw!r}"
            )

        params[
            name
        ] = value


json.dump(
    {
        "Params":
            params,

        "QoSClass":
            0,

        "QoSMaxRespT":
            -1.0,

        "CanDoOffloading":
            False,

        "Async":
            False,

        "ReturnOutput":
            True,
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)

print()

PY_BODY

check_node() {
  local label="$1"
  local host="$2"
  local port="$3"

  curl \
    -fsS \
    --max-time 3 \
    -o /dev/null \
    "http://${host}:${port}/status" ||
    fail \
      "$label non raggiungibile su http://${host}:${port}/status"
}

prewarm_node() {
  local label="$1"
  local host="$2"
  local port="$3"
  local log_file="$4"

  echo \
    "[prewarm] $label"

  "$CLI_BIN" prewarm \
    -f "$FUNCTION_NAME" \
    -c 1 \
    -H "$host" \
    -P "$port" |
    tee \
      "$log_file"
}

invoke_node() {
  local label="$1"
  local host="$2"
  local port="$3"
  local index="$4"
  local log_file="$5"

  echo \
    "[invoke] $label sample $index/$SAMPLES_PER_ARCH"

  curl \
    -fsS \
    --max-time 300 \
    -H \
      'Content-Type: application/json' \
    --data-binary \
      "@$INVOCATION_BODY" \
    "http://${host}:${port}/invoke/${FUNCTION_NAME}" \
    > "$log_file"
}

remote_line_count() {
  local target="$1"
  local path="$2"
  local port="$3"

  local -a ssh_args=(
    -o
    "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  )

  if [[ "$port" != "-" ]]; then

    ssh_args+=(
      -p
      "$port"
    )

  fi

  ssh \
    "${ssh_args[@]}" \
    "$target" \
    "if [ -f '$path' ]; then wc -l < '$path'; else echo 0; fi"
}

check_node \
  "$BOOTSTRAP_LABEL" \
  "$BOOTSTRAP_HOST" \
  "$BOOTSTRAP_PORT"

BOOTSTRAP_BASELINE_LINES="$(
  remote_line_count \
    "$BOOTSTRAP_SSH_TARGET" \
    "$BOOTSTRAP_REMOTE_PATH" \
    "$BOOTSTRAP_SSH_PORT"
)"

[[ "$BOOTSTRAP_BASELINE_LINES" =~ ^[0-9]+$ ]] ||
  fail \
    "Baseline profiling non valida per $BOOTSTRAP_NODE_NAME: $BOOTSTRAP_BASELINE_LINES"

echo \
  "[bootstrap] node=$BOOTSTRAP_NODE_NAME tag=$BOOTSTRAP_TAG raw_baseline_lines=$BOOTSTRAP_BASELINE_LINES"

prewarm_node \
  "$BOOTSTRAP_LABEL" \
  "$BOOTSTRAP_HOST" \
  "$BOOTSTRAP_PORT" \
  "$OUTPUT_DIR/prewarm-bootstrap.json"

sleep \
  "$PREWARM_SETTLE_SECONDS"

for index in $(
  seq \
    1 \
    "$SAMPLES_PER_ARCH"
); do

  invoke_node \
    "$BOOTSTRAP_LABEL" \
    "$BOOTSTRAP_HOST" \
    "$BOOTSTRAP_PORT" \
    "$index" \
    "$OUTPUT_DIR/invoke-bootstrap-${index}.json"

done

echo \
  "[collect] Raccolta dataset dai nodi..."

bash scripts/collect-profiling-datasets.sh \
  --manifest \
    "$COLLECTION_MANIFEST" \
  --experiment \
    "$EXPERIMENT_ID" \
  --output-root \
    "$COLLECTION_ROOT"

mkdir -p \
  "$FILTERED_DIR"

"$PYTHON_BIN" \
  - \
  "$COLLECTION_DIR" \
  "$FILTERED_DIR" \
  "$FUNCTION_NAME" \
  "$BOOTSTRAP_NODE_NAME" \
  "$BOOTSTRAP_TAG" \
  "$BOOTSTRAP_BASELINE_LINES" \
  "$SAMPLES_PER_ARCH" <<'PY_FILTER'

import json
import sys

from pathlib import Path


(
    collection_raw,
    filtered_raw,
    function_name,

    bootstrap_node,
    bootstrap_tag,
    bootstrap_baseline_raw,

    required_raw,
) = sys.argv[1:]

collection = Path(
    collection_raw
)

filtered = Path(
    filtered_raw
)

required = int(
    required_raw
)

expected = {
    bootstrap_node: (
        bootstrap_tag,
        int(
            bootstrap_baseline_raw
        ),
    ),
}


for (
        node_name,
        (
            machine_tag,
            baseline,
        ),
) in expected.items():

    source = (
            collection
            / node_name
            / "profiling-samples.jsonl"
    )

    if not source.is_file():

        raise SystemExit(
            "dataset raccolto "
            f"mancante per {node_name}: "
            f"{source}"
        )

    lines = (
        source
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    if len(
        lines
    ) < baseline:

        raise SystemExit(
            f"dataset {node_name} "
            "è più corto della baseline: "
            f"{len(lines)} < {baseline}"
        )

    selected = []

    warm = 0
    cold = 0
    eligible = 0

    for (
            line_number,
            raw,
    ) in enumerate(
        lines[
            baseline:
        ],
        baseline + 1,
    ):

        line = raw.strip()

        if not line:
            continue

        try:
            sample = json.loads(
                line
            )

        except json.JSONDecodeError as exc:

            raise SystemExit(
                f"{source}:"
                f"{line_number}: "
                f"JSON non valido: "
                f"{exc}"
            )

        if (
                sample.get(
                    "function_name"
                )
                != function_name
        ):
            continue

        if (
                str(
                    sample.get(
                        "node_name",
                        "",
                    )
                )
                != node_name
        ):
            continue

        if (
                str(
                    sample.get(
                        "machine_tag",
                        "",
                    )
                )
                != machine_tag
        ):

            raise SystemExit(
                "sample "
                f"{sample.get('request_id')}: "
                "machine_tag inatteso "
                f"per {node_name}"
            )

        selected.append(
            sample
        )

        if bool(
            sample.get(
                "warm_start"
            )
        ):
            warm += 1

        else:
            cold += 1

        if bool(
            (
                sample.get(
                    "eligibility"
                )
                or {}
            ).get(
                "resource_clustering"
            )
        ):
            eligible += 1

    node_dir = (
            filtered
            / node_name
    )

    node_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
            node_dir
            / "profiling-samples.jsonl"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for sample in selected:

            handle.write(
                json.dumps(
                    sample,
                    sort_keys=True,
                )
                + "\n"
            )

    print(
        f"[profile] "
        f"node={node_name} "
        f"tag={machine_tag} "
        f"target_samples={len(selected)} "
        f"warm={warm} "
        f"cold={cold} "
        f"eligible={eligible} "
        f"required={required}"
    )

    if eligible < required:

        raise SystemExit(
            "campioni warm/exclusive "
            f"insufficienti per {node_name}: "
            f"{eligible} < {required}"
        )

PY_FILTER

"$PROFILING_BIN" aggregate \
  --input-dir \
    "$FILTERED_DIR" \
  --samples \
    "$SAMPLES_PER_ARCH" \
  --output \
    "$FUNCTION_PROFILES"

"$PROFILING_BIN" export-csv \
  --input \
    "$FUNCTION_PROFILES" \
  --experiment-id \
    "$EXPERIMENT_ID" \
  --mean-output \
    "$MEAN_CSV" \
  --median-output \
    "$MEDIAN_CSV"

if [[ "$CATALOG_AGGREGATION" == "mean" ]]; then

  QUERY_INPUT_CSV="$MEAN_CSV"

else

  QUERY_INPUT_CSV="$MEDIAN_CSV"

fi

QUERY_ARGS=(
  "$PYTHON_BIN"
  analysis/profiling/transfer_query.py

  --input
  "$QUERY_INPUT_CSV"

  --catalog
  "$CATALOG"

  --model
  "$MODEL"

  --function
  "$FUNCTION_NAME"

  --query-id
  "$QUERY_ID"

  --output
  "$QUERY_JSON"
)

if [[ -n "$CLUSTER_LABEL" ]]; then

  QUERY_ARGS+=(
    --cluster-label
    "$CLUSTER_LABEL"
  )

fi

"${QUERY_ARGS[@]}"

SELECTION_ARGS=(
  "$PYTHON_BIN"
  analysis/profiling/similarity_selection.py

  --catalog
  "$CATALOG"

  --query
  "$QUERY_JSON"

  --run-id
  "$SELECTION_RUN_ID"

  --max-distance
  "$MAX_DISTANCE"

  --output-json
  "$SELECTION_JSON"

  --output-csv
  "$SELECTION_CSV"
)

if [[ "$REQUIRE_SAME_CLUSTER" -eq 1 ]]; then

  SELECTION_ARGS+=(
    --require-same-cluster
  )

fi

"${SELECTION_ARGS[@]}"

"$PYTHON_BIN" \
  - \
  "$SELECTION_JSON" \
  "$FUNCTION_NAME" \
  "$PRIOR_WEIGHT" \
  "$MIN_DONOR_OBSERVATIONS" \
  > "$CONTROL_REQUEST" <<'PY_CONTROL'

import json
import sys

from pathlib import Path


selection = json.loads(
    Path(
        sys.argv[1]
    ).read_text(
        encoding="utf-8"
    )
)

json.dump(
    {
        "target_function_name":
            sys.argv[2],

        "selection_artifact":
            selection,

        "prior_config": {
            "equivalent_observation_weight":
                float(
                    sys.argv[3]
                ),

            "min_real_observations_per_arm":
                int(
                    sys.argv[4]
                ),
        },
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)

print()

PY_CONTROL

echo \
  "[transfer] Inizializzazione MAB target nel load balancer live..."

HTTP_CODE="$(
  curl \
    -sS \
    --max-time 30 \
    -o "$CONTROL_RESPONSE" \
    -w '%{http_code}' \
    -H \
      'Content-Type: application/json' \
    --data-binary \
      "@$CONTROL_REQUEST" \
    "$LB_URL/mab/transfer/initialize"
)"

if [[ "$HTTP_CODE" != "200" ]]; then

  echo \
    "[transfer] HTTP $HTTP_CODE" \
    >&2

  cat \
    "$CONTROL_RESPONSE" \
    >&2 ||
    true

  fail \
    "Il load balancer ha rifiutato l'inizializzazione del target."
fi

"$PYTHON_BIN" \
  - \
  "$SELECTION_JSON" \
  "$CONTROL_RESPONSE" <<'PY_SUMMARY'

import json
import sys

from pathlib import Path


selection = json.loads(
    Path(
        sys.argv[1]
    ).read_text(
        encoding="utf-8"
    )
)

response = json.loads(
    Path(
        sys.argv[2]
    ).read_text(
        encoding="utf-8"
    )
)

selected = (
        selection.get(
            "selected_donor"
        )
        or {}
)

prior = (
        response.get(
            "prior"
        )
        or {}
)

print()

print(
    "============================================================"
)

print(
    "BOOTSTRAP COMPLETATO"
)

print(
    "============================================================"
)

print(
    "selection_status="
    f"{selection.get('status')}"
)

print(
    "selection_reason="
    f"{selection.get('reason') or 'none'}"
)

print(
    "selected_donor="
    f"{selected.get('function_name') or 'none'}"
)

print(
    "transfer_attempted="
    f"{response.get('transfer_attempted')}"
)

print(
    "transfer_applied="
    f"{response.get('transfer_applied')}"
)

print(
    "runtime_reason="
    f"{response.get('runtime_reason')}"
)

print(
    "prior_has_prior="
    f"{prior.get('has_prior', False)}"
)

print()

print(
    "Il target MAB è inizializzato. "
    "Le richieste successive vanno inviate"
)

print(
    "normalmente al load balancer, "
    "che userà UCB1/LinUCB."
)

PY_SUMMARY

echo \
  "artifacts=$OUTPUT_DIR"