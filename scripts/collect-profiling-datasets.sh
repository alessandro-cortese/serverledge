#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

OUTPUT_ROOT="${PROFILING_COLLECTION_ROOT:-data/profiling/raw}"
EXPECTED_SCHEMA_VERSION="${EXPECTED_SCHEMA_VERSION:-3}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"

MANIFEST=""
EXPERIMENT_ID=""
CHECK_MANIFEST=0

STAGING_DIR=""
COLLECTION_COMPLETED=0

usage() {
  cat <<'EOF'
Usage:
  collect-profiling-datasets.sh \
    --manifest <file> \
    [--experiment <id>] \
    [--output-root <directory>]

Options:
  --manifest <file>
      Node collection manifest.

  --experiment <id>
      Identifier of this experimental collection.
      Default: current timestamp.

  --output-root <directory>
      Root directory for collected datasets.
      Default: data/profiling/raw

  --check-manifest
      Validate the manifest without connecting to remote nodes.

  -h, --help
      Show this help.

Manifest format:

  node_name|machine_tag|ssh_target|remote_path|ssh_port

Example:

  arm-node-01|arm64|ale@192.168.1.101|/home/ale/serverledge/data/profiling/profiling-samples.jsonl|22
  x86-node-01|x86|serverledge-x86|/home/ale/serverledge/data/profiling/profiling-samples.jsonl|-

The collection must be performed after the experimental invocations have
finished, so the remote append-only JSONL files are not modified while they
are being copied.
EOF
}

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

trim() {
  local value="$1"

  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"

  printf '%s' "$value"
}

cleanup() {
  local status=$?

  if [[ "$COLLECTION_COMPLETED" -ne 1 ]] &&
    [[ -n "$STAGING_DIR" ]] &&
    [[ -d "$STAGING_DIR" ]]; then

    rm -rf "$STAGING_DIR"
  fi

  exit "$status"
}

trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
  --manifest)
    [[ $# -ge 2 ]] ||
      fail "--manifest richiede un valore."

    MANIFEST="$2"
    shift 2
    ;;

  --experiment)
    [[ $# -ge 2 ]] ||
      fail "--experiment richiede un valore."

    EXPERIMENT_ID="$2"
    shift 2
    ;;

  --output-root)
    [[ $# -ge 2 ]] ||
      fail "--output-root richiede un valore."

    OUTPUT_ROOT="$2"
    shift 2
    ;;

  --check-manifest)
    CHECK_MANIFEST=1
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

[[ -n "$MANIFEST" ]] ||
  fail "Specificare --manifest."

[[ -f "$MANIFEST" ]] ||
  fail "Manifest non trovato: $MANIFEST"

if [[ -z "$EXPERIMENT_ID" ]]; then
  EXPERIMENT_ID="$(date +%Y%m%d_%H%M%S)"
fi

EXPERIMENT_ID="$(trim "$EXPERIMENT_ID")"
OUTPUT_ROOT="$(trim "$OUTPUT_ROOT")"

[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail \
    "Experiment ID non valido: usare solo lettere, numeri, '.', '_' e '-'."

[[ -n "$OUTPUT_ROOT" ]] ||
  fail "Output root vuota."

[[ "$EXPECTED_SCHEMA_VERSION" =~ ^[0-9]+$ ]] ||
  fail "EXPECTED_SCHEMA_VERSION deve essere un intero."

[[ "$SSH_CONNECT_TIMEOUT" =~ ^[0-9]+$ ]] ||
  fail "SSH_CONNECT_TIMEOUT deve essere un intero."

command -v python3 >/dev/null 2>&1 ||
  fail "python3 non trovato."

if [[ "$CHECK_MANIFEST" -ne 1 ]]; then
  command -v scp >/dev/null 2>&1 ||
    fail "scp non trovato."

  command -v sha256sum >/dev/null 2>&1 ||
    fail "sha256sum non trovato."
fi

declare -a NODE_NAMES=()
declare -a MACHINE_TAGS=()
declare -a SSH_TARGETS=()
declare -a REMOTE_PATHS=()
declare -a SSH_PORTS=()

declare -A SEEN_NODE_NAMES=()

line_number=0

while IFS= read -r raw_line ||
  [[ -n "$raw_line" ]]; do

  line_number=$((line_number + 1))

  raw_line="${raw_line%$'\r'}"

  line="$(trim "$raw_line")"

  if [[ -z "$line" ]] ||
    [[ "$line" == \#* ]]; then

    continue
  fi

  IFS='|' read -r \
    node_name \
    machine_tag \
    ssh_target \
    remote_path \
    ssh_port \
    extra <<<"$line"

  node_name="$(trim "${node_name:-}")"
  machine_tag="$(trim "${machine_tag:-}")"
  ssh_target="$(trim "${ssh_target:-}")"
  remote_path="$(trim "${remote_path:-}")"
  ssh_port="$(trim "${ssh_port:-}")"
  extra="$(trim "${extra:-}")"

  [[ -z "$extra" ]] ||
    fail \
      "Manifest riga ${line_number}: troppe colonne."

  [[ -n "$node_name" ]] ||
    fail \
      "Manifest riga ${line_number}: node_name vuoto."

  [[ "$node_name" =~ ^[A-Za-z0-9._-]+$ ]] ||
    fail \
      "Manifest riga ${line_number}: node_name non valido: $node_name"

  [[ -n "$machine_tag" ]] ||
    fail \
      "Manifest riga ${line_number}: machine_tag vuoto."

  [[ "$machine_tag" =~ ^[A-Za-z0-9._/-]+$ ]] ||
    fail \
      "Manifest riga ${line_number}: machine_tag non valido: $machine_tag"

  [[ -n "$ssh_target" ]] ||
    fail \
      "Manifest riga ${line_number}: ssh_target vuoto."

  [[ "$ssh_target" =~ ^[A-Za-z0-9._@-]+$ ]] ||
    fail \
      "Manifest riga ${line_number}: ssh_target non valido: $ssh_target"

  [[ -n "$remote_path" ]] ||
    fail \
      "Manifest riga ${line_number}: remote_path vuoto."

  [[ "$remote_path" == /* ]] ||
    fail \
      "Manifest riga ${line_number}: remote_path deve essere assoluto."

  [[ "$remote_path" != *[[:space:]]* ]] ||
    fail \
      "Manifest riga ${line_number}: remote_path con spazi non supportato."

  [[ "$remote_path" != *"|"* ]] ||
    fail \
      "Manifest riga ${line_number}: remote_path contiene '|'."

  if [[ -z "$ssh_port" ]] ||
    [[ "$ssh_port" == "-" ]]; then

    ssh_port="-"
  else
    [[ "$ssh_port" =~ ^[0-9]+$ ]] ||
      fail \
        "Manifest riga ${line_number}: porta SSH non valida: $ssh_port"

    ((ssh_port >= 1 && ssh_port <= 65535)) ||
      fail \
        "Manifest riga ${line_number}: porta SSH fuori range: $ssh_port"
  fi

  if [[ -n "${SEEN_NODE_NAMES[$node_name]+x}" ]]; then
    fail \
      "Manifest riga ${line_number}: node_name duplicato: $node_name"
  fi

  SEEN_NODE_NAMES["$node_name"]=1

  NODE_NAMES+=(
    "$node_name"
  )

  MACHINE_TAGS+=(
    "$machine_tag"
  )

  SSH_TARGETS+=(
    "$ssh_target"
  )

  REMOTE_PATHS+=(
    "$remote_path"
  )

  SSH_PORTS+=(
    "$ssh_port"
  )
done <"$MANIFEST"

NODE_COUNT="${#NODE_NAMES[@]}"

((NODE_COUNT > 0)) ||
  fail "Il manifest non contiene nodi."

echo "Manifest valido: ${NODE_COUNT} nodi"

for index in "${!NODE_NAMES[@]}"; do
  echo \
    "[node] name=${NODE_NAMES[$index]} machine_tag=${MACHINE_TAGS[$index]} target=${SSH_TARGETS[$index]} port=${SSH_PORTS[$index]} path=${REMOTE_PATHS[$index]}"
done

if [[ "$CHECK_MANIFEST" -eq 1 ]]; then
  echo
  echo "[PASS] Manifest validato senza connessioni remote."
  exit 0
fi

FINAL_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}"

[[ ! -e "$FINAL_DIR" ]] ||
  fail \
    "La raccolta esiste già: $FINAL_DIR"

mkdir -p "$OUTPUT_ROOT"

STAGING_DIR="${OUTPUT_ROOT}/.${EXPERIMENT_ID}.collecting.$$"

mkdir -p "$STAGING_DIR"

METADATA_FILE="${STAGING_DIR}/.collection-records.tsv"

: >"$METADATA_FILE"

echo
echo "============================================================"
echo "RACCOLTA PROFILING DISTRIBUITA"
echo "============================================================"
echo "experiment_id: $EXPERIMENT_ID"
echo "output:        $FINAL_DIR"
echo "nodes:         $NODE_COUNT"
echo

for index in "${!NODE_NAMES[@]}"; do
  node_name="${NODE_NAMES[$index]}"
  machine_tag="${MACHINE_TAGS[$index]}"
  ssh_target="${SSH_TARGETS[$index]}"
  remote_path="${REMOTE_PATHS[$index]}"
  ssh_port="${SSH_PORTS[$index]}"

  node_dir="${STAGING_DIR}/${node_name}"

  mkdir -p "$node_dir"

  destination="${node_dir}/profiling-samples.jsonl"
  temporary="${destination}.part"

  echo \
    "[collect] ${node_name} (${machine_tag}) <- ${ssh_target}:${remote_path}"

  scp_args=(
    -o
    "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  )

  if [[ "$ssh_port" != "-" ]]; then
    scp_args+=(
      -P
      "$ssh_port"
    )
  fi

  scp \
    "${scp_args[@]}" \
    "${ssh_target}:${remote_path}" \
    "$temporary"

  [[ -s "$temporary" ]] ||
    fail \
      "Dataset vuoto ricevuto da $node_name."

  sample_count="$(
    python3 - \
      "$temporary" \
      "$node_name" \
      "$machine_tag" \
      "$EXPECTED_SCHEMA_VERSION" <<'PY'
import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
expected_node = sys.argv[2]
expected_machine_tag = sys.argv[3]
expected_schema = int(sys.argv[4])

raw = path.read_bytes()

if not raw:
    raise SystemExit(
        f"{path}: dataset vuoto"
    )

if not raw.endswith(b"\n"):
    raise SystemExit(
        f"{path}: newline finale mancante; "
        "il file potrebbe essere stato copiato durante una scrittura"
    )

records = []
request_ids = set()

for line_number, line in enumerate(
    raw.splitlines(),
    start=1,
):
    if not line.strip():
        continue

    try:
        sample = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path}: JSON non valido alla riga "
            f"{line_number}: {exc}"
        )

    if sample.get("schema_version") != expected_schema:
        raise SystemExit(
            f"{path}: schema_version non valido "
            f"alla riga {line_number}: "
            f"{sample.get('schema_version')!r}, "
            f"atteso {expected_schema}"
        )

    if sample.get("node_name") != expected_node:
        raise SystemExit(
            f"{path}: node_name non valido "
            f"alla riga {line_number}: "
            f"{sample.get('node_name')!r}, "
            f"atteso {expected_node!r}"
        )

    if sample.get("machine_tag") != expected_machine_tag:
        raise SystemExit(
            f"{path}: machine_tag non valido "
            f"alla riga {line_number}: "
            f"{sample.get('machine_tag')!r}, "
            f"atteso {expected_machine_tag!r}"
        )

    request_id = sample.get("request_id")

    if not isinstance(request_id, str) or not request_id.strip():
        raise SystemExit(
            f"{path}: request_id vuoto/non valido "
            f"alla riga {line_number}"
        )

    if request_id in request_ids:
        raise SystemExit(
            f"{path}: request_id duplicato "
            f"{request_id!r}"
        )

    request_ids.add(request_id)
    records.append(sample)

if not records:
    raise SystemExit(
        f"{path}: nessun InvocationSample"
    )

print(len(records))
PY
  )"

  mv \
    "$temporary" \
    "$destination"

  sha256="$(
    sha256sum \
      "$destination" |
      awk '{print $1}'
  )"

  relative_dataset="${node_name}/profiling-samples.jsonl"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$node_name" \
    "$machine_tag" \
    "$ssh_target" \
    "$remote_path" \
    "$relative_dataset" \
    "$sample_count" \
    "$sha256" \
    >>"$METADATA_FILE"

  echo \
    "[PASS] ${node_name}: ${sample_count} campioni validi, sha256=${sha256}"
done

python3 - \
  "$METADATA_FILE" \
  "$STAGING_DIR/collection-manifest.json" \
  "$EXPERIMENT_ID" \
  "$EXPECTED_SCHEMA_VERSION" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


metadata_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
experiment_id = sys.argv[3]
sample_schema = int(sys.argv[4])

nodes = []
global_request_ids = {}

for line_number, line in enumerate(
    metadata_path.read_text(
        encoding="utf-8"
    ).splitlines(),
    start=1,
):
    if not line.strip():
        continue

    fields = line.split("\t")

    if len(fields) != 7:
        raise SystemExit(
            "metadata interna non valida alla "
            f"riga {line_number}"
        )

    (
        node_name,
        machine_tag,
        ssh_target,
        remote_path,
        relative_dataset,
        sample_count,
        sha256,
    ) = fields

    dataset_path = (
        metadata_path.parent
        /
        relative_dataset
    )

    actual_count = 0

    for json_line_number, raw_line in enumerate(
        dataset_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        sample = json.loads(
            raw_line
        )

        request_id = sample[
            "request_id"
        ]

        if request_id in global_request_ids:
            previous = global_request_ids[
                request_id
            ]

            raise SystemExit(
                "request_id duplicato tra dataset: "
                f"{request_id!r} presente in "
                f"{previous!r} e {node_name!r}"
            )

        global_request_ids[
            request_id
        ] = node_name

        actual_count += 1

    if actual_count != int(
        sample_count
    ):
        raise SystemExit(
            f"conteggio inconsistente per {node_name}"
        )

    nodes.append(
        {
            "node_name":
                node_name,

            "machine_tag":
                machine_tag,

            "ssh_target":
                ssh_target,

            "remote_path":
                remote_path,

            "dataset":
                relative_dataset,

            "sample_count":
                int(
                    sample_count
                ),

            "sha256":
                sha256,
        }
    )

summary = {
    "schema_version":
        1,

    "experiment_id":
        experiment_id,

    "collected_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "invocation_sample_schema_version":
        sample_schema,

    "node_count":
        len(
            nodes
        ),

    "total_sample_count":
        sum(
            node[
                "sample_count"
            ]
            for node in nodes
        ),

    "nodes":
        nodes,
}

output_path.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)
PY

rm -f "$METADATA_FILE"

mv \
  "$STAGING_DIR" \
  "$FINAL_DIR"

STAGING_DIR=""
COLLECTION_COMPLETED=1

echo
echo "============================================================"
echo "RACCOLTA COMPLETATA"
echo "============================================================"
echo
echo "[PASS] Tutti i dataset remoti sono stati copiati."
echo "[PASS] Tutti i JSONL rispettano lo schema atteso."
echo "[PASS] NodeName e MachineTag corrispondono al manifest."
echo "[PASS] I request ID sono unici anche tra nodi differenti."
echo "[PASS] La raccolta è stata pubblicata atomicamente."
echo
echo "Dataset directory:"
echo "  $FINAL_DIR"
echo
echo "Collection manifest:"
echo "  $FINAL_DIR/collection-manifest.json"
echo
echo "Per aggregare:"
echo
echo "  bin/serverledge-profiling aggregate \\"
echo "    --input-dir \"$FINAL_DIR\" \\"
echo "    --output \"$FINAL_DIR/function-profiles.jsonl\" \\"
echo "    --samples 10"