#!/usr/bin/env bash
#
# prova_cpu.sh
#
# Verifica controllata dell'effetto di CPUDemand sul confronto
# GCP N2/x86 (Intel) vs T2A/ARM (Ampere Altra).
#
# Default: preflight economico
#   - 1 worker x86 + 1 worker ARM
#   - funzione amd_faster
#   - CPU=0.25
#   - 1 misura warm per architettura
#
#   ./prova_cpu.sh
#
# Esperimento completo:
#   MODE=full ./prova_cpu.sh
#
# Il test usa cluster OMOGENEI separati per architettura, perché il
# RoundRobin di Serverledge usa un hash ring e una funzione, in un cluster
# eterogeneo, non garantisce osservazioni su entrambe le architetture.
#
# Le VM vengono distrutte automaticamente sia in caso di errore sia dopo
# un PASS, per evitare consumo involontario di credito. Per conservarle:
#
#   AUTO_DESTROY=0 ./prova_cpu.sh
#   AUTO_DESTROY=0 MODE=full ./prova_cpu.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

MODE="${MODE:-preflight}"
AUTO_DESTROY="${AUTO_DESTROY:-1}"

case "$MODE" in
    preflight)
        TEST_N_X86=1
        TEST_N_ARM=1
        RIPETIZIONI=1
        VALORI_CPU="0.25"
        FUNZIONI=(
            "amd_faster|go125|amd_fasterV2.tar|1024|"
        )
        ;;
    full)
        TEST_N_X86="${TEST_N_X86:-3}"
        TEST_N_ARM="${TEST_N_ARM:-3}"
        RIPETIZIONI="${RIPETIZIONI:-5}"
        VALORI_CPU="${VALORI_CPU:-0.25 0.5 1.0 2.0}"
        FUNZIONI=(
            "amd_faster|go125|amd_fasterV2.tar|1024|"
            "arm_faster|go125|arm_fasterV2.tar|1024|"
            "readdisk|go125|readdisk.tar|1024|"
            "chacha20|go125|chacha20.tar|1024|"
            "primenumber|go125|primenum.tar|1024|"
            "linpack|python312ml|linpack.py|2048|linpack.handler"
        )
        ;;
    *)
        echo "MODE non valido: $MODE (usa preflight oppure full)" >&2
        exit 2
        ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${SCRIPT_DIR}/risultati/prova-cpu-${MODE}-${STAMP}"
RAW="${RESULT_DIR}/raw.tsv"
RUN_LOG="${RESULT_DIR}/run.log"
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
SUMMARY_TXT="${RESULT_DIR}/summary.txt"
INSTANCES="${RESULT_DIR}/instances.txt"
METADATA="${RESULT_DIR}/metadata.txt"

mkdir -p "$RESULT_DIR"
printf "cpu\texperiment_arch\tfunction\trepetition\tnode_arch\tresponse_time\n" > "$RAW"
: > "$RUN_LOG"

RUN_OK=0
LAST_ERROR_LINE="?"

on_err() {
    LAST_ERROR_LINE="$LINENO"
}
trap on_err ERR

on_exit() {
    local rc=$?
    trap - EXIT ERR
    set +e

    if (( rc != 0 )); then
        echo >&2
        echo "ERRORE prova_cpu.sh: exit=$rc, linea=${LAST_ERROR_LINE}" >&2
        echo "Log locale: $RUN_LOG" >&2
        echo "Raw locale: $RAW" >&2
    fi

    if [[ "$AUTO_DESTROY" == "1" ]]; then
        if (( rc == 0 )); then
            echo
            echo "AUTO_DESTROY=1: test terminato, distruggo le VM..."
        else
            echo "AUTO_DESTROY=1: errore, distruggo le VM..." >&2
        fi
        "${SCRIPT_DIR}/gcp_down.sh" || true
    else
        echo
        echo "AUTO_DESTROY=0: VM lasciate accese."
        echo "Quando hai finito: ${SCRIPT_DIR}/gcp_down.sh"
    fi

    exit "$rc"
}
trap on_exit EXIT

banner_local() {
    echo
    echo "############################################################"
    echo "# $1"
    echo "############################################################"
}

all_test_workers() {
    local i
    for ((i=1; i<=TEST_N_X86; i++)); do echo "sl-x86-${i}"; done
    for ((i=1; i<=TEST_N_ARM; i++)); do echo "sl-arm-${i}"; done
}

instance_exists() {
    gc compute instances describe "$1" --zone="$ZONE" >/dev/null 2>&1
}

stop_all_workers() {
    local h
    while read -r h; do
        [[ -n "$h" ]] || continue
        instance_exists "$h" || continue
        gc compute ssh "$h" \
            --zone="$ZONE" \
            --quiet \
            --command='sudo pkill -x serverledge >/dev/null 2>&1 || true' \
            </dev/null >/dev/null 2>&1 || true
    done < <(all_test_workers)
}

clear_function_containers() {
    local h
    while read -r h; do
        [[ -n "$h" ]] || continue
        instance_exists "$h" || continue
        gc compute ssh "$h" \
            --zone="$ZONE" \
            --quiet \
            --command='
                ids="$(sudo docker ps -aq)"
                if [ -n "$ids" ]; then
                    sudo docker rm -f $ids >/dev/null 2>&1 || true
                fi
            ' \
            </dev/null >/dev/null 2>&1 || true
    done < <(all_test_workers)
}

start_homogeneous_cluster() {
    local arch="$1"

    banner_local "RESET PRIMA DELLA FASE $arch"
    stop_all_workers
    clear_function_containers

    case "$arch" in
        amd64)
            banner_local "AVVIO CLUSTER x86-ONLY"
            N_X86="$TEST_N_X86" N_ARM=0 \
                "${SCRIPT_DIR}/gcp_start_cluster.sh" RoundRobin
            ;;
        arm64)
            banner_local "AVVIO CLUSTER ARM-ONLY"
            N_X86=0 N_ARM="$TEST_N_ARM" \
                "${SCRIPT_DIR}/gcp_start_cluster.sh" RoundRobin
            ;;
        *)
            echo "Architettura non supportata: $arch" >&2
            return 2
            ;;
    esac
}

banner_local "INFRASTRUTTURA — MODE=$MODE"
N_X86="$TEST_N_X86" N_ARM="$TEST_N_ARM" "${SCRIPT_DIR}/gcp_up.sh"

{
    echo "timestamp=$STAMP"
    echo "mode=$MODE"
    echo "project=$PROJECT"
    echo "zone=$ZONE"
    echo "x86_machine_type=$MT_X86"
    echo "arm_machine_type=$MT_ARM"
    echo "n_x86=$TEST_N_X86"
    echo "n_arm=$TEST_N_ARM"
    echo "cpu_values=$VALORI_CPU"
    echo "repetitions=$RIPETIZIONI"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
} > "$METADATA"

banner_local "PIATTAFORME CPU"
{
    printf "%-12s %-18s %-30s %-10s\n" \
        "instance" "machine_type" "cpu_platform" "status"

    while read -r h; do
        [[ -n "$h" ]] || continue
        instance_exists "$h" || continue

        name="$(gc compute instances describe "$h" --zone="$ZONE" --format='value(name)')"
        mt="$(gc compute instances describe "$h" --zone="$ZONE" --format='value(machineType.basename())')"
        cp="$(gc compute instances describe "$h" --zone="$ZONE" --format='value(cpuPlatform)')"
        st="$(gc compute instances describe "$h" --zone="$ZONE" --format='value(status)')"

        printf "%-12s %-18s %-30s %-10s\n" "$name" "$mt" "$cp" "$st"
    done < <(all_test_workers)
} | tee "$INSTANCES"

LB="$(
    gc compute instances describe "$NAME_LB" \
        --zone="$ZONE" \
        --format='value(networkInterfaces[0].networkIP)'
)"

cat > /tmp/misura_cpu_debug.sh <<'REMOTE'
#!/usr/bin/env bash
set -uo pipefail

die() {
    echo "REMOTE_FAIL: $*" >&2
    exit 1
}

LB="$1"
CPU="$2"
RIPETIZIONI="$3"
EXPECTED_ARCH="$4"
shift 4

CLI=/opt/serverledge/bin/serverledge-cli
E=/opt/serverledge/examples/experiments

[[ -x "$CLI" ]] || die "CLI assente/non eseguibile: $CLI"
[[ -d "$E" ]] || die "directory esperimenti assente: $E"

export SERVERLEDGE_HOST="$LB"
export SERVERLEDGE_PORT=1323
cd "$E" || die "cd $E fallita"

invoke_once() {
    local function="$1"
    local label="$2"
    local headers body http arch success response_time

    headers="$(mktemp)"
    body="$(mktemp)"

    if ! http="$(
        curl -sS -m 900 \
            -D "$headers" \
            -o "$body" \
            -w '%{http_code}' \
            -X POST "http://$LB:1323/invoke/$function" \
            -H 'Content-Type: application/json' \
            -d '{"params":{}}'
    )"; then
        echo "curl fallita: function=$function label=$label" >&2
        cat "$headers" >&2 || true
        cat "$body" >&2 || true
        rm -f "$headers" "$body"
        return 1
    fi

    # Serverledge usa attualmente:
    #     Serverledge-Node-Arch: amd64
    # Manteniamo anche Node-Arch come alias per compatibilita' con log/script
    # precedenti.
    arch="$(
        awk -F': *' '
            {
                key=tolower($1)
                gsub(/[[:space:]]/, "", key)
            }
            key == "serverledge-node-arch" || key == "node-arch" {
                gsub("\r", "", $2)
                print $2
                exit
            }
        ' "$headers"
    )"

    read -r success response_time < <(
        python3 - "$body" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        obj = json.load(f)
except Exception:
    print("parse_error", "NA")
    raise SystemExit

success = obj.get("Success")
rt = obj.get("ResponseTime")
print(str(success).lower(), "NA" if rt is None else rt)
PY
    )

    if [[ ! "$http" =~ ^2 ]]; then
        echo "HTTP non 2xx: $http function=$function label=$label" >&2
        echo "----- headers -----" >&2
        cat "$headers" >&2
        echo "----- body -----" >&2
        cat "$body" >&2
        rm -f "$headers" "$body"
        return 1
    fi

    if [[ "$arch" != "$EXPECTED_ARCH" ]]; then
        echo "Arch errata: function=$function label=$label attesa=$EXPECTED_ARCH ottenuta=${arch:-MISSING}" >&2
        echo "----- headers -----" >&2
        cat "$headers" >&2
        echo "----- body -----" >&2
        cat "$body" >&2
        rm -f "$headers" "$body"
        return 1
    fi

    if [[ "$success" != "true" ]]; then
        echo "Success!=true: function=$function label=$label success=$success" >&2
        cat "$body" >&2
        rm -f "$headers" "$body"
        return 1
    fi

    if [[ "$response_time" == "NA" ]]; then
        echo "ResponseTime mancante: function=$function label=$label" >&2
        cat "$body" >&2
        rm -f "$headers" "$body"
        return 1
    fi

    printf '%s\n' "$response_time"
    rm -f "$headers" "$body"
}

for spec in "$@"; do
    IFS='|' read -r nome runtime src mem handler <<< "$spec"

    echo "[REMOTE] function=$nome runtime=$runtime src=$src cpu=$CPU arch=$EXPECTED_ARCH" >&2
    [[ -f "$src" ]] || die "sorgente assente: $PWD/$src"

    handler_args=()
    [[ -n "$handler" ]] && handler_args=(--handler "$handler")

    echo "[CREATE] $nome" >&2
    if ! create_out="$(
        "$CLI" create \
            -f "$nome" \
            --runtime "$runtime" \
            --src "$src" \
            --memory "$mem" \
            --cpu "$CPU" \
            "${handler_args[@]}" \
            --update \
            2>&1
    )"; then
        echo "$create_out" >&2
        die "create fallita per $nome"
    fi
    echo "$create_out" >&2

    echo "[COLD] $nome" >&2
    invoke_once "$nome" cold >/dev/null || die "cold fallita per $nome"

    echo "[WARMUP] $nome" >&2
    invoke_once "$nome" warmup >/dev/null || die "warmup fallita per $nome"

    for i in $(seq 1 "$RIPETIZIONI"); do
        tempo="$(invoke_once "$nome" "warm-$i")" || die "warm-$i fallita per $nome"

        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$CPU" \
            "$EXPECTED_ARCH" \
            "$nome" \
            "$i" \
            "$EXPECTED_ARCH" \
            "$tempo"
    done
done
REMOTE

chmod +x /tmp/misura_cpu_debug.sh

gc compute scp \
    /tmp/misura_cpu_debug.sh \
    "$NAME_WORKLOAD:/tmp/" \
    --zone="$ZONE" \
    >/dev/null

for cpu in $VALORI_CPU; do
    for arch in amd64 arm64; do
        banner_local "MISURA CPU=$cpu ARCH=$arch"
        start_homogeneous_cluster "$arch"

        printf -v REMOTE_COMMAND '%q ' \
            /tmp/misura_cpu_debug.sh \
            "$LB" \
            "$cpu" \
            "$RIPETIZIONI" \
            "$arch" \
            "${FUNZIONI[@]}"

        set +e
        gc compute ssh "$NAME_WORKLOAD" \
            --zone="$ZONE" \
            --quiet \
            --command="$REMOTE_COMMAND" \
            </dev/null \
            2>&1 \
            | tee -a "$RUN_LOG" \
            | awk -F'\t' 'NF == 6 {print}' \
            >> "$RAW"

        ssh_rc=${PIPESTATUS[0]}
        set -e

        if (( ssh_rc != 0 )); then
            echo "La misura remota e' fallita: cpu=$cpu arch=$arch rc=$ssh_rc" >&2
            exit "$ssh_rc"
        fi
    done
done

banner_local "CALCOLO RISULTATI"

python3 - "$RAW" "$SUMMARY_CSV" "$SUMMARY_TXT" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

raw_path = Path(sys.argv[1])
csv_out = Path(sys.argv[2])
txt_out = Path(sys.argv[3])

values = defaultdict(list)

with raw_path.open(newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            values[
                (row["cpu"], row["function"], row["experiment_arch"])
            ].append(float(row["response_time"]))
        except (KeyError, ValueError):
            continue

cpus = sorted({k[0] for k in values}, key=float)
functions = sorted({k[1] for k in values})

rows = []

for cpu in cpus:
    for fn in functions:
        x = values.get((cpu, fn, "amd64"), [])
        a = values.get((cpu, fn, "arm64"), [])

        if not x or not a:
            continue

        mx = statistics.median(x)
        ma = statistics.median(a)
        delta = (ma - mx) / mx * 100.0

        rows.append({
            "cpu": cpu,
            "function": fn,
            "x86_n": len(x),
            "arm_n": len(a),
            "x86_median": mx,
            "arm_median": ma,
            "delta_arm_vs_x86_percent": delta,
        })

with csv_out.open("w", newline="") as f:
    fields = [
        "cpu",
        "function",
        "x86_n",
        "arm_n",
        "x86_median",
        "arm_median",
        "delta_arm_vs_x86_percent",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

lines = []
header = f"{'funzione':14}" + "".join(f"  cpu={c:<7}" for c in cpus)
lines.append(header)
lines.append("-" * len(header))

for fn in functions:
    line = f"{fn:14}"
    for cpu in cpus:
        r = next(
            (r for r in rows if r["cpu"] == cpu and r["function"] == fn),
            None,
        )
        line += (
            f"  {r['delta_arm_vs_x86_percent']:>+9.1f}%"
            if r
            else f"  {'n/d':>10}"
        )
    lines.append(line)

lines += [
    "",
    "Delta = (mediana ARM - mediana x86) / mediana x86 * 100.",
    "Positivo: x86 piu' veloce.",
    "Negativo: ARM piu' veloce.",
]

txt = "\n".join(lines) + "\n"
txt_out.write_text(txt)
print(txt)
PY

RUN_OK=1

banner_local "PASS"
echo "raw:      $RAW"
echo "summary:  $SUMMARY_CSV"
echo "report:   $SUMMARY_TXT"
echo "run log:  $RUN_LOG"
echo "metadata: $METADATA"