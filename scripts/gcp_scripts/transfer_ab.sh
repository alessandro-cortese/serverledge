#!/usr/bin/env bash
#
# Esperimento A/B sul transfer learning, da eseguire SUL GENERATORE DI WORKLOAD.
#
#     ./transfer_ab.sh <LB_IP> A     condizione senza prior
#     ./transfer_ab.sh <LB_IP> B     condizione con prior
#
# Va copiato sulla VM ed eseguito li', invece di essere passato inline a
# "gcloud compute ssh --command": le virgolette annidate di un comando remoto
# che contiene a sua volta python e curl si rompono con facilita', e la shell
# resta in attesa di una chiusura senza eseguire nulla.
#
# Le due condizioni vanno eseguite su cluster appena riavviati: il bandit non
# deve avere osservazioni precedenti sulla funzione target, altrimenti il prior
# si sovrappone a uno stato gia' formato e il confronto perde significato.

set -euo pipefail

LB="${1:?indicare l IP interno del load balancer}"
CONDITION="${2:?indicare A senza prior oppure B con prior}"

DONOR="${DONOR:-readmemory}"
TARGET="${TARGET:-readdisk}"
WARMUP="${WARMUP:-40}"
MEASURE="${MEASURE:-30}"
ARTIFACT="${ARTIFACT:-/tmp/selection-readdisk.json}"

CLI=/opt/serverledge/bin/serverledge-cli
EXPERIMENTS=/opt/serverledge/examples/experiments

export SERVERLEDGE_HOST="$LB"
export SERVERLEDGE_PORT=1323

cd "$EXPERIMENTS"

echo "=== condizione $CONDITION - donor=$DONOR target=$TARGET"

$CLI create -f "$DONOR"  --runtime go125 --src "${DONOR}.tar"  --memory 512 --cpu 0.25 --update
$CLI create -f "$TARGET" --runtime go125 --src "${TARGET}.tar" --memory 512 --cpu 0.25 --update

# Il weak prior viene costruito dal bandit VIVO del donor nel processo del load
# balancer, non dall'artefatto: senza abbastanza osservazioni reali per arm il
# trasferimento viene saltato per min_real_observations_per_arm.
echo "--- scaldo il donor ($WARMUP invocazioni)"
for i in $(seq 1 "$WARMUP"); do
    curl -s -m 120 -X POST "http://${LB}:1323/invoke/${DONOR}" \
        -H 'Content-Type: application/json' -d '{"params":{}}' > /dev/null
done
echo "    fatto"

if [ "$CONDITION" = "B" ]; then
    echo "--- invio artefatto di selezione"
    python3 - "$ARTIFACT" "$TARGET" > /tmp/payload.json <<'PY'
import json
import sys

artifact = json.load(open(sys.argv[1]))

payload = {
    "target_function_name": sys.argv[2],
    "selection_artifact": artifact,
    "prior_config": {
        "equivalent_observation_weight": 1.0,
        "min_real_observations_per_arm": 5,
    },
}

print(json.dumps(payload))
PY
    curl -s -X POST "http://${LB}:1323/mab/transfer/initialize" \
        -H 'Content-Type: application/json' --data @/tmp/payload.json
    echo
    echo "    verificare sopra che il prior sia stato applicato"
fi

echo "--- $MEASURE invocazioni del target"
for i in $(seq 1 "$MEASURE"); do
    arch=$(curl -s -i -m 120 -X POST "http://${LB}:1323/invoke/${TARGET}" \
        -H 'Content-Type: application/json' -d '{"params":{}}' \
        | grep -i 'Node-Arch' | tr -d '\r' | cut -d' ' -f2)
    echo "$i ${arch:-FALLITA}"
done
