#!/usr/bin/env bash
#
# Scarica in locale tutto ciò che serve ad analizzare un esperimento.
#
#     ./gcp_collect.sh LinUCB
#     ./gcp_collect.sh RoundRobin ~/tesi/risultati
#
# Va eseguito PRIMA di gcp_down.sh: le VM sono effimere e con esse spariscono
# CSV e log. Questo è il punto in cui un esperimento da venti minuti si
# trasforma in dati che sopravvivono alla sessione.
#
# Raccoglie tre cose:
#   - i CSV di locust e il CSV per-richiesta del locustfile
#   - il log del load balancer, che contiene gli eventi [LB][MAB]
#   - i log dei worker, con i profili delle invocazioni
#
# e scrive un file di manifest con la configurazione dell'esperimento, così che
# i dati restino interpretabili anche a mesi di distanza.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

POLICY="${1:-unknown}"
DEST_ROOT="${2:-./risultati}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="${DEST_ROOT}/${POLICY}_${STAMP}"

banner "RACCOLTA RISULTATI — $POLICY"

mkdir -p "${DEST}/locust" "${DEST}/logs"

echo "destinazione: $DEST"

fetch() {
    local host="$1"
    local remote_path="$2"
    local local_name="$3"

    if gc compute scp --zone="$ZONE" --quiet --recurse \
        "${host}:${remote_path}" "${DEST}/${local_name}" >/dev/null 2>&1; then
        echo "  $host:${remote_path}"
    else
        echo "  $host:${remote_path}  (assente)"
    fi
}

# --- Risultati di locust -----------------------------------------------------

banner "LOCUST"

gc compute ssh "$NAME_WORKLOAD" --zone="$ZONE" --quiet --command="
    sudo chmod -R a+r /opt/serverledge/results 2>/dev/null || true
    ls -1d /opt/serverledge/results/*/ 2>/dev/null | tail -5
" || true

fetch "$NAME_WORKLOAD" "/opt/serverledge/results/*" "locust/"

# --- Log del load balancer ---------------------------------------------------
#
# È il file più prezioso dopo i CSV: contiene gli eventi di selezione del
# bandit, la scoperta dei machine tag e gli aggiornamenti di reward.

banner "LOG"

gc compute ssh "$NAME_LB" --zone="$ZONE" --quiet \
    --command="sudo cp /var/log/serverledge-lb.log /tmp/lb.log && sudo chmod a+r /tmp/lb.log" \
    >/dev/null 2>&1 || true

fetch "$NAME_LB" "/tmp/lb.log" "logs/load-balancer.log"

# --- Log dei worker ----------------------------------------------------------

while read -r host; do
    [[ -z "$host" ]] && continue

    gc compute ssh "$host" --zone="$ZONE" --quiet \
        --command="sudo cp /var/log/serverledge-node.log /tmp/node.log && sudo chmod a+r /tmp/node.log" \
        >/dev/null 2>&1 || true

    fetch "$host" "/tmp/node.log" "logs/${host}.log"

done < <(x86_node_names; arm_node_names)

# --- Manifest ----------------------------------------------------------------
#
# Senza questo file, fra due mesi un CSV è una tabella di numeri senza contesto:
# non si sa con quale policy, quante macchine, quale commit del codice.

banner "MANIFEST"

CODE_COMMIT="$(
    gc compute ssh "$NAME_LB" --zone="$ZONE" --quiet \
        --command="sudo git -C /opt/serverledge rev-parse HEAD" 2>/dev/null | tr -d '[:space:]' || echo sconosciuto
)"

cat > "${DEST}/manifest.txt" <<EOF
esperimento:      ${POLICY}
raccolto il:      $(date -Iseconds)
zona:             ${ZONE}
progetto:         ${PROJECT}

nodi x86:         ${N_X86} (${MT_X86})
nodi ARM:         ${N_ARM} (${MT_ARM})
load balancer:    ${MT_LB}
registry:         ${MT_REGISTRY}
generatore:       ${MT_WORKLOAD}
provisioning:     $([[ "$SPOT" == "1" ]] && echo SPOT || echo on-demand)

commit del codice: ${CODE_COMMIT}

macchine attive al momento della raccolta:
$(gc compute instances list --format="table(name,machineType.basename(),status)" 2>/dev/null | sed 's/^/  /')
EOF

cat "${DEST}/manifest.txt"

banner "FATTO"

echo "Dati salvati in: ${DEST}"
echo
find "${DEST}" -type f | sed 's|^|  |'
echo
echo "Ora puoi distruggere il cluster senza perdere nulla:"
echo "    ./gcp_down.sh"
