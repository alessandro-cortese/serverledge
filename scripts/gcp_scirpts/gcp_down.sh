#!/usr/bin/env bash
#
# Distrugge il cluster e verifica che non resti nulla a consumare credito.
#
#     ./gcp-down.sh              # cancella istanze, poi controlla i residui
#     ./gcp-down.sh --all        # cancella anche le immagini personalizzate
#
# Il controllo finale è la parte che conta davvero: una VM cancellata male
# lascia il disco, e tre dischi da 50 GB dimenticati costano circa 15 dollari
# al mese — quasi un terzo del credito, per macchine che non stanno facendo
# niente.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp-config.sh"

DELETE_IMAGES=0

if [[ "${1:-}" == "--all" ]]; then
    DELETE_IMAGES=1
fi

banner "CLUSTER SERVERLEDGE — DISTRUZIONE"

# --- Istanze -----------------------------------------------------------------

RUNNING="$(
    gc compute instances list \
        --filter="labels.progetto=serverledge-tesi AND zone:($ZONE)" \
        --format="value(name)" \
    || true
)"

if [[ -z "$RUNNING" ]]; then
    echo "Nessuna istanza da cancellare."
else
    echo "Istanze da cancellare:"
    echo "$RUNNING" | sed 's/^/  /'
    echo

    # shellcheck disable=SC2086
    gc compute instances delete $RUNNING --zone="$ZONE" --quiet
    echo "Istanze cancellate."
fi

# --- Immagini ----------------------------------------------------------------

if [[ "$DELETE_IMAGES" == "1" ]]; then

    banner "IMMAGINI PERSONALIZZATE"

    for image in "$IMAGE_X86" "$IMAGE_ARM"; do
        if image_exists "$image"; then
            gc compute images delete "$image" --quiet
            echo "  $image cancellata"
        fi
    done
else
    echo
    echo "Immagini personalizzate conservate: costano pochi centesimi al mese"
    echo "ed evitano di rifare il setup alla prossima sessione."
    echo "Per cancellarle: ./gcp-down.sh --all"
fi

# --- Verifica dei residui ----------------------------------------------------

banner "VERIFICA RESIDUI"

RESIDUI=0

check_empty() {
    local label="$1"
    local output="$2"

    if [[ -n "$output" ]]; then
        echo "ATTENZIONE — $label ancora presenti:"
        echo "$output" | sed 's/^/  /'
        RESIDUI=1
    else
        echo "OK — nessun $label"
    fi
}

check_empty "istanza" "$(
    gc compute instances list --format="value(name,zone,status)" || true
)"

check_empty "disco" "$(
    gc compute disks list --format="value(name,zone,sizeGb,users)" || true
)"

check_empty "indirizzo IP statico" "$(
    gc compute addresses list --format="value(name,region,status)" || true
)"

check_empty "snapshot" "$(
    gc compute snapshots list --format="value(name,diskSizeGb)" || true
)"

echo
if [[ "$RESIDUI" == "1" ]]; then
    echo "Ci sono risorse residue: controllale, alcune fatturano anche da spente."
    echo "In particolare i dischi non collegati continuano a costare."
    exit 1
fi

echo "Nessuna risorsa residua. Il credito non sta più scendendo."
