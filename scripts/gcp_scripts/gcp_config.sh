#!/usr/bin/env bash
#
# Configurazione condivisa degli script GCP.
#
# Non va eseguito direttamente: gli altri script lo caricano con "source".
# Tutti i valori possono essere sovrascritti da variabili d'ambiente, quindi
# per una prova rapida basta ad esempio:
#
#     ZONE=us-central1-a ./gcp-up.sh
#
# Il conteggio delle macchine rispecchia la Tabella 4 della tesi di Filippo:
# 1 generatore di workload, 1 load balancer, 1 global registry, 3 nodi x86,
# 3 nodi ARM.

set -euo pipefail

# --- Progetto e zona ---------------------------------------------------------

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"
REGION="${REGION:-${ZONE%-*}}"

# --- Modello di provisioning -------------------------------------------------
#
# Finché la quota PREEMPTIBLE_CPUS resta a zero, SPOT non è utilizzabile.
# Quando l'aumento verrà approvato basterà esportare SPOT=1 per dimezzare il
# costo orario senza toccare altro.

SPOT="${SPOT:-0}"

if [[ "$SPOT" == "1" ]]; then
    PROVISIONING_FLAGS=(
        --provisioning-model=SPOT
        --instance-termination-action=DELETE
    )
else
    PROVISIONING_FLAGS=()
fi

# --- Tipi di macchina --------------------------------------------------------

MT_WORKLOAD="${MT_WORKLOAD:-e2-standard-2}"
MT_LB="${MT_LB:-n2-standard-4}"
MT_REGISTRY="${MT_REGISTRY:-n2-standard-2}"
MT_X86="${MT_X86:-n2-standard-4}"
MT_ARM="${MT_ARM:-t2a-standard-4}"

# --- Numero di nodi worker ---------------------------------------------------
#
# Per il primo giro di validazione conviene partire da 1+1: il cluster costa un
# terzo e verifica esattamente le stesse cose. Si passa a 3+3 solo per la
# campagna di misura vera.

N_X86="${N_X86:-3}"
N_ARM="${N_ARM:-3}"

# --- Immagini ----------------------------------------------------------------
#
# Se le immagini personalizzate esistono vengono usate quelle, altrimenti si
# parte dalle immagini pubbliche Ubuntu. Le prime evitano di rifare
# l'installazione su nove macchine a ogni sessione.

IMAGE_X86="${IMAGE_X86:-sl-x86}"
IMAGE_ARM="${IMAGE_ARM:-sl-arm}"

BASE_IMAGE_X86_FAMILY="ubuntu-2404-lts-amd64"
BASE_IMAGE_ARM_FAMILY="ubuntu-2404-lts-arm64"
BASE_IMAGE_PROJECT="ubuntu-os-cloud"

DISK_SIZE="${DISK_SIZE:-50GB}"

# --- Nomi delle istanze ------------------------------------------------------

NAME_WORKLOAD="sl-workload"
NAME_LB="sl-lb"
NAME_REGISTRY="sl-registry"

x86_node_names() {
    local index
    for ((index = 1; index <= N_X86; index++)); do
        echo "sl-x86-${index}"
    done
}

arm_node_names() {
    local index
    for ((index = 1; index <= N_ARM; index++)); do
        echo "sl-arm-${index}"
    done
}

all_instance_names() {
    echo "$NAME_WORKLOAD"
    echo "$NAME_LB"
    echo "$NAME_REGISTRY"
    x86_node_names
    arm_node_names
}

# --- Utilità -----------------------------------------------------------------

gc() {
    gcloud --project="$PROJECT" "$@"
}

# image_exists restituisce 0 se l'immagine personalizzata è già stata creata.
image_exists() {
    gc compute images describe "$1" >/dev/null 2>&1
}

# x86_image_flags e arm_image_flags scelgono fra immagine personalizzata e
# immagine pubblica, così gli stessi script funzionano prima e dopo la
# creazione delle immagini.
x86_image_flags() {
    if image_exists "$IMAGE_X86"; then
        echo "--image=$IMAGE_X86"
    else
        echo "--image-family=$BASE_IMAGE_X86_FAMILY --image-project=$BASE_IMAGE_PROJECT"
    fi
}

arm_image_flags() {
    if image_exists "$IMAGE_ARM"; then
        echo "--image=$IMAGE_ARM"
    else
        echo "--image-family=$BASE_IMAGE_ARM_FAMILY --image-project=$BASE_IMAGE_PROJECT"
    fi
}

banner() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}
