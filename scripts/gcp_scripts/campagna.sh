#!/usr/bin/env bash
#
# Campagna di caratterizzazione su una sola architettura.
#
#     ./campagna.sh x86 40m
#     ./campagna.sh arm 40m
#
# Esegue l'intera sequenza: creazione delle VM, ricostruzione dell'immagine
# runtime, avvio del cluster con profiling attivo, esperimento, raccolta dei
# campioni e aggregazione.
#
# Il cluster e' omogeneo di proposito: le campagne di caratterizzazione
# misurano ogni funzione su una architettura per volta, cosi' che le durate
# siano confrontabili fra le due e che tutte le funzioni vengano misurate su
# entrambe. Un cluster eterogeneo produrrebbe campioni sbilanciati, perche'
# l'hash ring inchioda ciascuna funzione a un solo nodo.
#
# Le VM sono on-demand: una preemption a meta' di quaranta minuti
# invaliderebbe la raccolta.

set -euo pipefail

ARCH="${1:?indicare x86 oppure arm}"
DURATA="${2:-40m}"

ZONE="${ZONE:-europe-west4-a}"
NODI="${NODI:-6}"
UTENTI="${UTENTI:-30}"

case "$ARCH" in
    x86) N_X86=$NODI; N_ARM=0; PREFISSO="sl-x86" ;;
    arm) N_X86=0; N_ARM=$NODI; PREFISSO="sl-arm" ;;
    *)   echo "Architettura non riconosciuta: $ARCH"; exit 1 ;;
esac

export N_X86 N_ARM

WORKERS=()
for ((i = 1; i <= NODI; i++)); do
    WORKERS+=("${PREFISSO}-${i}")
done

banner() {
    echo
    echo "############################################################"
    echo "# $1"
    echo "############################################################"
}

# pulisci_known_hosts rimuove le chiavi host memorizzate per gli indirizzi IP
# attualmente assegnati ai worker.
#
# GCP riassegna gli indirizzi pubblici fra un cluster e l'altro: un IP usato
# ieri da un nodo x86 puo' essere assegnato oggi a un nodo ARM. La chiave host
# e' diversa, e ssh rifiuta la connessione con
# "REMOTE HOST IDENTIFICATION HAS CHANGED" perche' non puo' distinguere questo
# caso da un attacco.
#
# Il problema si manifesta nella fase di raccolta, che usa scp: la copia
# fallisce e i campioni restano sulle VM. Se a quel punto il cluster viene
# distrutto, i dati della campagna sono persi.
pulisci_known_hosts() {
    local h ip

    for h in "${WORKERS[@]}"; do
        ip=$(gcloud compute instances describe "$h" --zone="$ZONE" \
             --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || true)

        [ -z "$ip" ] && continue

        ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$ip" >/dev/null 2>&1 || true
        echo "  $h ($ip)"
    done
}

banner "1/6  CREAZIONE VM — $NODI nodi $ARCH, on-demand"

./gcp_up.sh

echo
echo "Pulizia delle chiavi host per gli indirizzi appena assegnati:"
pulisci_known_hosts

banner "2/6  AVVIO CLUSTER con profiling"

PROFILING=1 ./gcp_start_cluster.sh RoundRobin

banner "3/6  IMMAGINE RUNTIME sui worker"

# Le immagini VM congelate non contengono openssl, aggiunto al Dockerfile dopo
# la loro creazione: senza, chacha20 fallisce.
for h in "${WORKERS[@]}"; do
    echo -n "  $h: "
    gcloud compute ssh "$h" --zone="$ZONE" --quiet --command='
        cd /opt/serverledge && sudo git pull --quiet
        sudo docker build -q -t grussorusso/serverledge-go125 \
            -f images/go125/Dockerfile . >/dev/null
        sudo docker run --rm grussorusso/serverledge-go125 openssl version
    ' </dev/null
done

banner "4/6  RIAVVIO CLUSTER dopo la ricostruzione delle immagini"

# Necessario: ricostruire le immagini mentre i worker girano lascia la
# contabilita' delle risorse disallineata rispetto ai container effettivi.
PROFILING=1 ./gcp_start_cluster.sh RoundRobin

# I campioni di eventuali sessioni precedenti vanno rimossi, altrimenti si
# mescolerebbero a quelli della campagna.
for h in "${WORKERS[@]}"; do
    gcloud compute ssh "$h" --zone="$ZONE" --quiet \
        --command='sudo rm -f /var/lib/serverledge/profiling-samples.jsonl' </dev/null
done

banner "5/6  ESPERIMENTO — $DURATA, $UTENTI utenti"

echo "Lo script si fermera' dopo la verifica delle trenta funzioni."
echo "Controlla che siano tutte OK prima di premere INVIO."
echo

USERS="$UTENTI" SPAWN_RATE="$UTENTI" ./gcp_run_experiment.sh RoundRobin "$DURATA"

banner "6/6  RACCOLTA E AGGREGAZIONE"

echo "Pulizia delle chiavi host prima della raccolta:"
pulisci_known_hosts

cd ../..

for h in "${WORKERS[@]}"; do
    gcloud compute ssh "$h" --zone="$ZONE" --quiet \
        --command='sudo chmod a+r /var/lib/serverledge/profiling-samples.jsonl' </dev/null
done

# Il manifest non puo' essere statico: i node_name sono generati dai worker a
# ogni avvio, quindi vanno letti dai campioni appena raccolti.
: > scripts/profiling-nodes.local.conf

for h in "${WORKERS[@]}"; do
    ip=$(gcloud compute instances describe "$h" --zone="$ZONE" \
         --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

    info=$(gcloud compute ssh "$h" --zone="$ZONE" --quiet --command="
        head -1 /var/lib/serverledge/profiling-samples.jsonl \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"node_name\"], d[\"machine_tag\"])'
    " </dev/null 2>/dev/null || true)

    if [ -z "$info" ]; then
        echo "  $h: nessun campione, escluso dal manifest"
        continue
    fi

    node=$(echo "$info" | cut -d' ' -f1)
    tag=$(echo "$info" | cut -d' ' -f2)

    echo "${node}|${tag}|ale@${ip}|/var/lib/serverledge/profiling-samples.jsonl|22" \
        >> scripts/profiling-nodes.local.conf

    echo "  $h -> $node ($tag)"
done

EXPERIMENT="campagna-${ARCH}-$(date +%Y%m%d)"

EXPECTED_SCHEMA_VERSION=4 bash scripts/collect-profiling-datasets.sh \
    --manifest scripts/profiling-nodes.local.conf \
    --experiment "$EXPERIMENT"

bin/serverledge-profiling aggregate \
    --input-dir "data/profiling/raw/${EXPERIMENT}" \
    --output "data/profiling/raw/${EXPERIMENT}/function-profiles.jsonl" \
    --samples 10

bin/serverledge-profiling export-csv \
    --input "data/profiling/raw/${EXPERIMENT}/function-profiles.jsonl" \
    --experiment-id "$EXPERIMENT"

# La raccolta e' l'unico momento in cui i dati passano dalle VM effimere al
# disco locale: se fallisce, distruggere il cluster significa perdere la
# campagna. Meglio interrompere con un errore esplicito.
if [ ! -f "data/profiling/raw/${EXPERIMENT}/function-profiles.jsonl" ]; then
    echo
    echo "ERRORE: l'aggregazione non ha prodotto il file dei profili."
    echo "I campioni sono ancora sulle VM: NON distruggere il cluster."
    exit 1
fi

banner "FATTO — $EXPERIMENT"

cat <<EOF
Dati in: data/profiling/raw/${EXPERIMENT}

Verifica sopra quanti profili sono stati costruiti e quanti saltati.
I campioni sono ora in locale: puoi distruggere il cluster.

    cd scripts/gcp_scripts && ./gcp_down.sh
EOF
