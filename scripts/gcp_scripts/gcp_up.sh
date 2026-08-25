#!/usr/bin/env bash
#
# Crea il cluster sperimentale su GCP.
#
#     ./gcp-up.sh
#     N_X86=1 N_ARM=1 ./gcp-up.sh     # cluster ridotto per la validazione
#     SPOT=1 ./gcp-up.sh              # quando la quota spot sarà approvata
#
# Le regole firewall vengono create solo se non esistono già, quindi lo script
# è ripetibile. Al termine stampa la tabella degli indirizzi IP interni, che
# sono quelli da usare nei file di configurazione di Serverledge: restare sugli
# indirizzi interni evita di esporre etcd su internet e non genera traffico a
# pagamento.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

banner "CLUSTER SERVERLEDGE — CREAZIONE"

echo "progetto:      $PROJECT"
echo "zona:          $ZONE"
echo "nodi x86:      $N_X86 ($MT_X86)"
echo "nodi ARM:      $N_ARM ($MT_ARM)"
echo "provisioning:  $([[ "$SPOT" == "1" ]] && echo SPOT || echo on-demand)"

if image_exists "$IMAGE_X86"; then
    echo "immagine x86:  $IMAGE_X86 (personalizzata)"
else
    echo "immagine x86:  $BASE_IMAGE_X86_FAMILY (pubblica, setup da fare)"
fi

if image_exists "$IMAGE_ARM"; then
    echo "immagine ARM:  $IMAGE_ARM (personalizzata)"
else
    echo "immagine ARM:  $BASE_IMAGE_ARM_FAMILY (pubblica, setup da fare)"
fi

# --- Firewall ----------------------------------------------------------------

banner "REGOLE FIREWALL"

MY_IP="$(curl -s --max-time 10 ifconfig.me || true)"

if [[ -z "$MY_IP" ]]; then
    echo "Impossibile determinare l'IP pubblico locale."
    echo "Imposta MY_IP a mano ed esegui di nuovo."
    exit 1
fi

echo "IP locale rilevato: $MY_IP"

create_rule() {
    local name="$1"
    shift

    if gc compute firewall-rules describe "$name" >/dev/null 2>&1; then
        echo "  $name già presente"
        return
    fi

    gc compute firewall-rules create "$name" "$@" >/dev/null
    echo "  $name creata"
}

# Il traffico fra i nodi del cluster resta interno alla VPC.
create_rule serverledge-internal \
    --allow=tcp,udp,icmp \
    --source-ranges=10.128.0.0/9 \
    --description="Traffico interno fra i nodi Serverledge"

create_rule serverledge-ssh \
    --allow=tcp:22 \
    --source-ranges="${MY_IP}/32" \
    --description="SSH dal portatile"

# La porta del load balancer serve solo se locust gira da fuori. Il generatore
# di workload sta dentro la zona, quindi questa regola è una comodità per le
# prove manuali con curl.
create_rule serverledge-lb \
    --allow=tcp:1323 \
    --source-ranges="${MY_IP}/32" \
    --description="Load balancer dal portatile"

# --- Istanze -----------------------------------------------------------------

banner "CREAZIONE ISTANZE"

create_instance() {
    local name="$1"
    local machine_type="$2"
    local arch="$3"

    if gc compute instances describe "$name" --zone="$ZONE" >/dev/null 2>&1; then
        echo "  $name già presente, salto"
        return
    fi

    local image_flags
    if [[ "$arch" == "arm" ]]; then
        image_flags="$(arm_image_flags)"
    else
        image_flags="$(x86_image_flags)"
    fi

    # shellcheck disable=SC2086
    gc compute instances create "$name" \
        --zone="$ZONE" \
        --machine-type="$machine_type" \
        $image_flags \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-auto-delete \
        --labels=progetto=serverledge-tesi \
        "${PROVISIONING_FLAGS[@]}" \
        >/dev/null

    echo "  $name creata ($machine_type)"
}

create_instance "$NAME_REGISTRY" "$MT_REGISTRY" x86
create_instance "$NAME_LB"       "$MT_LB"       x86
create_instance "$NAME_WORKLOAD" "$MT_WORKLOAD" x86

for name in $(x86_node_names); do
    create_instance "$name" "$MT_X86" x86
done

for name in $(arm_node_names); do
    create_instance "$name" "$MT_ARM" arm
done

# --- Riepilogo ---------------------------------------------------------------

banner "INDIRIZZI"

gc compute instances list \
    --filter="labels.progetto=serverledge-tesi" \
    --format="table(name,machineType.basename(),status,networkInterfaces[0].networkIP:label=IP_INTERNO,networkInterfaces[0].accessConfigs[0].natIP:label=IP_PUBBLICO)"

banner "PROSSIMI PASSI"

cat <<EOF
Verifica che i nodi ARM siano davvero ARM:

    gcloud compute ssh sl-arm-1 --zone=$ZONE --command="uname -m"     # aarch64

Nei file di configurazione usa l'IP INTERNO del registry come indirizzo etcd.

Quando hai finito, libera tutto con:

    ./gcp-down.sh

Il costo scatta da adesso: il cluster acceso costa circa 1,70 \$/h nella
configurazione completa a nove macchine.
EOF
