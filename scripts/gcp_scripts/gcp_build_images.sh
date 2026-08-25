#!/usr/bin/env bash
#
# Costruisce le due immagini personalizzate (x86 e ARM) con l'ambiente
# Serverledge già pronto.
#
#     ./gcp-build-images.sh
#
# Va eseguito UNA VOLTA SOLA. Da quel momento gcp-up.sh crea le nove macchine
# partendo da queste immagini, e il cluster è operativo in un paio di minuti
# invece che in un'ora e mezza.
#
# È la voce di risparmio più grande dell'intera campagna: rifare Docker, Go,
# clone e build su nove macchine a ogni sessione costerebbe più credito degli
# esperimenti stessi.
#
# Lo script crea due VM temporanee, esegue il setup tramite startup script,
# attende che finisca, spegne le VM, crea le immagini e cancella le VM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gcp_config.sh"

REPO_URL="${REPO_URL:-https://github.com/alessandro-cortese/serverledge.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
GO_VERSION="${GO_VERSION:-1.24.6}"

BUILDER_X86="sl-builder-x86"
BUILDER_ARM="sl-builder-arm"

banner "COSTRUZIONE IMMAGINI"

echo "repository: $REPO_URL ($REPO_BRANCH)"
echo "Go:         $GO_VERSION"
echo
echo "Servono circa 15-20 minuti. Le due VM temporanee sono attive solo"
echo "durante il setup e vengono cancellate al termine."

# --- Startup script ----------------------------------------------------------
#
# GOARCH viene ricavato da uname, così lo stesso script vale per entrambe le
# architetture. L'ultima riga scrive un file sentinella: è il segnale che il
# setup è finito, ed è più affidabile che indovinare un tempo di attesa.

STARTUP="$(mktemp)"
trap 'rm -f "$STARTUP"' EXIT

cat > "$STARTUP" <<EOF
#!/bin/bash
set -euxo pipefail

# Se una qualsiasi istruzione fallisce viene scritta una sentinella di errore:
# senza, chi attende il completamento resta bloccato fino al timeout anche
# quando il setup e' gia' morto dopo pochi secondi.
trap 'touch /var/log/serverledge-setup-failed' ERR

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl git make python3-pip python3-venv

# Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \\
    -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable" \\
    > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Go: l'archivio dipende dall'architettura della macchina.
case "\$(uname -m)" in
    x86_64)  GOARCH=amd64 ;;
    aarch64) GOARCH=arm64 ;;
    *)       echo "architettura non gestita: \$(uname -m)"; exit 1 ;;
esac

curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-\${GOARCH}.tar.gz" \\
    -o /tmp/go.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf /tmp/go.tar.gz
echo 'export PATH=\$PATH:/usr/local/go/bin' > /etc/profile.d/go.sh

# Lo startup script gira come root ma con un ambiente minimo in cui HOME non e'
# impostata. Go ricava GOPATH e GOMODCACHE da \$HOME e senza di essa fallisce
# con "module cache not found: neither GOMODCACHE nor GOPATH is set".
export HOME=/root
export GOPATH=/root/go
export GOMODCACHE=/root/go/pkg/mod
export GOCACHE=/root/.cache/go-build
export PATH=\$PATH:/usr/local/go/bin

mkdir -p "\$GOMODCACHE" "\$GOCACHE"

# Serverledge
git clone --branch "${REPO_BRANCH}" "${REPO_URL}" /opt/serverledge
cd /opt/serverledge
make

# Le immagini runtime vanno costruite sull'architettura di destinazione:
# le basi dei Dockerfile sono multi-arch, quindi su ARM si ottengono immagini
# arm64 con lo stesso tag che runtimes.go si aspetta.
#
# Un fallimento qui non deve interrompere il setup — l'immagine della VM resta
# comunque utile — ma va reso visibile invece di essere ingoiato.
if make images; then
    echo ok > /var/log/serverledge-images-status
else
    echo failed > /var/log/serverledge-images-status
fi

docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' \
    > /var/log/serverledge-images.txt 2>&1 || true

# L'utente che entrerà via SSH deve poter usare Docker senza sudo.
groupadd -f docker

touch /var/log/serverledge-setup-done
EOF

# --- Creazione delle VM temporanee -------------------------------------------

build_one() {
    local name="$1"
    local machine_type="$2"
    local image_family="$3"
    local target_image="$4"

    banner "IMMAGINE: $target_image"

    if image_exists "$target_image"; then
        echo "Immagine già presente, salto. Per rifarla:"
        echo "    gcloud compute images delete $target_image --quiet"
        return
    fi

    gc compute instances create "$name" \
        --zone="$ZONE" \
        --machine-type="$machine_type" \
        --image-family="$image_family" \
        --image-project="$BASE_IMAGE_PROJECT" \
        --boot-disk-size="$DISK_SIZE" \
        --metadata-from-file=startup-script="$STARTUP" \
        --labels=progetto=serverledge-tesi,ruolo=builder \
        >/dev/null

    echo "VM $name creata, attendo il completamento del setup..."

    local waited=0
    local timeout=5400

    while (( waited < timeout )); do

        local state
        state="$(
            gc compute ssh "$name" --zone="$ZONE" --quiet --command="
                if [ -f /var/log/serverledge-setup-done ]; then echo done;
                elif [ -f /var/log/serverledge-setup-failed ]; then echo failed;
                else echo running; fi
            " 2>/dev/null | tr -d '[:space:]' || echo unreachable
        )"

        case "$state" in
            done)
                echo "Setup completato dopo ${waited}s."
                break
                ;;
            failed)
                echo
                echo "Il setup e' fallito. Ultime righe del log:"
                gc compute ssh "$name" --zone="$ZONE" --quiet --command="
                    sudo journalctl -u google-startup-scripts --no-pager | tail -30
                " || true
                echo
                echo "La VM $name resta accesa per l'analisi. Cancellala con:"
                echo "    gcloud compute instances delete $name --zone=$ZONE --quiet"
                exit 1
                ;;
        esac

        sleep 30
        waited=$(( waited + 30 ))
        echo "  ...${waited}s  ($state)"
    done

    if (( waited >= timeout )); then
        echo "Timeout. Controlla il log con:"
        echo "    gcloud compute ssh $name --zone=$ZONE --command='sudo journalctl -u google-startup-scripts'"
        exit 1
    fi

    echo
    echo "Esito della costruzione delle immagini runtime:"
    gc compute ssh "$name" --zone="$ZONE" --quiet --command="
        echo -n '  make images: '
        cat /var/log/serverledge-images-status 2>/dev/null || echo sconosciuto
        echo '  immagini Docker presenti:'
        sed 's/^/    /' /var/log/serverledge-images.txt 2>/dev/null || echo '    (nessuna)'
        echo -n '  commit del repository: '
        git -C /opt/serverledge rev-parse --short HEAD 2>/dev/null || echo sconosciuto
    " || echo "  (lettura non riuscita)"
    echo

    gc compute instances stop "$name" --zone="$ZONE" --quiet

    gc compute images create "$target_image" \
        --source-disk="$name" \
        --source-disk-zone="$ZONE" \
        --family="serverledge" \
        >/dev/null

    echo "Immagine $target_image creata."

    gc compute instances delete "$name" --zone="$ZONE" --quiet
    echo "VM temporanea cancellata."
}

build_one "$BUILDER_X86" "$MT_X86" "$BASE_IMAGE_X86_FAMILY" "$IMAGE_X86"
build_one "$BUILDER_ARM" "$MT_ARM" "$BASE_IMAGE_ARM_FAMILY" "$IMAGE_ARM"

banner "FATTO"

gc compute images list --filter="family=serverledge" \
    --format="table(name,architecture,diskSizeGb,status)"

cat <<EOF

Da adesso ./gcp-up.sh userà queste immagini e il cluster sarà pronto in pochi
minuti. Le immagini restano anche dopo ./gcp-down.sh.

Se aggiorni il codice di Serverledge, rigenera le immagini con:

    gcloud compute images delete $IMAGE_X86 $IMAGE_ARM --quiet
    ./gcp-build-images.sh
EOF