#!/usr/bin/env bash
# Diagnostica cgroup/Docker sui due worker GCP, senza avviare, fermare o modificare VM.
# Uso: bash verifica_memoria_worker_gcp_sola_lettura.sh [--probe-running]
# Senza flag: SOLO stato delle VM (nessuna connessione SSH).
# --probe-running: sulle VM accese, apre SSH ed esegue SOLTANTO letture.
# Nota: gcloud compute ssh può gestire chiavi SSH nei metadati di accesso.
set -euo pipefail
MODE="${1:-status-only}"
if [[ "$MODE" != 'status-only' && "$MODE" != '--probe-running' ]]; then
  echo "Uso: $0 [--probe-running]"
  exit 2
fi
PROJECT="serverledge-tesi-ale"
ZONE="europe-west4-a"

command -v gcloud >/dev/null || { echo 'ERRORE: gcloud non installato'; exit 1; }

echo "Progetto: $PROJECT | Zona: $ZONE"
echo 'Nessuna VM sarà avviata o fermata. I comandi remoti sono di sola lettura.'
if [[ "$MODE" == 'status-only' ]]; then
  echo 'Modalità: inventario VM, SENZA SSH.'
else
  echo 'Modalità: SSH sui worker RUNNING. gcloud può aggiornare chiavi SSH di accesso.'
fi

for VM in sl-x86-1 sl-arm-1; do
  echo
  echo "==================== $VM ===================="
  # Esegue solo una chiamata di lettura dei metadati. Un risultato vuoto non prova
  # cancellazione se mancano privilegi o connettività: riportiamo l'errore.
  TMPERR="$(mktemp)"
  if ! STATUS="$(gcloud compute instances describe "$VM" \
        --project="$PROJECT" --zone="$ZONE" \
        --format='value(status)' 2>"$TMPERR")"; then
    echo "IMPOSSIBILE LEGGERE lo stato di $VM (potrebbe non esistere oppure mancare accesso):"
    sed -n '1,4p' "$TMPERR"
    rm -f "$TMPERR"
    continue
  fi
  rm -f "$TMPERR"
  echo "Stato: $STATUS"
  if [[ "$STATUS" != RUNNING ]]; then
    echo 'SKIP: worker spento o non avviato. Nessuna accensione automatica.'
    continue
  fi

  if [[ "$MODE" == 'status-only' ]]; then
    echo "Worker acceso: per verificare cgroup/Docker esegui questo script con --probe-running."
    continue
  fi

  # Il comando remoto contiene SOLTANTO letture. Nessun Docker run, alcuna
  # variazione di configurazione, né scrittura sulla VM.
  REMOTE=$(cat <<'SH'
set -u
printf '\n===== ARCHITETTURA / KERNEL =====\n'
uname -m; uname -r
printf '\n===== CGROUP MOUNT =====\n'
findmnt -T /sys/fs/cgroup -o TARGET,FSTYPE,OPTIONS
printf '\n===== CONTROLLERS =====\n'
if [ -r /sys/fs/cgroup/cgroup.controllers ]; then
    cat /sys/fs/cgroup/cgroup.controllers
else
    echo 'Cgroup v1 oppure controller non disponibile alla radice'
fi
printf '\n===== MEMORY COUNTERS DISPONIBILI =====\n'
find /sys/fs/cgroup -maxdepth 5 \( -name memory.current -o -name memory.peak -o -name memory.usage_in_bytes -o -name memory.stat \) 2>/dev/null | head -n 24 || true
printf '\n===== DOCKER INFO (SOLO LETTURA) =====\n'
if sudo -n docker info --format 'Docker: {{.ServerVersion}}; cgroup-version={{.CgroupVersion}}; driver={{.CgroupDriver}}' 2>/dev/null; then
    sudo -n docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}' 2>/dev/null || true
else
    echo 'Docker non consultabile con sudo non interattivo; verificare permessi.'
fi
printf '\n===== ESECUZIONE REMOTA COMPLETATA =====\n'
SH
)
  gcloud compute ssh "$VM" \
    --project="$PROJECT" --zone="$ZONE" --quiet \
    --command="$REMOTE"
done

echo
echo 'FINE: nessuna VM avviata o fermata. Verificare lo stato di accesso SSH se è stata usata la modalità remota.'
