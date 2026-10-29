#!/usr/bin/env bash
# Imposta limite MASSIMO di esecuzione, applicato direttamente da Google Cloud.
# Per modificare scheduling tutte le VM devono essere TERMINATED.
# Lascia i dischi di boot e i raw intatti; NON usa delete.
set -euo pipefail

PROJECT='serverledge-tesi-ale'
ZONE='europe-west4-a'
VM_LIST=(sl-registry sl-lb sl-workload sl-x86-1 sl-arm-1)

usage() {
  printf 'Uso: bash %s --check | --configure 12h\n' "$0" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
mode="$1"
case "$mode" in
  --check) [[ $# -eq 1 ]] || usage ;;
  --configure)
    [[ $# -eq 2 ]] || usage
    duration="$2"
    [[ "$duration" =~ ^([1-9]|[1-9][0-9])h$ ]] || {
      echo 'ERRORE: durata in ore 1h..99h' >&2; exit 2;
    }
    [[ "${PAPER_RF_FAILSAFE_CONFIRM:-}" == YES ]] || {
      echo 'ERRORE: serve PAPER_RF_FAILSAFE_CONFIRM=YES' >&2; exit 2;
    }
    ;;
  *) usage ;;
esac

for vm in "${VM_LIST[@]}"; do
  status="$(gcloud compute instances describe "$vm" --project="$PROJECT" --zone="$ZONE" --format='value(status)' --quiet)"
  echo "$vm: $status"
  if [[ "$mode" == --configure && "$status" != TERMINATED ]]; then
    echo "STOP: $vm non e TERMINATED; nessuna configurazione avviata" >&2
    exit 1
  fi
done

if [[ "$mode" == --configure ]]; then
  for vm in "${VM_LIST[@]}"; do
    echo "Configuro $vm: max-run-duration=$duration, action=STOP"
    gcloud compute instances set-scheduling "$vm" \
      --project="$PROJECT" --zone="$ZONE" --quiet \
      --max-run-duration="$duration" --instance-termination-action=STOP
  done
fi

echo '===== Protezione GCP per VM (controllo read-only) ====='
for vm in "${VM_LIST[@]}"; do
  echo "--- $vm ---"
  gcloud compute instances describe "$vm" --project="$PROJECT" --zone="$ZONE" --quiet \
    --format='json(scheduling.maxRunDuration,scheduling.instanceTerminationAction)'
done
