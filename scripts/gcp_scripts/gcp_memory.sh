#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv-analysis/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    printf 'ERRORE: Python non disponibile o non eseguibile: %s\n' "$PYTHON" >&2
    printf 'Crea/attiva .venv-analysis oppure passa PYTHON=/percorso/python.\n' >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    printf 'Uso: %s <azione> [opzioni]\n' "${0##*/}" >&2
    printf 'Azioni: preflight, init-pilot, init-full, build, provision, deploy, pilot, run, execute, finish, stop, abandon\n' >&2
    printf 'Esempio: bash scripts/gcp_scripts/gcp_memory.sh preflight\n' >&2
    printf 'Le azioni che modificano GCP richiedono --confirm.\n' >&2
    exit 2
fi

cd -- "$ROOT_DIR"
exec "$PYTHON" -m analysis.profiling.gcp_memory_campaign "$@"
