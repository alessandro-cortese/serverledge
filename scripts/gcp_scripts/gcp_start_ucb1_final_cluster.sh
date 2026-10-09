#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-}"

case "$MODE" in
    profiling|measurement)
        ;;
    *)
        echo "Uso: $0 {profiling|measurement}" >&2
        exit 2
        ;;
esac

EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT mancante}"
MAB_UCB1_C="${MAB_UCB1_C:?MAB_UCB1_C mancante}"

N_X86="${N_X86:-1}"
N_ARM="${N_ARM:-1}"

[[ "$N_X86" == "1" ]] || {
    echo "FATAL: N_X86 deve essere 1" >&2
    exit 1
}

[[ "$N_ARM" == "1" ]] || {
    echo "FATAL: N_ARM deve essere 1" >&2
    exit 1
}

case "$MODE" in
    profiling)
        PROFILING=1
        ;;
    measurement)
        PROFILING=0
        ;;
esac

echo
echo "============================================================"
echo "UCB1 FINAL CLUSTER"
echo "============================================================"
echo "mode:            $MODE"
echo "policy:          UCB1Decoupled"
echo "mab.ucb1.c:      $MAB_UCB1_C"
echo "profiling:       $PROFILING"
echo "x86 workers:     $N_X86"
echo "arm64 workers:   $N_ARM"
echo "expected commit: $EXPECTED_COMMIT"

# Reuse the already validated cluster reset/start implementation.
#
# We intentionally invoke "decoupled" also for No-TL experiments:
# baseline means "no prior applied", NOT a different MAB implementation.
(
    cd "$SCRIPT_DIR"

    EXPECTED_COMMIT="$EXPECTED_COMMIT" \
    N_X86="$N_X86" \
    N_ARM="$N_ARM" \
    PROFILING="$PROFILING" \
    MAB_UCB1_C="$MAB_UCB1_C" \
    ./gcp_start_transfer_cluster.sh decoupled
)