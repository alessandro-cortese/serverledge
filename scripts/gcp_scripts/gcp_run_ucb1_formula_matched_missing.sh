#!/usr/bin/env bash

set -euo pipefail


# ==============================================================================
# Paths and experiment constants
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MEASURE_RUNNER="${SCRIPT_DIR}/gcp_measure_ucb1_formula_matched.sh"
GCP_UP="${SCRIPT_DIR}/gcp_up.sh"
GCP_DOWN="${SCRIPT_DIR}/gcp_down.sh"

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

FROZEN_RUNTIME_ROOT="${HOME}/Documenti/GitHub/serverledge-ucb1-final-runtime"

FROZEN_RUNTIME_COMMIT="4c9f427a76e188c9cb026d0b65ccf44b4d116d6f"
FROZEN_BASE_COMMIT="4a7359156af6ad3d83841c2935efebd67905cb21"

FROZEN_DEPLOY_SCRIPT="${FROZEN_RUNTIME_ROOT}/scripts/gcp_scripts/gcp_deploy_exploration_ablation.sh"

LOG_DIR="${ROOT_DIR}/data/profiling/gcp-ucb1-formula-matched/campaign-logs"

mkdir -p "$LOG_DIR"

CAMPAIGN_ID="formula-matched-missing-$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${CAMPAIGN_ID}.log"

FROZEN_BUNDLE="${LOG_DIR}/${CAMPAIGN_ID}-runtime.bundle"
REMOTE_BUNDLE="/tmp/serverledge-ucb1-final-runtime.bundle"

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

CLOUD_TOUCHED=0


VMS=(
    "sl-registry"
    "sl-lb"
    "sl-workload"
    "sl-x86-1"
    "sl-arm-1"
)


BUILD_ARTIFACTS=(
    "build/gcp-exploration-ablation/amd64/lb"
    "build/gcp-exploration-ablation/amd64/serverledge"
    "build/gcp-exploration-ablation/arm64/serverledge"
)


# ==============================================================================
# Cleanup
# ==============================================================================

cleanup() {
    local status=$?
    local down_status=0

    trap - EXIT INT TERM

    echo
    echo "============================================================"
    echo "FORMULA-MATCHED CAMPAIGN CLEANUP"
    echo "============================================================"

    if (( CLOUD_TOUCHED == 1 )); then

        echo "Distruzione risorse GCP..."

        set +e

        "$GCP_DOWN"
        down_status=$?

        set -e

        if (( down_status != 0 )); then

            echo \
                "WARNING: gcp_down.sh exit=${down_status}" \
                >&2

            if (( status == 0 )); then
                status="$down_status"
            fi
        fi

    else

        echo "Nessuna creazione GCP avviata."

    fi


    # Ripristina solo i binari generati dal deploy.
    # Nessun sorgente viene modificato.
    if [[ -d "$FROZEN_RUNTIME_ROOT" ]]; then

        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            restore \
            -- \
            "${BUILD_ARTIFACTS[@]}" \
            >/dev/null 2>&1 \
            || true

    fi


    echo
    echo "campaign_log=$LOG"

    if [[ -f "$FROZEN_BUNDLE" ]]; then
        echo "runtime_bundle=$FROZEN_BUNDLE"
    fi

    exit "$status"
}


trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# Log dell'intera campagna.
exec > >(tee "$LOG") 2>&1


# ==============================================================================
# Utility
# ==============================================================================

fail() {
    echo "FATAL: $*" >&2
    exit 1
}


remote() {
    local vm="$1"
    shift

    gcloud compute ssh \
        "$vm" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        --command="$*"
}


copy_to() {
    local source="$1"
    local vm="$2"
    local destination="$3"

    gcloud compute scp \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet \
        "$source" \
        "${vm}:${destination}"
}


# ==============================================================================
# Frozen worktree local gate
# ==============================================================================

prepare_frozen_runtime() {

    echo
    echo "============================================================"
    echo "VERIFY FROZEN LOCAL RUNTIME"
    echo "============================================================"

    [[ -d "$FROZEN_RUNTIME_ROOT" ]] \
        || fail \
            "runtime worktree non trovato: $FROZEN_RUNTIME_ROOT"

    [[ -x "$FROZEN_DEPLOY_SCRIPT" ]] \
        || fail \
            "deploy script non eseguibile: $FROZEN_DEPLOY_SCRIPT"


    # Il deploy modifica questi tre binari.
    # Li riportiamo alla versione registrata prima del gate di pulizia.
    git \
        -C "$FROZEN_RUNTIME_ROOT" \
        restore \
        -- \
        "${BUILD_ARTIFACTS[@]}"


    local runtime_head

    runtime_head="$(
        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            rev-parse HEAD
    )"


    echo "runtime_root=$FROZEN_RUNTIME_ROOT"
    echo "runtime_head=$runtime_head"
    echo "expected=$FROZEN_RUNTIME_COMMIT"


    [[ "$runtime_head" == "$FROZEN_RUNTIME_COMMIT" ]] \
        || fail \
            "frozen runtime HEAD inatteso"


    local parent

    parent="$(
        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            rev-parse \
            "${FROZEN_RUNTIME_COMMIT}^"
    )"


    echo "runtime_parent=$parent"
    echo "expected_parent=$FROZEN_BASE_COMMIT"


    [[ "$parent" == "$FROZEN_BASE_COMMIT" ]] \
        || fail \
            "parent del runtime finale inatteso"


    local dirty

    dirty="$(
        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            status --short
    )"


    if [[ -n "$dirty" ]]; then

        echo "$dirty" >&2

        fail \
            "frozen runtime worktree non pulito"

    fi


    echo "FROZEN LOCAL RUNTIME: PASS"
}


# ==============================================================================
# Build tiny incremental Git bundle
# ==============================================================================

prepare_incremental_bundle() {

    echo
    echo "============================================================"
    echo "PREPARE FROZEN RUNTIME BUNDLE"
    echo "============================================================"


    # Aggiorna solo le remote refs locali.
    git \
        -C "$FROZEN_RUNTIME_ROOT" \
        fetch \
        --quiet \
        origin \
        main


    # Il commit base deve essere disponibile sul main remoto:
    # le VM lo riceveranno tramite "fetch origin main".
    git \
        -C "$FROZEN_RUNTIME_ROOT" \
        merge-base \
        --is-ancestor \
        "$FROZEN_BASE_COMMIT" \
        origin/main \
        || fail \
            "base commit non raggiungibile da origin/main"


    local unique_count

    unique_count="$(
        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            rev-list \
            --count \
            "$FROZEN_RUNTIME_COMMIT" \
            "^${FROZEN_BASE_COMMIT}"
    )"


    echo "unique_frozen_commits=$unique_count"


    [[ "$unique_count" == "1" ]] \
        || fail \
            "il runtime finale non è più un singolo commit sopra il base"


    rm -f "$FROZEN_BUNDLE"


    (
        cd "$FROZEN_RUNTIME_ROOT"

        git bundle create \
            "$FROZEN_BUNDLE" \
            HEAD \
            "^${FROZEN_BASE_COMMIT}"
    )


    git \
        -C "$FROZEN_RUNTIME_ROOT" \
        bundle verify \
        "$FROZEN_BUNDLE"


    local bundle_head

    bundle_head="$(
        git \
            -C "$FROZEN_RUNTIME_ROOT" \
            bundle list-heads \
            "$FROZEN_BUNDLE" \
            | awk '$2 == "HEAD" {print $1}'
    )"


    [[ "$bundle_head" == "$FROZEN_RUNTIME_COMMIT" ]] \
        || fail \
            "bundle HEAD mismatch"


    local bundle_size

    bundle_size="$(
        stat \
            -c '%s' \
            "$FROZEN_BUNDLE"
    )"


    # Il bundle atteso è di poche centinaia di byte.
    # 1 MiB è un limite volutamente molto conservativo.
    if (( bundle_size > 1048576 )); then

        fail \
            "bundle inatteso troppo grande: ${bundle_size} byte"

    fi


    FROZEN_BUNDLE_SHA="$(
        sha256sum \
            "$FROZEN_BUNDLE" \
            | awk '{print $1}'
    )"


    echo "bundle=$FROZEN_BUNDLE"
    echo "bundle_size_bytes=$bundle_size"
    echo "bundle_sha256=$FROZEN_BUNDLE_SHA"
    echo "bundle_head=$bundle_head"

    echo
    echo "FROZEN RUNTIME BUNDLE: PASS"
}


# ==============================================================================
# Verify that campaign VMs do not already exist
# ==============================================================================

verify_no_existing_vms() {

    echo
    echo "============================================================"
    echo "VERIFY NO EXISTING CAMPAIGN VMS"
    echo "============================================================"

    local vm
    local status

    for vm in "${VMS[@]}"; do

        status="$(
            gcloud compute instances describe \
                "$vm" \
                --project="$PROJECT" \
                --zone="$ZONE" \
                --format='value(status)' \
                2>/dev/null \
                || true
        )"


        if [[ -n "$status" ]]; then

            fail \
                "VM già esistente: $vm status=$status"

        fi


        echo "  $vm: absent"

    done


    echo "NO EXISTING CAMPAIGN VMS: PASS"
}


# ==============================================================================
# Seed the unpublished frozen commit onto all VMs
# ==============================================================================

seed_frozen_commit() {

    echo
    echo "============================================================"
    echo "SEED FROZEN RUNTIME COMMIT ON VMS"
    echo "============================================================"

    local vm


    for vm in "${VMS[@]}"; do

        echo
        echo "----- $vm -----"


        # ----------------------------------------------------------------------
        # First update origin/main.
        # This makes the prerequisite 4a735... available locally.
        # ----------------------------------------------------------------------

        remote "$vm" "
            set -euo pipefail

            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                fetch \
                --quiet \
                origin \
                main

            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                cat-file \
                -e \
                '${FROZEN_BASE_COMMIT}^{commit}'
        "


        echo "  base commit: PASS"


        # ----------------------------------------------------------------------
        # Copy the tiny incremental bundle.
        # ----------------------------------------------------------------------

        copy_to \
            "$FROZEN_BUNDLE" \
            "$vm" \
            "$REMOTE_BUNDLE"


        # ----------------------------------------------------------------------
        # Verify bundle bytes and import the frozen commit into the VM Git
        # object database.
        # ----------------------------------------------------------------------

        remote "$vm" "
            set -euo pipefail

            echo \
                '${FROZEN_BUNDLE_SHA}  ${REMOTE_BUNDLE}' \
                | sha256sum -c -

            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                fetch \
                --quiet \
                '${REMOTE_BUNDLE}' \
                HEAD

            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                cat-file \
                -e \
                '${FROZEN_RUNTIME_COMMIT}^{commit}'

            rm -f \
                '${REMOTE_BUNDLE}'
        "


        echo "  frozen commit: PASS"

    done


    echo
    echo "SEED FROZEN RUNTIME COMMIT: PASS"
}


# ==============================================================================
# One online measurement
# ==============================================================================

run_measurement() {

    local family="$1"
    local target="$2"
    local condition="$3"


    echo
    echo "============================================================"
    echo "FORMULA-MATCHED RUN"
    echo "============================================================"

    echo "family=$family"
    echo "target=$target"
    echo "condition=$condition"
    echo "runtime=$FROZEN_RUNTIME_COMMIT"


    # These are NOT contemporaneous paired runs with their historical
    # counterpart. They are the missing cells required to reconstruct the
    # formula-matched B and C comparisons.
    EXPECTED_COMMIT="$FROZEN_RUNTIME_COMMIT" \
    MEASURE_REQUESTS=50 \
    EXPERIMENT_BLOCK="formula-matched-missing-${family}-${target}" \
    ORDER_POSITION=0 \
    "$MEASURE_RUNNER" \
        "$family" \
        "$target" \
        r01 \
        "$condition"


    echo
    echo "RUN PASS: ${family}/${target}/${condition}"
}


# ==============================================================================
# Main
# ==============================================================================

main() {

    cd "$ROOT_DIR"


    # --------------------------------------------------------------------------
    # Local script gates
    # --------------------------------------------------------------------------

    [[ -x "$MEASURE_RUNNER" ]] \
        || fail \
            "runner non eseguibile: $MEASURE_RUNNER"

    [[ -x "$GCP_UP" ]] \
        || fail \
            "gcp_up.sh non eseguibile: $GCP_UP"

    [[ -x "$GCP_DOWN" ]] \
        || fail \
            "gcp_down.sh non eseguibile: $GCP_DOWN"


    prepare_frozen_runtime

    prepare_incremental_bundle

    verify_no_existing_vms


    echo
    echo "============================================================"
    echo "GCP UCB1 FORMULA-MATCHED — MISSING CONDITIONS"
    echo "============================================================"

    echo "campaign=$CAMPAIGN_ID"
    echo "runtime_root=$FROZEN_RUNTIME_ROOT"
    echo "runtime=$FROZEN_RUNTIME_COMMIT"
    echo "runtime_parent=$FROZEN_BASE_COMMIT"
    echo "bundle_sha256=$FROZEN_BUNDLE_SHA"
    echo "requests_per_run=50"
    echo "new_runs=8"

    echo
    echo "Family B:"
    echo "  missing formula = Ratio"
    echo "  w_R=0.05"
    echo "  w_E=64"
    echo "  c=0.2"

    echo
    echo "Family C:"
    echo "  missing formula = Difference"
    echo "  w_R=0.01"
    echo "  w_E=64"
    echo "  c=0.4"


    # --------------------------------------------------------------------------
    # Local-only preflight
    # --------------------------------------------------------------------------

    if [[ "$PREFLIGHT_ONLY" == "1" ]]; then

        echo
        echo "============================================================"
        echo "FORMULA-MATCHED CAMPAIGN PREFLIGHT: PASS"
        echo "============================================================"

        echo "Nessuna VM creata."
        echo "Nessuna misura eseguita."

        exit 0

    fi


    # ==========================================================================
    # Create five GCP VMs once
    # ==========================================================================

    echo
    echo "============================================================"
    echo "CREATE GCP INSTANCES"
    echo "============================================================"

    # Set BEFORE creation so a partial gcp_up failure also triggers cleanup.
    CLOUD_TOUCHED=1


    (
        cd "$SCRIPT_DIR"

        N_X86=1 \
        N_ARM=1 \
        "$GCP_UP"
    )


    # ==========================================================================
    # Make unpublished frozen commit available to every VM
    # ==========================================================================

    seed_frozen_commit


    # ==========================================================================
    # Deploy exactly the frozen runtime
    #
    # The original deployment script will:
    #
    # - build from 4c9f427;
    # - fetch origin/main;
    # - checkout 4c9f427 on all VMs;
    # - deploy amd64/arm64 binaries;
    # - run FINAL COMMIT GATE;
    # - run BINARY SHA256 GATE.
    #
    # It can now checkout 4c9f427 because seed_frozen_commit imported the
    # missing Git object into every VM.
    # ==========================================================================

    echo
    echo "============================================================"
    echo "DEPLOY FROZEN FINAL RUNTIME"
    echo "============================================================"

    echo "source=$FROZEN_RUNTIME_ROOT"
    echo "commit=$FROZEN_RUNTIME_COMMIT"


    (
        cd "$FROZEN_RUNTIME_ROOT"

        ABLATION_RUNTIME_COMMIT="$FROZEN_RUNTIME_COMMIT" \
        "$FROZEN_DEPLOY_SCRIPT"
    )


    # ==========================================================================
    # Eight missing measurements
    #
    # B:
    #   existing historical cell = Difference
    #   new missing cell          = Ratio
    #
    # C:
    #   existing historical cell = Ratio
    #   new missing cell          = Difference
    #
    # We alternate formulas across the campaign to avoid always measuring the
    # same formula first, but ORDER_POSITION remains 0 because these new runs
    # are not a fresh contemporaneous pair with the historical counterpart.
    # ==========================================================================

    run_measurement \
        B \
        billibench-knn \
        ratio

    run_measurement \
        C \
        billibench-knn \
        difference


    run_measurement \
        C \
        dynamic-html-py \
        difference

    run_measurement \
        B \
        dynamic-html-py \
        ratio


    run_measurement \
        B \
        functionbench-gzip \
        ratio

    run_measurement \
        C \
        functionbench-gzip \
        difference


    run_measurement \
        C \
        graph-bfs \
        difference

    run_measurement \
        B \
        graph-bfs \
        ratio


    echo
    echo "============================================================"
    echo "ALL 8 FORMULA-MATCHED RUNS: PASS"
    echo "============================================================"
}


main