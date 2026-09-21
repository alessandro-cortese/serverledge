#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/gcp_config.sh"

PROJECT="${PROJECT:-serverledge-tesi-ale}"
ZONE="${ZONE:-europe-west4-a}"

ABLATION_RUNTIME_COMMIT="${ABLATION_RUNTIME_COMMIT:-fc8bef0d95129c7bd2728b33f2892609bcc60974}"

NAME_REGISTRY="sl-registry"
NAME_LB="sl-lb"
NAME_WORKLOAD="sl-workload"
X86_WORKER="sl-x86-1"
ARM_WORKER="sl-arm-1"

BUILD_DIR="${ROOT_DIR}/build/gcp-exploration-ablation"

banner() {
    echo
    echo "============================================================"
    echo "$*"
    echo "============================================================"
}

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

remote() {
    local host="$1"
    shift

    gcloud \
        --project="$PROJECT" \
        compute ssh "$host" \
        --zone="$ZONE" \
        --quiet \
        --command="$*"
}

copy_to() {
    local source="$1"
    local host="$2"
    local destination="$3"

    gcloud \
        --project="$PROJECT" \
        compute scp \
        --zone="$ZONE" \
        --quiet \
        "$source" \
        "${host}:${destination}"
}

banner "VERIFY LOCAL ABLATION SOURCE"

LOCAL_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"

echo "expected: $ABLATION_RUNTIME_COMMIT"
echo "local:    $LOCAL_COMMIT"

[[ "$LOCAL_COMMIT" == "$ABLATION_RUNTIME_COMMIT" ]] \
    || fail "HEAD locale diverso dall'ablation commit"

git -C "$ROOT_DIR" diff --quiet \
    || fail "working tree con modifiche tracked non committate"

git -C "$ROOT_DIR" diff --cached --quiet \
    || fail "index con modifiche staged non committate"

echo "local source: PASS"

banner "VERIFY VM STATE"

for vm in \
    "$NAME_REGISTRY" \
    "$NAME_LB" \
    "$NAME_WORKLOAD" \
    "$X86_WORKER" \
    "$ARM_WORKER"
do
    status="$(
        gcloud compute instances describe "$vm" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --format='value(status)' \
            2>/dev/null \
            || true
    )"

    [[ "$status" == "RUNNING" ]] \
        || fail "VM $vm non RUNNING (${status:-missing})"

    echo "  $vm: RUNNING"
done

banner "BUILD RUNTIME BINARIES"

rm -rf "$BUILD_DIR"

mkdir -p \
    "${BUILD_DIR}/amd64" \
    "${BUILD_DIR}/arm64"

(
    cd "$ROOT_DIR"

    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
        go build \
        -o "${BUILD_DIR}/amd64/lb" \
        ./cmd/lb

    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
        go build \
        -o "${BUILD_DIR}/amd64/serverledge" \
        ./cmd/serverledge

    CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
        go build \
        -o "${BUILD_DIR}/arm64/serverledge" \
        ./cmd/serverledge
)

echo
sha256sum \
    "${BUILD_DIR}/amd64/lb" \
    "${BUILD_DIR}/amd64/serverledge" \
    "${BUILD_DIR}/arm64/serverledge"

banner "CHECKOUT ABLATION COMMIT ON VMS"

for vm in \
    "$NAME_REGISTRY" \
    "$NAME_LB" \
    "$NAME_WORKLOAD" \
    "$X86_WORKER" \
    "$ARM_WORKER"
do
    echo "  checkout $vm"

    remote "$vm" "
        set -euo pipefail

        sudo git \
            -c safe.directory=/opt/serverledge \
            -C /opt/serverledge \
            fetch --quiet origin main

        sudo git \
            -c safe.directory=/opt/serverledge \
            -C /opt/serverledge \
            checkout \
            --detach \
            --force \
            '${ABLATION_RUNTIME_COMMIT}'

        actual=\$(
            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                rev-parse HEAD
        )

        [[ \"\$actual\" == '${ABLATION_RUNTIME_COMMIT}' ]]
    "
done

banner "DEPLOY LOAD BALANCER"

remote "$NAME_LB" \
    "sudo pkill -x lb >/dev/null 2>&1 || true"

copy_to \
    "${BUILD_DIR}/amd64/lb" \
    "$NAME_LB" \
    "/tmp/serverledge-ablation-lb"

remote "$NAME_LB" "
    sudo install \
        -m 0755 \
        /tmp/serverledge-ablation-lb \
        /opt/serverledge/bin/lb

    rm -f /tmp/serverledge-ablation-lb
"

banner "DEPLOY X86 WORKER"

remote "$X86_WORKER" \
    "sudo pkill -x serverledge >/dev/null 2>&1 || true"

copy_to \
    "${BUILD_DIR}/amd64/serverledge" \
    "$X86_WORKER" \
    "/tmp/serverledge-ablation-worker"

remote "$X86_WORKER" "
    sudo install \
        -m 0755 \
        /tmp/serverledge-ablation-worker \
        /opt/serverledge/bin/serverledge

    rm -f /tmp/serverledge-ablation-worker
"

banner "DEPLOY ARM64 WORKER"

remote "$ARM_WORKER" \
    "sudo pkill -x serverledge >/dev/null 2>&1 || true"

copy_to \
    "${BUILD_DIR}/arm64/serverledge" \
    "$ARM_WORKER" \
    "/tmp/serverledge-ablation-worker"

remote "$ARM_WORKER" "
    sudo install \
        -m 0755 \
        /tmp/serverledge-ablation-worker \
        /opt/serverledge/bin/serverledge

    rm -f /tmp/serverledge-ablation-worker
"

banner "FINAL COMMIT GATE"

for vm in \
    "$NAME_REGISTRY" \
    "$NAME_LB" \
    "$NAME_WORKLOAD" \
    "$X86_WORKER" \
    "$ARM_WORKER"
do
    actual="$(
        remote "$vm" "
            sudo git \
                -c safe.directory=/opt/serverledge \
                -C /opt/serverledge \
                rev-parse HEAD
        " | tr -d '[:space:]'
    )"

    [[ "$actual" == "$ABLATION_RUNTIME_COMMIT" ]] \
        || fail "$vm commit=$actual"

    echo "  $vm: $actual"
done

banner "BINARY SHA256 GATE"

LOCAL_LB_SHA="$(
    sha256sum "${BUILD_DIR}/amd64/lb" |
    awk '{print $1}'
)"

REMOTE_LB_SHA="$(
    remote "$NAME_LB" \
        "sha256sum /opt/serverledge/bin/lb | awk '{print \$1}'" |
    tr -d '[:space:]'
)"

[[ "$LOCAL_LB_SHA" == "$REMOTE_LB_SHA" ]] \
    || fail "LB binary SHA mismatch"

echo "  LB amd64: PASS $REMOTE_LB_SHA"

LOCAL_X86_SHA="$(
    sha256sum "${BUILD_DIR}/amd64/serverledge" |
    awk '{print $1}'
)"

REMOTE_X86_SHA="$(
    remote "$X86_WORKER" \
        "sha256sum /opt/serverledge/bin/serverledge | awk '{print \$1}'" |
    tr -d '[:space:]'
)"

[[ "$LOCAL_X86_SHA" == "$REMOTE_X86_SHA" ]] \
    || fail "x86 worker binary SHA mismatch"

echo "  worker amd64: PASS $REMOTE_X86_SHA"

LOCAL_ARM_SHA="$(
    sha256sum "${BUILD_DIR}/arm64/serverledge" |
    awk '{print $1}'
)"

REMOTE_ARM_SHA="$(
    remote "$ARM_WORKER" \
        "sha256sum /opt/serverledge/bin/serverledge | awk '{print \$1}'" |
    tr -d '[:space:]'
)"

[[ "$LOCAL_ARM_SHA" == "$REMOTE_ARM_SHA" ]] \
    || fail "ARM worker binary SHA mismatch"

echo "  worker arm64: PASS $REMOTE_ARM_SHA"

banner "EXPLORATION ABLATION DEPLOY: PASS"

echo "runtime commit: $ABLATION_RUNTIME_COMMIT"
echo
echo "Il cluster non è stato avviato."
