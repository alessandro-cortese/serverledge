#!/usr/bin/env bash
# Deployment dedicato alla campagna paper-RF/schema 6 (21 ottobre 2026).
# Non avvia VM/cluster, non modifica dati raw e non tocca i worktree storici.
#
# Modalita':
#   bash scripts/gcp_scripts/gcp_paper_rf_deploy.sh --build       # solo locale
#   bash scripts/gcp_scripts/gcp_paper_rf_deploy.sh --check-vms   # GCP: sola lettura
#   PAPER_RF_DEPLOY_CONFIRM=YES bash scripts/gcp_scripts/gcp_paper_rf_deploy.sh --deploy
#   bash scripts/gcp_scripts/gcp_paper_rf_deploy.sh --verify      # GCP: sola lettura
#
# Le VM devono esistere ed essere RUNNING per --check-vms/--deploy/--verify.
# Il commit su VM e' aggiornato con fetch + checkout detached SENZA --force;
# un checkout con modifiche tracked o processi in esecuzione blocca il deploy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/gcp_config.sh"

PIN="9f9e9cfc5631b41f525f6d91fadad4415b93a8c1"
BUILD_DIR="${ROOT_DIR}/build/gcp-paper-rf-schema6-${PIN:0:12}"
BIN_DIR="${BUILD_DIR}/bin"
DEPLOY_TAG="${PIN:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
VM_LIST=("$NAME_REGISTRY" "$NAME_LB" "$NAME_WORKLOAD" sl-x86-1 sl-arm-1)

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
pass() { printf 'PASS: %s\n' "$*"; }
info() { printf 'INFO: %s\n' "$*"; }
usage() {
    cat <<'EOF_USAGE'
Uso: gcp_paper_rf_deploy.sh {--build|--check-vms|--deploy|--verify}
  --build       Compila localmente 5 binari dal commit Git congelato (nessun GCP)
  --check-vms   Controlla, senza modifiche, cinque VM esistenti e processi attivi
  --deploy      CHECK + checkout pinned + install binari; richiede PAPER_RF_DEPLOY_CONFIRM=YES
  --verify      Controlla commit, architettura e SHA dei binari installati
Non crea, avvia, ferma o cancella VM. Non avvia i servizi serverledge/etcd.
EOF_USAGE
}

[[ $# -eq 1 ]] || { usage; exit 2; }
MODE="$1"
case "$MODE" in
    --build|--check-vms|--deploy|--verify) ;;
    *) usage; exit 2 ;;
esac

command -v git >/dev/null || fail 'git assente'
command -v sha256sum >/dev/null || fail 'sha256sum assente'
[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || fail "Non e' un Git repository: $ROOT_DIR"
[[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" == "$PIN" ]] \
    || fail "HEAD locale non corrisponde al runtime congelato $PIN"
git -C "$ROOT_DIR" cat-file -e "${PIN}^{commit}" || fail 'Commit congelato non disponibile localmente'

# Per l'input di build usiamo SOLO git archive del commit: eventuali modifiche
# non committate agli script di orchestrazione non entrano nella compilazione.
build_local() {
    command -v go >/dev/null || fail 'Go non presente sul PATH locale'
    command -v tar >/dev/null || fail 'tar assente'
    mkdir -p "$BIN_DIR"
    TMP_SRC="$(mktemp -d)"
    trap '[[ -z "${TMP_SRC:-}" ]] || rm -rf -- "$TMP_SRC"' EXIT

    info "Esporto sorgenti Git verificati: $PIN"
    git -C "$ROOT_DIR" archive "$PIN" | tar -x -C "$TMP_SRC"

    info 'Compilo componenti amd64/arm64 (CGO_ENABLED=0, senza modificare i sorgenti)'
    (
        cd "$TMP_SRC"
        CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -buildvcs=false \
            -o "$BIN_DIR/lb-amd64" ./cmd/lb
        CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -buildvcs=false \
            -o "$BIN_DIR/serverledge-amd64" ./cmd/serverledge
        CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -buildvcs=false \
            -o "$BIN_DIR/serverledge-arm64" ./cmd/serverledge
        # I target CLI e profiling seguono gli entrypoint presenti nel Makefile fornito.
        CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -buildvcs=false \
            -o "$BIN_DIR/serverledge-cli-amd64" ./cmd/cli/main.go
        CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -buildvcs=false \
            -o "$BIN_DIR/serverledge-profiling-amd64" ./cmd/profiling
    )

    printf '%s\n' "$PIN" > "$BUILD_DIR/SOURCE_COMMIT"
    go version > "$BUILD_DIR/GO_VERSION"
    (
        cd "$BIN_DIR"
        sha256sum lb-amd64 serverledge-amd64 serverledge-arm64 \
            serverledge-cli-amd64 serverledge-profiling-amd64 > "$BUILD_DIR/SHA256SUMS"
        sha256sum -c "$BUILD_DIR/SHA256SUMS"
    )
    rm -rf -- "$TMP_SRC"
    trap - EXIT
    pass "Build locale completata: $BUILD_DIR"
    info 'Il deployment NON e stato eseguito; nessuna VM GCP coinvolta.'
}

check_build() {
    [[ -f "$BUILD_DIR/SOURCE_COMMIT" ]] || fail 'Build assente: esegui prima --build'
    [[ "$(cat "$BUILD_DIR/SOURCE_COMMIT")" == "$PIN" ]] || fail 'La build non appartiene al commit congelato'
    [[ -f "$BUILD_DIR/SHA256SUMS" ]] || fail 'Manifest SHA256 non presente'
    (cd "$BIN_DIR" && sha256sum -c "$BUILD_DIR/SHA256SUMS" >/dev/null) \
        || fail 'Verifica SHA256 dei binari locali fallita'
    pass 'Binari locali presenti e manifest SHA256 coerente'
}

require_cloud() {
    command -v gcloud >/dev/null || fail 'gcloud non disponibile sul PATH'
}

vm_state() {
    gc compute instances describe "$1" --zone="$ZONE" --format='value(status)' 2>/dev/null || true
}

remote() {
    local vm="$1"; shift
    gc compute ssh "$vm" --zone="$ZONE" --quiet --command="$*"
}

copy_to() {
    local vm="$1" src="$2" dest="$3"
    gc compute scp --zone="$ZONE" --quiet "$src" "${vm}:${dest}"
}

expected_arch() {
    [[ "$1" == sl-arm-1 ]] && echo aarch64 || echo x86_64
}

# Non usa un filtro per label: controlliamo ESATTAMENTE le cinque VM attese.
check_vms() {
    require_cloud
    local vm status actual_arch expected repo_status running
    info "Progetto=$PROJECT zona=$ZONE; controllo VM senza modifiche"
    for vm in "${VM_LIST[@]}"; do
        status="$(vm_state "$vm")"
        [[ "$status" == RUNNING ]] || fail "$vm: status=${status:-NOT_FOUND}; VM non avviata"
        expected="$(expected_arch "$vm")"
        actual_arch="$(remote "$vm" 'uname -m' | tr -d '\r[:space:]')"
        [[ "$actual_arch" == "$expected" ]] || fail "$vm: arch=$actual_arch, attesa=$expected"
        remote "$vm" 'sudo test -d /opt/serverledge/.git' \
            || fail "$vm: checkout /opt/serverledge non presente"
        repo_status="$(remote "$vm" \
            'sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge status --porcelain --untracked-files=no' | tr -d '\r')"
        [[ -z "$repo_status" ]] || fail "$vm: file Git tracciati modificati; non faccio checkout:\n$repo_status"
        # -x confronta il nome del processo: evita falsi match con la shell SSH.
        running="$(remote "$vm" \
            "if pgrep -x serverledge >/dev/null || pgrep -x lb >/dev/null; then echo ACTIVE; else echo idle; fi" \
            | tr -d '\r[:space:]')"
        [[ "$running" == idle ]] || fail "$vm: serverledge/lb in esecuzione; non interferisco con la campagna esistente"
        info "$vm: RUNNING, $actual_arch, Git tracked pulito, nessun serverledge/lb attivo"
    done
    pass 'Gate VM superato (nessuna mutazione remota)'
}

# Installazione singolo artefatto: staging in /tmp, verifica hash PRIMA e DOPO,
# backup della versione precedente sul disco VM, nessuna cancellazione dei raw.
install_binary() {
    local vm="$1" src="$2" dest="$3" expected tmp_path remote_sha
    expected="$(sha256sum "$BIN_DIR/$src" | awk '{print $1}')"
    tmp_path="/tmp/paper-rf-${PIN:0:12}-${src}"
    info "$vm: trasferisco $src"
    copy_to "$vm" "$BIN_DIR/$src" "$tmp_path"
    remote_sha="$(remote "$vm" "sha256sum '$tmp_path' | awk '{print \$1}'" | tr -d '\r[:space:]')"
    [[ "$remote_sha" == "$expected" ]] || fail "$vm: hash staging errato per $src"
    remote "$vm" "
        set -e
        sudo mkdir -p /opt/serverledge/bin/.paper_rf_backups
        if sudo test -e '/opt/serverledge/bin/$dest'; then
            sudo cp -p '/opt/serverledge/bin/$dest' \
                '/opt/serverledge/bin/.paper_rf_backups/${dest}-${DEPLOY_TAG}-before'
        fi
        sudo install -m 0755 '$tmp_path' '/opt/serverledge/bin/$dest'
        rm -f '$tmp_path'
    "
    remote_sha="$(remote "$vm" \
        "sha256sum '/opt/serverledge/bin/$dest' | awk '{print \$1}'" | tr -d '\r[:space:]')"
    [[ "$remote_sha" == "$expected" ]] || fail "$vm: hash binario installato diverso per $dest"
    pass "$vm: bin/$dest = $expected"
}

# Nessun --force né reset, per non distruggere eventuali modifiche tracked.
checkout_pinned() {
    local vm="$1" got
    info "$vm: fetch origin main e checkout detached del commit congelato"
    remote "$vm" "
        set -e
        sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge fetch --quiet origin main
        sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge cat-file -e '${PIN}^{commit}'
        sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge checkout --detach '$PIN'
    "
    got="$(remote "$vm" \
        'sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD' \
        | tr -d '\r[:space:]')"
    [[ "$got" == "$PIN" ]] || fail "$vm: checkout SHA inatteso: $got"
    pass "$vm: checkout=$PIN"
}

deploy() {
    [[ "${PAPER_RF_DEPLOY_CONFIRM:-}" == YES ]] \
        || fail 'Per il deployment usa PAPER_RF_DEPLOY_CONFIRM=YES (operazione remota esplicita)'
    check_build
    check_vms
    info 'Checkouts e installazioni sulla VM: non accendo VM o avvio cluster'
    local vm
    for vm in "${VM_LIST[@]}"; do checkout_pinned "$vm"; done
    install_binary "$NAME_LB" lb-amd64 lb
    install_binary sl-x86-1 serverledge-amd64 serverledge
    install_binary sl-arm-1 serverledge-arm64 serverledge
    install_binary "$NAME_WORKLOAD" serverledge-cli-amd64 serverledge-cli
    install_binary "$NAME_WORKLOAD" serverledge-profiling-amd64 serverledge-profiling
    verify
    pass 'Deployment paper-RF completato: il cluster NON e stato avviato'
}

verify() {
    check_build
    require_cloud
    local vm status sha actual_arch expected
    for vm in "${VM_LIST[@]}"; do
        status="$(vm_state "$vm")"
        [[ "$status" == RUNNING ]] || fail "$vm: non RUNNING"
        actual_arch="$(remote "$vm" 'uname -m' | tr -d '\r[:space:]')"
        expected="$(expected_arch "$vm")"
        [[ "$actual_arch" == "$expected" ]] || fail "$vm: arch=$actual_arch, attesa=$expected"
        sha="$(remote "$vm" \
            'sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD' \
            | tr -d '\r[:space:]')"
        [[ "$sha" == "$PIN" ]] || fail "$vm: SHA Git errato: $sha"
    done
    verify_one() {
        local vm="$1" local_name="$2" dest="$3" expected got
        expected="$(sha256sum "$BIN_DIR/$local_name" | awk '{print $1}')"
        got="$(remote "$vm" "sha256sum '/opt/serverledge/bin/$dest' | awk '{print \$1}'" | tr -d '\r[:space:]')"
        [[ "$got" == "$expected" ]] || fail "$vm bin/$dest: hash diverso, atteso $expected, letto $got"
        pass "$vm: bin/$dest SHA256 corretto"
    }
    verify_one "$NAME_LB" lb-amd64 lb
    verify_one sl-x86-1 serverledge-amd64 serverledge
    verify_one sl-arm-1 serverledge-arm64 serverledge
    verify_one "$NAME_WORKLOAD" serverledge-cli-amd64 serverledge-cli
    verify_one "$NAME_WORKLOAD" serverledge-profiling-amd64 serverledge-profiling
    pass "Tutte le VM al commit $PIN e tutti i binari verificati"
}

case "$MODE" in
    --build) build_local ;;
    --check-vms) check_vms ;;
    --deploy) deploy ;;
    --verify) verify ;;
esac
