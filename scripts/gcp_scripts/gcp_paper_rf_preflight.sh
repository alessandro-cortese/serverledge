#!/usr/bin/env bash
# Read-only preflight for the Serverledge schema-6/paper-RF GCP campaign.
# Usage:
#   bash scripts/gcp_scripts/gcp_paper_rf_preflight.sh
#   bash scripts/gcp_scripts/gcp_paper_rf_preflight.sh --remote  # once VMs are running
# This script NEVER starts VMs, changes remote Git checkouts, or removes data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/gcp_config.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/functions_catalog.sh"

EXPECTED_SHA="9f9e9cfc5631b41f525f6d91fadad4415b93a8c1"
CHECK_REMOTE=0
case "${1:-}" in
  "") ;;
  --remote) CHECK_REMOTE=1 ;;
  *) echo "Usage: $0 [--remote]" >&2; exit 2 ;;
esac

fail() { echo "ERROR: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

cd "$ROOT_DIR"
actual_sha="$(git rev-parse HEAD)"
[[ "$actual_sha" == "$EXPECTED_SHA" ]] || fail "HEAD=$actual_sha; expected $EXPECTED_SHA. Do not use the historical worktrees."
pass "HEAD pinned to $EXPECTED_SHA"

if ! git diff --quiet HEAD -- . \
  ':(exclude)scripts/gcp_scripts/gcp_paper_rf_preflight.sh'; then
  fail "Frozen repository files have uncommitted changes."
fi
pass "Frozen repository files unchanged"

for required in \
  cmd/profiling/export_paper_rf_csv.go \
  internal/profiling/paper_rf_csv.go \
  internal/profiling/paper_rf_features.go \
  internal/profiling/sample.go \
  examples/experiments/locustfile_collection.py \
  scripts/gcp_scripts/gcp_start_cluster.sh; do
  [[ -f "$required" ]] || fail "Required file absent: $required"
done
pass "Schema-6/paper-RF sources available"

[[ "$(git show HEAD:internal/profiling/sample.go | grep -Ec '^const InvocationSampleSchemaVersion = 6$')" == "1" ]] \
  || fail "InvocationSampleSchemaVersion is not 6"
pass "Raw schema = 6"

[[ "${#FUNCTIONS[@]}" -gt 0 ]] || fail "Empty function catalog"

# Validate exact names, specifications, and referenced sources (never rewrite them).
declare -A seen=()
missing=0
for spec in "${FUNCTIONS[@]}"; do
  IFS='|' read -r name runtime src memory handler <<< "$spec"
  [[ -n "$name" && -n "$runtime" && -n "$src" && "$memory" =~ ^[0-9]+$ ]] \
    || fail "Malformed catalog record: $spec"
  [[ -z "${seen[$name]:-}" ]] || fail "Duplicate function name: $name"
  seen[$name]=1
  if [[ ! -f "examples/experiments/$src" ]]; then
    echo "MISSING SOURCE: $name => examples/experiments/$src" >&2
    missing=$((missing+1))
  fi
done
[[ "$missing" -eq 0 ]] || fail "$missing function source paths missing (may require corpus preparation)"
pass "${#FUNCTIONS[@]} catalog records, unique and with local source files"

# Compare expected Go exporter entrypoint with Locust options.
grep -Fq 'export-paper-rf-csv' cmd/profiling/main.go || fail "paper-RF CLI subcommand absent"
grep -Fq 'MAX_REQUESTS_PER_FUNCTION' examples/experiments/locustfile_collection.py || fail "Locust request cap absent"
pass "Exporter command and bounded Locust invocation supported"

# No build here: this preflight remains fast and doesn't need a Go toolchain.
if [[ "$CHECK_REMOTE" -eq 0 ]]; then
  echo
  echo "LOCAL PREFLIGHT PASS. No GCP resources were accessed."
  echo "For read-only VM state checks (only once VMs are running):"
  echo "  bash scripts/gcp_scripts/gcp_paper_rf_preflight.sh --remote"
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || fail "gcloud not on PATH"
echo
printf '%-18s %-12s %-10s %-42s %s\n' VM STATE UNAME SHA BINARY_SHA256

for vm in "$NAME_REGISTRY" "$NAME_LB" "$NAME_WORKLOAD" sl-x86-1 sl-arm-1; do
  state="$(gc compute instances describe "$vm" --zone="$ZONE" --format='value(status)' 2>/dev/null || true)"
  [[ "$state" == "RUNNING" ]] || fail "$vm is ${state:-MISSING}, expected RUNNING"
  read -r uname vmsha binarysha < <(
    gc compute ssh "$vm" --zone="$ZONE" --quiet \
      --command="printf '%s ' \"\$(uname -m)\"; sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD | tr '\\n' ' '; if [ -f /opt/serverledge/bin/serverledge ]; then sha256sum /opt/serverledge/bin/serverledge | cut -d' ' -f1; elif [ -f /opt/serverledge/bin/lb ]; then sha256sum /opt/serverledge/bin/lb | cut -d' ' -f1; else echo MISSING; fi" \
      | tr -d '\r'
  )
  printf '%-18s %-12s %-10s %-42s %s\n' "$vm" "$state" "$uname" "$vmsha" "$binarysha"
  [[ "$vmsha" == "$EXPECTED_SHA" ]] || fail "$vm has wrong Git checkout ($vmsha); deployment needed"
  [[ "$binarysha" != "MISSING" ]] || echo "WARN: no relevant worker/lb binary found on $vm"
  case "$vm" in
    sl-arm-1) [[ "$uname" == "aarch64" ]] || fail "ARM worker has $uname" ;;
    sl-x86-1) [[ "$uname" == "x86_64" ]] || fail "x86 worker has $uname" ;;
  esac
done
pass "Remote Git SHAs and worker architectures verified; compare binary hashes with a pinned local build before collecting"
