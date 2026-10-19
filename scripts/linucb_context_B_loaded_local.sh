#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENARIO=B \
  "$SCRIPT_DIR/linucb_context_local_run.sh"
