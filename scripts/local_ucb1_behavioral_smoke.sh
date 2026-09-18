#!/usr/bin/env bash
set -euo pipefail

ROOT="${SERVERLEDGE_ROOT:-$HOME/Documenti/GitHub/serverledge}"
cd "$ROOT"

echo "=== Serverledge local UCB1 behavioral smoke ==="
echo "repo: $ROOT"
echo

SERVERLEDGE_LOCAL_UCB_SMOKE=1 \
go test ./internal/mab \
  -count=1 \
  -v \
  -run '^TestLocalUCB1BehavioralSmoke$'
