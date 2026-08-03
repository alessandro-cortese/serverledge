#!/usr/bin/env bash
set -euo pipefail

BASE_SCRIPT="/tmp/mab_decision_tracking_local_test.sh"
TEST_SCRIPT="/tmp/mab_fallback_penalty_local_test.sh"
POLICY="${POLICY:-UCB1}"
PENALTY="${PENALTY:--12.0}"
REQUESTS="${REQUESTS:-16}"
SLEEP_SECONDS="${SLEEP_SECONDS:-4}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
    echo "[FAIL] Non trovo $BASE_SCRIPT"
    echo "       Ricrea prima lo script usato per la prova locale della Modifica 04."
    exit 1
fi

python3 - "$BASE_SCRIPT" "$TEST_SCRIPT" "$POLICY" "$PENALTY" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
policy = sys.argv[3]
penalty = sys.argv[4]

text = source.read_text(encoding="utf-8")

text = re.sub(
    r"(?m)^mab\.policy:\s*.*$",
    f"mab.policy: {policy}",
    text,
    count=1,
)

text = re.sub(
    r"(?m)^mab\.fallback\.penalty:\s*.*\n?",
    "",
    text,
)

text = re.sub(
    rf"(?m)^(mab\.policy:\s*{re.escape(policy)}\s*)$",
    rf"\1\nmab.fallback.penalty: {penalty}",
    text,
    count=1,
)

text = re.sub(
    r'''FINAL_TOTAL="\$\(\s*\(\s*FINAL_DIRECT\s*\+\s*FINAL_FALLBACK\s*\+\s*FINAL_CANCELLED\s*\)\s*\)"''',
    "FINAL_TOTAL=$((FINAL_DIRECT + FINAL_FALLBACK + FINAL_CANCELLED))",
    text,
    flags=re.MULTILINE,
)

target.write_text(text, encoding="utf-8")
target.chmod(0o755)
PY

bash -n "$TEST_SCRIPT"

echo "[test] Policy=$POLICY penalty=$PENALTY requests=$REQUESTS sleep=${SLEEP_SECONDS}s"

set +e
REQUESTS="$REQUESTS" \
SLEEP_SECONDS="$SLEEP_SECONDS" \
"$TEST_SCRIPT"
RUNTIME_STATUS=$?
set -e

LOG_DIR="$(ls -dt logs/mab_decision_local_* 2>/dev/null | head -1 || true)"

if [[ -z "$LOG_DIR" || ! -f "$LOG_DIR/lb.log" ]]; then
    echo "[FAIL] Non trovo il log dell'esecuzione."
    exit 1
fi

LB_LOG="$LOG_DIR/lb.log"

echo
echo "============================================================"
echo "VALIDAZIONE PENALITÀ SINTETICA"
echo "============================================================"

python3 - "$LB_LOG" "$PENALTY" <<'PY'
import math
import re
import sys
from collections import defaultdict

log_path = sys.argv[1]
expected_penalty = float(sys.argv[2])
field_re = re.compile(r"([A-Za-z_]+)=([^\s]*)")


def parse(line):
    return {m.group(1): m.group(2) for m in field_re.finditer(line)}


created = {}
resolved = {}
cancelled = {}
synthetic = defaultdict(list)
in_flight = []
mismatches = 0

with open(log_path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        data = parse(line)
        event = data.get("event")
        request_id = data.get("request_id", "")

        if event == "decision_created" and request_id:
            created[request_id] = data
        elif event == "decision_resolved" and request_id:
            resolved[request_id] = data
        elif event == "decision_cancelled" and request_id:
            cancelled[request_id] = data
        elif event == "synthetic_reward" and request_id:
            synthetic[request_id].append(data)
        elif event == "in_flight_changed":
            in_flight.append(data)
        elif event == "execution_arm_mismatch":
            mismatches += 1

expected_ids = {
    request_id
    for request_id, data in resolved.items()
    if data.get("fallback") == "true"
}
expected_ids.update(
    request_id
    for request_id, data in cancelled.items()
    if data.get("reason") == "no_candidate_after_fallback"
    or data.get("fallback") == "true"
)

direct_ids = {
    request_id
    for request_id, data in resolved.items()
    if data.get("fallback") == "false"
}

failures = []
warnings = []

for request_id in sorted(expected_ids):
    events = synthetic.get(request_id, [])

    if len(events) != 1:
        failures.append(
            f"{request_id}: attesa 1 synthetic_reward, trovate {len(events)}"
        )
        continue

    event = events[0]
    selected_arm = created.get(request_id, {}).get("selected_arm", "")

    if event.get("arm") != selected_arm:
        failures.append(
            f"{request_id}: penalizzata arm={event.get('arm')}, selected_arm={selected_arm}"
        )

    try:
        value = float(event.get("reward", "nan"))
    except ValueError:
        value = math.nan

    if not math.isfinite(value) or not math.isclose(
        value,
        expected_penalty,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        failures.append(
            f"{request_id}: reward={value}, attesa={expected_penalty}"
        )

unexpected_ids = sorted(set(synthetic) - expected_ids)
if unexpected_ids:
    failures.append(
        f"synthetic_reward inattese sui request_id: {unexpected_ids}"
    )

if direct_ids & set(synthetic):
    failures.append(
        "almeno una direct execution ha ricevuto una penalità sintetica"
    )

if not synthetic:
    failures.append("nessuna synthetic_reward osservata")

fallback_ids = {
    request_id
    for request_id, data in resolved.items()
    if data.get("fallback") == "true"
}

cancelled_no_candidate_ids = {
    request_id
    for request_id, data in cancelled.items()
    if data.get("reason") == "no_candidate_after_fallback"
}

if not fallback_ids:
    warnings.append("nessun fallback completato osservato")
if not cancelled_no_candidate_ids:
    warnings.append("nessuna cancellazione no_candidate_after_fallback osservata")

last_total = None
for data in in_flight:
    try:
        last_total = int(data.get("total_in_flight", ""))
    except ValueError:
        pass

if last_total != 0:
    failures.append(f"total_in_flight finale={last_total}, atteso 0")

print(f"decisioni create:                     {len(created)}")
print(f"fallback completati:                  {len(fallback_ids)}")
print(f"cancellazioni no_candidate:           {len(cancelled_no_candidate_ids)}")
print(f"request_id che richiedono penalità:   {len(expected_ids)}")
print(f"synthetic_reward osservate:           {sum(len(v) for v in synthetic.values())}")
print(f"execution_arm_mismatch:               {mismatches}")
print(f"total_in_flight finale:                {last_total}")
print()

for request_id in sorted(synthetic):
    event = synthetic[request_id][0]
    terminal = resolved.get(request_id) or cancelled.get(request_id) or {}
    print(
        "[TRACE] "
        f"request_id={request_id} "
        f"selected_arm={created.get(request_id, {}).get('selected_arm', '')} "
        f"execution_arm={terminal.get('execution_arm', '')} "
        f"fallback={terminal.get('fallback', '')} "
        f"reward={event.get('reward', '')} "
        f"reason={event.get('reason', '')}"
    )

print()

if failures:
    for failure in failures:
        print(f"[FAIL] {failure}")
else:
    print("[PASS] Ogni fallback/fallimento attribuibile riceve una sola penalità.")
    print("[PASS] La penalità è applicata al SelectedArm.")
    print(f"[PASS] Il valore della penalità è {expected_penalty:.6f}.")
    print("[PASS] Le direct execution non ricevono penalità.")
    print("[PASS] total_in_flight finale è 0.")

for warning in warnings:
    print(f"[WARN] {warning}")

sys.exit(1 if failures else 0)
PY

PENALTY_STATUS=$?

echo
echo "[done] Runtime script status: $RUNTIME_STATUS"
echo "[done] Penalty validation status: $PENALTY_STATUS"
echo "[done] Log: $LOG_DIR"

if [[ "$RUNTIME_STATUS" -ne 0 || "$PENALTY_STATUS" -ne 0 ]]; then
    exit 1
fi
