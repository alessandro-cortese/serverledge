#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONCURRENCY="${CONCURRENCY:-2}"
ROUNDS="${ROUNDS:-2}"
CPU_SECONDS="${CPU_SECONDS:-3.0}"
REWARD_ZONE="${REWARD_ZONE:-package}"
KEPLER_URL="${KEPLER_URL:-http://127.0.0.1:28282/metrics}"
API_PORT="${API_PORT:-1323}"
UDP_PORT="${UDP_PORT:-9876}"
POOL_MEMORY_MB="${POOL_MEMORY_MB:-512}"
POOL_CPUS="${POOL_CPUS:-4}"
FUNCTION_MEMORY_MB="${FUNCTION_MEMORY_MB:-128}"
CONTAINER_EXPIRATION="${CONTAINER_EXPIRATION:-60}"
KEPLER_DISCOVERY_TIMEOUT="${KEPLER_DISCOVERY_TIMEOUT:-30}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-kepler-concurrency-isolation-local-${STAMP}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"
FUNCTION_NAME="${FUNCTION_NAME:-kepler_concurrency_${STAMP}}"

mkdir -p "$LOG_DIR/requests"
NODE_CONF="$LOG_DIR/node.yaml"
NODE_LOG="$LOG_DIR/node.log"
ETCD_LOG="$LOG_DIR/etcd.log"
RECORDS_JSONL="$LOG_DIR/records.jsonl"
SUMMARY_JSON="$LOG_DIR/summary.json"
PIDS=()
PREWARM_IDS=()
ETCD_STARTED=0

fail() { echo "[FAIL] $*" >&2; exit 1; }

cleanup() {
  status=$?
  set +e
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -INT "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -KILL "$pid" >/dev/null 2>&1 || true
  done
  for cid in "${PREWARM_IDS[@]:-}"; do
    [[ -n "$cid" ]] && docker rm -f "$cid" >/dev/null 2>&1 || true
  done
  [[ "$ETCD_STARTED" -eq 1 ]] && docker rm -f Etcd-server >/dev/null 2>&1 || true
  echo "[cleanup] Artefatti: $LOG_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1" pid="$2" log="$3"
  for _ in $(seq 1 100); do
    kill -0 "$pid" 2>/dev/null || { tail -100 "$log" || true; fail "processo terminato: $log"; }
    curl -fsS --max-time 1 -o /dev/null "$url" 2>/dev/null && return 0
    sleep 0.25
  done
  tail -100 "$log" || true
  fail "timeout aspettando $url"
}

warm_count() {
  curl -fsS "http://127.0.0.1:${API_PORT}/status" \
    | python3 -c 'import json, sys; data=json.load(sys.stdin); name=sys.argv[1]; print(int((data.get("AvailableWarmContainers") or {}).get(name, 0)))' "$FUNCTION_NAME"
}

for cmd in docker curl python3 make; do
  command -v "$cmd" >/dev/null 2>&1 || fail "comando mancante: $cmd"
done
[[ "$CONCURRENCY" =~ ^[2-9][0-9]*$ ]] || fail "CONCURRENCY deve essere >= 2"
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || fail "ROUNDS deve essere >= 1"
python3 - "$CPU_SECONDS" <<'PY'
import math, sys
v=float(sys.argv[1])
assert math.isfinite(v) and v>0, "CPU_SECONDS deve essere > 0"
PY
(( CONCURRENCY * FUNCTION_MEMORY_MB <= POOL_MEMORY_MB )) || fail "pool memoria insufficiente"
[[ -f examples/kepler_cpu_smoke.py ]] || fail "manca examples/kepler_cpu_smoke.py"

echo "============================================================"
echo "KEPLER CONCURRENCY / CONTAINER ISOLATION LOCAL"
echo "============================================================"
echo "concurrency: $CONCURRENCY"
echo "rounds:      $ROUNDS"
echo "CPU seconds: $CPU_SECONDS"
echo "zone:        $REWARD_ZONE"
echo

echo "[preflight] Kepler..."
curl -fsS --max-time 5 "$KEPLER_URL" > "$LOG_DIR/kepler-preflight.prom" || fail "Kepler non raggiungibile"
grep -q '^kepler_build_info' "$LOG_DIR/kepler-preflight.prom" || fail "kepler_build_info assente"

echo "[setup] Build..."
make >/dev/null
docker rm -f Etcd-server >/dev/null 2>&1 || true

cat > "$NODE_CONF" <<YAML
etcd.address: "127.0.0.1:2379"
api.ip: "127.0.0.1"
api.port: ${API_PORT}
registry.area: KEPLER_CONCURRENCY_${STAMP}
registry.node.id: kepler-concurrency-${STAMP}
registry.udp.port: ${UDP_PORT}
node.machine_tag: x86
container.pool.memory: ${POOL_MEMORY_MB}
container.pool.cpus: ${POOL_CPUS}
container.expiration: ${CONTAINER_EXPIRATION}
profiling.enabled: false
profiling.export.enabled: false
profiling.kepler.enabled: true
profiling.kepler.url: "${KEPLER_URL}"
profiling.kepler.timeout_ms: 3000
profiling.kepler.poll_interval_ms: 100
profiling.kepler.refresh_timeout_ms: 5000
scheduler.queue.capacity: 0
YAML

bash scripts/start-etcd.sh > "$ETCD_LOG" 2>&1
ETCD_STARTED=1
for _ in $(seq 1 30); do
  docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 && break
  sleep 0.5
done
docker exec Etcd-server etcdctl endpoint health >/dev/null 2>&1 || fail "etcd non disponibile"

bin/serverledge "$NODE_CONF" > "$NODE_LOG" 2>&1 &
node_pid=$!
PIDS+=("$node_pid")
wait_http "http://127.0.0.1:${API_PORT}/status" "$node_pid" "$NODE_LOG"

bin/serverledge-cli create \
  -f "$FUNCTION_NAME" \
  --memory "$FUNCTION_MEMORY_MB" \
  --cpu 1 \
  --max_concurrency 1 \
  --src examples/kepler_cpu_smoke.py \
  --runtime python314 \
  --handler "kepler_cpu_smoke.handler" \
  -H 127.0.0.1 -P "$API_PORT" > "$LOG_DIR/create.json"

docker ps --filter ancestor=grussorusso/serverledge-python314 --format '{{.ID}}' | sort > "$LOG_DIR/before.txt"

bin/serverledge-cli prewarm -f "$FUNCTION_NAME" -c "$CONCURRENCY" -H 127.0.0.1 -P "$API_PORT" > "$LOG_DIR/prewarm.json"
sleep 0.5

docker ps --filter ancestor=grussorusso/serverledge-python314 --format '{{.ID}}' | sort > "$LOG_DIR/after.txt"
mapfile -t short_ids < <(comm -13 "$LOG_DIR/before.txt" "$LOG_DIR/after.txt" | sed '/^$/d')
[[ "${#short_ids[@]}" == "$CONCURRENCY" ]] || fail "nuovi container=${#short_ids[@]}, atteso=$CONCURRENCY"
for id in "${short_ids[@]}"; do PREWARM_IDS+=("$(docker inspect --format '{{.Id}}' "$id")"); done
printf '%s\n' "${PREWARM_IDS[@]}" | sort > "$LOG_DIR/prewarm-container-ids.txt"

echo "[setup] Container prewarmed:"
printf '  %s\n' "${PREWARM_IDS[@]}"

# Wait until every container is visible in Kepler with the selected zone.
deadline=$(( $(date +%s) + KEPLER_DISCOVERY_TIMEOUT ))
while true; do
  curl -fsS "$KEPLER_URL" > "$LOG_DIR/kepler-discovery.prom"
  ok=1
  for cid in "${PREWARM_IDS[@]}"; do
    grep '^kepler_container_cpu_joules_total' "$LOG_DIR/kepler-discovery.prom" \
      | grep -F "container_id=\"${cid}\"" \
      | grep -Fq "zone=\"${REWARD_ZONE}\"" || { ok=0; break; }
  done
  [[ "$ok" -eq 1 ]] && break
  (( $(date +%s) < deadline )) || fail "Kepler non vede tutti i container/zone"
  sleep 0.5
done

echo "[setup] Kepler vede tutti i container."
: > "$RECORDS_JSONL"

for round in $(seq 1 "$ROUNDS"); do
  echo
  echo "--- ROUND $round/$ROUNDS ---"

  ROUND_DIR="$LOG_DIR/requests/round-$(printf '%02d' "$round")"
  mkdir -p "$ROUND_DIR"
  invoke_pids=()
  round_start_ns="$(date +%s%N)"

  for slot in $(seq 1 "$CONCURRENCY"); do
    out="$ROUND_DIR/invoke-$(printf '%02d' "$slot").json"
    (
      bin/serverledge-cli invoke \
        -f "$FUNCTION_NAME" \
        -p "seconds:${CPU_SECONDS}" \
        --return_output \
        -H 127.0.0.1 -P "$API_PORT" > "$out" 2>&1
    ) &
    invoke_pids+=("$!")
  done

  for pid in "${invoke_pids[@]}"; do wait "$pid" || fail "invoke concorrente fallita"; done
  round_end_ns="$(date +%s%N)"

  python3 - "$ROUND_DIR" "$round" "$CONCURRENCY" "$REWARD_ZONE" \
    "$LOG_DIR/prewarm-container-ids.txt" "$RECORDS_JSONL" "$round_start_ns" "$round_end_ns" <<'PY'
import json, math, sys
from datetime import datetime
from pathlib import Path

round_dir, rnd, conc, zone, ids_path, records_path, start_ns, end_ns = sys.argv[1:]
rnd=int(rnd); conc=int(conc)
prewarm=set(Path(ids_path).read_text().split())
files=sorted(Path(round_dir).glob('invoke-*.json'))
if len(files)!=conc: raise SystemExit(f"round {rnd}: file={len(files)} atteso={conc}")
rows=[]
for slot,p in enumerate(files,1):
    d=json.loads(p.read_text())
    if d.get('Success') is not True: raise SystemExit(f"round {rnd} slot {slot}: Success=false")
    if d.get('IsWarmStart') is not True: raise SystemExit(f"round {rnd} slot {slot}: non warm")
    e=d.get('KeplerEnergy') or {}
    if e.get('available') is not True: raise SystemExit(f"round {rnd} slot {slot}: Kepler unavailable {e.get('invalid_reason')}")
    cid=e.get('container_id','')
    if cid not in prewarm: raise SystemExit(f"round {rnd} slot {slot}: container inatteso {cid}")
    zones=e.get('cpu_joules_by_zone') or {}
    if zone not in zones: raise SystemExit(f"round {rnd} slot {slot}: zona {zone} assente")
    j=float(zones[zone])
    if not math.isfinite(j) or j<=0: raise SystemExit(f"round {rnd} slot {slot}: energia non valida {j}")
    b=datetime.fromisoformat(e['before_read_at'].replace('Z','+00:00'))
    a=datetime.fromisoformat(e['after_read_at'].replace('Z','+00:00'))
    if a<=b: raise SystemExit(f"round {rnd} slot {slot}: finestra non positiva")
    rows.append((slot,cid,j,b,a,float(d.get('Duration',0)),e['before_read_at'],e['after_read_at']))

ids=[r[1] for r in rows]
if len(set(ids))!=conc: raise SystemExit(f"round {rnd}: container distinti={len(set(ids))}/{conc}")
overlap=(min(r[4] for r in rows)-max(r[3] for r in rows)).total_seconds()
if overlap<=0: raise SystemExit(f"round {rnd}: finestre Kepler non sovrapposte ({overlap:.6f}s)")
wall=(int(end_ns)-int(start_ns))/1e9
with Path(records_path).open('a') as f:
    for slot,cid,j,b,a,dur,braw,araw in rows:
        f.write(json.dumps({'round':rnd,'slot':slot,'container_id':cid,'energy_j':j,'duration_s':dur,'before_read_at':braw,'after_read_at':araw,'overlap_seconds_round':overlap,'round_wall_seconds':wall},sort_keys=True)+'\n')
print(f"ROUND {rnd}: PASS | distinct={len(set(ids))}/{conc} | overlap={overlap:.6f}s | wall={wall:.6f}s")
for slot,cid,j,_,_,dur,_,_ in rows:
    print(f"  slot={slot} container={cid[:12]} {zone}={j:.9f}J duration={dur:.6f}s")
PY

  sleep 1
  current_warm="$(warm_count)"
  [[ "$current_warm" == "$CONCURRENCY" ]] || fail "dopo round $round warm=$current_warm, atteso=$CONCURRENCY"
done

EXPECTED=$((CONCURRENCY * ROUNDS))
python3 - "$RECORDS_JSONL" "$ROUNDS" "$CONCURRENCY" "$REWARD_ZONE" "$LOG_DIR/prewarm-container-ids.txt" "$SUMMARY_JSON" <<'PY'
import json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
records_path, rounds, conc, zone, ids_path, summary_path=sys.argv[1:]
rounds=int(rounds); conc=int(conc); expected=rounds*conc
rows=[json.loads(x) for x in Path(records_path).read_text().splitlines() if x.strip()]
if len(rows)!=expected: raise SystemExit(f"records={len(rows)} atteso={expected}")
prewarm=Path(ids_path).read_text().split()
counts=Counter(r['container_id'] for r in rows)
if set(counts)!=set(prewarm): raise SystemExit("container misurati != container prewarmed")
by=defaultdict(list)
for r in rows: by[r['container_id']].append(float(r['energy_j']))
summary={'status':'PASS','rounds':rounds,'concurrency':conc,'valid_measurements':len(rows),'expected_measurements':expected,'energy_coverage':len(rows)/expected,'reward_zone':zone,'measurements_per_container':dict(counts),'mean_energy_j_per_container':{k:statistics.mean(v) for k,v in by.items()},'round_overlap_seconds':{str(r):next(x['overlap_seconds_round'] for x in rows if x['round']==r) for r in range(1,rounds+1)}}
Path(summary_path).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print("FINAL CONCURRENCY / ISOLATION CHECK: PASS")
print(f"valid measurements={len(rows)}/{expected}")
print("energy coverage=100%")
for cid in prewarm: print(f"  container={cid[:12]} measurements={counts[cid]} mean_{zone}={statistics.mean(by[cid]):.9f}J")
PY

echo
echo "===== SUMMARY ====="
cat "$SUMMARY_JSON"
echo
echo "============================================================"
echo "KEPLER CONCURRENCY / CONTAINER ISOLATION LOCAL: PASS"
echo "============================================================"
echo "records: $RECORDS_JSONL"
echo "summary: $SUMMARY_JSON"
echo
echo "Interpretazione: attribuzione per-container e coverage sotto concorrenza validate localmente."
echo "NON dimostra l'esattezza fisica assoluta della stima energetica."
