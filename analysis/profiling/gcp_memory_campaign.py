#!/usr/bin/env python3
"""GCP UCB1Decoupled memory campaign. Native lifecycle logs; strict export-first cleanup.

Phase A: `preflight` offline. Phase B: `provision` -> `deploy` -> `pilot` ->
`run`. Phase C: `finish` after all checkpoints are locally verified. Every
mutation requires a confirmation argument. On a controlled failure the program
attempts STOP; GCP max-run-duration STOP covers host/network loss.

This is a NEW experimental runner. Never overwrite historical UCB1/RF scripts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

from analysis.profiling.gcp_memory_prepare import CONFIGS, read_catalog
from analysis.profiling.gcp_memory_analyze_lifecycle import analyze, parse_event_lines

ROOT = Path(__file__).resolve().parents[2]
PIN = '9f9e9cfc5631b41f525f6d91fadad4415b93a8c1'
PROJECT = 'serverledge-tesi-ale'
ZONE = 'europe-west4-a'
PLAN = ROOT / 'data/profiling/gcp-memory/planning/target_selection_v1/target_selection.csv'
PRIOR_ROOT = ROOT / 'data/profiling/gcp-memory/priors'
DEFAULT_RUNS = ROOT / 'data/profiling/gcp-memory/cloud-runs'
INFRA = ('sl-registry', 'sl-lb', 'sl-workload')
WORKERS = ('sl-x86-1', 'sl-arm-1')
VMS = INFRA + WORKERS
TYPES = {'sl-registry':'n2-standard-2', 'sl-lb':'n2-standard-4',
         'sl-workload':'e2-standard-2', 'sl-x86-1':'n2-standard-4',
         'sl-arm-1':'t2a-standard-4'}
IMAGES = {'amd64':'sl-x86-20260919', 'arm64':'sl-arm-20260919'}
HARD_LIMIT = '8h'  # Max RUNNING duration per VM, not full campaign duration.
EXPIRATION_S = 12
JANITOR_S = 2
POOL_MB = 12000
SAFE_NAME = re.compile(r'^[a-zA-Z0-9_.-]{1,120}$')
VALID_CONDITIONS = {row[0] for row in CONFIGS}

ACTUAL_MEMORY_CONFIG_IDS = (
    "N02",
    "R02",
    "N04",
    "R04",
    "N08",
    "R08",
)

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class CampaignError(RuntimeError):
    pass


def need(ok: bool, message: str):
    if not ok:
        raise CampaignError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def replace_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def command(argv: list[str], *, input: str | None = None, timeout: int = 240,
            capture: bool = True) -> str:
    result = subprocess.run(argv, input=input, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise CampaignError(f'command exit={result.returncode}: {shlex.join(argv)}\n'
                            f'{result.stderr[-1200:]}\n{result.stdout[-500:]}')
    if not capture and result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return result.stdout.strip()


def gc(*args, timeout=180):
    return command(['gcloud', f'--project={PROJECT}', *args], timeout=timeout)


def remote(host: str, script: str, *, timeout=180) -> str:
    need(host in VMS, f'Unrecognized host {host}')
    # Using bash -se via SSH stdin: eliminates two layers of interpolated quoting.
    return command(['gcloud', f'--project={PROJECT}', 'compute', 'ssh', host,
                    f'--zone={ZONE}', '--quiet', '--command=bash -se'],
                   input='set -euo pipefail\n' + script + '\n', timeout=timeout)


def upload(source: Path, host: str, dest: str):
    need(host in VMS and dest.startswith('/tmp/mem-'), 'Unsafe scp destination')
    return gc('compute', 'scp', f'--zone={ZONE}', '--quiet', str(source), f'{host}:{dest}', timeout=240)


def download(host: str, remote_path: str, local_path: Path):
    need(host in VMS and remote_path.startswith('/tmp/mem-'), 'Unsafe scp source')
    need(not local_path.exists(), f'File already exists: {local_path}')
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gc('compute', 'scp', f'--zone={ZONE}', '--quiet', f'{host}:{remote_path}', str(local_path), timeout=240)
    need(local_path.is_file(), f'Download missing {local_path}')


def run_remote_file(host: str, source: str, path: Path):
    need(source.startswith('/var/log/serverledge-'), 'Unsafe log source')
    tmp = '/tmp/mem-' + re.sub(r'[^a-zA-Z0-9-]', '-', path.parent.parent.name + '-' + host + '-' + path.name)
    remote(host, f'sudo cp {shlex.quote(source)} {shlex.quote(tmp)}\n'
                 f'sudo chmod 644 {shlex.quote(tmp)}')
    download(host, tmp, path)


def instances() -> dict:
    content = gc('compute','instances','list',f'--zones={ZONE}','--format=json')
    entries = json.loads(content)
    need(isinstance(entries, list), 'Malformed gcloud instance list')
    return {e['name']:e for e in entries}


def selected_targets() -> list[str]:
    need(PLAN.is_file(), f'Missing target plan: {PLAN}')
    with PLAN.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    names = [r['function_name'] for r in rows]
    need(len(names) == 10 and len(set(names)) == 10, 'Expected 10 distinct targets')
    return names


def preflight() -> dict:
    catalog = read_catalog(ROOT)
    targets = selected_targets()
    evidence_info = {}
    all_shas = {}
    for target in targets:
        need(target in catalog, f'{target}: missing catalog entry')
        path = PRIOR_ROOT / f'{target}-r01' / 'grid-manifest.json'
        need(path.is_file(), f'{target}: grid manifest missing')
        manifest = load(path)
        need(manifest.get('target') == target and manifest.get('replica') == 'r01', f'{target}: grid metadata')
        need(manifest.get('status') == 'OFFLINE_PREPARED_NOT_GCP_VALIDATED',
             f'{target}: unexpected grid status')
        entries = manifest.get('conditions', [])
        need(len(entries) == 9, f'{target}: expected nine conditions')
        by_id = {r.get('config_id'):r for r in entries}
        need(set(by_id) == VALID_CONDITIONS, f'{target}: wrong config IDs')
        for cid, mode, wr, we, c in CONFIGS:
            entry = by_id[cid]
            need(entry.get('formula') == mode and entry.get('policy') == 'UCB1Decoupled',
                 f'{target}/{cid}: wrong policy/formula')
            need(math.isclose(float(entry['c']), c, rel_tol=0, abs_tol=1e-12), f'{target}/{cid}: c mismatch')
            need(entry.get('prior_applied') is (mode != 'none'), f'{target}/{cid}: wrong prior flag')
            if mode != 'none':
                need(math.isclose(float(entry['w_R']), wr, abs_tol=1e-12)
                     and math.isclose(float(entry['w_E']), we, abs_tol=1e-12),
                     f'{target}/{cid}: weight mismatch')
                folder = path.parent / cid
                prior_path = folder / 'frozen-prior.json'
                request = folder / 'materialized-request.json'
                source_manifest = folder / 'manifest.json'
                need(prior_path.is_file() and request.is_file() and source_manifest.is_file(),
                     f'{target}/{cid}: files missing')
                actual_sha = sha(prior_path)
                prior = load(prior_path)
                applied = load(request)
                meta = load(source_manifest)
                need(actual_sha == entry.get('prior_sha256') == meta.get('prior_sha256'),
                     f'{target}/{cid}: SHA mismatch')
                need(applied.get('target_function_name') == target and applied.get('prior') == prior,
                     f'{target}/{cid}: request/prior mismatch')
                need(prior.get('policy') == 'UCB1Decoupled'
                     and prior.get('donor_function_name') == manifest.get('donor')
                     and prior['config']['ucb1_reference_anchor']['mode'] == mode,
                     f'{target}/{cid}: prior mismatch')
                for arch in ('amd64','arm64'):
                    u = prior['arms'][arch]['ucb1']
                    need(math.isclose(float(u['observation_weight']), wr, abs_tol=1e-12)
                         and math.isclose(float(u['exploration_observation_weight']), we, abs_tol=1e-12),
                         f'{target}/{cid}: materialized weights mismatch')
                all_shas[f'{target}/{cid}'] = actual_sha
        evidence_info[target] = {'donor':manifest['donor'], 'grid_sha256':sha(path),
                                 'bootstrap_commit':manifest.get('bootstrap_commit'),
                                 'catalog': catalog[target]}
    need(HARD_LIMIT and PIN, 'Internal configuration missing')
    result = {'schema_version':1, 'status':'OFFLINE_PASS_NOT_CLOUD_VALIDATED',
              'runtime_pin':PIN, 'targets':targets, 'grid': [v[0] for v in CONFIGS],
              'grid_frozen': True, 'prewarm_target': False,
              'container_max_concurrency':1, 'memory_pool_mb': POOL_MB,
              'container_expiration_s':EXPIRATION_S,'janitor_s':JANITOR_S,
              'function_data':evidence_info, 'prior_sha256':all_shas,
              'target_selection_csv_sha256':sha(PLAN)}
    print(f'PASS offline: {len(targets)} target / {len(CONFIGS)} condizioni / {len(all_shas)} prior')
    for fn in targets:
        print(f'  {fn}: donor={evidence_info[fn]["donor"]}')
    return result


def store_campaign(dest: Path, data: dict):
    write_new(dest / 'campaign.json', data)
    print(f'Campaign: {dest}')


def campaign(dest: Path) -> dict:
    path = dest / 'campaign.json'
    need(path.is_file(), f'Missing campaign: {path}')
    doc = load(path)
    need(doc.get('runtime_pin') == PIN and doc.get('project') == PROJECT
         and doc.get('zone') == ZONE and doc.get('replicas') in (1,2)
         and doc.get('mode') in ('pilot','full'), 'Campaign manifest invalid')
    need(doc.get('target_selection_csv_sha256') == sha(PLAN), 'Target plan changed')
    need(not doc.get('deleted'), 'VMs for this campaign already deleted')
    return doc


def run_id_now() -> str:
    return 'mem' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')


def vm_guard(doc: dict, *, require_running: bool = True) -> dict:
    need(doc.get('vm_ids'), 'No VM IDs recorded: provision first')
    cloud = instances()
    for name, uid in doc['vm_ids'].items():
        found = cloud.get(name)
        need(found is not None, f'{name} missing')
        need(str(found['id']) == str(uid), f'{name}: changed GCP instance ID')
        need((found.get('labels') or {}).get('memrun') == doc['run_id'],
             f'{name}: ownership label mismatch')
        if require_running:
            need(found.get('status') == 'RUNNING', f'{name}: status={found.get("status")}')
    return cloud


def check_firewalls_and_images():
    for name in ('serverledge-internal','serverledge-ssh'):
        gc('compute','firewall-rules','describe',name,'--format=json')
    for image in IMAGES.values():
        gc('compute','images','describe',image,'--format=json')


def provision(doc: dict, dest: Path):
    verified_build(dest)  # Avoid spending cloud time compiling local Go.
    need(not doc.get('vm_ids'), 'Already provisioned')
    check_firewalls_and_images()
    present = instances()
    conflict = set(VMS) & set(present)
    need(not conflict, f'VM names already in use (do not reuse!): {sorted(conflict)}')
    leftovers=[name for name,entry in present.items()
               if (entry.get('labels') or {}).get('progetto')=='serverledge-tesi']
    need(not leftovers, f'Other Serverledge VMs exist: {sorted(leftovers)}; check costs before proceeding')
    doc['vm_ids'] = {}
    replace_json(dest/'campaign.json', doc)
    for name in VMS:
        arch = 'arm64' if name.startswith('sl-arm-') else 'amd64'
        print(f'CREATE VM {name}', flush=True)
        gc('compute','instances','create',name,f'--zone={ZONE}',
           f'--machine-type={TYPES[name]}',f'--image={IMAGES[arch]}',
           '--boot-disk-size=50GB','--boot-disk-auto-delete',
           f'--labels=progetto=serverledge-tesi,memrun={doc["run_id"]}',
           f'--max-run-duration={HARD_LIMIT}', '--instance-termination-action=STOP',
           '--quiet', timeout=360)
        current = instances()
        need(name in current and current[name].get('status') == 'RUNNING',
             f'Provisioned {name} but status invalid')
        doc['vm_ids'][name] = str(current[name]['id'])
        replace_json(dest/'campaign.json', doc)
    doc['provisioned'] = True
    replace_json(dest/'campaign.json', doc)
    print('PASS provisioning 5 VM, 8h GCP STOP failsafe')


def build_runtime(dest: Path) -> dict:
    need(command(['git','-C',str(ROOT),'rev-parse','HEAD']) == PIN,
         f'Local HEAD must match pinned runtime {PIN}')
    command(['git','-C',str(ROOT),'cat-file','-e',f'{PIN}^{{commit}}'])
    artifacts = dest / 'binaries'
    need(not artifacts.exists(), f'Build output already exists: {artifacts}')
    artifacts.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='memory-pin-') as tmp:
        # Git archive isolates checked-in sources, never reads local uncommitted Go files.
        import tarfile
        archive_bytes = subprocess.run(['git','-C',str(ROOT),'archive','--format=tar',PIN],
                                       capture_output=True,check=True).stdout
        import io
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
            archive.extractall(tmp, filter='data')
        build_commands = (
            ('lb-amd64','amd64','./cmd/lb'),
            ('serverledge-amd64','amd64','./cmd/serverledge'),
            ('serverledge-arm64','arm64','./cmd/serverledge'),
            ('serverledge-cli-amd64','amd64','./cmd/cli/main.go'),
        )
        for name, arch, src in build_commands:
            env = os.environ.copy()
            env.update(CGO_ENABLED='0',GOOS='linux',GOARCH=arch)
            proc = subprocess.run(['go','build','-trimpath','-buildvcs=false',
                                   '-o',str(artifacts/name),src],cwd=tmp,env=env,
                                  text=True,capture_output=True,timeout=540)
            need(proc.returncode == 0, f'Build failed {name}: {proc.stderr[-1000:]}')
    manifest = {f.name:sha(f) for f in artifacts.iterdir() if f.is_file()}
    need(len(manifest) == 4, 'Incomplete bin set')
    write_new(dest/'build-sha256.json', {'pin':PIN,'sha256':manifest})
    return manifest


def verified_build(dest: Path) -> dict:
    meta_path = dest / 'build-sha256.json'
    need(meta_path.is_file(), 'Local build missing: run `build` before provisioning')
    meta = load(meta_path)
    need(meta.get('pin') == PIN, 'Build commit mismatch')
    checks = meta.get('sha256', {})
    need(set(checks) == {'lb-amd64','serverledge-amd64','serverledge-arm64',
                         'serverledge-cli-amd64'}, 'Build manifest missing binaries')
    for name, digest in checks.items():
        binary = dest / 'binaries' / name
        need(binary.is_file() and sha(binary) == digest, f'Build SHA mismatch: {name}')
    return checks


def deploy(doc:dict, dest:Path):
    vm_guard(doc)
    shas = verified_build(dest)
    for host in VMS:
        # Refuse dirty tracked checkouts and active application processes.
        probe = remote(host, 'sudo test -d /opt/serverledge/.git\n'
                             "sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge status --porcelain --untracked-files=no\n"
                             "! pgrep -x serverledge >/dev/null\n! pgrep -x lb >/dev/null")
        need(not probe, f'{host}: dirty tracked checkout / unexpected output')
    for host in VMS:
        remote(host, "sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge fetch --quiet origin main\n"
                     f"sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge checkout --detach {PIN}\n"
                     f"test \"$(sudo git -c safe.directory=/opt/serverledge -C /opt/serverledge rev-parse HEAD)\" = {PIN}")
    roles = {'sl-lb':[('lb-amd64','lb')],
             'sl-x86-1':[('serverledge-amd64','serverledge')],
             'sl-arm-1':[('serverledge-arm64','serverledge')],
             'sl-workload':[('serverledge-cli-amd64','serverledge-cli')]}
    for host, binaries in roles.items():
        for name, target in binaries:
            staged = '/tmp/mem-'+name
            upload(dest/'binaries'/name, host, staged)
            remote(host, f'actual=$(sha256sum {shlex.quote(staged)} | cut -d" " -f1)\n'
                         f'test "$actual" = {shas[name]}\n'
                         f'sudo install -m 0755 {shlex.quote(staged)} /opt/serverledge/bin/{target}\n'
                         f'test "$(sha256sum /opt/serverledge/bin/{target} | cut -d" " -f1)" = {shas[name]}')
    # Runtimes from the pinned images; detect unsupported/absent ones before billable measurements.
    meta = preflight()['function_data']
    paths = sorted({v['catalog']['src'] for v in meta.values()})
    script = 'cd /opt/serverledge/examples/experiments\n' + '\n'.join('test -f '+shlex.quote(p) for p in paths)
    remote('sl-workload', script)
    doc['deployed'] = True
    doc['build_sha256'] = shas
    replace_json(dest/'campaign.json', doc)
    print('PASS deployment verified source PIN, binaries, target file paths')


def internal_ip(name: str) -> str:
    need(name in VMS, 'Unsafe VM lookup')
    data = json.loads(gc('compute','instances','describe',name,f'--zone={ZONE}','--format=json'))
    ip = data['networkInterfaces'][0]['networkIP']
    need(re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', ip) is not None, 'Bad internal IP')
    return ip


def fresh_runtime(doc: dict, target: str, cid: str, run_stamp: str):
    vm_guard(doc)
    entry = next(r for r in CONFIGS if r[0] == cid)
    c = entry[-1]
    registry_ip = internal_ip('sl-registry')

    for host in ('sl-lb', *WORKERS):
        remote(
            host,
            'sudo pkill -x lb || true\n'
            'sudo pkill -x serverledge || true\n'
            'sleep 2'
        )

    for host in WORKERS:
        count = remote(host, 'sudo docker ps -q | wc -l').strip()
        need(
            count == '0',
            f'{host}: Docker containers still active: {count}'
        )

    remote('sl-registry', '''sudo docker rm -f Etcd-server >/dev/null 2>&1 || true
sudo docker run -d -p 2379:2379 -p 2380:2380 --name Etcd-server \
  gcr.io/etcd-development/etcd:v3.5.13 /usr/local/bin/etcd \
  --name s1 --data-dir /etcd-data --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster s1=http://0.0.0.0:2380 \
  --initial-cluster-token tkn --initial-cluster-state new \
  --max-request-bytes 33554432 --quota-backend-bytes 8589934592 >/dev/null''')

    for _ in range(20):
        try:
            remote(
                'sl-registry',
                'sudo docker exec Etcd-server etcdctl endpoint health',
                timeout=45
            )
            break
        except CampaignError:
            time.sleep(2)
    else:
        raise CampaignError('etcd not healthy')

    for host in WORKERS:
        arch = 'arm64' if 'arm' in host else 'amd64'

        yaml = f'''registry.area: "cloud-region"
registry.node.id: "{host}"
registry.udp.port: 9877
etcd.address: "{registry_ip}:2379"
node.machine_tag: "{arch}"
container.pool.memory: {POOL_MB}
container.pool.cpus: 4
container.expiration: {EXPIRATION_S}
janitor.interval: {JANITOR_S}
factory.images.refresh: false
profiling.enabled: false
profiling.export.enabled: false
'''

        script = (
                "sudo tee /opt/serverledge/worker-memory.yaml >/dev/null <<'MEM_YAML'\n"
                + yaml
                + "MEM_YAML\n"
                + "sudo sh -c 'cd /opt/serverledge && "
                + "SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1 "
                + "SERVERLEDGE_CONTAINER_MEMORY_SAMPLE_MS=1000 "
                + "nohup ./bin/serverledge worker-memory.yaml "
                + "> /var/log/serverledge-node.log 2>&1 </dev/null &'"
        )

        remote(host, script)


    lb_yml = f'''registry.area: "cloud-region"
registry.udp.port: 9877
lb.replicas: 128
lb.mode: "MAB"
lb.arch_awareness: true
lb.refresh_interval: 5
mab.policy: "UCB1Decoupled"
mab.ucb1.c: {c}
mab.reward.mode: "latency"
mab.transfer.control.enabled: true
etcd.address: "{registry_ip}:2379"
'''

    remote(
        'sl-lb',
        "sudo tee /opt/serverledge/lb-memory.yaml >/dev/null <<'MEM_YAML'\n"
        + lb_yml
        + "MEM_YAML\nsudo sh -c 'cd /opt/serverledge && "
        + "nohup ./bin/lb lb-memory.yaml "
        + "> /var/log/serverledge-lb.log 2>&1 </dev/null &'"
    )

    for _ in range(25):
        try:
            for host in (*WORKERS, 'sl-lb'):
                remote(
                    host,
                    'curl -fsS --max-time 5 '
                    'http://127.0.0.1:1323/status >/dev/null',
                    timeout=45
                )

            registered = remote(
                'sl-registry',
                "sudo docker exec Etcd-server etcdctl get "
                "registry/cloud-region/ --prefix --keys-only "
                "| grep -E '^registry/cloud-region/(amd64|arm64)/' "
                "| wc -l"
            )

            need(
                registered.strip() == '2',
                f'expected 2 registered workers: {registered}'
            )

            remote(
                'sl-lb',
                '''grep -q 'Running ArchitectureAwareLB' /var/log/serverledge-lb.log
grep -q 'LB mode set to MAB' /var/log/serverledge-lb.log
grep -q 'event=new_machine_tag_discovered tag=amd64 arch=amd64' /var/log/serverledge-lb.log
grep -q 'event=new_machine_tag_discovered tag=arm64 arch=arm64' /var/log/serverledge-lb.log'''
            )

            break

        except CampaignError:
            time.sleep(2)
    else:
        raise CampaignError(
            'Runtime not healthy, 2 workers and LB must be active'
        )

    for host in WORKERS:
        env = remote(
            host,
            "sudo python3 - <<'PY_ENV'\n"
            "import subprocess\n"
            "pid = subprocess.check_output("
            "['pgrep', '-x', 'serverledge'], "
            "text=True).splitlines()[0]\n"
            "value = open('/proc/' + pid + '/environ', "
            "'rb').read().split(b'\\x00')\n"
            "assert b'SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1' in value\n"
            "assert b'SERVERLEDGE_CONTAINER_MEMORY_SAMPLE_MS=1000' in value\n"
            "print('SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1 "
            "SERVERLEDGE_CONTAINER_MEMORY_SAMPLE_MS=1000')\n"
            "PY_ENV"
        )

        need(
            'SERVERLEDGE_CONTAINER_LIFECYCLE_LOG=1' in env
            and 'SERVERLEDGE_CONTAINER_MEMORY_SAMPLE_MS=1000' in env,
            f'{host}: container memory telemetry not enabled'
        )

    print(
        f'PASS fresh runtime {target}/{cid}: '
        f'c={c}, policy UCB1Decoupled, '
        'memory=1000ms, no target prewarm'
    )


def registration_catalog() -> dict:
    """Extract actual function definitions from the validated preflight structure.

    preflight()['function_data'][target] includes donor/provenance alongside
    the nested 'catalog' record. Passing the parent object to register_target
    caused KeyError('runtime') on the first real GCP pilot.
    """
    source = preflight()['function_data']
    result = {}
    for name, record in source.items():
        need(isinstance(record, dict) and isinstance(record.get('catalog'), dict),
             f'{name}: missing nested catalog metadata')
        meta = record['catalog']
        need(all(field in meta for field in ('runtime', 'src', 'memory_mb', 'handler')),
             f'{name}: incomplete catalog metadata')
        result[name] = meta
    return result


def register_target(target: str, catalog: dict):
    need(target in catalog, f'{target}: target missing in registration catalog')
    meta = catalog[target]
    # Original GCP measurement runner registers functions via the LB endpoint;
    # do not bind function registration to the x86 worker only.
    host = internal_ip('sl-lb')
    args = ['-f',target, '--runtime',meta['runtime'], '--src',meta['src'],
            '--memory',str(meta['memory_mb']), '--cpu','1','--max_concurrency','1','--update']
    if meta.get('handler'):
        args += ['--handler',meta['handler']]
    cmd = ('cd /opt/serverledge/examples/experiments\n'
           f'export SERVERLEDGE_HOST={host}\nexport SERVERLEDGE_PORT=1323\n'
           '/opt/serverledge/bin/serverledge-cli create '+shlex.join(args))
    remote('sl-workload',cmd,timeout=120)
    assert_no_target_container_created(target, phase='function registration')


def assert_no_target_container_created(target: str, *, phase: str):
    for host in WORKERS:
        needle = shlex.quote('function="' + target + '"')
        found = remote(host, f"if sudo grep -F {needle} /var/log/serverledge-node.log "
                             "| grep -q 'event=container_created'; then echo FOUND; else echo EMPTY; fi")
        need(found == 'EMPTY', f'{host}: target created during {phase}, no-prewarm invariant failed')


def apply_prior(target: str, cid: str, run_stamp: str, folder: Path):
    entry = next(x for x in CONFIGS if x[0] == cid)
    if entry[1] == 'none':
        return None
    req_path = PRIOR_ROOT / (target+'-r01') / cid / 'materialized-request.json'
    remote_req = '/tmp/mem-'+run_stamp+'-request.json'
    remote_res = '/tmp/mem-'+run_stamp+'-transfer.json'
    upload(req_path,'sl-workload',remote_req)
    lb_ip = internal_ip('sl-lb')
    remote('sl-workload', 'curl -fsS --max-time 30 -H "Content-Type: application/json" '
          f'--data-binary @{shlex.quote(remote_req)} '
          f'-o {shlex.quote(remote_res)} http://{lb_ip}:1323/mab/transfer/initialize-materialized')
    dest = folder / 'transfer-response.json'
    download('sl-workload',remote_res,dest)
    result = load(dest)
    need(result.get('transfer_applied') is True and result.get('transfer_attempted') is True
         and result.get('initialization_source') == 'materialized_prior'
         and result.get('target_function_name') == target and result.get('prior') == load(req_path)['prior'],
         f'{target}/{cid}: prior rejected/changed')
    return sha(req_path)


def _log_fields(line: str) -> dict[str, str]:
    """Parse the key=value tokens used by the pinned Go MAB event log."""
    return dict(re.findall(r'(?<!\S)([A-Za-z_][A-Za-z_0-9]*)=([^\s]*)', line))


def check_decisions(lb_text: str, target: str, count: int) -> dict:
    """Preserve fallback executions; reject missing/ambiguous MAB outcomes.

    The MAB reward penalty for a fallback is *not* a latency observation.
    A fallback changes which architecture actually executes the request,
    so both architectures and the fallback reason must be retained.
    """
    resolved = []
    for line in lb_text.splitlines():
        if 'event=decision_resolved' not in line:
            continue
        fields = _log_fields(line)
        if fields.get('function') == target and fields.get('event') == 'decision_resolved':
            resolved.append((line, fields))
    need(len(resolved) == count, f'Expected {count} MAB decision_resolved, got {len(resolved)}')
    ids = set()
    selected_counts = Counter()
    execution_counts = Counter()
    fallback_reasons = Counter()
    fallback_events = []
    for line, fields in resolved:
        rid = fields.get('request_id')
        selected = fields.get('selected_arm')
        execution = fields.get('execution_arm')
        fallback = fields.get('fallback')
        need(bool(rid) and rid not in ids, f'Duplicate/missing MAB request_id: {rid!r}')
        ids.add(rid)
        need(selected in ('amd64', 'arm64') and execution in ('amd64', 'arm64'),
             f'Invalid selected/execution arm for MAB request {rid}')
        need(fallback in ('true', 'false'), f'Missing/invalid fallback flag: {rid}')
        is_fallback = fallback == 'true'
        need(is_fallback == (selected != execution),
             f'Inconsistent selected/execution/fallback for {rid}')
        selected_counts[selected] += 1
        execution_counts[execution] += 1
        if is_fallback:
            reason = fields.get('fallback_reason')
            need(bool(reason), f'Missing fallback reason for request {rid}')
            fallback_reasons[reason] += 1
            fallback_events.append({
                'request_id': rid,
                'selected_arm': selected,
                'execution_arm': execution,
                'reason': reason,
                'resolved_log_line': line,
            })
    # Preserve penalty diagnostics, without treating synthetic rewards as latency.
    synthetic = []
    for line in lb_text.splitlines():
        if 'event=synthetic_reward' not in line:
            continue
        fields = _log_fields(line)
        if fields.get('event') == 'synthetic_reward' and fields.get('function') == target:
            synthetic.append({
                'request_id': fields.get('request_id'),
                'arm': fields.get('arm'),
                'reward': fields.get('reward'),
                'reason': fields.get('reason'),
                'raw_line': line,
            })
    synthetic_by_id = {event['request_id'] for event in synthetic if event['request_id']}
    for event in fallback_events:
        event['synthetic_reward_logged'] = event['request_id'] in synthetic_by_id
    return {
        'decisions': len(resolved),
        'fallbacks': len(fallback_events),
        'selected_arm_counts': dict(selected_counts),
        'execution_arm_counts': dict(execution_counts),
        'fallback_reasons': dict(fallback_reasons),
        'fallback_events': fallback_events,
        'synthetic_rewards': synthetic,
        'comparison_class': ('WITH_FALLBACK' if fallback_events else 'NO_FALLBACK'),
        'raw_selected_execution_lines': [line for line, _ in resolved],
    }

def validate_actual_memory(
        worker_text: dict[str, str],
        target: str,
) -> tuple[dict, list[dict]]:

    lifecycle = {}
    samples = {}
    stops = set()
    errors = []

    for host, content in worker_text.items():
        for line in content.splitlines():

            if (
                    "[CONTAINER_LIFECYCLE]" not in line
                    and "[CONTAINER_MEMORY]" not in line
            ):
                continue

            is_memory = "[CONTAINER_MEMORY]" in line

            marker = (
                "[CONTAINER_MEMORY]"
                if is_memory
                else "[CONTAINER_LIFECYCLE]"
            )

            try:
                fields = dict(
                    piece.split("=", 1)
                    for piece in shlex.split(
                        line.split(marker, 1)[1]
                    )
                    if "=" in piece
                )
            except ValueError as exc:
                raise CampaignError(
                    f"Invalid container log on {host}: {line}"
                ) from exc

            if fields.get("function") != target:
                continue

            key = (host, fields.get("container_id"))

            need(
                bool(key[1]),
                f"Missing container ID on {host}: {line}",
            )

            event = fields.get("event")

            if not is_memory and event in (
                    "container_created",
                    "container_destroyed",
            ):
                need(
                    key not in lifecycle
                    or event not in lifecycle[key],
                    f"Duplicate {event} for {key}",
                    )

                lifecycle.setdefault(key, {})[event] = fields

            elif is_memory and event == "memory_sample":
                samples.setdefault(key, []).append(fields)

            elif is_memory and event == "monitor_stopped":
                need(
                    key not in stops,
                    f"Duplicate monitor_stopped for {key}",
                    )
                stops.add(key)

            elif is_memory and event in (
                    "sample_error",
                    "sample_invalid",
            ):
                errors.append(
                    (key, event, fields.get("error"))
                )

    need(
        not errors,
        f"Memory reading errors for {target}: {errors[:3]}",
    )

    need(
        bool(lifecycle),
        f"No lifecycle events for {target}",
    )

    need(
        set(samples) == set(lifecycle),
        f"Memory container mismatch: "
        f"missing={set(lifecycle) - set(samples)}, "
        f"extra={set(samples) - set(lifecycle)}",
        )

    need(
        stops == set(lifecycle),
        f"Missing/extra monitor_stopped: "
        f"{set(lifecycle) ^ stops}",
        )

    per_container = []

    for (host, container_id), pair in sorted(
            lifecycle.items()
    ):
        need(
            set(pair) == {
                "container_created",
                "container_destroyed",
            },
            f"Incomplete lifecycle {host}/{container_id}",
            )

        start = int(
            pair["container_created"]["ts"]
        )

        end = int(
            pair["container_destroyed"]["ts"]
        )

        need(
            end > start,
            f"Invalid lifetime {host}/{container_id}",
            )

        container_samples = samples[
            (host, container_id)
        ]

        need(
            len(container_samples) >= 2,
            f"Too few samples {host}/{container_id}",
            )

        container_samples.sort(
            key=lambda row: int(row["ts"])
        )

        times = [
            int(row["ts"])
            for row in container_samples
        ]

        usage = [
            int(row["usage_bytes"])
            for row in container_samples
        ]

        limits = [
            int(row["limit_bytes"])
            for row in container_samples
        ]

        reasons = [
            row.get("reason")
            for row in container_samples
        ]

        need(
            all(u > 0 for u in usage),
            f"Invalid RAM usage {host}/{container_id}",
        )

        need(
            all(limit > 0 for limit in limits),
            f"Invalid RAM limit {host}/{container_id}",
        )

        need(
            all(start <= t <= end for t in times),
            f"Sample outside lifetime {host}/{container_id}",
        )

        need(
            all(
                b > a
                for a, b in zip(times, times[1:])
            ),
            f"Non-increasing timestamps {host}/{container_id}",
        )

        need(
            "start" in reasons
            and "before_destroy" in reasons,
            f"Missing boundary samples {host}/{container_id}",
            )

        lifetime_ms = end - start

        if lifetime_ms >= 2000:
            need(
                "periodic" in reasons,
                f"No periodic samples {host}/{container_id}",
                )

        gaps = [
            b - a
            for a, b in zip(times, times[1:])
        ]

        need(
            max(gaps) <= 3500,
            f"Sampling gap > 3.5 s {host}/{container_id}",
            )

        coverage = (
                           times[-1] - times[0]
                   ) / lifetime_ms

        need(
            coverage >= 0.90,
            f"RAM coverage below 90% "
            f"{host}/{container_id}: {coverage:.3f}",
            )

        # Integrale trapezoidale sulle sole osservazioni.
        # byte_seconds = byte * secondi.
        byte_seconds = sum(
            (u0 + u1) / 2 * (t1 - t0) / 1000.0
            for u0, u1, t0, t1 in zip(
                usage,
                usage[1:],
                times,
                times[1:],
            )
        )

        per_container.append({
            "worker": host,
            "container_id": container_id,
            "function_name": target,
            "machine_tag":
                pair["container_created"]["machine_tag"],
            "lifetime_ms": lifetime_ms,
            "sample_count": len(container_samples),
            "periodic_sample_count": sum(
                reason == "periodic"
                for reason in reasons
            ),
            "coverage_ratio": coverage,
            "max_sample_gap_ms": max(gaps),
            "observed_usage_peak_bytes": max(usage),
            "actual_memory_MB_s":
                byte_seconds / 1_000_000,
            "actual_memory_MiB_s":
                byte_seconds / (1024 ** 2),
            "unmeasured_start_ms": times[0] - start,
            "unmeasured_end_ms": end - times[-1],
        })

    summary = {
        "actual_memory_MB_s": sum(
            row["actual_memory_MB_s"]
            for row in per_container
        ),
        "actual_memory_MiB_s": sum(
            row["actual_memory_MiB_s"]
            for row in per_container
        ),
        "actual_memory_container_count":
            len(per_container),
        "actual_memory_sample_count": sum(
            row["sample_count"]
            for row in per_container
        ),
        "actual_memory_min_coverage": min(
            row["coverage_ratio"]
            for row in per_container
        ),
        "actual_memory_integration":
            "trapezoid_observed_interval_no_boundary_imputation",
    }

    return summary, per_container


def validate_local_measurement(folder: Path, target: str, cid: str,
                               concurrency: int, *, require_multi: bool = False
                               ) -> tuple[dict, list[dict], dict]:
    """Recompute all assertions solely from exported local files.

    Used both for fresh measurements and for recovery of the interrupted
    R04 pilot. No cloud API, no side effects, no dependency on existing
    derived summaries. Existing raw files remain the source of truth.
    """
    requests_path = folder / 'requests.jsonl'
    lb_path = folder / 'raw/sl-lb.log'
    need(requests_path.is_file() and lb_path.is_file(),
         f'{folder.name}: request or LB raw log missing')
    requests = [json.loads(line) for line in requests_path.read_text(encoding='utf-8').splitlines()
                if line.strip()]
    need(len(requests) == 50 and {r.get('index') for r in requests} == set(range(1, 51))
         and all(r.get('success') is True for r in requests),
         f'{folder.name}: request data malformed/incomplete')
    need(all(isinstance(r.get('client_elapsed_ms'), (int, float))
                 and math.isfinite(r['client_elapsed_ms']) and r['client_elapsed_ms'] > 0
                 for r in requests), f'{folder.name}: client latency invalid')
    # Cloud client retains response headers independently of the MAB logs.
    # Per-request correlation is unavailable unless the HTTP response exposes
    # the same request ID as the LB event; do not invent such a join.
    need(all(r.get('execution_header_arm') in ('amd64', 'arm64') for r in requests),
         f'{folder.name}: response architecture header absent/invalid')
    worker_text = {}
    for host in WORKERS:
        raw_path = folder / 'raw' / f'{host}.log'
        need(raw_path.is_file(), f'{folder.name}: missing raw log for {host}')
        worker_text[host] = raw_path.read_text(encoding='utf-8')
    summary, rows = analyze(worker_text, target, allow_prewarm=False,
                            allowed_destroy_reasons=('expired',))
    actual_memory_summary, _ = validate_actual_memory(
        worker_text,
        target,
    )

    summary.update(actual_memory_summary)
    decisions = check_decisions(lb_path.read_text(encoding='utf-8'), target, 50)
    actual_arch_counts = Counter(r['execution_header_arm'] for r in requests)
    # We can validate aggregate execution counts; there is no guaranteed
    # request-ID correlation between HTTP results and MAB event records.
    need(dict(actual_arch_counts) == decisions['execution_arm_counts'],
         f'{folder.name}: LB execution counts differ from client response headers '
         f'({decisions["execution_arm_counts"]} != {dict(actual_arch_counts)})')
    if cid != '':
        entry = next((x for x in CONFIGS if x[0] == cid), None)
        need(entry is not None, f'Unknown frozen configuration {cid}')
        if entry[1] != 'none':
            response_path = folder / 'transfer-response.json'
            source_request = PRIOR_ROOT / (target + '-r01') / cid / 'materialized-request.json'
            need(response_path.is_file() and source_request.is_file(),
                 f'{folder.name}: missing transfer application proof')
            response = load(response_path)
            original = load(source_request)
            need(response.get('transfer_applied') is True
                 and response.get('transfer_attempted') is True
                 and response.get('initialization_source') == 'materialized_prior'
                 and response.get('target_function_name') == target
                 and response.get('prior') == original.get('prior'),
                 f'{folder.name}: materialized prior not verified')
        else:
            need(not (folder / 'transfer-response.json').exists(),
                 f'{folder.name}: No-TL run has a transfer response')
    summary.update({
        'run_key': folder.name,
        'config_id': cid,
        'policy': 'UCB1Decoupled',
        'concurrency': concurrency,
        'requests': 50,
        'runtime_pin': PIN,
        'request_sha256': sha(requests_path),
        'latency_client_mean_ms': sum(r['client_elapsed_ms'] for r in requests) / 50,
        'warm_invocations': sum(r.get('is_warm_start') is True for r in requests),
        'cold_invocations': sum(r.get('is_warm_start') is False for r in requests),
        'mab': {k:v for k,v in decisions.items() if k != 'raw_selected_execution_lines'},
        'validation_class': 'COMPLETE_WITH_FALLBACK' if decisions['fallbacks'] else 'COMPLETE_NO_FALLBACK',
        'fallback_analysis_note': (
            'Fallback executions are valid observations of deployed behavior, '
            'but must be stratified for causal comparisons. Synthetic rewards '
            'are not latency samples.'),
    })
    summary['max_containers_one_worker'] = max(summary['peak_resident_containers_by_worker'].values())
    if require_multi:
        need(summary['max_containers_one_worker'] > 1,
             f'{folder.name}: no actual multi-container worker')
    return summary, rows, decisions


def finalize_local_measurement(
        folder: Path,
        target: str,
        cid: str,
        concurrency: int,
        *,
        require_multi: bool = False,
) -> dict:

    summary, rows, decisions = validate_local_measurement(
        folder,
        target,
        cid,
        concurrency,
        require_multi=require_multi,
    )

    output_names = (
        "memory-summary.json",
        "container-residency.csv",
        "container-actual-memory.csv",
        "mab-decisions.log",
        "verified.json",
    )

    need(
        not any(
            (folder / name).exists()
            for name in output_names
        ),
        f"{folder.name}: some finalized outputs already exist; "
        "refuse overwrite",
    )

    worker_logs = {}

    for host in WORKERS:

        log_path = folder / "raw" / f"{host}.log"

        need(
            log_path.is_file(),
            f"{folder.name}: missing worker log {host}",
        )

        worker_logs[host] = log_path.read_text(
            encoding="utf-8"
        )

    actual_memory_summary, actual_memory_rows = (
        validate_actual_memory(
            worker_logs,
            target,
        )
    )

    need(
        bool(actual_memory_rows),
        f"{folder.name}: no actual memory measurements",
    )

    need(
        summary["actual_memory_container_count"]
        == len(actual_memory_rows),
        f"{folder.name}: actual memory container count mismatch",
        )

    need(
        math.isclose(
            summary["actual_memory_MiB_s"],
            actual_memory_summary["actual_memory_MiB_s"],
            rel_tol=1e-10,
            abs_tol=1e-6,
        ),
        f"{folder.name}: actual memory summary mismatch",
    )

    with (
            folder / "mab-decisions.log"
    ).open(
        "x",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(
                decisions["raw_selected_execution_lines"]
            ) + "\n"
        )

    write_new(
        folder / "memory-summary.json",
        summary,
        )

    with (
            folder / "container-residency.csv"
    ).open(
        "x",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    with (
            folder / "container-actual-memory.csv"
    ).open(
        "x",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(actual_memory_rows[0]),
        )

        writer.writeheader()
        writer.writerows(actual_memory_rows)

    files = {
        str(path.relative_to(folder)): sha(path)

        for path in folder.rglob("*")

        if path.is_file()
           and path.name not in (
               "status.json",
               "verified.json",
           )
    }

    write_new(
        folder / "verified.json",
        {
            "state": "PASS",
            "validation_class": summary["validation_class"],
            "fallbacks": decisions["fallbacks"],
            "sha256": files,
            "containers": summary["created_containers"],
            "actual_memory_containers":
                summary["actual_memory_container_count"],
            "target": target,
            "condition": cid,
            "requests": 50,
        },
        )

    status_path = folder / "status.json"

    info = load(status_path)

    need(
        info.get("status") == "STARTED",
        f"{folder.name}: status is not STARTED",
        )

    info["status"] = "VERIFIED_LOCAL"

    info["validation_class"] = (
        summary["validation_class"]
    )

    info["fallbacks"] = decisions["fallbacks"]

    info["completed_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    replace_json(
        status_path,
        info,
    )

    need(
        verified(folder),
        f"{folder.name}: generated checkpoint verification failed",
    )


    print(
        f"PASS {folder.name}: "
        f"{summary['created_containers']} containers; "
        f"configured={summary['total_memory_MB_s']:.3f} MB.s; "
        f"actual={summary['actual_memory_MiB_s']:.3f} MiB.s; "
        f"fallbacks={decisions['fallbacks']} "
        f"({summary['validation_class']})",
        flush=True,
    )

    return summary


def collect_one(doc: dict, dest: Path, target: str, cid: str, replica: int,
                concurrency: int, *, require_multi: bool = False) -> dict:
    key = f'{target}-{cid}-r{replica:02d}-q{concurrency}'
    need(SAFE_NAME.fullmatch(key) is not None, f'Unsafe run ID {key}')
    folder = dest / 'runs' / key
    need(not folder.exists(), f'Existing run, refusing overwrite: {folder}')
    folder.mkdir(parents=True)
    stamp = doc['run_id']+'-'+key
    info = {'key':key,'target':target,'condition':cid,'replica':replica,
            'concurrency':concurrency,'requests':50,'status':'STARTED',
            'runtime_pin':PIN,'started_utc':datetime.now(timezone.utc).isoformat()}
    replace_json(folder/'status.json',info)
    # Start each condition from fresh MAB state and clean container pool.
    fresh_runtime(doc,target,cid,stamp)
    catalog = registration_catalog()
    register_target(target,catalog)
    prior_sha = apply_prior(target,cid,stamp,folder)
    info['prior_request_sha256'] = prior_sha
    assert_no_target_container_created(target, phase='prior/no-TL initialization')
    remote_client = '/tmp/mem-'+stamp+'-client.py'
    remote_output = '/tmp/mem-'+stamp+'-requests.jsonl'
    client = ROOT/'analysis/profiling/gcp_memory_workload.py'
    upload(client,'sl-workload',remote_client)
    ip = internal_ip('sl-lb')
    remote('sl-workload', f'python3 {shlex.quote(remote_client)} '
          f'--lb-ip {shlex.quote(ip)} --target {shlex.quote(target)} '
          f'--count 50 --concurrency {concurrency} --output {shlex.quote(remote_output)}', timeout=2700)
    download('sl-workload',remote_output,folder/'requests.jsonl')
    # Validated from raw local data, including client success, MAB decision
    # identity, actual execution architecture, and container create/destroy.
    # Fallback is a reported outcome, not an automatic failure.
    # Wait until all cold-created containers expire naturally and log destruction.
    max_wait = max(90, EXPIRATION_S + JANITOR_S + 30)
    for elapsed in range(0, max_wait, 5):
        logs = {}
        for host in WORKERS:
            logs[host] = remote(host, 'sudo cat /var/log/serverledge-node.log', timeout=90)
        c = sum(len([e for e in parse_event_lines(s, host, target) if e['event'] == 'container_created'])
                for host, s in logs.items())
        d = sum(len([e for e in parse_event_lines(s, host, target) if e['event'] == 'container_destroyed'])
                for host, s in logs.items())
        failures = sum(len([e for e in parse_event_lines(s, host, target)
                            if e['event'] == 'container_destroy_failed'])
                       for host, s in logs.items())
        if c and c == d and failures == 0:
            break
        time.sleep(5)
    else:
        raise CampaignError(f'{key}: timed out waiting natural container expiry '
                            f'(created={c}, destroyed={d}, failed={failures})')
    for host in WORKERS:
        run_remote_file(host, '/var/log/serverledge-node.log', folder / 'raw' / f'{host}.log')
    run_remote_file('sl-lb', '/var/log/serverledge-lb.log', folder / 'raw' / 'sl-lb.log')
    return finalize_local_measurement(folder, target, cid, concurrency,
                                      require_multi=require_multi)


def inspect_existing_run(doc: dict, dest: Path, key: str, *, recover: bool = False) -> dict:
    """Audit (or recover) one STARTED/verified run without touching GCP.

    A recovery never reruns invocations, edits the original logs, or
    changes MAB/prior parameters. Reject unrelated paths and unexpected
    status/manifest metadata. Evidence is stored beneath the existing run.
    """
    need(SAFE_NAME.fullmatch(key) is not None, 'Unsafe run key')
    folder = dest / 'runs' / key
    need(folder.is_dir(), f'Missing run directory: {folder}')
    state = load(folder / 'status.json')
    need(state.get('key') == key and state.get('runtime_pin') == PIN
         and state.get('requests') == 50, f'{key}: invalid run metadata')
    target = state.get('target')
    cid = state.get('condition')
    q = state.get('concurrency')
    replica = state.get('replica')
    need(isinstance(q, int) and 1 <= q <= 50, f'{key}: invalid concurrency')
    need(target in doc['targets'] and cid in doc['configs']
         and key == f'{target}-{cid}-r{replica:02d}-q{q}'
         and key in set(expected_keys(doc, q)),
         f'{key}: manifest fields do not match run identity')
    need(state.get('status') in ('STARTED', 'VERIFIED_LOCAL'),
         f'{key}: unexpected status {state.get("status")}')
    summary, rows, decisions = validate_local_measurement(folder, target, cid, q)
    print(f'AUDIT {key}: requests=50, decisions=50, lifecycle={len(rows)} closed, '
          f'fallbacks={decisions["fallbacks"]}, '
          f'class={summary["validation_class"]}', flush=True)
    if not recover:
        return summary
    need(state['status'] == 'STARTED' and not (folder / 'verified.json').exists(),
         f'{key}: already finalized; recovery refuses overwrite')
    # Durable proof of the exact inputs used in the recovery. Leave raw evidence
    # and any archived earlier attempts untouched.
    evidence_files = [folder / 'status.json', folder / 'requests.jsonl',
                      folder / 'raw/sl-lb.log'] + [
                          folder / 'raw' / f'{host}.log' for host in WORKERS]
    response = folder / 'transfer-response.json'
    if response.is_file():
        evidence_files.append(response)
    write_new(folder / 'local-recovery-provenance.json', {
        'schema_version': 1,
        'recovery_kind': 'OFFLINE_VALIDATION_OF_EXISTING_MEASUREMENT',
        'no_cloud_calls': True,
        'recovered_utc': datetime.now(timezone.utc).isoformat(),
        'raw_input_sha256': {str(p.relative_to(folder)): sha(p) for p in evidence_files},
        'original_status': state,
        'comparison_class': decisions['comparison_class'],
    })
    return finalize_local_measurement(folder, target, cid, q)


def audit_local_grid(
        doc: dict,
        dest: Path,
        concurrency: int,
) -> None:

    need(
        1 <= concurrency <= 50,
        'Invalid audit grid concurrency'
    )

    total_fallbacks = 0

    for key in expected_keys(doc, concurrency):
        folder = dest / 'runs' / key

        need(
            verified(folder),
            f'Not locally verified: {key}'
        )

        state = load(folder / 'status.json')

        summary, rows, decisions = validate_local_measurement(
            folder,
            state['target'],
            state['condition'],
            concurrency,
        )

        earlier = load(folder / 'memory-summary.json')

        need(
            math.isclose(
                summary['total_memory_MB_s'],
                earlier['total_memory_MB_s'],
                rel_tol=1e-10,
                abs_tol=1e-6,
            ),
            f'{key}: configured memory changed vs raw logs'
        )

        need(
            math.isclose(
                summary['actual_memory_MiB_s'],
                earlier['actual_memory_MiB_s'],
                rel_tol=1e-10,
                abs_tol=1e-6,
            ),
            f'{key}: actual memory changed vs raw logs'
        )

        need(
            summary['actual_memory_container_count']
            == summary['created_containers'],
            f'{key}: missing monitored containers'
        )

        need(
            decisions['fallbacks']
            == earlier['mab']['fallbacks'],
            f'{key}: fallback count changed vs summary'
        )

        total_fallbacks += decisions['fallbacks']

        print(
            f'PASS {key}: '
            f'{len(rows)} containers, '
            f'configured={summary["total_memory_MB_s"]:.3f} MB.s, '
            f'actual={summary["actual_memory_MiB_s"]:.3f} MiB.s, '
            f'fallbacks={decisions["fallbacks"]}',
            flush=True
        )

    expected = len(expected_keys(doc, concurrency))

    print(
        f'PASS OFFLINE GRID q={concurrency}: '
        f'{expected} runs, '
        f'{expected * 50} requests, '
        f'{total_fallbacks} fallback(s) recorded',
        flush=True
    )

def verified(folder: Path) -> bool:
    v=folder/'verified.json'
    if not v.is_file() or not (folder/'status.json').is_file():
        return False
    doc=load(v)
    if doc.get('state')!='PASS' or load(folder/'status.json').get('status')!='VERIFIED_LOCAL':
        return False
    paths=doc.get('sha256',{})
    return bool(paths) and all((folder/p).is_file() and sha(folder/p)==digest for p,digest in paths.items())


def expected_keys(doc:dict, concurrency:int) -> list[str]:
    # Predeclared two-replica counterbalancing, reproducible after interruption.
    # Each condition is run once per target per replica; r02 runs reverse order.
    result=[]
    for r in range(1,doc['replicas']+1):
        targets=doc['targets'] if r==1 else list(reversed(doc['targets']))
        configs=doc['configs'] if r==1 else list(reversed(doc['configs']))
        result.extend(f'{target}-{cid}-r{r:02d}-q{concurrency}'
                      for target in targets for cid in configs)
    return result


def run_grid(doc: dict, dest: Path, *, concurrency: int,
             max_new_runs: int | None = None) -> bool:
    """Run one bounded batch, preserving all already verified checkpoints.

    Returns True only when the entire declared grid is locally verified.
    The full campaign MUST be batched: its 180 measurements can exceed the
    independent eight-hour instance STOP limit if launched in one session.
    """
    need(doc.get('deployed') is True, 'Deploy has not passed')
    need(1 <= concurrency <= 50, 'Invalid concurrency')
    if max_new_runs is not None:
        need(isinstance(max_new_runs, int) and 1 <= max_new_runs <= 20,
             '--max-runs must be between 1 and 20')
    if doc.get('mode') == 'full':
        need(max_new_runs is not None,
             'Full campaign requires --max-runs; refuse unbounded 180-run session')
        locked = doc.get('run_concurrency')
        need(locked is None or locked == concurrency,
             f'Full campaign locked to concurrency {locked}, requested {concurrency}')
        if locked is None:
            doc['run_concurrency'] = concurrency
            replace_json(dest / 'campaign.json', doc)
    elif max_new_runs is not None:
        raise CampaignError('Pilot must run its declared complete three-condition grid')

    keys = expected_keys(doc, concurrency)
    completed_new: list[str] = []
    for key in keys:
        folder = dest / 'runs' / key
        if folder.exists():
            need(verified(folder), f'{key}: incomplete/corrupted run, refuse silent resume')
            print(f'SKIP verified {key}', flush=True)
            continue
        if max_new_runs is not None and len(completed_new) >= max_new_runs:
            break
        target_c, rest = key.rsplit('-r', 1)
        replica = int(rest.split('-q')[0])
        cid = next(k for k in doc['configs'] if target_c.endswith('-' + k))
        target = target_c[:-len(cid)-1]
        collect_one(doc, dest, target, cid, replica, concurrency)
        need(verified(folder), f'{key}: collector returned without VERIFIED_LOCAL')
        completed_new.append(key)

    # All previously accepted measurements must remain valid as well.
    done = sum(verified(dest / 'runs' / key) for key in keys)
    complete = done == len(keys)
    if complete:
        doc['finished_grid_concurrency'] = concurrency
    doc.setdefault('batch_history', []).append({
        'utc': datetime.now(timezone.utc).isoformat(),
        'concurrency': concurrency,
        'new_verified_keys': completed_new,
        'cumulative_verified': done,
        'total_expected': len(keys),
        'complete': complete,
    })
    replace_json(dest / 'campaign.json', doc)
    print(f"PASS BATCH: {len(completed_new)} new VERIFIED_LOCAL; "
          f"{done}/{len(keys)} total; complete={complete}", flush=True)
    return complete


def salvage(dest:Path, doc:dict):
    # Read-only rescue of active logs; no claim of valid experiment completion.
    rescue=dest/'rescue'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rescue.mkdir(parents=True,exist_ok=True)
    for host,path in [(x,'/var/log/serverledge-node.log') for x in WORKERS]+[
        ('sl-lb','/var/log/serverledge-lb.log')]:
        try:
            run_remote_file(host,path,rescue/'raw'/f'{host}.log')
        except Exception as exc:
            print(f'WARN rescue {host}: {exc}',file=sys.stderr,flush=True)


def stop_owned(doc:dict):
    cloud=instances()
    # Recover also an instance created just before the launcher lost its connection,
    # as long as its *name* and *unique campaign ownership label* match.
    safe=[]
    for name in VMS:
        vm=cloud.get(name)
        if vm is None or (vm.get('labels') or {}).get('memrun')!=doc['run_id']:
            continue
        recorded=(doc.get('vm_ids') or {}).get(name)
        need(recorded is None or str(recorded)==str(vm['id']),
             f'{name}: recorded and actual VM ID differ, refusing STOP')
        safe.append(name)
    running=[n for n in safe if cloud[n].get('status') not in ('TERMINATED','STOPPING')]
    if running:
        gc('compute','instances','stop',*running,f'--zone={ZONE}','--quiet',timeout=900)
    cloud=instances()
    statuses={n:cloud[n]['status'] for n in safe if n in cloud}
    need(all(s=='TERMINATED' for s in statuses.values()),f'STOP not verified: {statuses}')
    print(f'PASS STOP: {statuses}',flush=True)



def resume_owned(doc: dict):
    """Restart only this campaign's existing STOPPED VMs; never recreate them.

    Failed runs remain separate and require explicit archival before rerun.
    This operation deliberately does not modify VM IDs, ownership labels,
    deployment records, or the set of experimental conditions.
    """
    need(doc.get('provisioned') is True and doc.get('deployed') is True,
         'Cannot resume before provision and deployment pass')
    need(set(doc.get('vm_ids', {})) == set(VMS),
         'Refusing resume: incomplete or extra VM IDs in campaign')
    cloud = vm_guard(doc, require_running=False)
    states = {name: cloud[name].get('status') for name in VMS}
    need(all(state == 'TERMINATED' for state in states.values()),
         f'Resume requires all five VMs TERMINATED; actual: {states}')
    print(f'START existing owned VMs: {list(VMS)}', flush=True)
    gc('compute', 'instances', 'start', *VMS, f'--zone={ZONE}', '--quiet', timeout=900)
    for _ in range(36):
        cloud = vm_guard(doc, require_running=False)
        states = {name: cloud[name].get('status') for name in VMS}
        if all(status == 'RUNNING' for status in states.values()):
            print('PASS resume: same five VM IDs and labels; all RUNNING', flush=True)
            return
        time.sleep(5)
    raise CampaignError(f'Resume start not fully verified: {states}')


def finish(doc:dict,dest:Path,concurrency:int):
    keys=expected_keys(doc,concurrency)
    need(doc.get('finished_grid_concurrency')==concurrency,'Grid has not completed')
    for key in keys:
        need(verified(dest/'runs'/key),f'Missing/corrupt local result: {key}')
    # If pilot was run at multiple concurrency levels, preserve/validate ALL of them.
    for folder in (dest/'runs').iterdir():
        if folder.is_dir():
            need(verified(folder), f'Unverified extra run: {folder.name}')
    cloud=vm_guard(doc,require_running=False)
    # Exact recorded numeric IDs, immutable ownership labels, and auto-delete disks.
    owned=[]
    boot_disks=set()
    for n in VMS:
        need(n in doc['vm_ids'],f'Missing VM ID for {n}')
        need(n in cloud and str(cloud[n]['id'])==str(doc['vm_ids'][n]),f'Changed VM ID for {n}')
        disks=cloud[n].get('disks',[])
        need(len(disks)==1 and disks[0].get('boot') is True,
             f'{n}: unexpected attached disk layout')
        disk=disks[0]
        need(disk.get('autoDelete') is True,f'{n}: boot disk not auto-delete')
        boot_disks.add(disk['source'].rsplit('/',1)[-1])
        owned.append(n)
    final={'state':'LOCAL_EXPORT_CHECKSUMS_PASS','keys':keys,'run_id':doc['run_id'],
           'observations':len(keys)*50,'verified_utc':datetime.now(timezone.utc).isoformat()}
    write_new(dest/'final-verification.json',final)
    print(f'PASS {len(keys)} runs, {len(keys)*50} requests, locally verified: DELETE owned VM only',flush=True)
    gc('compute','instances','delete',*owned,f'--zone={ZONE}','--quiet',timeout=1200)
    after=instances()
    need(not(set(owned)&set(after)),f'Owned VM still exists: {set(owned)&set(after)}')
    # Check no detached boot disks whose names equal owned VM names.
    disk_rows=json.loads(gc('compute','disks','list','--format=json'))
    leftovers=[d['name'] for d in disk_rows if d.get('name') in boot_disks]
    need(not leftovers,f'Boot disks remained: {leftovers}')
    final['state']='DELETE_VERIFIED'
    replace_json(dest/'final-verification.json',final)
    doc['deleted']=True
    replace_json(dest/'campaign.json',doc)
    print('PASS: all five campaign VMs deleted and known boot disks absent',flush=True)


def init_campaign(dest:Path, mode:str, configs:list[str], targets:list[str], replicas:int):
    p=preflight()
    need(len(configs)==len(set(configs)) and set(configs)<=VALID_CONDITIONS,'Invalid config subset')
    need(len(targets)==len(set(targets)) and set(targets)<=set(p['targets']),'Invalid target subset')
    need(0 < replicas <=2,'replicas must be 1 or 2')
    doc={'schema_version':1,'runtime_pin':PIN,'project':PROJECT,'zone':ZONE,
         'run_id':run_id_now(),'mode':mode,'targets':targets,'configs':configs,
         'replicas':replicas,'target_selection_csv_sha256':p['target_selection_csv_sha256'],
         'created_utc':datetime.now(timezone.utc).isoformat(),
         'invariants':{'prewarm_target':False,'max_concurrency_container':1,
                       'expiry_seconds':EXPIRATION_S,'janitor_seconds':JANITOR_S},
         'vm_ids':{},'provisioned':False,'deployed':False,'deleted':False}
    need(not dest.exists(),'Campaign output dir exists; no overwrite')
    dest.mkdir(parents=True)
    store_campaign(dest,doc)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight','init-pilot','init-full','build','provision','deploy','resume','pilot','run','execute','finish','stop','abandon','audit-run','recover-run','audit-grid'))
    p.add_argument('--dest',type=Path)
    p.add_argument('--concurrency',type=int)
    p.add_argument('--max-runs',type=int,help='Mandatory full-campaign batch size (1..20 new verified runs per call)')
    p.add_argument('--key',help='Exact existing run ID for offline audit/recovery')
    p.add_argument('--confirm',action='store_true',help='Required for all GCP mutations')
    p.add_argument('--stop-after',action='store_true',
                   help='STOP owned VMs after successful pilot/run (also attempt STOP on failure)')
    p.add_argument('--discard-unverified-data',action='store_true',help='Manual emergency DELETE even if local data incomplete')
    a=p.parse_args()
    if a.action=='preflight':
        preflight();return
    need(a.dest is not None,'--dest is required')
    dest=a.dest.resolve()
    if a.action in ('init-pilot','init-full'):
        if a.action=='init-pilot':
            init_campaign(dest,'pilot',['N04'],['billibench-knn'],1)
        else:
            need(
                set(ACTUAL_MEMORY_CONFIG_IDS) <= VALID_CONDITIONS,
                "Missing No-TL/Ratio configurations",
            )

            init_campaign(
                dest,
                'full',
                list(ACTUAL_MEMORY_CONFIG_IDS),
                selected_targets(),
                2,
            )
        return
    doc=campaign(dest)
    if a.action == 'audit-grid':
        need(a.concurrency is not None, '--concurrency required for audit-grid')
        audit_local_grid(doc, dest, a.concurrency)
        return
    if a.action in ('audit-run', 'recover-run'):
        need(a.key is not None, '--key is required for offline audit/recovery')
        inspect_existing_run(doc, dest, a.key, recover=(a.action == 'recover-run'))
        return
    if a.action == 'build':
        build_runtime(dest)
        return
    need(a.confirm, 'Cloud mutation requires --confirm')
    try:
        if a.action=='provision':
            provision(doc,dest)
        elif a.action=='deploy':
            deploy(doc,dest)
        elif a.action=='resume':
            resume_owned(doc)
        elif a.action in ('run','pilot','execute'):
            need(a.concurrency is not None,'--concurrency must be provided')
            if a.action=='execute':
                need(doc['mode']=='full','execute is restricted to full campaigns')
            complete = run_grid(doc, dest, concurrency=a.concurrency,
                                max_new_runs=a.max_runs)
            if a.action == 'execute':
                # Each bounded session ends with STOP, until all checkpoints
                # exist; the final session verifies and DELETEs owned VMs.
                if complete:
                    finish(doc, dest, a.concurrency)
                else:
                    stop_owned(doc)
            elif a.stop_after:
                stop_owned(doc)
        elif a.action=='finish':
            need(a.concurrency is not None,'--concurrency must be provided')
            finish(doc,dest,a.concurrency)
        elif a.action=='stop':
            stop_owned(doc)
        elif a.action=='abandon':
            need(a.discard_unverified_data,'abandon requires --discard-unverified-data')
            cloud=instances()
            owned=[]
            for name in VMS:
                vm=cloud.get(name)
                if vm is None or (vm.get('labels') or {}).get('memrun')!=doc['run_id']:
                    continue
                record=(doc.get('vm_ids') or {}).get(name)
                need(record is None or str(record)==str(vm['id']),
                     f'{name}: changed VM ID in emergency cleanup')
                owned.append(name)
            if owned:
                gc('compute','instances','delete',*owned,f'--zone={ZONE}','--quiet',timeout=1200)
            remaining=instances()
            need(not (set(owned)&set(remaining)), 'Emergency DELETE not verified')
            doc['deleted']=True
            doc['abandoned_incomplete']=True
            replace_json(dest/'campaign.json',doc)
            print('PASS emergency DELETE; local dataset explicitly marked INCOMPLETE')
    except BaseException as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        # On Ctrl+C, do not spend minutes attempting SCP rescue while the
        # instances keep billing. The logs remain on the persistent VM disks.
        if not isinstance(exc, KeyboardInterrupt):
            try:
                salvage(dest, doc)
            except BaseException as rescue_exc:
                print(f'WARN salvage: {rescue_exc}', file=sys.stderr, flush=True)
        # A *second* Ctrl+C used to interrupt the old emergency STOP. Shield
        # this cleanup from ordinary terminal SIGINT/SIGTERM. SIGKILL, power
        # loss, or cloud/network failure can still prevent a verified STOP;
        # keep the independent GCP max-run-duration failsafe.
        prior_handlers = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                prior_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_owned(doc)
            except BaseException as stop_exc:
                print(f'CRITICAL: GCP STOP not verified: {stop_exc}',
                      file=sys.stderr, flush=True)
        finally:
            for sig, handler in prior_handlers.items():
                signal.signal(sig, handler)
        raise


if __name__=='__main__':
    main()
