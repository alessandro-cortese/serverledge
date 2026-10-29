#!/usr/bin/env python3
"""Workload client executed on sl-workload. One response JSON line per request.

No target ground-truth knowledge and no synthetic prewarm. A request is sent via
architecture-aware LB rather than directly to a worker. Timeouts are explicit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def invoke(host: str, target: str, index: int) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    request = Request(
        f'http://{host}:1323/invoke/{target}',
        data=b'{"params":{}}',
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=300) as response:
        raw = response.read()
        arch = (response.headers.get('Serverledge-Node-Arch') or '').strip()
    elapsed = 1000.0 * (time.perf_counter() - start)
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise RuntimeError(f'{index}: response non-JSON object')
    if document.get('Success') is not True:
        raise RuntimeError(f'{index}: invocation failed: {document!r}')
    if arch not in ('amd64', 'arm64'):
        raise RuntimeError(f'{index}: missing/unknown Serverledge-Node-Arch: {arch!r}')
    return {'index': index, 'submitted_utc': started_at, 'client_elapsed_ms': elapsed,
            'execution_header_arm': arch, 'is_warm_start': document.get('IsWarmStart'),
            'duration_seconds': document.get('Duration'),
            'response_time_seconds': document.get('ResponseTime'),
            'success': True, 'response': document}


def execute(host: str, target: str, count: int, concurrency: int, output: Path) -> None:
    if count != 50 or not 1 <= concurrency <= 50:
        raise ValueError('Expected 50 requests with concurrency 1..50')
    if output.exists():
        raise FileExistsError(output)
    # Atomic export: no complete-looking file until all 50 responses succeed.
    temp = output.with_name(output.name + '.partial')
    if temp.exists():
        raise FileExistsError(temp)
    responses = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(invoke, host, target, i): i for i in range(1, count + 1)}
            for future in as_completed(futures):
                responses.append(future.result())
                print(f'PASS request {len(responses)}/{count} index={futures[future]}', flush=True)
        with temp.open('w', encoding='utf-8') as out:
            for response in sorted(responses, key=lambda r: r['index']):
                out.write(json.dumps(response, sort_keys=True) + '\n')
            out.flush()
            os.fsync(out.fileno())
        temp.replace(output)
    except BaseException:
        # Leave a diagnostic marker; the supervisor snapshots LB and node logs.
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lb-ip', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--concurrency', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    execute(a.lb_ip, a.target, a.count, a.concurrency, a.output)


if __name__ == '__main__':
    main()
