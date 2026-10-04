"""
ServerlessBench Testcase1 - Resource Efficiency - ALU.

Source:
  ServerlessBench/Testcase1-Resource-efficiency/code/alu.py

Adaptation for Serverledge:
- AWS Lambda wrapper removed.
- boto3/S3 phase is not part of this ALU function and is not included.
- Original ALU multiprocessing structure is preserved:
    10,000,000 total iterations
    100 worker processes
- Python 3.14/forkserver compatible.

The computational kernel and original default workload size are preserved.
"""

import multiprocessing as mp
import random


DEFAULT_LOOP_TIME = 10_000_000
DEFAULT_PARALLEL_INDEX = 100


def _do_alu(times, child_conn, client_id):
    a = random.randint(10, 100)
    b = random.randint(10, 100)

    temp = 0

    for i in range(times):
        if i % 4 == 0:
            temp = a + b
        elif i % 4 == 1:
            temp = a - b
        elif i % 4 == 2:
            temp = a * b
        else:
            temp = a / b

    child_conn.send(temp)
    child_conn.close()


def _alu(times, parallel_index):
    per_times = int(times / parallel_index)

    processes = []
    parent_conns = []

    for i in range(parallel_index):
        parent_conn, child_conn = mp.Pipe()

        parent_conns.append(parent_conn)

        process = mp.Process(
            target=_do_alu,
            args=(per_times, child_conn, i),
        )

        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results = []

    for connection in parent_conns:
        results.append(connection.recv())
        connection.close()

    return results


def handler(params, context):
    params = params or {}

    loop_time = int(
        params.get(
            "loop_time",
            DEFAULT_LOOP_TIME,
        )
    )

    parallel_index = int(
        params.get(
            "parallel_index",
            DEFAULT_PARALLEL_INDEX,
        )
    )

    if loop_time <= 0:
        raise ValueError("loop_time must be > 0")

    if parallel_index <= 0:
        raise ValueError("parallel_index must be > 0")

    results = _alu(
        loop_time,
        parallel_index,
    )

    return {
        "benchmark": "serverlessbench-alu",
        "loop_time": loop_time,
        "parallel_index": parallel_index,
        "results_count": len(results),
        "checksum": round(
            sum(float(x) for x in results),
            9,
        ),
    }
