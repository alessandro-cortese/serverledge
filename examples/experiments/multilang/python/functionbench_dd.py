"""
FunctionBench dd disk-I/O port for Serverledge.

Original workload:
  FunctionBench/aws/disk/dd

Uses the same Unix dd utility with /dev/zero and /tmp.
"""

import os
import subprocess


OUTPUT_PATH = "/tmp/functionbench-dd.out"


def handler(params, context):
    params = params or {}

    bs = str(params.get("bs", "1M"))
    count = int(params.get("count", 256))

    if count <= 0:
        raise ValueError("count must be > 0")

    try:
        result = subprocess.run(
            [
                "dd",
                "if=/dev/zero",
                f"of={OUTPUT_PATH}",
                f"bs={bs}",
                f"count={count}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        size = os.path.getsize(OUTPUT_PATH)

        return {
            "benchmark": "functionbench-dd",
            "bs": bs,
            "count": count,
            "bytes_written": size,
            "dd_summary": result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "",
        }

    finally:
        try:
            os.remove(OUTPUT_PATH)
        except FileNotFoundError:
            pass
