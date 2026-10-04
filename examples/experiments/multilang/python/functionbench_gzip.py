"""
FunctionBench gzip_compression port for Serverledge.

Original workload:
  FunctionBench/aws/disk/gzip_compression

The workload preserves:
  1. generation/write of file_size MiB using os.urandom
  2. gzip compression of that file in /tmp
"""

import gzip
import os


INPUT_PATH = "/tmp/functionbench-gzip-input"
OUTPUT_PATH = "/tmp/functionbench-gzip-result.gz"


def handler(params, context):
    params = params or {}

    file_size = int(params.get("file_size", 16))

    if file_size <= 0:
        raise ValueError("file_size must be > 0")

    try:
        with open(INPUT_PATH, "wb") as f:
            f.write(os.urandom(file_size * 1024 * 1024))

        with open(INPUT_PATH, "rb") as src:
            with gzip.open(OUTPUT_PATH, "wb") as dst:
                dst.writelines(src)

        return {
            "benchmark": "functionbench-gzip",
            "file_size_mb": file_size,
            "input_bytes": os.path.getsize(INPUT_PATH),
            "compressed_bytes": os.path.getsize(OUTPUT_PATH),
        }

    finally:
        for path in (INPUT_PATH, OUTPUT_PATH):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
