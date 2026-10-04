"""
FunctionBench random_disk_io port for Serverledge.

Original workload:
  FunctionBench/aws/disk/random_disk_io

Compatibility adaptation:
Python 2-style
    range(total_file_bytes / byte_size)
is converted to integer division // for Python 3.

A deterministic random seed is used for reproducible seek positions.
"""

import os
import random


FILE_PATH = "/tmp/functionbench-random-disk"


def handler(params, context):
    params = params or {}

    file_size = int(params.get("file_size", 32))
    byte_size = int(params.get("byte_size", 4096))
    seed = int(params.get("seed", 42))

    if file_size <= 0:
        raise ValueError("file_size must be > 0")

    if byte_size <= 0:
        raise ValueError("byte_size must be > 0")

    total_bytes = file_size * 1024 * 1024
    total_file_bytes = total_bytes - byte_size

    if total_file_bytes <= 0:
        raise ValueError("file_size must be larger than byte_size")

    iterations = total_file_bytes // byte_size

    rng = random.Random(seed)
    block = os.urandom(byte_size)

    try:
        with open(FILE_PATH, "wb") as f:
            for _ in range(iterations):
                f.seek(rng.randrange(total_file_bytes))
                f.write(block)

            f.flush()
            os.fsync(f.fileno())

        rng = random.Random(seed)

        bytes_read = 0

        with open(FILE_PATH, "rb") as f:
            for _ in range(iterations):
                f.seek(rng.randrange(total_file_bytes))
                data = f.read(byte_size)
                bytes_read += len(data)

        return {
            "benchmark": "functionbench-random-disk-io",
            "file_size_mb": file_size,
            "byte_size": byte_size,
            "iterations": iterations,
            "bytes_read": bytes_read,
            "seed": seed,
        }

    finally:
        try:
            os.remove(FILE_PATH)
        except FileNotFoundError:
            pass
