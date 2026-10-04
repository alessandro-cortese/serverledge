"""
FunctionBench sequential_disk_io port for Serverledge.

Original workload:
  FunctionBench/aws/disk/sequential_disk_io

Compatibility adaptation:
binary EOF comparison is changed from "" to b"" for Python 3.
"""

import os


FILE_PATH = "/tmp/functionbench-sequential-disk"


def handler(params, context):
    params = params or {}

    file_size = int(params.get("file_size", 64))
    byte_size = int(params.get("byte_size", 1024 * 1024))

    if file_size <= 0:
        raise ValueError("file_size must be > 0")

    if byte_size <= 0:
        raise ValueError("byte_size must be > 0")

    try:
        with open(FILE_PATH, "wb", buffering=byte_size) as f:
            f.write(os.urandom(file_size * 1024 * 1024))
            f.flush()
            os.fsync(f.fileno())

        bytes_read = 0

        with open(FILE_PATH, "rb", buffering=byte_size) as f:
            chunk = f.read(byte_size)

            while chunk != b"":
                bytes_read += len(chunk)
                chunk = f.read(byte_size)

        return {
            "benchmark": "functionbench-sequential-disk-io",
            "file_size_mb": file_size,
            "byte_size": byte_size,
            "bytes_read": bytes_read,
        }

    finally:
        try:
            os.remove(FILE_PATH)
        except FileNotFoundError:
            pass
