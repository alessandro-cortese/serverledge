"""
FunctionBench matrix multiplication port for Serverledge.

Original workload:
  FunctionBench/aws/cpu-memory/matmul

The original NumPy matrix multiplication kernel is preserved.
A deterministic seed is used for architecture-comparable inputs.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


def handler(params, context):
    params = params or {}

    n = int(params.get("n", 1200))
    seed = int(params.get("seed", 42))

    if n <= 0:
        raise ValueError("n must be > 0")

    rng = np.random.RandomState(seed)

    A = rng.rand(n, n)
    B = rng.rand(n, n)

    C = np.matmul(A, B)

    return {
        "benchmark": "functionbench-matmul",
        "n": n,
        "seed": seed,
        "checksum": float(C[0, 0] + C[-1, -1]),
    }
