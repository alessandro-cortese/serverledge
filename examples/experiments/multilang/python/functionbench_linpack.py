"""
FunctionBench LINPACK port for Serverledge.

Original workload:
  FunctionBench/aws/cpu-memory/linpack

The original dense linear-system kernel is preserved.
A deterministic seed is used to make x86/ARM invocations comparable.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


def handler(params, context):
    params = params or {}

    n = int(params.get("n", 1250))
    seed = int(params.get("seed", 42))

    if n <= 0:
        raise ValueError("n must be > 0")

    rng = np.random.RandomState(seed)

    # Original FunctionBench construction:
    # random values in [-0.5, 0.5), B = row sums of A.
    A = rng.random_sample((n, n)) - 0.5
    B = A.sum(axis=1)

    A = np.matrix(A)
    B = np.matrix(B.reshape((n, 1)))

    x = np.linalg.solve(A, B)

    return {
        "benchmark": "functionbench-linpack",
        "n": n,
        "seed": seed,
        "solution_checksum": float(np.asarray(x).sum()),
    }
