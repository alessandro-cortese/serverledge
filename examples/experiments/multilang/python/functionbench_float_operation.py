"""
FunctionBench float_operation port for Serverledge.

Original workload:
  FunctionBench/aws/cpu-memory/float_operation

The computational kernel is preserved:
for every i, compute sin(i), cos(i), sqrt(i).

Only the serverless-provider wrapper has been replaced.
"""

import math


def handler(params, context):
    params = params or {}
    n = int(params.get("n", 1_000_000))

    if n <= 0:
        raise ValueError("n must be > 0")

    sin_i = 0.0
    cos_i = 0.0
    sqrt_i = 0.0

    for i in range(n):
        sin_i = math.sin(i)
        cos_i = math.cos(i)
        sqrt_i = math.sqrt(i)

    return {
        "benchmark": "functionbench-float-operation",
        "n": n,
        "last_sin": sin_i,
        "last_cos": cos_i,
        "last_sqrt": sqrt_i,
    }
