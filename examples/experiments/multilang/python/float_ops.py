import math


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}

    iterations = _int_param(params, "iterations", 6_000_000, 100_000, 20_000_000)

    acc = 0.0
    x = 0.123456789

    for i in range(1, iterations + 1):
        x = (x * 1.0000001192092896 + 0.00000035762786865) % 1.0
        acc += math.sin(x + (i % 31) * 0.001) * math.cos(x * 0.5)
        acc += math.sqrt(x + 1.0) / (1.0 + (i % 17))

    return {
        "benchmark": "float_ops",
        "iterations": iterations,
        "result": round(acc, 9),
    }
