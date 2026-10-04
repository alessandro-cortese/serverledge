import random


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def alu(times):
    # Preserved from ServerlessBench Testcase2 Parallel-composition,
    # sequential.py.
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

    return temp


def handler(params, context):
    params = params or {}

    # The original ServerlessBench aluEvent.json uses n = 10,000,000.
    times = _int_param(params, "n", 10_000_000, 1, 100_000_000)

    result = alu(times)

    return {
        "benchmark": "serverlessbench-alu",
        "n": times,
        "result": result,
    }
