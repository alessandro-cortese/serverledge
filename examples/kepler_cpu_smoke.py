import time


def handler(params, context):
    seconds = float(params.get("seconds", 2.5))

    if seconds <= 0:
        raise ValueError("seconds must be positive")

    deadline = time.perf_counter() + seconds
    checksum = 0

    while time.perf_counter() < deadline:
        for value in range(1, 10000):
            checksum = (
                checksum + value * value
            ) % 1000000007

    return {
        "seconds": seconds,
        "checksum": checksum,
    }
