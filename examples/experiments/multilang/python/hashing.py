import hashlib


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}

    block_kb = _int_param(params, "block_kb", 64, 1, 1024)
    iterations = _int_param(params, "iterations", 20_000, 1, 200_000)
    size = block_kb * 1024

    pattern = bytes(range(256))
    block = bytearray((pattern * ((size + 255) // 256))[:size])
    digest = bytes(32)

    for _ in range(iterations):
        block[:32] = digest
        digest = hashlib.sha256(block).digest()

    return {
        "benchmark": "hashing",
        "iterations": iterations,
        "processed_mb": (block_kb * iterations) / 1024.0,
        "final_digest": digest.hex(),
    }
