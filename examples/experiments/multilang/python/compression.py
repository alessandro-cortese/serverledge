import zlib


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _make_payload(size_bytes):
    block_size = 65536
    block = bytearray(block_size)
    state = 0xC0FFEE12

    for i in range(block_size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF

        if i % 32 < 24:
            block[i] = (i * 17 + (i >> 3)) & 0xFF
        else:
            block[i] = (state >> 24) & 0xFF

    repetitions, remainder = divmod(size_bytes, block_size)
    return bytes(block) * repetitions + bytes(block[:remainder])


def _fnv1a(data):
    checksum = 2166136261

    for byte in data:
        checksum ^= byte
        checksum = (checksum * 16777619) & 0xFFFFFFFF

    return checksum


def handler(params, context):
    params = params or {}

    size_mb = _int_param(params, "size_mb", 4, 1, 16)
    rounds = _int_param(params, "rounds", 8, 1, 30)
    level = _int_param(params, "level", 6, 1, 9)

    payload = _make_payload(size_mb * 1024 * 1024)
    expected_checksum = _fnv1a(payload)

    compressed_bytes = 0
    checksum = 0

    for r in range(rounds):
        compressed = zlib.compress(payload, level)
        restored = zlib.decompress(compressed)

        if len(restored) != len(payload):
            raise RuntimeError("decompressed payload size mismatch")

        current = _fnv1a(restored)

        if current != expected_checksum:
            raise RuntimeError("decompressed payload checksum mismatch")

        compressed_bytes += len(compressed)
        checksum = (checksum + current + r) & 0xFFFFFFFF

    return {
        "benchmark": "compression",
        "size_mb": size_mb,
        "rounds": rounds,
        "level": level,
        "input_bytes": len(payload) * rounds,
        "compressed_bytes": compressed_bytes,
        "payload_checksum": expected_checksum,
        "checksum": checksum,
    }
