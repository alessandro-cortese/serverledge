import base64


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}

    size_mb = _int_param(params, "size_mb", 16, 1, 64)
    rounds = _int_param(params, "rounds", 30, 1, 100)
    size = size_mb * 1024 * 1024

    pattern = bytes(range(256))
    source = (pattern * ((size + 255) // 256))[:size]

    total_decoded = 0
    checksum = 0

    for _ in range(rounds):
        encoded = base64.b64encode(source)
        decoded = base64.b64decode(encoded, validate=True)

        total_decoded += len(decoded)
        checksum = (checksum + decoded[0] + decoded[-1] + len(decoded)) & 0xFFFFFFFF

    return {
        "benchmark": "base64stream",
        "size_mb": size_mb,
        "rounds": rounds,
        "processed_gb": total_decoded / 1024.0 / 1024.0 / 1024.0,
        "checksum": checksum,
    }
