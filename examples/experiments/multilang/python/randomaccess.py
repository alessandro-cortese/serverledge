MASK64 = (1 << 64) - 1


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _xorshift64(state):
    state ^= (state << 13) & MASK64
    state &= MASK64
    state ^= state >> 7
    state &= MASK64
    state ^= (state << 17) & MASK64
    return state & MASK64


def handler(params, context):
    params = params or {}

    size_mb = _int_param(params, "size_mb", 256, 1, 512)
    accesses = _int_param(params, "accesses", 40_000_000, 1, 100_000_000)
    size = size_mb * 1024 * 1024

    # Equivalent to the Go initialization buffer[i] = byte(i), but avoids a
    # Python-level loop over the entire buffer.
    pattern = bytearray(range(256))
    repetitions, remainder = divmod(size, 256)
    buffer = pattern * repetitions

    if remainder:
        buffer.extend(pattern[:remainder])

    state = 88172645463325252
    checksum = 0

    for _ in range(accesses):
        state = _xorshift64(state)
        checksum += buffer[state % size]

    return {
        "benchmark": "randomaccess",
        "size_mb": size_mb,
        "accesses": accesses,
        "checksum": checksum,
    }
