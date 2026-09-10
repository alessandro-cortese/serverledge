from array import array


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}

    elements = _int_param(
        params,
        "elements",
        3_000_000,
        100_000,
        4_000_000,
    )

    rounds = _int_param(
        params,
        "rounds",
        40,
        1,
        200,
    )

    data = array(
        "I",
        (
            (i * 2654435761) & 0xFFFFFFFF
            for i in range(elements)
        ),
    )

    mask = 0xFFFFFFFF
    checksum = 0

    for r in range(rounds):
        step = 97 + (r % 13)
        offset = (r * 8191) % elements

        for i in range(0, elements, step):
            idx = (i + offset) % elements
            value = data[idx]

            value = (
                            value
                            ^ ((idx + r) * 2246822519)
                    ) & mask

            data[idx] = value

            checksum = (
                               checksum + value
                       ) & mask

        local = 0

        for i in range(0, elements, 4):
            local = (
                            local + data[i]
                    ) & mask

        checksum ^= local

    return {
        "benchmark": "memory_rw",
        "elements": elements,
        "rounds": rounds,
        "bytes": elements * data.itemsize,
        "checksum": checksum,
    }