from array import array

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

# Squiggle-style 2-bit DNA encoding: T=00, C=01, A=10, G=11.
BASES = b"TCAG"
BASE_TO_CODE = {
    ord("T"): 0,
    ord("C"): 1,
    ord("A"): 2,
    ord("G"): 3,
}


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


def _generate_sequence(length, seed):
    sequence = bytearray(length)
    state = seed & MASK64

    for i in range(length):
        state = _xorshift64(state)
        sequence[i] = BASES[state & 0x3]

    return sequence


def _transform_squiggle(sequence):
    xs = array("d", [0.0])
    ys = array("d", [0.0])
    x2 = 0
    y2 = 0
    checksum = 2166136261

    for base in sequence:
        code = BASE_TO_CODE[base]

        # Process the high bit first, then the low bit. Coordinates are stored
        # in half-step units so the core update can stay integer-based.
        for bit in ((code >> 1) & 1, code & 1):
            x2 += 1
            y2 += 1 if bit else -1

            xs.append(x2 * 0.5)
            ys.append(y2 * 0.5)

            checksum ^= (x2 * 31 + (y2 & MASK32)) & MASK32
            checksum = (checksum * 16777619) & MASK32

    return xs, ys, checksum, x2, y2


def handler(params, context):
    params = params or {}

    length = _int_param(params, "length", 1_000_000, 1_000, 5_000_000)
    rounds = _int_param(params, "rounds", 3, 1, 10)
    seed = _int_param(params, "seed", 88_172_645, 1, (1 << 31) - 1)

    sequence = _generate_sequence(length, seed)

    xs = None
    ys = None
    checksum = 0
    x2 = 0
    y2 = 0

    for _ in range(rounds):
        xs, ys, checksum, x2, y2 = _transform_squiggle(sequence)

    return {
        "benchmark": "dna_visualisation",
        "length": length,
        "rounds": rounds,
        "points": len(xs),
        "final_x": x2 * 0.5,
        "final_y": y2 * 0.5,
        "coordinate_bytes": len(xs) * 8 + len(ys) * 8,
        "checksum": checksum,
    }
