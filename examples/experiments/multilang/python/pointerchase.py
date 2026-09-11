from array import array

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

    nodes = _int_param(params, "nodes", 8_000_000, 2, 20_000_000)
    steps = _int_param(params, "steps", 60_000_000, 1, 200_000_000)

    # Keep int32 storage, matching the Go benchmark's []int32 representation.
    nxt = array("i", range(nodes))
    state = 88172645463325252

    # Same deterministic Fisher-Yates shuffle used by the Go benchmark.
    for i in range(nodes - 1, 0, -1):
        state = _xorshift64(state)
        j = state % (i + 1)
        nxt[i], nxt[j] = nxt[j], nxt[i]

    current = 0
    for _ in range(steps):
        current = nxt[current]

    return {
        "benchmark": "pointerchase",
        "nodes": nodes,
        "steps": steps,
        "final": int(current),
    }
