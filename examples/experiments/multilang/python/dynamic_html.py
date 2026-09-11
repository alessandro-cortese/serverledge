def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _fnv1a_update(checksum, data):
    for byte in data.encode("utf-8"):
        checksum ^= byte
        checksum = (checksum * 16777619) & 0xFFFFFFFF
    return checksum


def handler(params, context):
    params = params or {}

    entries = _int_param(params, "entries", 12_000, 500, 30_000)
    rounds = _int_param(params, "rounds", 24, 1, 60)
    username = str(params.get("username", "serverledge-user"))

    total_bytes = 0
    checksum = 2166136261

    for r in range(rounds):
        parts = [
            "<!doctype html><html><head><title>Serverless Benchmark</title></head><body>",
            f"<h1>Hello {username}</h1>",
            f'<section data-round="{r}"><ul>',
        ]

        state = (0x12345678 ^ r) & 0xFFFFFFFF

        for i in range(entries):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            value = state % 1_000_000
            parts.append(
                f'<li data-index="{i}" data-value="{value}">item-{i:06d}:{value:06d}</li>'
            )

        parts.append("</ul></section></body></html>")
        html = "".join(parts)

        total_bytes += len(html.encode("utf-8"))
        checksum = _fnv1a_update(checksum, html)

    return {
        "benchmark": "dynamic_html",
        "entries": entries,
        "rounds": rounds,
        "total_bytes": total_bytes,
        "checksum": checksum,
    }
