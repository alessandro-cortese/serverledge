import json


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}

    records = _int_param(params, "records", 20_000, 100, 100_000)
    rounds = _int_param(params, "rounds", 20, 1, 100)

    source = []

    for i in range(records):
        source.append(
            {
                "id": i,
                "name": f"entity-{i}",
                "tags": ["alpha", "beta", "gamma"],
                "metadata": {"region": "eu", "tier": f"t{i % 5}"},
                "values": [float(i), float(i) * 1.5, float(i) * 2.25],
            }
        )

    total_bytes = 0
    total_parsed = 0
    checksum = 0

    for _ in range(rounds):
        encoded = json.dumps(source, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        total_bytes += len(encoded)

        decoded = json.loads(encoded)
        total_parsed += len(decoded)

        if decoded:
            checksum = (
                checksum + decoded[0]["id"] + decoded[-1]["id"] + len(decoded[-1]["tags"])
            ) & 0xFFFFFFFF

    return {
        "benchmark": "jsonparse",
        "records": records,
        "rounds": rounds,
        "total_parsed": total_parsed,
        "total_mb": total_bytes / 1024.0 / 1024.0,
        "checksum": checksum,
    }
