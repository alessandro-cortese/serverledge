import json


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def handler(params, context):
    params = params or {}
    records = _int_param(params, "records", 12000, 1000, 50000)
    rounds = _int_param(params, "rounds", 20, 1, 100)

    payload = [
        {
            "id": i,
            "group": i % 97,
            "active": (i % 3) != 0,
            "score": ((i * 2654435761) & 0xFFFFFFFF) / 4294967296.0,
            "name": f"record-{i:06d}",
            "tags": [i % 11, i % 17, i % 23],
        }
        for i in range(records)
    ]

    checksum = 0
    encoded_bytes = 0

    for _ in range(rounds):
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)

        encoded_bytes += len(encoded)
        checksum = (
                           checksum
                           + decoded[-1]["id"]
                           + decoded[records // 2]["group"]
                   ) & 0xFFFFFFFF

    return {
        "benchmark": "json_dumps",
        "records": records,
        "rounds": rounds,
        "encoded_bytes": encoded_bytes,
        "checksum": checksum,
    }