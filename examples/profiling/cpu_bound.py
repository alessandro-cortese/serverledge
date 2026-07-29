def handler(params, context):
    iterations = int(params.get("iterations", 8_000_000))

    value = 2_166_136_261

    for i in range(iterations):
        value ^= i & 0xFFFFFFFF
        value = (value * 16_777_619) & 0xFFFFFFFF

    return "iterations={}, checksum={}".format(iterations, value)