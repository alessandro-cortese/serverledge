def handler(params, context):
    megabytes = int(params.get("megabytes", 48))
    size = megabytes * 1024 * 1024

    data = bytearray(size)

    checksum = 0

    for offset in range(0, size, 4096):
        value = (offset // 4096) % 251
        data[offset] = value
        checksum = (checksum + data[offset]) & 0xFFFFFFFF

    return "allocated_mb={}, checksum={}".format(
        megabytes,
        checksum,
    )