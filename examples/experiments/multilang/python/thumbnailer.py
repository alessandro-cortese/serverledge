MASK32 = (1 << 32) - 1
ONE = 1 << 16
DEN = 1 << 32
HALF_DEN = 1 << 31


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _generate_rgb(width, height):
    image = bytearray(width * height * 3)
    pos = 0

    for y in range(height):
        for x in range(width):
            image[pos] = (x * 13 + y * 7 + (x ^ y)) & 0xFF
            image[pos + 1] = (x * 3 + y * 17 + (x * y)) & 0xFF
            image[pos + 2] = (x * 19 + y * 5 + ((x + y) * 11)) & 0xFF
            pos += 3

    return image


def _axis_map(src_size, dst_size):
    if dst_size <= 1:
        return [(0, 0, 0)]

    scale = ((src_size - 1) * ONE) // (dst_size - 1)
    result = []

    for d in range(dst_size):
        fp = d * scale
        lo = fp >> 16
        frac = fp & 0xFFFF
        hi = lo + 1 if lo + 1 < src_size else lo
        result.append((lo, hi, frac))

    return result


def _resize_bilinear(src, src_width, src_height, dst_width, dst_height):
    xmap = _axis_map(src_width, dst_width)
    ymap = _axis_map(src_height, dst_height)
    out = bytearray(dst_width * dst_height * 3)
    out_pos = 0

    for y0, y1, fy in ymap:
        wy0 = ONE - fy
        wy1 = fy
        row0 = y0 * src_width * 3
        row1 = y1 * src_width * 3

        for x0, x1, fx in xmap:
            wx0 = ONE - fx
            wx1 = fx
            w00 = wx0 * wy0
            w10 = wx1 * wy0
            w01 = wx0 * wy1
            w11 = wx1 * wy1

            p00 = row0 + x0 * 3
            p10 = row0 + x1 * 3
            p01 = row1 + x0 * 3
            p11 = row1 + x1 * 3

            for channel in range(3):
                weighted = (
                    src[p00 + channel] * w00
                    + src[p10 + channel] * w10
                    + src[p01 + channel] * w01
                    + src[p11 + channel] * w11
                )
                out[out_pos] = (weighted + HALF_DEN) // DEN
                out_pos += 1

    return out


def _fnv1a32(data):
    value = 2166136261

    for byte in data:
        value ^= byte
        value = (value * 16777619) & MASK32

    return value


def handler(params, context):
    params = params or {}

    src_width = _int_param(params, "src_width", 1024, 64, 4096)
    src_height = _int_param(params, "src_height", 768, 64, 4096)
    dst_width = _int_param(params, "dst_width", 320, 16, src_width)
    dst_height = _int_param(params, "dst_height", 240, 16, src_height)
    rounds = _int_param(params, "rounds", 8, 1, 50)

    source = _generate_rgb(src_width, src_height)
    resized = b""

    for _ in range(rounds):
        resized = _resize_bilinear(source, src_width, src_height, dst_width, dst_height)

    return {
        "benchmark": "thumbnailer",
        "source_width": src_width,
        "source_height": src_height,
        "thumbnail_width": dst_width,
        "thumbnail_height": dst_height,
        "rounds": rounds,
        "processed_pixels": dst_width * dst_height * rounds,
        "output_bytes": len(resized),
        "checksum": _fnv1a32(resized),
    }
