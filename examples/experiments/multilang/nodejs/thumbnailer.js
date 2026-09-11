const ONE = 1 << 16;
const DEN = 4294967296;
const HALF_DEN = 2147483648;

function intParam(params, name, defaultValue, minimum, maximum) {
  const raw = params && params[name] !== undefined ? Number(params[name]) : defaultValue;
  const value = Number.isFinite(raw) ? Math.trunc(raw) : defaultValue;
  return Math.max(minimum, Math.min(maximum, value));
}

function generateRgb(width, height) {
  const image = Buffer.allocUnsafe(width * height * 3);
  let pos = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      image[pos] = (x * 13 + y * 7 + (x ^ y)) & 0xff;
      image[pos + 1] = (x * 3 + y * 17 + x * y) & 0xff;
      image[pos + 2] = (x * 19 + y * 5 + (x + y) * 11) & 0xff;
      pos += 3;
    }
  }
  return image;
}

function axisMap(srcSize, dstSize) {
  if (dstSize <= 1) {
    return [[0, 0, 0]];
  }
  const scale = Math.floor(((srcSize - 1) * ONE) / (dstSize - 1));
  const result = new Array(dstSize);
  for (let d = 0; d < dstSize; d++) {
    const fp = d * scale;
    const lo = Math.floor(fp / ONE);
    const frac = fp % ONE;
    const hi = lo + 1 < srcSize ? lo + 1 : lo;
    result[d] = [lo, hi, frac];
  }
  return result;
}

function resizeBilinear(src, srcWidth, srcHeight, dstWidth, dstHeight) {
  const xmap = axisMap(srcWidth, dstWidth);
  const ymap = axisMap(srcHeight, dstHeight);
  const out = Buffer.allocUnsafe(dstWidth * dstHeight * 3);
  let outPos = 0;
  for (const [y0, y1, fy] of ymap) {
    const wy0 = ONE - fy;
    const wy1 = fy;
    const row0 = y0 * srcWidth * 3;
    const row1 = y1 * srcWidth * 3;
    for (const [x0, x1, fx] of xmap) {
      const wx0 = ONE - fx;
      const wx1 = fx;
      const w00 = wx0 * wy0;
      const w10 = wx1 * wy0;
      const w01 = wx0 * wy1;
      const w11 = wx1 * wy1;
      const p00 = row0 + x0 * 3;
      const p10 = row0 + x1 * 3;
      const p01 = row1 + x0 * 3;
      const p11 = row1 + x1 * 3;
      for (let channel = 0; channel < 3; channel++) {
        const weighted =
          src[p00 + channel] * w00 + src[p10 + channel] * w10 + src[p01 + channel] * w01 + src[p11 + channel] * w11;
        out[outPos] = Math.floor((weighted + HALF_DEN) / DEN);
        outPos += 1;
      }
    }
  }
  return out;
}

function fnv1a32(data) {
  let value = 2166136261 >>> 0;
  for (const byte of data) {
    value ^= byte;
    value = Math.imul(value, 16777619) >>> 0;
  }
  return value >>> 0;
}

module.exports = async function handler(params, context) {
  params = params || {};
  const srcWidth = intParam(params, "src_width", 1024, 64, 4096);
  const srcHeight = intParam(params, "src_height", 768, 64, 4096);
  const dstWidth = intParam(params, "dst_width", 320, 16, srcWidth);
  const dstHeight = intParam(params, "dst_height", 240, 16, srcHeight);
  const rounds = intParam(params, "rounds", 8, 1, 50);
  const source = generateRgb(srcWidth, srcHeight);
  let resized = Buffer.alloc(0);
  for (let round = 0; round < rounds; round++) {
    resized = resizeBilinear(source, srcWidth, srcHeight, dstWidth, dstHeight);
  }
  return {
    benchmark: "thumbnailer",
    source_width: srcWidth,
    source_height: srcHeight,
    thumbnail_width: dstWidth,
    thumbnail_height: dstHeight,
    rounds,
    processed_pixels: dstWidth * dstHeight * rounds,
    output_bytes: resized.length,
    checksum: fnv1a32(resized),
  };
};
