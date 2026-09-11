const zlib = require("zlib");

function intParam(params, name, defaultValue, min, max) {
  const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
  const value = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, value));
}

function makePayload(sizeBytes) {
  const blockSize = 65536;
  const block = Buffer.allocUnsafe(blockSize);
  let state = 0xc0ffee12 >>> 0;
  for (let i = 0; i < blockSize; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    if (i % 32 < 24) {
      block[i] = (i * 17 + (i >>> 3)) & 0xff;
    } else {
      block[i] = (state >>> 24) & 0xff;
    }
  }
  const payload = Buffer.allocUnsafe(sizeBytes);
  let offset = 0;
  while (offset < sizeBytes) {
    const length = Math.min(blockSize, sizeBytes - offset);
    block.copy(payload, offset, 0, length);
    offset += length;
  }
  return payload;
}

function fnv1a(data) {
  let checksum = 2166136261;
  for (let i = 0; i < data.length; i++) {
    checksum ^= data[i];
    checksum = Math.imul(checksum, 16777619) >>> 0;
  }
  return checksum >>> 0;
}

module.exports = (params, ctx) => {
  params = params || {};
  const sizeMb = intParam(params, "size_mb", 4, 1, 16);
  const rounds = intParam(params, "rounds", 8, 1, 30);
  const level = intParam(params, "level", 6, 1, 9);
  const payload = makePayload(sizeMb * 1024 * 1024);
  const expectedChecksum = fnv1a(payload);
  let compressedBytes = 0;
  let checksum = 0;
  for (let r = 0; r < rounds; r++) {
    const compressed = zlib.deflateSync(payload, { level });
    const restored = zlib.inflateSync(compressed);
    if (restored.length !== payload.length) {
      throw new Error("decompressed payload size mismatch");
    }
    const current = fnv1a(restored);
    if (current !== expectedChecksum) {
      throw new Error("decompressed payload checksum mismatch");
    }
    compressedBytes += compressed.length;
    checksum = (checksum + current + r) >>> 0;
  }
  return {
    benchmark: "compression",
    size_mb: sizeMb,
    rounds,
    level,
    input_bytes: payload.length * rounds,
    compressed_bytes: compressedBytes,
    payload_checksum: expectedChecksum,
    checksum: checksum >>> 0,
  };
};
