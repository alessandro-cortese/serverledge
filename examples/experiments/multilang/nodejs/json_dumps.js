function intParam(params, name, defaultValue, min, max) {
  const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
  const value = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, value));
}

module.exports = (params, ctx) => {
  params = params || {};
  const records = intParam(params, "records", 12000, 1000, 50000);
  const rounds = intParam(params, "rounds", 20, 1, 100);
  const payload = new Array(records);
  for (let i = 0; i < records; i++) {
    payload[i] = {
      active: i % 3 !== 0,
      group: i % 97,
      id: i,
      name: `record-${String(i).padStart(6, "0")}`,
      score: (Math.imul(i, 0x9e3779b1) >>> 0) / 4294967296.0,
      tags: [i % 11, i % 17, i % 23],
    };
  }
  let checksum = 0;
  let encodedBytes = 0;
  for (let r = 0; r < rounds; r++) {
    const encoded = JSON.stringify(payload);
    const decoded = JSON.parse(encoded);
    encodedBytes += Buffer.byteLength(encoded, "utf8");
    checksum = (checksum + decoded[records - 1].id + decoded[Math.floor(records / 2)].group) >>> 0;
  }
  return {
    benchmark: "json_dumps",
    records,
    rounds,
    encoded_bytes: encodedBytes,
    checksum,
  };
};
