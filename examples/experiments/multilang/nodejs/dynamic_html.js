function intParam(params, name, defaultValue, min, max) {
  const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
  const value = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, value));
}

function fnv1aUpdate(checksum, text) {
  const data = Buffer.from(text, "utf8");
  let hash = checksum >>> 0;
  for (let i = 0; i < data.length; i++) {
    hash ^= data[i];
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

module.exports = (params, ctx) => {
  params = params || {};
  const entries = intParam(params, "entries", 12000, 500, 30000);
  const rounds = intParam(params, "rounds", 24, 1, 60);
  const username = String(params.username ?? "serverledge-user");
  let totalBytes = 0;
  let checksum = 2166136261;
  for (let r = 0; r < rounds; r++) {
    const parts = [
      "<!doctype html><html><head><title>Serverless Benchmark</title></head><body>",
      `<h1>Hello ${username}</h1>`,
      `<section data-round="${r}"><ul>`,
    ];
    let state = (0x12345678 ^ r) >>> 0;
    for (let i = 0; i < entries; i++) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      const value = state % 1000000;
      parts.push(
        `<li data-index="${i}" data-value="${value}">` +
          `item-${String(i).padStart(6, "0")}:` +
          `${String(value).padStart(6, "0")}</li>`,
      );
    }
    parts.push("</ul></section></body></html>");
    const html = parts.join("");
    totalBytes += Buffer.byteLength(html, "utf8");
    checksum = fnv1aUpdate(checksum, html);
  }
  return {
    benchmark: "dynamic_html",
    entries,
    rounds,
    total_bytes: totalBytes,
    checksum: checksum >>> 0,
  };
};
