function intParam(params, name, defaultValue, min, max) {
  const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
  const value = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, value));
}

function buildGraph(nodes, degree) {
  const edges = new Uint32Array(nodes * degree);
  for (let u = 0; u < nodes; u++) {
    const base = u * degree;
    let state = (Math.imul(u, 0x9e3779b1) + 1013904223) >>> 0;
    for (let j = 0; j < degree; j++) {
      state = (Math.imul(1664525, state) + 1013904223 + j) >>> 0;
      let v = state % nodes;
      if (v === u) {
        v = (v + j + 1) % nodes;
      }
      edges[base + j] = v;
    }
  }
  return edges;
}

function pagerank(nodes, degree, edges, iterations, damping) {
  let ranks = new Float64Array(nodes);
  ranks.fill(1.0 / nodes);
  const baseRank = (1.0 - damping) / nodes;
  for (let iter = 0; iter < iterations; iter++) {
    const newRanks = new Float64Array(nodes);
    newRanks.fill(baseRank);
    for (let u = 0; u < nodes; u++) {
      const contribution = (damping * ranks[u]) / degree;
      const offset = u * degree;
      for (let j = 0; j < degree; j++) {
        newRanks[edges[offset + j]] += contribution;
      }
    }
    ranks = newRanks;
  }
  return ranks;
}

module.exports = (params, ctx) => {
  params = params || {};
  const nodes = intParam(params, "nodes", 100000, 5000, 200000);
  const degree = intParam(params, "degree", 8, 2, 16);
  const iterations = intParam(params, "iterations", 50, 5, 80);
  const damping = 0.85;
  const edges = buildGraph(nodes, degree);
  const ranks = pagerank(nodes, degree, edges, iterations, damping);
  let rankSum = 0.0;
  for (let i = 0; i < nodes; i++) {
    rankSum += ranks[i];
  }
  let sampleSum = 0.0;
  const step = Math.max(1, Math.floor(nodes / 1000));
  for (let i = 0; i < nodes; i += step) {
    sampleSum += ranks[i];
  }
  return {
    benchmark: "graph_pagerank",
    nodes,
    degree,
    iterations,
    rank0: Number(ranks[0].toFixed(12)),
    rank_sum: Number(rankSum.toFixed(12)),
    sample_sum: Number(sampleSum.toFixed(12)),
  };
};
