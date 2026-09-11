function intParam(params, name, defaultValue, min, max) {
  const parsed = Number.parseInt(params?.[name] ?? defaultValue, 10);
  const value = Number.isFinite(parsed) ? parsed : defaultValue;
  return Math.max(min, Math.min(max, value));
}

function buildGraph(nodes, degree) {
  const graph = new Array(nodes);
  for (let u = 0; u < nodes; u++) {
    const neighbors = new Uint32Array(degree);
    const seed = (Math.imul(u, 1103515245) + 12345) & 0x7fffffff;
    for (let j = 0; j < degree; j++) {
      let v = (seed + Math.imul(j + 1, 0x9e3779b1)) >>> 0;
      v %= nodes;
      if (v === u) {
        v = (v + 1) % nodes;
      }
      neighbors[j] = v;
    }
    graph[u] = neighbors;
  }
  return graph;
}

function bfs(graph, start) {
  const nodes = graph.length;
  const seen = new Uint8Array(nodes);
  const queue = new Uint32Array(nodes);
  let head = 0;
  let tail = 0;
  let visited = 0;
  let checksum = 0;
  seen[start] = 1;
  queue[tail++] = start;
  while (head < tail) {
    const u = queue[head++];
    visited++;
    checksum = (checksum + u) >>> 0;
    const neighbors = graph[u];
    for (let j = 0; j < neighbors.length; j++) {
      const v = neighbors[j];
      if (seen[v] === 0) {
        seen[v] = 1;
        queue[tail++] = v;
      }
    }
  }
  return [visited, checksum];
}

module.exports = (params, ctx) => {
  params = params || {};
  const nodes = intParam(params, "nodes", 180000, 5000, 250000);
  const degree = intParam(params, "degree", 8, 2, 16);
  const rounds = intParam(params, "rounds", 12, 1, 20);
  const graph = buildGraph(nodes, degree);
  let totalVisited = 0;
  let checksum = 0;
  for (let r = 0; r < rounds; r++) {
    const start = (r * 7919) % nodes;
    const [visited, current] = bfs(graph, start);
    totalVisited += visited;
    checksum = (checksum + current + Math.imul(visited, r + 1)) >>> 0;
  }
  return {
    benchmark: "graph_bfs",
    nodes,
    degree,
    rounds,
    visited: totalVisited,
    checksum,
  };
};
