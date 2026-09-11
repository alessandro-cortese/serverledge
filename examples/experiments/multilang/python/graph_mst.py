MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _xorshift64(state):
    state ^= (state << 13) & MASK64
    state &= MASK64
    state ^= state >> 7
    state &= MASK64
    state ^= (state << 17) & MASK64
    return state & MASK64


def _find(parent, node):
    root = node

    while parent[root] != root:
        root = parent[root]

    while parent[node] != node:
        next_node = parent[node]
        parent[node] = root
        node = next_node

    return root


def _generate_graph(nodes, m, seed):
    initial = m + 1
    edges = []
    repeated = []
    state = seed & MASK64

    # Initial complete graph.
    for u in range(initial):
        for v in range(u + 1, initial):
            state = _xorshift64(state)
            weight = ((state >> 32) & 0xFFFF) + 1
            edges.append((weight, u, v))
            repeated.append(u)
            repeated.append(v)

    # Preferential attachment, Barabasi-Albert style.
    for node in range(initial, nodes):
        targets = set()

        while len(targets) < m:
            state = _xorshift64(state)
            target = repeated[state % len(repeated)]
            targets.add(target)

        for target in targets:
            state = _xorshift64(state)
            weight = ((state >> 32) & 0xFFFF) + 1
            edges.append((weight, node, target))
            repeated.append(node)
            repeated.append(target)

    return edges


def _kruskal(nodes, edges):
    parent = list(range(nodes))
    rank = bytearray(nodes)

    selected = 0
    total_weight = 0
    checksum = 2166136261

    for weight, u, v in sorted(edges):
        ru = _find(parent, u)
        rv = _find(parent, v)

        if ru == rv:
            continue

        if rank[ru] < rank[rv]:
            ru, rv = rv, ru

        parent[rv] = ru

        if rank[ru] == rank[rv]:
            rank[ru] += 1

        selected += 1
        total_weight += weight
        checksum ^= (u * 31 + v * 17 + weight) & MASK32
        checksum = (checksum * 16777619) & MASK32

        if selected == nodes - 1:
            break

    return selected, total_weight, checksum


def handler(params, context):
    params = params or {}

    nodes = _int_param(params, "nodes", 60_000, 100, 150_000)
    m = _int_param(params, "m", 8, 1, 16)

    if m >= nodes:
        m = nodes - 1

    seed = _int_param(params, "seed", 88_172_645, 1, (1 << 31) - 1)
    edges = _generate_graph(nodes, m, seed)
    selected, total_weight, checksum = _kruskal(nodes, edges)

    return {
        "benchmark": "graph_mst",
        "nodes": nodes,
        "m": m,
        "edges": len(edges),
        "mst_edges": selected,
        "total_weight": total_weight,
        "checksum": checksum,
    }
