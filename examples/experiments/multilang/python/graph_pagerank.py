from array import array


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _build_graph(nodes, degree):
    edges = array("I", [0]) * (nodes * degree)

    for u in range(nodes):
        base = u * degree
        state = (u * 2654435761 + 1013904223) & 0xFFFFFFFF

        for j in range(degree):
            state = (1664525 * state + 1013904223 + j) & 0xFFFFFFFF
            v = state % nodes

            if v == u:
                v = (v + j + 1) % nodes

            edges[base + j] = v

    return edges


def _pagerank(nodes, degree, edges, iterations, damping):
    ranks = [1.0 / nodes] * nodes
    base_rank = (1.0 - damping) / nodes

    for _ in range(iterations):
        new_ranks = [base_rank] * nodes

        for u in range(nodes):
            contribution = damping * ranks[u] / degree
            offset = u * degree

            for j in range(degree):
                new_ranks[edges[offset + j]] += contribution

        ranks = new_ranks

    return ranks


def handler(params, context):
    params = params or {}

    nodes = _int_param(params, "nodes", 100_000, 5_000, 200_000)
    degree = _int_param(params, "degree", 8, 2, 16)
    iterations = _int_param(params, "iterations", 50, 5, 80)
    damping = 0.85

    edges = _build_graph(nodes, degree)
    ranks = _pagerank(nodes, degree, edges, iterations, damping)

    rank_sum = sum(ranks)
    sample_sum = 0.0
    step = max(1, nodes // 1000)

    for i in range(0, nodes, step):
        sample_sum += ranks[i]

    return {
        "benchmark": "graph_pagerank",
        "nodes": nodes,
        "degree": degree,
        "iterations": iterations,
        "rank0": round(ranks[0], 12),
        "rank_sum": round(rank_sum, 12),
        "sample_sum": round(sample_sum, 12),
    }
