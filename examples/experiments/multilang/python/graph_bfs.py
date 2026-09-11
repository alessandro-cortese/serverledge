from collections import deque


def _int_param(params, name, default, minimum, maximum):
    try:
        value = int(params.get(name, default))
    except (TypeError, ValueError, AttributeError):
        value = default
    return max(minimum, min(maximum, value))


def _build_graph(nodes, degree):
    graph = [[] for _ in range(nodes)]

    for u in range(nodes):
        seed = (u * 1103515245 + 12345) & 0x7FFFFFFF

        for j in range(degree):
            v = (seed + ((j + 1) * 2654435761) & 0xFFFFFFFF) & 0xFFFFFFFF
            v %= nodes

            if v == u:
                v = (v + 1) % nodes

            graph[u].append(v)

    return graph


def _bfs(graph, start):
    seen = bytearray(len(graph))
    queue = deque([start])
    seen[start] = 1

    visited = 0
    checksum = 0

    while queue:
        u = queue.popleft()
        visited += 1
        checksum = (checksum + u) & 0xFFFFFFFF

        for v in graph[u]:
            if not seen[v]:
                seen[v] = 1
                queue.append(v)

    return visited, checksum


def handler(params, context):
    params = params or {}

    nodes = _int_param(params, "nodes", 180_000, 5_000, 250_000)
    degree = _int_param(params, "degree", 8, 2, 16)
    rounds = _int_param(params, "rounds", 12, 1, 20)

    graph = _build_graph(nodes, degree)

    total_visited = 0
    checksum = 0

    for r in range(rounds):
        start = (r * 7919) % nodes
        visited, current = _bfs(graph, start)

        total_visited += visited
        checksum = (checksum + current + visited * (r + 1)) & 0xFFFFFFFF

    return {
        "benchmark": "graph_bfs",
        "nodes": nodes,
        "degree": degree,
        "rounds": rounds,
        "visited": total_visited,
        "checksum": checksum,
    }
