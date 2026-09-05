package main

// graph-mst — albero ricoprente minimo con l'algoritmo di Prim, come nel
// benchmark 502.graph-mst di SeBS.
//
// Profilo di risorse atteso: CPU-bound dominato dalle operazioni sulla coda
// di priorità. Rispetto alla BFS l'accesso alla memoria è meno irregolare —
// l'heap è un array contiguo — ma il numero di confronti e scambi è molto più
// alto. Serve quindi a occupare una regione dello spazio delle feature
// distinta da quella della visita in ampiezza.

import (
	"container/heap"
	"math/rand"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

type edge struct {
	to     int32
	weight float64
}

type queueItem struct {
	node   int32
	weight float64
}

// priorityQueue implementa heap.Interface sui nodi di frontiera.
type priorityQueue []queueItem

func (q priorityQueue) Len() int { return len(q) }

func (q priorityQueue) Less(i int, j int) bool {
	return q[i].weight < q[j].weight
}

func (q priorityQueue) Swap(i int, j int) {
	q[i], q[j] = q[j], q[i]
}

func (q *priorityQueue) Push(item interface{}) {
	*q = append(*q, item.(queueItem))
}

func (q *priorityQueue) Pop() interface{} {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[:n-1]

	return item
}

// buildWeightedGraph costruisce un grafo connesso e pesato.
//
// La catena iniziale garantisce la connessione, così l'albero ricoprente
// esiste sempre e il costo dell'algoritmo non dipende dal seme; gli archi
// aggiuntivi casuali ne aumentano la densità.
func buildWeightedGraph(n int, extraEdges int, seed int64) [][]edge {
	rng := rand.New(rand.NewSource(seed))

	adjacency := make([][]edge, n)
	for i := range adjacency {
		adjacency[i] = make([]edge, 0, 4)
	}

	addEdge := func(a int32, b int32, w float64) {
		adjacency[a] = append(adjacency[a], edge{to: b, weight: w})
		adjacency[b] = append(adjacency[b], edge{to: a, weight: w})
	}

	for v := 1; v < n; v++ {
		addEdge(int32(v), int32(rng.Intn(v)), rng.Float64())
	}

	for i := 0; i < extraEdges; i++ {
		a := int32(rng.Intn(n))
		b := int32(rng.Intn(n))

		if a != b {
			addEdge(a, b, rng.Float64())
		}
	}

	return adjacency
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	nodes := 100000
	if val, ok := params["nodes"].(float64); ok {
		nodes = int(val)
	}

	extraEdges := nodes * 5
	if val, ok := params["edges"].(float64); ok {
		extraEdges = int(val)
	}

	seed := int64(42)
	if val, ok := params["seed"].(float64); ok {
		seed = int64(val)
	}

	adjacency := buildWeightedGraph(nodes, extraEdges, seed)

	inTree := make([]bool, nodes)
	queue := &priorityQueue{{node: 0, weight: 0}}
	heap.Init(queue)

	totalWeight := 0.0
	treeEdges := 0

	for queue.Len() > 0 {
		item := heap.Pop(queue).(queueItem)

		// Un nodo può comparire più volte nella coda con pesi diversi:
		// le occorrenze successive alla prima vengono scartate qui, evitando
		// di dover implementare la decrease-key.
		if inTree[item.node] {
			continue
		}

		inTree[item.node] = true
		totalWeight += item.weight
		treeEdges++

		for _, e := range adjacency[item.node] {
			if !inTree[e.to] {
				heap.Push(queue, queueItem{node: e.to, weight: e.weight})
			}
		}
	}

	return map[string]interface{}{
		"message":      "MST completed",
		"nodes":        nodes,
		"tree_edges":   treeEdges - 1,
		"total_weight": totalWeight,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
