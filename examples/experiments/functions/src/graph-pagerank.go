package main

// graph-pagerank — PageRank iterativo, come nel benchmark 503.graph-pagerank
// di SeBS.
//
// Profilo di risorse atteso: CPU-bound in virgola mobile, con letture
// ripetute sulla stessa struttura per un numero fisso di iterazioni. A
// differenza di BFS e MST, che attraversano il grafo una volta sola, qui lo
// stesso insieme di dati viene riletto decine di volte: il comportamento
// rispetto alla cache è quindi diverso, ed è la ragione per cui le tre
// funzioni occupano regioni distinte dello spazio delle feature.

import (
	"math"
	"math/rand"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// buildDirectedGraph costruisce un grafo orientato con archi uscenti casuali.
//
// Restituisce la lista degli archi entranti per ciascun nodo, che è la forma
// in cui PageRank li consuma, e il grado uscente, necessario a ripartire il
// rango fra i successori.
func buildDirectedGraph(n int, outDegree int, seed int64) ([][]int32, []int32) {
	rng := rand.New(rand.NewSource(seed))

	incoming := make([][]int32, n)
	for i := range incoming {
		incoming[i] = make([]int32, 0, outDegree)
	}

	outDegrees := make([]int32, n)

	for v := 0; v < n; v++ {
		for k := 0; k < outDegree; k++ {
			target := rng.Intn(n)
			if target == v {
				continue
			}

			incoming[target] = append(incoming[target], int32(v))
			outDegrees[v]++
		}
	}

	return incoming, outDegrees
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	nodes := 100000
	if val, ok := params["nodes"].(float64); ok {
		nodes = int(val)
	}

	outDegree := 10
	if val, ok := params["edges"].(float64); ok {
		outDegree = int(val)
	}

	iterations := 30
	if val, ok := params["iterations"].(float64); ok {
		iterations = int(val)
	}

	damping := 0.85
	seed := int64(42)
	if val, ok := params["seed"].(float64); ok {
		seed = int64(val)
	}

	incoming, outDegrees := buildDirectedGraph(nodes, outDegree, seed)

	rank := make([]float64, nodes)
	next := make([]float64, nodes)

	initial := 1.0 / float64(nodes)
	for i := range rank {
		rank[i] = initial
	}

	delta := 0.0

	for iteration := 0; iteration < iterations; iteration++ {

		// Il rango dei nodi senza archi uscenti viene ridistribuito
		// uniformemente: senza questa correzione la somma dei ranghi non si
		// conserverebbe e il risultato dipenderebbe dal numero di iterazioni.
		danglingMass := 0.0
		for v := 0; v < nodes; v++ {
			if outDegrees[v] == 0 {
				danglingMass += rank[v]
			}
		}

		base := (1.0-damping)/float64(nodes) +
			damping*danglingMass/float64(nodes)

		for v := 0; v < nodes; v++ {
			sum := 0.0

			for _, u := range incoming[v] {
				sum += rank[u] / float64(outDegrees[u])
			}

			next[v] = base + damping*sum
		}

		delta = 0.0
		for v := 0; v < nodes; v++ {
			delta += math.Abs(next[v] - rank[v])
		}

		rank, next = next, rank
	}

	best := 0
	for v := 1; v < nodes; v++ {
		if rank[v] > rank[best] {
			best = v
		}
	}

	return map[string]interface{}{
		"message":     "PageRank completed",
		"nodes":       nodes,
		"iterations":  iterations,
		"final_delta": delta,
		"top_node":    best,
		"arch":        runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
