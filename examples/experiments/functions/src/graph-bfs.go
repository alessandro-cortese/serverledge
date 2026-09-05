package main

// graph-bfs — visita in ampiezza su un grafo generato con il modello
// Barabási–Albert, come nel benchmark 501.graph-bfs di SeBS.
//
// Profilo di risorse atteso: CPU-bound con accesso alla memoria fortemente
// irregolare. La lista di adiacenza viene attraversata seguendo puntatori,
// quindi la funzione genera molti cache miss e page fault rispetto a un
// carico che scorre la memoria in modo sequenziale.

import (
	"math/rand"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// buildBarabasiAlbert costruisce un grafo a invarianza di scala.
//
// Ogni nuovo nodo si collega a m nodi esistenti scelti con probabilità
// proporzionale al loro grado. Il risultato ha pochi nodi molto connessi e
// molti nodi poco connessi, distribuzione che rende la visita irregolare.
func buildBarabasiAlbert(n int, m int, seed int64) [][]int32 {
	rng := rand.New(rand.NewSource(seed))

	adjacency := make([][]int32, n)
	for i := range adjacency {
		adjacency[i] = make([]int32, 0, m)
	}

	// targets contiene ogni nodo ripetuto una volta per arco incidente:
	// estrarre a caso da questa lista equivale a estrarre proporzionalmente
	// al grado, senza dover ricalcolare la distribuzione a ogni passo.
	targets := make([]int32, 0, 2*n*m)

	for i := 0; i < m && i < n; i++ {
		for j := 0; j < m && j < n; j++ {
			if i == j {
				continue
			}
			adjacency[i] = append(adjacency[i], int32(j))
		}
		targets = append(targets, int32(i))
	}

	for v := m; v < n; v++ {
		chosen := make(map[int32]struct{}, m)

		for len(chosen) < m && len(targets) > 0 {
			candidate := targets[rng.Intn(len(targets))]
			chosen[candidate] = struct{}{}
		}

		for u := range chosen {
			adjacency[v] = append(adjacency[v], u)
			adjacency[u] = append(adjacency[u], int32(v))
			targets = append(targets, u)
		}

		targets = append(targets, int32(v))
	}

	return adjacency
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	nodes := 100000
	if val, ok := params["nodes"].(float64); ok {
		nodes = int(val)
	}

	edgesPerNode := 10
	if val, ok := params["edges"].(float64); ok {
		edgesPerNode = int(val)
	}

	// Il seme è fisso per default: due invocazioni della stessa funzione
	// devono attraversare lo stesso grafo, altrimenti i campioni di
	// profilazione non sarebbero confrontabili fra loro.
	seed := int64(42)
	if val, ok := params["seed"].(float64); ok {
		seed = int64(val)
	}

	adjacency := buildBarabasiAlbert(nodes, edgesPerNode, seed)

	distance := make([]int32, nodes)
	for i := range distance {
		distance[i] = -1
	}

	queue := make([]int32, 0, nodes)
	queue = append(queue, 0)
	distance[0] = 0

	visited := 0
	maxDistance := int32(0)

	for head := 0; head < len(queue); head++ {
		current := queue[head]
		visited++

		for _, neighbour := range adjacency[current] {
			if distance[neighbour] >= 0 {
				continue
			}

			distance[neighbour] = distance[current] + 1
			if distance[neighbour] > maxDistance {
				maxDistance = distance[neighbour]
			}

			queue = append(queue, neighbour)
		}
	}

	return map[string]interface{}{
		"message":      "BFS completed",
		"nodes":        nodes,
		"visited":      visited,
		"eccentricity": maxDistance,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
