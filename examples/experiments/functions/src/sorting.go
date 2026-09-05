package main

// sorting — SINTETICA. Ordinamento di un grande array di interi.
//
// Profilo atteso: CPU utente alta e banda di memoria elevata. A differenza di
// matmul, che riusa gli stessi blocchi, l'ordinamento attraversa ripetutamente
// l'intero array con pattern che cambiano a ogni passata: e' un carico
// intermedio fra il calcolo puro e la scansione di memoria.

import (
	"runtime"
	"sort"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	// Ridotto da 20 a 5 milioni dopo la caratterizzazione: con 20 milioni la
	// funzione superava i 35 secondi gia' su una macchina a 32 core, e sulle VM
	// sperimentali non avrebbe raggiunto i dieci campioni eleggibili nella
	// finestra di venti minuti.
	elements := 5 * 1000 * 1000
	if val, ok := params["elements"].(float64); ok {
		elements = int(val)
	}

	rounds := 2
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	data := make([]int, elements)

	checksum := 0

	for round := 0; round < rounds; round++ {

		// L'array viene rigenerato a ogni giro con un generatore xorshift:
		// ordinare due volte lo stesso array gia' ordinato misurerebbe solo
		// il caso migliore dell'algoritmo.
		state := uint64(88172645463325252 + round)

		for i := range data {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			data[i] = int(state % 1000000007)
		}

		sort.Ints(data)

		checksum += data[0] + data[len(data)-1]
	}

	return map[string]interface{}{
		"message":  "Sorting completed",
		"elements": elements,
		"rounds":   rounds,
		"checksum": checksum,
		"arch":     runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
