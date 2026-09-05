package main

// pointerchase — SINTETICA. Inseguimento di puntatori lungo una lista
// concatenata dispersa in memoria.
//
// Profilo atteso: il caso estremo di accesso irregolare. Ogni lettura dipende
// dalla precedente, quindi la CPU non puo' anticipare il prossimo indirizzo e
// il prefetch hardware e' inutile. Rispetto a randomaccess, dove gli accessi
// sono casuali ma indipendenti e quindi parallelizzabili, qui la catena e'
// strettamente sequenziale: e' il carico piu' sensibile alla latenza della
// memoria, ed e' una dimensione su cui le due architetture differiscono
// sensibilmente.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	nodes := 8 * 1000 * 1000
	if val, ok := params["nodes"].(float64); ok {
		nodes = int(val)
	}

	// Ridotto da 60 a 15 milioni in fase di taratura: ogni passo dipende dal
	// precedente, quindi il tempo cresce linearmente con la latenza di
	// memoria e sulle VM il carico originario risultava eccessivo.
	steps := 15 * 1000 * 1000
	if val, ok := params["steps"].(float64); ok {
		steps = int(val)
	}

	// La catena e' rappresentata da una permutazione: next[i] indica il nodo
	// successivo. Costruirla come permutazione garantisce che l'inseguimento
	// non entri mai in un ciclo corto e attraversi tutti i nodi.
	next := make([]int32, nodes)
	for i := range next {
		next[i] = int32(i)
	}

	state := uint64(88172645463325252)
	for i := nodes - 1; i > 0; i-- {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17

		j := int(state % uint64(i+1))
		next[i], next[j] = next[j], next[i]
	}

	current := int32(0)
	for s := 0; s < steps; s++ {
		current = next[current]
	}

	return map[string]interface{}{
		"message": "Pointer chase completed",
		"nodes":   nodes,
		"steps":   steps,
		"final":   current,
		"arch":    runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
