package main

// randomaccess — SINTETICA, gruppo "riempimento".
//
// Scopo: occupare la regione dello spazio delle feature ad alto
// page_faults_delta. Alloca un'area di memoria molto piu' grande della cache e
// vi accede in ordine pseudo-casuale, cosi' che ogni lettura ricada quasi
// sempre su una pagina diversa dalla precedente.
//
// Profilo atteso: page fault molto alti, CPU utente moderata, CPU kernel bassa.
// E' il complemento di readmemory, che scorre la memoria in modo sequenziale e
// beneficia del prefetch hardware.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 256
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	accesses := 40 * 1000 * 1000
	if val, ok := params["accesses"].(float64); ok {
		accesses = int(val)
	}

	size := sizeMB * 1024 * 1024
	buffer := make([]byte, size)

	// Riempimento iniziale: forza il sistema operativo ad assegnare
	// fisicamente tutte le pagine, altrimenti i page fault verrebbero
	// contati durante la fase di misura invece che qui.
	for i := range buffer {
		buffer[i] = byte(i)
	}

	// Generatore congruenziale lineare inline: evita il costo di math/rand
	// e mantiene il ciclo dominato dall'accesso alla memoria anziche' dalla
	// generazione dei numeri.
	state := uint64(88172645463325252)
	checksum := uint64(0)

	for i := 0; i < accesses; i++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17

		checksum += uint64(buffer[state%uint64(size)])
	}

	return map[string]interface{}{
		"message":  "Random access completed",
		"size_mb":  sizeMB,
		"accesses": accesses,
		"checksum": checksum,
		"arch":     runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
