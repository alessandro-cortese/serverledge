package main

// twin-readmemory — SINTETICA, gruppo "gemelle".
//
// Scopo: replicare il profilo di readmemory con un'implementazione diversa.
// L'originale invoca "sysbench memory"; qui si scorre un buffer in modo
// sequenziale con passo pari alla linea di cache.
//
// L'accesso e' sequenziale e prevedibile, quindi il prefetch hardware lavora
// bene e i page fault restano bassi: e' l'opposto di randomaccess, che opera
// sullo stesso volume di memoria ma in ordine casuale. Le tre funzioni
// insieme — readmemory, twin-readmemory, randomaccess — permettono di
// verificare se lo spazio delle feature separi il pattern di accesso dal
// volume di memoria toccato.
//
// VERIFICA ATTESA: il donor selezionato per questa funzione deve essere
// readmemory, non randomaccess.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 256
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	passes := 30
	if val, ok := params["passes"].(float64); ok {
		passes = int(val)
	}

	size := sizeMB * 1024 * 1024
	buffer := make([]byte, size)

	for i := range buffer {
		buffer[i] = byte(i)
	}

	// Passo di 64 byte: una linea di cache. Leggere ogni byte sarebbe
	// ridondante, perche' il primo accesso a una linea porta in cache anche
	// i successivi 63.
	const stride = 64

	checksum := uint64(0)

	for p := 0; p < passes; p++ {
		for i := 0; i < size; i += stride {
			checksum += uint64(buffer[i])
		}
	}

	return map[string]interface{}{
		"message":      "Sequential memory scan completed",
		"size_mb":      sizeMB,
		"passes":       passes,
		"processed_gb": (float64(size) * float64(passes)) / 1024 / 1024 / 1024,
		"checksum":     checksum,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
