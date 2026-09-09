package main

// twin-readmemory — SINTETICA, gruppo "gemelle".
//
// Replica il PROFILO DI RISORSE di readmemory con un'implementazione diversa.
// L'originale invoca "sysbench memory"; qui si scorre un buffer con passo pari
// alla linea di cache.
//
// TARATURA. I valori predefiniti colpiscono il profilo misurato
// dell'originale sulla campagna x86 a trenta funzioni:
//
//     readmemory         page fault 778    CPU utente 14.046 ms    kernel 16 ms
//
// La prima versione usava passes=30 su un buffer allocato una volta sola, e
// produceva 764 ms di CPU utente e ZERO page fault. Entrambi i valori erano
// sbagliati rispetto al bersaglio, e per ragioni diverse.
//
// Il tempo CPU si corregge con piu' passate. I page fault richiedono invece un
// cambiamento strutturale: un buffer allocato una volta e poi riletto non ne
// genera, perche' le pagine restano assegnate. sysbench memory alloca e
// rilascia ripetutamente, ed e' quello a produrre i 778 page fault
// dell'originale. Qui il buffer viene quindi riallocato a blocchi durante
// l'esecuzione.
//
// VERIFICA ATTESA: il donor selezionato deve essere readmemory, non
// randomaccess, benche' quest'ultima operi su un volume di memoria analogo:
// la differenza sta nel pattern di accesso, sequenziale contro casuale, ed e'
// precisamente cio' che lo spazio delle feature dovrebbe distinguere.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 256
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	// 550 passate portano la CPU utente da 764 ms a circa 14 secondi, il
	// valore misurato per readmemory.
	passes := 550
	if val, ok := params["passes"].(float64); ok {
		passes = int(val)
	}

	// Ogni quante passate rilasciare e riallocare il buffer. La riallocazione
	// costringe il sistema operativo a riassegnare le pagine, generando i page
	// fault che l'originale produce e che una singola allocazione non
	// genererebbe.
	reallocEvery := 25
	if val, ok := params["realloc_every"].(float64); ok {
		reallocEvery = int(val)
	}

	size := sizeMB * 1024 * 1024

	// Passo di 64 byte: una linea di cache. Leggere ogni byte sarebbe
	// ridondante, perche' il primo accesso a una linea porta in cache anche i
	// successivi 63.
	const stride = 64

	checksum := uint64(0)
	riallocazioni := 0

	var buffer []byte

	for p := 0; p < passes; p++ {
		if p%reallocEvery == 0 {
			buffer = make([]byte, size)

			for i := 0; i < size; i += stride {
				buffer[i] = byte(i)
			}

			riallocazioni++
		}

		for i := 0; i < size; i += stride {
			checksum += uint64(buffer[i])
		}
	}

	return map[string]interface{}{
		"message":       "Sequential memory scan completed",
		"size_mb":       sizeMB,
		"passes":        passes,
		"reallocations": riallocazioni,
		"processed_gb":  (float64(size) * float64(passes)) / 1024 / 1024 / 1024,
		"checksum":      checksum,
		"arch":          runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
