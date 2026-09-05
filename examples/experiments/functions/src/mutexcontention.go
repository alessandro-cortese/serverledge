package main

// mutexcontention — SINTETICA. Contesa su un mutex fra piu' goroutine.
//
// Profilo atteso: CPU kernel alta per i cambi di contesto, CPU utente bassa.
// Si distingue da thread, che con sysbench misura la creazione e la
// terminazione dei thread: qui i thread esistono gia' e il costo e' tutto
// nella contesa.
//
// E' anche la funzione piu' sensibile al numero di core disponibili, quindi
// utile per lo scenario con famiglie di macchine di potenza diversa.

import (
	"runtime"
	"sync"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	workers := runtime.NumCPU() * 4
	if val, ok := params["workers"].(float64); ok {
		workers = int(val)
	}

	// Ridotto da 400.000 a 150.000 in fase di taratura. Il numero di worker
	// resta legato a NumCPU() di proposito: questa e' l'unica funzione la cui
	// durata dipende esplicitamente dal numero di core, ed e' quindi la piu'
	// utile nello scenario con famiglie di macchine di potenza diversa.
	incrementsPerWorker := 150000
	if val, ok := params["increments"].(float64); ok {
		incrementsPerWorker = int(val)
	}

	var mu sync.Mutex
	var wg sync.WaitGroup

	counter := 0

	for w := 0; w < workers; w++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for i := 0; i < incrementsPerWorker; i++ {
				// Sezione critica minima: il costo e' quasi interamente
				// nell'acquisizione del lock, non nel lavoro protetto.
				mu.Lock()
				counter++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	return map[string]interface{}{
		"message":  "Mutex contention completed",
		"workers":  workers,
		"counter":  counter,
		"num_cpu":  runtime.NumCPU(),
		"arch":     runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
