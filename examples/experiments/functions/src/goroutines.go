package main

// goroutines — SINTETICA. Creazione massiva di goroutine con scambio di
// messaggi su canale.
//
// Profilo atteso: pressione sullo scheduler del runtime Go, con CPU utente e
// kernel entrambe non trascurabili e allocazioni frequenti per gli stack.
// A differenza di mutexcontention, dove poche goroutine si contendono una
// risorsa, qui il costo e' nella creazione e nella distruzione continua.

import (
	"runtime"
	"sync"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	// Ridotto da 400 a 150 in fase di taratura.
	batches := 150
	if val, ok := params["batches"].(float64); ok {
		batches = int(val)
	}

	perBatch := 2000
	if val, ok := params["per_batch"].(float64); ok {
		perBatch = int(val)
	}

	total := 0

	for b := 0; b < batches; b++ {
		results := make(chan int, perBatch)
		var wg sync.WaitGroup

		for g := 0; g < perBatch; g++ {
			wg.Add(1)

			go func(seed int) {
				defer wg.Done()

				// Lavoro minimo: quel che si vuole misurare e' il costo di
				// creare, schedulare e distruggere la goroutine, non il
				// calcolo che esegue.
				acc := 0
				for i := 0; i < 100; i++ {
					acc += (seed + i) % 7
				}

				results <- acc
			}(g)
		}

		wg.Wait()
		close(results)

		for value := range results {
			total += value
		}
	}

	return map[string]interface{}{
		"message":    "Goroutine storm completed",
		"batches":    batches,
		"goroutines": batches * perBatch,
		"total":      total,
		"arch":       runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
