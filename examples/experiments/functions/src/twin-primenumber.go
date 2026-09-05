package main

// twin-primenumber — SINTETICA, gruppo "gemelle".
//
// Scopo: replicare il profilo di risorse di primenumber con un algoritmo
// diverso. L'originale invoca "sysbench cpu", che verifica la primalita' per
// divisioni successive; qui si usa un crivello di Eratostene segmentato.
//
// Il carico e' in entrambi i casi CPU utente puro su un'area di memoria
// contenuta, quindi le due funzioni dovrebbero risultare vicine nello spazio
// delle feature pur essendo implementazioni indipendenti.
//
// VERIFICA ATTESA: interrogando il catalogo con questa funzione, il donor
// selezionato deve essere primenumber. E' un criterio di correttezza per il
// meccanismo di similarita': si conosce in anticipo la risposta giusta.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	limit := 30 * 1000 * 1000
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
	}

	// Crivello segmentato: il segmento sta nella cache, mentre il crivello
	// completo su 30 milioni di elementi non ci starebbe. E' la stessa
	// ragione per cui sysbench cpu resta CPU-bound senza saturare la memoria.
	segment := 1 << 20

	base := make([]bool, segment+1)
	primes := make([]int, 0, 4096)

	for i := 2; i*i <= limit; i++ {
		if base[i] {
			continue
		}

		primes = append(primes, i)

		for j := i * i; j <= segment && j <= limit; j += i {
			base[j] = true
		}
	}

	count := 0
	for i := 2; i <= segment && i <= limit; i++ {
		if !base[i] {
			count++
		}
	}

	window := make([]bool, segment)

	for low := segment + 1; low <= limit; low += segment {
		high := low + segment - 1
		if high > limit {
			high = limit
		}

		for i := range window {
			window[i] = false
		}

		for _, p := range primes {
			start := ((low + p - 1) / p) * p
			if start < p*p {
				start = p * p
			}

			for j := start; j <= high; j += p {
				window[j-low] = true
			}
		}

		for i := 0; i <= high-low; i++ {
			if !window[i] {
				count++
			}
		}
	}

	return map[string]interface{}{
		"message":     "Prime sieve completed",
		"limit":       limit,
		"prime_count": count,
		"arch":        runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
