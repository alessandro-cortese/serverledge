package main

// twin-primenumber — SINTETICA, gruppo "gemelle".
//
// Replica il PROFILO DI RISORSE di primenumber con un algoritmo diverso.
// L'originale invoca "sysbench cpu", che verifica la primalita' per divisioni
// successive; qui si usa un crivello di Eratostene segmentato.
//
// TARATURA. I valori predefiniti non sono scelti per comodita' ma per colpire
// il profilo misurato dell'originale sulla campagna x86 a trenta funzioni:
//
//     primenumber        page fault 520    CPU utente 52.293 ms    kernel 10 ms
//
// La prima versione di questa funzione usava limit=30.000.000 senza
// ripetizioni, e produceva 111 ms di CPU utente: quattrocentosettanta volte
// meno dell'originale. Le due funzioni risultavano quindi distanti nello
// spazio delle feature, e il confronto che dovevano rendere possibile non era
// eseguibile.
//
// La ripetizione del crivello e' preferibile all'aumento del limite: un
// crivello su un miliardo e mezzo di interi richiederebbe gigabyte di memoria
// e sposterebbe il profilo verso l'uso di memoria, mentre l'originale e'
// CPU-bound con footprint contenuto.
//
// VERIFICA ATTESA: interrogando il catalogo con questa funzione il donor
// selezionato deve essere primenumber. Se non lo e', il problema sta nel
// vettore delle feature o nella metrica di distanza, non nei dati.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// crivelloSegmentato conta i primi fino a limit usando segmenti che stanno
// nella cache, come fa sysbench restando CPU-bound senza saturare la memoria.
func crivelloSegmentato(limit int, segment int) int {
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

	return count
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	limit := 30 * 1000 * 1000
	if val, ok := params["limit"].(float64); ok {
		limit = int(val)
	}

	// 470 ripetizioni portano la CPU utente da 111 ms a circa 52 secondi, il
	// valore misurato per primenumber.
	rounds := 470
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	segment := 1 << 20

	count := 0

	for r := 0; r < rounds; r++ {
		count = crivelloSegmentato(limit, segment)
	}

	return map[string]interface{}{
		"message":     "Prime sieve completed",
		"limit":       limit,
		"rounds":      rounds,
		"prime_count": count,
		"arch":        runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
