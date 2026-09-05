package main

// dna-visualisation — adattamento del benchmark 504.dna-visualisation di SeBS.
//
// L'originale legge una sequenza di DNA da storage e ne produce una
// visualizzazione bidimensionale con l'algoritmo Squiggle. Qui la sequenza
// viene generata in-funzione: la dipendenza da uno storage esterno renderebbe
// la funzione non profilabile in modo confrontabile con le altre, perche' il
// tempo di rete entrerebbe nella misura.
//
// Profilo atteso: CPU utente in virgola mobile su un array lungo, con accesso
// sequenziale. E' un carico misto fra il calcolo di linpack e la scansione di
// readmemory, quindi dovrebbe collocarsi in una regione intermedia.

import (
	"math"
	"math/rand"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// squiggle converte una sequenza di basi in una passeggiata bidimensionale.
//
// E' la trasformazione "squiggle" dell'omonima libreria: ogni base impone una
// direzione, e la curva risultante rende visivamente confrontabili sequenze
// diverse.
func squiggle(sequence []byte) ([]float64, []float64) {
	x := make([]float64, 0, len(sequence)*2+1)
	y := make([]float64, 0, len(sequence)*2+1)

	cx := 0.0
	cy := 0.0

	x = append(x, cx)
	y = append(y, cy)

	for _, base := range sequence {
		var dy float64

		switch base {
		case 'A':
			dy = -1
		case 'T':
			dy = 1
		case 'G':
			dy = 1
		case 'C':
			dy = -1
		}

		// Ogni base produce due segmenti, in salita e in discesa, cosi' che
		// la curva torni sempre alla quota di partenza: e' questo a rendere
		// la rappresentazione leggibile come sequenza di onde.
		cx += 0.5
		cy += dy * 0.5
		x = append(x, cx)
		y = append(y, cy)

		cx += 0.5
		cy -= dy * 0.5
		x = append(x, cx)
		y = append(y, cy)
	}

	return x, y
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	length := 2 * 1000 * 1000
	if val, ok := params["length"].(float64); ok {
		length = int(val)
	}

	seed := int64(42)
	if val, ok := params["seed"].(float64); ok {
		seed = int64(val)
	}

	rng := rand.New(rand.NewSource(seed))
	bases := []byte{'A', 'C', 'G', 'T'}

	sequence := make([]byte, length)
	for i := range sequence {
		sequence[i] = bases[rng.Intn(4)]
	}

	x, y := squiggle(sequence)

	// Statistiche sulla curva: costringono a rileggere entrambi gli array,
	// aggiungendo una seconda passata sequenziale sui dati appena prodotti.
	minY := math.Inf(1)
	maxY := math.Inf(-1)
	sumY := 0.0

	for _, v := range y {
		if v < minY {
			minY = v
		}
		if v > maxY {
			maxY = v
		}
		sumY += v
	}

	return map[string]interface{}{
		"message":     "DNA visualisation completed",
		"length":      length,
		"points":      len(x),
		"y_min":       minY,
		"y_max":       maxY,
		"y_mean":      sumY / float64(len(y)),
		"final_x":     x[len(x)-1],
		"arch":        runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
