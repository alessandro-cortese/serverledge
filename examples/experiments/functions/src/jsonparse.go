package main

// jsonparse — SINTETICA. Serializzazione e deserializzazione JSON ripetuta.
//
// Profilo atteso: CPU utente alta accompagnata da page fault elevati. La
// deserializzazione alloca molti oggetti di piccola dimensione, il garbage
// collector interviene di frequente, e la memoria viene continuamente
// richiesta e restituita al sistema operativo.
//
// E' un profilo che nessuna funzione esistente occupa: randomaccess ha page
// fault alti ma su un'area preallocata, senza pressione sull'allocatore.

import (
	"encoding/json"
	"fmt"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

type record struct {
	ID       int               `json:"id"`
	Name     string            `json:"name"`
	Tags     []string          `json:"tags"`
	Metadata map[string]string `json:"metadata"`
	Values   []float64         `json:"values"`
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	records := 20000
	if val, ok := params["records"].(float64); ok {
		records = int(val)
	}

	// Ridotto da 20 a 8 in fase di taratura.
	rounds := 8
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	source := make([]record, records)

	for i := range source {
		source[i] = record{
			ID:   i,
			Name: fmt.Sprintf("entity-%d", i),
			Tags: []string{"alpha", "beta", "gamma"},
			Metadata: map[string]string{
				"region": "eu",
				"tier":   fmt.Sprintf("t%d", i%5),
			},
			Values: []float64{float64(i), float64(i) * 1.5, float64(i) * 2.25},
		}
	}

	totalBytes := 0
	totalParsed := 0

	for round := 0; round < rounds; round++ {
		encoded, err := json.Marshal(source)
		if err != nil {
			return nil, err
		}

		totalBytes += len(encoded)

		// Il risultato viene deserializzato in una struttura nuova a ogni
		// giro: riusare la stessa slice ridurrebbe le allocazioni e
		// snaturerebbe il profilo che si vuole ottenere.
		var decoded []record
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			return nil, err
		}

		totalParsed += len(decoded)
	}

	return map[string]interface{}{
		"message":      "JSON round-trip completed",
		"records":      records,
		"rounds":       rounds,
		"total_parsed": totalParsed,
		"total_mb":     float64(totalBytes) / 1024 / 1024,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
