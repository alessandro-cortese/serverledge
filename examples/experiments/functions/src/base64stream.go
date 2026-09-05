package main

// base64stream — SINTETICA. Codifica e decodifica Base64 ripetuta.
//
// Profilo atteso: trasformazione lineare pura, con lettura e scrittura
// strettamente sequenziali e nessuna dipendenza fra iterazioni successive.
// E' il carico piu' regolare dell'insieme, ideale come estremo opposto di
// pointerchase nello spazio delle feature.

import (
	"encoding/base64"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 16
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	rounds := 30
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	size := sizeMB * 1024 * 1024
	source := make([]byte, size)

	for i := range source {
		source[i] = byte(i)
	}

	// I buffer sono allocati una volta sola: riallocarli a ogni giro
	// sposterebbe il profilo verso la pressione sull'allocatore, che e' gia'
	// coperta da jsonparse.
	encoded := make([]byte, base64.StdEncoding.EncodedLen(size))
	decoded := make([]byte, size)

	totalDecoded := 0

	for r := 0; r < rounds; r++ {
		base64.StdEncoding.Encode(encoded, source)

		n, err := base64.StdEncoding.Decode(decoded, encoded)
		if err != nil {
			return nil, err
		}

		totalDecoded += n
	}

	return map[string]interface{}{
		"message":      "Base64 round-trip completed",
		"size_mb":      sizeMB,
		"rounds":       rounds,
		"processed_gb": float64(totalDecoded) / 1024 / 1024 / 1024,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
