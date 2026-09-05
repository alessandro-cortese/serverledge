package main

// hashing — SINTETICA. Calcolo ripetuto di digest SHA-256.
//
// Profilo atteso: CPU utente alta con accesso strettamente sequenziale alla
// memoria. E' il complemento crittografico di chacha20, che usa openssl: qui
// il calcolo avviene interamente in spazio utente senza invocare processi
// esterni, quindi la CPU kernel resta bassa.
//
// Su ARMv8 e su x86 recenti SHA-256 dispone di istruzioni dedicate, ma la
// libreria standard di Go le usa in modo diverso sulle due architetture:
// la funzione e' quindi un candidato interessante per far emergere differenze
// architetturali.

import (
	"crypto/sha256"
	"encoding/hex"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	blockKB := 64
	if val, ok := params["block_kb"].(float64); ok {
		blockKB = int(val)
	}

	iterations := 20000
	if val, ok := params["iterations"].(float64); ok {
		iterations = int(val)
	}

	block := make([]byte, blockKB*1024)
	for i := range block {
		block[i] = byte(i)
	}

	digest := make([]byte, 32)

	for i := 0; i < iterations; i++ {
		// Il digest precedente viene reimmesso nel blocco: la catena impedisce
		// al compilatore di considerare invariante il calcolo e di eliminarlo.
		copy(block, digest)

		sum := sha256.Sum256(block)
		copy(digest, sum[:])
	}

	return map[string]interface{}{
		"message":      "Hashing completed",
		"iterations":   iterations,
		"processed_mb": float64(blockKB*iterations) / 1024,
		"final_digest": hex.EncodeToString(digest),
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
