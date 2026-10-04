package main

// vswarm-aes — adattamento del benchmark AES di vSwarm.
//
// Il kernel resta AES-128 in modalità CTR come nell'implementazione originale.
// Il benchmark originale cifra un singolo plaintext; quel plaintext predefinito
// è troppo piccolo per produrre un profilo stabile in Serverledge.
//
// Per la campagna di profiling viene quindi cifrato un plaintext logico più
// grande, processato a chunk per evitare un enorme buffer e un enorme output.
// Lo stream CTR rimane continuo tra i chunk: il lavoro equivale quindi alla
// cifratura sequenziale di un unico plaintext di dimensione
// chunk_mb * chunks.
//
// Default calibrato:
//   16 MiB * 64 chunk = 1024 MiB cifrati per invocazione.
//
// L'output resta compatto per evitare che serializzazione e rete dominino
// il profilo della funzione.

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/hex"
	"fmt"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

const (
	defaultAESKeyHex = "6368616e676520746869732070617373"
	defaultPlaintext = "defaultplaintext"

	defaultChunkMB = 16
	defaultChunks  = 64
)

func aesCTRWorkload(
	chunkMB int,
	chunks int,
	keyHex string,
	pattern []byte,
) (uint64, error) {

	if chunkMB <= 0 {
		return 0, fmt.Errorf("chunk_mb must be > 0")
	}

	if chunks <= 0 {
		return 0, fmt.Errorf("chunks must be > 0")
	}

	if len(pattern) == 0 {
		return 0, fmt.Errorf("plaintext must not be empty")
	}

	key, err := hex.DecodeString(keyHex)
	if err != nil {
		return 0, fmt.Errorf("invalid hexadecimal AES key: %w", err)
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return 0, fmt.Errorf("invalid AES key: %w", err)
	}

	size := chunkMB * 1024 * 1024

	plaintext := make([]byte, size)
	ciphertext := make([]byte, size)

	for i := range plaintext {
		plaintext[i] = pattern[i%len(pattern)]
	}

	// vSwarm usa un IV nullo per rendere il risultato riproducibile.
	iv := make([]byte, aes.BlockSize)

	// Un solo stream continuo: i chunk rappresentano parti successive
	// dello stesso plaintext logico.
	stream := cipher.NewCTR(block, iv)

	var checksum uint64

	for i := 0; i < chunks; i++ {
		stream.XORKeyStream(ciphertext, plaintext)

		// Consuma una piccola parte del risultato senza serializzare
		// l'intero ciphertext.
		checksum += uint64(ciphertext[0])
		checksum += uint64(ciphertext[len(ciphertext)-1])
	}

	return checksum, nil
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	chunkMB := defaultChunkMB
	if val, ok := params["chunk_mb"].(float64); ok {
		chunkMB = int(val)
	}

	chunks := defaultChunks
	if val, ok := params["chunks"].(float64); ok {
		chunks = int(val)
	}

	keyHex := defaultAESKeyHex
	if val, ok := params["key"].(string); ok && val != "" {
		keyHex = val
	}

	plaintext := defaultPlaintext
	if val, ok := params["plaintext"].(string); ok && val != "" && val != "world" {
		plaintext = val
	}

	checksum, err := aesCTRWorkload(
		chunkMB,
		chunks,
		keyHex,
		[]byte(plaintext),
	)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"message":      "vSwarm AES-CTR completed",
		"chunk_mb":     chunkMB,
		"chunks":       chunks,
		"processed_mb": chunkMB * chunks,
		"checksum":     checksum,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
