package main

// twin-chacha20 — SINTETICA, gruppo "gemelle".
//
// Implementa ChaCha20 in Go puro, mentre la funzione chacha20 della tesi
// precedente invoca "openssl enc -chacha20" come processo esterno.
//
// E' il caso piu' interessante fra le gemelle: l'algoritmo e' lo stesso, ma
// l'implementazione e' diversa. La versione con openssl paga la creazione del
// processo, la pipe e le chiamate di sistema, quindi ha CPU kernel non
// trascurabile; questa versione e' interamente in spazio utente.
//
// VERIFICA ATTESA: se lo spazio delle feature descrive il COMPORTAMENTO e non
// l'ALGORITMO, le due funzioni dovrebbero risultare distanti nonostante
// calcolino la stessa cosa. E' un test del significato stesso della metrica di
// similarita', e vale la pena riportarne l'esito qualunque esso sia.

import (
	"encoding/binary"
	"math/bits"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// quarterRound e' l'operazione elementare di ChaCha20: quattro addizioni,
// quattro XOR e quattro rotazioni sui registri dello stato.
func quarterRound(a, b, c, d uint32) (uint32, uint32, uint32, uint32) {
	a += b
	d ^= a
	d = bits.RotateLeft32(d, 16)

	c += d
	b ^= c
	b = bits.RotateLeft32(b, 12)

	a += b
	d ^= a
	d = bits.RotateLeft32(d, 8)

	c += d
	b ^= c
	b = bits.RotateLeft32(b, 7)

	return a, b, c, d
}

// chachaBlock produce 64 byte di flusso cifrante a partire da chiave, contatore
// e nonce, applicando venti round alternati per colonne e per diagonali.
func chachaBlock(out []byte, key []uint32, counter uint32, nonce []uint32) {
	var state [16]uint32

	state[0] = 0x61707865
	state[1] = 0x3320646e
	state[2] = 0x79622d32
	state[3] = 0x6b206574

	copy(state[4:12], key)

	state[12] = counter
	copy(state[13:16], nonce)

	working := state

	for round := 0; round < 10; round++ {
		working[0], working[4], working[8], working[12] =
			quarterRound(working[0], working[4], working[8], working[12])
		working[1], working[5], working[9], working[13] =
			quarterRound(working[1], working[5], working[9], working[13])
		working[2], working[6], working[10], working[14] =
			quarterRound(working[2], working[6], working[10], working[14])
		working[3], working[7], working[11], working[15] =
			quarterRound(working[3], working[7], working[11], working[15])

		working[0], working[5], working[10], working[15] =
			quarterRound(working[0], working[5], working[10], working[15])
		working[1], working[6], working[11], working[12] =
			quarterRound(working[1], working[6], working[11], working[12])
		working[2], working[7], working[8], working[13] =
			quarterRound(working[2], working[7], working[8], working[13])
		working[3], working[4], working[9], working[14] =
			quarterRound(working[3], working[4], working[9], working[14])
	}

	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(out[i*4:], working[i]+state[i])
	}
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 250
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	size := sizeMB * 1024 * 1024

	key := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	nonce := []uint32{9, 10, 11}

	block := make([]byte, 64)
	checksum := uint64(0)

	blocks := size / 64

	for counter := 0; counter < blocks; counter++ {
		chachaBlock(block, key, uint32(counter), nonce)

		// La somma dei primi byte impedisce al compilatore di eliminare il
		// calcolo, senza aggiungere un costo confrontabile con quello della
		// cifratura.
		checksum += uint64(block[0]) + uint64(block[63])
	}

	return map[string]interface{}{
		"message":      "ChaCha20 keystream completed",
		"size_mb":      sizeMB,
		"blocks":       blocks,
		"checksum":     checksum,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
