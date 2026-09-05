package main

// compression — adattamento del benchmark 311.compression di SeBS.
//
// L'originale comprime un archivio letto da storage; qui i dati vengono
// generati in-funzione per evitare la dipendenza esterna.
//
// Profilo atteso: CPU utente alta con memoria moderata. La compressione
// alterna letture sequenziali e ricerche nella finestra scorrevole, quindi
// il pattern di accesso e' intermedio fra quello di hashing, strettamente
// sequenziale, e quello di sorting.

import (
	"bytes"
	"compress/gzip"
	"io"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	sizeMB := 32
	if val, ok := params["size_mb"].(float64); ok {
		sizeMB = int(val)
	}

	// Ridotto da 6 a 2 in fase di taratura.
	rounds := 2
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	size := sizeMB * 1024 * 1024
	source := make([]byte, size)

	// Dati parzialmente comprimibili: byte casuali sarebbero incomprimibili e
	// il compressore si limiterebbe a copiarli, mentre dati costanti
	// verrebbero compressi istantaneamente. La struttura ripetitiva con
	// rumore riproduce un carico realistico.
	state := uint64(88172645463325252)
	for i := range source {
		if i%16 < 12 {
			source[i] = byte(i % 251)
		} else {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			source[i] = byte(state)
		}
	}

	compressedBytes := 0
	decompressedBytes := 0

	for round := 0; round < rounds; round++ {
		var buffer bytes.Buffer

		writer, err := gzip.NewWriterLevel(&buffer, gzip.BestSpeed)
		if err != nil {
			return nil, err
		}

		if _, err := writer.Write(source); err != nil {
			return nil, err
		}

		if err := writer.Close(); err != nil {
			return nil, err
		}

		compressedBytes += buffer.Len()

		reader, err := gzip.NewReader(&buffer)
		if err != nil {
			return nil, err
		}

		written, err := io.Copy(io.Discard, reader)
		if err != nil {
			return nil, err
		}

		_ = reader.Close()

		decompressedBytes += int(written)
	}

	return map[string]interface{}{
		"message":       "Compression completed",
		"size_mb":       sizeMB,
		"rounds":        rounds,
		"ratio":         float64(compressedBytes) / float64(size*rounds),
		"decompressed":  decompressedBytes,
		"arch":          runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
