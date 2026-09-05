package main

// tempfileio — SINTETICA, gruppo "riempimento".
//
// Scopo: occupare la regione I/O intensiva scrivendo e rileggendo file
// temporanei. A differenza di readdisk, che con sysbench fileio misura la
// lettura su file preesistenti, qui il carico e' misto scrittura-lettura e
// include la creazione e la rimozione dei file.
//
// Profilo atteso: CPU kernel alta, page fault medi, CPU utente bassa.

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	files := 200
	if val, ok := params["files"].(float64); ok {
		files = int(val)
	}

	fileSizeKB := 512
	if val, ok := params["file_size_kb"].(float64); ok {
		fileSizeKB = int(val)
	}

	rounds := 4
	if val, ok := params["rounds"].(float64); ok {
		rounds = int(val)
	}

	dir, err := os.MkdirTemp("", "tempfileio")
	if err != nil {
		return nil, err
	}

	// La rimozione e' differita: se la funzione fallisce a meta', i file
	// temporanei non restano sul nodo a occupare spazio fra un'invocazione
	// e l'altra.
	defer os.RemoveAll(dir)

	block := make([]byte, fileSizeKB*1024)
	for i := range block {
		block[i] = byte(i)
	}

	bytesWritten := 0
	bytesRead := 0

	for round := 0; round < rounds; round++ {
		for f := 0; f < files; f++ {
			path := filepath.Join(dir, "chunk")

			if err := os.WriteFile(path, block, 0o600); err != nil {
				return nil, err
			}

			bytesWritten += len(block)

			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}

			bytesRead += len(data)

			if err := os.Remove(path); err != nil {
				return nil, err
			}
		}
	}

	return map[string]interface{}{
		"message":       "Temp file I/O completed",
		"files":         files * rounds,
		"bytes_written": bytesWritten,
		"bytes_read":    bytesRead,
		"arch":          runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
