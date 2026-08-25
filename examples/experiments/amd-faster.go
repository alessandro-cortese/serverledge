package main

import (
	"crypto/aes"
	"crypto/cipher"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// Handler is the entry point for Serverledge.
// It performs AES-GCM encryption on a generated dataset.
func myHandler(params map[string]interface{}) (interface{}, error) {
	// Default data size: 128MB
	dataSize := 256 * 1024 * 1024
	if val, ok := params["size"].(float64); ok {
		dataSize = int(val)
	}

	// Default passes: 5
	passes := 5

	// Prepare random-like data (deterministic fill to avoid generation overhead)
	data := make([]byte, dataSize)
	for i := 0; i < len(data); i++ {
		data[i] = byte(i)
	}

	key := make([]byte, 32) // 256-bit key
	nonce := make([]byte, 12)

	block, err := aes.NewCipher(key)
	if err != nil {
		panic(0)
	}

	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		panic(0)
	}

	// Destination buffer with capacity for overhead
	dst := make([]byte, 0, len(data)+aesgcm.Overhead())

	// Execution: Encrypt data multiple times on a single thread
	// This stresses the CPU pipeline and vector instructions (VAES/NEON)
	for i := 0; i < passes; i++ {
		dst = aesgcm.Seal(dst[:0], nonce, data, nil)
	}

	return map[string]interface{}{
		"message":      "AES encryption completed",
		"processed_gb": (float64(dataSize) * float64(passes)) / 1024 / 1024 / 1024,
		"passes":       passes,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	// Start the Serverledge runtime. This is a blocking function since it will start an HTTP server inside the
	// container, waiting for the signal to execute the function.
	serverledge.Start(myHandler)
}
