package main

// Adaptation of the vSwarm AES standalone benchmark.
// Original benchmark: vSwarm/benchmarks/aes/go/server.go
// Copyright (c) 2022 EASE lab — MIT License.
//
// Adaptation policy:
// - AES-128 CTR kernel preserved;
// - original default key and plaintext preserved;
// - gRPC, AWS Lambda glue and distributed tracing removed;
// - Serverledge handler added.

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/hex"
	"fmt"
	"runtime"
	"strings"

	"github.com/serverledge-faas/serverledge/serverledge"
)

const (
	defaultAESKeyHex    = "6368616e676520746869732070617373"
	defaultAESPlaintext = "defaultplaintext"
)

func aesModeCTR(plaintext []byte, keyHex string) ([]byte, error) {
	key, err := hex.DecodeString(keyHex)
	if err != nil {
		return nil, fmt.Errorf("invalid hexadecimal AES key: %w", err)
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("invalid AES key: %w", err)
	}

	iv := make([]byte, aes.BlockSize)
	ciphertext := make([]byte, len(plaintext))

	stream := cipher.NewCTR(block, iv)
	stream.XORKeyStream(ciphertext, plaintext)

	return ciphertext, nil
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	plaintext := defaultAESPlaintext
	if value, ok := params["plaintext"].(string); ok {
		value = strings.TrimSpace(value)
		if value != "" && value != "world" {
			plaintext = value
		}
	}

	keyHex := defaultAESKeyHex
	if value, ok := params["key"].(string); ok {
		value = strings.TrimSpace(value)
		if value != "" {
			keyHex = value
		}
	}

	ciphertext, err := aesModeCTR([]byte(plaintext), keyHex)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"benchmark":      "vswarm-aes",
		"plaintext":      plaintext,
		"ciphertext_hex": hex.EncodeToString(ciphertext),
		"arch":           runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
