package main

// vswarm-fibonacci — adattamento del benchmark Fibonacci di vSwarm.
//
// Il kernel fibonacci() è mantenuto nella stessa forma dell'implementazione
// Go originale di vSwarm, usando math/big e riutilizzando tre big.Int.
//
// Il default originale usato nel primo preflight (n=10000) risultava troppo
// breve per la profilazione Serverledge. La calibrazione locale del kernel
// originale ha portato alla scelta di:
//
//     n = 200000
//
// che mantiene invariato l'algoritmo ma rende il campione sufficientemente
// lungo rispetto all'overhead del profiler.
//
// Il risultato completo non viene convertito in una gigantesca stringa
// decimale: vengono restituiti metadati compatti sul BigInt, così il profilo
// resta concentrato sul kernel Fibonacci.

import (
	"fmt"
	"math/big"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

const defaultFibonacciN = 200000

// Kernel mantenuto equivalente all'implementazione Go originale di vSwarm.
func fibonacci(num int) *big.Int {
	if num <= 0 {
		return big.NewInt(0)
	}

	num1 := big.NewInt(0)
	num2 := big.NewInt(1)
	sum := big.NewInt(0)

	for i := 0; i < num; i++ {
		sum.Add(num1, num2)
		num1.Set(num2)
		num2.Set(sum)
	}

	return num1
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	n := defaultFibonacciN

	if val, ok := params["n"].(float64); ok {
		n = int(val)
	}

	if n < 0 {
		return nil, fmt.Errorf("n must be >= 0")
	}

	result := fibonacci(n)

	return map[string]interface{}{
		"message":    "vSwarm Fibonacci completed",
		"n":          n,
		"bit_length": result.BitLen(),
		"low_64":     result.Uint64(),
		"arch":       runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
