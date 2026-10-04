package main

// Adaptation of the vSwarm Fibonacci standalone benchmark.
// Original benchmark: vSwarm/benchmarks/fibonacci/go/server.go
// Copyright (c) 2022 EASE lab — MIT License.
//
// Adaptation policy:
// - original iterative math/big Fibonacci kernel preserved;
// - gRPC and distributed tracing removed;
// - Serverledge handler added;
// - default n is only a Serverledge profiling input and will be
//   calibrated during the local preflight.

import (
	"math/big"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

const defaultFibonacciN = 10000

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
	if value, ok := params["n"].(float64); ok {
		n = int(value)
	}

	result := fibonacci(n)

	return map[string]interface{}{
		"benchmark": "vswarm-fibonacci",
		"n":         n,
		"result":    result.String(),
		"arch":      runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
