package main

// matmul — SINTETICA. Moltiplicazione di matrici a blocchi.
//
// Profilo atteso: CPU utente in virgola mobile molto alta, page fault bassi.
// Il blocking rende l'accesso favorevole alla cache, quindi il carico resta
// dominato dall'aritmetica invece che dalla banda di memoria. E' il
// corrispettivo in Go di linpack, che pero' usa numpy e quindi BLAS ottimizzate.

import (
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	n := 512
	if val, ok := params["size"].(float64); ok {
		n = int(val)
	}

	// Dimensione del blocco scelta perche' tre sottomatrici da 64×64 float64
	// stiano comodamente nella cache di secondo livello.
	block := 64

	a := make([]float64, n*n)
	b := make([]float64, n*n)
	c := make([]float64, n*n)

	for i := range a {
		a[i] = float64(i%97) * 0.5
		b[i] = float64(i%89) * 0.25
	}

	for ii := 0; ii < n; ii += block {
		for jj := 0; jj < n; jj += block {
			for kk := 0; kk < n; kk += block {

				iMax := ii + block
				if iMax > n {
					iMax = n
				}

				for i := ii; i < iMax; i++ {
					jMax := jj + block
					if jMax > n {
						jMax = n
					}

					for j := jj; j < jMax; j++ {
						sum := c[i*n+j]

						kMax := kk + block
						if kMax > n {
							kMax = n
						}

						for k := kk; k < kMax; k++ {
							sum += a[i*n+k] * b[k*n+j]
						}

						c[i*n+j] = sum
					}
				}
			}
		}
	}

	trace := 0.0
	for i := 0; i < n; i++ {
		trace += c[i*n+i]
	}

	return map[string]interface{}{
		"message": "Matrix multiplication completed",
		"size":    n,
		"trace":   trace,
		"arch":    runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
