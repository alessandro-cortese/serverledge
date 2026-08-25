package main

import (
	"compress/gzip"
	"io"
	"runtime"
	"sync"

	"github.com/serverledge-faas/serverledge/serverledge"
	_ "go.uber.org/automaxprocs"
)

// Handler is the entry point for Serverledge.
// It performs parallel Gzip compression utilizing all available logical CPUs.
func myHandler(params map[string]interface{}) (interface{}, error) {
	// Default base size: 128MB
	baseSize := 256 * 1024 * 1024
	if val, ok := params["size"].(float64); ok {
		baseSize = int(val)
	}

	// Multiplier to increase total workload: default 4
	multiplier := 10

	// Prepare data
	data := make([]byte, baseSize)
	for i := 0; i < len(data); i++ {
		data[i] = byte(i)
	}

	// Detect available CPUs to spawn correct number of goroutines
	numThreads := runtime.NumCPU()

	// Calculate total workload and chunk size per thread
	totalDataSize := baseSize * multiplier
	chunkSize := totalDataSize / numThreads
	if chunkSize > len(data) {
		chunkSize = len(data)
	}

	var wg sync.WaitGroup

	// Execution: Parallel compression
	for t := 0; t < numThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// We define the source slice based on chunk size.
			// Note: We are compressing the same memory region repeatedly if total > base,
			// but this is fine for testing CPU throughput.
			src := data[:chunkSize]

			w := gzip.NewWriter(io.Discard)
			_, _ = w.Write(src)
			w.Close()
		}()
	}

	wg.Wait()

	return map[string]interface{}{
		"message":    "Gzip compression completed",
		"total_size": totalDataSize,
		"arch":       runtime.GOARCH,
		"cpu":        runtime.NumCPU(),
	}, nil
}

func main() {
	// Start the Serverledge runtime. This is a blocking function since it will start an HTTP server inside the
	// container, waiting for the signal to execute the function.
	serverledge.Start(myHandler)
}
