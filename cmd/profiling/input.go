package main

import (
	"fmt"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

func resolveInvocationSampleInputs(inputPaths stringListFlag, inputDir string) ([]string, error) {
	inputDir = strings.TrimSpace(inputDir)

	if len(inputPaths) > 0 &&
		inputDir != "" {
		return nil, fmt.Errorf("use either --input or --input-dir, not both")
	}

	if len(inputPaths) > 0 {
		return append([]string(nil), inputPaths...), nil
	}

	if inputDir != "" {
		return profiling.DiscoverInvocationSampleDatasets(inputDir)
	}

	return []string{profiling.DefaultInvocationSampleExportPath}, nil
}
