package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

// runExportSamplesCSV writes one CSV row per eligible warm invocation, without
// aggregating.
//
// The aggregated dataset produced by `aggregate` + `export-csv` remains the
// input of clustering and donor selection. This command serves the supervised
// classification of the architecture preference, which the reference paper
// performs on individual runs and then resolves by majority vote: with one row
// per function-configuration there would be too few examples to train on.
func runExportSamplesCSV(args []string) {

	flags := flag.NewFlagSet("export-samples-csv", flag.ExitOnError)
	var inputPaths stringListFlag
	flags.Var(&inputPaths, "input", "input InvocationSample JSONL dataset; may be repeated for multiple nodes")
	inputDir := flags.String("input-dir", "", "directory recursively containing per-node profiling-samples.jsonl datasets")
	experimentID := flags.String("experiment-id", "", "experiment identifier stored as CSV metadata (required)")
	outputPath := flags.String("output", "", "feature vector CSV output path (required)")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "Usage: serverledge-profiling export-samples-csv [options]\n\n")
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		log.Fatal(err)
	}

	*experimentID = strings.TrimSpace(*experimentID)
	if *experimentID == "" {
		log.Fatal("--experiment-id is required")
	}

	*outputPath = strings.TrimSpace(*outputPath)
	if *outputPath == "" {
		log.Fatal("--output is required")
	}

	resolvedInputs, err := resolveInvocationSampleInputs(inputPaths, *inputDir)

	if err != nil {
		log.Fatal(err)
	}

	samples, err := profiling.LoadInvocationSamplesJSONLFiles(resolvedInputs)
	if err != nil {
		log.Fatal(err)
	}

	vectors, err := profiling.BuildProfilingFeatureVectors(samples)
	if err != nil {
		log.Fatal(err)
	}

	if len(vectors) == 0 {
		log.Fatal("no sample eligible for resource clustering in the input datasets")
	}

	if err := profiling.ExportProfilingFeatureVectorsCSV(*outputPath, *experimentID, vectors); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("raw samples: %d\n", len(samples))
	fmt.Printf("eligible feature vectors: %d\n", len(vectors))
	fmt.Printf("output: %s\n", *outputPath)
}
