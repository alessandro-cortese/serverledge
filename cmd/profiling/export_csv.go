package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

func runExportCSV(args []string) {

	flags := flag.NewFlagSet("export-csv", flag.ExitOnError)
	inputPath := flags.String("input", profiling.DefaultFunctionProfileExportPath, "input FunctionProfile JSONL dataset")
	experimentID := flags.String("experiment-id", "", "experiment identifier stored as CSV metadata (required)")
	meanOutputPath := flags.String("mean-output", "", "mean CSV output path; default is next to the input dataset")
	medianOutputPath := flags.String("median-output", "", "median CSV output path; default is next to the input dataset")

	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "Usage: serverledge-profiling export-csv [options]\n\n")
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		log.Fatal(err)
	}

	*experimentID = strings.TrimSpace(*experimentID)
	if *experimentID == "" {
		log.Fatal("--experiment-id is required")
	}

	inputDirectory := filepath.Dir(*inputPath)
	if strings.TrimSpace(*meanOutputPath) == "" {
		*meanOutputPath = filepath.Join(inputDirectory, "function-profiles-mean.csv")
	}

	if strings.TrimSpace(*medianOutputPath) == "" {
		*medianOutputPath = filepath.Join(inputDirectory, "function-profiles-median.csv")
	}

	profiles, err := profiling.LoadFunctionProfilesJSONL(*inputPath)
	if err != nil {
		log.Fatal(err)
	}

	if err := profiling.ExportFunctionProfilesCSV(*meanOutputPath, *experimentID, profiling.FunctionProfileAggregationMean, profiles); err != nil {
		log.Fatal(err)
	}

	if err := profiling.ExportFunctionProfilesCSV(*medianOutputPath, *experimentID, profiling.FunctionProfileAggregationMedian, profiles); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("profiles=%d experiment_id=%s\n", len(profiles), *experimentID)
	fmt.Printf("mean_output=%s\n", *meanOutputPath)
	fmt.Printf("median_output=%s\n", *medianOutputPath)
}
