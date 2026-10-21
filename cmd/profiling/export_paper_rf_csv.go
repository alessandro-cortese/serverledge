package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

// runExportPaperRFCSV exports one row per usable schema-6 warm invocation using
// the 14-feature mapping that mirrors the published SAAF Random Forest artifact.
func runExportPaperRFCSV(args []string) {
	flags :=
		flag.NewFlagSet(
			"export-paper-rf-csv",
			flag.ExitOnError,
		)

	var inputPaths stringListFlag

	flags.Var(
		&inputPaths,
		"input",
		"input InvocationSample schema-6 JSONL dataset; may be repeated for multiple nodes",
	)

	inputDir :=
		flags.String(
			"input-dir",
			"",
			"directory recursively containing per-node profiling-samples.jsonl datasets",
		)

	experimentID :=
		flags.String(
			"experiment-id",
			"",
			"experiment identifier stored as CSV metadata (required)",
		)

	outputPath :=
		flags.String(
			"output",
			"",
			"paper RF CSV output path (required)",
		)

	flags.Usage = func() {
		fmt.Fprintf(
			flags.Output(),
			"Usage: serverledge-profiling export-paper-rf-csv [options]\n\n",
		)

		flags.PrintDefaults()
	}

	if err :=
		flags.Parse(
			args,
		); err != nil {

		log.Fatal(
			err,
		)
	}

	*experimentID =
		strings.TrimSpace(
			*experimentID,
		)

	if *experimentID == "" {
		log.Fatal(
			"--experiment-id is required",
		)
	}

	*outputPath =
		strings.TrimSpace(
			*outputPath,
		)

	if *outputPath == "" {
		log.Fatal(
			"--output is required",
		)
	}

	resolvedInputs, err :=
		resolveInvocationSampleInputs(
			inputPaths,
			*inputDir,
		)

	if err != nil {
		log.Fatal(
			err,
		)
	}

	samples, err :=
		profiling.LoadInvocationSamplesJSONLFiles(
			resolvedInputs,
		)

	if err != nil {
		log.Fatal(
			err,
		)
	}

	vectors, err :=
		profiling.BuildPaperRFFeatureVectors(
			samples,
		)

	if err != nil {
		log.Fatal(
			err,
		)
	}

	if len(vectors) == 0 {
		log.Fatal(
			"no schema-6 warm sample contains all measurements required by the paper RF adapter",
		)
	}

	if err :=
		profiling.ExportPaperRFFeatureVectorsCSV(
			*outputPath,
			*experimentID,
			vectors,
		); err != nil {

		log.Fatal(
			err,
		)
	}

	fmt.Printf(
		"input datasets: %d\n",
		len(resolvedInputs),
	)

	fmt.Printf(
		"raw samples: %d\n",
		len(samples),
	)

	fmt.Printf(
		"paper RF feature vectors: %d\n",
		len(vectors),
	)

	fmt.Printf(
		"ignored samples: %d\n",
		len(samples)-len(vectors),
	)

	fmt.Printf(
		"output: %s\n",
		*outputPath,
	)
}
