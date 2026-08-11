package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

type stringListFlag []string

func (values *stringListFlag) String() string {
	return strings.Join(
		*values,
		",",
	)
}

func (values *stringListFlag) Set(
	value string,
) error {
	value = strings.TrimSpace(
		value,
	)

	if value == "" {
		return fmt.Errorf(
			"input path cannot be empty",
		)
	}

	*values = append(
		*values,
		value,
	)

	return nil
}

func main() {
	log.SetFlags(
		0,
	)

	if len(os.Args) < 2 {
		printUsage()

		os.Exit(
			2,
		)
	}

	switch os.Args[1] {

	case "aggregate":
		runAggregate(
			os.Args[2:],
		)

	case "help",
		"-h",
		"--help":

		printUsage()

	default:
		log.Fatalf(
			"unknown profiling command %q",
			os.Args[1],
		)
	}
}

func runAggregate(
	args []string,
) {
	flags := flag.NewFlagSet(
		"aggregate",
		flag.ExitOnError,
	)

	var inputPaths stringListFlag

	flags.Var(
		&inputPaths,
		"input",
		"input InvocationSample JSONL dataset; may be repeated for multiple nodes",
	)

	inputDir := flags.String(
		"input-dir",
		"",
		"directory recursively containing per-node profiling-samples.jsonl datasets",
	)

	outputPath := flags.String(
		"output",
		profiling.DefaultFunctionProfileExportPath,
		"output FunctionProfile JSONL dataset",
	)

	samplesPerProfile := flags.Int(
		"samples",
		profiling.MaxFunctionProfileSamples,
		"number of most recent eligible warm samples per function profile (10-20)",
	)

	flags.Usage =
		func() {
			fmt.Fprintf(
				flags.Output(),
				"Usage: serverledge-profiling aggregate [options]\n\n",
			)

			flags.PrintDefaults()
		}

	if err := flags.Parse(
		args,
	); err != nil {

		log.Fatal(
			err,
		)
	}

	// Use either explicitly listed input files or one collection directory.
	// Mixing the two forms is rejected to avoid accidentally loading the same
	// raw dataset twice.
	if len(inputPaths) > 0 &&
		strings.TrimSpace(
			*inputDir,
		) != "" {

		log.Fatal(
			"use either --input or --input-dir, not both",
		)
	}

	resolvedInputs := make(
		[]string,
		0,
		len(inputPaths),
	)

	if len(inputPaths) > 0 {
		resolvedInputs = append(
			resolvedInputs,
			inputPaths...,
		)
	} else if strings.TrimSpace(
		*inputDir,
	) != "" {

		discovered, err :=
			profiling.DiscoverInvocationSampleDatasets(
				*inputDir,
			)

		if err != nil {
			log.Fatal(
				err,
			)
		}

		resolvedInputs = append(
			resolvedInputs,
			discovered...,
		)
	} else {
		// Preserve the behavior of the original single-node command.
		resolvedInputs = []string{
			profiling.DefaultInvocationSampleExportPath,
		}
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

	result, err :=
		profiling.BuildFunctionProfilesByGroup(
			samples,
			*samplesPerProfile,
		)

	if err != nil {
		log.Fatal(
			err,
		)
	}

	if len(result.Profiles) == 0 {
		for _, group := range result.Groups {
			fmt.Printf(
				"[skip] function=%s machine_tag=%s cpus=%g memory_mb=%d eligible=%d required=%d\n",
				group.FunctionName,
				group.MachineTag,
				group.FunctionConfiguration.ConfiguredCPUs,
				group.FunctionConfiguration.ConfiguredMemoryMB,
				group.EligibleSampleCount,
				*samplesPerProfile,
			)
		}

		log.Fatalf(
			"no complete profiling groups: need at least %d eligible samples per group",
			*samplesPerProfile,
		)
	}

	if err :=
		profiling.ExportFunctionProfilesJSONL(
			*outputPath,
			result.Profiles,
		); err != nil {

		log.Fatal(
			err,
		)
	}

	fmt.Printf(
		"input_datasets=%d raw_samples=%d eligible_samples=%d ignored_samples=%d groups=%d profiles=%d samples_per_profile=%d\n",
		len(resolvedInputs),
		result.RawSampleCount,
		result.EligibleSamples,
		result.IgnoredSamples,
		len(result.Groups),
		len(result.Profiles),
		*samplesPerProfile,
	)

	for _, input := range resolvedInputs {
		fmt.Printf(
			"[input] %s\n",
			input,
		)
	}

	for _, group := range result.Groups {
		status := "skipped"

		if group.Built {
			status = "built"
		}

		fmt.Printf(
			"[%s] function=%s machine_tag=%s cpus=%g memory_mb=%d eligible=%d selected=%d\n",
			status,
			group.FunctionName,
			group.MachineTag,
			group.FunctionConfiguration.ConfiguredCPUs,
			group.FunctionConfiguration.ConfiguredMemoryMB,
			group.EligibleSampleCount,
			group.SelectedSampleCount,
		)
	}

	fmt.Printf(
		"output=%s\n",
		*outputPath,
	)
}

func printUsage() {
	fmt.Println(
		"Usage: serverledge-profiling <command> [options]",
	)

	fmt.Println()

	fmt.Println(
		"Commands:",
	)

	fmt.Println(
		"  aggregate    build FunctionProfile records from one or more raw InvocationSample JSONL datasets",
	)
}
