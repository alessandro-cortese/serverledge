package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

func runAggregateCold(args []string) {

	var inputPaths stringListFlag
	flags := flag.NewFlagSet("aggregate-cold", flag.ExitOnError)
	flags.Var(&inputPaths, "input", "input InvocationSample JSONL dataset; may be repeated for multiple nodes")

	inputDir := flags.String("input-dir", "", "directory recursively containing per-node profiling-samples.jsonl datasets")
	outputPath := flags.String("output", profiling.DefaultColdStartProfileExportPath, "output ColdStartProfile JSONL dataset")

	samplesPerProfile := flags.Int("samples", 0, "number of most recent eligible cold starts per profile (required; use 1 only for local smoke tests)")

	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "Usage: serverledge-profiling aggregate-cold [options]\n\n")
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		log.Fatal(err)
	}

	if *samplesPerProfile <= 0 {
		log.Fatal("--samples must be explicitly set to a positive value")
	}

	resolvedInputs, err := resolveInvocationSampleInputs(inputPaths, *inputDir)
	if err != nil {
		log.Fatal(err)
	}

	samples, err := profiling.LoadInvocationSamplesJSONLFiles(resolvedInputs)
	if err != nil {
		log.Fatal(err)
	}

	result, err := profiling.BuildColdStartProfilesByGroup(samples, *samplesPerProfile)
	if err != nil {
		log.Fatal(err)
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

		log.Fatalf("no complete cold-start groups: need at least %d eligible cold starts per group", *samplesPerProfile)
	}

	if err := profiling.ExportColdStartProfilesJSONL(*outputPath, result.Profiles); err != nil {
		log.Fatal(err)
	}

	fmt.Printf(
		"input_datasets=%d raw_samples=%d eligible_cold_samples=%d ignored_samples=%d groups=%d profiles=%d samples_per_profile=%d\n",
		len(resolvedInputs),
		result.RawSampleCount,
		result.EligibleSamples,
		result.IgnoredSamples,
		len(result.Groups),
		len(result.Profiles),
		*samplesPerProfile,
	)

	for _, input := range resolvedInputs {
		fmt.Printf("[input] %s\n", input)
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

	fmt.Printf("output=%s\n", *outputPath)
}

func runExportColdCSV(args []string) {

	flags := flag.NewFlagSet("export-cold-csv", flag.ExitOnError)
	inputPath := flags.String("input", profiling.DefaultColdStartProfileExportPath, "input ColdStartProfile JSONL dataset")
	experimentID := flags.String("experiment-id", "", "experiment identifier stored as CSV metadata (required)")
	outputPath := flags.String("output", "", "cold-start CSV output path; default is next to the input dataset")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "Usage: serverledge-profiling export-cold-csv [options]\n\n")
		flags.PrintDefaults()
	}

	if err := flags.Parse(args); err != nil {
		log.Fatal(err)
	}

	*experimentID = strings.TrimSpace(*experimentID)
	if *experimentID == "" {
		log.Fatal("--experiment-id is required")
	}

	if strings.TrimSpace(*outputPath) == "" {
		*outputPath = filepath.Join(filepath.Dir(*inputPath), filepath.Base(profiling.DefaultColdStartProfileCSVPath))
	}

	profiles, err := profiling.LoadColdStartProfilesJSONL(*inputPath)
	if err != nil {
		log.Fatal(err)
	}

	if err := profiling.ExportColdStartProfilesCSV(*outputPath, *experimentID, profiles); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("profiles=%d experiment_id=%s\n", len(profiles), *experimentID)
	fmt.Printf("output=%s\n", *outputPath)
}
