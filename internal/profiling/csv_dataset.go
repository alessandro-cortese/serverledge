package profiling

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const FunctionProfileCSVSchemaVersion = 1

type FunctionProfileAggregation string

const (
	FunctionProfileAggregationMean FunctionProfileAggregation = "mean"

	FunctionProfileAggregationMedian FunctionProfileAggregation = "median"
)

// FunctionProfileCSVMetadataNames returns the columns that identify and
// describe a FunctionProfile but must not automatically become clustering
// dimensions.
func FunctionProfileCSVMetadataNames() []string {
	return []string{
		"csv_schema_version",
		"experiment_id",
		"aggregation",
		"function_profile_schema_version",
		"function_name",
		"machine_tag",
		"configured_cpus",
		"configured_memory_mb",
		"sample_count",
	}
}

// FunctionProfileCSVHeader returns the complete and stable CSV schema.
//
// Metadata columns come first. The final columns are exactly the numeric
// profiling dimensions returned by RandomForestFeatureNames.
func FunctionProfileCSVHeader() []string {
	header :=
		FunctionProfileCSVMetadataNames()

	return append(
		header,
		RandomForestFeatureNames()...,
	)
}

// LoadFunctionProfilesJSONL loads the derived FunctionProfile dataset.
//
// Every non-empty line must contain one valid FunctionProfile using the
// currently supported schema.
func LoadFunctionProfilesJSONL(
	path string,
) ([]FunctionProfile, error) {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return nil,
			fmt.Errorf(
				"function profile input path cannot be empty",
			)
	}

	file, err :=
		os.Open(
			path,
		)

	if err != nil {
		return nil,
			fmt.Errorf(
				"failed to open function profile dataset %s: %w",
				path,
				err,
			)
	}

	defer file.Close()

	scanner :=
		bufio.NewScanner(
			file,
		)

	scanner.Buffer(
		make(
			[]byte,
			64*1024,
		),
		4*1024*1024,
	)

	profiles :=
		make(
			[]FunctionProfile,
			0,
		)

	lineNumber := 0

	for scanner.Scan() {
		lineNumber++

		line :=
			strings.TrimSpace(
				scanner.Text(),
			)

		if line == "" {
			continue
		}

		var profile FunctionProfile

		if err :=
			json.Unmarshal(
				[]byte(
					line,
				),
				&profile,
			); err != nil {

			return nil,
				fmt.Errorf(
					"invalid function profile JSON at line %d: %w",
					lineNumber,
					err,
				)
		}

		if err :=
			validateFunctionProfileForCSV(
				profile,
			); err != nil {

			return nil,
				fmt.Errorf(
					"invalid function profile at line %d: %w",
					lineNumber,
					err,
				)
		}

		profiles =
			append(
				profiles,
				profile,
			)
	}

	if err :=
		scanner.Err(); err != nil {

		return nil,
			fmt.Errorf(
				"failed to read function profile dataset %s: %w",
				path,
				err,
			)
	}

	if len(
		profiles,
	) == 0 {

		return nil,
			fmt.Errorf(
				"function profile dataset %s contains no profiles",
				path,
			)
	}

	if err :=
		validateUniqueFunctionProfilesForCSV(
			profiles,
		); err != nil {

		return nil,
			err
	}

	return profiles, nil
}

// ExportFunctionProfilesCSV writes one derived CSV dataset using either the
// mean or median representation of each FunctionProfile.
//
// The output is sorted deterministically and rewritten atomically. The
// original FunctionProfile JSONL remains unchanged.
func ExportFunctionProfilesCSV(
	path string,
	experimentID string,
	aggregation FunctionProfileAggregation,
	profiles []FunctionProfile,
) error {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return fmt.Errorf(
			"function profile CSV output path cannot be empty",
		)
	}

	experimentID =
		strings.TrimSpace(
			experimentID,
		)

	if experimentID == "" {
		return fmt.Errorf(
			"experiment ID cannot be empty",
		)
	}

	if aggregation !=
		FunctionProfileAggregationMean &&
		aggregation !=
			FunctionProfileAggregationMedian {

		return fmt.Errorf(
			"unsupported function profile aggregation %q",
			aggregation,
		)
	}

	if len(
		profiles,
	) == 0 {

		return fmt.Errorf(
			"cannot export an empty function profile dataset",
		)
	}

	for index, profile := range profiles {

		if err :=
			validateFunctionProfileForCSV(
				profile,
			); err != nil {

			return fmt.Errorf(
				"invalid function profile at index %d: %w",
				index,
				err,
			)
		}
	}

	if err :=
		validateUniqueFunctionProfilesForCSV(
			profiles,
		); err != nil {

		return err
	}

	orderedProfiles :=
		append(
			[]FunctionProfile(nil),
			profiles...,
		)

	sort.SliceStable(
		orderedProfiles,
		func(
			i int,
			j int,
		) bool {
			first :=
				orderedProfiles[i]

			second :=
				orderedProfiles[j]

			if first.FunctionName !=
				second.FunctionName {

				return first.FunctionName <
					second.FunctionName
			}

			if first.MachineTag !=
				second.MachineTag {

				return first.MachineTag <
					second.MachineTag
			}

			if first.
				FunctionConfiguration.
				ConfiguredCPUs !=
				second.
					FunctionConfiguration.
					ConfiguredCPUs {

				return first.
					FunctionConfiguration.
					ConfiguredCPUs <
					second.
						FunctionConfiguration.
						ConfiguredCPUs
			}

			return first.
				FunctionConfiguration.
				ConfiguredMemoryMB <
				second.
					FunctionConfiguration.
					ConfiguredMemoryMB
		},
	)

	directory :=
		filepath.Dir(
			path,
		)

	if directory != "." {
		if err :=
			os.MkdirAll(
				directory,
				0o755,
			); err != nil {

			return fmt.Errorf(
				"failed to create function profile CSV directory %s: %w",
				directory,
				err,
			)
		}
	}

	tempFile, err :=
		os.CreateTemp(
			directory,
			".function-profiles-*.csv.tmp",
		)

	if err != nil {
		return fmt.Errorf(
			"failed to create temporary function profile CSV: %w",
			err,
		)
	}

	tempPath :=
		tempFile.Name()

	cleanupTemp := true

	defer func() {
		if cleanupTemp {
			_ =
				os.Remove(
					tempPath,
				)
		}
	}()

	writer :=
		csv.NewWriter(
			tempFile,
		)

	if err :=
		writer.Write(
			FunctionProfileCSVHeader(),
		); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to write function profile CSV header: %w",
			err,
		)
	}

	for index, profile := range orderedProfiles {

		record, err :=
			functionProfileCSVRecord(
				experimentID,
				aggregation,
				profile,
			)

		if err != nil {
			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to build function profile CSV row %d: %w",
				index,
				err,
			)
		}

		if err :=
			writer.Write(
				record,
			); err != nil {

			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to write function profile CSV row %d: %w",
				index,
				err,
			)
		}
	}

	writer.Flush()

	if err :=
		writer.Error(); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to flush function profile CSV: %w",
			err,
		)
	}

	if err :=
		tempFile.Sync(); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to sync temporary function profile CSV: %w",
			err,
		)
	}

	if err :=
		tempFile.Chmod(
			0o644,
		); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to set function profile CSV permissions: %w",
			err,
		)
	}

	if err :=
		tempFile.Close(); err != nil {

		return fmt.Errorf(
			"failed to close temporary function profile CSV: %w",
			err,
		)
	}

	if err :=
		os.Rename(
			tempPath,
			path,
		); err != nil {

		return fmt.Errorf(
			"failed to replace function profile CSV %s: %w",
			path,
			err,
		)
	}

	cleanupTemp = false

	return nil
}

func functionProfileCSVRecord(
	experimentID string,
	aggregation FunctionProfileAggregation,
	profile FunctionProfile,
) ([]string, error) {
	var features RandomForestProfilingFeatures

	switch aggregation {

	case FunctionProfileAggregationMean:
		features =
			profile.Mean

	case FunctionProfileAggregationMedian:
		features =
			profile.Median

	default:
		return nil,
			fmt.Errorf(
				"unsupported function profile aggregation %q",
				aggregation,
			)
	}

	record :=
		[]string{
			strconv.Itoa(
				FunctionProfileCSVSchemaVersion,
			),

			experimentID,

			string(
				aggregation,
			),

			strconv.Itoa(
				profile.SchemaVersion,
			),

			profile.FunctionName,

			profile.MachineTag,

			strconv.FormatFloat(
				profile.
					FunctionConfiguration.
					ConfiguredCPUs,
				'g',
				-1,
				64,
			),

			strconv.FormatInt(
				profile.
					FunctionConfiguration.
					ConfiguredMemoryMB,
				10,
			),

			strconv.Itoa(
				profile.SampleCount,
			),
		}

	for _, value := range features.Values() {

		record =
			append(
				record,
				strconv.FormatFloat(
					value,
					'g',
					-1,
					64,
				),
			)
	}

	return record, nil
}

func validateFunctionProfileForCSV(
	profile FunctionProfile,
) error {
	if profile.SchemaVersion !=
		FunctionProfileSchemaVersion {

		return fmt.Errorf(
			"unsupported function profile schema version %d",
			profile.SchemaVersion,
		)
	}

	if strings.TrimSpace(
		profile.FunctionName,
	) == "" {

		return fmt.Errorf(
			"function name is empty",
		)
	}

	if strings.TrimSpace(
		profile.MachineTag,
	) == "" {

		return fmt.Errorf(
			"machine tag is empty",
		)
	}

	if !finitePositive(
		profile.
			FunctionConfiguration.
			ConfiguredCPUs,
	) {
		return fmt.Errorf(
			"configured CPUs are invalid: %v",
			profile.
				FunctionConfiguration.
				ConfiguredCPUs,
		)
	}

	if profile.
		FunctionConfiguration.
		ConfiguredMemoryMB <= 0 {

		return fmt.Errorf(
			"configured memory is invalid: %d MB",
			profile.
				FunctionConfiguration.
				ConfiguredMemoryMB,
		)
	}

	if profile.SampleCount <
		MinFunctionProfileSamples ||
		profile.SampleCount >
			MaxFunctionProfileSamples {

		return fmt.Errorf(
			"sample count must be between %d and %d, got %d",
			MinFunctionProfileSamples,
			MaxFunctionProfileSamples,
			profile.SampleCount,
		)
	}

	if len(
		profile.SourceRequestIDs,
	) != profile.SampleCount {

		return fmt.Errorf(
			"source request ID count %d does not match sample count %d",
			len(
				profile.SourceRequestIDs,
			),
			profile.SampleCount,
		)
	}

	seenRequestIDs :=
		make(
			map[string]struct{},
			len(
				profile.SourceRequestIDs,
			),
		)

	for _, requestID := range profile.SourceRequestIDs {

		requestID =
			strings.TrimSpace(
				requestID,
			)

		if requestID == "" {
			return fmt.Errorf(
				"source request ID is empty",
			)
		}

		if _, exists :=
			seenRequestIDs[requestID]; exists {

			return fmt.Errorf(
				"duplicate source request ID %q",
				requestID,
			)
		}

		seenRequestIDs[requestID] =
			struct{}{}
	}

	if err :=
		validateAggregatedFeaturesForCSV(
			"mean",
			profile.Mean,
		); err != nil {

		return err
	}

	if err :=
		validateAggregatedFeaturesForCSV(
			"median",
			profile.Median,
		); err != nil {

		return err
	}

	return nil
}

func validateAggregatedFeaturesForCSV(
	aggregationName string,
	features RandomForestProfilingFeatures,
) error {
	names :=
		RandomForestFeatureNames()

	values :=
		features.Values()

	for index, value := range values {

		if !finiteNonNegative(
			value,
		) {
			return fmt.Errorf(
				"%s feature %s is invalid: %v",
				aggregationName,
				names[index],
				value,
			)
		}
	}

	return nil
}

type functionProfileCSVIdentity struct {
	FunctionName string
	MachineTag   string

	ConfiguredCPUs     float64
	ConfiguredMemoryMB int64
}

func validateUniqueFunctionProfilesForCSV(
	profiles []FunctionProfile,
) error {
	seen := make(map[functionProfileCSVIdentity]struct{}, len(profiles))

	for _, profile := range profiles {
		identity := functionProfileCSVIdentity{
			FunctionName:       profile.FunctionName,
			MachineTag:         profile.MachineTag,
			ConfiguredCPUs:     profile.FunctionConfiguration.ConfiguredCPUs,
			ConfiguredMemoryMB: profile.FunctionConfiguration.ConfiguredMemoryMB,
		}

		if _, exists := seen[identity]; exists {
			return fmt.Errorf(
				"duplicate function profile for function %q, machine tag %q, CPUs %g, memory %d MB",
				identity.FunctionName,
				identity.MachineTag,
				identity.ConfiguredCPUs,
				identity.ConfiguredMemoryMB,
			)
		}
		seen[identity] = struct{}{}
	}
	return nil
}
