package profiling

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

const ProfilingFeatureVectorCSVSchemaVersion = 1

// ProfilingFeatureVectorCSVMetadataNames returns the columns that identify one
// per-invocation feature vector without becoming model dimensions.
//
// The set mirrors FunctionProfileCSVMetadataNames, with two differences that
// follow from the granularity: the row is identified by request_id instead of
// sample_count, and there is no aggregation column because no aggregation has
// been applied.
func ProfilingFeatureVectorCSVMetadataNames() []string {
	return []string{
		"sample_csv_schema_version",
		"experiment_id",
		"feature_vector_schema_version",
		"request_id",
		"function_name",
		"machine_tag",
		"configured_cpus",
		"configured_memory_mb",
	}
}

// ProfilingFeatureVectorCSVHeader returns metadata columns followed by the six
// Random Forest features, in the stable order of RandomForestFeatureNames.
func ProfilingFeatureVectorCSVHeader() []string {
	header :=
		ProfilingFeatureVectorCSVMetadataNames()

	return append(
		header,
		RandomForestFeatureNames()...,
	)
}

// BuildProfilingFeatureVectors converts the samples eligible for resource
// clustering into per-invocation feature vectors, without aggregating them.
//
// This is the same conversion BuildFunctionProfileFromSamples performs before
// aggregating: the difference is only that the vectors are returned as they
// are. Samples not eligible for resource clustering are ignored; an eligible
// sample that cannot be converted is an error, because it indicates an
// inconsistency in the dataset.
func BuildProfilingFeatureVectors(
	samples []InvocationSample,
) ([]ProfilingFeatureVector, error) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			0,
			len(samples),
		)

	for index, sample := range samples {

		if !sample.
			Eligibility.
			ResourceClustering {

			continue
		}

		vector, err :=
			BuildProfilingFeatureVector(
				sample,
			)

		if err != nil {
			return nil,
				fmt.Errorf(
					"failed to build feature vector from eligible sample at index %d: %w",
					index,
					err,
				)
		}

		vectors =
			append(
				vectors,
				vector,
			)
	}

	return vectors, nil
}

// ExportProfilingFeatureVectorsCSV writes one row per eligible invocation.
//
// The aggregated FunctionProfile CSV remains the input of clustering and donor
// selection. This dataset serves the supervised classification of the
// architecture preference, which the reference paper performs on individual
// runs and then resolves by majority vote, and would be untrainable on one row
// per function-configuration.
//
// Rows are sorted deterministically and the file is rewritten atomically.
func ExportProfilingFeatureVectorsCSV(
	path string,
	experimentID string,
	vectors []ProfilingFeatureVector,
) error {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return fmt.Errorf(
			"feature vector CSV output path cannot be empty",
		)
	}

	experimentID =
		strings.TrimSpace(
			experimentID,
		)

	if experimentID == "" {
		return fmt.Errorf(
			"feature vector CSV experiment id cannot be empty",
		)
	}

	if len(vectors) == 0 {
		return fmt.Errorf(
			"cannot export an empty feature vector dataset",
		)
	}

	seenRequestIDs :=
		make(
			map[string]struct{},
			len(vectors),
		)

	for index, vector := range vectors {

		if err :=
			validateFeatureVectorForCSV(
				vector,
			); err != nil {

			return fmt.Errorf(
				"invalid feature vector at index %d: %w",
				index,
				err,
			)
		}

		if _, duplicate :=
			seenRequestIDs[vector.RequestID]; duplicate {

			return fmt.Errorf(
				"duplicate request id %q in feature vector dataset",
				vector.RequestID,
			)
		}

		seenRequestIDs[vector.RequestID] =
			struct{}{}
	}

	ordered :=
		make(
			[]ProfilingFeatureVector,
			len(vectors),
		)

	copy(
		ordered,
		vectors,
	)

	// Deterministic order: function, machine tag, configuration, then request
	// id as the final tie-breaker, which is unique by the check above.
	sort.SliceStable(
		ordered,
		func(
			i int,
			j int,
		) bool {
			if ordered[i].FunctionName !=
				ordered[j].FunctionName {

				return ordered[i].FunctionName <
					ordered[j].FunctionName
			}

			if ordered[i].MachineTag !=
				ordered[j].MachineTag {

				return ordered[i].MachineTag <
					ordered[j].MachineTag
			}

			if ordered[i].
				FunctionConfiguration.
				ConfiguredCPUs !=
				ordered[j].
					FunctionConfiguration.
					ConfiguredCPUs {

				return ordered[i].
					FunctionConfiguration.
					ConfiguredCPUs <
					ordered[j].
						FunctionConfiguration.
						ConfiguredCPUs
			}

			if ordered[i].
				FunctionConfiguration.
				ConfiguredMemoryMB !=
				ordered[j].
					FunctionConfiguration.
					ConfiguredMemoryMB {

				return ordered[i].
					FunctionConfiguration.
					ConfiguredMemoryMB <
					ordered[j].
						FunctionConfiguration.
						ConfiguredMemoryMB
			}

			return ordered[i].RequestID <
				ordered[j].RequestID
		},
	)

	tempFile, tempPath, err :=
		createAtomicFile(
			path,
			".profiling-feature-vectors-*.csv.tmp",
			"feature vector CSV",
		)

	if err != nil {
		return err
	}

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
			ProfilingFeatureVectorCSVHeader(),
		); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to write feature vector CSV header: %w",
			err,
		)
	}

	for index, vector := range ordered {

		if err :=
			writer.Write(
				profilingFeatureVectorCSVRecord(
					experimentID,
					vector,
				),
			); err != nil {

			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to write feature vector at index %d: %w",
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
			"failed to flush feature vector CSV: %w",
			err,
		)
	}

	if err :=
		finalizeAtomicFile(
			tempFile,
			tempPath,
			path,
			"feature vector CSV",
		); err != nil {

		return err
	}

	cleanupTemp = false

	return nil
}

func profilingFeatureVectorCSVRecord(
	experimentID string,
	vector ProfilingFeatureVector,
) []string {
	record :=
		[]string{
			strconv.Itoa(
				ProfilingFeatureVectorCSVSchemaVersion,
			),

			experimentID,

			strconv.Itoa(
				vector.SchemaVersion,
			),

			vector.RequestID,

			vector.FunctionName,

			vector.MachineTag,

			strconv.FormatFloat(
				vector.
					FunctionConfiguration.
					ConfiguredCPUs,
				'g',
				-1,
				64,
			),

			strconv.FormatInt(
				vector.
					FunctionConfiguration.
					ConfiguredMemoryMB,
				10,
			),
		}

	for _, value := range vector.Features.Values() {

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

	return record
}

func validateFeatureVectorForCSV(
	vector ProfilingFeatureVector,
) error {
	if vector.SchemaVersion !=
		ProfilingFeatureVectorSchemaVersion {

		return fmt.Errorf(
			"unsupported feature vector schema version %d: expected %d",
			vector.SchemaVersion,
			ProfilingFeatureVectorSchemaVersion,
		)
	}

	if strings.TrimSpace(
		vector.RequestID,
	) == "" {

		return fmt.Errorf(
			"request id cannot be empty",
		)
	}

	if strings.TrimSpace(
		vector.FunctionName,
	) == "" {

		return fmt.Errorf(
			"function name cannot be empty",
		)
	}

	if strings.TrimSpace(
		vector.MachineTag,
	) == "" {

		return fmt.Errorf(
			"machine tag cannot be empty",
		)
	}

	return validateAggregatedFeaturesForCSV(
		"sample",
		vector.Features,
	)
}
