package profiling

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

const PaperRFFeatureVectorCSVSchemaVersion = 1

// PaperRFFeatureVectorCSVMetadataNames returns metadata columns that identify a
// per-invocation example without becoming Random Forest dimensions.
func PaperRFFeatureVectorCSVMetadataNames() []string {
	return []string{
		"paper_rf_csv_schema_version",
		"experiment_id",
		"paper_rf_feature_vector_schema_version",
		"request_id",
		"function_name",
		"machine_tag",
		"configured_cpus",
		"configured_memory_mb",
	}
}

// PaperRFFeatureVectorCSVHeader returns metadata followed by the exact 14
// paper-artifact feature names.
func PaperRFFeatureVectorCSVHeader() []string {
	header :=
		PaperRFFeatureVectorCSVMetadataNames()

	return append(
		header,
		PaperRFFeatureNames()...,
	)
}

// ExportPaperRFFeatureVectorsCSV writes one row per usable warm invocation.
// Rows are sorted deterministically and the file is replaced atomically.
func ExportPaperRFFeatureVectorsCSV(
	path string,
	experimentID string,
	vectors []PaperRFFeatureVector,
) error {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return fmt.Errorf(
			"paper RF CSV output path cannot be empty",
		)
	}

	experimentID =
		strings.TrimSpace(
			experimentID,
		)

	if experimentID == "" {
		return fmt.Errorf(
			"paper RF CSV experiment id cannot be empty",
		)
	}

	if len(vectors) == 0 {
		return fmt.Errorf(
			"cannot export an empty paper RF feature vector dataset",
		)
	}

	seenRequestIDs :=
		make(
			map[string]struct{},
			len(vectors),
		)

	for index, vector := range vectors {
		if err :=
			validatePaperRFFeatureVectorForCSV(
				vector,
			); err != nil {

			return fmt.Errorf(
				"invalid paper RF feature vector at index %d: %w",
				index,
				err,
			)
		}

		if _, duplicate :=
			seenRequestIDs[vector.RequestID]; duplicate {

			return fmt.Errorf(
				"duplicate request id %q in paper RF feature vector dataset",
				vector.RequestID,
			)
		}

		seenRequestIDs[vector.RequestID] =
			struct{}{}
	}

	ordered :=
		make(
			[]PaperRFFeatureVector,
			len(vectors),
		)

	copy(
		ordered,
		vectors,
	)

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
			".paper-rf-feature-vectors-*.csv.tmp",
			"paper RF feature vector CSV",
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
			PaperRFFeatureVectorCSVHeader(),
		); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to write paper RF CSV header: %w",
			err,
		)
	}

	for index, vector := range ordered {
		if err :=
			writer.Write(
				paperRFFeatureVectorCSVRecord(
					experimentID,
					vector,
				),
			); err != nil {

			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to write paper RF feature vector at index %d: %w",
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
			"failed to flush paper RF feature vector CSV: %w",
			err,
		)
	}

	if err :=
		finalizeAtomicFile(
			tempFile,
			tempPath,
			path,
			"paper RF feature vector CSV",
		); err != nil {

		return err
	}

	cleanupTemp = false

	return nil
}

func paperRFFeatureVectorCSVRecord(
	experimentID string,
	vector PaperRFFeatureVector,
) []string {
	record :=
		[]string{
			strconv.Itoa(
				PaperRFFeatureVectorCSVSchemaVersion,
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

func validatePaperRFFeatureVectorForCSV(
	vector PaperRFFeatureVector,
) error {
	if vector.SchemaVersion !=
		PaperRFFeatureVectorSchemaVersion {

		return fmt.Errorf(
			"unsupported paper RF feature vector schema version %d: expected %d",
			vector.SchemaVersion,
			PaperRFFeatureVectorSchemaVersion,
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

	names := PaperRFFeatureNames()
	values := vector.Features.Values()

	for index, value := range values {
		if !finiteNonNegative(value) {
			return fmt.Errorf(
				"paper RF feature %s is invalid: %v",
				names[index],
				value,
			)
		}
	}

	if !finitePositive(vector.Features.TotalMemory) {
		return fmt.Errorf(
			"paper RF feature totalMemory must be positive",
		)
	}

	return nil
}
