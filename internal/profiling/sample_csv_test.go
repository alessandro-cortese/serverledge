package profiling

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sampleForFeatureVector builds a minimal InvocationSample that is eligible for
// resource clustering, so that the tests below can focus on the export instead
// of on the construction of the profiles.
func sampleForFeatureVector(
	requestID string,
	functionName string,
	machineTag string,
	configuredCPUs float64,
	configuredMemoryMB int64,
	eligible bool,
) InvocationSample {
	return InvocationSample{
		RequestID:    requestID,
		FunctionName: functionName,
		MachineTag:   machineTag,

		FunctionConfiguration: InvocationFunctionConfiguration{
			ConfiguredCPUs:     configuredCPUs,
			ConfiguredMemoryMB: configuredMemoryMB,
		},

		Timing: InvocationTiming{
			DurationMs: 100,
		},

		Profile: &InvocationResourceProfile{
			PageFaultsAvailable: true,
			PageFaultsDelta:     12,

			CPUUsageUserDeltaNs: 60_000_000,

			CPUUsageKernelDeltaNs: 20_000_000,

			ProfilingStartOverheadMs: 2.5,
		},

		NodeEnvironment: &NodeResourceProfile{
			MemoryAvailable: true,

			FreeMemoryBeforeBytes: 1024 *
				1024 *
				2048,

			SnapshotStartOverheadMs: 1.5,
		},

		Eligibility: InvocationEligibility{
			ResourceClustering: eligible,
		},
	}
}

func readCSV(
	t *testing.T,
	path string,
) [][]string {
	t.Helper()

	file, err :=
		os.Open(
			path,
		)

	require.NoError(
		t,
		err,
	)

	defer file.Close()

	records, err :=
		csv.NewReader(
			file,
		).ReadAll()

	require.NoError(
		t,
		err,
	)

	return records
}

// TestProfilingFeatureVectorCSVHeaderIsStable pins the exported contract.
//
// The header is duplicated by hand in analysis/profiling/classify_architecture.py
// (SAMPLE_METADATA + preprocess.FEATURE_NAMES): a silent change here would only
// surface at runtime, on the first dataset the classifier tries to read.
func TestProfilingFeatureVectorCSVHeaderIsStable(
	t *testing.T,
) {
	assert.Equal(
		t,
		[]string{
			"sample_csv_schema_version",
			"experiment_id",
			"feature_vector_schema_version",
			"request_id",
			"function_name",
			"machine_tag",
			"configured_cpus",
			"configured_memory_mb",
			"page_faults_delta",
			"utilized_cpus",
			"free_memory_mb",
			"cpu_user_delta_ms",
			"cpu_kernel_delta_ms",
			"framework_runtime_ms",
		},
		ProfilingFeatureVectorCSVHeader(),
	)

	// The feature block must remain the tail of the header, in the same order
	// the clustering pipeline expects.
	header :=
		ProfilingFeatureVectorCSVHeader()

	assert.Equal(
		t,
		RandomForestFeatureNames(),
		header[len(ProfilingFeatureVectorCSVMetadataNames()):],
	)
}

// TestBuildProfilingFeatureVectorsKeepsOnlyEligibleSamples documents that the
// filter matches the one applied before aggregation: cold starts and failed or
// non-exclusive invocations never reach the training dataset.
func TestBuildProfilingFeatureVectorsKeepsOnlyEligibleSamples(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			sampleForFeatureVector(
				"request-1",
				"alpha",
				"x86",
				1,
				256,
				true,
			),
			sampleForFeatureVector(
				"request-2",
				"alpha",
				"x86",
				1,
				256,
				false,
			),
			sampleForFeatureVector(
				"request-3",
				"beta",
				"x86",
				1,
				256,
				true,
			),
		}

	vectors, err :=
		BuildProfilingFeatureVectors(
			samples,
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		vectors,
		2,
	)

	assert.Equal(
		t,
		"request-1",
		vectors[0].RequestID,
	)

	assert.Equal(
		t,
		"request-3",
		vectors[1].RequestID,
	)
}

// TestBuildProfilingFeatureVectorsRejectsInconsistentEligibleSample covers the
// contract of the eligibility flag: ResourceClustering=true promises that the
// sample is convertible, so a failure is a dataset inconsistency, not a sample
// to skip silently.
func TestBuildProfilingFeatureVectorsRejectsInconsistentEligibleSample(
	t *testing.T,
) {
	sample :=
		sampleForFeatureVector(
			"request-broken",
			"alpha",
			"x86",
			1,
			256,
			true,
		)

	sample.NodeEnvironment = nil

	_, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sample,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"index 0",
	)
}

func TestExportProfilingFeatureVectorsCSVWritesOneRowPerInvocation(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			sampleForFeatureVector(
				"request-1",
				"alpha",
				"x86",
				1,
				256,
				true,
			),
			sampleForFeatureVector(
				"request-2",
				"alpha",
				"x86",
				1,
				256,
				true,
			),
			sampleForFeatureVector(
				"request-3",
				"alpha",
				"x86",
				1,
				256,
				true,
			),
		}

	vectors, err :=
		BuildProfilingFeatureVectors(
			samples,
		)

	require.NoError(
		t,
		err,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"nested",
			"samples.csv",
		)

	require.NoError(
		t,
		ExportProfilingFeatureVectorsCSV(
			path,
			"experiment-a",
			vectors,
		),
	)

	records :=
		readCSV(
			t,
			path,
		)

	// One header row plus one row per eligible invocation: this is the whole
	// point of the dataset, since the aggregated CSV would have produced a
	// single row for these three samples.
	require.Len(
		t,
		records,
		4,
	)

	assert.Equal(
		t,
		ProfilingFeatureVectorCSVHeader(),
		records[0],
	)

	for _, record := range records[1:] {

		assert.Len(
			t,
			record,
			len(ProfilingFeatureVectorCSVHeader()),
		)

		assert.Equal(
			t,
			"experiment-a",
			record[1],
		)

		assert.Equal(
			t,
			"alpha",
			record[4],
		)
	}
}

// TestExportProfilingFeatureVectorsCSVOrdersDeterministically guarantees that
// two runs over the same samples produce byte-identical files, which is what
// makes a dataset checksum meaningful.
func TestExportProfilingFeatureVectorsCSVOrdersDeterministically(
	t *testing.T,
) {
	vectors, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sampleForFeatureVector(
					"request-c",
					"beta",
					"x86",
					1,
					256,
					true,
				),
				sampleForFeatureVector(
					"request-a",
					"alpha",
					"x86",
					2,
					512,
					true,
				),
				sampleForFeatureVector(
					"request-b",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
			},
		)

	require.NoError(
		t,
		err,
	)

	directory :=
		t.TempDir()

	first :=
		filepath.Join(
			directory,
			"first.csv",
		)

	second :=
		filepath.Join(
			directory,
			"second.csv",
		)

	require.NoError(
		t,
		ExportProfilingFeatureVectorsCSV(
			first,
			"experiment-a",
			vectors,
		),
	)

	// Reversed input order must not change the output.
	reversed :=
		make(
			[]ProfilingFeatureVector,
			0,
			len(vectors),
		)

	for index := len(vectors) - 1; index >= 0; index-- {
		reversed =
			append(
				reversed,
				vectors[index],
			)
	}

	require.NoError(
		t,
		ExportProfilingFeatureVectorsCSV(
			second,
			"experiment-a",
			reversed,
		),
	)

	firstContent, err :=
		os.ReadFile(
			first,
		)

	require.NoError(
		t,
		err,
	)

	secondContent, err :=
		os.ReadFile(
			second,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		string(
			firstContent,
		),
		string(
			secondContent,
		),
	)

	records :=
		readCSV(
			t,
			first,
		)

	// Sorted by function, then machine tag, then configuration, then request
	// id: alpha/1cpu/256MB precedes alpha/2cpu/512MB, which precedes beta.
	assert.Equal(
		t,
		[]string{
			"request-b",
			"request-a",
			"request-c",
		},
		[]string{
			records[1][3],
			records[2][3],
			records[3][3],
		},
	)
}

func TestExportProfilingFeatureVectorsCSVRejectsDuplicateRequestID(
	t *testing.T,
) {
	vectors, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sampleForFeatureVector(
					"request-1",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
				sampleForFeatureVector(
					"request-1",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
			},
		)

	require.NoError(
		t,
		err,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"samples.csv",
		)

	err =
		ExportProfilingFeatureVectorsCSV(
			path,
			"experiment-a",
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"duplicate request id",
	)

	// The rejection must happen before any file is published.
	_, statErr :=
		os.Stat(
			path,
		)

	assert.True(
		t,
		os.IsNotExist(
			statErr,
		),
	)
}

func TestExportProfilingFeatureVectorsCSVRejectsInvalidInput(
	t *testing.T,
) {
	valid, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sampleForFeatureVector(
					"request-1",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
			},
		)

	require.NoError(
		t,
		err,
	)

	directory :=
		t.TempDir()

	testCases :=
		[]struct {
			name         string
			path         string
			experimentID string
			vectors      []ProfilingFeatureVector
			expected     string
		}{
			{
				name:         "empty path",
				path:         "  ",
				experimentID: "experiment-a",
				vectors:      valid,
				expected:     "output path cannot be empty",
			},
			{
				name: "empty experiment id",
				path: filepath.Join(
					directory,
					"a.csv",
				),
				experimentID: "  ",
				vectors:      valid,
				expected:     "experiment id cannot be empty",
			},
			{
				name: "no vectors",
				path: filepath.Join(
					directory,
					"b.csv",
				),
				experimentID: "experiment-a",
				vectors:      nil,
				expected:     "empty feature vector dataset",
			},
		}

	for _, testCase := range testCases {

		t.Run(
			testCase.name,
			func(
				t *testing.T,
			) {
				err :=
					ExportProfilingFeatureVectorsCSV(
						testCase.path,
						testCase.experimentID,
						testCase.vectors,
					)

				require.Error(
					t,
					err,
				)

				assert.Contains(
					t,
					err.Error(),
					testCase.expected,
				)
			},
		)
	}
}

func TestExportProfilingFeatureVectorsCSVRejectsUnsupportedSchemaVersion(
	t *testing.T,
) {
	vectors, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sampleForFeatureVector(
					"request-1",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
			},
		)

	require.NoError(
		t,
		err,
	)

	vectors[0].SchemaVersion =
		ProfilingFeatureVectorSchemaVersion + 1

	err =
		ExportProfilingFeatureVectorsCSV(
			filepath.Join(
				t.TempDir(),
				"samples.csv",
			),
			"experiment-a",
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"unsupported feature vector schema version",
	)
}

// TestExportProfilingFeatureVectorsCSVRewritesInsteadOfAppending mirrors the
// guarantee already covered for the aggregated dataset: re-running the export
// replaces the file atomically and never leaves a temporary behind.
func TestExportProfilingFeatureVectorsCSVRewritesInsteadOfAppending(
	t *testing.T,
) {
	vectors, err :=
		BuildProfilingFeatureVectors(
			[]InvocationSample{
				sampleForFeatureVector(
					"request-1",
					"alpha",
					"x86",
					1,
					256,
					true,
				),
			},
		)

	require.NoError(
		t,
		err,
	)

	directory :=
		t.TempDir()

	path :=
		filepath.Join(
			directory,
			"samples.csv",
		)

	for range 2 {
		require.NoError(
			t,
			ExportProfilingFeatureVectorsCSV(
				path,
				"experiment-a",
				vectors,
			),
		)
	}

	records :=
		readCSV(
			t,
			path,
		)

	assert.Len(
		t,
		records,
		2,
	)

	entries, err :=
		os.ReadDir(
			directory,
		)

	require.NoError(
		t,
		err,
	)

	for _, entry := range entries {

		assert.False(
			t,
			strings.HasSuffix(
				entry.Name(),
				".tmp",
			),
			"temporary file left behind: %s",
			entry.Name(),
		)
	}
}
