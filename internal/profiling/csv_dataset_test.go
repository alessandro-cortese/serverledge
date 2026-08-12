package profiling

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFunctionProfileCSVHeaderSeparatesMetadataAndFeatures(
	t *testing.T,
) {
	metadata :=
		FunctionProfileCSVMetadataNames()

	header :=
		FunctionProfileCSVHeader()

	featureNames :=
		RandomForestFeatureNames()

	require.Len(
		t,
		header,
		len(metadata)+
			len(featureNames),
	)

	assert.Equal(
		t,
		metadata,
		header[:len(metadata)],
	)

	assert.Equal(
		t,
		featureNames,
		header[len(metadata):],
	)
}

func TestLoadFunctionProfilesJSONL(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles.jsonl",
		)

	profiles :=
		[]FunctionProfile{
			validFunctionProfileForCSVTest(
				"beta",
				"x86",
				10,
			),

			validFunctionProfileForCSVTest(
				"alpha",
				"arm64",
				10,
			),
		}

	file, err :=
		os.Create(
			path,
		)

	require.NoError(
		t,
		err,
	)

	encoder :=
		json.NewEncoder(
			file,
		)

	for _, profile := range profiles {

		require.NoError(
			t,
			encoder.Encode(
				profile,
			),
		)
	}

	require.NoError(
		t,
		file.Close(),
	)

	loaded, err :=
		LoadFunctionProfilesJSONL(
			path,
		)

	require.NoError(
		t,
		err,
	)

	require.Equal(
		t,
		profiles,
		loaded,
	)
}

func TestLoadFunctionProfilesJSONLRejectsInvalidProfile(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles.jsonl",
		)

	profile :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	profile.SchemaVersion =
		FunctionProfileSchemaVersion +
			1

	payload, err :=
		json.Marshal(
			profile,
		)

	require.NoError(
		t,
		err,
	)

	require.NoError(
		t,
		os.WriteFile(
			path,
			append(
				payload,
				'\n',
			),
			0o600,
		),
	)

	_, err =
		LoadFunctionProfilesJSONL(
			path,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"unsupported function profile schema version",
	)
}

func TestLoadFunctionProfilesJSONLRejectsDuplicateIdentity(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles.jsonl",
		)

	first :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	second :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	for index := range second.SourceRequestIDs {

		second.SourceRequestIDs[index] =
			fmt.Sprintf(
				"second-request-%02d",
				index,
			)
	}

	file, err :=
		os.Create(
			path,
		)

	require.NoError(
		t,
		err,
	)

	encoder :=
		json.NewEncoder(
			file,
		)

	require.NoError(
		t,
		encoder.Encode(
			first,
		),
	)

	require.NoError(
		t,
		encoder.Encode(
			second,
		),
	)

	require.NoError(
		t,
		file.Close(),
	)

	_, err =
		LoadFunctionProfilesJSONL(
			path,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"duplicate function profile",
	)
}

func TestExportFunctionProfilesCSVWritesStableMeanDataset(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles-mean.csv",
		)

	beta :=
		validFunctionProfileForCSVTest(
			"beta",
			"x86",
			10,
		)

	beta.Mean =
		testRandomForestFeaturesForCSV(
			20,
		)

	alpha :=
		validFunctionProfileForCSVTest(
			"alpha",
			"arm64",
			10,
		)

	alpha.Mean =
		testRandomForestFeaturesForCSV(
			10,
		)

	require.NoError(
		t,
		ExportFunctionProfilesCSV(
			path,
			"experiment-001",
			FunctionProfileAggregationMean,
			[]FunctionProfile{
				beta,
				alpha,
			},
		),
	)

	records :=
		readCSVForTest(
			t,
			path,
		)

	require.Len(
		t,
		records,
		3,
	)

	assert.Equal(
		t,
		FunctionProfileCSVHeader(),
		records[0],
	)

	// The exporter sorts by function identity rather than preserving
	// arbitrary input order.
	assert.Equal(
		t,
		"alpha",
		records[1][4],
	)

	assert.Equal(
		t,
		"arm64",
		records[1][5],
	)

	assert.Equal(
		t,
		"mean",
		records[1][2],
	)

	assert.Equal(
		t,
		"experiment-001",
		records[1][1],
	)

	// First numeric feature: page_faults_delta.
	assert.Equal(
		t,
		"10",
		records[1][9],
	)

	// Second numeric feature: utilized_cpus.
	assert.Equal(
		t,
		"10.1",
		records[1][10],
	)

	assert.Equal(
		t,
		"beta",
		records[2][4],
	)

	assert.Equal(
		t,
		"20",
		records[2][9],
	)
}

func TestExportFunctionProfilesCSVWritesMedianDataset(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles-median.csv",
		)

	profile :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	profile.Median =
		testRandomForestFeaturesForCSV(
			30,
		)

	require.NoError(
		t,
		ExportFunctionProfilesCSV(
			path,
			"experiment-002",
			FunctionProfileAggregationMedian,
			[]FunctionProfile{
				profile,
			},
		),
	)

	records :=
		readCSVForTest(
			t,
			path,
		)

	require.Len(
		t,
		records,
		2,
	)

	assert.Equal(
		t,
		"median",
		records[1][2],
	)

	assert.Equal(
		t,
		"30",
		records[1][9],
	)

	assert.Equal(
		t,
		"30.1",
		records[1][10],
	)
}

func TestExportFunctionProfilesCSVRewritesInsteadOfAppending(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"function-profiles-mean.csv",
		)

	first :=
		validFunctionProfileForCSVTest(
			"first",
			"x86",
			10,
		)

	second :=
		validFunctionProfileForCSVTest(
			"second",
			"x86",
			10,
		)

	require.NoError(
		t,
		ExportFunctionProfilesCSV(
			path,
			"experiment-001",
			FunctionProfileAggregationMean,
			[]FunctionProfile{
				first,
				second,
			},
		),
	)

	require.NoError(
		t,
		ExportFunctionProfilesCSV(
			path,
			"experiment-001",
			FunctionProfileAggregationMean,
			[]FunctionProfile{
				first,
			},
		),
	)

	records :=
		readCSVForTest(
			t,
			path,
		)

	require.Len(
		t,
		records,
		2,
	)

	assert.Equal(
		t,
		"first",
		records[1][4],
	)
}

func TestExportFunctionProfilesCSVRejectsInvalidAggregation(
	t *testing.T,
) {
	profile :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	err :=
		ExportFunctionProfilesCSV(
			filepath.Join(
				t.TempDir(),
				"invalid.csv",
			),
			"experiment-001",
			FunctionProfileAggregation(
				"unknown",
			),
			[]FunctionProfile{
				profile,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"unsupported function profile aggregation",
	)
}

func TestExportFunctionProfilesCSVRejectsInvalidFeature(
	t *testing.T,
) {
	profile :=
		validFunctionProfileForCSVTest(
			"alpha",
			"x86",
			10,
		)

	profile.Mean.UtilizedCPUs =
		math.NaN()

	err :=
		ExportFunctionProfilesCSV(
			filepath.Join(
				t.TempDir(),
				"invalid.csv",
			),
			"experiment-001",
			FunctionProfileAggregationMean,
			[]FunctionProfile{
				profile,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"mean feature utilized_cpus is invalid",
	)
}

func validFunctionProfileForCSVTest(
	functionName string,
	machineTag string,
	sampleCount int,
) FunctionProfile {
	requestIDs :=
		make(
			[]string,
			0,
			sampleCount,
		)

	for index := 0; index < sampleCount; index++ {

		requestIDs =
			append(
				requestIDs,
				fmt.Sprintf(
					"%s-request-%02d",
					functionName,
					index,
				),
			)
	}

	return FunctionProfile{
		SchemaVersion: FunctionProfileSchemaVersion,

		FunctionName: functionName,

		MachineTag: machineTag,

		FunctionConfiguration: InvocationFunctionConfiguration{
			ConfiguredCPUs: 1,

			ConfiguredMemoryMB: 128,
		},

		SampleCount: sampleCount,

		SourceRequestIDs: requestIDs,

		Mean: testRandomForestFeaturesForCSV(
			1,
		),

		Median: testRandomForestFeaturesForCSV(
			2,
		),
	}
}

func testRandomForestFeaturesForCSV(
	base float64,
) RandomForestProfilingFeatures {
	return RandomForestProfilingFeatures{
		PageFaultsDelta: base,

		UtilizedCPUs: base +
			0.1,

		FreeMemoryMB: base +
			0.2,

		CPUUserDeltaMs: base +
			0.3,

		CPUKernelDeltaMs: base +
			0.4,

		FrameworkRuntimeMs: base +
			0.5,
	}
}

func readCSVForTest(
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
