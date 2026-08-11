package profiling

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadInvocationSamplesJSONL(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"samples.jsonl",
		)

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

	for index := 0; index < 2; index++ {

		sample :=
			testInvocationSampleForAggregation(
				index,
			)

		sample.TimestampMs =
			int64(
				100 +
					index,
			)

		require.NoError(
			t,
			encoder.Encode(
				sample,
			),
		)
	}

	require.NoError(
		t,
		file.Close(),
	)

	samples, err :=
		LoadInvocationSamplesJSONL(
			path,
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		samples,
		2,
	)

	assert.Equal(
		t,
		"eligible-00",
		samples[0].
			RequestID,
	)

	assert.Equal(
		t,
		"eligible-01",
		samples[1].
			RequestID,
	)
}

func TestLoadInvocationSamplesJSONLRejectsMalformedLine(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"samples.jsonl",
		)

	require.NoError(
		t,
		os.WriteFile(
			path,
			[]byte(
				"{not-json}\n",
			),
			0o600,
		),
	)

	_, err :=
		LoadInvocationSamplesJSONL(
			path,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"line 1",
	)
}

func TestLoadInvocationSamplesJSONLRejectsUnsupportedSchema(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"samples.jsonl",
		)

	sample :=
		testInvocationSampleForAggregation(
			0,
		)

	sample.SchemaVersion =
		InvocationSampleSchemaVersion -
			1

	payload, err :=
		json.Marshal(
			sample,
		)

	require.NoError(
		t,
		err,
	)

	payload =
		append(
			payload,
			'\n',
		)

	require.NoError(
		t,
		os.WriteFile(
			path,
			payload,
			0o600,
		),
	)

	_, err =
		LoadInvocationSamplesJSONL(
			path,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"unsupported invocation sample schema version",
	)
}

func TestBuildFunctionProfilesByGroupBuildsCompleteGroupsAndSkipsIncomplete(
	t *testing.T,
) {
	samples :=
		make(
			[]InvocationSample,
			0,
			26,
		)

	// 15 x86 samples.
	//
	// With samplesPerProfile=10, only the ten most recent must be selected.
	for index := 0; index < 15; index++ {

		sample :=
			testInvocationSampleForAggregation(
				index,
			)

		sample.TimestampMs =
			int64(
				1000 +
					index,
			)

		samples =
			append(
				samples,
				sample,
			)
	}

	// 9 ARM samples: incomplete group.
	for index := 0; index < 9; index++ {

		sample :=
			testInvocationSampleForAggregation(
				100 +
					index,
			)

		sample.MachineTag =
			"arm64"

		sample.TimestampMs =
			int64(
				2000 +
					index,
			)

		samples =
			append(
				samples,
				sample,
			)
	}

	// Two raw samples not eligible for clustering.
	samples =
		append(
			samples,
			InvocationSample{
				RequestID: "cold",
			},
			InvocationSample{
				RequestID: "failed",
			},
		)

	result, err :=
		BuildFunctionProfilesByGroup(
			samples,
			10,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		26,
		result.RawSampleCount,
	)

	assert.Equal(
		t,
		24,
		result.EligibleSamples,
	)

	assert.Equal(
		t,
		2,
		result.IgnoredSamples,
	)

	require.Len(
		t,
		result.Groups,
		2,
	)

	require.Len(
		t,
		result.Profiles,
		1,
	)

	profile :=
		result.Profiles[0]

	assert.Equal(
		t,
		"x86",
		profile.MachineTag,
	)

	assert.Equal(
		t,
		10,
		profile.SampleCount,
	)

	require.Len(
		t,
		profile.SourceRequestIDs,
		10,
	)

	// Samples 0..4 are the oldest and must not be selected.
	assert.NotContains(
		t,
		profile.SourceRequestIDs,
		"eligible-00",
	)

	assert.NotContains(
		t,
		profile.SourceRequestIDs,
		"eligible-04",
	)

	assert.Contains(
		t,
		profile.SourceRequestIDs,
		"eligible-05",
	)

	assert.Contains(
		t,
		profile.SourceRequestIDs,
		"eligible-14",
	)

	var armStatus *FunctionProfileGroupStatus

	for index := range result.Groups {

		if result.Groups[index].
			MachineTag ==
			"arm64" {

			armStatus =
				&result.Groups[index]
		}
	}

	require.NotNil(
		t,
		armStatus,
	)

	assert.False(
		t,
		armStatus.Built,
	)

	assert.Equal(
		t,
		9,
		armStatus.
			EligibleSampleCount,
	)

	assert.Zero(
		t,
		armStatus.
			SelectedSampleCount,
	)
}

func TestBuildFunctionProfilesByGroupSeparatesConfigurations(
	t *testing.T,
) {
	samples :=
		make(
			[]InvocationSample,
			0,
			20,
		)

	for index := 0; index < 10; index++ {

		sample :=
			testInvocationSampleForAggregation(
				index,
			)

		sample.TimestampMs =
			int64(
				index +
					1,
			)

		samples =
			append(
				samples,
				sample,
			)
	}

	for index := 0; index < 10; index++ {

		sample :=
			testInvocationSampleForAggregation(
				20 +
					index,
			)

		sample.
			FunctionConfiguration.
			ConfiguredCPUs =
			2.0

		sample.TimestampMs =
			int64(
				100 +
					index,
			)

		samples =
			append(
				samples,
				sample,
			)
	}

	result, err :=
		BuildFunctionProfilesByGroup(
			samples,
			10,
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		result.Profiles,
		2,
	)

	assert.NotEqual(
		t,
		result.Profiles[0].
			FunctionConfiguration.
			ConfiguredCPUs,

		result.Profiles[1].
			FunctionConfiguration.
			ConfiguredCPUs,
	)
}

func TestBuildFunctionProfilesByGroupRejectsInvalidRequestedSampleCount(
	t *testing.T,
) {
	for _, count := range []int{
		9,
		21,
	} {

		t.Run(
			fmt.Sprintf(
				"count-%d",
				count,
			),
			func(
				t *testing.T,
			) {
				_, err :=
					BuildFunctionProfilesByGroup(
						nil,
						count,
					)

				require.Error(
					t,
					err,
				)

				assert.Contains(
					t,
					err.Error(),
					"samples per profile must be between",
				)
			},
		)
	}
}

func TestExportFunctionProfilesJSONLRewritesDerivedDataset(
	t *testing.T,
) {
	path :=
		filepath.Join(
			t.TempDir(),
			"nested",
			"function-profiles.jsonl",
		)

	profiles :=
		[]FunctionProfile{
			{
				SchemaVersion: FunctionProfileSchemaVersion,

				FunctionName: "first",

				MachineTag: "x86",

				SampleCount: 10,
			},
			{
				SchemaVersion: FunctionProfileSchemaVersion,

				FunctionName: "second",

				MachineTag: "arm64",

				SampleCount: 10,
			},
		}

	require.NoError(
		t,
		ExportFunctionProfilesJSONL(
			path,
			profiles,
		),
	)

	file, err :=
		os.Open(
			path,
		)

	require.NoError(
		t,
		err,
	)

	defer file.Close()

	scanner :=
		bufio.NewScanner(
			file,
		)

	var decoded []FunctionProfile

	for scanner.Scan() {
		var profile FunctionProfile

		require.NoError(
			t,
			json.Unmarshal(
				scanner.Bytes(),
				&profile,
			),
		)

		decoded =
			append(
				decoded,
				profile,
			)
	}

	require.NoError(
		t,
		scanner.Err(),
	)

	require.Len(
		t,
		decoded,
		2,
	)

	assert.Equal(
		t,
		"first",
		decoded[0].
			FunctionName,
	)

	assert.Equal(
		t,
		"second",
		decoded[1].
			FunctionName,
	)

	// A second export must replace the derived dataset instead of appending
	// duplicate aggregate profiles.
	require.NoError(
		t,
		ExportFunctionProfilesJSONL(
			path,
			profiles[:1],
		),
	)

	content, err :=
		os.ReadFile(
			path,
		)

	require.NoError(
		t,
		err,
	)

	lines := 0

	for _, line := range strings.Split(
		string(
			content,
		),
		"\n",
	) {

		if strings.TrimSpace(
			line,
		) != "" {

			lines++
		}
	}

	assert.Equal(
		t,
		1,
		lines,
	)
}

func TestDiscoverInvocationSampleDatasetsFindsPerNodeFiles(
	t *testing.T,
) {
	root := t.TempDir()

	nodeA := filepath.Join(
		root,
		"x86-node-01",
	)

	nodeB := filepath.Join(
		root,
		"arm-node-01",
	)

	require.NoError(
		t,
		os.MkdirAll(
			nodeA,
			0o755,
		),
	)

	require.NoError(
		t,
		os.MkdirAll(
			nodeB,
			0o755,
		),
	)

	for _, directory := range []string{
		nodeA,
		nodeB,
	} {
		require.NoError(
			t,
			os.WriteFile(
				filepath.Join(
					directory,
					filepath.Base(
						DefaultInvocationSampleExportPath,
					),
				),
				[]byte{},
				0o600,
			),
		)
	}

	// An unrelated JSONL file must not be interpreted as a raw
	// InvocationSample dataset.
	require.NoError(
		t,
		os.WriteFile(
			filepath.Join(
				root,
				"other.jsonl",
			),
			[]byte{},
			0o600,
		),
	)

	paths, err :=
		DiscoverInvocationSampleDatasets(
			root,
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		paths,
		2,
	)

	combined :=
		paths[0] +
			paths[1]

	assert.Contains(
		t,
		combined,
		"arm-node-01",
	)

	assert.Contains(
		t,
		combined,
		"x86-node-01",
	)
}

func TestLoadInvocationSamplesJSONLFilesMergesMultipleNodes(
	t *testing.T,
) {
	root := t.TempDir()

	x86Path := filepath.Join(
		root,
		"x86",
		"profiling-samples.jsonl",
	)

	armPath := filepath.Join(
		root,
		"arm",
		"profiling-samples.jsonl",
	)

	writeInvocationDatasetForTest(
		t,
		x86Path,
		buildNodeSamplesForTest(
			0,
			10,
			"x86",
		),
	)

	writeInvocationDatasetForTest(
		t,
		armPath,
		buildNodeSamplesForTest(
			100,
			10,
			"arm64",
		),
	)

	samples, err :=
		LoadInvocationSamplesJSONLFiles(
			[]string{
				x86Path,
				armPath,
			},
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		samples,
		20,
	)

	result, err :=
		BuildFunctionProfilesByGroup(
			samples,
			10,
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		result.Profiles,
		2,
	)

	machineTags :=
		[]string{
			result.Profiles[0].MachineTag,
			result.Profiles[1].MachineTag,
		}

	assert.ElementsMatch(
		t,
		[]string{
			"x86",
			"arm64",
		},
		machineTags,
	)
}

func TestLoadInvocationSamplesJSONLFilesRejectsDuplicateRequestIDAcrossNodes(
	t *testing.T,
) {
	root := t.TempDir()

	firstPath := filepath.Join(
		root,
		"node-a",
		"profiling-samples.jsonl",
	)

	secondPath := filepath.Join(
		root,
		"node-b",
		"profiling-samples.jsonl",
	)

	first :=
		testInvocationSampleForAggregation(
			0,
		)

	second :=
		testInvocationSampleForAggregation(
			1,
		)

	second.RequestID =
		first.RequestID

	second.MachineTag =
		"arm64"

	writeInvocationDatasetForTest(
		t,
		firstPath,
		[]InvocationSample{
			first,
		},
	)

	writeInvocationDatasetForTest(
		t,
		secondPath,
		[]InvocationSample{
			second,
		},
	)

	_, err :=
		LoadInvocationSamplesJSONLFiles(
			[]string{
				firstPath,
				secondPath,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"duplicate request ID",
	)
}

func buildNodeSamplesForTest(
	start int,
	count int,
	machineTag string,
) []InvocationSample {
	samples := make(
		[]InvocationSample,
		0,
		count,
	)

	for index := 0; index < count; index++ {
		sample :=
			testInvocationSampleForAggregation(
				start +
					index,
			)

		sample.MachineTag =
			machineTag

		sample.TimestampMs =
			int64(
				start +
					index +
					1,
			)

		samples = append(
			samples,
			sample,
		)
	}

	return samples
}

func writeInvocationDatasetForTest(
	t *testing.T,
	path string,
	samples []InvocationSample,
) {
	t.Helper()

	require.NoError(
		t,
		os.MkdirAll(
			filepath.Dir(
				path,
			),
			0o755,
		),
	)

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

	for _, sample := range samples {
		require.NoError(
			t,
			encoder.Encode(
				sample,
			),
		)
	}

	require.NoError(
		t,
		file.Close(),
	)
}
