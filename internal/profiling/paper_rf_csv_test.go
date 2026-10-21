package profiling

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPaperRFFeatureVectorCSVHeaderIsStable(
	t *testing.T,
) {
	assert.Equal(
		t,
		[]string{
			"paper_rf_csv_schema_version",
			"experiment_id",
			"paper_rf_feature_vector_schema_version",
			"request_id",
			"function_name",
			"machine_tag",
			"configured_cpus",
			"configured_memory_mb",
			"totalMemory",
			"freeMemory",
			"frameworkRuntime",
			"userRuntime",
			"cpuUserDelta",
			"cpuKernelDelta",
			"cpuIdleDelta",
			"cpuIOWaitDelta",
			"cpuSoftIrqDelta",
			"cpuStealDelta",
			"pageFaultsDelta",
			"utilizedCPUs",
			"frameworkRuntimeDeltas",
			"runtime",
		},
		PaperRFFeatureVectorCSVHeader(),
	)

	header :=
		PaperRFFeatureVectorCSVHeader()

	assert.Equal(
		t,
		PaperRFFeatureNames(),
		header[len(PaperRFFeatureVectorCSVMetadataNames()):],
	)
}

func TestExportPaperRFFeatureVectorsCSVWritesOneRowPerInvocation(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			sampleForPaperRF(
				"request-2",
				"alpha",
				"amd64",
			),
			sampleForPaperRF(
				"request-1",
				"alpha",
				"amd64",
			),
		}

	vectors, err :=
		BuildPaperRFFeatureVectors(
			samples,
		)

	require.NoError(
		t,
		err,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"paper-rf.csv",
		)

	require.NoError(
		t,
		ExportPaperRFFeatureVectorsCSV(
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

	require.Len(
		t,
		records,
		3,
	)

	assert.Equal(
		t,
		PaperRFFeatureVectorCSVHeader(),
		records[0],
	)

	assert.Equal(
		t,
		"request-1",
		records[1][3],
	)

	assert.Equal(
		t,
		"request-2",
		records[2][3],
	)

	for _, record := range records[1:] {
		assert.Len(
			t,
			record,
			len(PaperRFFeatureVectorCSVHeader()),
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

func TestExportPaperRFFeatureVectorsCSVRejectsDuplicateRequestID(
	t *testing.T,
) {
	first :=
		sampleForPaperRF(
			"request-1",
			"alpha",
			"amd64",
		)

	second :=
		sampleForPaperRF(
			"request-1",
			"beta",
			"amd64",
		)

	vectors, err :=
		BuildPaperRFFeatureVectors(
			[]InvocationSample{
				first,
				second,
			},
		)

	require.NoError(
		t,
		err,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"paper-rf.csv",
		)

	err =
		ExportPaperRFFeatureVectorsCSV(
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
