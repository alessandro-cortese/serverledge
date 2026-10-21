package profiling

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func sampleForPaperRF(
	requestID string,
	functionName string,
	machineTag string,
) InvocationSample {
	return InvocationSample{
		SchemaVersion:      PaperRFInvocationSampleSchemaVersion,
		RequestID:          requestID,
		FunctionName:       functionName,
		MachineTag:         machineTag,
		WarmStart:          true,
		ExecutionSucceeded: true,

		FunctionConfiguration: InvocationFunctionConfiguration{
			ConfiguredCPUs:     1,
			ConfiguredMemoryMB: 1024,
		},

		Timing: InvocationTiming{
			DurationMs:          100,
			ExecutionWallTimeMs: 100,
		},

		Profile: &InvocationResourceProfile{
			Enabled:                  true,
			Collected:                true,
			Valid:                    true,
			ExclusiveContainer:       true,
			ProfilingStartOverheadMs: 2.5,
			ProfilingEndOverheadMs:   3.5,
			ProfilingTotalOverheadMs: 6.0,
		},

		NodeEnvironment: &NodeResourceProfile{
			Collected:               true,
			Complete:                true,
			CPUAvailable:            true,
			MemoryAvailable:         true,
			VMStatAvailable:         true,
			AvailableCPUs:           4,
			ExecutionWallTimeMs:     100,
			CPUClockTicksAvailable:  true,
			CPUClockTicksPerSecond:  100,
			CPUUserDeltaTicks:       6,
			CPUKernelDeltaTicks:     2,
			CPUIdleDeltaTicks:       30,
			CPUIOWaitDeltaTicks:     1,
			CPUSoftIRQDeltaTicks:    2,
			CPUStealDeltaTicks:      1,
			TotalMemoryBeforeBytes:  8 * 1024 * 1024 * 1024,
			FreeMemoryBeforeBytes:   2 * 1024 * 1024 * 1024,
			PageFaultsDelta:         12,
			SnapshotStartOverheadMs: 1.5,
			SnapshotEndOverheadMs:   1.0,
			SnapshotTotalOverheadMs: 2.5,
		},
	}
}

func TestBuildPaperRFFeatureVectorMapsFourteenFeatures(
	t *testing.T,
) {
	vector, err :=
		BuildPaperRFFeatureVector(
			sampleForPaperRF(
				"request-1",
				"alpha",
				"amd64",
			),
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		PaperRFFeatureVectorSchemaVersion,
		vector.SchemaVersion,
	)

	assert.Equal(
		t,
		[]string{
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
		PaperRFFeatureNames(),
	)

	assert.Equal(
		t,
		14,
		len(vector.Features.Values()),
	)

	assert.InDelta(
		t,
		8*1024*1024.0,
		vector.Features.TotalMemory,
		1e-9,
	)

	assert.InDelta(
		t,
		2*1024*1024.0,
		vector.Features.FreeMemory,
		1e-9,
	)

	assert.InDelta(
		t,
		4.0,
		vector.Features.FrameworkRuntime,
		1e-9,
	)

	assert.InDelta(
		t,
		100.0,
		vector.Features.UserRuntime,
		1e-9,
	)

	assert.InDelta(
		t,
		60.0,
		vector.Features.CPUUserDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		20.0,
		vector.Features.CPUKernelDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		300.0,
		vector.Features.CPUIdleDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		10.0,
		vector.Features.CPUIOWaitDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		20.0,
		vector.Features.CPUSoftIRQDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		10.0,
		vector.Features.CPUStealDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		12.0,
		vector.Features.PageFaultsDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		0.8,
		vector.Features.UtilizedCPUs,
		1e-9,
	)

	assert.InDelta(
		t,
		4.5,
		vector.Features.FrameworkRuntimeDeltas,
		1e-9,
	)

	assert.InDelta(
		t,
		108.5,
		vector.Features.Runtime,
		1e-9,
	)
}

func TestBuildPaperRFFeatureVectorRoundsUtilizedCPUsLikeSAAF(
	t *testing.T,
) {
	sample :=
		sampleForPaperRF(
			"request-round",
			"alpha",
			"amd64",
		)

	sample.Timing.DurationMs = 123
	sample.NodeEnvironment.CPUUserDeltaTicks = 5
	sample.NodeEnvironment.CPUKernelDeltaTicks = 1

	vector, err :=
		BuildPaperRFFeatureVector(
			sample,
		)

	require.NoError(
		t,
		err,
	)

	assert.InDelta(
		t,
		0.488,
		vector.Features.UtilizedCPUs,
		1e-12,
	)
}

func TestBuildPaperRFFeatureVectorUsesRawProcStatTicksInsteadOfNormalizedCPUFields(
	t *testing.T,
) {
	sample :=
		sampleForPaperRF(
			"request-raw-ticks",
			"alpha",
			"amd64",
		)

	// Deliberately make the historical normalized-ms fields disagree with the
	// raw jiffy representation. The paper adapter must follow the latter.
	sample.NodeEnvironment.CPUUserDeltaMs = 999
	sample.NodeEnvironment.CPUKernelDeltaMs = 999
	sample.NodeEnvironment.CPUUserDeltaTicks = 6
	sample.NodeEnvironment.CPUKernelDeltaTicks = 2
	sample.NodeEnvironment.CPUClockTicksPerSecond = 100

	vector, err :=
		BuildPaperRFFeatureVector(
			sample,
		)

	require.NoError(
		t,
		err,
	)

	assert.InDelta(
		t,
		60.0,
		vector.Features.CPUUserDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		20.0,
		vector.Features.CPUKernelDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		0.8,
		vector.Features.UtilizedCPUs,
		1e-9,
	)
}

func TestBuildPaperRFFeatureVectorsSkipsUnusableMeasurements(
	t *testing.T,
) {
	valid :=
		sampleForPaperRF(
			"request-valid",
			"alpha",
			"amd64",
		)

	cold :=
		sampleForPaperRF(
			"request-cold",
			"alpha",
			"amd64",
		)

	cold.WarmStart = false

	failed :=
		sampleForPaperRF(
			"request-failed",
			"alpha",
			"amd64",
		)

	failed.ExecutionSucceeded = false

	incompleteNode :=
		sampleForPaperRF(
			"request-node",
			"alpha",
			"amd64",
		)

	incompleteNode.NodeEnvironment.VMStatAvailable = false

	vectors, err :=
		BuildPaperRFFeatureVectors(
			[]InvocationSample{
				valid,
				cold,
				failed,
				incompleteNode,
			},
		)

	require.NoError(
		t,
		err,
	)

	require.Len(
		t,
		vectors,
		1,
	)

	assert.Equal(
		t,
		"request-valid",
		vectors[0].RequestID,
	)
}

func TestBuildPaperRFFeatureVectorsRejectsOlderRawSchema(
	t *testing.T,
) {
	sample :=
		sampleForPaperRF(
			"request-old",
			"alpha",
			"amd64",
		)

	sample.SchemaVersion =
		PaperRFInvocationSampleSchemaVersion - 1

	_, err :=
		BuildPaperRFFeatureVectors(
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
		"requires schema 6",
	)
}
