package profiling

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildProfilingFeatureVectorExtractsRandomForestFeatures(
	t *testing.T,
) {
	sample :=
		InvocationSample{
			RequestID:    "request-rf",
			FunctionName: "fibonacci",
			MachineTag:   "x86",

			FunctionConfiguration: InvocationFunctionConfiguration{
				ConfiguredCPUs:     1.5,
				ConfiguredMemoryMB: 512,
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
				ResourceClustering: true,
			},
		}

	vector, err :=
		BuildProfilingFeatureVector(
			sample,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		ProfilingFeatureVectorSchemaVersion,
		vector.SchemaVersion,
	)

	assert.Equal(
		t,
		"request-rf",
		vector.RequestID,
	)

	assert.Equal(
		t,
		"fibonacci",
		vector.FunctionName,
	)

	assert.Equal(
		t,
		"x86",
		vector.MachineTag,
	)

	assert.InDelta(
		t,
		1.5,
		vector.FunctionConfiguration.
			ConfiguredCPUs,
		1e-9,
	)

	assert.Equal(
		t,
		int64(512),
		vector.FunctionConfiguration.
			ConfiguredMemoryMB,
	)

	assert.InDelta(
		t,
		12.0,
		vector.Features.
			PageFaultsDelta,
		1e-9,
	)

	// (60 ms user + 20 ms kernel) / 100 ms = 0.8 CPU.
	assert.InDelta(
		t,
		0.8,
		vector.Features.
			UtilizedCPUs,
		1e-9,
	)

	assert.InDelta(
		t,
		2048.0,
		vector.Features.
			FreeMemoryMB,
		1e-9,
	)

	assert.InDelta(
		t,
		60.0,
		vector.Features.
			CPUUserDeltaMs,
		1e-9,
	)

	assert.InDelta(
		t,
		20.0,
		vector.Features.
			CPUKernelDeltaMs,
		1e-9,
	)

	// 2.5 ms container snapshot + 1.5 ms node snapshot.
	assert.InDelta(
		t,
		4.0,
		vector.Features.
			FrameworkRuntimeMs,
		1e-9,
	)

	assert.Equal(
		t,
		[]string{
			"page_faults_delta",
			"utilized_cpus",
			"free_memory_mb",
			"cpu_user_delta_ms",
			"cpu_kernel_delta_ms",
			"framework_runtime_ms",
		},
		RandomForestFeatureNames(),
	)

	assert.Equal(
		t,
		[]float64{
			12,
			0.8,
			2048,
			60,
			20,
			4,
		},
		vector.Features.Values(),
	)
}

func TestBuildProfilingFeatureVectorRejectsIneligibleSample(
	t *testing.T,
) {
	_, err :=
		BuildProfilingFeatureVector(
			InvocationSample{
				RequestID: "cold-or-failed",
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"not eligible",
	)
}

func TestBuildProfilingFeatureVectorRequiresContainerPageFaults(
	t *testing.T,
) {
	sample :=
		validFeatureVectorSampleForTest()

	sample.Profile.PageFaultsAvailable =
		false

	_, err :=
		BuildProfilingFeatureVector(
			sample,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"page-fault",
	)
}

func TestBuildProfilingFeatureVectorRequiresNodeMemory(
	t *testing.T,
) {
	sample :=
		validFeatureVectorSampleForTest()

	sample.NodeEnvironment.MemoryAvailable =
		false

	_, err :=
		BuildProfilingFeatureVector(
			sample,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"node memory",
	)
}

func TestBuildProfilingFeatureVectorRequiresPositiveDuration(
	t *testing.T,
) {
	sample :=
		validFeatureVectorSampleForTest()

	sample.Timing.DurationMs =
		0

	_, err :=
		BuildProfilingFeatureVector(
			sample,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"duration_ms",
	)
}

func validFeatureVectorSampleForTest() InvocationSample {
	return InvocationSample{
		RequestID: "request-valid",

		Timing: InvocationTiming{
			DurationMs: 10,
		},

		Profile: &InvocationResourceProfile{
			PageFaultsAvailable: true,
		},

		NodeEnvironment: &NodeResourceProfile{
			MemoryAvailable: true,
		},

		Eligibility: InvocationEligibility{
			ResourceClustering: true,
		},
	}
}
