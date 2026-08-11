package profiling

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAggregateProfilingFeatureVectorsCalculatesMeanAndMedian(
	t *testing.T,
) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			0,
			10,
		)

	for index := 0; index < 10; index++ {

		value :=
			float64(
				index + 1,
			)

		vectors =
			append(
				vectors,
				testProfilingFeatureVector(
					index,
					RandomForestProfilingFeatures{
						PageFaultsDelta: value,

						UtilizedCPUs: value /
							10.0,

						FreeMemoryMB: 100.0 +
							float64(index)*
								10.0,

						CPUUserDeltaMs: 10.0 +
							float64(index),

						CPUKernelDeltaMs: float64(index) *
							2.0,

						FrameworkRuntimeMs: 1.0 +
							float64(index)/
								10.0,
					},
				),
			)
	}

	profile, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		FunctionProfileSchemaVersion,
		profile.SchemaVersion,
	)

	assert.Equal(
		t,
		"test-function",
		profile.FunctionName,
	)

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

	assert.Len(
		t,
		profile.SourceRequestIDs,
		10,
	)

	assert.InDelta(
		t,
		1.0,
		profile.FunctionConfiguration.
			ConfiguredCPUs,
		1e-9,
	)

	assert.Equal(
		t,
		int64(128),
		profile.FunctionConfiguration.
			ConfiguredMemoryMB,
	)

	// Valori 1..10:
	// mean = median = 5.5.
	assert.InDelta(
		t,
		5.5,
		profile.Mean.
			PageFaultsDelta,
		1e-9,
	)

	assert.InDelta(
		t,
		5.5,
		profile.Median.
			PageFaultsDelta,
		1e-9,
	)

	// 0.1..1.0:
	// mean = median = 0.55.
	assert.InDelta(
		t,
		0.55,
		profile.Mean.
			UtilizedCPUs,
		1e-9,
	)

	assert.InDelta(
		t,
		0.55,
		profile.Median.
			UtilizedCPUs,
		1e-9,
	)

	// 100, 110, ..., 190:
	// mean = median = 145.
	assert.InDelta(
		t,
		145.0,
		profile.Mean.
			FreeMemoryMB,
		1e-9,
	)

	assert.InDelta(
		t,
		145.0,
		profile.Median.
			FreeMemoryMB,
		1e-9,
	)

	// 10..19:
	// mean = median = 14.5.
	assert.InDelta(
		t,
		14.5,
		profile.Mean.
			CPUUserDeltaMs,
		1e-9,
	)

	assert.InDelta(
		t,
		14.5,
		profile.Median.
			CPUUserDeltaMs,
		1e-9,
	)

	// 0, 2, ..., 18:
	// mean = median = 9.
	assert.InDelta(
		t,
		9.0,
		profile.Mean.
			CPUKernelDeltaMs,
		1e-9,
	)

	assert.InDelta(
		t,
		9.0,
		profile.Median.
			CPUKernelDeltaMs,
		1e-9,
	)

	// 1.0..1.9:
	// mean = median = 1.45.
	assert.InDelta(
		t,
		1.45,
		profile.Mean.
			FrameworkRuntimeMs,
		1e-9,
	)

	assert.InDelta(
		t,
		1.45,
		profile.Median.
			FrameworkRuntimeMs,
		1e-9,
	)

	assert.Equal(
		t,
		profile.Mean.Values(),
		profile.MeanValues(),
	)

	assert.Equal(
		t,
		profile.Median.Values(),
		profile.MedianValues(),
	)
}

func TestAggregateProfilingFeatureVectorsCalculatesOddMedian(
	t *testing.T,
) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			0,
			11,
		)

	for index := 0; index < 11; index++ {

		value :=
			float64(
				index + 1,
			)

		vectors =
			append(
				vectors,
				testProfilingFeatureVector(
					index,
					RandomForestProfilingFeatures{
						PageFaultsDelta: value,

						UtilizedCPUs: value,

						FreeMemoryMB: value,

						CPUUserDeltaMs: value,

						CPUKernelDeltaMs: value,

						FrameworkRuntimeMs: value,
					},
				),
			)
	}

	profile, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		11,
		profile.SampleCount,
	)

	for _, value := range profile.MedianValues() {

		assert.InDelta(
			t,
			6.0,
			value,
			1e-9,
		)
	}
}

func TestAggregateProfilingFeatureVectorsAcceptsTwentySamples(
	t *testing.T,
) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			0,
			MaxFunctionProfileSamples,
		)

	for index := 0; index <
		MaxFunctionProfileSamples; index++ {

		vectors =
			append(
				vectors,
				testProfilingFeatureVector(
					index,
					testRandomForestFeatures(
						float64(
							index+1,
						),
					),
				),
			)
	}

	profile, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		MaxFunctionProfileSamples,
		profile.SampleCount,
	)
}

func TestAggregateProfilingFeatureVectorsRejectsTooFewSamples(
	t *testing.T,
) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			MinFunctionProfileSamples-1,
		)

	for index := range vectors {

		vectors[index] =
			testProfilingFeatureVector(
				index,
				testRandomForestFeatures(
					float64(
						index+1,
					),
				),
			)
	}

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"not enough profiling samples",
	)
}

func TestAggregateProfilingFeatureVectorsRejectsTooManySamples(
	t *testing.T,
) {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			MaxFunctionProfileSamples+1,
		)

	for index := range vectors {

		vectors[index] =
			testProfilingFeatureVector(
				index,
				testRandomForestFeatures(
					float64(
						index+1,
					),
				),
			)
	}

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"too many profiling samples",
	)
}

func TestAggregateProfilingFeatureVectorsRejectsMixedFunction(
	t *testing.T,
) {
	vectors :=
		validProfilingFeatureVectorsForTest(
			10,
		)

	vectors[5].FunctionName =
		"another-function"

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"mixed function names",
	)
}

func TestAggregateProfilingFeatureVectorsRejectsMixedMachineTag(
	t *testing.T,
) {
	vectors :=
		validProfilingFeatureVectorsForTest(
			10,
		)

	vectors[5].MachineTag =
		"arm64"

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"mixed machine tags",
	)
}

func TestAggregateProfilingFeatureVectorsRejectsMixedConfiguration(
	t *testing.T,
) {
	vectors :=
		validProfilingFeatureVectorsForTest(
			10,
		)

	vectors[5].
		FunctionConfiguration.
		ConfiguredCPUs =
		2.0

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"mixed function configurations",
	)
}

func TestAggregateProfilingFeatureVectorsRejectsDuplicateRequestID(
	t *testing.T,
) {
	vectors :=
		validProfilingFeatureVectorsForTest(
			10,
		)

	vectors[9].RequestID =
		vectors[0].RequestID

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
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

func TestAggregateProfilingFeatureVectorsRejectsInvalidFeature(
	t *testing.T,
) {
	vectors :=
		validProfilingFeatureVectorsForTest(
			10,
		)

	vectors[3].
		Features.
		UtilizedCPUs =
		math.NaN()

	_, err :=
		AggregateProfilingFeatureVectors(
			vectors,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"invalid feature",
	)
}

func TestBuildFunctionProfileFromSamplesIgnoresIneligibleSamples(
	t *testing.T,
) {
	samples :=
		make(
			[]InvocationSample,
			0,
			12,
		)

	// Due campioni non eleggibili: non devono contribuire
	// al FunctionProfile.
	samples =
		append(
			samples,
			InvocationSample{
				RequestID: "cold-1",
			},
			InvocationSample{
				RequestID: "failed-1",
			},
		)

	for index := 0; index < 10; index++ {

		samples =
			append(
				samples,
				testInvocationSampleForAggregation(
					index,
				),
			)
	}

	profile, err :=
		BuildFunctionProfileFromSamples(
			samples,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		10,
		profile.SampleCount,
	)

	assert.Len(
		t,
		profile.SourceRequestIDs,
		10,
	)
}

func testProfilingFeatureVector(
	index int,
	features RandomForestProfilingFeatures,
) ProfilingFeatureVector {
	return ProfilingFeatureVector{
		SchemaVersion: ProfilingFeatureVectorSchemaVersion,

		RequestID: fmt.Sprintf(
			"request-%02d",
			index,
		),

		FunctionName: "test-function",

		MachineTag: "x86",

		FunctionConfiguration: InvocationFunctionConfiguration{
			ConfiguredCPUs: 1.0,

			ConfiguredMemoryMB: 128,
		},

		Features: features,
	}
}

func testRandomForestFeatures(
	value float64,
) RandomForestProfilingFeatures {
	return RandomForestProfilingFeatures{
		PageFaultsDelta: value,

		UtilizedCPUs: value,

		FreeMemoryMB: value,

		CPUUserDeltaMs: value,

		CPUKernelDeltaMs: value,

		FrameworkRuntimeMs: value,
	}
}

func validProfilingFeatureVectorsForTest(
	count int,
) []ProfilingFeatureVector {
	vectors :=
		make(
			[]ProfilingFeatureVector,
			0,
			count,
		)

	for index := 0; index < count; index++ {

		vectors =
			append(
				vectors,
				testProfilingFeatureVector(
					index,
					testRandomForestFeatures(
						float64(
							index+1,
						),
					),
				),
			)
	}

	return vectors
}

func testInvocationSampleForAggregation(
	index int,
) InvocationSample {
	return InvocationSample{
		SchemaVersion: InvocationSampleSchemaVersion,

		RequestID: fmt.Sprintf(
			"eligible-%02d",
			index,
		),

		FunctionName: "test-function",

		MachineTag: "x86",

		FunctionConfiguration: InvocationFunctionConfiguration{
			ConfiguredCPUs: 1.0,

			ConfiguredMemoryMB: 128,
		},

		WarmStart: true,

		ExecutionSucceeded: true,

		Timing: InvocationTiming{
			DurationMs: 100.0,
		},

		Profile: &InvocationResourceProfile{
			Enabled: true,

			Collected: true,

			Valid: true,

			ExclusiveContainer: true,

			PageFaultsAvailable: true,

			PageFaultsDelta: uint64(
				index + 1,
			),

			CPUUsageUserDeltaNs: uint64(
				50_000_000 +
					index*
						1_000_000,
			),

			CPUUsageKernelDeltaNs: 10_000_000,

			ProfilingStartOverheadMs: 2.0,
		},

		NodeEnvironment: &NodeResourceProfile{
			MemoryAvailable: true,

			FreeMemoryBeforeBytes: uint64(
				2048 *
					1024 *
					1024,
			),

			SnapshotStartOverheadMs: 1.0,
		},

		Eligibility: InvocationEligibility{
			ResourceClustering: true,

			PerformanceAnalysis: true,
		},
	}
}
