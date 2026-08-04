package profiling

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildInvocationSampleMarksWarmValidProfileForClustering(
	t *testing.T,
) {
	profile :=
		&InvocationResourceProfile{
			Enabled:            true,
			Collected:          true,
			Valid:              true,
			ExclusiveContainer: true,
			ContainerID:        "container-a",
		}

	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				Timestamp: time.UnixMilli(
					1234,
				),

				RequestID: "request-a",

				FunctionName: "fibonacci",

				MachineTag: "x86",

				NodeName: "node-a",

				WarmStart: true,

				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					DurationMs: 10,

					ResponseTimeMs: 12,

					InitTimeMs: 0,
				},

				Profile: profile,
			},
		)

	assert.Equal(
		t,
		InvocationSampleSchemaVersion,
		sample.SchemaVersion,
	)

	assert.Equal(
		t,
		int64(1234),
		sample.TimestampMs,
	)

	assert.Equal(
		t,
		"container-a",
		sample.ContainerID,
	)

	assert.True(
		t,
		sample.Eligibility.
			ResourceClustering,
	)

	assert.False(
		t,
		sample.Eligibility.
			ColdStartAnalysis,
	)

	assert.True(
		t,
		sample.Eligibility.
			PerformanceAnalysis,
	)

	assert.Empty(
		t,
		sample.Eligibility.
			ExclusionReasons,
	)
}

func TestBuildInvocationSamplePreservesColdInitTimeForAnalysis(
	t *testing.T,
) {
	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				RequestID: "request-cold",

				FunctionName: "fibonacci",

				MachineTag: "arm",

				NodeName: "node-arm",

				ContainerID: "container-cold",

				WarmStart: false,

				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					InitTimeMs: 325,
				},

				Profile: nil,
			},
		)

	assert.InDelta(
		t,
		325.0,
		sample.Timing.InitTimeMs,
		1e-9,
	)

	assert.Equal(
		t,
		"container-cold",
		sample.ContainerID,
	)

	assert.Nil(
		t,
		sample.Profile,
	)

	assert.False(
		t,
		sample.Eligibility.
			ResourceClustering,
	)

	assert.True(
		t,
		sample.Eligibility.
			ColdStartAnalysis,
	)

	assert.False(
		t,
		sample.Eligibility.
			PerformanceAnalysis,
	)

	assert.Contains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"cold_start",
	)

	assert.NotContains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"profile_missing",
	)
}

func TestBuildInvocationSampleExplainsUnusableResourceProfile(
	t *testing.T,
) {
	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				RequestID: "request-failed",

				WarmStart: true,

				ExecutionSucceeded: false,

				ExecutionError: "boom",

				Timing: InvocationTiming{
					DurationMs: 10,

					ResponseTimeMs: 11,
				},

				Profile: &InvocationResourceProfile{
					Enabled: true,

					Collected: false,

					Valid: false,

					ExclusiveContainer: true,
				},
			},
		)

	require.False(
		t,
		sample.Eligibility.
			ResourceClustering,
	)

	assert.False(
		t,
		sample.Eligibility.
			ColdStartAnalysis,
	)

	assert.False(
		t,
		sample.Eligibility.
			PerformanceAnalysis,
	)

	assert.Contains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"execution_failed",
	)

	assert.Contains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"profile_not_collected",
	)

	assert.Contains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"profile_invalid",
	)
}

func TestBuildInvocationSampleWarmWithoutProfileIsNotEligibleForClustering(
	t *testing.T,
) {
	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				RequestID: "request-warm-no-profile",

				FunctionName: "fibonacci",

				MachineTag: "x86",

				NodeName: "node-x86",

				ContainerID: "container-warm",

				WarmStart: true,

				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					DurationMs: 10,

					ResponseTimeMs: 12,
				},

				Profile: nil,
			},
		)

	assert.False(
		t,
		sample.Eligibility.
			ResourceClustering,
	)

	assert.False(
		t,
		sample.Eligibility.
			ColdStartAnalysis,
	)

	assert.True(
		t,
		sample.Eligibility.
			PerformanceAnalysis,
	)

	assert.Contains(
		t,
		sample.Eligibility.
			ExclusionReasons,
		"profile_missing",
	)
}
