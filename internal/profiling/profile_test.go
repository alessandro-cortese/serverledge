package profiling

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildInvocationResourceProfileCalculatesDeltas(
	t *testing.T,
) {
	before :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageUserNs: 700_000_000,

			CPUUsageKernelNs: 300_000_000,

			PageFaults: 100,

			PageFaultsAvailable: true,
		}

	after :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageUserNs: 900_000_000,

			CPUUsageKernelNs: 350_000_000,

			PageFaults: 140,

			PageFaultsAvailable: true,
		}

	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			true,
			before,
			after,
			2*time.Second,
			3*time.Millisecond,
		)

	require.NotNil(
		t,
		profile,
	)

	require.True(
		t,
		profile.Valid,
		profile.InvalidReason,
	)

	assert.True(
		t,
		profile.Collected,
	)

	assert.Equal(
		t,
		uint64(200_000_000),
		profile.CPUUsageUserDeltaNs,
	)

	assert.Equal(
		t,
		uint64(50_000_000),
		profile.CPUUsageKernelDeltaNs,
	)

	assert.Equal(
		t,
		uint64(40),
		profile.PageFaultsDelta,
	)

	assert.True(
		t,
		profile.PageFaultsAvailable,
	)

	assert.InDelta(
		t,
		3.0,
		profile.ProfilingStartOverheadMs,
		1e-9,
	)

	assert.Equal(
		t,
		"container-a",
		profile.ContainerID,
	)
}

// TestBuildInvocationResourceProfileWithoutPageFaultCounters documents that a
// missing page-fault counter does not invalidate the profile: it only makes the
// sample ineligible for the feature vector, which checks PageFaultsAvailable.
func TestBuildInvocationResourceProfileWithoutPageFaultCounters(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			true,
			ResourceSnapshot{
				OSType: "linux",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			time.Second,
			0,
		)

	require.True(
		t,
		profile.Valid,
		profile.InvalidReason,
	)

	assert.False(
		t,
		profile.PageFaultsAvailable,
	)

	assert.Equal(
		t,
		uint64(0),
		profile.PageFaultsDelta,
	)
}

func TestBuildInvocationResourceProfileMarksNonExclusiveCollectionInvalid(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			false,
			ResourceSnapshot{
				OSType: "linux",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			time.Second,
			0,
		)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Collected,
	)

	assert.False(
		t,
		profile.Valid,
	)

	assert.False(
		t,
		profile.ExclusiveContainer,
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"container_concurrency_not_exclusive",
	)
}

func TestBuildInvocationResourceProfileRejectsRegressingCounters(
	t *testing.T,
) {
	before :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageUserNs: 2_000_000_000,

			CPUUsageKernelNs: 500_000_000,

			PageFaults: 200,

			PageFaultsAvailable: true,
		}

	after :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageUserNs: 1_000_000_000,

			CPUUsageKernelNs: 400_000_000,

			PageFaults: 100,

			PageFaultsAvailable: true,
		}

	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			true,
			before,
			after,
			time.Second,
			0,
		)

	require.False(
		t,
		profile.Valid,
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"cpu_user_counter_regressed",
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"cpu_kernel_counter_regressed",
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"page_fault_counter_regressed",
	)
}

// TestBuildInvocationResourceProfileRejectsNonPositiveWallTime covers the only
// surviving validity check on the measurement window.
func TestBuildInvocationResourceProfileRejectsNonPositiveWallTime(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			true,
			ResourceSnapshot{
				OSType: "linux",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			0,
			0,
		)

	require.False(
		t,
		profile.Valid,
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"non_positive_execution_wall_time",
	)
}

func TestBuildInvocationResourceProfileRejectsNonLinuxContainer(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			true,
			ResourceSnapshot{
				OSType: "windows",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			time.Second,
			0,
		)

	require.False(
		t,
		profile.Valid,
	)

	assert.Contains(
		t,
		profile.InvalidReason,
		"unsupported_os_before",
	)
}

func TestNewInvalidInvocationResourceProfileRecordsReason(
	t *testing.T,
) {
	profile :=
		NewInvalidInvocationResourceProfile(
			"container-a",
			true,
			"snapshot_before_failed: boom",
			2*time.Millisecond,
		)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Enabled,
	)

	assert.False(
		t,
		profile.Collected,
	)

	assert.False(
		t,
		profile.Valid,
	)

	assert.Equal(
		t,
		"snapshot_before_failed: boom",
		profile.InvalidReason,
	)

	assert.InDelta(
		t,
		2.0,
		profile.ProfilingStartOverheadMs,
		1e-9,
	)
}

// TestInvocationResourceProfileUsesStableJSONKeys pins the exported schema.
//
// The JSONL dataset is consumed by the offline Python pipeline and by the
// validation scripts, so the key names must not follow Go identifiers: renaming
// a struct field must never silently change the dataset.
func TestInvocationResourceProfileUsesStableJSONKeys(
	t *testing.T,
) {
	encoded, err :=
		json.Marshal(
			BuildInvocationResourceProfile(
				"container-a",
				true,
				ResourceSnapshot{
					OSType:              "linux",
					PageFaultsAvailable: true,
				},
				ResourceSnapshot{
					OSType:              "linux",
					PageFaultsAvailable: true,
				},
				time.Second,
				0,
			),
		)

	require.NoError(
		t,
		err,
	)

	var decoded map[string]any

	require.NoError(
		t,
		json.Unmarshal(
			encoded,
			&decoded,
		),
	)

	expectedKeys :=
		[]string{
			"enabled",
			"collected",
			"valid",
			"container_id",
			"exclusive_container",
			"cpu_usage_user_delta_ns",
			"cpu_usage_kernel_delta_ns",
			"page_faults_delta",
			"page_faults_available",
			"profiling_start_overhead_ms",
		}

	for _, key := range expectedKeys {

		assert.Contains(
			t,
			decoded,
			key,
		)
	}

	// invalid_reason is omitted when the profile is valid.
	assert.Len(
		t,
		decoded,
		len(expectedKeys),
	)
}
