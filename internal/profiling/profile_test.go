package profiling

import (
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

			CPUUsageTotalNs: 1_000_000_000,

			CPUUsageUserNs: 700_000_000,

			CPUUsageKernelNs: 300_000_000,

			CPUThrottledTimeNs: 10_000,

			CPUThrottledPeriods: 2,

			OnlineCPUs: 2,

			MemoryUsageBytes: 1000,

			MemoryLimitBytes: 4096,

			PageFaults: 100,

			MajorPageFaults: 4,

			PageFaultsAvailable: true,

			MajorPageFaultsAvailable: true,

			BlockReadBytes: 1000,

			BlockWriteBytes: 2000,

			BlockReadOps: 10,

			BlockWriteOps: 20,

			BlockIOBytesAvailable: true,

			BlockIOOpsAvailable: true,

			NetworkRxBytes: 500,

			NetworkTxBytes: 800,

			NetworkAvailable: true,

			PIDs: 2,
		}

	after :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageTotalNs: 2_500_000_000,

			CPUUsageUserNs: 1_700_000_000,

			CPUUsageKernelNs: 800_000_000,

			CPUThrottledTimeNs: 20_000,

			CPUThrottledPeriods: 5,

			OnlineCPUs: 2,

			MemoryUsageBytes: 1300,

			MemoryLimitBytes: 4096,

			PageFaults: 130,

			MajorPageFaults: 7,

			PageFaultsAvailable: true,

			MajorPageFaultsAvailable: true,

			BlockReadBytes: 1500,

			BlockWriteBytes: 2700,

			BlockReadOps: 15,

			BlockWriteOps: 28,

			BlockIOBytesAvailable: true,

			BlockIOOpsAvailable: true,

			NetworkRxBytes: 900,

			NetworkTxBytes: 1400,

			NetworkAvailable: true,

			PIDs: 3,
		}

	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			1,
			true,
			4*time.Millisecond,
			before,
			after,
			time.Second,
			2*time.Millisecond,
			3*time.Millisecond,
		)

	assert.InDelta(
		t,
		4.0,
		profile.ProfilingLockWaitMs,
		1e-9,
	)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Collected,
	)

	assert.True(
		t,
		profile.Valid,
	)

	assert.Empty(
		t,
		profile.InvalidReason,
	)

	assert.True(
		t,
		profile.ExclusiveContainer,
	)

	assert.Equal(
		t,
		uint64(
			1_500_000_000,
		),
		profile.CPUUsageTotalDeltaNs,
	)

	assert.Equal(
		t,
		uint64(
			1_000_000_000,
		),
		profile.CPUUsageUserDeltaNs,
	)

	assert.Equal(
		t,
		uint64(
			500_000_000,
		),
		profile.CPUUsageKernelDeltaNs,
	)

	assert.InDelta(
		t,
		1.5,
		profile.UtilizedCPUs,
		1e-9,
	)

	assert.Equal(
		t,
		int64(300),
		profile.MemoryUsageDeltaBytes,
	)

	assert.Equal(
		t,
		uint64(30),
		profile.PageFaultsDelta,
	)

	assert.Equal(
		t,
		uint64(3),
		profile.MajorPageFaultsDelta,
	)

	assert.Equal(
		t,
		uint64(500),
		profile.BlockReadBytesDelta,
	)

	assert.Equal(
		t,
		uint64(700),
		profile.BlockWriteBytesDelta,
	)

	assert.Equal(
		t,
		uint64(5),
		profile.BlockReadOpsDelta,
	)

	assert.Equal(
		t,
		uint64(8),
		profile.BlockWriteOpsDelta,
	)

	assert.Equal(
		t,
		uint64(400),
		profile.NetworkRxBytesDelta,
	)

	assert.Equal(
		t,
		uint64(600),
		profile.NetworkTxBytesDelta,
	)

	assert.InDelta(
		t,
		2.0,
		profile.ProfilingStartOverheadMs,
		1e-9,
	)

	assert.InDelta(
		t,
		3.0,
		profile.ProfilingEndOverheadMs,
		1e-9,
	)

	assert.InDelta(
		t,
		5.0,
		profile.ProfilingTotalOverheadMs,
		1e-9,
	)

	assert.True(
		t,
		profile.BlockIOAvailable,
	)

	assert.True(
		t,
		profile.BlockIOBytesAvailable,
	)

	assert.True(
		t,
		profile.BlockIOOpsAvailable,
	)
}

func TestBuildInvocationResourceProfileMarksNonExclusiveCollectionInvalid(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			2,
			false,
			0,
			ResourceSnapshot{
				OSType: "linux",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			time.Second,
			0,
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

			CPUUsageTotalNs: 2_000_000_000,
		}

	after :=
		ResourceSnapshot{
			OSType: "linux",

			CPUUsageTotalNs: 1_000_000_000,
		}

	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			1,
			true,
			0,
			before,
			after,
			time.Second,
			0,
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

	assert.Contains(
		t,
		profile.InvalidReason,
		"cpu_total_counter_regressed",
	)
}

func TestBuildInvocationResourceProfileSupportsBlockBytesWithoutOps(
	t *testing.T,
) {
	before := ResourceSnapshot{
		OSType: "linux",

		BlockWriteBytes: 1_000,

		BlockIOBytesAvailable: true,
		BlockIOOpsAvailable:   false,
	}

	after := ResourceSnapshot{
		OSType: "linux",

		BlockWriteBytes: 1_000 + 32*1024*1024,

		BlockIOBytesAvailable: true,
		BlockIOOpsAvailable:   false,
	}

	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			1,
			true,
			4*time.Millisecond,
			before,
			after,
			time.Second,
			2*time.Millisecond,
			3*time.Millisecond,
		)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Valid,
	)

	assert.True(
		t,
		profile.BlockIOAvailable,
	)

	assert.True(
		t,
		profile.BlockIOBytesAvailable,
	)

	assert.False(
		t,
		profile.BlockIOOpsAvailable,
	)

	assert.Equal(
		t,
		uint64(32*1024*1024),
		profile.BlockWriteBytesDelta,
	)

	assert.Zero(
		t,
		profile.BlockWriteOpsDelta,
	)
}

func TestBuildInvocationResourceProfileAcceptsExclusiveCollectionWithConfiguredConcurrency(
	t *testing.T,
) {
	profile :=
		BuildInvocationResourceProfile(
			"container-a",
			4,
			true,
			10*time.Millisecond,
			ResourceSnapshot{
				OSType: "linux",
			},
			ResourceSnapshot{
				OSType: "linux",
			},
			time.Second,
			0,
			0,
		)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Valid,
	)

	assert.True(
		t,
		profile.ExclusiveContainer,
	)

	assert.Equal(
		t,
		int16(4),
		profile.MaxConcurrency,
	)

	assert.InDelta(
		t,
		10.0,
		profile.ProfilingLockWaitMs,
		1e-9,
	)

	assert.NotContains(
		t,
		profile.InvalidReason,
		"container_concurrency_not_exclusive",
	)
}
