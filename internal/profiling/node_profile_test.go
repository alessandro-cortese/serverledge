package profiling

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseProcStat(
	t *testing.T,
) {
	data :=
		[]byte(
			`cpu  100 20 30 400 50 6 7 8 9 10
cpu0 50 10 15 200 25 3 4 4 5 5
cpu1 50 10 15 200 25 3 3 4 4 5
intr 1
`,
		)

	result, err :=
		parseProcStat(
			data,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		uint64(100),
		result.UserTicks,
	)

	assert.Equal(
		t,
		uint64(20),
		result.NiceTicks,
	)

	assert.Equal(
		t,
		uint64(30),
		result.KernelTicks,
	)

	assert.Equal(
		t,
		uint64(400),
		result.IdleTicks,
	)

	assert.Equal(
		t,
		uint64(50),
		result.IOWaitTicks,
	)

	assert.Equal(
		t,
		uint64(6),
		result.IRQTicks,
	)

	assert.Equal(
		t,
		uint64(7),
		result.SoftIRQTicks,
	)

	assert.Equal(
		t,
		uint64(8),
		result.StealTicks,
	)

	assert.Equal(
		t,
		uint64(9),
		result.GuestTicks,
	)

	assert.Equal(
		t,
		uint64(10),
		result.GuestNiceTicks,
	)

	assert.Equal(
		t,
		2,
		result.AvailableCPUs,
	)
}

func TestParseProcStatAcceptsMissingOptionalFields(
	t *testing.T,
) {
	data :=
		[]byte(
			`cpu 10 2 3 40
cpu0 10 2 3 40
`,
		)

	result, err :=
		parseProcStat(
			data,
		)

	require.NoError(
		t,
		err,
	)

	assert.Zero(
		t,
		result.IOWaitTicks,
	)

	assert.Zero(
		t,
		result.GuestTicks,
	)

	assert.Equal(
		t,
		1,
		result.AvailableCPUs,
	)
}

func TestParseProcMemInfo(
	t *testing.T,
) {
	data :=
		[]byte(
			`MemTotal:       1024 kB
MemFree:         256 kB
MemAvailable:    512 kB
Buffers:          10 kB
`,
		)

	result, err :=
		parseProcMemInfo(
			data,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		uint64(1024*1024),
		result.TotalBytes,
	)

	assert.Equal(
		t,
		uint64(256*1024),
		result.FreeBytes,
	)

	assert.Equal(
		t,
		uint64(512*1024),
		result.AvailableBytes,
	)
}

func TestParseProcVMStat(
	t *testing.T,
) {
	data :=
		[]byte(
			`nr_free_pages 10
pgfault 1234
pgmajfault 12
`,
		)

	result, err :=
		parseProcVMStat(
			data,
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		uint64(1234),
		result.PageFaults,
	)
}

// TestParseProcVMStatRequiresPageFaults documents that pgfault is the only
// mandatory counter: /proc/vmstat content without it must be rejected, while
// the absence of any other line is irrelevant.
func TestParseProcVMStatRequiresPageFaults(
	t *testing.T,
) {
	data :=
		[]byte(
			`nr_free_pages 10
pgmajfault 12
`,
		)

	_, err :=
		parseProcVMStat(
			data,
		)

	require.Error(
		t,
		err,
	)
}

func TestReadNodeResourceSnapshotSupportsPartialAvailability(
	t *testing.T,
) {
	root :=
		t.TempDir()

	err :=
		os.WriteFile(
			filepath.Join(
				root,
				"stat",
			),
			[]byte(
				"cpu 10 0 2 30 0 0 0 0 0 0\n"+
					"cpu0 10 0 2 30 0 0 0 0 0 0\n",
			),
			0o600,
		)

	require.NoError(
		t,
		err,
	)

	snapshot, err :=
		readNodeResourceSnapshotFromRoot(
			root,
		)

	require.NoError(
		t,
		err,
	)

	assert.True(
		t,
		snapshot.CPUAvailable,
	)

	assert.False(
		t,
		snapshot.MemoryAvailable,
	)

	assert.False(
		t,
		snapshot.VMStatAvailable,
	)

	assert.NotEmpty(
		t,
		snapshot.CollectionErrors,
	)
}

func TestBuildNodeResourceProfileCalculatesDeltas(
	t *testing.T,
) {
	before :=
		NodeResourceSnapshot{
			CPUAvailable:    true,
			MemoryAvailable: true,
			VMStatAvailable: true,

			CPU: NodeCPUStatSnapshot{
				UserTicks:      100,
				NiceTicks:      10,
				KernelTicks:    20,
				IdleTicks:      300,
				IOWaitTicks:    5,
				IRQTicks:       2,
				SoftIRQTicks:   3,
				StealTicks:     1,
				GuestTicks:     4,
				GuestNiceTicks: 2,
				AvailableCPUs:  2,
			},

			Memory: NodeMemorySnapshot{
				TotalBytes:     1000,
				FreeBytes:      400,
				AvailableBytes: 600,
			},

			VMStat: NodeVMStatSnapshot{
				PageFaults: 1000,
			},
		}

	after :=
		NodeResourceSnapshot{
			CPUAvailable:    true,
			MemoryAvailable: true,
			VMStatAvailable: true,

			CPU: NodeCPUStatSnapshot{
				UserTicks:      130,
				NiceTicks:      14,
				KernelTicks:    30,
				IdleTicks:      340,
				IOWaitTicks:    10,
				IRQTicks:       3,
				SoftIRQTicks:   5,
				StealTicks:     2,
				GuestTicks:     6,
				GuestNiceTicks: 3,
				AvailableCPUs:  2,
			},

			Memory: NodeMemorySnapshot{
				TotalBytes:     1000,
				FreeBytes:      300,
				AvailableBytes: 500,
			},

			VMStat: NodeVMStatSnapshot{
				PageFaults: 1040,
			},
		}

	profile :=
		BuildNodeResourceProfile(
			before,
			after,
			1000*time.Millisecond,
			2*time.Millisecond,
			3*time.Millisecond,
		)

	require.True(
		t,
		profile.Collected,
	)

	require.True(
		t,
		profile.Complete,
	)

	require.True(
		t,
		profile.CPUAvailable,
	)

	assert.Equal(
		t,
		2,
		profile.AvailableCPUs,
	)

	// Tick deltas are no longer exported: only the normalized millisecond
	// values are. The expected values are therefore expressed through the
	// same relation the implementation applies:
	//
	//	deltaMs = wallTimeMs * availableCPUs * (modeTicks / totalTicks)
	//
	// with guest time already subtracted from user time and guest_nice
	// already subtracted from nice time:
	//
	//	user  = (130 - 100) - (6 - 4) = 28
	//	nice  = (14 - 10) - (3 - 2)   = 3
	//	total = 28+3+10+40+5+1+2+1+2+1 = 93
	const (
		expectedTotalTicks   = 93.0
		expectedAvailableCPU = 2.0
		expectedWallTimeMs   = 1000.0
	)

	availableCPUTimeMs :=
		expectedWallTimeMs *
			expectedAvailableCPU

	expectedMs :=
		func(
			modeTicks float64,
		) float64 {
			return availableCPUTimeMs *
				(modeTicks /
					expectedTotalTicks)
		}

	assert.InDelta(
		t,
		expectedMs(28),
		profile.CPUUserDeltaMs,
		1e-6,
	)

	assert.InDelta(
		t,
		expectedMs(3),
		profile.CPUNiceDeltaMs,
		1e-6,
	)

	assert.InDelta(
		t,
		expectedMs(10),
		profile.CPUKernelDeltaMs,
		1e-6,
	)

	assert.InDelta(
		t,
		expectedMs(40),
		profile.CPUIdleDeltaMs,
		1e-6,
	)

	assert.InDelta(
		t,
		expectedMs(5),
		profile.CPUIOWaitDeltaMs,
		1e-6,
	)

	assert.InDelta(
		t,
		expectedMs(2),
		profile.CPUGuestDeltaMs,
		1e-6,
	)

	// The invariant of Linux CPU time accounting: for every second of wall
	// time each available CPU contributes one second distributed across the
	// modes, so the modes must still sum to the total available CPU time.
	assert.InDelta(
		t,
		availableCPUTimeMs,
		sumNodeCPUMilliseconds(
			profile,
		),
		1e-6,
	)

	assert.Equal(
		t,
		uint64(400),
		profile.FreeMemoryBeforeBytes,
	)

	assert.Equal(
		t,
		uint64(300),
		profile.FreeMemoryAfterBytes,
	)

	assert.Equal(
		t,
		uint64(1000),
		profile.PageFaultsBefore,
	)

	assert.Equal(
		t,
		uint64(1040),
		profile.PageFaultsAfter,
	)

	assert.Equal(
		t,
		uint64(40),
		profile.PageFaultsDelta,
	)

	assert.InDelta(
		t,
		5.0,
		profile.SnapshotTotalOverheadMs,
		1e-9,
	)
}

func TestBuildNodeResourceProfileRejectsRegressingCPUCounter(
	t *testing.T,
) {
	before :=
		NodeResourceSnapshot{
			CPUAvailable: true,

			CPU: NodeCPUStatSnapshot{
				UserTicks:     100,
				IdleTicks:     100,
				AvailableCPUs: 1,
			},
		}

	after :=
		NodeResourceSnapshot{
			CPUAvailable: true,

			CPU: NodeCPUStatSnapshot{
				UserTicks:     90,
				IdleTicks:     110,
				AvailableCPUs: 1,
			},
		}

	profile :=
		BuildNodeResourceProfile(
			before,
			after,
			time.Second,
			0,
			0,
		)

	assert.False(
		t,
		profile.CPUAvailable,
	)

	assert.False(
		t,
		profile.Complete,
	)

	assert.Contains(
		t,
		profile.Errors,
		"node_cpu_user_counter_regressed",
	)
}

func TestBuildInvocationSamplePreservesNodeEnvironment(
	t *testing.T,
) {
	nodeEnvironment :=
		&NodeResourceProfile{
			Collected:     true,
			Complete:      true,
			CPUAvailable:  true,
			AvailableCPUs: 2,
		}

	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				WarmStart:          true,
				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					DurationMs:     10,
					ResponseTimeMs: 11,
				},

				Profile: &InvocationResourceProfile{
					Enabled:            true,
					Collected:          true,
					Valid:              true,
					ExclusiveContainer: true,
				},

				NodeEnvironment: nodeEnvironment,
			},
		)

	assert.Same(
		t,
		nodeEnvironment,
		sample.NodeEnvironment,
	)

	assert.Equal(
		t,
		InvocationSampleSchemaVersion,
		sample.SchemaVersion,
	)
}

func sumNodeCPUMilliseconds(
	profile *NodeResourceProfile,
) float64 {
	return profile.CPUUserDeltaMs +
		profile.CPUNiceDeltaMs +
		profile.CPUKernelDeltaMs +
		profile.CPUIdleDeltaMs +
		profile.CPUIOWaitDeltaMs +
		profile.CPUIRQDeltaMs +
		profile.CPUSoftIRQDeltaMs +
		profile.CPUStealDeltaMs +
		profile.CPUGuestDeltaMs +
		profile.CPUGuestNiceDeltaMs
}
