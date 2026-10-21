package profiling

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPopulateNodeCPUProfileRetainsRawProcStatTickDeltas(
	t *testing.T,
) {
	profile :=
		&NodeResourceProfile{
			ExecutionWallTimeMs: 1000,
		}

	before :=
		NodeCPUStatSnapshot{
			UserTicks:      100,
			NiceTicks:      20,
			KernelTicks:    50,
			IdleTicks:      1000,
			IOWaitTicks:    10,
			IRQTicks:       5,
			SoftIRQTicks:   7,
			StealTicks:     3,
			GuestTicks:     10,
			GuestNiceTicks: 2,
			AvailableCPUs:  4,
		}

	after :=
		NodeCPUStatSnapshot{
			UserTicks:      120,
			NiceTicks:      25,
			KernelTicks:    58,
			IdleTicks:      1300,
			IOWaitTicks:    12,
			IRQTicks:       6,
			SoftIRQTicks:   10,
			StealTicks:     4,
			GuestTicks:     15,
			GuestNiceTicks: 3,
			AvailableCPUs:  4,
		}

	err :=
		populateNodeCPUProfile(
			profile,
			before,
			after,
		)

	require.NoError(
		t,
		err,
	)

	// The raw user/nice deltas intentionally include guest/guest_nice, exactly
	// as the first two aggregate /proc/stat fields do and exactly as SAAF reads
	// them before multiplying by 1000/CLK_TCK.
	assert.Equal(
		t,
		uint64(20),
		profile.CPUUserDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(5),
		profile.CPUNiceDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(8),
		profile.CPUKernelDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(300),
		profile.CPUIdleDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(2),
		profile.CPUIOWaitDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(1),
		profile.CPUIRQDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(3),
		profile.CPUSoftIRQDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(1),
		profile.CPUStealDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(5),
		profile.CPUGuestDeltaTicks,
	)

	assert.Equal(
		t,
		uint64(1),
		profile.CPUGuestNiceDeltaTicks,
	)

	// The historical normalized-ms fields remain valid and unchanged in
	// semantics; the new raw fields are additive rather than a replacement.
	assert.GreaterOrEqual(
		t,
		profile.CPUUserDeltaMs,
		0.0,
	)

	assert.GreaterOrEqual(
		t,
		profile.CPUKernelDeltaMs,
		0.0,
	)

	// getconf is deliberately non-fatal for the historical pipeline. On normal
	// Linux hosts, including the intended GCP workers, it should be available.
	if profile.CPUClockTicksAvailable {
		assert.Greater(
			t,
			profile.CPUClockTicksPerSecond,
			uint64(0),
		)
	}
}
