package profiling

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// ResourceSnapshot is a point-in-time view of cumulative resource counters
// exposed by Docker for one container.
//
// Most counters are cumulative since container creation. Per-invocation values
// are obtained by subtracting a snapshot taken before the invocation from one
// taken after it.
type ResourceSnapshot struct {
	OSType string
	ReadAt time.Time

	CPUUsageTotalNs  uint64
	CPUUsageUserNs   uint64
	CPUUsageKernelNs uint64

	CPUThrottledTimeNs  uint64
	CPUThrottledPeriods uint64
	OnlineCPUs          uint32

	MemoryUsageBytes uint64
	MemoryLimitBytes uint64

	PageFaults               uint64
	MajorPageFaults          uint64
	PageFaultsAvailable      bool
	MajorPageFaultsAvailable bool

	BlockReadBytes  uint64
	BlockWriteBytes uint64
	BlockReadOps    uint64
	BlockWriteOps   uint64

	BlockIOBytesAvailable bool
	BlockIOOpsAvailable   bool

	NetworkRxBytes   uint64
	NetworkTxBytes   uint64
	NetworkAvailable bool

	PIDs uint64
}

// InvocationResourceProfile contains resource deltas measured around one
// function invocation.
//
// The profile is observability data only. It does not influence the MAB reward,
// arm selection, or contextual features in the current implementation.
type InvocationResourceProfile struct {
	Enabled bool

	Collected     bool
	Valid         bool
	InvalidReason string

	ContainerID        string
	ExclusiveContainer bool
	MaxConcurrency     int16

	// ProfilingLockWaitMs is the time spent waiting for the per-container
	// profiling critical section. It is kept separate from snapshot overhead
	// and execution wall time.
	ProfilingLockWaitMs float64

	CPUUsageTotalDeltaNs  uint64
	CPUUsageUserDeltaNs   uint64
	CPUUsageKernelDeltaNs uint64

	CPUThrottledTimeDeltaNs  uint64
	CPUThrottledPeriodsDelta uint64
	OnlineCPUs               uint32
	UtilizedCPUs             float64

	MemoryUsageBeforeBytes uint64
	MemoryUsageAfterBytes  uint64
	MemoryUsageDeltaBytes  int64
	MemoryLimitBytes       uint64

	PageFaultsDelta          uint64
	MajorPageFaultsDelta     uint64
	PageFaultsAvailable      bool
	MajorPageFaultsAvailable bool

	BlockReadBytesDelta  uint64
	BlockWriteBytesDelta uint64
	BlockReadOpsDelta    uint64
	BlockWriteOpsDelta   uint64

	BlockIOAvailable      bool
	BlockIOBytesAvailable bool
	BlockIOOpsAvailable   bool

	NetworkRxBytesDelta uint64
	NetworkTxBytesDelta uint64
	NetworkAvailable    bool

	PIDsBefore uint64
	PIDsAfter  uint64

	ExecutionWallTimeMs      float64
	ProfilingStartOverheadMs float64
	ProfilingEndOverheadMs   float64
	ProfilingTotalOverheadMs float64
}

// NewInvalidInvocationResourceProfile creates a profile that records why
// profiling could not produce a usable per-invocation sample.
func NewInvalidInvocationResourceProfile(
	containerID string,
	maxConcurrency int16,
	exclusiveContainer bool,
	profilingLockWait time.Duration,
	reason string,
	startOverhead time.Duration,
	endOverhead time.Duration,
) *InvocationResourceProfile {
	return &InvocationResourceProfile{
		Enabled:                  true,
		Collected:                false,
		Valid:                    false,
		InvalidReason:            reason,
		ContainerID:              containerID,
		ExclusiveContainer:       exclusiveContainer,
		MaxConcurrency:           maxConcurrency,
		ProfilingLockWaitMs:      durationMilliseconds(profilingLockWait),
		ProfilingStartOverheadMs: durationMilliseconds(startOverhead),
		ProfilingEndOverheadMs:   durationMilliseconds(endOverhead),
		ProfilingTotalOverheadMs: durationMilliseconds(
			startOverhead + endOverhead,
		),
	}
}

// BuildInvocationResourceProfile calculates per-invocation deltas from two
// Docker snapshots.
func BuildInvocationResourceProfile(
	containerID string,
	maxConcurrency int16,
	exclusiveContainer bool,
	profilingLockWait time.Duration,
	before ResourceSnapshot,
	after ResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
	endOverhead time.Duration,
) *InvocationResourceProfile {
	profile := &InvocationResourceProfile{
		Enabled:            true,
		Collected:          true,
		Valid:              true,
		ContainerID:        containerID,
		ExclusiveContainer: exclusiveContainer,
		MaxConcurrency:     maxConcurrency,
		ProfilingLockWaitMs: durationMilliseconds(
			profilingLockWait,
		),
		OnlineCPUs: after.OnlineCPUs,

		MemoryUsageBeforeBytes: before.MemoryUsageBytes,
		MemoryUsageAfterBytes:  after.MemoryUsageBytes,
		MemoryUsageDeltaBytes: signedDelta(
			after.MemoryUsageBytes,
			before.MemoryUsageBytes,
		),
		MemoryLimitBytes: after.MemoryLimitBytes,

		PageFaultsAvailable: before.PageFaultsAvailable &&
			after.PageFaultsAvailable,

		MajorPageFaultsAvailable: before.MajorPageFaultsAvailable &&
			after.MajorPageFaultsAvailable,

		BlockIOBytesAvailable: before.BlockIOBytesAvailable &&
			after.BlockIOBytesAvailable,

		BlockIOOpsAvailable: before.BlockIOOpsAvailable &&
			after.BlockIOOpsAvailable,

		NetworkAvailable: before.NetworkAvailable &&
			after.NetworkAvailable,

		PIDsBefore: before.PIDs,
		PIDsAfter:  after.PIDs,

		ExecutionWallTimeMs: durationMilliseconds(
			executionWallTime,
		),

		ProfilingStartOverheadMs: durationMilliseconds(
			startOverhead,
		),

		ProfilingEndOverheadMs: durationMilliseconds(
			endOverhead,
		),

		ProfilingTotalOverheadMs: durationMilliseconds(
			startOverhead + endOverhead,
		),
	}

	profile.BlockIOAvailable =
		profile.BlockIOBytesAvailable ||
			profile.BlockIOOpsAvailable

	var invalidReasons []string

	if before.OSType != "" &&
		!strings.EqualFold(
			before.OSType,
			"linux",
		) {

		invalidReasons =
			append(
				invalidReasons,
				"unsupported_os_before",
			)
	}

	if after.OSType != "" &&
		!strings.EqualFold(
			after.OSType,
			"linux",
		) {

		invalidReasons =
			append(
				invalidReasons,
				"unsupported_os_after",
			)
	}

	var ok bool

	profile.CPUUsageTotalDeltaNs, ok =
		counterDelta(
			after.CPUUsageTotalNs,
			before.CPUUsageTotalNs,
		)

	if !ok {
		invalidReasons =
			append(
				invalidReasons,
				"cpu_total_counter_regressed",
			)
	}

	profile.CPUUsageUserDeltaNs, ok =
		counterDelta(
			after.CPUUsageUserNs,
			before.CPUUsageUserNs,
		)

	if !ok {
		invalidReasons =
			append(
				invalidReasons,
				"cpu_user_counter_regressed",
			)
	}

	profile.CPUUsageKernelDeltaNs, ok =
		counterDelta(
			after.CPUUsageKernelNs,
			before.CPUUsageKernelNs,
		)

	if !ok {
		invalidReasons =
			append(
				invalidReasons,
				"cpu_kernel_counter_regressed",
			)
	}

	profile.CPUThrottledTimeDeltaNs, ok =
		counterDelta(
			after.CPUThrottledTimeNs,
			before.CPUThrottledTimeNs,
		)

	if !ok {
		invalidReasons =
			append(
				invalidReasons,
				"cpu_throttled_time_counter_regressed",
			)
	}

	profile.CPUThrottledPeriodsDelta, ok =
		counterDelta(
			after.CPUThrottledPeriods,
			before.CPUThrottledPeriods,
		)

	if !ok {
		invalidReasons =
			append(
				invalidReasons,
				"cpu_throttled_periods_counter_regressed",
			)
	}

	if profile.PageFaultsAvailable {
		profile.PageFaultsDelta, ok =
			counterDelta(
				after.PageFaults,
				before.PageFaults,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"page_fault_counter_regressed",
				)
		}
	}

	if profile.MajorPageFaultsAvailable {
		profile.MajorPageFaultsDelta, ok =
			counterDelta(
				after.MajorPageFaults,
				before.MajorPageFaults,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"major_page_fault_counter_regressed",
				)
		}
	}

	if profile.BlockIOBytesAvailable {
		profile.BlockReadBytesDelta, ok =
			counterDelta(
				after.BlockReadBytes,
				before.BlockReadBytes,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"block_read_bytes_counter_regressed",
				)
		}

		profile.BlockWriteBytesDelta, ok =
			counterDelta(
				after.BlockWriteBytes,
				before.BlockWriteBytes,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"block_write_bytes_counter_regressed",
				)
		}
	}

	if profile.BlockIOOpsAvailable {
		profile.BlockReadOpsDelta, ok =
			counterDelta(
				after.BlockReadOps,
				before.BlockReadOps,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"block_read_ops_counter_regressed",
				)
		}

		profile.BlockWriteOpsDelta, ok =
			counterDelta(
				after.BlockWriteOps,
				before.BlockWriteOps,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"block_write_ops_counter_regressed",
				)
		}
	}

	if profile.NetworkAvailable {
		profile.NetworkRxBytesDelta, ok =
			counterDelta(
				after.NetworkRxBytes,
				before.NetworkRxBytes,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"network_rx_counter_regressed",
				)
		}

		profile.NetworkTxBytesDelta, ok =
			counterDelta(
				after.NetworkTxBytes,
				before.NetworkTxBytes,
			)

		if !ok {
			invalidReasons =
				append(
					invalidReasons,
					"network_tx_counter_regressed",
				)
		}
	}

	if executionWallTime <= 0 {
		invalidReasons =
			append(
				invalidReasons,
				"non_positive_execution_wall_time",
			)
	} else {
		profile.UtilizedCPUs =
			float64(
				profile.CPUUsageTotalDeltaNs,
			) /
				float64(
					executionWallTime.Nanoseconds(),
				)

		if math.IsNaN(
			profile.UtilizedCPUs,
		) ||
			math.IsInf(
				profile.UtilizedCPUs,
				0,
			) {

			invalidReasons =
				append(
					invalidReasons,
					"non_finite_utilized_cpus",
				)
		}
	}

	if !profile.ExclusiveContainer {
		invalidReasons =
			append(
				invalidReasons,
				"container_concurrency_not_exclusive",
			)
	}

	if len(invalidReasons) > 0 {
		profile.Valid = false

		profile.InvalidReason =
			strings.Join(
				invalidReasons,
				",",
			)
	}

	return profile
}

func counterDelta(
	after uint64,
	before uint64,
) (uint64, bool) {
	if after < before {
		return 0, false
	}

	return after - before, true
}

func signedDelta(
	after uint64,
	before uint64,
) int64 {
	if after >= before {
		delta :=
			after - before

		if delta > math.MaxInt64 {
			return math.MaxInt64
		}

		return int64(delta)
	}

	delta :=
		before - after

	if delta > math.MaxInt64 {
		return math.MinInt64
	}

	return -int64(delta)
}

func durationMilliseconds(
	value time.Duration,
) float64 {
	return float64(value) /
		float64(time.Millisecond)
}

func (p InvocationResourceProfile) String() string {
	return fmt.Sprintf(
		"valid=%t reason=%s cpu_total_delta_ns=%d cpu_user_delta_ns=%d cpu_kernel_delta_ns=%d utilized_cpus=%.6f page_faults_delta=%d memory_delta_bytes=%d",
		p.Valid,
		p.InvalidReason,
		p.CPUUsageTotalDeltaNs,
		p.CPUUsageUserDeltaNs,
		p.CPUUsageKernelDeltaNs,
		p.UtilizedCPUs,
		p.PageFaultsDelta,
		p.MemoryUsageDeltaBytes,
	)
}
