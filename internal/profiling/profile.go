package profiling

import (
	"strings"
	"time"
)

// ResourceSnapshot contains the counters read by Docker.
//
// CPU and page fault counters are used for invocation profiling.
// The memory counters are absolute readings used exclusively
// for optional container lifecycle monitoring.
//
// Memory collection does not affect the MAB reward,
// the classifier or the choice of architecture.
type ResourceSnapshot struct {
	OSType              string
	CPUUsageUserNs      uint64
	CPUUsageKernelNs    uint64
	PageFaults          uint64
	PageFaultsAvailable bool

	// Absolute container memory counters.
	MemoryUsageBytes        uint64
	MemoryLimitBytes        uint64
	MemoryPeakBytes         uint64
	MemoryAnonBytes         uint64
	MemoryFileBytes         uint64
	MemoryKernelBytes       uint64
	MemoryInactiveFileBytes uint64

	MemoryAnonAvailable         bool
	MemoryFileAvailable         bool
	MemoryKernelAvailable       bool
	MemoryInactiveFileAvailable bool
}

// InvocationResourceProfile contains the container-scoped deltas measured
// around one function invocation.
//
// Every field is either an input of BuildProfilingFeatureVector or a condition
// used by BuildInvocationSample to decide whether the sample is eligible for
// resource clustering. The profile does not influence the MAB reward, arm
// selection, or the LinUCB context.
//
// The JSON field names are explicit and stable: the exported dataset must not
// depend on Go identifiers.
type InvocationResourceProfile struct {
	Enabled            bool   `json:"enabled"`
	Collected          bool   `json:"collected"`
	Valid              bool   `json:"valid"`
	InvalidReason      string `json:"invalid_reason,omitempty"`
	ContainerID        string `json:"container_id"`
	ExclusiveContainer bool   `json:"exclusive_container"`

	// cpuUserDelta and cpuKernelDelta of the reference paper. They are also
	// the two terms of the derived utilizedCPUs feature.
	CPUUsageUserDeltaNs   uint64 `json:"cpu_usage_user_delta_ns"`
	CPUUsageKernelDeltaNs uint64 `json:"cpu_usage_kernel_delta_ns"`

	// pageFaultsDelta of the reference paper, container-scoped so that it is
	// attributable to the single invocation.
	PageFaultsDelta     uint64 `json:"page_faults_delta"`
	PageFaultsAvailable bool   `json:"page_faults_available"`

	// Container-side profiling overheads.
	//
	// ProfilingStartOverheadMs measures the snapshot collected before the
	// invocation. ProfilingEndOverheadMs measures the corresponding snapshot
	// collected after the invocation.
	//
	// ProfilingTotalOverheadMs is their sum.
	ProfilingStartOverheadMs float64 `json:"profiling_start_overhead_ms"`
	ProfilingEndOverheadMs   float64 `json:"profiling_end_overhead_ms"`
	ProfilingTotalOverheadMs float64 `json:"profiling_total_overhead_ms"`
}

// NewInvalidInvocationResourceProfile creates a profile that records why
// profiling could not produce a usable per-invocation sample.
//
// This legacy-compatible constructor only records the initial overhead.
func NewInvalidInvocationResourceProfile(
	containerID string,
	exclusiveContainer bool,
	reason string,
	startOverhead time.Duration,
) *InvocationResourceProfile {
	return &InvocationResourceProfile{
		Enabled:                  true,
		Collected:                false,
		Valid:                    false,
		InvalidReason:            reason,
		ContainerID:              containerID,
		ExclusiveContainer:       exclusiveContainer,
		ProfilingStartOverheadMs: durationMilliseconds(startOverhead),
	}
}

// NewInvalidInvocationResourceProfileWithOverheads records both the initial
// and final container snapshot overheads when they are available.
func NewInvalidInvocationResourceProfileWithOverheads(
	containerID string,
	exclusiveContainer bool,
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
		ProfilingStartOverheadMs: durationMilliseconds(startOverhead),
		ProfilingEndOverheadMs:   durationMilliseconds(endOverhead),
		ProfilingTotalOverheadMs: durationMilliseconds(startOverhead + endOverhead),
	}
}

// BuildInvocationResourceProfile computes the container-scoped deltas for one
// profiled invocation.
//
// This legacy-compatible entry point keeps the previous API and records only
// the initial snapshot overhead.
func BuildInvocationResourceProfile(
	containerID string,
	exclusiveContainer bool,
	before ResourceSnapshot,
	after ResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
) *InvocationResourceProfile {
	return buildInvocationResourceProfile(
		containerID,
		exclusiveContainer,
		before,
		after,
		executionWallTime,
		startOverhead,
		0,
		false,
	)
}

// BuildInvocationResourceProfileWithOverheads computes the same resource
// deltas while also retaining the initial and final container snapshot costs.
func BuildInvocationResourceProfileWithOverheads(
	containerID string,
	exclusiveContainer bool,
	before ResourceSnapshot,
	after ResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
	endOverhead time.Duration,
) *InvocationResourceProfile {
	return buildInvocationResourceProfile(
		containerID,
		exclusiveContainer,
		before,
		after,
		executionWallTime,
		startOverhead,
		endOverhead,
		true,
	)
}

func buildInvocationResourceProfile(
	containerID string,
	exclusiveContainer bool,
	before ResourceSnapshot,
	after ResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
	endOverhead time.Duration,
	includeEndOverhead bool,
) *InvocationResourceProfile {

	profile :=
		&InvocationResourceProfile{
			Enabled:                  true,
			Collected:                true,
			Valid:                    true,
			ContainerID:              containerID,
			ExclusiveContainer:       exclusiveContainer,
			PageFaultsAvailable:      before.PageFaultsAvailable && after.PageFaultsAvailable,
			ProfilingStartOverheadMs: durationMilliseconds(startOverhead),
		}

	if includeEndOverhead {
		profile.ProfilingEndOverheadMs =
			durationMilliseconds(endOverhead)

		profile.ProfilingTotalOverheadMs =
			durationMilliseconds(startOverhead + endOverhead)
	}

	var invalidReasons []string

	if before.OSType != "" &&
		!strings.EqualFold(before.OSType, "linux") {
		invalidReasons =
			append(
				invalidReasons,
				"unsupported_os_before",
			)
	}

	if after.OSType != "" &&
		!strings.EqualFold(after.OSType, "linux") {
		invalidReasons =
			append(
				invalidReasons,
				"unsupported_os_after",
			)
	}

	var ok bool

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

	// The measurement window must be positive: a non-positive wall time means
	// the invocation was not actually observed.
	if executionWallTime <= 0 {
		invalidReasons =
			append(
				invalidReasons,
				"non_positive_execution_wall_time",
			)
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
			strings.Join(invalidReasons, ",")
	}

	return profile
}

// counterDelta subtracts two cumulative counters, reporting whether the counter
// progressed monotonically.
func counterDelta(
	after uint64,
	before uint64,
) (uint64, bool) {
	if after < before {
		return 0, false
	}

	return after - before, true
}

func durationMilliseconds(
	value time.Duration,
) float64 {
	return float64(value) /
		float64(time.Millisecond)
}
