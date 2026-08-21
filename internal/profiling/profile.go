package profiling

import (
	"strings"
	"time"
)

// ResourceSnapshot is a point-in-time view of the cumulative Docker counters
// that are actually consumed by the profiling pipeline.
//
// Counters are cumulative since container creation: per-invocation values are
// obtained by subtracting a snapshot taken before the invocation from one taken
// after it.
//
// Only the counters needed to build the Random Forest feature vector are kept.
// Docker exposes many more (block I/O, network, PIDs, throttling, memory), but
// none of them is an input of the clustering pipeline, so they are neither
// stored nor exported.
type ResourceSnapshot struct {
	OSType string

	CPUUsageUserNs   uint64
	CPUUsageKernelNs uint64

	PageFaults          uint64
	PageFaultsAvailable bool
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
	Enabled bool `json:"enabled"`

	Collected     bool   `json:"collected"`
	Valid         bool   `json:"valid"`
	InvalidReason string `json:"invalid_reason,omitempty"`

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

	// Container-side half of the frameworkRuntime feature. The node-side half
	// is NodeResourceProfile.SnapshotStartOverheadMs.
	ProfilingStartOverheadMs float64 `json:"profiling_start_overhead_ms"`
}

// NewInvalidInvocationResourceProfile creates a profile that records why
// profiling could not produce a usable per-invocation sample.
func NewInvalidInvocationResourceProfile(
	containerID string,
	exclusiveContainer bool,
	reason string,
	startOverhead time.Duration,
) *InvocationResourceProfile {
	return &InvocationResourceProfile{
		Enabled:            true,
		Collected:          false,
		Valid:              false,
		InvalidReason:      reason,
		ContainerID:        containerID,
		ExclusiveContainer: exclusiveContainer,

		ProfilingStartOverheadMs: durationMilliseconds(
			startOverhead,
		),
	}
}

// BuildInvocationResourceProfile computes the container-scoped deltas for one
// profiled invocation.
//
// A profile is valid only when every counter progressed monotonically, the
// measurement window is positive, and the container served the invocation
// exclusively.
func BuildInvocationResourceProfile(
	containerID string,
	exclusiveContainer bool,
	before ResourceSnapshot,
	after ResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
) *InvocationResourceProfile {
	profile :=
		&InvocationResourceProfile{
			Enabled:            true,
			Collected:          true,
			Valid:              true,
			ContainerID:        containerID,
			ExclusiveContainer: exclusiveContainer,

			PageFaultsAvailable: before.PageFaultsAvailable &&
				after.PageFaultsAvailable,

			ProfilingStartOverheadMs: durationMilliseconds(
				startOverhead,
			),
		}

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
			strings.Join(
				invalidReasons,
				",",
			)
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
