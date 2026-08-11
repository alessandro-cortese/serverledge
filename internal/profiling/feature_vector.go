package profiling

import (
	"fmt"
	"math"
)

const ProfilingFeatureVectorSchemaVersion = 1

// RandomForestProfilingFeatures contains the six profiling dimensions that
// the reference paper reports as the most important for its Random Forest
// classifier.
//
// The terminology follows the paper, while the mapping deliberately uses the
// Serverledge measurements that are most attributable to one invocation.
type RandomForestProfilingFeatures struct {
	PageFaultsDelta    float64 `json:"page_faults_delta"`
	UtilizedCPUs       float64 `json:"utilized_cpus"`
	FreeMemoryMB       float64 `json:"free_memory_mb"`
	CPUUserDeltaMs     float64 `json:"cpu_user_delta_ms"`
	CPUKernelDeltaMs   float64 `json:"cpu_kernel_delta_ms"`
	FrameworkRuntimeMs float64 `json:"framework_runtime_ms"`
}

// ProfilingFeatureVector keeps identifiers and allocation metadata outside the
// actual feature set. This prevents metadata such as request ID, function name,
// machine tag or configured resources from accidentally becoming clustering
// dimensions.
type ProfilingFeatureVector struct {
	SchemaVersion int `json:"schema_version"`

	RequestID    string `json:"request_id"`
	FunctionName string `json:"function_name"`
	MachineTag   string `json:"machine_tag"`

	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`

	Features RandomForestProfilingFeatures `json:"features"`
}

// RandomForestFeatureNames returns the stable order used by Values.
//
// Keeping the order explicit is important because clustering algorithms will
// later consume []float64 rather than a Go struct.
func RandomForestFeatureNames() []string {
	return []string{
		"page_faults_delta",
		"utilized_cpus",
		"free_memory_mb",
		"cpu_user_delta_ms",
		"cpu_kernel_delta_ms",
		"framework_runtime_ms",
	}
}

// Values returns the numeric vector in exactly the same order returned by
// RandomForestFeatureNames.
func (f RandomForestProfilingFeatures) Values() []float64 {
	return []float64{
		f.PageFaultsDelta,
		f.UtilizedCPUs,
		f.FreeMemoryMB,
		f.CPUUserDeltaMs,
		f.CPUKernelDeltaMs,
		f.FrameworkRuntimeMs,
	}
}

// BuildProfilingFeatureVector extracts the six paper-inspired Random Forest
// features from one warm, successful and exclusively profiled invocation.
//
// Serverledge mapping:
//
//	pageFaultsDelta
//	    container-scoped Docker/cgroup page-fault delta.
//
//	utilizedCPUs
//	    (container user CPU + container kernel CPU) / function duration.
//
//	freeMemory
//	    node-scoped MemFree observed before the invocation.
//
//	cpuUserDelta
//	    container-scoped user CPU delta.
//
//	cpuKernelDelta
//	    container-scoped kernel CPU delta.
//
//	frameworkRuntime
//	    Serverledge equivalent of the initial profiling overhead:
//	    container snapshot overhead + node snapshot overhead.
//
// The mapping intentionally keeps function-attributable CPU/page-fault
// measurements container-scoped, while freeMemory remains an environmental
// node-scoped feature as in the reference profiling approach.
func BuildProfilingFeatureVector(
	sample InvocationSample,
) (ProfilingFeatureVector, error) {
	if !sample.Eligibility.ResourceClustering {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q is not eligible for resource clustering",
				sample.RequestID,
			)
	}

	if sample.Profile == nil {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q has no container profile",
				sample.RequestID,
			)
	}

	if sample.NodeEnvironment == nil {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q has no node environment",
				sample.RequestID,
			)
	}

	profile :=
		sample.Profile

	nodeEnvironment :=
		sample.NodeEnvironment

	if !profile.PageFaultsAvailable {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q has no container page-fault counters",
				sample.RequestID,
			)
	}

	if !nodeEnvironment.MemoryAvailable {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q has no node memory snapshot",
				sample.RequestID,
			)
	}

	if !finitePositive(
		sample.Timing.DurationMs,
	) {
		return ProfilingFeatureVector{},
			fmt.Errorf(
				"sample %q has invalid duration_ms %.6f",
				sample.RequestID,
				sample.Timing.DurationMs,
			)
	}

	cpuUserDeltaMs :=
		float64(
			profile.CPUUsageUserDeltaNs,
		) /
			1_000_000.0

	cpuKernelDeltaMs :=
		float64(
			profile.CPUUsageKernelDeltaNs,
		) /
			1_000_000.0

	// This is the Serverledge equivalent closest to the paper's utilizedCPUs:
	// CPU time actually consumed by the invocation divided by user/runtime
	// wall time.
	utilizedCPUs :=
		(cpuUserDeltaMs +
			cpuKernelDeltaMs) /
			sample.Timing.DurationMs

	freeMemoryMB :=
		float64(
			nodeEnvironment.FreeMemoryBeforeBytes,
		) /
			(1024.0 * 1024.0)

	// SAAF measures the overhead of its initial inspection phase.
	// Serverledge performs the corresponding profiling externally, therefore
	// we sum the two initial snapshot overheads.
	frameworkRuntimeMs :=
		profile.ProfilingStartOverheadMs +
			nodeEnvironment.SnapshotStartOverheadMs

	features :=
		RandomForestProfilingFeatures{
			PageFaultsDelta: float64(
				profile.PageFaultsDelta,
			),

			UtilizedCPUs: utilizedCPUs,

			FreeMemoryMB: freeMemoryMB,

			CPUUserDeltaMs: cpuUserDeltaMs,

			CPUKernelDeltaMs: cpuKernelDeltaMs,

			FrameworkRuntimeMs: frameworkRuntimeMs,
		}

	names :=
		RandomForestFeatureNames()

	values :=
		features.Values()

	for index, value := range values {

		if !finiteNonNegative(
			value,
		) {
			return ProfilingFeatureVector{},
				fmt.Errorf(
					"sample %q produced invalid feature %s=%v",
					sample.RequestID,
					names[index],
					value,
				)
		}
	}

	return ProfilingFeatureVector{
		SchemaVersion: ProfilingFeatureVectorSchemaVersion,

		RequestID: sample.RequestID,

		FunctionName: sample.FunctionName,

		MachineTag: sample.MachineTag,

		FunctionConfiguration: sample.FunctionConfiguration,

		Features: features,
	}, nil
}

func finitePositive(
	value float64,
) bool {
	return value > 0 &&
		!math.IsNaN(
			value,
		) &&
		!math.IsInf(
			value,
			0,
		)
}
