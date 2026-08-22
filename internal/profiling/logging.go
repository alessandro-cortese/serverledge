package profiling

import (
	"log"
	"time"
)

const logPrefix = "[PROFILING]"

// LogInvocationResourceProfile emits one structured logfmt-style line for
// experiment data collection.
//
// The line reports exactly the fields the profile carries: the two CPU deltas
// and the page-fault delta that feed the Random Forest features, the
// container-side profiling overhead, and the validity conditions that decide
// whether the sample is eligible for clustering.
func LogInvocationResourceProfile(requestID string, functionName string, machineTag string, nodeName string, warmStart bool, profile *InvocationResourceProfile) {
	if profile == nil {
		return
	}

	log.Printf(
		"%s event=invocation_resource_profile ts=%d request_id=%q function=%s machine_tag=%s node=%s container_id=%s warm_start=%t collected=%t valid=%t invalid_reason=%q exclusive_container=%t cpu_user_delta_ns=%d cpu_kernel_delta_ns=%d page_faults_available=%t page_faults_delta=%d profiling_start_overhead_ms=%.6f\n",
		logPrefix,
		time.Now().UnixMilli(),
		requestID,
		functionName,
		machineTag,
		nodeName,
		profile.ContainerID,
		warmStart,
		profile.Collected,
		profile.Valid,
		profile.InvalidReason,
		profile.ExclusiveContainer,
		profile.CPUUsageUserDeltaNs,
		profile.CPUUsageKernelDeltaNs,
		profile.PageFaultsAvailable,
		profile.PageFaultsDelta,
		profile.ProfilingStartOverheadMs,
	)
}
