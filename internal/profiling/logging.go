package profiling

import (
	"log"
	"time"
)

const logPrefix = "[PROFILING]"

// LogInvocationResourceProfile emits one structured logfmt-style line for
// experiment data collection. The profile remains separate from the MAB state.
func LogInvocationResourceProfile(
	requestID string,
	functionName string,
	machineTag string,
	nodeName string,
	warmStart bool,
	profile *InvocationResourceProfile,
) {
	if profile == nil {
		return
	}

	log.Printf(
		"%s event=invocation_resource_profile ts=%d request_id=%q function=%s machine_tag=%s node=%s container_id=%s warm_start=%t collected=%t valid=%t invalid_reason=%q exclusive_container=%t max_concurrency=%d execution_wall_time_ms=%.6f profiling_start_overhead_ms=%.6f profiling_end_overhead_ms=%.6f profiling_total_overhead_ms=%.6f cpu_total_delta_ns=%d cpu_user_delta_ns=%d cpu_kernel_delta_ns=%d cpu_throttled_time_delta_ns=%d cpu_throttled_periods_delta=%d online_cpus=%d utilized_cpus=%.6f memory_before_bytes=%d memory_after_bytes=%d memory_delta_bytes=%d memory_limit_bytes=%d page_faults_available=%t page_faults_delta=%d major_page_faults_available=%t major_page_faults_delta=%d block_io_available=%t block_io_bytes_available=%t block_io_ops_available=%t block_read_bytes_delta=%d block_write_bytes_delta=%d block_read_ops_delta=%d block_write_ops_delta=%d network_available=%t network_rx_bytes_delta=%d network_tx_bytes_delta=%d pids_before=%d pids_after=%d\n",
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
		profile.MaxConcurrency,
		profile.ExecutionWallTimeMs,
		profile.ProfilingStartOverheadMs,
		profile.ProfilingEndOverheadMs,
		profile.ProfilingTotalOverheadMs,
		profile.CPUUsageTotalDeltaNs,
		profile.CPUUsageUserDeltaNs,
		profile.CPUUsageKernelDeltaNs,
		profile.CPUThrottledTimeDeltaNs,
		profile.CPUThrottledPeriodsDelta,
		profile.OnlineCPUs,
		profile.UtilizedCPUs,
		profile.MemoryUsageBeforeBytes,
		profile.MemoryUsageAfterBytes,
		profile.MemoryUsageDeltaBytes,
		profile.MemoryLimitBytes,
		profile.PageFaultsAvailable,
		profile.PageFaultsDelta,
		profile.MajorPageFaultsAvailable,
		profile.MajorPageFaultsDelta,
		profile.BlockIOAvailable,
		profile.BlockIOBytesAvailable,
		profile.BlockIOOpsAvailable,
		profile.BlockReadBytesDelta,
		profile.BlockWriteBytesDelta,
		profile.BlockReadOpsDelta,
		profile.BlockWriteOpsDelta,
		profile.NetworkAvailable,
		profile.NetworkRxBytesDelta,
		profile.NetworkTxBytesDelta,
		profile.PIDsBefore,
		profile.PIDsAfter,
	)
}
