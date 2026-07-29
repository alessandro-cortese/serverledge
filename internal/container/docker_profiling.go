package container

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

const dockerStatsTimeout = 2 * time.Second

// GetResourceSnapshot reads one non-streaming Docker statistics sample for a
// running container.
func (cf *DockerFactory) GetResourceSnapshot(
	contID ContainerID,
) (profiling.ResourceSnapshot, error) {
	ctx, cancel :=
		context.WithTimeout(
			cf.ctx,
			dockerStatsTimeout,
		)

	defer cancel()

	statsReader, err :=
		cf.cli.ContainerStatsOneShot(
			ctx,
			contID,
		)

	if err != nil {
		return profiling.ResourceSnapshot{},
			fmt.Errorf(
				"failed to read Docker stats for container %s: %w",
				contID,
				err,
			)
	}

	defer statsReader.Body.Close()

	var stats dockercontainer.StatsResponse

	if err :=
		json.NewDecoder(
			statsReader.Body,
		).
			Decode(
				&stats,
			); err != nil {

		return profiling.ResourceSnapshot{},
			fmt.Errorf(
				"failed to decode Docker stats for container %s: %w",
				contID,
				err,
			)
	}

	return resourceSnapshotFromDockerStats(
		statsReader.OSType,
		stats,
	), nil
}

func resourceSnapshotFromDockerStats(
	osType string,
	stats dockercontainer.StatsResponse,
) profiling.ResourceSnapshot {
	pageFaults, pageFaultsAvailable :=
		dockerMemoryStat(
			stats.MemoryStats.Stats,
			"total_pgfault",
			"pgfault",
		)

	majorPageFaults, majorPageFaultsAvailable :=
		dockerMemoryStat(
			stats.MemoryStats.Stats,
			"total_pgmajfault",
			"pgmajfault",
		)

	blockReadBytes, blockWriteBytes :=
		aggregateDockerBlockIO(
			stats.BlkioStats.
				IoServiceBytesRecursive,
		)

	blockReadOps, blockWriteOps :=
		aggregateDockerBlockIO(
			stats.BlkioStats.
				IoServicedRecursive,
		)

	blockIOBytesAvailable :=
		len(
			stats.BlkioStats.
				IoServiceBytesRecursive,
		) > 0

	blockIOOpsAvailable :=
		len(
			stats.BlkioStats.
				IoServicedRecursive,
		) > 0

	networkRxBytes, networkTxBytes :=
		aggregateDockerNetworkIO(
			stats.Networks,
		)

	return profiling.ResourceSnapshot{
		OSType: osType,
		ReadAt: stats.Read,

		CPUUsageTotalNs: stats.CPUStats.
			CPUUsage.
			TotalUsage,

		CPUUsageUserNs: stats.CPUStats.
			CPUUsage.
			UsageInUsermode,

		CPUUsageKernelNs: stats.CPUStats.
			CPUUsage.
			UsageInKernelmode,

		CPUThrottledTimeNs: stats.CPUStats.
			ThrottlingData.
			ThrottledTime,

		CPUThrottledPeriods: stats.CPUStats.
			ThrottlingData.
			ThrottledPeriods,

		OnlineCPUs: stats.CPUStats.
			OnlineCPUs,

		MemoryUsageBytes: stats.MemoryStats.
			Usage,

		MemoryLimitBytes: stats.MemoryStats.
			Limit,

		PageFaults: pageFaults,

		MajorPageFaults: majorPageFaults,

		PageFaultsAvailable: pageFaultsAvailable,

		MajorPageFaultsAvailable: majorPageFaultsAvailable,

		BlockReadBytes: blockReadBytes,

		BlockWriteBytes: blockWriteBytes,

		BlockReadOps: blockReadOps,

		BlockWriteOps: blockWriteOps,

		BlockIOBytesAvailable: blockIOBytesAvailable,

		BlockIOOpsAvailable: blockIOOpsAvailable,

		NetworkRxBytes: networkRxBytes,

		NetworkTxBytes: networkTxBytes,

		NetworkAvailable: len(
			stats.Networks,
		) > 0,

		PIDs: stats.PidsStats.
			Current,
	}
}

func dockerMemoryStat(
	stats map[string]uint64,
	keys ...string,
) (uint64, bool) {
	for _, key := range keys {

		value, ok :=
			stats[key]

		if ok {
			return value, true
		}
	}

	return 0, false
}

func aggregateDockerBlockIO(
	entries []dockercontainer.BlkioStatEntry,
) (
	read uint64,
	write uint64,
) {
	for _, entry := range entries {

		switch strings.ToLower(
			entry.Op,
		) {
		case "read":
			read +=
				entry.Value

		case "write":
			write +=
				entry.Value
		}
	}

	return read, write
}

func aggregateDockerNetworkIO(
	networks map[string]dockercontainer.NetworkStats,
) (
	rx uint64,
	tx uint64,
) {
	for _, network := range networks {

		rx +=
			network.RxBytes

		tx +=
			network.TxBytes
	}

	return rx, tx
}
