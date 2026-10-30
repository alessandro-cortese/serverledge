package container

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

const dockerStatsTimeout = 2 * time.Second

// GetResourceSnapshot reads one non-streaming Docker statistics sample for a
// running container.
func (cf *DockerFactory) GetResourceSnapshot(contID ContainerID) (profiling.ResourceSnapshot, error) {

	ctx, cancel := context.WithTimeout(cf.ctx, dockerStatsTimeout)
	defer cancel()
	statsReader, err := cf.cli.ContainerStatsOneShot(ctx, contID)
	if err != nil {
		return profiling.ResourceSnapshot{}, fmt.Errorf("failed to read Docker stats for container %s: %w", contID, err)
	}

	defer statsReader.Body.Close()
	var stats dockercontainer.StatsResponse
	if err := json.NewDecoder(statsReader.Body).Decode(&stats); err != nil {
		return profiling.ResourceSnapshot{}, fmt.Errorf("failed to decode Docker stats for container %s: %w", contID, err)
	}

	return resourceSnapshotFromDockerStats(statsReader.OSType, stats), nil
}

// resourceSnapshotFromDockerStats extracts Docker counters.
//
// It retains the CPU and page fault counters already used by profiling
// and adds absolute information about the container’s memory.
//
// The new counters are not used to modify MAB decisions.
func resourceSnapshotFromDockerStats(osType string, stats dockercontainer.StatsResponse) profiling.ResourceSnapshot {

	pageFaults, pageFaultsAvailable := dockerMemoryStat(
		stats.MemoryStats.Stats,
		"total_pgfault",
		"pgfault",
	)

	// Anonymous memory: compatibility with cgroup keys v1/v2
	anon, hasAnon := dockerMemoryStat(
		stats.MemoryStats.Stats,
		"anon",
		"total_rss",
		"rss",
	)

	// Memory associated with files and the cache
	file, hasFile := dockerMemoryStat(
		stats.MemoryStats.Stats,
		"file",
		"total_cache",
		"cache",
	)

	// Kernel memory, where available
	kernel, hasKernel := dockerMemoryStat(
		stats.MemoryStats.Stats,
		"kernel",
	)

	// Cache inactive
	inactiveFile, hasInactiveFile := dockerMemoryStat(
		stats.MemoryStats.Stats,
		"inactive_file",
		"total_inactive_file",
	)

	return profiling.ResourceSnapshot{
		OSType: osType,

		CPUUsageUserNs:   stats.CPUStats.CPUUsage.UsageInUsermode,
		CPUUsageKernelNs: stats.CPUStats.CPUUsage.UsageInKernelmode,

		PageFaults:          pageFaults,
		PageFaultsAvailable: pageFaultsAvailable,

		// Memory currently accounted for by the cgroup
		MemoryUsageBytes: stats.MemoryStats.Usage,

		// Configured limit
		MemoryLimitBytes: stats.MemoryStats.Limit,

		// This may not be available in some versions of cgroup
		MemoryPeakBytes: stats.MemoryStats.MaxUsage,

		MemoryAnonBytes:         anon,
		MemoryFileBytes:         file,
		MemoryKernelBytes:       kernel,
		MemoryInactiveFileBytes: inactiveFile,

		MemoryAnonAvailable:         hasAnon,
		MemoryFileAvailable:         hasFile,
		MemoryKernelAvailable:       hasKernel,
		MemoryInactiveFileAvailable: hasInactiveFile,
	}
}

func dockerMemoryStat(stats map[string]uint64, keys ...string) (uint64, bool) {
	for _, key := range keys {
		value, ok := stats[key]
		if ok {
			return value, true
		}
	}
	return 0, false
}
