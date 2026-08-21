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

func resourceSnapshotFromDockerStats(osType string, stats dockercontainer.StatsResponse) profiling.ResourceSnapshot {

	pageFaults, pageFaultsAvailable := dockerMemoryStat(stats.MemoryStats.Stats, "total_pgfault", "pgfault")

	// Only the counters consumed by the profiling pipeline are mapped. Docker
	// also reports block I/O, network, PIDs, throttling and memory usage, but
	// none of them is an input of the Random Forest feature vector.
	return profiling.ResourceSnapshot{
		OSType:              osType,
		CPUUsageUserNs:      stats.CPUStats.CPUUsage.UsageInUsermode,
		CPUUsageKernelNs:    stats.CPUStats.CPUUsage.UsageInKernelmode,
		PageFaults:          pageFaults,
		PageFaultsAvailable: pageFaultsAvailable,
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
