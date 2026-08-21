package container

import (
	"testing"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/stretchr/testify/assert"
)

// TestResourceSnapshotFromDockerStats verifies that only the counters consumed
// by the profiling pipeline are mapped, and that everything else reported by
// Docker is deliberately ignored.
func TestResourceSnapshotFromDockerStats(
	t *testing.T,
) {
	stats :=
		dockercontainer.StatsResponse{
			CPUStats: dockercontainer.CPUStats{
				CPUUsage: dockercontainer.CPUUsage{
					TotalUsage: 1000,

					UsageInUsermode: 700,

					UsageInKernelmode: 300,
				},

				OnlineCPUs: 2,
			},

			MemoryStats: dockercontainer.MemoryStats{
				Usage: 2048,

				Limit: 4096,

				Stats: map[string]uint64{
					"pgfault": 20,

					"pgmajfault": 2,
				},
			},

			BlkioStats: dockercontainer.BlkioStats{
				IoServiceBytesRecursive: []dockercontainer.BlkioStatEntry{
					{
						Op: "Read",

						Value: 100,
					},
				},
			},

			PidsStats: dockercontainer.PidsStats{
				Current: 5,
			},
		}

	snapshot :=
		resourceSnapshotFromDockerStats(
			"linux",
			stats,
		)

	assert.Equal(
		t,
		"linux",
		snapshot.OSType,
	)

	assert.Equal(
		t,
		uint64(700),
		snapshot.CPUUsageUserNs,
	)

	assert.Equal(
		t,
		uint64(300),
		snapshot.CPUUsageKernelNs,
	)

	assert.Equal(
		t,
		uint64(20),
		snapshot.PageFaults,
	)

	assert.True(
		t,
		snapshot.PageFaultsAvailable,
	)
}

// TestResourceSnapshotWithoutPageFaultCounters covers the cgroup configuration
// in which Docker does not expose page-fault counters at all.
func TestResourceSnapshotWithoutPageFaultCounters(
	t *testing.T,
) {
	snapshot :=
		resourceSnapshotFromDockerStats(
			"linux",
			dockercontainer.StatsResponse{
				MemoryStats: dockercontainer.MemoryStats{
					Stats: map[string]uint64{},
				},
			},
		)

	assert.False(
		t,
		snapshot.PageFaultsAvailable,
	)

	assert.Equal(
		t,
		uint64(0),
		snapshot.PageFaults,
	)
}

func TestDockerMemoryStatPrefersTotalCgroupV1Counter(
	t *testing.T,
) {
	value, ok :=
		dockerMemoryStat(
			map[string]uint64{
				"total_pgfault": 100,

				"pgfault": 20,
			},
			"total_pgfault",
			"pgfault",
		)

	assert.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		uint64(100),
		value,
	)
}
