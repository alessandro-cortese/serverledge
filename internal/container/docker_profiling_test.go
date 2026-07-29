package container

import (
	"testing"

	dockercontainer "github.com/docker/docker/api/types/container"
	"github.com/stretchr/testify/assert"
)

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

				ThrottlingData: dockercontainer.ThrottlingData{
					ThrottledTime: 50,

					ThrottledPeriods: 4,
				},
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
					{
						Op: "Write",

						Value: 200,
					},
					{
						Op: "Total",

						Value: 300,
					},
				},

				IoServicedRecursive: []dockercontainer.BlkioStatEntry{
					{
						Op: "read",

						Value: 3,
					},
					{
						Op: "write",

						Value: 4,
					},
				},
			},

			Networks: map[string]dockercontainer.NetworkStats{
				"eth0": {
					RxBytes: 400,

					TxBytes: 500,
				},

				"eth1": {
					RxBytes: 40,

					TxBytes: 50,
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
		uint64(1000),
		snapshot.CPUUsageTotalNs,
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
		uint32(2),
		snapshot.OnlineCPUs,
	)

	assert.Equal(
		t,
		uint64(2048),
		snapshot.MemoryUsageBytes,
	)

	assert.Equal(
		t,
		uint64(20),
		snapshot.PageFaults,
	)

	assert.Equal(
		t,
		uint64(2),
		snapshot.MajorPageFaults,
	)

	assert.True(
		t,
		snapshot.PageFaultsAvailable,
	)

	assert.Equal(
		t,
		uint64(100),
		snapshot.BlockReadBytes,
	)

	assert.Equal(
		t,
		uint64(200),
		snapshot.BlockWriteBytes,
	)

	assert.Equal(
		t,
		uint64(3),
		snapshot.BlockReadOps,
	)

	assert.Equal(
		t,
		uint64(4),
		snapshot.BlockWriteOps,
	)

	assert.True(
		t,
		snapshot.BlockIOBytesAvailable,
	)

	assert.True(
		t,
		snapshot.BlockIOOpsAvailable,
	)

	assert.Equal(
		t,
		uint64(440),
		snapshot.NetworkRxBytes,
	)

	assert.Equal(
		t,
		uint64(550),
		snapshot.NetworkTxBytes,
	)

	assert.Equal(
		t,
		uint64(5),
		snapshot.PIDs,
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

func TestResourceSnapshotDistinguishesBlockIOBytesFromOperations(
	t *testing.T,
) {
	stats := dockercontainer.StatsResponse{
		BlkioStats: dockercontainer.BlkioStats{
			IoServiceBytesRecursive: []dockercontainer.BlkioStatEntry{
				{
					Op:    "Write",
					Value: 32 * 1024 * 1024,
				},
			},
		},
	}

	snapshot :=
		resourceSnapshotFromDockerStats(
			"linux",
			stats,
		)

	assert.True(
		t,
		snapshot.BlockIOBytesAvailable,
	)

	assert.False(
		t,
		snapshot.BlockIOOpsAvailable,
	)

	assert.Equal(
		t,
		uint64(32*1024*1024),
		snapshot.BlockWriteBytes,
	)

	assert.Zero(
		t,
		snapshot.BlockWriteOps,
	)
}
