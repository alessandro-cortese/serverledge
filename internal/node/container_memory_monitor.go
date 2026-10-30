package node

import (
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/serverledge-faas/serverledge/internal/container"
)

const (
	memoryMonitorEnv    = "SERVERLEDGE_CONTAINER_MEMORY_SAMPLE_MS"
	memoryMonitorPrefix = "[CONTAINER_MEMORY]"
)

// memoryMonitorInterval returns the configured frequency
//
// 0 or variable omitted: monitoring disabled
// Intervals between 100 and 10,000 milliseconds are permitted
func memoryMonitorInterval() time.Duration {
	value := strings.TrimSpace(os.Getenv(memoryMonitorEnv))

	if value == "" || value == "0" {
		return 0
	}

	ms, err := strconv.Atoi(value)

	if err != nil || ms < 100 || ms > 10000 {
		log.Printf(
			"%s event=invalid_config variable=%q value=%q",
			memoryMonitorPrefix,
			memoryMonitorEnv,
			value,
		)
		return 0
	}

	return time.Duration(ms) * time.Millisecond
}

// A monitor is associated with a single container
type containerMemoryMonitor struct {
	id           container.ContainerID
	functionName string
	machineTag   string
	interval     time.Duration
	stop         chan struct{}
	done         chan struct{}
}

// Monitor currently running
var runningMemoryMonitors sync.Map

// Monitors are stopped before they are destroyed
// They are temporarily retained to handle any errors
var stoppedMemoryMonitors sync.Map

// startContainerMemoryMonitor starts monitoring the container
//
// Statistics are read in a goroutine
// No external process is created
func startContainerMemoryMonitor(
	id container.ContainerID,
	functionName string,
) {
	interval := memoryMonitorInterval()

	if interval <= 0 || id == "" || functionName == "" {
		return
	}

	monitor := &containerMemoryMonitor{
		id:           id,
		functionName: functionName,
		machineTag:   LocalNode.MachineTag,
		interval:     interval,
		stop:         make(chan struct{}),
		done:         make(chan struct{}),
	}

	_, alreadyPresent := runningMemoryMonitors.LoadOrStore(
		id,
		monitor,
	)

	if alreadyPresent {
		log.Printf(
			"%s event=duplicate_start container_id=%q",
			memoryMonitorPrefix,
			id,
		)
		return
	}

	go monitor.run()
}

// takeSample reads from memory using the Docker APIs already available via the Serverledge Factory
func (m *containerMemoryMonitor) takeSample(reason string) {
	started := time.Now()

	snapshot, err := container.GetFactory().GetResourceSnapshot(m.id)

	timestamp := time.Now().UnixMilli()
	overhead := time.Since(started).Milliseconds()

	if err != nil {
		log.Printf(
			"%s event=sample_error ts=%d "+
				"container_id=%q function=%q machine_tag=%q "+
				"reason=%q overhead_ms=%d error=%q",
			memoryMonitorPrefix,
			timestamp,
			m.id,
			m.functionName,
			m.machineTag,
			reason,
			overhead,
			err.Error(),
		)
		return
	}

	// Zero could indicate that a sample is unavailable
	// I do not include it silently in the integral calculation
	if snapshot.MemoryUsageBytes == 0 {
		log.Printf(
			"%s event=sample_invalid ts=%d "+
				"container_id=%q function=%q machine_tag=%q "+
				"reason=%q overhead_ms=%d",
			memoryMonitorPrefix,
			timestamp,
			m.id,
			m.functionName,
			m.machineTag,
			reason,
			overhead,
		)
		return
	}

	log.Printf(
		"%s event=memory_sample ts=%d "+
			"container_id=%q function=%q machine_tag=%q "+
			"reason=%q "+
			"usage_bytes=%d limit_bytes=%d peak_bytes=%d "+
			"anon_bytes=%d anon_available=%t "+
			"file_bytes=%d file_available=%t "+
			"kernel_bytes=%d kernel_available=%t "+
			"inactive_file_bytes=%d inactive_file_available=%t "+
			"overhead_ms=%d",

		memoryMonitorPrefix,
		timestamp,
		m.id,
		m.functionName,
		m.machineTag,
		reason,

		snapshot.MemoryUsageBytes,
		snapshot.MemoryLimitBytes,
		snapshot.MemoryPeakBytes,

		snapshot.MemoryAnonBytes,
		snapshot.MemoryAnonAvailable,

		snapshot.MemoryFileBytes,
		snapshot.MemoryFileAvailable,

		snapshot.MemoryKernelBytes,
		snapshot.MemoryKernelAvailable,

		snapshot.MemoryInactiveFileBytes,
		snapshot.MemoryInactiveFileAvailable,

		overhead,
	)
}

// run performs an initial measurement, then continues at regular intervals
//
// The monitor remains active even when the container is warm
func (m *containerMemoryMonitor) run() {
	defer close(m.done)

	m.takeSample("start")

	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stop:
			m.takeSample("before_destroy")
			return

		case <-ticker.C:
			m.takeSample("periodic")
		}
	}
}

// stopContainerMemoryMonitor stops monitoring and waits for any final read operation to complete
func stopContainerMemoryMonitor(id container.ContainerID) {
	value, found := runningMemoryMonitors.LoadAndDelete(id)

	if !found {
		return
	}

	monitor := value.(*containerMemoryMonitor)

	close(monitor.stop)

	<-monitor.done

	stoppedMemoryMonitors.Store(id, monitor)

	log.Printf(
		"%s event=monitor_stopped ts=%d "+
			"container_id=%q function=%q machine_tag=%q",
		memoryMonitorPrefix,
		time.Now().UnixMilli(),
		id,
		monitor.functionName,
		monitor.machineTag,
	)
}

// If Docker is unable to destroy a container, monitoring can be restarted
func resumeContainerMemoryMonitor(id container.ContainerID) {
	value, found := stoppedMemoryMonitors.LoadAndDelete(id)

	if !found {
		return
	}

	monitor := value.(*containerMemoryMonitor)

	log.Printf(
		"%s event=monitor_restart_after_destroy_failure "+
			"ts=%d container_id=%q function=%q machine_tag=%q",
		memoryMonitorPrefix,
		time.Now().UnixMilli(),
		id,
		monitor.functionName,
		monitor.machineTag,
	)

	startContainerMemoryMonitor(id, monitor.functionName)
}

// Removes the temporary state after destruction
func forgetStoppedContainerMemoryMonitor(id container.ContainerID) {
	stoppedMemoryMonitors.Delete(id)
}
