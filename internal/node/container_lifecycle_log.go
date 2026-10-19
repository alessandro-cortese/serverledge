package node

import (
	"log"
	"os"
	"strings"
	"time"

	"github.com/serverledge-faas/serverledge/internal/container"
	"github.com/serverledge-faas/serverledge/internal/function"
)

const containerLifecycleLogEnv = "SERVERLEDGE_CONTAINER_LIFECYCLE_LOG"
const containerLifecycleLogPrefix = "[CONTAINER_LIFECYCLE]"

// containerLifecycleLoggingEnabled keeps the experimental instrumentation
// completely opt-in.
//
// Accepted values:
//
//	1, true, yes, on
//
// When disabled, the lifecycle helpers preserve the legacy behaviour and do
// not emit additional log lines.
func containerLifecycleLoggingEnabled() bool {
	value := strings.ToLower(
		strings.TrimSpace(
			os.Getenv(containerLifecycleLogEnv),
		),
	)

	switch value {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// logContainerCreated records the successful creation of a container.
//
// startAsIdle distinguishes containers created explicitly for the warm pool
// (prewarm) from containers created on demand by a cold-start request.
func logContainerCreated(
	cont *container.Container,
	fun *function.Function,
	startAsIdle bool,
) {
	if !containerLifecycleLoggingEnabled() {
		return
	}

	if cont == nil || fun == nil {
		return
	}

	creationKind := "cold_start"
	initialState := "busy"

	if startAsIdle {
		creationKind = "prewarm"
		initialState = "warm"
	}

	log.Printf(
		"%s event=container_created "+
			"ts=%d "+
			"container_id=%q "+
			"function=%q "+
			"arch=%q "+
			"machine_tag=%q "+
			"node=%q "+
			"memory_mb=%d "+
			"creation_kind=%q "+
			"initial_state=%q",
		containerLifecycleLogPrefix,
		time.Now().UnixMilli(),
		cont.ID,
		fun.Name,
		LocalNode.Arch,
		LocalNode.MachineTag,
		LocalNode.Key,
		fun.MemoryMB,
		creationKind,
		initialState,
	)
}

// destroyContainerWithLifecycle wraps the legacy container.Destroy call.
//
// With lifecycle logging disabled, this is intentionally equivalent to:
//
//	container.Destroy(id)
//
// With logging enabled, the function reads the configured container memory
// before destruction and emits the destruction event only after a successful
// destroy.
func destroyContainerWithLifecycle(
	id container.ContainerID,
	functionName string,
	reason string,
) error {
	if !containerLifecycleLoggingEnabled() {
		return container.Destroy(id)
	}

	memoryMB, memoryErr := container.GetMemoryMB(id)
	memoryLookupOK := memoryErr == nil

	if memoryErr != nil {
		memoryMB = -1
	}

	err := container.Destroy(id)

	if err != nil {
		log.Printf(
			"%s event=container_destroy_failed "+
				"ts=%d "+
				"container_id=%q "+
				"function=%q "+
				"arch=%q "+
				"machine_tag=%q "+
				"node=%q "+
				"memory_mb=%d "+
				"memory_lookup_ok=%t "+
				"reason=%q "+
				"error=%q",
			containerLifecycleLogPrefix,
			time.Now().UnixMilli(),
			id,
			functionName,
			LocalNode.Arch,
			LocalNode.MachineTag,
			LocalNode.Key,
			memoryMB,
			memoryLookupOK,
			reason,
			err.Error(),
		)

		return err
	}

	log.Printf(
		"%s event=container_destroyed "+
			"ts=%d "+
			"container_id=%q "+
			"function=%q "+
			"arch=%q "+
			"machine_tag=%q "+
			"node=%q "+
			"memory_mb=%d "+
			"memory_lookup_ok=%t "+
			"reason=%q",
		containerLifecycleLogPrefix,
		time.Now().UnixMilli(),
		id,
		functionName,
		LocalNode.Arch,
		LocalNode.MachineTag,
		LocalNode.Key,
		memoryMB,
		memoryLookupOK,
		reason,
	)

	return nil
}

// containerPoolFunctionName returns the function owning a pool.
//
// Callers in pool.go already hold LocalResources' lock in the paths where this
// helper is used, therefore this helper deliberately does not acquire another
// lock.
func containerPoolFunctionName(target *ContainerPool) string {
	for functionName, pool := range LocalResources.containerPools {
		if pool == target {
			return functionName
		}
	}

	return ""
}
