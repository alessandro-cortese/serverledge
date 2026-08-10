package container

import (
	"fmt"
	"time"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/executor"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

// ExecuteProfiled invokes a function and, when profiling is enabled, surrounds
// the invocation with Docker/container and procfs/node snapshots.
//
// Profiling failures never prevent the function from running. They are reported
// through invalid or partially collected profiles instead.
func ExecuteProfiled(
	cont *Container,
	req *executor.InvocationRequest,
	maxConcurrency int16,
	collectResourceProfile bool,
) (
	*executor.InvocationResult,
	time.Duration,
	time.Duration,
	*profiling.InvocationResourceProfile,
	*profiling.NodeResourceProfile,
	error,
) {
	if cont == nil {
		return nil,
			0,
			0,
			nil,
			nil,
			fmt.Errorf(
				"cannot execute on a nil container",
			)
	}

	contID :=
		cont.ID

	if !config.GetBool(
		config.FUNCTION_PROFILING_ENABLED,
		false,
	) ||
		!collectResourceProfile {

		executionStartedAt :=
			time.Now()

		response,
			invocationWait,
			err :=
			Execute(
				contID,
				req,
			)

		executionWallTime :=
			time.Since(
				executionStartedAt,
			)

		return response,
			invocationWait,
			executionWallTime,
			nil,
			nil,
			err
	}

	profilingLockStartedAt :=
		time.Now()

	cont.profilingMu.Lock()

	profilingLockWait :=
		time.Since(
			profilingLockStartedAt,
		)

	defer cont.profilingMu.Unlock()

	// Lo snapshot del nodo viene eseguito per primo, così il relativo
	// intervallo contiene anche l'intervallo degli snapshot del container.
	// Entrambi gli snapshot finali vengono comunque completati prima di
	// rilasciare il mutex di profilazione del container.
	nodeStartSnapshotStartedAt :=
		time.Now()

	nodeBefore,
		nodeBeforeErr :=
		profiling.ReadNodeResourceSnapshot()

	nodeStartOverhead :=
		time.Since(
			nodeStartSnapshotStartedAt,
		)

	containerStartSnapshotStartedAt :=
		time.Now()

	containerBefore,
		containerBeforeErr :=
		cf.GetResourceSnapshot(
			contID,
		)

	containerStartOverhead :=
		time.Since(
			containerStartSnapshotStartedAt,
		)

	executionStartedAt :=
		time.Now()

	response,
		invocationWait,
		executionErr :=
		Execute(
			contID,
			req,
		)

	executionWallTime :=
		time.Since(
			executionStartedAt,
		)

	var containerAfter profiling.ResourceSnapshot
	var containerAfterErr error
	var containerEndOverhead time.Duration

	if containerBeforeErr == nil {
		containerEndSnapshotStartedAt :=
			time.Now()

		containerAfter,
			containerAfterErr =
			cf.GetResourceSnapshot(
				contID,
			)

		containerEndOverhead =
			time.Since(
				containerEndSnapshotStartedAt,
			)
	}

	nodeEndSnapshotStartedAt :=
		time.Now()

	nodeAfter,
		nodeAfterErr :=
		profiling.ReadNodeResourceSnapshot()

	nodeEndOverhead :=
		time.Since(
			nodeEndSnapshotStartedAt,
		)

	var resourceProfile *profiling.InvocationResourceProfile

	switch {
	case containerBeforeErr != nil:
		resourceProfile =
			profiling.NewInvalidInvocationResourceProfile(
				contID,
				maxConcurrency,
				true,
				profilingLockWait,
				fmt.Sprintf(
					"snapshot_before_failed: %v",
					containerBeforeErr,
				),
				containerStartOverhead,
				containerEndOverhead,
			)

	case containerAfterErr != nil:
		resourceProfile =
			profiling.NewInvalidInvocationResourceProfile(
				contID,
				maxConcurrency,
				true,
				profilingLockWait,
				fmt.Sprintf(
					"snapshot_after_failed: %v",
					containerAfterErr,
				),
				containerStartOverhead,
				containerEndOverhead,
			)

	default:
		resourceProfile =
			profiling.BuildInvocationResourceProfile(
				contID,
				maxConcurrency,
				true,
				profilingLockWait,
				containerBefore,
				containerAfter,
				executionWallTime,
				containerStartOverhead,
				containerEndOverhead,
			)
	}

	var nodeProfile *profiling.NodeResourceProfile

	switch {
	case nodeBeforeErr != nil:
		nodeProfile =
			profiling.NewInvalidNodeResourceProfile(
				fmt.Sprintf(
					"node_snapshot_before_failed: %v",
					nodeBeforeErr,
				),
				nodeStartOverhead,
				nodeEndOverhead,
			)

	case nodeAfterErr != nil:
		nodeProfile =
			profiling.NewInvalidNodeResourceProfile(
				fmt.Sprintf(
					"node_snapshot_after_failed: %v",
					nodeAfterErr,
				),
				nodeStartOverhead,
				nodeEndOverhead,
			)

	default:
		nodeProfile =
			profiling.BuildNodeResourceProfile(
				nodeBefore,
				nodeAfter,
				executionWallTime,
				nodeStartOverhead,
				nodeEndOverhead,
			)
	}

	return response,
		invocationWait,
		executionWallTime,
		resourceProfile,
		nodeProfile,
		executionErr
}
