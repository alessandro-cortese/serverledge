package container

import (
	"fmt"
	"time"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/executor"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

// ExecuteProfiled invokes a function and, when profiling is enabled, surrounds
// the invocation with two Docker statistics snapshots.
//
// Profiling failures never prevent the function from running. They are reported
// through an invalid InvocationResourceProfile instead.
func ExecuteProfiled(
	cont *Container,
	req *executor.InvocationRequest,
	maxConcurrency int16,
) (
	*executor.InvocationResult,
	time.Duration,
	time.Duration,
	*profiling.InvocationResourceProfile,
	error,
) {
	if cont == nil {
		return nil,
			0,
			0,
			nil,
			fmt.Errorf("cannot execute on a nil container")
	}

	contID := cont.ID

	if !config.GetBool(
		config.FUNCTION_PROFILING_ENABLED,
		false,
	) {
		executionStartedAt :=
			time.Now()

		response, invocationWait, err :=
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

	startSnapshotStartedAt :=
		time.Now()

	before, beforeErr :=
		cf.GetResourceSnapshot(
			contID,
		)

	startOverhead :=
		time.Since(
			startSnapshotStartedAt,
		)

	executionStartedAt :=
		time.Now()

	response, invocationWait, executionErr :=
		Execute(
			contID,
			req,
		)

	executionWallTime :=
		time.Since(
			executionStartedAt,
		)

	var after profiling.ResourceSnapshot
	var afterErr error
	var endOverhead time.Duration

	if beforeErr == nil {
		endSnapshotStartedAt :=
			time.Now()

		after, afterErr =
			cf.GetResourceSnapshot(
				contID,
			)

		endOverhead =
			time.Since(
				endSnapshotStartedAt,
			)
	}

	if beforeErr != nil {
		return response,
			invocationWait,
			executionWallTime,
			profiling.NewInvalidInvocationResourceProfile(
				contID,
				maxConcurrency,
				true,
				profilingLockWait,
				fmt.Sprintf(
					"snapshot_before_failed: %v",
					beforeErr,
				),
				startOverhead,
				endOverhead,
			),
			executionErr
	}

	if afterErr != nil {
		return response,
			invocationWait,
			executionWallTime,
			profiling.NewInvalidInvocationResourceProfile(
				contID,
				maxConcurrency,
				true,
				profilingLockWait,
				fmt.Sprintf(
					"snapshot_after_failed: %v",
					afterErr,
				),
				startOverhead,
				endOverhead,
			),
			executionErr
	}

	profile :=
		profiling.BuildInvocationResourceProfile(
			contID,
			maxConcurrency,
			true,
			profilingLockWait,
			before,
			after,
			executionWallTime,
			startOverhead,
			endOverhead,
		)

	return response,
		invocationWait,
		executionWallTime,
		profile,
		executionErr
}
