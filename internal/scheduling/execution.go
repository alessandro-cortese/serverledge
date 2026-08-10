package scheduling

import (
	"fmt"
	"log"
	"time"

	"github.com/serverledge-faas/serverledge/internal/container"
	"github.com/serverledge-faas/serverledge/internal/executor"
	"github.com/serverledge-faas/serverledge/internal/node"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

const HANDLER_DIR = "/app"

// Execute serves a request on the specified container.
func Execute(
	cont *container.Container,
	r *scheduledRequest,
	isWarm bool,
) error {
	log.Printf(
		"[%s] Executing on container: %v",
		r.Fun,
		cont.ID,
	)

	var req executor.InvocationRequest

	if r.Fun.Runtime == container.CUSTOM_RUNTIME {
		req = executor.InvocationRequest{
			Params:       r.Params,
			ReturnOutput: r.ReturnOutput,
		}
	} else {
		cmd := container.RuntimeToInfo[r.Fun.Runtime].InvocationCmd

		req = executor.InvocationRequest{
			Command:      cmd,
			Params:       r.Params,
			Handler:      r.Fun.Handler,
			HandlerDir:   HANDLER_DIR,
			ReturnOutput: r.ReturnOutput,
		}
	}

	t0 := time.Now()

	initTime := t0.Sub(r.Arrival).Seconds()

	response,
		invocationWait,
		executionWallTime,
		resourceProfile,
		nodeEnvironment,
		executionErr :=
		container.ExecuteProfiled(
			cont,
			&req,
			r.Fun.MaxConcurrency,
			isWarm,
		)

	// Record timing and placement for both successful and failed invocations.
	// This keeps the raw dataset auditable.
	r.ResourceProfile = resourceProfile
	r.IsWarmStart = isWarm

	r.Duration =
		executionWallTime.Seconds() -
			invocationWait.Seconds()

	r.ResponseTime =
		time.Since(r.Arrival).Seconds()

	// Initializing containers may require invocation retries. InitTime is
	// retained even when cold samples are excluded from warm clustering.
	r.InitTime =
		initTime +
			invocationWait.Seconds()

	r.ExecutionArea = node.LocalNode.Area
	r.ExecutionNode = node.LocalNode.Key

	executionSucceeded :=
		executionErr == nil &&
			response != nil &&
			response.Success

	executionErrorText := ""

	switch {
	case executionErr != nil:
		executionErrorText = executionErr.Error()

	case response == nil:
		executionErrorText = "missing_invocation_result"

	case !response.Success:
		executionErrorText = "function_execution_failed"
	}

	profiling.LogInvocationResourceProfile(
		r.Id(),
		r.Fun.Name,
		node.LocalNode.MachineTag,
		node.LocalNode.Key,
		isWarm,
		resourceProfile,
	)

	sampleTiming :=
		profiling.InvocationTiming{}

	if isWarm {
		sampleTiming.DurationMs =
			r.Duration * 1000.0

		sampleTiming.ResponseTimeMs =
			r.ResponseTime * 1000.0

		sampleTiming.QueueingTimeMs =
			r.QueueingTime * 1000.0

		sampleTiming.OffloadLatencyMs =
			r.OffloadLatency * 1000.0

		sampleTiming.InvocationWaitMs =
			float64(invocationWait) /
				float64(time.Millisecond)

		sampleTiming.ExecutionWallTimeMs =
			float64(executionWallTime) /
				float64(time.Millisecond)
	} else {
		// Per le invocazioni cold conserviamo soltanto il tempo
		// di inizializzazione, come stabilito per la profilazione.
		sampleTiming.InitTimeMs =
			r.InitTime * 1000.0
	}

	sample :=
		profiling.BuildInvocationSample(
			profiling.InvocationSampleInput{
				RequestID:    r.Id(),
				FunctionName: r.Fun.Name,
				MachineTag:   node.LocalNode.MachineTag,
				NodeName:     node.LocalNode.Key,
				ContainerID:  cont.ID,

				WarmStart:          isWarm,
				ExecutionSucceeded: executionSucceeded,
				ExecutionError:     executionErrorText,

				Timing:          sampleTiming,
				Profile:         resourceProfile,
				NodeEnvironment: nodeEnvironment,
			},
		)

	// The completion notification is sent before the deferred export runs.
	// Consequently a filesystem write cannot keep the container unavailable.
	defer func() {
		if err :=
			profiling.ExportInvocationSample(
				sample,
			); err != nil {

			log.Printf(
				"[PROFILING] event=sample_export_failed request_id=%q function=%s error=%q",
				r.Id(),
				r.Fun.Name,
				err.Error(),
			)
		}
	}()

	if executionErr != nil {
		logs, errLog :=
			container.GetLog(
				cont.ID,
			)

		if errLog == nil {
			fmt.Println(logs)
		} else {
			fmt.Printf(
				"Failed to get log: %v\n",
				errLog,
			)
		}

		completions <- &completionNotification{
			funcName:  r.Fun.Name,
			offloaded: r.offloaded,
			cont:      cont,
			failed:    true,
		}

		return fmt.Errorf(
			"[%s] Execution failed on container %v: %v ",
			r,
			cont.ID,
			executionErr,
		)
	}

	if response == nil ||
		!response.Success {

		completions <- &completionNotification{
			funcName:  r.Fun.Name,
			offloaded: r.offloaded,
			cont:      cont,
			failed:    true,
		}

		return fmt.Errorf(
			"[%s] Function execution failed %v",
			r,
			cont.ID,
		)
	}

	r.Result = response.Result
	r.Output = response.Output

	completions <- &completionNotification{
		funcName:  r.Fun.Name,
		offloaded: r.offloaded,
		report:    *r.ExecutionReport,
		cont:      cont,
		failed:    false,
	}

	return nil
}
