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
		err := container.ExecuteProfiled(
		cont.ID,
		&req,
		r.Fun.MaxConcurrency,
	)

	r.ResourceProfile = resourceProfile

	profiling.LogInvocationResourceProfile(
		r.Id(),
		r.Fun.Name,
		node.LocalNode.MachineTag,
		node.LocalNode.Key,
		isWarm,
		resourceProfile,
	)

	if err != nil {
		logs, errLog := container.GetLog(cont.ID)

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
			err,
		)
	}

	if !response.Success {
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
	r.IsWarmStart = isWarm

	r.Duration =
		executionWallTime.Seconds() -
			invocationWait.Seconds()

	r.ResponseTime =
		time.Since(r.Arrival).Seconds()

	// Initializing containers may require invocation retries,
	// adding latency to InitTime.
	r.InitTime =
		initTime +
			invocationWait.Seconds()

	r.ExecutionArea = node.LocalNode.Area
	r.ExecutionNode = node.LocalNode.Key

	completions <- &completionNotification{
		funcName:  r.Fun.Name,
		offloaded: r.offloaded,
		report:    *r.ExecutionReport,
		cont:      cont,
		failed:    false,
	}

	return nil
}
