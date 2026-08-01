package mab

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/function"
)

func UpdateBandit(
	body []byte,
	reqPath string,
	executionTag string,
	decision DecisionRecord,
	feedback ExecutionFeedback,
) (err error) {
	resolved := false

	failureReason :=
		"feedback_processing_failed"

	defer func() {
		if resolved {
			return
		}

		ResolveDecisionWithoutFeedback(
			decision,
			failureReason,
		)
	}()

	if GlobalBanditManager == nil {
		failureReason =
			"bandit_manager_unavailable"

		return fmt.Errorf(
			"bandit manager is not initialized",
		)
	}

	bandit :=
		GlobalBanditManager.
			GetBandit(
				decision.FunctionName,
			)

	var response function.Response

	if err :=
		json.Unmarshal(
			body,
			&response,
		); err != nil {

		failureReason =
			"malformed_response_body"

		return fmt.Errorf(
			"failed to unmarshal response body: %w",
			err,
		)
	}

	pathParts :=
		strings.Split(
			strings.Trim(
				reqPath,
				"/",
			),
			"/",
		)

	if len(pathParts) != 2 ||
		pathParts[0] != "invoke" ||
		pathParts[1] == "" {

		failureReason =
			"invalid_invoke_path"

		return fmt.Errorf(
			"could not extract function name from URL: %s",
			reqPath,
		)
	}

	functionName :=
		pathParts[1]

	if functionName !=
		decision.FunctionName {

		failureReason =
			"function_mismatch"

		return fmt.Errorf(
			"function mismatch: decision is for %q, response path is for %q",
			decision.FunctionName,
			functionName,
		)
	}

	report :=
		response.ExecutionReport

	feedback.DurationMs =
		report.Duration * 1000.0

	feedback.ResponseTimeMs =
		report.ResponseTime * 1000.0

	feedback.InitTimeMs =
		report.InitTime * 1000.0

	feedback.QueueingTimeMs =
		report.QueueingTime * 1000.0

	feedback.OffloadLatencyMs =
		report.OffloadLatency * 1000.0

	feedback.IsWarmStart =
		report.IsWarmStart

	feedback.ExecutionNode =
		report.ExecutionNode

	if feedback.NodeName == "" {
		feedback.NodeName =
			feedback.ExecutionNode
	}

	if feedback.ExecutionNode != "" &&
		feedback.NodeName != "" &&
		feedback.ExecutionNode !=
			feedback.NodeName {

		log.Printf(
			"[MAB] event=execution_node_mismatch function=%s node_name=%s execution_node=%s",
			functionName,
			feedback.NodeName,
			feedback.ExecutionNode,
		)
	}

	if executionTag == "" {
		failureReason =
			"missing_execution_tag"

		return fmt.Errorf(
			"Serverledge-Node-Tag header missing",
		)
	}

	logMABExecutionTiming(
		string(
			bandit.GetType(),
		),
		functionName,
		executionTag,
		feedback,
	)

	if !ResolveDecisionWithFeedback(
		decision,
		executionTag,
		feedback,
	) {
		failureReason =
			"decision_resolution_failed"

		return fmt.Errorf(
			"failed to resolve MAB decision for function %q",
			functionName,
		)
	}

	resolved = true

	if report.Duration <= 0 {
		return fmt.Errorf(
			"invalid execution duration: %f",
			report.Duration,
		)
	}

	return nil
}
