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
	tag string,
	reqID string,
	feedback ExecutionFeedback,
) error { // Read the body
	// Modified function to use machineTag parameter instead of ach
	// Parse the body to a Response object
	var response function.Response
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("failed to unmarshal response body: %v", err)
	}

	report := response.ExecutionReport

	feedback.DurationMs = report.Duration * 1000.0

	feedback.ResponseTimeMs = report.ResponseTime * 1000.0

	feedback.InitTimeMs = report.InitTime * 1000.0

	feedback.QueueingTimeMs = report.QueueingTime * 1000.0

	feedback.OffloadLatencyMs = report.OffloadLatency * 1000.0

	feedback.IsWarmStart = report.IsWarmStart

	feedback.ExecutionNode = report.ExecutionNode

	if feedback.NodeName == "" {
		feedback.NodeName =
			feedback.ExecutionNode
	}

	// get the url of the request, to extract the function name, so that we can update the related MAB.
	pathParts := strings.Split(reqPath, "/")
	if len(pathParts) < 3 || pathParts[len(pathParts)-2] != "invoke" {
		return fmt.Errorf("could not extract function name from URL: %s", reqPath)
	}
	functionName := pathParts[len(pathParts)-1]

	bandit := GlobalBanditManager.GetBandit(functionName)
	ctx := GlobalContextStorage.RetrieveAndDelete(reqID)

	if feedback.ExecutionNode != "" &&
		feedback.NodeName != "" &&
		feedback.ExecutionNode != feedback.NodeName {

		log.Printf(
			"[MAB] event=execution_node_mismatch function=%s node_name=%s execution_node=%s",
			functionName,
			feedback.NodeName,
			feedback.ExecutionNode,
		)
	}

	if tag == "" {
		log.Println("Serverledge-Node-Tag header missing")
		panic(0) // should never happen
	}

	logMABExecutionTiming(
		string(bandit.GetType()),
		functionName,
		tag,
		feedback,
	)

	// Calculate the reward for this execution
	if response.ExecutionReport.Duration <= 0 {
		log.Printf("invalid execution duration: %f", response.ExecutionReport.Duration)
		panic(1) // should never happen
	}

	// Reward = 1 / Duration (we don't consider cold start delay, since we want to focus on architectures' performance)
	// durationMs := response.ExecutionReport.Duration * 1000.0 // s to ms

	// finally update the reward for the bandit. This is thread safe since internally it has a mutex
	bandit.UpdateReward(
		tag,
		ctx,
		feedback,
	)

	return nil
}
