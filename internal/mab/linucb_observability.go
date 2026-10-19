package mab

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
)

const linUCBObservabilityEnv = "SERVERLEDGE_LINUCB_OBSERVABILITY"

const linUCBObservabilityPrefix = "[LINUCB_OBSERVABILITY]"

// linUCBObservabilityEnabled keeps the additional LinUCB experiment logging
// completely opt-in.
//
// Accepted values:
//
//	1, true, yes, on
//
// When disabled, the helper returns immediately and the normal MAB behaviour
// remains unchanged.
func linUCBObservabilityEnabled() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(linUCBObservabilityEnv)))

	switch value {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// formatLinUCBContext returns a deterministic representation of the complete
// ring-level memory-utilization snapshot used for the routing decision.
//
// The keys are sorted so experiment logs are reproducible and easy to parse.
func formatLinUCBContext(ctx *Context) string {
	if ctx == nil || ctx.ArchMemUsage == nil {
		return ""
	}

	arms := make([]string, 0, len(ctx.ArchMemUsage))

	for arm := range ctx.ArchMemUsage {
		arms = append(arms, arm)
	}

	sort.Strings(arms)

	values := make([]string, 0, len(arms))

	for _, arm := range arms {
		values = append(values, fmt.Sprintf("%s=%.6f", arm, ctx.ArchMemUsage[arm]))
	}

	return strings.Join(values, ",")
}

// linUCBContextHasArm reports whether the decision snapshot explicitly
// contained an utilization value for the requested arm.
//
// This is kept separate from armUtilization because armUtilization deliberately
// maps missing telemetry to zero. For experiment auditing we also want to know
// whether zero was an actual observation or the fallback for missing context.
func linUCBContextHasArm(ctx *Context, arm string) bool {
	if ctx == nil || ctx.ArchMemUsage == nil || arm == "" {
		return false
	}

	_, ok := ctx.ArchMemUsage[arm]

	return ok
}

// logLinUCBRequestObservation records one complete request-level observation.
//
// It deliberately does not modify the LinUCB model and does not calculate a
// second reward. The event only records:
//   - the context snapshot used at decision time;
//   - the selected and actual execution arms;
//   - the execution measurements returned by the node.
//
// Reward values can therefore be reconstructed independently from the recorded
// measurements without coupling experimental logging to the reward formula.
func logLinUCBRequestObservation(policy BanditType, decision DecisionRecord, feedback ExecutionFeedback) {
	if !linUCBObservabilityEnabled() {
		return
	}

	if policy != LinUCB {
		return
	}

	contextPresent := decision.Context != nil && decision.Context.ArchMemUsage != nil

	selectedContextPresent := linUCBContextHasArm(decision.Context, decision.SelectedArm)
	executionContextPresent := linUCBContextHasArm(decision.Context, decision.ExecutionArm)

	// These are the exact utilization values LinUCB would use after applying
	// the existing missing-context and clamping semantics.
	selectedUtilization := armUtilization(decision.Context, decision.SelectedArm)
	executionUtilization := armUtilization(decision.Context, decision.ExecutionArm)

	log.Printf(
		"%s event=request_observation "+
			"request_id=%q "+
			"policy=%s "+
			"function=%q "+
			"selected_arm=%q "+
			"execution_arm=%q "+
			"fallback=%t "+
			"fallback_reason=%q "+
			"context_present=%t "+
			"context=%q "+
			"selected_context_present=%t "+
			"selected_utilization=%.6f "+
			"execution_context_present=%t "+
			"execution_utilization=%.6f "+
			"duration_ms=%.6f "+
			"response_time_ms=%.6f "+
			"init_time_ms=%.6f "+
			"queueing_time_ms=%.6f "+
			"offload_latency_ms=%.6f "+
			"warm_start=%t "+
			"node_name=%q "+
			"execution_node=%q",
		linUCBObservabilityPrefix,
		decision.RequestID,
		policy,
		decision.FunctionName,
		decision.SelectedArm,
		decision.ExecutionArm,
		decision.Fallback,
		decision.FallbackReason,
		contextPresent,
		formatLinUCBContext(decision.Context),
		selectedContextPresent,
		selectedUtilization,
		executionContextPresent,
		executionUtilization,
		feedback.DurationMs,
		feedback.ResponseTimeMs,
		feedback.InitTimeMs,
		feedback.QueueingTimeMs,
		feedback.OffloadLatencyMs,
		feedback.IsWarmStart,
		feedback.NodeName,
		feedback.ExecutionNode,
	)
}
