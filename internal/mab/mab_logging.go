package mab

import (
	"log"
	"sort"
	"strings"
	"time"
)

const mabLogPrefix = "[MAB]"

func nowMillis() int64 {
	return time.Now().UnixMilli()
}

func formatArmsFromMap[T any](arms map[string]T) string {
	keys := make([]string, 0, len(arms))
	for arm := range arms {
		keys = append(keys, arm)
	}

	sort.Strings(keys)

	return strings.Join(keys, ",")
}

func logMABArmAdded(policy string, arm string, functionName string, arms string) {
	log.Printf(
		"%s event=arm_added ts=%d policy=%s function=%s arm=%s arms=[%s]\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		arms,
	)
}

func logMABSelectArm(
	policy string,
	functionName string,
	selectedArm string,
	reason string,
	score float64,
	totalCounts int64,
	totalInFlight int64,
	arms string,
) {
	log.Printf(
		"%s event=select_arm ts=%d policy=%s function=%s selected_arm=%s reason=%s score=%.6f total_counts=%d total_in_flight=%d effective_total_counts=%d arms=[%s]\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		selectedArm,
		reason,
		score,
		totalCounts,
		totalInFlight,
		totalCounts+totalInFlight,
		arms,
	)
}

func logMABUCB1ArmScore(
	policy string,
	functionName string,
	arm string,
	score float64,
	explorationBonus float64,
	count int64,
	inFlight int64,
	effectiveCount int64,
	avgReward float64,
	totalCounts int64,
	totalInFlight int64,
	effectiveTotalCounts int64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f count=%d in_flight=%d effective_count=%d avg_reward=%.6f exploration_bonus=%.6f total_counts=%d total_in_flight=%d effective_total_counts=%d contextual=false\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		count,
		inFlight,
		effectiveCount,
		avgReward,
		explorationBonus,
		totalCounts,
		totalInFlight,
		effectiveTotalCounts,
	)
}

func logMABUpdateReward(
	policy string,
	functionName string,
	arm string,
	durationMs float64,
	isWarmStart bool,
	reward float64,
	count int64,
	avgReward float64,
	totalCounts int64,
) {
	log.Printf(
		"%s event=update_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f warm_start=%t reward=%.6f count=%d avg_reward=%.6f total_counts=%d\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		durationMs,
		isWarmStart,
		reward,
		count,
		avgReward,
		totalCounts,
	)
}

func logMABSkipColdStart(
	policy string,
	functionName string,
	arm string,
	durationMs float64,
) {
	log.Printf(
		"%s event=skip_cold_start ts=%d policy=%s function=%s arm=%s duration_ms=%.6f reason=cold_start\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		durationMs,
	)
}

func logMABContextualSelectArm(
	policy string,
	functionName string,
	selectedArm string,
	reason string,
	score float64,
	totalInFlight int64,
	arms string,
) {
	log.Printf(
		"%s event=select_arm ts=%d policy=%s function=%s selected_arm=%s reason=%s score=%.6f total_in_flight=%d arms=[%s] contextual=true\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		selectedArm,
		reason,
		score,
		totalInFlight,
		arms,
	)
}

func logMABContextualUpdateReward(
	policy string,
	functionName string,
	arm string,
	durationMs float64,
	isWarmStart bool,
	utilization float64,
	reward float64,
) {
	log.Printf(
		"%s event=update_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f warm_start=%t utilization=%.6f explicit_utilization_penalty=false reward=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		durationMs,
		isWarmStart,
		utilization,
		reward,
	)
}

func logMABContextualArmScore(
	policy string,
	functionName string,
	arm string,
	score float64,
	expectedReward float64,
	confidence float64,
	utilization float64,
	inFlight int64,
	totalInFlight int64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f expected_reward=%.6f confidence=%.6f contextual=true utilization=%.6f explicit_utilization_penalty=false in_flight=%d total_in_flight=%d\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		expectedReward,
		confidence,
		utilization,
		inFlight,
		totalInFlight,
	)
}

func logMABInFlightChanged(
	policy string,
	functionName string,
	arm string,
	action string,
	inFlight int64,
	totalInFlight int64,
) {
	log.Printf(
		"%s event=in_flight_changed ts=%d policy=%s function=%s arm=%s action=%s in_flight=%d total_in_flight=%d\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		action,
		inFlight,
		totalInFlight,
	)
}

func logMABInFlightIgnored(
	policy string,
	functionName string,
	arm string,
	reason string,
) {
	log.Printf(
		"%s event=in_flight_ignored ts=%d policy=%s function=%s arm=%s reason=%s\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		reason,
	)
}

func logMABDecisionCreated(
	decision DecisionRecord,
) {
	log.Printf(
		"%s event=decision_created ts=%d request_id=%s function=%s selected_arm=%s\n",
		mabLogPrefix,
		nowMillis(),
		decision.RequestID,
		decision.FunctionName,
		decision.SelectedArm,
	)
}

func logMABDecisionPlanned(
	decision DecisionRecord,
) {
	log.Printf(
		"%s event=decision_planned ts=%d request_id=%s function=%s selected_arm=%s execution_arm=%s fallback=%t fallback_reason=%s\n",
		mabLogPrefix,
		nowMillis(),
		decision.RequestID,
		decision.FunctionName,
		decision.SelectedArm,
		decision.ExecutionArm,
		decision.Fallback,
		decision.FallbackReason,
	)
}

func logMABDecisionResolved(
	decision DecisionRecord,
	stats DecisionStatsSnapshot,
) {
	log.Printf(
		"%s event=decision_resolved ts=%d request_id=%s function=%s selected_arm=%s execution_arm=%s fallback=%t fallback_reason=%s direct_executions=%d fallback_executions=%d cancelled_decisions=%d\n",
		mabLogPrefix,
		nowMillis(),
		decision.RequestID,
		decision.FunctionName,
		decision.SelectedArm,
		decision.ExecutionArm,
		decision.Fallback,
		decision.FallbackReason,
		stats.DirectExecutions,
		stats.FallbackExecutions,
		stats.CancelledDecisions,
	)
}

func logMABDecisionCancelled(
	decision DecisionRecord,
	reason string,
	stats DecisionStatsSnapshot,
	banditManagerAvailable bool,
) {
	log.Printf(
		"%s event=decision_cancelled ts=%d request_id=%s function=%s selected_arm=%s execution_arm=%s fallback=%t fallback_reason=%s reason=%s direct_executions=%d fallback_executions=%d cancelled_decisions=%d bandit_manager_available=%t\n",
		mabLogPrefix,
		nowMillis(),
		decision.RequestID,
		decision.FunctionName,
		decision.SelectedArm,
		decision.ExecutionArm,
		decision.Fallback,
		decision.FallbackReason,
		reason,
		stats.DirectExecutions,
		stats.FallbackExecutions,
		stats.CancelledDecisions,
		banditManagerAvailable,
	)
}

func logMABExecutionArmMismatch(
	decision DecisionRecord,
	observedExecutionArm string,
) {
	log.Printf(
		"%s event=execution_arm_mismatch ts=%d request_id=%s function=%s selected_arm=%s planned_execution_arm=%s observed_execution_arm=%s\n",
		mabLogPrefix,
		nowMillis(),
		decision.RequestID,
		decision.FunctionName,
		decision.SelectedArm,
		decision.ExecutionArm,
		observedExecutionArm,
	)
}

func logMABRewardBreakdown(
	policy string,
	functionName string,
	arm string,
	durationMs float64,
	breakdown CostBreakdown,
) {
	info := ParseMachineTag(arm)
	capabilities :=
		strings.Join(
			info.Capabilities,
			",",
		)

	log.Printf(
		"%s event=reward_breakdown ts=%d policy=%s function=%s arm=%s base_tag=%s architecture=%s specialization=%s capabilities=[%s] duration_ms=%.6f latency_reward=%.6f cost_weight=%.6f cost_factor=%.6f cost_term=%.6f energy_weight=%.6f energy_factor=%.6f energy_term=%.6f final_reward=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		info.BaseTag,
		info.Architecture,
		info.Specialization,
		capabilities,
		durationMs,
		breakdown.LatencyReward,
		breakdown.CostWeight,
		breakdown.CostFactor,
		breakdown.CostTerm,
		breakdown.EnergyWeight,
		breakdown.EnergyFactor,
		breakdown.EnergyTerm,
		breakdown.FinalReward,
	)
}

func logMABExecutionTiming(
	policy string,
	functionName string,
	arm string,
	feedback ExecutionFeedback,
) {
	responseMinusDurationMs :=
		feedback.ResponseTimeMs -
			feedback.DurationMs

	log.Printf(
		"%s event=execution_timing ts=%d policy=%s function=%s arm=%s node_name=%s execution_node=%s warm_start=%t response_time_ms=%.6f duration_ms=%.6f init_time_ms=%.6f queueing_time_ms=%.6f offload_latency_ms=%.6f response_minus_duration_ms=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		feedback.NodeName,
		feedback.ExecutionNode,
		feedback.IsWarmStart,
		feedback.ResponseTimeMs,
		feedback.DurationMs,
		feedback.InitTimeMs,
		feedback.QueueingTimeMs,
		feedback.OffloadLatencyMs,
		responseMinusDurationMs,
	)
}

func logMABUseColdStartSample(
	policy string,
	functionName string,
	arm string,
	durationMs float64,
) {
	log.Printf(
		"%s event=use_cold_start_sample ts=%d policy=%s function=%s arm=%s duration_ms=%.6f mode=execution reward_time_source=duration_ms\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		durationMs,
	)
}

func logMABInvalidExecutionFeedback(
	policy string,
	functionName string,
	arm string,
	reason string,
	feedback ExecutionFeedback,
) {
	log.Printf(
		"%s event=invalid_execution_feedback ts=%d policy=%s function=%s arm=%s reason=%s warm_start=%t duration_ms=%.6f response_time_ms=%.6f init_time_ms=%.6f queueing_time_ms=%.6f offload_latency_ms=%.6f cost_factor=%.6f energy_factor=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		reason,
		feedback.IsWarmStart,
		feedback.DurationMs,
		feedback.ResponseTimeMs,
		feedback.InitTimeMs,
		feedback.QueueingTimeMs,
		feedback.OffloadLatencyMs,
		feedback.CostFactor,
		feedback.EnergyFactor,
	)
}

func logMABInvalidObservabilityMetric(
	policy string,
	functionName string,
	arm string,
	metric string,
	value float64,
) {
	log.Printf(
		"%s event=invalid_observability_metric ts=%d policy=%s function=%s arm=%s metric=%s value=%.6f action=continue_if_reward_feedback_valid\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		metric,
		value,
	)
}

func logMABColdStartStats(
	policy string,
	functionName string,
	arm string,
	stats ColdStartStatsSnapshot,
) {
	log.Printf(
		"%s event=cold_start_stats ts=%d policy=%s function=%s arm=%s total_invocations=%d cold_observed=%d warm_observed=%d cold_accepted=%d cold_skipped=%d invalid_feedback=%d cold_duration_samples=%d warm_duration_samples=%d cold_init_time_samples=%d avg_cold_duration_ms=%.6f avg_warm_duration_ms=%.6f avg_cold_init_time_ms=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		stats.TotalInvocations,
		stats.ColdObserved,
		stats.WarmObserved,
		stats.ColdAccepted,
		stats.ColdSkipped,
		stats.InvalidFeedback,
		stats.ColdDurationSamples,
		stats.WarmDurationSamples,
		stats.ColdInitTimeSamples,
		stats.AvgColdDurationMs,
		stats.AvgWarmDurationMs,
		stats.AvgColdInitTimeMs,
	)
}
