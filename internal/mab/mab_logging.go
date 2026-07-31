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
	arms string,
) {
	log.Printf(
		"%s event=select_arm ts=%d policy=%s function=%s selected_arm=%s reason=%s score=%.6f total_counts=%d arms=[%s]\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		selectedArm,
		reason,
		score,
		totalCounts,
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
	avgReward float64,
	totalCounts int64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f base_score=%.6f count=%d avg_reward=%.6f exploration_bonus=%.6f total_counts=%d contextual=false\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		score,
		count,
		avgReward,
		explorationBonus,
		totalCounts,
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
	arms string,
) {
	log.Printf(
		"%s event=select_arm ts=%d policy=%s function=%s selected_arm=%s reason=%s score=%.6f arms=[%s]\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		selectedArm,
		reason,
		score,
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
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f expected_reward=%.6f confidence=%.6f contextual=true utilization=%.6f explicit_utilization_penalty=false\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		expectedReward,
		confidence,
		utilization,
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
