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

func logMABArmScore(
	policy string,
	functionName string,
	arm string,
	score float64,
	count int64,
	avgReward float64,
	totalCounts int64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f count=%d avg_reward=%.6f total_counts=%d\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		count,
		avgReward,
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
	memUsage float64,
	lambda float64,
	reward float64,
) {
	log.Printf(
		"%s event=update_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f warm_start=%t mem_usage=%.6f lambda=%.6f reward=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		durationMs,
		isWarmStart,
		memUsage,
		lambda,
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
	memUsage float64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f expected_reward=%.6f confidence=%.6f mem_usage=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		expectedReward,
		confidence,
		memUsage,
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
	capabilities := strings.Join(info.Capabilities, ",")

	log.Printf(
		"%s event=reward_breakdown ts=%d policy=%s function=%s arm=%s base_tag=%s architecture=%s specialization=%s capabilities=[%s] duration_ms=%.6f latency_reward=%.6f cost_weight=%.6f cost_factor=%.6f cost_term=%.6f energy_weight=%.6f energy_factor=%.6f energy_term=%.6f memory_weight=%.6f memory_penalty=%.6f memory_term=%.6f final_reward=%.6f\n",
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
		breakdown.MemoryWeight,
		breakdown.MemoryPenalty,
		breakdown.MemoryTerm,
		breakdown.FinalReward,
	)
}
