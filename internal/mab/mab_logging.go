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

func logMABUCB1UtilizationArmScore(
	policy string,
	functionName string,
	arm string,
	score float64,
	baseScore float64,
	explorationBonus float64,
	count int64,
	avgReward float64,
	totalCounts int64,
	utilization UtilizationScoreBreakdown,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f base_score=%.6f count=%d avg_reward=%.6f exploration_bonus=%.6f total_counts=%d contextual=true utilization=%.6f utilization_threshold=%.6f utilization_penalty=%.6f utilization_weight=%.6f utilization_term=%.6f\n",
		mabLogPrefix,
		nowMillis(),
		policy,
		functionName,
		arm,
		score,
		baseScore,
		count,
		avgReward,
		explorationBonus,
		totalCounts,
		utilization.Utilization,
		utilization.UtilizationThreshold,
		utilization.UtilizationPenalty,
		utilization.UtilizationWeight,
		utilization.UtilizationTerm,
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
		"%s event=update_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f warm_start=%t utilization=%.6f reward=%.6f\n",
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
