package mab

import (
	"log"
	"math"
	"strings"
	"sync"
)

// UCB1DecoupledBandit is the experimental transfer-aware UCB1 variant in
// which the statistical strength of transferred reward knowledge and the
// exploration pseudo-count are independent.
//
// For arm j:
//
//	mu_eff_j = (S_j + w_R * mu_prior_j) / (n_j + w_R)
//
//	bonus_j = c * sqrt(
//	    ln(N + sum_k w_E,k) / (n_j + w_E,j)
//	)
//
// where:
//   - w_R is ArmStats.PriorObservationWeight;
//   - w_E is ArmStats.PriorExplorationObservationWeight.
//
// In-flight selections participate only in the exploration counts exactly as
// in the historical UCB1 implementation. When w_R == w_E for every arm, this
// policy is mathematically equivalent to the coupled transfer formula in
// bandit_ucb1.go.
//
// The policy intentionally has its own BanditType so that coupled and
// decoupled experiments can be selected and reproduced from configuration
// without replacing the historical implementation.
type UCB1DecoupledBandit struct {
	FunctionName  string
	TotalCounts   int64
	Arms          map[string]*ArmStats
	mu            sync.RWMutex
	c             float64
	TotalInFlight int64

	// PriorDonorFunctionName is provenance only.
	PriorDonorFunctionName string
}

func NewUCB1DecoupledBandit(functionName string, exploration float64) *UCB1DecoupledBandit {
	return &UCB1DecoupledBandit{
		FunctionName: functionName,
		Arms:         make(map[string]*ArmStats),
		c:            exploration,
	}
}

func (b *UCB1DecoupledBandit) InitArm(arm string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.Arms[arm]; !exists {
		b.Arms[arm] = &ArmStats{Count: 0, SumRewards: 0, AvgReward: 0}
	}

	logMABArmAdded(
		string(b.GetType()),
		arm,
		b.FunctionName,
		formatArmsFromMap(b.Arms),
	)
}

func (b *UCB1DecoupledBandit) SelectArm(ctx *Context) string {
	return b.SelectArmFrom(ctx, nil)
}

// SelectArmFrom implements the decoupled UCB1 score over the supplied action
// mask. The reward prior weight affects only exploitation; the exploration
// prior weight affects only the UCB pseudo-count.
func (b *UCB1DecoupledBandit) SelectArmFrom(ctx *Context, allowedArms []string) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	_ = ctx // UCB1 remains context-free.

	candidateArms := filterAllowedArms(
		b.Arms,
		allowedArms,
	)

	if len(candidateArms) == 0 {
		log.Printf(
			"[MAB] event=no_allowed_arms ts=%d policy=%s function=%s\n",
			nowMillis(),
			string(b.GetType()),
			b.FunctionName,
		)
		return ""
	}

	// The force-exploration check uses the exploration count. A fresh arm is
	// therefore untried only when it has no live observation, no pending request
	// and no exploration prior evidence.
	leastTriedArm := ""
	currentMinEffectiveCount := math.MaxFloat64

	for _, arm := range candidateArms {
		stats := b.Arms[arm]
		effectiveCount := b.explorationArmObservationWeightLocked(stats)

		if effectiveCount <= 0.0 && effectiveCount < currentMinEffectiveCount {
			currentMinEffectiveCount = effectiveCount
			leastTriedArm = arm
		}
	}

	if leastTriedArm != "" {
		b.markSelectionLocked(leastTriedArm)
		logMABSelectArm(
			string(b.GetType()),
			b.FunctionName,
			leastTriedArm,
			"least_tried",
			0.0,
			b.TotalCounts,
			b.TotalInFlight,
			strings.Join(candidateArms, ","),
		)
		return leastTriedArm
	}

	bestScore := -math.MaxFloat64
	bestArm := ""
	totalPriorExplorationWeight := b.totalPriorExplorationObservationWeightLocked()
	effectiveTotalCounts := float64(b.TotalCounts) + totalPriorExplorationWeight + float64(b.TotalInFlight)

	if effectiveTotalCounts < 1.0 {
		effectiveTotalCounts = 1.0
	}

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		explorationCount := b.explorationArmObservationWeightLocked(stats)
		if explorationCount <= 0.0 {
			explorationCount = 1.0
		}

		effectiveAvgReward := b.effectiveArmAverageRewardLocked(stats)
		explorationBonus := 0.0
		if b.c != 0.0 {
			explorationBonus = b.c * math.Sqrt(math.Log(effectiveTotalCounts)/explorationCount)
		}
		score := effectiveAvgReward + explorationBonus

		logMABUCB1DecoupledArmScore(
			b.FunctionName,
			arm,
			score,
			explorationBonus,
			stats.Count,
			stats.InFlight,
			stats.PriorObservationWeight,
			stats.PriorExplorationObservationWeight,
			float64(stats.Count)+stats.PriorObservationWeight,
			explorationCount,
			effectiveAvgReward,
			b.TotalCounts,
			b.TotalInFlight,
			totalPriorExplorationWeight,
			effectiveTotalCounts,
		)

		if score > bestScore {
			bestScore = score
			bestArm = arm
		}
	}

	if bestArm == "" {
		log.Printf(
			"[MAB] event=no_arm_selected ts=%d policy=%s function=%s allowed_arms=%v\n",
			nowMillis(),
			string(b.GetType()),
			b.FunctionName,
			candidateArms,
		)
		return ""
	}

	b.markSelectionLocked(bestArm)
	logMABSelectArm(
		string(b.GetType()),
		b.FunctionName,
		bestArm,
		"ucb_score",
		bestScore,
		b.TotalCounts,
		b.TotalInFlight,
		strings.Join(candidateArms, ","),
	)

	return bestArm
}

func (b *UCB1DecoupledBandit) UpdateReward(arm string, ctx *Context, feedback ExecutionFeedback) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.updateRewardLocked(arm, ctx, feedback)
}

func (b *UCB1DecoupledBandit) ResolveSelection(selectedArm string, executionArm string, ctx *Context, feedback *ExecutionFeedback, selectedArmReward *SyntheticReward) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.completeSelectionLocked(selectedArm) {
		return
	}

	if selectedArmReward != nil {
		b.updateSyntheticRewardLocked(selectedArm, *selectedArmReward)
	}

	if feedback == nil || executionArm == "" {
		return
	}

	b.updateRewardLocked(executionArm, ctx, *feedback)
}

func (b *UCB1DecoupledBandit) updateSyntheticRewardLocked(arm string, synthetic SyntheticReward) {
	stats, ok := b.Arms[arm]
	if !ok {
		log.Printf(
			"[MAB] event=unknown_synthetic_reward_arm policy=%s function=%s arm=%s reason=%s\n",
			string(b.GetType()),
			b.FunctionName,
			arm,
			synthetic.Reason,
		)
		return
	}

	if !isFiniteNumber(synthetic.Value) {
		log.Printf(
			"[MAB] event=invalid_synthetic_reward policy=%s function=%s arm=%s reward=%f reason=%s\n",
			string(b.GetType()),
			b.FunctionName,
			arm,
			synthetic.Value,
			synthetic.Reason,
		)
		return
	}

	stats.Count++
	stats.SyntheticCount++
	b.TotalCounts++
	stats.SumRewards += synthetic.Value
	stats.AvgReward = stats.SumRewards / float64(stats.Count)

	logMABSyntheticReward(
		string(b.GetType()),
		b.FunctionName,
		arm,
		synthetic,
		0.0,
		stats.Count,
		stats.AvgReward,
		b.TotalCounts,
	)
}

func (b *UCB1DecoupledBandit) markSelectionLocked(arm string) {
	stats, ok := b.Arms[arm]
	if !ok {
		return
	}

	startInFlightSelection(
		string(b.GetType()),
		b.FunctionName,
		arm,
		&stats.InFlight,
		&b.TotalInFlight,
	)
}

func (b *UCB1DecoupledBandit) completeSelectionLocked(arm string) bool {
	stats, ok := b.Arms[arm]
	if !ok {
		logMABInFlightIgnored(
			string(b.GetType()),
			b.FunctionName,
			arm,
			"no_pending_selection",
		)
		return false
	}

	return completeInFlightSelection(
		string(b.GetType()),
		b.FunctionName,
		arm,
		&stats.InFlight,
		&b.TotalInFlight,
	)
}

func (b *UCB1DecoupledBandit) updateRewardLocked(arm string, ctx *Context, feedback ExecutionFeedback) {
	_ = ctx
	stats, ok := b.Arms[arm]

	if !ok {
		log.Printf(
			"[MAB] event=unknown_feedback_arm policy=%s function=%s arm=%s\n",
			string(b.GetType()),
			b.FunctionName,
			arm,
		)
		return
	}

	policy := string(b.GetType())
	if !validateExecutionFeedback(policy, b.FunctionName, arm, feedback) {
		return
	}

	if !shouldUpdateRewardFromFeedback(policy, b.FunctionName, arm, feedback) {
		return
	}

	rewardResult, err := CalculateExecutionReward(feedback)
	if err != nil {
		log.Printf(
			"[MAB] event=reward_calculation_failed policy=%s function=%s arm=%s error=%v",
			UCB1Decoupled,
			b.FunctionName,
			arm,
			err,
		)
		return
	}

	reward := rewardResult.Value
	if !isFiniteNumber(reward) {
		recordInvalidExecutionFeedback(policy, b.FunctionName, arm, "non_finite_reward", feedback)
		return
	}

	logMABRewardBreakdown(
		policy,
		b.FunctionName,
		arm,
		feedback.DurationMs,
		rewardResult,
	)

	stats.Count++
	stats.RealCount++
	b.TotalCounts++
	stats.SumRewards += reward
	stats.AvgReward = stats.SumRewards / float64(stats.Count)
	stats.RealSumRewards += reward
	stats.RealAvgReward = stats.RealSumRewards / float64(stats.RealCount)
	recordAcceptedExecutionFeedback(policy, b.FunctionName, arm, feedback)

	logMABUpdateReward(
		policy,
		b.FunctionName,
		arm,
		feedback.DurationMs,
		feedback.IsWarmStart,
		reward,
		stats.Count,
		stats.AvgReward,
		b.TotalCounts,
	)
}

// effectiveArmAverageRewardLocked implements the exploitation part of the
// decoupled formula. PriorExplorationObservationWeight is deliberately absent.
func (b *UCB1DecoupledBandit) effectiveArmAverageRewardLocked(stats *ArmStats) float64 {
	if stats.PriorObservationWeight <= 0.0 {
		return stats.AvgReward
	}

	completedWeight := float64(stats.Count) + stats.PriorObservationWeight
	if completedWeight <= 0.0 {
		return 0.0
	}

	liveRewardSum := stats.AvgReward * float64(stats.Count)
	return (liveRewardSum + stats.PriorRewardSum) / completedWeight
}

// explorationArmObservationWeightLocked implements the denominator of the
// decoupled exploration term. The reward prior weight is deliberately absent.
func (b *UCB1DecoupledBandit) explorationArmObservationWeightLocked(stats *ArmStats) float64 {
	return float64(stats.Count) + stats.PriorExplorationObservationWeight + float64(stats.InFlight)
}

func (b *UCB1DecoupledBandit) totalPriorExplorationObservationWeightLocked() float64 {
	total := 0.0
	for _, stats := range b.Arms {
		total += stats.PriorExplorationObservationWeight
	}
	return total
}

func (b *UCB1DecoupledBandit) GetType() BanditType {
	return UCB1Decoupled
}

// TransferableKnowledge exports only accepted real execution feedback.
// Neither reward-prior nor exploration-prior pseudo-observations are exported,
// so transferred knowledge can never recursively become "real" evidence.
func (b *UCB1DecoupledBandit) TransferableKnowledge() TransferableMABKnowledge {
	b.mu.RLock()
	defer b.mu.RUnlock()

	arms := make(map[string]TransferableArmKnowledge, len(b.Arms))
	var totalReal int64
	var totalSynthetic int64

	for arm, stats := range b.Arms {
		totalReal += stats.RealCount
		totalSynthetic += stats.SyntheticCount
		arms[arm] = TransferableArmKnowledge{
			RealObservationCount:              stats.RealCount,
			ExcludedSyntheticObservationCount: stats.SyntheticCount,
			UCB1: &TransferableUCB1ArmKnowledge{
				RealSumRewards: stats.RealSumRewards,
				RealAvgReward:  stats.RealAvgReward,
			},
		}
	}

	return TransferableMABKnowledge{
		SchemaVersion:                     TransferableMABKnowledgeSchemaVersion,
		FunctionName:                      b.FunctionName,
		Policy:                            UCB1Decoupled,
		HasRealKnowledge:                  totalReal > 0,
		RealObservationCount:              totalReal,
		ExcludedSyntheticObservationCount: totalSynthetic,
		Arms:                              arms,
	}
}

func logMABUCB1DecoupledArmScore(
	functionName string,
	arm string,
	score float64,
	explorationBonus float64,
	count int64,
	inFlight int64,
	rewardPriorWeight float64,
	explorationPriorWeight float64,
	rewardEffectiveCount float64,
	explorationEffectiveCount float64,
	avgReward float64,
	totalCounts int64,
	totalInFlight int64,
	totalPriorExplorationWeight float64,
	effectiveTotalCounts float64,
) {
	log.Printf(
		"%s event=arm_score ts=%d policy=%s function=%s arm=%s score=%.6f count=%d in_flight=%d reward_prior_weight=%.6f exploration_prior_weight=%.6f reward_effective_count=%.6f exploration_effective_count=%.6f avg_reward=%.6f exploration_bonus=%.6f total_counts=%d total_in_flight=%d total_prior_exploration_weight=%.6f effective_total_counts=%.6f contextual=false decoupled_prior=true\n",
		mabLogPrefix,
		nowMillis(),
		UCB1Decoupled,
		functionName,
		arm,
		score,
		count,
		inFlight,
		rewardPriorWeight,
		explorationPriorWeight,
		rewardEffectiveCount,
		explorationEffectiveCount,
		avgReward,
		explorationBonus,
		totalCounts,
		totalInFlight,
		totalPriorExplorationWeight,
		effectiveTotalCounts,
	)
}
