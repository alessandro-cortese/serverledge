package mab

import (
	"log"
	"math"
	"strings"
	"sync"
)

// NOTE: Since nomenclature may be confusing: 'ARM' is the architecture, 'arm' is the arm of the Multi-Armed Bandit (MAB)

// ArmStats maintains information about a single arm dedicated to a single function
type ArmStats struct {
	Count      int64   // Count is the number of accepted learning observations, including synthetic fallback penalties.
	SumRewards float64 // SumRewards contains accepted live learning rewards (real + synthetic), excluding prior evidence.
	AvgReward  float64 // AvgReward is the accepted live mean; selection combines it with weak prior evidence when present.
	InFlight   int64

	// Real* tracks only accepted execution feedback.
	//
	// These values are deliberately kept separate from Count/SumRewards/
	// AvgReward because the live UCB1 state also includes synthetic fallback
	// penalties. Transfer learning must be able to export function knowledge
	// without treating those penalties as intrinsic observations of the
	// function.
	RealCount      int64
	RealSumRewards float64
	RealAvgReward  float64

	// SyntheticCount is diagnostic only.
	//
	// Synthetic observations continue to affect the live UCB1 statistics
	// exactly as before, but they are excluded from transferable knowledge.
	SyntheticCount int64

	// Prior* contains donor-derived weak statistical evidence applied before
	// the target function starts learning. Prior state influences selection,
	// but it is never counted as real or synthetic target experience.
	PriorObservationWeight float64
	PriorRewardSum         float64
}

// UCB1Bandit is the bandit that handles decision for ONE function
type UCB1Bandit struct {
	FunctionName string               // TODO: Ask if can do this
	TotalCounts  int64                // TotalCounts is the total number of accepted learning observations across all arms for this function.
	Arms         map[string]*ArmStats // Map "amd64" -> Stats, "arm64" -> Stats for each arm
	mu           sync.RWMutex         // Mutex per thread-safety
	c            float64              // Exploration parameter C (usually sqrt(2) ~= 1.41, but can be tuned)
	// Higher values lead to more exploration. Lower values lead to more exploitation.
	TotalInFlight int64

	// PriorDonorFunctionName is provenance only. It is populated when a weak
	// prior is applied and does not participate directly in the UCB1 formula.
	PriorDonorFunctionName string
}

func NewUCB1Bandit(
	functionName string,
	exploration float64,
) *UCB1Bandit {
	return &UCB1Bandit{
		FunctionName: functionName,
		Arms:         make(map[string]*ArmStats),
		c:            exploration,
	}
}

// InitArm adds a new arm to the bandit
func (b *UCB1Bandit) InitArm(arm string) {
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

// SelectArm chooses among all arms known by the policy.
// It delegates to SelectArmFrom with no action mask.
func (b *UCB1Bandit) SelectArm(ctx *Context) string {
	return b.SelectArmFrom(ctx, nil)
}

// SelectArmFrom implements UCB1 over the supplied action mask.
//
// Count and TotalCounts contain accepted learning observations, including
// synthetic fallback penalties.
//
// RealCount/RealSumRewards/RealAvgReward separately track only accepted
// execution feedback for future transfer learning.
//
// InFlight and TotalInFlight account for selections that have not completed yet.
// Pending selections affect exploration through Count+InFlight, but they never
// enter AvgReward before a learning observation is accepted.
func (b *UCB1Bandit) SelectArmFrom(
	ctx *Context,
	allowedArms []string,
) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Classic UCB1 is intentionally context-free.
	_ = ctx

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

	// An arm is unexplored only when neither live observations, pending
	// evaluations nor weak prior evidence are available. Prior weight can be
	// fractional, therefore the zero check cannot rely on an integer count.
	leastTriedArm := ""
	currentMinEffectiveCount := math.MaxFloat64

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		effectiveCount :=
			b.effectiveArmObservationWeightLocked(stats)

		if effectiveCount <= 0.0 &&
			effectiveCount < currentMinEffectiveCount {

			currentMinEffectiveCount =
				effectiveCount

			leastTriedArm =
				arm
		}
	}

	if leastTriedArm != "" {
		b.markSelectionLocked(
			leastTriedArm,
		)

		logMABSelectArm(
			string(b.GetType()),
			b.FunctionName,
			leastTriedArm,
			"least_tried",
			0.0,
			b.TotalCounts,
			b.TotalInFlight,
			strings.Join(
				candidateArms,
				",",
			),
		)

		return leastTriedArm
	}

	bestScore := -math.MaxFloat64
	bestArm := ""

	totalPriorObservationWeight :=
		b.totalPriorObservationWeightLocked()

	effectiveTotalCounts :=
		float64(b.TotalCounts) +
			totalPriorObservationWeight +
			float64(
				b.TotalInFlight,
			)

	if effectiveTotalCounts < 1.0 {
		effectiveTotalCounts = 1.0
	}

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		effectiveCount :=
			b.effectiveArmObservationWeightLocked(stats)

		if effectiveCount <= 0.0 {
			effectiveCount = 1.0
		}

		effectiveAvgReward :=
			b.effectiveArmAverageRewardLocked(stats)

		explorationBonus :=
			b.c *
				math.Sqrt(
					math.Log(
						effectiveTotalCounts,
					)/
						effectiveCount,
				)

		score :=
			effectiveAvgReward +
				explorationBonus

		logMABUCB1ArmScore(
			string(b.GetType()),
			b.FunctionName,
			arm,
			score,
			explorationBonus,
			stats.Count,
			stats.InFlight,
			stats.PriorObservationWeight,
			effectiveCount,
			effectiveAvgReward,
			b.TotalCounts,
			b.TotalInFlight,
			totalPriorObservationWeight,
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

	b.markSelectionLocked(
		bestArm,
	)

	logMABSelectArm(
		string(b.GetType()),
		b.FunctionName,
		bestArm,
		"ucb_score",
		bestScore,
		b.TotalCounts,
		b.TotalInFlight,
		strings.Join(
			candidateArms,
			",",
		),
	)

	return bestArm
}

// UpdateReward applies a standalone feedback sample. It does not resolve an
// in-flight selection; production request handling uses ResolveSelection.
func (b *UCB1Bandit) UpdateReward(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.updateRewardLocked(
		arm,
		ctx,
		feedback,
	)
}

// ResolveSelection atomically closes the selected arm's pending request and,
// when feedback is available, updates the arm that actually executed it.
func (b *UCB1Bandit) ResolveSelection(
	selectedArm string,
	executionArm string,
	ctx *Context,
	feedback *ExecutionFeedback,
	selectedArmReward *SyntheticReward,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.completeSelectionLocked(
		selectedArm,
	) {
		return
	}

	if selectedArmReward != nil {
		b.updateSyntheticRewardLocked(
			selectedArm,
			*selectedArmReward,
		)
	}

	if feedback == nil ||
		executionArm == "" {

		return
	}

	b.updateRewardLocked(
		executionArm,
		ctx,
		*feedback,
	)
}

func (b *UCB1Bandit) updateSyntheticRewardLocked(
	arm string,
	synthetic SyntheticReward,
) {
	stats, ok :=
		b.Arms[arm]

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

	if !isFiniteNumber(
		synthetic.Value,
	) {
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

	stats.SumRewards +=
		synthetic.Value

	stats.AvgReward =
		stats.SumRewards /
			float64(
				stats.Count,
			)

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

func (b *UCB1Bandit) markSelectionLocked(
	arm string,
) {
	stats, ok :=
		b.Arms[arm]

	if !ok {
		return
	}

	stats.InFlight++
	b.TotalInFlight++

	logMABInFlightChanged(
		string(b.GetType()),
		b.FunctionName,
		arm,
		"started",
		stats.InFlight,
		b.TotalInFlight,
	)
}

func (b *UCB1Bandit) completeSelectionLocked(
	arm string,
) bool {
	stats, ok :=
		b.Arms[arm]

	if !ok ||
		stats.InFlight <= 0 {

		logMABInFlightIgnored(
			string(b.GetType()),
			b.FunctionName,
			arm,
			"no_pending_selection",
		)

		return false
	}

	stats.InFlight--

	if b.TotalInFlight > 0 {
		b.TotalInFlight--
	}

	logMABInFlightChanged(
		string(b.GetType()),
		b.FunctionName,
		arm,
		"resolved",
		stats.InFlight,
		b.TotalInFlight,
	)

	return true
}

func (b *UCB1Bandit) updateRewardLocked(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	_ = ctx

	stats, ok :=
		b.Arms[arm]

	if !ok {
		log.Printf(
			"[MAB] event=unknown_feedback_arm policy=%s function=%s arm=%s\n",
			string(b.GetType()),
			b.FunctionName,
			arm,
		)

		return
	}

	policy :=
		string(
			b.GetType(),
		)

	if !validateExecutionFeedback(
		policy,
		b.FunctionName,
		arm,
		feedback,
	) {
		return
	}

	if !shouldUpdateRewardFromFeedback(
		policy,
		b.FunctionName,
		arm,
		feedback,
	) {
		return
	}

	latencyReward :=
		-math.Log(
			feedback.DurationMs,
		)

	reward :=
		latencyReward

	if !isFiniteNumber(
		reward,
	) {
		recordInvalidExecutionFeedback(
			policy,
			b.FunctionName,
			arm,
			"non_finite_reward",
			feedback,
		)

		return
	}

	logMABRewardBreakdown(
		policy,
		b.FunctionName,
		arm,
		feedback.DurationMs,
		reward,
	)

	stats.Count++
	stats.RealCount++
	b.TotalCounts++

	stats.SumRewards +=
		reward

	stats.AvgReward =
		stats.SumRewards /
			float64(
				stats.Count,
			)

	stats.RealSumRewards +=
		reward

	stats.RealAvgReward =
		stats.RealSumRewards /
			float64(
				stats.RealCount,
			)

	recordAcceptedExecutionFeedback(
		policy,
		b.FunctionName,
		arm,
		feedback,
	)

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

func (b *UCB1Bandit) effectiveArmObservationWeightLocked(
	stats *ArmStats,
) float64 {
	return float64(stats.Count) +
		stats.PriorObservationWeight +
		float64(stats.InFlight)
}

func (b *UCB1Bandit) effectiveArmAverageRewardLocked(
	stats *ArmStats,
) float64 {
	// Preserve the historical UCB1 state exactly when no transfer prior is
	// present. AvgReward remains the authoritative live mean maintained by the
	// existing update path.
	if stats.PriorObservationWeight <= 0.0 {
		return stats.AvgReward
	}

	completedWeight :=
		float64(stats.Count) +
			stats.PriorObservationWeight

	if completedWeight <= 0.0 {
		return 0.0
	}

	liveRewardSum :=
		stats.AvgReward *
			float64(stats.Count)

	return (liveRewardSum + stats.PriorRewardSum) /
		completedWeight
}

func (b *UCB1Bandit) totalPriorObservationWeightLocked() float64 {
	total := 0.0

	for _, stats := range b.Arms {
		total +=
			stats.PriorObservationWeight
	}

	return total
}

func (b *UCB1Bandit) GetType() BanditType {
	return UCB1
}
