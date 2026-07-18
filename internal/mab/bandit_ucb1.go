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
	Count      int64   // Count is the number of valid feedback samples used to estimate this arm.
	SumRewards float64 // SumRewards is the sum of valid rewards observed for this arm.
	AvgReward  float64 // AvgReward is the empirical mean reward of this arm.
}

// UCB1Bandit is the bandit that handles decision for ONE function
type UCB1Bandit struct {
	FunctionName string               // TODO: Ask if can do this
	TotalCounts  int64                // TotalCounts is the total number of valid feedback samples observed across all arms for this function.
	Arms         map[string]*ArmStats // Map "amd64" -> Stats, "arm64" -> Stats for each arm
	mu           sync.RWMutex         // Mutex per thread-safety
	c            float64              // Exploration parameter C (usually sqrt(2) ~= 1.41, but can be tuned)
	// Higher values lead to more exploration. Lower values lead to more exploitation.
	policyType BanditType
}

func NewUCB1Bandit(
	functionName string,
	exploration float64,
) *UCB1Bandit {
	return &UCB1Bandit{
		FunctionName: functionName,
		Arms:         make(map[string]*ArmStats),
		c:            exploration,
		policyType:   UCB1,
	}
}

func NewUCB1UtilizationAwareBandit(
	functionName string,
	exploration float64,
) *UCB1Bandit {
	return &UCB1Bandit{
		FunctionName: functionName,
		Arms:         make(map[string]*ArmStats),
		c:            exploration,
		policyType:   UCB1UtilizationAware,
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
// allowedArms semantics:
//   - nil: consider every arm known by the bandit;
//   - non-nil: consider only the listed, known arms.
//
// Selection does not change Count or TotalCounts. Those counters are updated
// only when UpdateReward receives a valid feedback sample.
func (b *UCB1Bandit) SelectArmFrom(
	ctx *Context,
	allowedArms []string,
) string {
	b.mu.Lock()
	defer b.mu.Unlock()

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

	minSampleCount := int64(1)
	currentMinSample := int64(math.MaxInt64)
	leastTriedArm := ""

	// Initial exploration is restricted to the allowed action set.
	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		if stats.Count < minSampleCount &&
			stats.Count < currentMinSample {

			currentMinSample = stats.Count
			leastTriedArm = arm
		}
	}

	if leastTriedArm != "" {
		logMABSelectArm(
			string(b.GetType()),
			b.FunctionName,
			leastTriedArm,
			"least_tried",
			0.0,
			b.TotalCounts,
			strings.Join(candidateArms, ","),
		)

		return leastTriedArm
	}

	bestScore := -math.MaxFloat64
	bestArm := ""

	// If every candidate has at least one sample, TotalCounts should be positive.
	// The guard protects against manually initialized or inconsistent test states.
	totalCounts := b.TotalCounts
	if totalCounts < 1 {
		totalCounts = 1
	}

	utilizationAware :=
		b.GetType() == UCB1UtilizationAware

	utilizationWeight := 0.0

	if utilizationAware {
		utilizationWeight =
			configuredUCB1UtilizationWeight()
	}

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		explorationBonus := b.c *
			math.Sqrt(
				math.Log(float64(totalCounts))/
					float64(stats.Count),
			)

		baseScore :=
			stats.AvgReward +
				explorationBonus

		score := baseScore

		if utilizationAware {
			breakdown :=
				buildUtilizationScoreBreakdown(
					arm,
					baseScore,
					ctx,
					utilizationWeight,
				)

			score = breakdown.FinalScore

			logMABUCB1UtilizationArmScore(
				string(b.GetType()),
				b.FunctionName,
				arm,
				score,
				baseScore,
				explorationBonus,
				stats.Count,
				stats.AvgReward,
				b.TotalCounts,
				breakdown,
			)
		} else {
			logMABUCB1ArmScore(
				string(b.GetType()),
				b.FunctionName,
				arm,
				score,
				explorationBonus,
				stats.Count,
				stats.AvgReward,
				b.TotalCounts,
			)
		}

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

	logMABSelectArm(
		string(b.GetType()),
		b.FunctionName,
		bestArm,
		"ucb_score",
		bestScore,
		b.TotalCounts,
		strings.Join(candidateArms, ","),
	)

	return bestArm
}

// UpdateReward updates the empirical reward of an arm after a valid execution.
// The historical reward contains latency, node-specific cost and node-specific
// energy. Plain UCB1 does not use context during selection, while the optional
// UCB1UtilizationAware variant applies utilization only to its selection score.
func (b *UCB1Bandit) UpdateReward(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {

	_ = ctx
	b.mu.Lock()
	defer b.mu.Unlock()

	stats, ok := b.Arms[arm]
	if !ok {
		return
	}

	if !feedback.IsWarmStart {
		logMABSkipColdStart(
			string(b.GetType()),
			b.FunctionName,
			arm,
			feedback.DurationMs,
		)
		return
	}

	if feedback.DurationMs <= 0 {
		log.Printf(
			"[MAB] event=skip_invalid_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f reason=non_positive_duration\n",
			nowMillis(),
			string(b.GetType()),
			b.FunctionName,
			arm,
			feedback.DurationMs,
		)
		return
	}

	latencyReward :=
		-math.Log(feedback.DurationMs)

	breakdown :=
		buildCostBreakdown(
			latencyReward,
			feedback,
		)

	reward := breakdown.FinalReward

	logMABRewardBreakdown(
		string(b.GetType()),
		b.FunctionName,
		arm,
		feedback.DurationMs,
		breakdown,
	)

	stats.Count++
	b.TotalCounts++

	stats.SumRewards += reward
	stats.AvgReward =
		stats.SumRewards /
			float64(stats.Count)

	logMABUpdateReward(
		string(b.GetType()),
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

func (b *UCB1Bandit) GetType() BanditType {
	if b.policyType == "" {
		return UCB1
	}

	return b.policyType
}
