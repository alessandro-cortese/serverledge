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
	InFlight   int64
}

// UCB1Bandit is the bandit that handles decision for ONE function
type UCB1Bandit struct {
	FunctionName string               // TODO: Ask if can do this
	TotalCounts  int64                // TotalCounts is the total number of valid feedback samples observed across all arms for this function.
	Arms         map[string]*ArmStats // Map "amd64" -> Stats, "arm64" -> Stats for each arm
	mu           sync.RWMutex         // Mutex per thread-safety
	c            float64              // Exploration parameter C (usually sqrt(2) ~= 1.41, but can be tuned)
	// Higher values lead to more exploration. Lower values lead to more exploitation.
	TotalInFlight int64
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
// Count and TotalCounts contain only completed, valid feedback samples.
// InFlight and TotalInFlight account for selections that have not completed yet.
// Pending selections affect exploration through Count+InFlight, but they never
// enter AvgReward before a real reward is observed.
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

	// An arm is still considered unexplored only when it has neither a
	// completed feedback nor a pending evaluation.
	leastTriedArm := ""
	currentMinEffectiveCount := int64(math.MaxInt64)

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		effectiveCount :=
			stats.Count +
				stats.InFlight

		if effectiveCount < 1 &&
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

	effectiveTotalCounts :=
		b.TotalCounts +
			b.TotalInFlight

	if effectiveTotalCounts < 1 {
		effectiveTotalCounts = 1
	}

	for _, arm := range candidateArms {
		stats := b.Arms[arm]

		effectiveCount :=
			stats.Count +
				stats.InFlight

		if effectiveCount < 1 {
			effectiveCount = 1
		}

		explorationBonus :=
			b.c *
				math.Sqrt(
					math.Log(
						float64(
							effectiveTotalCounts,
						),
					)/
						float64(
							effectiveCount,
						),
				)

		score :=
			stats.AvgReward +
				explorationBonus

		logMABUCB1ArmScore(
			string(b.GetType()),
			b.FunctionName,
			arm,
			score,
			explorationBonus,
			stats.Count,
			stats.InFlight,
			effectiveCount,
			stats.AvgReward,
			b.TotalCounts,
			b.TotalInFlight,
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
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Un feedback può essere applicato soltanto se esiste davvero una
	// selezione pendente da risolvere.
	if !b.completeSelectionLocked(
		selectedArm,
	) {
		return
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

	breakdown :=
		buildCostBreakdown(
			latencyReward,
			feedback,
		)

	reward :=
		breakdown.FinalReward

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
		breakdown,
	)

	stats.Count++
	b.TotalCounts++

	stats.SumRewards +=
		reward

	stats.AvgReward =
		stats.SumRewards /
			float64(
				stats.Count,
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
func (b *UCB1Bandit) GetType() BanditType {
	return UCB1
}
