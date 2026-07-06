package mab

import (
	"log"
	"math"
	"sync"
)

// NOTE: Since nomenclature may be confusing: 'ARM' is the architecture, 'arm' is the arm of the Multi-Armed Bandit (MAB)

// ArmStats maintains information about a single arm dedicated to a single function
type ArmStats struct {
	Count      int64   // UCB needs to know hom many times we chose that arm/architecture
	SumRewards float64 // Sum of rewards
	AvgReward  float64 // Avg Reward (Q value in the formula)
}

// UCB1Bandit is the bandit that handles decision for ONE function
type UCB1Bandit struct {
	FunctionName string               // TODO: Ask if can do this
	TotalCounts  int64                // number of total executions (t)
	Arms         map[string]*ArmStats // Map "amd64" -> Stats, "arm64" -> Stats for each arm
	mu           sync.RWMutex         // Mutex per thread-safety
	c            float64              // Exploration parameter C (usually sqrt(2) ~= 1.41, but can be tuned)
	// Higher values lead to more exploration. Lower values lead to more exploitation.
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

// SelectArm implements UCB-1 formulas
// Returns the suggested architecture to use ("amd64" o "arm64").
// ctx *Ctx is necessary even if not used to be compliant with the interface.
func (b *UCB1Bandit) SelectArm(ctx *Context) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	ctx = nil // not used, favor garbage collection
	minSampleCount := int64(1)
	currentMinSample := int64(math.MaxInt64)
	leastTriedArch := ""

	// If an arm hasn't tried at least minSampleCount times, it has to be tried
	// If more than one arm hasn't reached this threshold, choose the least tried one
	for arch, stats := range b.Arms {
		if stats.Count < minSampleCount && stats.Count < currentMinSample {
			currentMinSample = stats.Count
			leastTriedArch = arch
		}
	}

	if leastTriedArch != "" {

		logMABSelectArm(
			string(b.GetType()),
			b.FunctionName,
			leastTriedArch,
			"least_tried",
			0.0,
			b.TotalCounts,
			formatArmsFromMap(b.Arms),
		)

		return leastTriedArch
	}

	bestScore := -math.MaxFloat64
	bestArch := ""

	// Calculate UCB1 score for each architecture
	for arch, stats := range b.Arms {
		explorationBonus := b.c * math.Sqrt(math.Log(float64(b.TotalCounts))/float64(stats.Count))
		score := stats.AvgReward + explorationBonus

		logMABArmScore(
			string(b.GetType()),
			b.FunctionName,
			arch,
			score,
			stats.Count,
			stats.AvgReward,
			b.TotalCounts,
		)

		if score > bestScore {
			bestScore = score
			bestArch = arch
		}
	}

	if bestArch == "" {
		log.Printf("Couldn't select any ARM. Panic\n")
		panic(1)
	}

	logMABSelectArm(
		string(b.GetType()),
		b.FunctionName,
		bestArch,
		"ucb_score",
		bestScore,
		b.TotalCounts,
		formatArmsFromMap(b.Arms),
	)

	return bestArch
}

// UpdateReward updates bandit stats after execution. For now reward is 1.0 / executionTime (not considering setup time).
// It may be fine-tuned in the future. ctx *Context is need even if it's unused to be compliant with the interface.
func (b *UCB1Bandit) UpdateReward(arch string, ctx *Context, isWarmStart bool, durationMs float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	stats, ok := b.Arms[arch]
	if !ok {
		return // Should not happen
	}

	if !isWarmStart {
		// Redact this run if it was not a warm start. Likely to be an outlier.
		logMABSkipColdStart(
			string(b.GetType()),
			b.FunctionName,
			arch,
			durationMs,
		)
		return
	}

	if durationMs <= 0 {
		log.Printf(
			"[MAB] event=skip_invalid_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f reason=non_positive_duration\n",
			nowMillis(),
			string(b.GetType()),
			b.FunctionName,
			arch,
			durationMs,
		)
		return
	}

	latencyReward := -math.Log(durationMs)
	breakdown := buildCostBreakdown(
		arch,
		latencyReward,
		ctx,
		0.0,
		0.0,
	)
	reward := breakdown.FinalReward

	logMABRewardBreakdown(
		string(b.GetType()),
		b.FunctionName,
		arch,
		durationMs,
		breakdown,
	)

	stats.Count++
	b.TotalCounts++

	// Update average reward.
	stats.SumRewards += reward
	stats.AvgReward = stats.SumRewards / float64(stats.Count)

	logMABUpdateReward(
		string(b.GetType()),
		b.FunctionName,
		arch,
		durationMs,
		isWarmStart,
		reward,
		stats.Count,
		stats.AvgReward,
		b.TotalCounts,
	)
}

func (b *UCB1Bandit) GetType() BanditType {
	return UCB1
}
