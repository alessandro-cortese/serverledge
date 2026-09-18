package mab

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLocalUCB1BehavioralSmoke is an opt-in behavioral smoke test for the
// UCB1 transfer-learning experiment. It is deliberately synthetic: its goal is
// to verify the decision semantics of the Go runtime before spending GCP
// budget, not to estimate real hardware performance.
//
// Run with:
//
//	SERVERLEDGE_LOCAL_UCB_SMOKE=1 go test ./internal/mab \
//	  -count=1 -v -run '^TestLocalUCB1BehavioralSmoke$'
//
// Controlled setup:
//   - donor real means: x86=100 ms, arm64=80 ms;
//   - target reference profile: x86=200 ms;
//   - reference anchoring therefore predicts arm64=160 ms;
//   - reward is r=-ln(duration_ms);
//   - one identical real target observation (arm64=140 ms) is replayed into
//     the comparison scenarios before the next decision.
//
// The smoke verifies:
//  1. no-transfer still force-explores an unseen arm;
//  2. UCB1 coupled w=.25 and UCB1Decoupled wR=.25,wE=.25 are identical;
//  3. changing only wE from .25 to 1 leaves exploitation means unchanged but
//     changes exploration bonuses and can change the selected arm;
//  4. wR=1 gives an exact 50/50 exploitation mean after one real feedback;
//  5. with c=0, exploration bonus is zero and the prior alone drives the
//     initial decision for wR=.25/.5/1.
func TestLocalUCB1BehavioralSmoke(t *testing.T) {
	if os.Getenv("SERVERLEDGE_LOCAL_UCB_SMOKE") != "1" {
		t.Skip("set SERVERLEDGE_LOCAL_UCB_SMOKE=1 to run the local UCB1 behavioral smoke")
	}

	viper.Reset()
	t.Cleanup(viper.Reset)
	viper.Set(config.MAB_REWARD_MODE, string(RewardModeLatency))

	const (
		c                  = 0.8
		targetX86ProfileMs = 200.0
		targetArmLiveMs    = 140.0
	)

	donor := localSmokeDonorKnowledge()
	targetReferenceReward := -math.Log(targetX86ProfileMs)

	anchor := &UCB1ReferenceAnchorConfig{
		Enabled:                   true,
		ReferenceArm:              "x86",
		TargetReferenceMeanReward: targetReferenceReward,
	}

	// Build the priors from exactly the same donor state. Only the target
	// formula/configuration changes.
	coupledPrior := localSmokeBuildPrior(t, donor, UCB1, WeakMABPriorConfig{
		EquivalentObservationWeight: 0.25,
		MinRealObservationsPerArm:   1,
		UCB1ReferenceAnchor:         anchor,
	})

	diagonalPrior := localSmokeBuildPrior(t, donor, UCB1Decoupled, WeakMABPriorConfig{
		RewardObservationWeight:      0.25,
		ExplorationObservationWeight: 0.25,
		MinRealObservationsPerArm:    1,
		UCB1ReferenceAnchor:          anchor,
	})

	decoupledPrior := localSmokeBuildPrior(t, donor, UCB1Decoupled, WeakMABPriorConfig{
		RewardObservationWeight:      0.25,
		ExplorationObservationWeight: 1.0,
		MinRealObservationsPerArm:    1,
		UCB1ReferenceAnchor:          anchor,
	})

	fiftyFiftyPrior := localSmokeBuildPrior(t, donor, UCB1Decoupled, WeakMABPriorConfig{
		RewardObservationWeight:      1.0,
		ExplorationObservationWeight: 1.0,
		MinRealObservationsPerArm:    1,
		UCB1ReferenceAnchor:          anchor,
	})

	// The donor's 100 -> 80 ms architecture effect is a 1.25x speedup on ARM.
	// Anchoring the target at 200 ms must therefore produce an ARM prior of
	// 160 ms without observing target ARM before transfer.
	anchoredX86 := coupledPrior.Arms["x86"].UCB1.MeanReward
	anchoredARM := coupledPrior.Arms["arm64"].UCB1.MeanReward
	require.InDelta(t, targetX86ProfileMs, math.Exp(-anchoredX86), 1e-9)
	require.InDelta(t, 160.0, math.Exp(-anchoredARM), 1e-9)

	t.Logf("ANCHOR donor=(x86=100ms, arm64=80ms) target_x86=%.0fms -> prior=(x86=%.3fms, arm64=%.3fms)",
		targetX86ProfileMs,
		math.Exp(-anchoredX86),
		math.Exp(-anchoredARM),
	)

	feedback := ExecutionFeedback{
		DurationMs:  targetArmLiveMs,
		IsWarmStart: true,
	}

	// No-transfer baseline: after observing ARM once, x86 is still unseen and
	// classic UCB1 must force-explore it.
	noTransfer := NewUCB1Bandit("smoke-no-transfer", c)
	localSmokeInitArms(noTransfer)
	noTransfer.UpdateReward("arm64", nil, feedback)
	noTransferScores := localSmokeCoupledScores(noTransfer)
	noTransferSelected := noTransfer.SelectArm(nil)
	require.Equal(t, "x86", noTransferSelected)

	// Coupled historical formula, w=.25.
	coupled := NewUCB1Bandit("smoke-coupled-w025", c)
	localSmokeInitArms(coupled)
	require.NoError(t, ApplyWeakMABPrior(coupled, coupledPrior))
	coupled.UpdateReward("arm64", nil, feedback)
	coupledScores := localSmokeCoupledScores(coupled)
	coupledSelected := coupled.SelectArm(nil)

	// Decoupled diagonal, wR=wE=.25. This must exactly reproduce the coupled
	// formula under the same prior and live state.
	diagonal := NewUCB1DecoupledBandit("smoke-decoupled-diagonal", c)
	localSmokeInitArms(diagonal)
	require.NoError(t, ApplyWeakMABPrior(diagonal, diagonalPrior))
	diagonal.UpdateReward("arm64", nil, feedback)
	diagonalScores := localSmokeDecoupledScores(diagonal)
	diagonalSelected := diagonal.SelectArm(nil)

	// Decoupled candidate motivated by the offline replay: same reward trust
	// wR=.25, but a full exploration pseudo-count wE=1.
	decoupled := NewUCB1DecoupledBandit("smoke-decoupled-wR025-wE1", c)
	localSmokeInitArms(decoupled)
	require.NoError(t, ApplyWeakMABPrior(decoupled, decoupledPrior))
	decoupled.UpdateReward("arm64", nil, feedback)
	decoupledScores := localSmokeDecoupledScores(decoupled)
	decoupledSelected := decoupled.SelectArm(nil)

	for _, arm := range []string{"x86", "arm64"} {
		assert.InDelta(t, coupledScores[arm].EffectiveMean, diagonalScores[arm].EffectiveMean, 1e-12)
		assert.InDelta(t, coupledScores[arm].ExplorationBonus, diagonalScores[arm].ExplorationBonus, 1e-12)
		assert.InDelta(t, coupledScores[arm].Score, diagonalScores[arm].Score, 1e-12)

		// wR is unchanged between coupled and the decoupled candidate, therefore
		// the exploitation mean must remain exactly the same.
		assert.InDelta(t, coupledScores[arm].EffectiveMean, decoupledScores[arm].EffectiveMean, 1e-12)
	}

	require.Equal(t, coupledSelected, diagonalSelected)

	// This controlled state is intentionally chosen to expose the effect of wE:
	// coupled/diagonal explores x86, while wE=1 is confident enough to keep the
	// currently better ARM arm. This is a semantic demonstration, not a claim
	// about which choice is universally better.
	require.Equal(t, "x86", coupledSelected)
	require.Equal(t, "arm64", decoupledSelected)

	// Explicit 50/50 case. After one real ARM observation, wR=1 means exactly
	// one prior equivalent observation + one real observation in exploitation.
	fiftyFifty := NewUCB1DecoupledBandit("smoke-fifty-fifty", c)
	localSmokeInitArms(fiftyFifty)
	require.NoError(t, ApplyWeakMABPrior(fiftyFifty, fiftyFiftyPrior))
	fiftyFifty.UpdateReward("arm64", nil, feedback)
	fiftyScores := localSmokeDecoupledScores(fiftyFifty)
	fiftySelected := fiftyFifty.SelectArm(nil)

	priorARMReward := fiftyFiftyPrior.Arms["arm64"].UCB1.MeanReward
	realARMReward := -math.Log(targetArmLiveMs)
	expectedFiftyFiftyMean := (priorARMReward + realARMReward) / 2.0
	assert.InDelta(t, expectedFiftyFiftyMean, fiftyScores["arm64"].EffectiveMean, 1e-12)

	t.Log("CONTROLLED STATE: one identical real target observation arm64=140ms before the next decision")
	t.Log("scenario                         selected   arm     n   wR     wE     eff_mean    bonus       score")
	localSmokeLogScenario(t, "no-transfer", noTransferSelected, noTransferScores)
	localSmokeLogScenario(t, "coupled w=.25", coupledSelected, coupledScores)
	localSmokeLogScenario(t, "decoupled wR=.25 wE=.25", diagonalSelected, diagonalScores)
	localSmokeLogScenario(t, "decoupled wR=.25 wE=1", decoupledSelected, decoupledScores)
	localSmokeLogScenario(t, "50/50 wR=1 wE=1", fiftySelected, fiftyScores)

	// Knowledge-only modes: c=0 means the exploration term is exactly zero.
	// wE is intentionally kept at 1 to demonstrate that it becomes irrelevant
	// when c=0. No live target feedback is inserted before this first decision.
	for _, wR := range []float64{0.25, 0.5, 1.0} {
		prior := localSmokeBuildPrior(t, donor, UCB1Decoupled, WeakMABPriorConfig{
			RewardObservationWeight:      wR,
			ExplorationObservationWeight: 1.0,
			MinRealObservationsPerArm:    1,
			UCB1ReferenceAnchor:          anchor,
		})

		bandit := NewUCB1DecoupledBandit(fmt.Sprintf("smoke-knowledge-only-%.2f", wR), 0.0)
		localSmokeInitArms(bandit)
		require.NoError(t, ApplyWeakMABPrior(bandit, prior))

		scores := localSmokeDecoupledScores(bandit)
		selected := bandit.SelectArm(nil)

		require.Equal(t, "arm64", selected)
		for _, arm := range []string{"x86", "arm64"} {
			assert.Zero(t, scores[arm].ExplorationBonus)
		}

		localSmokeLogScenario(t, fmt.Sprintf("knowledge-only c=0 wR=%.2f", wR), selected, scores)
	}

	t.Log("SMOKE RESULT: PASS — coupled diagonal equivalence, independent exploration weight, 50/50 exploitation, and c=0 knowledge-only semantics are all active in the Go runtime")
}

type localSmokeScore struct {
	LiveCount              int64
	RewardPriorWeight      float64
	ExplorationPriorWeight float64
	EffectiveMean          float64
	ExplorationBonus       float64
	Score                  float64
	ForceExplore           bool
}

func localSmokeDonorKnowledge() TransferableMABKnowledge {
	const realCount int64 = 4

	x86Mean := -math.Log(100.0)
	armMean := -math.Log(80.0)

	return TransferableMABKnowledge{
		SchemaVersion:        TransferableMABKnowledgeSchemaVersion,
		FunctionName:         "smoke-donor",
		Policy:               UCB1,
		HasRealKnowledge:     true,
		RealObservationCount: 2 * realCount,
		Arms: map[string]TransferableArmKnowledge{
			"x86": {
				RealObservationCount: realCount,
				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: x86Mean * float64(realCount),
					RealAvgReward:  x86Mean,
				},
			},
			"arm64": {
				RealObservationCount: realCount,
				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: armMean * float64(realCount),
					RealAvgReward:  armMean,
				},
			},
		},
	}
}

func localSmokeBuildPrior(
	t *testing.T,
	donor TransferableMABKnowledge,
	targetPolicy BanditType,
	cfg WeakMABPriorConfig,
) WeakMABPrior {
	t.Helper()

	prior, err := BuildWeakMABPriorForTarget(donor, targetPolicy, cfg)
	require.NoError(t, err)
	require.True(t, prior.HasPrior)
	require.Equal(t, 2, prior.TransferredArmCount)
	return prior
}

func localSmokeInitArms(policy interface{ InitArm(string) }) {
	policy.InitArm("x86")
	policy.InitArm("arm64")
}

func localSmokeCoupledScores(b *UCB1Bandit) map[string]localSmokeScore {
	b.mu.RLock()
	defer b.mu.RUnlock()

	totalPrior := b.totalPriorObservationWeightLocked()
	effectiveTotal := float64(b.TotalCounts) + totalPrior + float64(b.TotalInFlight)
	if effectiveTotal < 1.0 {
		effectiveTotal = 1.0
	}

	out := make(map[string]localSmokeScore, len(b.Arms))
	for _, arm := range []string{"x86", "arm64"} {
		stats := b.Arms[arm]
		effectiveCount := b.effectiveArmObservationWeightLocked(stats)
		entry := localSmokeScore{
			LiveCount:              stats.Count,
			RewardPriorWeight:      stats.PriorObservationWeight,
			ExplorationPriorWeight: stats.PriorObservationWeight,
			EffectiveMean:          b.effectiveArmAverageRewardLocked(stats),
			ForceExplore:           effectiveCount <= 0.0,
		}

		if !entry.ForceExplore {
			entry.ExplorationBonus = b.c * math.Sqrt(math.Log(effectiveTotal)/effectiveCount)
			entry.Score = entry.EffectiveMean + entry.ExplorationBonus
		}
		out[arm] = entry
	}
	return out
}

func localSmokeDecoupledScores(b *UCB1DecoupledBandit) map[string]localSmokeScore {
	b.mu.RLock()
	defer b.mu.RUnlock()

	totalPrior := b.totalPriorExplorationObservationWeightLocked()
	effectiveTotal := float64(b.TotalCounts) + totalPrior + float64(b.TotalInFlight)
	if effectiveTotal < 1.0 {
		effectiveTotal = 1.0
	}

	out := make(map[string]localSmokeScore, len(b.Arms))
	for _, arm := range []string{"x86", "arm64"} {
		stats := b.Arms[arm]
		explorationCount := b.explorationArmObservationWeightLocked(stats)
		entry := localSmokeScore{
			LiveCount:              stats.Count,
			RewardPriorWeight:      stats.PriorObservationWeight,
			ExplorationPriorWeight: stats.PriorExplorationObservationWeight,
			EffectiveMean:          b.effectiveArmAverageRewardLocked(stats),
			ForceExplore:           explorationCount <= 0.0,
		}

		if !entry.ForceExplore {
			entry.ExplorationBonus = b.c * math.Sqrt(math.Log(effectiveTotal)/explorationCount)
			entry.Score = entry.EffectiveMean + entry.ExplorationBonus
		}
		out[arm] = entry
	}
	return out
}

func localSmokeLogScenario(t *testing.T, scenario string, selected string, scores map[string]localSmokeScore) {
	t.Helper()
	for _, arm := range []string{"x86", "arm64"} {
		s := scores[arm]
		if s.ForceExplore {
			t.Logf("%-32s %-10s %-7s %d   %.2f   %.2f   %-11s %-11s %-11s",
				scenario,
				selected,
				arm,
				s.LiveCount,
				s.RewardPriorWeight,
				s.ExplorationPriorWeight,
				"FORCE",
				"FORCE",
				"FORCE",
			)
			continue
		}

		t.Logf("%-32s %-10s %-7s %d   %.2f   %.2f   %+.6f   %+.6f   %+.6f",
			scenario,
			selected,
			arm,
			s.LiveCount,
			s.RewardPriorWeight,
			s.ExplorationPriorWeight,
			s.EffectiveMean,
			s.ExplorationBonus,
			s.Score,
		)
	}
}
