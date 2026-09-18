package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildReferenceAnchoredDecoupledPriorTransfersArchitectureEffect(t *testing.T) {
	source := testUCBTransferableKnowledge(
		"donor",
		UCB1Decoupled,
		map[string]float64{
			"x86":   -4.0,
			"arm64": -3.5,
		},
	)

	prior, err := BuildWeakMABPrior(
		source,
		WeakMABPriorConfig{
			RewardObservationWeight:      0.25,
			ExplorationObservationWeight: 1.0,
			MinRealObservationsPerArm:    1,
			UCB1ReferenceAnchor: &UCB1ReferenceAnchorConfig{
				Enabled:                   true,
				ReferenceArm:              "x86",
				TargetReferenceMeanReward: -6.0,
			},
		},
	)
	require.NoError(t, err)

	x86 := prior.Arms["x86"].UCB1
	arm64 := prior.Arms["arm64"].UCB1
	require.NotNil(t, x86)
	require.NotNil(t, arm64)

	assert.InDelta(t, -6.0, x86.MeanReward, 1e-12)
	assert.InDelta(t, -5.5, arm64.MeanReward, 1e-12)
	assert.InDelta(t, 0.5, arm64.MeanReward-x86.MeanReward, 1e-12)
	assert.InDelta(t, 0.25, x86.ObservationWeight, 1e-12)
	assert.InDelta(t, 1.0, x86.ExplorationObservationWeight, 1e-12)
	assert.InDelta(t, -1.5, x86.RewardSum, 1e-12)
	assert.InDelta(t, -1.375, arm64.RewardSum, 1e-12)
}

func TestBuildReferenceAnchoredPriorAlsoWorksWithLegacyCoupledUCB1(t *testing.T) {
	source := testUCBTransferableKnowledge(
		"donor",
		UCB1,
		map[string]float64{
			"x86":   -4.0,
			"arm64": -3.5,
		},
	)

	prior, err := BuildWeakMABPrior(
		source,
		WeakMABPriorConfig{
			EquivalentObservationWeight: 0.5,
			MinRealObservationsPerArm:   1,
			UCB1ReferenceAnchor: &UCB1ReferenceAnchorConfig{
				Enabled:                   true,
				ReferenceArm:              "x86",
				TargetReferenceMeanReward: -6.0,
			},
		},
	)
	require.NoError(t, err)

	assert.InDelta(t, -6.0, prior.Arms["x86"].UCB1.MeanReward, 1e-12)
	assert.InDelta(t, -5.5, prior.Arms["arm64"].UCB1.MeanReward, 1e-12)
	assert.InDelta(t, 0.5, prior.Arms["x86"].UCB1.ObservationWeight, 1e-12)
	assert.InDelta(t, 0.5, prior.Arms["x86"].UCB1.ExplorationObservationWeight, 1e-12)
}

func TestApplyDecoupledPriorStoresIndependentWeights(t *testing.T) {
	prior := buildDecoupledApplicationPrior(t, 0.25, 1.0, false)

	target := NewUCB1DecoupledBandit("target", 0.8)
	target.InitArm("x86")
	target.InitArm("arm64")

	require.NoError(t, ApplyWeakMABPrior(target, prior))

	assert.InDelta(t, 0.25, target.Arms["x86"].PriorObservationWeight, 1e-12)
	assert.InDelta(t, 1.0, target.Arms["x86"].PriorExplorationObservationWeight, 1e-12)
	assert.Zero(t, target.Arms["x86"].Count)
	assert.Zero(t, target.Arms["x86"].RealCount)
}

func TestDecoupledDiagonalMatchesCoupledFormula(t *testing.T) {
	for _, weight := range []float64{0.25, 0.5, 1.0} {
		coupled := NewUCB1Bandit("coupled", 0.8)
		decoupled := NewUCB1DecoupledBandit("decoupled", 0.8)

		for _, arm := range []string{"arm64", "x86"} {
			coupled.InitArm(arm)
			decoupled.InitArm(arm)
		}

		// Identical live state and identical diagonal prior state.
		coupled.TotalCounts = 5
		decoupled.TotalCounts = 5
		coupled.TotalInFlight = 1
		decoupled.TotalInFlight = 1

		states := map[string]struct {
			count     int64
			avg       float64
			inFlight  int64
			priorMean float64
		}{
			"x86":   {count: 3, avg: -5.0, inFlight: 1, priorMean: -4.8},
			"arm64": {count: 2, avg: -4.9, inFlight: 0, priorMean: -4.7},
		}

		for arm, state := range states {
			for _, stats := range []*ArmStats{coupled.Arms[arm], decoupled.Arms[arm]} {
				stats.Count = state.count
				stats.AvgReward = state.avg
				stats.SumRewards = state.avg * float64(state.count)
				stats.InFlight = state.inFlight
				stats.PriorObservationWeight = weight
				stats.PriorExplorationObservationWeight = weight
				stats.PriorRewardSum = state.priorMean * weight
			}
		}

		coupledPriorTotal := coupled.totalPriorObservationWeightLocked()
		coupledTotal := float64(coupled.TotalCounts) + coupledPriorTotal + float64(coupled.TotalInFlight)
		if coupledTotal < 1.0 {
			coupledTotal = 1.0
		}

		decoupledPriorTotal := decoupled.totalPriorExplorationObservationWeightLocked()
		decoupledTotal := float64(decoupled.TotalCounts) + decoupledPriorTotal + float64(decoupled.TotalInFlight)
		if decoupledTotal < 1.0 {
			decoupledTotal = 1.0
		}

		assert.InDelta(t, coupledTotal, decoupledTotal, 1e-12)

		for arm := range states {
			oldStats := coupled.Arms[arm]
			newStats := decoupled.Arms[arm]

			oldCount := coupled.effectiveArmObservationWeightLocked(oldStats)
			newCount := decoupled.explorationArmObservationWeightLocked(newStats)
			oldMean := coupled.effectiveArmAverageRewardLocked(oldStats)
			newMean := decoupled.effectiveArmAverageRewardLocked(newStats)
			oldBonus := coupled.c * math.Sqrt(math.Log(coupledTotal)/oldCount)
			newBonus := decoupled.c * math.Sqrt(math.Log(decoupledTotal)/newCount)

			assert.InDelta(t, oldCount, newCount, 1e-12)
			assert.InDelta(t, oldMean, newMean, 1e-12)
			assert.InDelta(t, oldBonus, newBonus, 1e-12)
			assert.InDelta(t, oldMean+oldBonus, newMean+newBonus, 1e-12)
		}
	}
}

func TestDecoupledRewardWeightOneIsFiftyFiftyAfterFirstRealFeedback(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)

	prior := buildDecoupledApplicationPrior(t, 1.0, 1.0, false)
	target := NewUCB1DecoupledBandit("target", 0.0)
	target.InitArm("x86")
	target.InitArm("arm64")
	require.NoError(t, ApplyWeakMABPrior(target, prior))

	priorMean := target.Arms["x86"].PriorRewardSum / target.Arms["x86"].PriorObservationWeight

	feedback := ExecutionFeedback{
		DurationMs:  math.Exp(5.0), // reward = -5
		IsWarmStart: true,
	}
	target.UpdateReward("x86", nil, feedback)

	got := target.effectiveArmAverageRewardLocked(target.Arms["x86"])
	expected := (priorMean + (-5.0)) / 2.0
	assert.InDelta(t, expected, got, 1e-12)
}

func TestDecoupledKnowledgeOnlyUsesPriorWithoutExplorationBonus(t *testing.T) {
	prior := buildDecoupledApplicationPrior(t, 0.5, 1.0, false)
	target := NewUCB1DecoupledBandit("target", 0.0)
	target.InitArm("x86")
	target.InitArm("arm64")
	require.NoError(t, ApplyWeakMABPrior(target, prior))

	selected := target.SelectArm(nil)
	assert.Equal(t, "arm64", selected)
}

func TestBanditManagerCreatesDecoupledUCB1(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)

	viper.Set(config.MAB_POLICY, "UCB1Decoupled")
	viper.Set(config.MAB_UCB1_C, 0.8)

	manager := &BanditManager{
		bandits:   make(map[string]Policy),
		knownArms: []string{"x86", "arm64"},
	}

	policy := manager.GetBandit("target")
	assert.Equal(t, UCB1Decoupled, policy.GetType())
	_, ok := policy.(*UCB1DecoupledBandit)
	assert.True(t, ok)
}

func TestDecoupledTransferableKnowledgeExcludesPrior(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)

	prior := buildDecoupledApplicationPrior(t, 0.25, 1.0, false)
	target := NewUCB1DecoupledBandit("target", 0.8)
	target.InitArm("x86")
	target.InitArm("arm64")
	require.NoError(t, ApplyWeakMABPrior(target, prior))

	target.UpdateReward(
		"x86",
		nil,
		ExecutionFeedback{DurationMs: 100.0, IsWarmStart: true},
	)

	snapshot := target.TransferableKnowledge()
	assert.Equal(t, UCB1Decoupled, snapshot.Policy)
	assert.EqualValues(t, 1, snapshot.RealObservationCount)
	assert.EqualValues(t, 1, snapshot.Arms["x86"].RealObservationCount)
	assert.EqualValues(t, 0, snapshot.Arms["arm64"].RealObservationCount)
	assert.InDelta(t, -math.Log(100.0), snapshot.Arms["x86"].UCB1.RealSumRewards, 1e-12)
}

func TestDecoupledConfigRejectsAmbiguousLegacyWeight(t *testing.T) {
	source := testUCBTransferableKnowledge(
		"donor",
		UCB1Decoupled,
		map[string]float64{"x86": -4.0},
	)

	_, err := BuildWeakMABPrior(
		source,
		WeakMABPriorConfig{
			EquivalentObservationWeight:  0.5,
			RewardObservationWeight:      0.25,
			ExplorationObservationWeight: 1.0,
			MinRealObservationsPerArm:    1,
		},
	)
	assert.Error(t, err)
}

func testUCBTransferableKnowledge(
	functionName string,
	policy BanditType,
	means map[string]float64,
) TransferableMABKnowledge {
	const realCount int64 = 4

	arms := make(map[string]TransferableArmKnowledge, len(means))
	var total int64

	for arm, mean := range means {
		arms[arm] = TransferableArmKnowledge{
			RealObservationCount: realCount,
			UCB1: &TransferableUCB1ArmKnowledge{
				RealSumRewards: mean * float64(realCount),
				RealAvgReward:  mean,
			},
		}
		total += realCount
	}

	return TransferableMABKnowledge{
		SchemaVersion:        TransferableMABKnowledgeSchemaVersion,
		FunctionName:         functionName,
		Policy:               policy,
		HasRealKnowledge:     total > 0,
		RealObservationCount: total,
		Arms:                 arms,
	}
}

func buildDecoupledApplicationPrior(
	t *testing.T,
	rewardWeight float64,
	explorationWeight float64,
	anchored bool,
) WeakMABPrior {
	t.Helper()

	source := testUCBTransferableKnowledge(
		"donor",
		UCB1Decoupled,
		map[string]float64{
			"x86":   -6.0,
			"arm64": -5.0,
		},
	)

	config := WeakMABPriorConfig{
		RewardObservationWeight:      rewardWeight,
		ExplorationObservationWeight: explorationWeight,
		MinRealObservationsPerArm:    1,
	}

	if anchored {
		config.UCB1ReferenceAnchor = &UCB1ReferenceAnchorConfig{
			Enabled:                   true,
			ReferenceArm:              "x86",
			TargetReferenceMeanReward: -7.0,
		}
	}

	prior, err := BuildWeakMABPrior(source, config)
	require.NoError(t, err)
	return prior
}
