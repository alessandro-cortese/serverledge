package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRuntimeTransferAllowsUCB1DonorToDecoupledTarget(t *testing.T) {
	resetUCBFamilyCompatibilityConfig(t)

	viper.Set(config.MAB_POLICY, "UCB1")

	manager := newUCBFamilyCompatibilityTestManager("x86", "arm64")
	donor := manager.GetBandit("donor-ucb1").(*UCB1Bandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86",
			nil,
			ExecutionFeedback{
				DurationMs:  100.0,
				IsWarmStart: true,
			},
		)
		donor.UpdateReward(
			"arm64",
			nil,
			ExecutionFeedback{
				DurationMs:  80.0,
				IsWarmStart: true,
			},
		)
	}

	// Only the target changes formula. The donor remains the classic UCB1
	// policy that collected the real feedback above.
	viper.Set(config.MAB_POLICY, "UCB1Decoupled")

	targetReferenceMeanReward := -math.Log(200.0)

	result, err := manager.InitializeTargetFromDonor(
		"target-decoupled",
		"donor-ucb1",
		WeakMABPriorConfig{
			RewardObservationWeight:      0.25,
			ExplorationObservationWeight: 1.0,
			MinRealObservationsPerArm:    2,
			UCB1ReferenceAnchor: &UCB1ReferenceAnchorConfig{
				Enabled:                   true,
				ReferenceArm:              "x86",
				TargetReferenceMeanReward: targetReferenceMeanReward,
			},
		},
	)

	require.NoError(t, err)
	assert.True(t, result.Applied)
	assert.Equal(t, UCB1Decoupled, result.Policy)
	assert.Equal(t, UCB1Decoupled, result.Prior.Policy)

	// The donor policy itself is unchanged.
	assert.Equal(t, UCB1, donor.GetType())

	target, ok := manager.bandits["target-decoupled"].(*UCB1DecoupledBandit)
	require.True(t, ok)

	x86 := target.Arms["x86"]
	arm64 := target.Arms["arm64"]

	assert.InDelta(t, 0.25, x86.PriorObservationWeight, 1e-12)
	assert.InDelta(t, 1.0, x86.PriorExplorationObservationWeight, 1e-12)
	assert.InDelta(t, 0.25, arm64.PriorObservationWeight, 1e-12)
	assert.InDelta(t, 1.0, arm64.PriorExplorationObservationWeight, 1e-12)

	// Reference anchoring calibrates the target x86 prior to the target's own
	// measured x86 reward while preserving the donor x86->ARM reward delta.
	assert.InDelta(
		t,
		targetReferenceMeanReward,
		x86.PriorRewardSum/x86.PriorObservationWeight,
		1e-12,
	)

	donorDelta := -math.Log(80.0) - (-math.Log(100.0))
	assert.InDelta(
		t,
		targetReferenceMeanReward+donorDelta,
		arm64.PriorRewardSum/arm64.PriorObservationWeight,
		1e-12,
	)
}

func TestRuntimeTransferAllowsDecoupledDonorToCoupledTarget(t *testing.T) {
	resetUCBFamilyCompatibilityConfig(t)

	viper.Set(config.MAB_POLICY, "UCB1Decoupled")

	manager := newUCBFamilyCompatibilityTestManager("x86", "arm64")
	donor := manager.GetBandit("donor-decoupled").(*UCB1DecoupledBandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86",
			nil,
			ExecutionFeedback{
				DurationMs:  100.0,
				IsWarmStart: true,
			},
		)
		donor.UpdateReward(
			"arm64",
			nil,
			ExecutionFeedback{
				DurationMs:  125.0,
				IsWarmStart: true,
			},
		)
	}

	// Reuse the exact same donor real-feedback state with the historical
	// coupled target formula.
	viper.Set(config.MAB_POLICY, "UCB1")

	result, err := manager.InitializeTargetFromDonor(
		"target-coupled",
		"donor-decoupled",
		WeakMABPriorConfig{
			EquivalentObservationWeight: 0.5,
			MinRealObservationsPerArm:   2,
		},
	)

	require.NoError(t, err)
	assert.True(t, result.Applied)
	assert.Equal(t, UCB1, result.Policy)
	assert.Equal(t, UCB1, result.Prior.Policy)
	assert.Equal(t, UCB1Decoupled, donor.GetType())

	target, ok := manager.bandits["target-coupled"].(*UCB1Bandit)
	require.True(t, ok)

	for _, arm := range []string{"x86", "arm64"} {
		stats := target.Arms[arm]
		assert.InDelta(t, 0.5, stats.PriorObservationWeight, 1e-12)
		assert.InDelta(t, 0.5, stats.PriorExplorationObservationWeight, 1e-12)
	}
}

func TestRuntimeTransferStillRejectsUCB1DonorToLinUCBTarget(t *testing.T) {
	resetUCBFamilyCompatibilityConfig(t)

	viper.Set(config.MAB_POLICY, "UCB1")

	manager := newUCBFamilyCompatibilityTestManager("x86")
	donor := manager.GetBandit("donor-ucb1").(*UCB1Bandit)
	donor.UpdateReward(
		"x86",
		nil,
		ExecutionFeedback{
			DurationMs:  100.0,
			IsWarmStart: true,
		},
	)

	viper.Set(config.MAB_POLICY, "LinUCB")

	_, err := manager.InitializeTargetFromDonor(
		"target-linucb",
		"donor-ucb1",
		WeakMABPriorConfig{
			EquivalentObservationWeight: 0.5,
			MinRealObservationsPerArm:   1,
		},
	)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "incompatible transfer policies")
	_, exists := manager.bandits["target-linucb"]
	assert.False(t, exists)
}

func resetUCBFamilyCompatibilityConfig(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)
	viper.Set(config.MAB_REWARD_MODE, "latency")
}

func newUCBFamilyCompatibilityTestManager(arms ...string) *BanditManager {
	return &BanditManager{
		bandits: make(map[string]Policy),
		knownArms: append(
			[]string(nil),
			arms...,
		),
	}
}
