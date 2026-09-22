package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newMaterializedPriorTestManager(
	arms ...string,
) *BanditManager {
	return &BanditManager{
		bandits:   make(map[string]Policy),
		knownArms: append([]string(nil), arms...),
	}
}

func TestInitializeTargetFromMaterializedPriorAppliesToFreshUCB1(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	// Exploitation-only target.
	viper.Set(
		config.MAB_UCB1_C,
		0.0,
	)

	prior := buildUCB1ApplicationPrior(
		t,
		0.25,
		map[string]TransferableUCB1ArmKnowledge{
			"amd64": {
				RealSumRewards: -16.0,
				RealAvgReward:  -4.0,
			},
			"arm64": {
				RealSumRewards: -12.0,
				RealAvgReward:  -3.0,
			},
		},
	)

	manager :=
		newMaterializedPriorTestManager(
			"amd64",
			"arm64",
		)

	result, err :=
		manager.InitializeTargetFromMaterializedPrior(
			"target-materialized",
			prior,
		)

	require.NoError(t, err)

	assert.True(t, result.Applied)
	assert.Equal(
		t,
		RuntimeTransferReasonApplied,
		result.Reason,
	)
	assert.Equal(t, UCB1, result.Policy)
	assert.Equal(
		t,
		prior.DonorFunctionName,
		result.DonorFunctionName,
	)

	target :=
		manager.GetBandit(
			"target-materialized",
		).(*UCB1Bandit)

	assert.Equal(
		t,
		prior.DonorFunctionName,
		target.PriorDonorFunctionName,
	)

	// Materialized prior must not become fake real experience.
	assert.Zero(t, target.TotalCounts)
	assert.Zero(
		t,
		target.Arms["amd64"].Count,
	)
	assert.Zero(
		t,
		target.Arms["arm64"].Count,
	)
	assert.Zero(
		t,
		target.Arms["amd64"].RealCount,
	)
	assert.Zero(
		t,
		target.Arms["arm64"].RealCount,
	)

	assert.InDelta(
		t,
		0.25,
		target.Arms["amd64"].
			PriorObservationWeight,
		1e-12,
	)
	assert.InDelta(
		t,
		0.25,
		target.Arms["arm64"].
			PriorObservationWeight,
		1e-12,
	)

	// With c=0, selection is exploitation-only.
	// arm64 has the better prior reward (-3 > -4).
	selected :=
		target.SelectArmFrom(
			nil,
			[]string{
				"amd64",
				"arm64",
			},
		)

	assert.Equal(
		t,
		"arm64",
		selected,
	)
}

func TestInitializeTargetFromMaterializedPriorAppliesToFreshUCB1Decoupled(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1Decoupled",
	)

	// Start from the same coupled reward prior used by the existing
	// materialized UCB1 test.
	prior := buildUCB1ApplicationPrior(
		t,
		0.25,
		map[string]TransferableUCB1ArmKnowledge{
			"amd64": {
				RealSumRewards: -16.0,
				RealAvgReward:  -4.0,
			},
			"arm64": {
				RealSumRewards: -12.0,
				RealAvgReward:  -3.0,
			},
		},
	)

	// Derive the decoupled twin:
	// reward knowledge stays at w_R=0.25,
	// exploration pseudo-count becomes w_E=1.0.
	prior.Policy = UCB1Decoupled

	prior.Config.EquivalentObservationWeight = 0.0
	prior.Config.RewardObservationWeight = 0.25
	prior.Config.ExplorationObservationWeight = 1.0

	for armName, armPrior := range prior.Arms {
		if !armPrior.Transferred {
			continue
		}

		require.NotNil(t, armPrior.UCB1)

		armPrior.AppliedExplorationObservationWeight = 1.0
		armPrior.UCB1.ExplorationObservationWeight = 1.0

		prior.Arms[armName] = armPrior
	}

	manager := newMaterializedPriorTestManager(
		"amd64",
		"arm64",
	)

	result, err := manager.InitializeTargetFromMaterializedPrior(
		"target-materialized-decoupled",
		prior,
	)

	require.NoError(t, err)

	assert.True(t, result.Applied)
	assert.Equal(
		t,
		RuntimeTransferReasonApplied,
		result.Reason,
	)
	assert.Equal(
		t,
		UCB1Decoupled,
		result.Policy,
	)
	assert.Equal(
		t,
		prior.DonorFunctionName,
		result.DonorFunctionName,
	)

	target := manager.GetBandit(
		"target-materialized-decoupled",
	).(*UCB1DecoupledBandit)

	assert.Equal(
		t,
		prior.DonorFunctionName,
		target.PriorDonorFunctionName,
	)

	// A materialized prior must not create fake target executions.
	assert.Zero(t, target.TotalCounts)

	for _, arm := range []string{"amd64", "arm64"} {
		stats := target.Arms[arm]

		assert.Zero(t, stats.Count)
		assert.Zero(t, stats.RealCount)

		assert.InDelta(
			t,
			0.25,
			stats.PriorObservationWeight,
			1e-12,
		)

		assert.InDelta(
			t,
			1.0,
			stats.PriorExplorationObservationWeight,
			1e-12,
		)

		assert.InDelta(
			t,
			prior.Arms[arm].UCB1.RewardSum,
			stats.PriorRewardSum,
			1e-12,
		)
	}

	assert.InDelta(
		t,
		-1.0,
		target.Arms["amd64"].PriorRewardSum,
		1e-12,
	)

	assert.InDelta(
		t,
		-0.75,
		target.Arms["arm64"].PriorRewardSum,
		1e-12,
	)
}

func TestInitializeTargetFromMaterializedPriorRejectsExistingTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	prior := buildUCB1ApplicationPrior(
		t,
		0.25,
		map[string]TransferableUCB1ArmKnowledge{
			"amd64": {
				RealSumRewards: -16.0,
				RealAvgReward:  -4.0,
			},
		},
	)

	manager :=
		newMaterializedPriorTestManager(
			"amd64",
		)

	manager.GetBandit(
		"existing-target",
	)

	_, err :=
		manager.InitializeTargetFromMaterializedPrior(
			"existing-target",
			prior,
		)

	require.Error(t, err)

	assert.Contains(
		t,
		err.Error(),
		"already has a MAB policy",
	)
}

func TestInitializeTargetFromMaterializedPriorRejectsPolicyMismatch(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	prior := buildUCB1ApplicationPrior(
		t,
		0.25,
		map[string]TransferableUCB1ArmKnowledge{
			"amd64": {
				RealSumRewards: -16.0,
				RealAvgReward:  -4.0,
			},
		},
	)

	// The frozen prior is UCB1, while the new target
	// runtime is deliberately configured as UCB1Decoupled.
	viper.Set(
		config.MAB_POLICY,
		"UCB1Decoupled",
	)

	manager :=
		newMaterializedPriorTestManager(
			"amd64",
		)

	_, err :=
		manager.InitializeTargetFromMaterializedPrior(
			"target-policy-mismatch",
			prior,
		)

	require.Error(t, err)

	assert.Contains(
		t,
		err.Error(),
		"does not match target policy",
	)
}
