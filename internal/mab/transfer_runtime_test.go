package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRuntimeTransferUCB1WorksWithSingleLocalArm(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	donor :=
		manager.
			GetBandit(
				"donor-local",
			).(*UCB1Bandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			nil,
			ExecutionFeedback{
				DurationMs:  10.0,
				IsWarmStart: true,
			},
		)
	}

	result, err :=
		manager.InitializeTargetFromDonor(
			"target-local",
			"donor-local",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 0.5,

				MinRealObservationsPerArm: 2,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.True(
		t,
		result.Applied,
	)

	assert.Equal(
		t,
		RuntimeTransferReasonApplied,
		result.Reason,
	)

	assert.Equal(
		t,
		UCB1,
		result.Policy,
	)

	assert.Equal(
		t,
		int64(4),
		result.Prior.
			SourceRealObservationCount,
	)

	target :=
		manager.
			GetBandit(
				"target-local",
			).(*UCB1Bandit)

	stats :=
		target.Arms["x86-local"]

	assert.Equal(
		t,
		"donor-local",
		target.PriorDonorFunctionName,
	)

	// The target has received prior knowledge, but no fake target execution
	// has been created.
	assert.Zero(
		t,
		target.TotalCounts,
	)

	assert.Zero(
		t,
		stats.Count,
	)

	assert.Zero(
		t,
		stats.RealCount,
	)

	assert.InDelta(
		t,
		0.5,
		stats.PriorObservationWeight,
		1e-12,
	)

	assert.NotZero(
		t,
		stats.PriorRewardSum,
	)

	targetSnapshot :=
		target.TransferableKnowledge()

	assert.False(
		t,
		targetSnapshot.HasRealKnowledge,
	)

	assert.Zero(
		t,
		targetSnapshot.RealObservationCount,
	)

	// Now execute the target itself. Its own experience must remain separate
	// from the donor prior.
	target.UpdateReward(
		"x86-local",
		nil,
		ExecutionFeedback{
			DurationMs:  20.0,
			IsWarmStart: true,
		},
	)

	targetSnapshot =
		target.TransferableKnowledge()

	assert.True(
		t,
		targetSnapshot.HasRealKnowledge,
	)

	assert.Equal(
		t,
		int64(1),
		targetSnapshot.RealObservationCount,
	)

	assert.Equal(
		t,
		int64(1),
		targetSnapshot.
			Arms["x86-local"].
			RealObservationCount,
	)
}

func TestRuntimeTransferLinUCBWorksWithSingleLocalArm(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"LinUCB",
	)

	viper.Set(
		config.MAB_LINUCB_ALPHA,
		0.1,
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	donor :=
		manager.
			GetBandit(
				"donor-lin-local",
			).(*LinUCBDisjointPolicy)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				"x86-local": 0.35,
			},
		}

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			ctx,
			ExecutionFeedback{
				DurationMs:  12.0,
				IsWarmStart: true,
			},
		)
	}

	result, err :=
		manager.InitializeTargetFromDonor(
			"target-lin-local",
			"donor-lin-local",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 0.5,

				MinRealObservationsPerArm: 2,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.True(
		t,
		result.Applied,
	)

	assert.Equal(
		t,
		LinUCB,
		result.Policy,
	)

	target :=
		manager.
			GetBandit(
				"target-lin-local",
			).(*LinUCBDisjointPolicy)

	state :=
		target.Arms["x86-local"]

	assert.Equal(
		t,
		"donor-lin-local",
		target.PriorDonorFunctionName,
	)

	assert.InDelta(
		t,
		0.5,
		state.PriorObservationWeight,
		1e-12,
	)

	assert.False(
		t,
		denseIsZero(
			state.PriorAContribution,
			target.Dim,
			target.Dim,
		),
	)

	assert.False(
		t,
		vectorIsZero(
			state.PriorBContribution,
			target.Dim,
		),
	)

	// Again, donor prior must not become target real knowledge.
	assert.Zero(
		t,
		state.RealObservationCount,
	)

	assert.True(
		t,
		denseIsZero(
			state.RealAContribution,
			target.Dim,
			target.Dim,
		),
	)

	assert.True(
		t,
		vectorIsZero(
			state.RealBContribution,
			target.Dim,
		),
	)

	targetSnapshot :=
		target.TransferableKnowledge()

	assert.False(
		t,
		targetSnapshot.HasRealKnowledge,
	)

	assert.Zero(
		t,
		targetSnapshot.RealObservationCount,
	)
}

func TestRuntimeTransferCreatesFreshTargetWhenDonorHasNoUsablePrior(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	manager.GetBandit(
		"empty-donor",
	)

	result, err :=
		manager.InitializeTargetFromDonor(
			"fresh-target",
			"empty-donor",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.False(
		t,
		result.Applied,
	)

	assert.Equal(
		t,
		RuntimeTransferReasonNoTransferablePrior,
		result.Reason,
	)

	assert.False(
		t,
		result.Prior.HasPrior,
	)

	target, exists :=
		manager.bandits["fresh-target"]

	require.True(
		t,
		exists,
	)

	ucb :=
		target.(*UCB1Bandit)

	assert.Empty(
		t,
		ucb.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		ucb.Arms["x86-local"].PriorObservationWeight,
	)
}

func TestRuntimeTransferRejectsUnknownDonorWithoutPublishingTarget(
	t *testing.T,
) {
	viper.Reset()
	t.Cleanup(
		viper.Reset,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	_, err :=
		manager.InitializeTargetFromDonor(
			"target",
			"missing-donor",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["target"]

	assert.False(
		t,
		exists,
	)
}

func TestRuntimeTransferRejectsExistingTarget(
	t *testing.T,
) {
	viper.Reset()
	t.Cleanup(
		viper.Reset,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	manager.GetBandit(
		"donor",
	)

	manager.GetBandit(
		"target",
	)

	_, err :=
		manager.InitializeTargetFromDonor(
			"target",
			"donor",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.Error(
		t,
		err,
	)
}

func TestRuntimeTransferPolicyMismatchDoesNotPublishTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	donor :=
		manager.
			GetBandit(
				"ucb-donor",
			).(*UCB1Bandit)

	donor.UpdateReward(
		"x86-local",
		nil,
		ExecutionFeedback{
			DurationMs:  10.0,
			IsWarmStart: true,
		},
	)

	// Simulate a configuration change after donor creation.
	viper.Set(
		config.MAB_POLICY,
		"LinUCB",
	)

	_, err :=
		manager.InitializeTargetFromDonor(
			"lin-target",
			"ucb-donor",
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["lin-target"]

	assert.False(
		t,
		exists,
	)
}

func newRuntimeTransferTestManager(
	arms ...string,
) *BanditManager {
	return &BanditManager{
		bandits: make(
			map[string]Policy,
		),

		knownArms: append(
			[]string(nil),
			arms...,
		),
	}
}
