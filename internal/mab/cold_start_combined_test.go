package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestColdStartExecutionUsesConcreteNodeFactors(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeExecution,
		),
	)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.5,
	)

	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.25,
	)

	bandit :=
		NewUCB1Bandit(
			"cold-node-factor-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:    10.0,
			InitTimeMs:    400.0,
			IsWarmStart:   false,
			NodeName:      "expensive-node",
			ExecutionNode: "expensive-node",
			CostFactor:    3.0,
			EnergyFactor:  2.0,
		},
	)

	expectedReward :=
		-math.Log(10.0) -
			0.5*3.0 -
			0.25*2.0

	stats, ok :=
		bandit.Arms[arm]

	require.True(t, ok)

	assert.Equal(
		t,
		int64(1),
		stats.Count,
	)

	assert.InDelta(
		t,
		expectedReward,
		stats.AvgReward,
		1e-9,
	)

	snapshot :=
		GlobalColdStartStats.
			Snapshot(
				"cold-node-factor-test",
				arm,
			)

	assert.Equal(
		t,
		int64(1),
		snapshot.ColdObserved,
	)

	assert.Equal(
		t,
		int64(1),
		snapshot.ColdAccepted,
	)

	assert.Zero(
		t,
		snapshot.ColdSkipped,
	)
}

func TestColdStartSkipIgnoresConcreteNodeFactors(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeSkip,
		),
	)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.5,
	)

	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.25,
	)

	bandit :=
		NewUCB1Bandit(
			"cold-skip-factor-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:   10.0,
			InitTimeMs:   400.0,
			IsWarmStart:  false,
			CostFactor:   3.0,
			EnergyFactor: 2.0,
		},
	)

	assert.Zero(
		t,
		bandit.Arms[arm].Count,
	)

	assert.Zero(
		t,
		bandit.TotalCounts,
	)

	snapshot :=
		GlobalColdStartStats.
			Snapshot(
				"cold-skip-factor-test",
				arm,
			)

	assert.Equal(
		t,
		int64(1),
		snapshot.ColdObserved,
	)

	assert.Equal(
		t,
		int64(1),
		snapshot.ColdSkipped,
	)

	assert.Zero(
		t,
		snapshot.ColdAccepted,
	)
}

func TestColdStartExecutionLinUCBUsesDecisionContext(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeExecution,
		),
	)

	bandit :=
		NewLinUCBDisjointPolicy(
			"cold-linucb-context-test",
			0.1,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				arm: 0.35,
			},
		}

	features :=
		bandit.computeFeatures(
			0.35,
		)

	bandit.UpdateReward(
		arm,
		ctx,
		ExecutionFeedback{
			DurationMs:   10.0,
			InitTimeMs:   300.0,
			IsWarmStart:  false,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	state, ok :=
		bandit.Arms[arm]

	require.True(t, ok)

	expectedReward :=
		-math.Log(10.0)

	for i := 0; i < bandit.Dim; i++ {
		assert.InDelta(
			t,
			expectedReward*
				features.AtVec(i),
			state.b.AtVec(i),
			1e-9,
		)
	}
}

func TestColdStartExecutionRespectsActionMask(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeExecution,
		),
	)

	bandit :=
		NewUCB1Bandit(
			"cold-action-mask-test",
			0.0,
		)

	bandit.InitArm("arm-a")
	bandit.InitArm("arm-b")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-b",
			},
		)

	require.Equal(
		t,
		"arm-b",
		selected,
	)

	feedback := ExecutionFeedback{
		DurationMs:   10.0,
		IsWarmStart:  false,
		CostFactor:   1.0,
		EnergyFactor: 1.0,
	}

	bandit.ResolveSelection(
		selected,
		selected,
		nil,
		&feedback,
		nil,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-b"].Count,
	)
}
