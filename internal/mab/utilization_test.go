package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetMABConfigAfterTest(t *testing.T) {
	t.Helper()

	viper.Reset()
	GlobalColdStartStats.Reset()

	t.Cleanup(func() {
		viper.Reset()
		GlobalColdStartStats.Reset()
	})
}

func TestUCB1ClassicIgnoresUtilizationContext(
	t *testing.T,
) {
	bandit := NewUCB1Bandit(
		"ucb1-classic-context-free-test",
		0.0,
	)

	bandit.InitArm("faster-but-busy")
	bandit.InitArm("slower-but-free")

	bandit.Arms["faster-but-busy"].Count = 1
	bandit.Arms["faster-but-busy"].AvgReward = 1.0

	bandit.Arms["slower-but-free"].Count = 1
	bandit.Arms["slower-but-free"].AvgReward = 0.0

	bandit.TotalCounts = 2

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"faster-but-busy": 1.00,
			"slower-but-free": 0.00,
		},
	}

	selected := bandit.SelectArmFrom(
		ctx,
		[]string{
			"faster-but-busy",
			"slower-but-free",
		},
	)

	assert.Equal(
		t,
		"faster-but-busy",
		selected,
	)
}

func TestLinUCBUsesLegacyUtilizationTransformation(
	t *testing.T,
) {
	bandit := NewLinUCBDisjointPolicy(
		"linucb-context-feature-test",
		0.1,
	)

	lowUtilization := 0.20
	highUtilization := 0.90

	low := bandit.computeFeatures(lowUtilization)
	high := bandit.computeFeatures(highUtilization)

	require.Equal(t, 2, low.Len())
	require.Equal(t, 2, high.Len())

	assert.InDelta(t, 1.0, low.AtVec(0), 1e-9)
	assert.InDelta(t, 1.0, high.AtVec(0), 1e-9)

	expectedLow :=
		1.0 /
			(1.0 - lowUtilization + linUCBUtilizationEpsilon)

	expectedHigh :=
		1.0 /
			(1.0 - highUtilization + linUCBUtilizationEpsilon)

	assert.InDelta(
		t,
		expectedLow,
		low.AtVec(1),
		1e-9,
	)

	assert.InDelta(
		t,
		expectedHigh,
		high.AtVec(1),
		1e-9,
	)

	assert.Greater(
		t,
		high.AtVec(1),
		low.AtVec(1),
	)
}

func TestLinUCBRewardHasNoExplicitUtilizationPenalty(
	t *testing.T,
) {
	resetMABConfigAfterTest(t)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.0,
	)
	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.0,
	)

	const (
		arm       = "shared-ring"
		duration  = 100.0
		lowUsage  = 0.20
		highUsage = 0.90
	)

	lowBandit := NewLinUCBDisjointPolicy(
		"linucb-low-utilization-reward-test",
		0.0,
	)

	highBandit := NewLinUCBDisjointPolicy(
		"linucb-high-utilization-reward-test",
		0.0,
	)

	lowBandit.InitArm(arm)
	highBandit.InitArm(arm)

	feedback := ExecutionFeedback{
		DurationMs:   duration,
		IsWarmStart:  true,
		CostFactor:   1.0,
		EnergyFactor: 1.0,
	}

	lowContext := &Context{
		ArchMemUsage: map[string]float64{
			arm: lowUsage,
		},
	}

	highContext := &Context{
		ArchMemUsage: map[string]float64{
			arm: highUsage,
		},
	}

	lowBandit.UpdateReward(
		arm,
		lowContext,
		feedback,
	)

	highBandit.UpdateReward(
		arm,
		highContext,
		feedback,
	)

	lowState, ok := lowBandit.Arms[arm]
	require.True(t, ok)

	highState, ok := highBandit.Arms[arm]
	require.True(t, ok)

	expectedReward := -math.Log(duration)

	assert.InDelta(
		t,
		expectedReward,
		lowState.b.AtVec(0),
		1e-9,
	)

	assert.InDelta(
		t,
		expectedReward,
		highState.b.AtVec(0),
		1e-9,
	)

	assert.NotEqual(
		t,
		lowState.b.AtVec(1),
		highState.b.AtVec(1),
	)
}

func TestLinUCBLearnsContextDependentSelectionWithoutExplicitPenalty(
	t *testing.T,
) {
	resetMABConfigAfterTest(t)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.0,
	)
	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.0,
	)

	bandit := NewLinUCBDisjointPolicy(
		"linucb-context-dependent-selection-test",
		0.0,
	)

	const (
		contextSensitiveArm = "context-sensitive"
		stableArm           = "stable"
	)

	bandit.InitArm(contextSensitiveArm)
	bandit.InitArm(stableArm)

	warmFeedback := func(
		durationMs float64,
	) ExecutionFeedback {
		return ExecutionFeedback{
			DurationMs:   durationMs,
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		}
	}

	contextForArm := func(
		arm string,
		utilization float64,
	) *Context {
		return &Context{
			ArchMemUsage: map[string]float64{
				arm: utilization,
			},
		}
	}

	for i := 0; i < 10; i++ {
		bandit.UpdateReward(
			contextSensitiveArm,
			contextForArm(
				contextSensitiveArm,
				0.10,
			),
			warmFeedback(10.0),
		)

		bandit.UpdateReward(
			contextSensitiveArm,
			contextForArm(
				contextSensitiveArm,
				0.90,
			),
			warmFeedback(1000.0),
		)

		bandit.UpdateReward(
			stableArm,
			contextForArm(
				stableArm,
				0.10,
			),
			warmFeedback(30.0),
		)

		bandit.UpdateReward(
			stableArm,
			contextForArm(
				stableArm,
				0.90,
			),
			warmFeedback(30.0),
		)
	}

	lowUtilizationContext := &Context{
		ArchMemUsage: map[string]float64{
			contextSensitiveArm: 0.10,
			stableArm:           0.10,
		},
	}

	highUtilizationContext := &Context{
		ArchMemUsage: map[string]float64{
			contextSensitiveArm: 0.90,
			stableArm:           0.90,
		},
	}

	assert.Equal(
		t,
		contextSensitiveArm,
		bandit.SelectArmFrom(
			lowUtilizationContext,
			[]string{
				contextSensitiveArm,
				stableArm,
			},
		),
	)

	assert.Equal(
		t,
		stableArm,
		bandit.SelectArmFrom(
			highUtilizationContext,
			[]string{
				contextSensitiveArm,
				stableArm,
			},
		),
	)
}
