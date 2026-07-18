package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetMABConfigAfterTest(t *testing.T) {
	t.Helper()

	viper.Reset()
	t.Cleanup(viper.Reset)
}

func TestUtilizationPenaltyUsesThresholdAndQuadraticGrowth(
	t *testing.T,
) {
	assert.InDelta(
		t,
		0.0,
		utilizationPenalty(0.50, 0.70),
		1e-9,
	)

	assert.InDelta(
		t,
		0.0,
		utilizationPenalty(0.70, 0.70),
		1e-9,
	)

	assert.InDelta(
		t,
		0.25,
		utilizationPenalty(0.85, 0.70),
		1e-9,
	)

	assert.InDelta(
		t,
		1.0,
		utilizationPenalty(1.00, 0.70),
		1e-9,
	)
}

func TestUtilizationPenaltyClampsOutOfRangeValues(
	t *testing.T,
) {
	assert.InDelta(
		t,
		0.0,
		utilizationPenalty(-0.50, 0.70),
		1e-9,
	)

	assert.InDelta(
		t,
		1.0,
		utilizationPenalty(1.50, 0.70),
		1e-9,
	)

	assert.InDelta(
		t,
		0.0,
		utilizationPenalty(0.90, 1.00),
		1e-9,
	)
}

func TestUCB1UtilizationAwarePenalizesCurrentlySaturatedArm(
	t *testing.T,
) {
	resetMABConfigAfterTest(t)

	viper.Set(
		config.MAB_UCB1_UTILIZATION_WEIGHT,
		1.0,
	)
	viper.Set(
		config.MAB_UCB1_UTILIZATION_THRESHOLD,
		0.70,
	)

	bandit := NewUCB1UtilizationAwareBandit(
		"ucb1-utilization-test",
		0.0,
	)

	bandit.InitArm("saturated")
	bandit.InitArm("available")

	bandit.Arms["saturated"].Count = 1
	bandit.Arms["saturated"].AvgReward = 0.0

	bandit.Arms["available"].Count = 1
	bandit.Arms["available"].AvgReward = 0.0

	bandit.TotalCounts = 2

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"saturated": 1.00,
			"available": 0.20,
		},
	}

	selected := bandit.SelectArmFrom(
		ctx,
		[]string{
			"saturated",
			"available",
		},
	)

	assert.Equal(
		t,
		"available",
		selected,
	)
}

func TestUCB1ClassicIgnoresUtilizationContext(
	t *testing.T,
) {
	resetMABConfigAfterTest(t)

	viper.Set(
		config.MAB_UCB1_UTILIZATION_WEIGHT,
		100.0,
	)
	viper.Set(
		config.MAB_UCB1_UTILIZATION_THRESHOLD,
		0.70,
	)

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

func TestLinUCBUsesUtilizationAsContextualFeature(
	t *testing.T,
) {
	bandit := NewLinUCBDisjointPolicy(
		"linucb-context-feature-test",
		0.1,
	)

	low := bandit.computeFeatures(0.20)
	high := bandit.computeFeatures(0.90)

	require.Equal(t, 2, low.Len())
	require.Equal(t, 2, high.Len())

	assert.InDelta(t, 1.0, low.AtVec(0), 1e-9)
	assert.InDelta(t, 1.0, high.AtVec(0), 1e-9)

	assert.Greater(
		t,
		high.AtVec(1),
		low.AtVec(1),
	)
}
