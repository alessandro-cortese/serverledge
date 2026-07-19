package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUCB1RejectsInvalidDurationFeedback(
	t *testing.T,
) {
	tests := []struct {
		name     string
		duration float64
	}{
		{
			name:     "zero duration",
			duration: 0,
		},
		{
			name:     "negative duration",
			duration: -1,
		},
		{
			name:     "nan duration",
			duration: math.NaN(),
		},
		{
			name:     "infinite duration",
			duration: math.Inf(1),
		},
	}

	for _, test := range tests {

		t.Run(
			test.name,
			func(t *testing.T) {
				resetExecutionFeedbackConfig(t)

				bandit :=
					NewUCB1Bandit(
						"invalid-duration-test",
						0.0,
					)

				const arm = "shared-ring"

				bandit.InitArm(arm)

				bandit.UpdateReward(
					arm,
					nil,
					ExecutionFeedback{
						DurationMs:   test.duration,
						IsWarmStart:  true,
						CostFactor:   1.0,
						EnergyFactor: 1.0,
					},
				)

				stats, ok :=
					bandit.Arms[arm]

				require.True(t, ok)

				assert.Zero(
					t,
					stats.Count,
				)

				assert.Zero(
					t,
					bandit.TotalCounts,
				)

				assert.Zero(
					t,
					stats.SumRewards,
				)

				snapshot :=
					GlobalColdStartStats.
						Snapshot(
							"invalid-duration-test",
							arm,
						)

				assert.Equal(
					t,
					int64(1),
					snapshot.WarmObserved,
				)

				assert.Equal(
					t,
					int64(1),
					snapshot.InvalidFeedback,
				)
			},
		)
	}
}

func TestLinUCBRejectsInvalidDurationFeedback(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-invalid-duration-test",
			0.1,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				arm: 0.20,
			},
		}

	bandit.UpdateReward(
		arm,
		ctx,
		ExecutionFeedback{
			DurationMs:   math.NaN(),
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	state, ok :=
		bandit.Arms[arm]

	require.True(t, ok)

	for i := 0; i < bandit.Dim; i++ {
		for j := 0; j < bandit.Dim; j++ {
			expected := 0.0

			if i == j {
				expected = 1.0
			}

			assert.InDelta(
				t,
				expected,
				state.A.At(i, j),
				1e-9,
			)
		}

		assert.Zero(
			t,
			state.b.AtVec(i),
		)
	}
}

func TestInvalidObservabilityMetricsDoNotBlockValidReward(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"invalid-observability-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:       10.0,
			ResponseTimeMs:   math.NaN(),
			InitTimeMs:       -1.0,
			QueueingTimeMs:   math.Inf(1),
			OffloadLatencyMs: -2.0,
			IsWarmStart:      true,
			CostFactor:       1.0,
			EnergyFactor:     1.0,
		},
	)

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
		-math.Log(10.0),
		stats.AvgReward,
		1e-9,
	)
}

func TestNonFiniteNodeFactorsRejectFeedback(
	t *testing.T,
) {
	tests := []struct {
		name         string
		costFactor   float64
		energyFactor float64
	}{
		{
			name:         "nan cost factor",
			costFactor:   math.NaN(),
			energyFactor: 1.0,
		},
		{
			name:         "infinite energy factor",
			costFactor:   1.0,
			energyFactor: math.Inf(1),
		},
	}

	for _, test := range tests {

		t.Run(
			test.name,
			func(t *testing.T) {
				resetExecutionFeedbackConfig(t)

				viper.Set(
					config.MAB_COST_WEIGHT,
					0.5,
				)

				viper.Set(
					config.MAB_ENERGY_WEIGHT,
					0.5,
				)

				bandit :=
					NewUCB1Bandit(
						"invalid-factor-test",
						0.0,
					)

				const arm = "shared-ring"

				bandit.InitArm(arm)

				bandit.UpdateReward(
					arm,
					nil,
					ExecutionFeedback{
						DurationMs:   10.0,
						IsWarmStart:  true,
						CostFactor:   test.costFactor,
						EnergyFactor: test.energyFactor,
					},
				)

				assert.Zero(
					t,
					bandit.Arms[arm].Count,
				)

				assert.Equal(
					t,
					int64(1),
					GlobalColdStartStats.
						Snapshot(
							"invalid-factor-test",
							arm,
						).
						InvalidFeedback,
				)
			},
		)
	}
}
