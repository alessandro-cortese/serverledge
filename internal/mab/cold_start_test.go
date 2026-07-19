package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfiguredColdStartMode(
	t *testing.T,
) {
	tests := []struct {
		name         string
		configured   bool
		value        string
		expectedMode ColdStartMode
	}{
		{
			name:         "default is legacy skip",
			expectedMode: ColdStartModeSkip,
		},
		{
			name:         "explicit skip",
			configured:   true,
			value:        "skip",
			expectedMode: ColdStartModeSkip,
		},
		{
			name:         "execution is normalized",
			configured:   true,
			value:        "  EXECUTION  ",
			expectedMode: ColdStartModeExecution,
		},
		{
			name:         "unknown mode falls back to skip",
			configured:   true,
			value:        "unsupported",
			expectedMode: ColdStartModeSkip,
		},
	}

	for _, test := range tests {
		t.Run(
			test.name,
			func(t *testing.T) {
				resetExecutionFeedbackConfig(t)

				if test.configured {
					viper.Set(
						config.MAB_COLD_START_MODE,
						test.value,
					)
				}

				assert.Equal(
					t,
					test.expectedMode,
					configuredColdStartMode(),
				)
			},
		)
	}
}

func TestUCB1ColdStartModes(
	t *testing.T,
) {
	tests := []struct {
		name           string
		mode           ColdStartMode
		expectedCount  int64
		expectedReward float64
	}{
		{
			name:          "skip preserves legacy behavior",
			mode:          ColdStartModeSkip,
			expectedCount: 0,
		},
		{
			name:           "execution learns from duration",
			mode:           ColdStartModeExecution,
			expectedCount:  1,
			expectedReward: -math.Log(10.0),
		},
	}

	for _, test := range tests {
		t.Run(
			test.name,
			func(t *testing.T) {
				resetExecutionFeedbackConfig(t)

				viper.Set(
					config.MAB_COLD_START_MODE,
					string(test.mode),
				)

				bandit :=
					NewUCB1Bandit(
						"ucb1-cold-start-test",
						0.0,
					)

				const arm = "shared-ring"

				bandit.InitArm(arm)

				bandit.UpdateReward(
					arm,
					nil,
					ExecutionFeedback{
						DurationMs:  10.0,
						IsWarmStart: false,
					},
				)

				stats, ok :=
					bandit.Arms[arm]

				require.True(
					t,
					ok,
				)

				assert.Equal(
					t,
					test.expectedCount,
					stats.Count,
				)

				assert.Equal(
					t,
					test.expectedCount,
					bandit.TotalCounts,
				)

				assert.InDelta(
					t,
					test.expectedReward,
					stats.SumRewards,
					1e-9,
				)

				assert.InDelta(
					t,
					test.expectedReward,
					stats.AvgReward,
					1e-9,
				)
			},
		)
	}
}

func TestLinUCBColdStartModes(
	t *testing.T,
) {
	tests := []struct {
		name         string
		mode         ColdStartMode
		shouldUpdate bool
	}{
		{
			name: "skip preserves legacy behavior",
			mode: ColdStartModeSkip,
		},
		{
			name:         "execution learns from duration",
			mode:         ColdStartModeExecution,
			shouldUpdate: true,
		},
	}

	for _, test := range tests {
		t.Run(
			test.name,
			func(t *testing.T) {
				resetExecutionFeedbackConfig(t)

				viper.Set(
					config.MAB_COLD_START_MODE,
					string(test.mode),
				)

				bandit :=
					NewLinUCBDisjointPolicy(
						"linucb-cold-start-test",
						0.1,
					)

				const arm = "shared-ring"

				bandit.InitArm(arm)

				ctx := &Context{
					ArchMemUsage: map[string]float64{
						arm: 0.20,
					},
				}

				features :=
					bandit.computeFeatures(
						0.20,
					)

				bandit.UpdateReward(
					arm,
					ctx,
					ExecutionFeedback{
						DurationMs:  10.0,
						IsWarmStart: false,
					},
				)

				state, ok :=
					bandit.Arms[arm]

				require.True(
					t,
					ok,
				)

				expectedReward :=
					-math.Log(10.0)

				for i := 0; i < bandit.Dim; i++ {
					for j := 0; j < bandit.Dim; j++ {
						expectedA := 0.0

						// A is initialized as the identity matrix.
						if i == j {
							expectedA = 1.0
						}

						// In execution mode:
						//
						//     A = I + x*x^T
						if test.shouldUpdate {
							expectedA +=
								features.AtVec(i) *
									features.AtVec(j)
						}

						assert.InDelta(
							t,
							expectedA,
							state.A.At(i, j),
							1e-9,
						)
					}

					expectedB := 0.0

					// In execution mode:
					//
					//     b = reward*x
					if test.shouldUpdate {
						expectedB =
							expectedReward *
								features.AtVec(i)
					}

					assert.InDelta(
						t,
						expectedB,
						state.b.AtVec(i),
						1e-9,
					)
				}
			},
		)
	}
}
