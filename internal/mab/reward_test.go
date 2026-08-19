package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetRewardConfigForTest(
	t *testing.T,
) {
	t.Helper()

	viper.Reset()

	t.Cleanup(func() {
		viper.Reset()
	})
}

func TestRewardDefaultsToLatency(
	t *testing.T,
) {
	resetRewardConfigForTest(t)

	result,
		err :=
		CalculateExecutionReward(
			ExecutionFeedback{
				DurationMs: 10.0,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		RewardModeLatency,
		result.Mode,
	)

	assert.Equal(
		t,
		"duration_ms",
		result.InputName,
	)

	assert.Equal(
		t,
		"ms",
		result.InputUnit,
	)

	assert.InDelta(
		t,
		10.0,
		result.InputValue,
		1e-9,
	)

	assert.InDelta(
		t,
		-math.Log(10.0),
		result.Value,
		1e-9,
	)
}

func TestRewardExplicitLatencyMode(
	t *testing.T,
) {
	resetRewardConfigForTest(t)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeLatency,
		),
	)

	result,
		err :=
		CalculateExecutionReward(
			ExecutionFeedback{
				DurationMs: 25.0,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		RewardModeLatency,
		result.Mode,
	)

	assert.InDelta(
		t,
		-math.Log(25.0),
		result.Value,
		1e-9,
	)
}

func TestRewardRejectsUnknownMode(
	t *testing.T,
) {
	resetRewardConfigForTest(t)

	viper.Set(
		config.MAB_REWARD_MODE,
		"unsupported-reward",
	)

	err :=
		ValidateRewardConfiguration()

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"unsupported MAB reward mode",
	)
}

func TestLatencyRewardRejectsInvalidDuration(
	t *testing.T,
) {
	resetRewardConfigForTest(t)

	tests := []struct {
		name     string
		duration float64
	}{
		{
			name:     "zero",
			duration: 0,
		},
		{
			name:     "negative",
			duration: -1,
		},
		{
			name:     "nan",
			duration: math.NaN(),
		},
		{
			name:     "positive infinity",
			duration: math.Inf(1),
		},
	}

	for _, test := range tests {

		t.Run(
			test.name,
			func(t *testing.T) {
				_,
					err :=
					CalculateExecutionReward(
						ExecutionFeedback{
							DurationMs: test.duration,
						},
					)

				require.Error(
					t,
					err,
				)
			},
		)
	}
}

func TestRewardModeNormalization(
	t *testing.T,
) {
	resetRewardConfigForTest(t)

	viper.Set(
		config.MAB_REWARD_MODE,
		"  LATENCY  ",
	)

	mode,
		err :=
		ConfiguredRewardMode()

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		RewardModeLatency,
		mode,
	)
}
