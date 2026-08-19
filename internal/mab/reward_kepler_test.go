package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetKeplerRewardConfigForTest(
	t *testing.T,
) {
	t.Helper()

	viper.Reset()

	t.Cleanup(func() {
		viper.Reset()
	})
}

func validKeplerRewardFeedback() ExecutionFeedback {
	return ExecutionFeedback{
		DurationMs: 10.0,

		KeplerEnergy: &KeplerExecutionEnergyFeedback{
			Available: true,

			ContainerID: "container-test",

			CPUJoulesByZone: map[string]float64{
				"core":    2.1,
				"package": 2.5,
			},
		},
	}
}

func TestConfiguredKeplerRewardZoneRequiresExplicitValue(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	_,
		err :=
		ConfiguredKeplerRewardZone()

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		config.MAB_REWARD_KEPLER_ZONE,
	)
}

func TestConfiguredKeplerRewardZoneNormalizesValue(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"  PACKAGE  ",
	)

	zone,
		err :=
		ConfiguredKeplerRewardZone()

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		"package",
		zone,
	)
}

func TestSelectKeplerRewardInputSelectsConfiguredZoneOnly(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	input,
		err :=
		SelectKeplerRewardInput(
			validKeplerRewardFeedback(),
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		"package",
		input.Zone,
	)

	assert.InDelta(
		t,
		2.5,
		input.Joules,
		1e-9,
	)

	// The selector must not implicitly sum package and core.
	assert.NotEqual(
		t,
		4.6,
		input.Joules,
	)
}

func TestSelectKeplerRewardInputCanSelectCore(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"core",
	)

	input,
		err :=
		SelectKeplerRewardInput(
			validKeplerRewardFeedback(),
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		"core",
		input.Zone,
	)

	assert.InDelta(
		t,
		2.1,
		input.Joules,
		1e-9,
	)
}

func TestSelectKeplerRewardInputRejectsMissingFeedback(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	_,
		err :=
		SelectKeplerRewardInput(
			ExecutionFeedback{},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"feedback is missing",
	)
}

func TestSelectKeplerRewardInputRejectsUnavailableMeasurement(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	_,
		err :=
		SelectKeplerRewardInput(
			ExecutionFeedback{
				KeplerEnergy: &KeplerExecutionEnergyFeedback{
					Available: false,

					InvalidReason: "refresh timeout",
				},
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"refresh timeout",
	)
}

func TestSelectKeplerRewardInputRejectsUnavailableZone(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"dram",
	)

	_,
		err :=
		SelectKeplerRewardInput(
			validKeplerRewardFeedback(),
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		`zone "dram" is unavailable`,
	)

	assert.Contains(
		t,
		err.Error(),
		"core",
	)

	assert.Contains(
		t,
		err.Error(),
		"package",
	)
}

func TestSelectKeplerRewardInputRejectsInvalidEnergyValues(
	t *testing.T,
) {
	tests :=
		[]struct {
			name   string
			joules float64
		}{
			{
				name:   "negative",
				joules: -1.0,
			},
			{
				name:   "nan",
				joules: math.NaN(),
			},
			{
				name:   "positive infinity",
				joules: math.Inf(1),
			},
		}

	for _, test := range tests {

		t.Run(
			test.name,
			func(t *testing.T) {
				resetKeplerRewardConfigForTest(
					t,
				)

				viper.Set(
					config.MAB_REWARD_KEPLER_ZONE,
					"package",
				)

				feedback :=
					validKeplerRewardFeedback()

				feedback.
					KeplerEnergy.
					CPUJoulesByZone["package"] =
					test.joules

				_,
					err :=
					SelectKeplerRewardInput(
						feedback,
					)

				require.Error(
					t,
					err,
				)
			},
		)
	}
}

func TestSelectKeplerRewardInputAcceptsZeroMeasurement(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	feedback :=
		validKeplerRewardFeedback()

	feedback.
		KeplerEnergy.
		CPUJoulesByZone["package"] =
		0.0

	input,
		err :=
		SelectKeplerRewardInput(
			feedback,
		)

	require.NoError(
		t,
		err,
	)

	assert.Zero(
		t,
		input.Joules,
	)
}

func TestValidateRewardConfigurationRequiresZoneForKeplerEnergy(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
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
		config.MAB_REWARD_KEPLER_ZONE,
	)
}

func TestValidateRewardConfigurationAcceptsKeplerEnergyWithZone(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	require.NoError(
		t,
		ValidateRewardConfiguration(),
	)
}

func TestValidateRewardConfigurationLatencyDoesNotRequireKeplerZone(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeLatency,
		),
	)

	require.NoError(
		t,
		ValidateRewardConfiguration(),
	)
}

func TestCalculateKeplerEnergyRewardUsesNegativeLogarithm(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	result,
		err :=
		CalculateExecutionReward(
			validKeplerRewardFeedback(),
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		RewardModeKeplerEnergy,
		result.Mode,
	)

	assert.Equal(
		t,
		"kepler_energy_package",
		result.InputName,
	)

	assert.Equal(
		t,
		"J",
		result.InputUnit,
	)

	assert.InDelta(
		t,
		2.5,
		result.InputValue,
		1e-9,
	)

	assert.InDelta(
		t,
		-math.Log(2.5),
		result.Value,
		1e-9,
	)
}

func TestCalculateKeplerEnergyRewardUsesConfiguredCoreZone(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"core",
	)

	result,
		err :=
		CalculateExecutionReward(
			validKeplerRewardFeedback(),
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		"kepler_energy_core",
		result.InputName,
	)

	assert.InDelta(
		t,
		2.1,
		result.InputValue,
		1e-9,
	)

	assert.InDelta(
		t,
		-math.Log(2.1),
		result.Value,
		1e-9,
	)
}

func TestCalculateKeplerEnergyRewardPrefersLowerEnergy(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	lowEnergy :=
		validKeplerRewardFeedback()

	lowEnergy.
		KeplerEnergy.
		CPUJoulesByZone["package"] =
		0.5

	highEnergy :=
		validKeplerRewardFeedback()

	highEnergy.
		KeplerEnergy.
		CPUJoulesByZone["package"] =
		2.0

	lowResult,
		err :=
		CalculateExecutionReward(
			lowEnergy,
		)

	require.NoError(
		t,
		err,
	)

	highResult,
		err :=
		CalculateExecutionReward(
			highEnergy,
		)

	require.NoError(
		t,
		err,
	)

	assert.Greater(
		t,
		lowResult.Value,
		highResult.Value,
	)
}

func TestCalculateKeplerEnergyRewardRejectsZero(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	feedback :=
		validKeplerRewardFeedback()

	feedback.
		KeplerEnergy.
		CPUJoulesByZone["package"] =
		0.0

	_,
		err :=
		CalculateExecutionReward(
			feedback,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"must be positive",
	)
}

func TestUCB1UsesKeplerEnergyReward(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	bandit :=
		NewUCB1Bandit(
			"kepler-energy-ucb1-test",
			0.8,
		)

	const arm = "x86"

	bandit.InitArm(
		arm,
	)

	feedback :=
		validKeplerRewardFeedback()

	feedback.IsWarmStart =
		true

	bandit.UpdateReward(
		arm,
		nil,
		feedback,
	)

	stats,
		ok :=
		bandit.Arms[arm]

	require.True(
		t,
		ok,
	)

	require.Equal(
		t,
		int64(1),
		stats.Count,
	)

	assert.InDelta(
		t,
		-math.Log(2.5),
		stats.AvgReward,
		1e-9,
	)
}

func TestLinUCBUsesKeplerEnergyReward(
	t *testing.T,
) {
	resetKeplerRewardConfigForTest(
		t,
	)

	viper.Set(
		config.MAB_REWARD_MODE,
		string(
			RewardModeKeplerEnergy,
		),
	)

	viper.Set(
		config.MAB_REWARD_KEPLER_ZONE,
		"package",
	)

	bandit :=
		NewLinUCBDisjointPolicy(
			"kepler-energy-linucb-test",
			0.1,
		)

	const arm = "x86"

	bandit.InitArm(
		arm,
	)

	const utilization = 0.20

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				arm: utilization,
			},
		}

	features :=
		bandit.computeFeatures(
			utilization,
		)

	feedback :=
		validKeplerRewardFeedback()

	feedback.IsWarmStart =
		true

	bandit.UpdateReward(
		arm,
		ctx,
		feedback,
	)

	state,
		ok :=
		bandit.Arms[arm]

	require.True(
		t,
		ok,
	)

	expectedReward :=
		-math.Log(
			2.5,
		)

	for i := 0; i < bandit.Dim; i++ {

		assert.InDelta(
			t,
			expectedReward*
				features.AtVec(
					i,
				),
			state.b.AtVec(
				i,
			),
			1e-9,
		)
	}
}
