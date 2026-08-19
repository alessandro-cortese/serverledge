package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/function"
	"github.com/serverledge-faas/serverledge/internal/profiling"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionEnergyFeedbackFromReportPreservesAvailableMeasurement(
	t *testing.T,
) {
	report :=
		function.ExecutionReport{
			KeplerEnergy: &profiling.KeplerInvocationEnergyProfile{
				Available:   true,
				ContainerID: "container-a",
				CPUJoulesByZone: map[string]float64{
					"core":    0.40,
					"package": 0.70,
				},
			},
		}

	feedback :=
		executionEnergyFeedbackFromReport(
			report,
		)

	require.NotNil(
		t,
		feedback,
	)

	assert.True(
		t,
		feedback.Available,
	)

	assert.Equal(
		t,
		"container-a",
		feedback.ContainerID,
	)

	assert.InDelta(
		t,
		0.40,
		feedback.CPUJoulesByZone["core"],
		1e-9,
	)

	assert.InDelta(
		t,
		0.70,
		feedback.CPUJoulesByZone["package"],
		1e-9,
	)

	// Verify that the MAB feedback owns an independent copy of the zone map.
	report.KeplerEnergy.CPUJoulesByZone["core"] =
		9.99

	assert.InDelta(
		t,
		0.40,
		feedback.CPUJoulesByZone["core"],
		1e-9,
	)
}

func TestExecutionEnergyFeedbackFromReportPreservesUnavailableMeasurement(
	t *testing.T,
) {
	report :=
		function.ExecutionReport{
			KeplerEnergy: &profiling.KeplerInvocationEnergyProfile{
				Available:     false,
				InvalidReason: "Kepler refresh timeout",
				ContainerID:   "container-b",
			},
		}

	feedback :=
		executionEnergyFeedbackFromReport(
			report,
		)

	require.NotNil(
		t,
		feedback,
	)

	assert.False(
		t,
		feedback.Available,
	)

	assert.Equal(
		t,
		"Kepler refresh timeout",
		feedback.InvalidReason,
	)

	assert.Equal(
		t,
		"container-b",
		feedback.ContainerID,
	)

	assert.Nil(
		t,
		feedback.CPUJoulesByZone,
	)
}

func TestExecutionEnergyFeedbackFromReportReturnsNilWhenKeplerWasNotCollected(
	t *testing.T,
) {
	feedback :=
		executionEnergyFeedbackFromReport(
			function.ExecutionReport{},
		)

	assert.Nil(
		t,
		feedback,
	)
}

func TestLatencyRewardIgnoresKeplerEnergyFeedback(
	t *testing.T,
) {
	viper.Reset()

	t.Cleanup(func() {
		viper.Reset()
	})

	withoutEnergy,
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

	withEnergy,
		err :=
		CalculateExecutionReward(
			ExecutionFeedback{
				DurationMs: 25.0,

				KeplerEnergy: &KeplerExecutionEnergyFeedback{
					Available:   true,
					ContainerID: "container-a",

					CPUJoulesByZone: map[string]float64{
						"core":    1000.0,
						"package": 2000.0,
					},
				},
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		RewardModeLatency,
		withEnergy.Mode,
	)

	assert.Equal(
		t,
		withoutEnergy.Value,
		withEnergy.Value,
	)

	assert.Equal(
		t,
		withoutEnergy.InputName,
		withEnergy.InputName,
	)

	assert.Equal(
		t,
		withoutEnergy.InputValue,
		withEnergy.InputValue,
	)

	assert.Equal(
		t,
		withoutEnergy.InputUnit,
		withEnergy.InputUnit,
	)
}
