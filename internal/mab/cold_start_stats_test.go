package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
)

func TestColdStartStatsSeparateObservedAcceptedSkippedAndInvalid(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"cold-stats-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	// Valid cold sample accepted through execution mode.
	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeExecution,
		),
	)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:   10.0,
			InitTimeMs:   100.0,
			IsWarmStart:  false,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	// Valid cold sample intentionally skipped.
	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeSkip,
		),
	)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:   20.0,
			InitTimeMs:   200.0,
			IsWarmStart:  false,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	// Valid warm sample.
	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:   5.0,
			InitTimeMs:   1.0,
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	// Cold sample invalid because DurationMs is zero.
	viper.Set(
		config.MAB_COLD_START_MODE,
		string(
			ColdStartModeExecution,
		),
	)

	bandit.UpdateReward(
		arm,
		nil,
		ExecutionFeedback{
			DurationMs:   0.0,
			InitTimeMs:   300.0,
			IsWarmStart:  false,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	stats :=
		GlobalColdStartStats.
			Snapshot(
				"cold-stats-test",
				arm,
			)

	assert.Equal(
		t,
		int64(4),
		stats.TotalInvocations,
	)

	assert.Equal(
		t,
		int64(3),
		stats.ColdObserved,
	)

	assert.Equal(
		t,
		int64(1),
		stats.WarmObserved,
	)

	assert.Equal(
		t,
		int64(1),
		stats.ColdAccepted,
	)

	assert.Equal(
		t,
		int64(1),
		stats.ColdSkipped,
	)

	assert.Equal(
		t,
		int64(1),
		stats.InvalidFeedback,
	)

	assert.Equal(
		t,
		int64(2),
		stats.ColdDurationSamples,
	)

	assert.Equal(
		t,
		int64(1),
		stats.WarmDurationSamples,
	)

	assert.Equal(
		t,
		int64(3),
		stats.ColdInitTimeSamples,
	)

	assert.InDelta(
		t,
		30.0,
		stats.ColdDurationSumMs,
		1e-9,
	)

	assert.InDelta(
		t,
		15.0,
		stats.AvgColdDurationMs,
		1e-9,
	)

	assert.InDelta(
		t,
		5.0,
		stats.AvgWarmDurationMs,
		1e-9,
	)

	assert.InDelta(
		t,
		600.0,
		stats.ColdInitTimeSumMs,
		1e-9,
	)

	assert.InDelta(
		t,
		200.0,
		stats.AvgColdInitTimeMs,
		1e-9,
	)

	// Only the accepted cold sample and the valid warm sample update UCB1.
	assert.Equal(
		t,
		int64(2),
		bandit.Arms[arm].Count,
	)

	assert.Equal(
		t,
		int64(2),
		bandit.TotalCounts,
	)
}
