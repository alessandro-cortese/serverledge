package mab

import (
	"sync"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
)

func TestUCB1InitialRequestsUseDifferentUntriedArms(
	t *testing.T,
) {
	bandit :=
		NewUCB1Bandit(
			"in-flight-exploration",
			0.8,
		)

	bandit.InitArm("arm-a")
	bandit.InitArm("arm-b")

	first :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
				"arm-b",
			},
		)

	second :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
				"arm-b",
			},
		)

	assert.Equal(t, "arm-a", first)
	assert.Equal(t, "arm-b", second)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].InFlight,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-b"].InFlight,
	)

	assert.Equal(
		t,
		int64(2),
		bandit.TotalInFlight,
	)

	assert.Zero(
		t,
		bandit.TotalCounts,
	)
}

func TestUCB1ValidFeedbackMovesInFlightToCount(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"in-flight-feedback",
			0.0,
		)

	bandit.InitArm("arm-a")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
		)

	feedback := ExecutionFeedback{
		DurationMs:   10.0,
		IsWarmStart:  true,
		CostFactor:   1.0,
		EnergyFactor: 1.0,
	}

	bandit.ResolveSelection(
		selected,
		selected,
		nil,
		&feedback,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].InFlight,
	)

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.TotalCounts,
	)
}

func TestUCB1ColdSkipReleasesInFlightWithoutCount(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(ColdStartModeSkip),
	)

	bandit :=
		NewUCB1Bandit(
			"in-flight-cold-skip",
			0.0,
		)

	bandit.InitArm("arm-a")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
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
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].InFlight,
	)

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].Count,
	)

	assert.Zero(
		t,
		bandit.TotalCounts,
	)
}

func TestUCB1FallbackUpdatesActualExecutionArm(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"in-flight-fallback",
			0.0,
		)

	bandit.InitArm("arm-a")
	bandit.InitArm("arm-b")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
		)

	feedback := ExecutionFeedback{
		DurationMs:   10.0,
		IsWarmStart:  true,
		CostFactor:   1.0,
		EnergyFactor: 1.0,
	}

	bandit.ResolveSelection(
		selected,
		"arm-b",
		nil,
		&feedback,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].InFlight,
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

func TestUCB1RepeatedResolutionDoesNotApplyRewardTwice(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"in-flight-double-resolution",
			0.0,
		)

	bandit.InitArm("arm-a")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
		)

	feedback := ExecutionFeedback{
		DurationMs:   10.0,
		IsWarmStart:  true,
		CostFactor:   1.0,
		EnergyFactor: 1.0,
	}

	bandit.ResolveSelection(
		selected,
		selected,
		nil,
		&feedback,
	)

	bandit.ResolveSelection(
		selected,
		selected,
		nil,
		&feedback,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.TotalCounts,
	)
}

func TestUCB1ConcurrentCancellationsRemainBalanced(
	t *testing.T,
) {
	bandit :=
		NewUCB1Bandit(
			"in-flight-concurrent",
			0.8,
		)

	bandit.InitArm("arm-a")

	const requests = 100

	var wg sync.WaitGroup
	wg.Add(requests)

	for i := 0; i < requests; i++ {
		go func() {
			defer wg.Done()

			selected :=
				bandit.SelectArmFrom(
					nil,
					[]string{
						"arm-a",
					},
				)

			bandit.ResolveSelection(
				selected,
				"",
				nil,
				nil,
			)
		}()
	}

	wg.Wait()

	assert.Zero(
		t,
		bandit.Arms["arm-a"].InFlight,
	)

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)
}
