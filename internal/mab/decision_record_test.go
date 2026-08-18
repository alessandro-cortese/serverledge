package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecisionStorageTracksDirectExecutionPlan(
	t *testing.T,
) {
	storage :=
		&DecisionStorage{}

	storage.Store(
		"request-direct",
		DecisionRecord{
			FunctionName: "hello",
			SelectedArm:  "arm-a",
		},
	)

	planned, ok :=
		storage.SetExecutionPlan(
			"request-direct",
			"arm-a",
			"",
		)

	require.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		"request-direct",
		planned.RequestID,
	)

	assert.Equal(
		t,
		"arm-a",
		planned.SelectedArm,
	)

	assert.Equal(
		t,
		"arm-a",
		planned.ExecutionArm,
	)

	assert.False(
		t,
		planned.Fallback,
	)

	assert.Empty(
		t,
		planned.FallbackReason,
	)

	retrieved, ok :=
		storage.RetrieveAndDelete(
			"request-direct",
		)

	require.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		planned,
		retrieved,
	)
}

func TestDecisionStorageTracksFallbackExecutionPlan(
	t *testing.T,
) {
	storage :=
		&DecisionStorage{}

	storage.Store(
		"request-fallback",
		DecisionRecord{
			FunctionName: "hello",
			SelectedArm:  "arm-a",
		},
	)

	planned, ok :=
		storage.SetExecutionPlan(
			"request-fallback",
			"arm-b",
			FallbackReasonSelectedArmNoCandidate,
		)

	require.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		"arm-a",
		planned.SelectedArm,
	)

	assert.Equal(
		t,
		"arm-b",
		planned.ExecutionArm,
	)

	assert.True(
		t,
		planned.Fallback,
	)

	assert.Equal(
		t,
		FallbackReasonSelectedArmNoCandidate,
		planned.FallbackReason,
	)
}

func TestResolveDecisionWithFeedbackRecordsFallback(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_FALLBACK_PENALTY,
		-12.0,
	)

	previousManager :=
		GlobalBanditManager

	GlobalDecisionStats.Reset()

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager

		GlobalDecisionStats.Reset()
	})

	bandit :=
		NewUCB1Bandit(
			"decision-fallback",
			0.0,
		)

	bandit.InitArm(
		"arm-a",
	)

	bandit.InitArm(
		"arm-b",
	)

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
		)

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"decision-fallback": bandit,
			},
		}

	resolved :=
		ResolveDecisionWithFeedback(
			DecisionRecord{
				RequestID: "request-fallback",

				FunctionName: "decision-fallback",

				SelectedArm: selected,

				ExecutionArm: "arm-b",

				Fallback: true,

				FallbackReason: FallbackReasonSelectedArmNoCandidate,
			},
			"arm-b",
			ExecutionFeedback{
				DurationMs: 10.0,

				IsWarmStart: true,
			},
		)

	require.True(
		t,
		resolved,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].
			InFlight,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].Count,
	)

	assert.InDelta(
		t,
		-12.0,
		bandit.Arms["arm-a"].AvgReward,
		1e-9,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-b"].
			Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].Count,
	)

	assert.InDelta(
		t,
		-12.0,
		bandit.Arms["arm-a"].AvgReward,
		1e-9,
	)

	stats :=
		GlobalDecisionStats.
			Snapshot()

	assert.Zero(
		t,
		stats.DirectExecutions,
	)

	assert.Equal(
		t,
		int64(1),
		stats.FallbackExecutions,
	)

	assert.Zero(
		t,
		stats.CancelledDecisions,
	)
}

func TestResolveDecisionWithoutFeedbackRecordsCancellation(
	t *testing.T,
) {

	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_FALLBACK_PENALTY,
		-12.0,
	)

	previousManager :=
		GlobalBanditManager

	GlobalDecisionStats.Reset()

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager

		GlobalDecisionStats.Reset()
	})

	bandit :=
		NewUCB1Bandit(
			"decision-cancelled",
			0.0,
		)

	bandit.InitArm(
		"arm-a",
	)

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"arm-a",
			},
		)

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"decision-cancelled": bandit,
			},
		}

	resolved :=
		ResolveDecisionWithoutFeedback(
			DecisionRecord{
				RequestID: "request-cancelled",

				FunctionName: "decision-cancelled",

				SelectedArm: selected,
			},
			DecisionFailureReasonNoCandidateAfterFallback,
		)

	require.True(
		t,
		resolved,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].
			InFlight,
	)

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.TotalCounts,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["arm-a"].Count,
	)

	assert.InDelta(
		t,
		-12.0,
		bandit.Arms["arm-a"].AvgReward,
		1e-9,
	)

	stats :=
		GlobalDecisionStats.
			Snapshot()

	assert.Zero(
		t,
		stats.DirectExecutions,
	)

	assert.Zero(
		t,
		stats.FallbackExecutions,
	)

	assert.Equal(
		t,
		int64(1),
		stats.CancelledDecisions,
	)
}
