package mab

import (
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnrelatedCancellationDoesNotApplyFallbackPenalty(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	previousManager :=
		GlobalBanditManager

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager
	})

	bandit :=
		NewUCB1Bandit(
			"unrelated-cancellation",
			0.0,
		)

	bandit.InitArm("arm-a")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{"arm-a"},
		)

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"unrelated-cancellation": bandit,
			},
		}

	resolved :=
		ResolveDecisionWithoutFeedback(
			DecisionRecord{
				RequestID: "request-proxy-error",

				FunctionName: "unrelated-cancellation",

				SelectedArm: selected,

				ExecutionArm: "arm-a",
			},
			"proxy_completed_without_claimed_feedback",
		)

	require.True(t, resolved)

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)

	assert.Zero(
		t,
		bandit.TotalCounts,
	)

	assert.Zero(
		t,
		bandit.Arms["arm-a"].Count,
	)
}

func TestFallbackPenaltyAppliesWhenActualColdFeedbackIsSkipped(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_FALLBACK_PENALTY,
		-12.0,
	)

	viper.Set(
		config.MAB_COLD_START_MODE,
		string(ColdStartModeSkip),
	)

	previousManager :=
		GlobalBanditManager

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager
	})

	bandit :=
		NewUCB1Bandit(
			"cold-fallback",
			0.0,
		)

	bandit.InitArm("arm-a")
	bandit.InitArm("arm-b")

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{"arm-a"},
		)

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"cold-fallback": bandit,
			},
		}

	resolved :=
		ResolveDecisionWithFeedback(
			DecisionRecord{
				RequestID: "request-cold-fallback",

				FunctionName: "cold-fallback",

				SelectedArm: selected,

				ExecutionArm: "arm-b",

				Fallback: true,

				FallbackReason: FallbackReasonSelectedArmNoCandidate,
			},
			"arm-b",
			ExecutionFeedback{
				DurationMs: 10.0,

				IsWarmStart: false,
			},
		)

	require.True(t, resolved)

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

	assert.Zero(
		t,
		bandit.Arms["arm-b"].Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.TotalCounts,
	)
}

func TestLinUCBFallbackPenaltyUpdatesSelectedAndActualModels(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_FALLBACK_PENALTY,
		-12.0,
	)

	previousManager :=
		GlobalBanditManager

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager
	})

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-fallback",
			0.0,
		)

	bandit.InitArm("arm-a")
	bandit.InitArm("arm-b")

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				"arm-a": 0.20,
				"arm-b": 0.70,
			},
		}

	selected :=
		bandit.SelectArmFrom(
			ctx,
			[]string{"arm-a"},
		)

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"linucb-fallback": bandit,
			},
		}

	resolved :=
		ResolveDecisionWithFeedback(
			DecisionRecord{
				RequestID: "request-linucb-fallback",

				FunctionName: "linucb-fallback",

				SelectedArm: selected,

				ExecutionArm: "arm-b",

				Context: ctx,

				Fallback: true,

				FallbackReason: FallbackReasonSelectedArmNoCandidate,
			},
			"arm-b",
			ExecutionFeedback{
				DurationMs: 10.0,

				IsWarmStart: true,
			},
		)

	require.True(t, resolved)

	selectedFeatures :=
		bandit.computeFeatures(0.20)

	executionFeatures :=
		bandit.computeFeatures(0.70)

	for i := 0; i < bandit.Dim; i++ {
		assert.InDelta(
			t,
			-12.0*selectedFeatures.AtVec(i),
			bandit.Arms["arm-a"].b.AtVec(i),
			1e-9,
		)

		assert.InDelta(
			t,
			-math.Log(10.0)*
				executionFeatures.AtVec(i),
			bandit.Arms["arm-b"].b.AtVec(i),
			1e-9,
		)
	}

	assert.Zero(
		t,
		bandit.TotalInFlight,
	)
}

func TestInvalidFallbackPenaltyUsesDefault(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_FALLBACK_PENALTY,
		1.0,
	)

	assert.InDelta(
		t,
		defaultFallbackPenalty,
		configuredFallbackPenalty(),
		1e-9,
	)
}
