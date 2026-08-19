package mab

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/function"
	"github.com/serverledge-faas/serverledge/internal/profiling"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func resetExecutionFeedbackConfig(
	t *testing.T,
) {
	t.Helper()

	viper.Reset()
	GlobalColdStartStats.Reset()

	t.Cleanup(func() {
		viper.Reset()
		GlobalColdStartStats.Reset()
	})
}

func TestUCB1RewardDependsOnlyOnDuration(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"latency-only-reward-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	firstFeedback := ExecutionFeedback{
		DurationMs:    10.0,
		IsWarmStart:   true,
		NodeName:      "node-a",
		ExecutionNode: "node-a",
	}

	secondFeedback := ExecutionFeedback{
		DurationMs:    10.0,
		IsWarmStart:   true,
		NodeName:      "node-b",
		ExecutionNode: "node-b",
	}

	expectedReward :=
		-math.Log(10.0)

	bandit.UpdateReward(
		arm,
		nil,
		firstFeedback,
	)

	stats, ok :=
		bandit.Arms[arm]

	require.True(t, ok)

	require.Equal(
		t,
		int64(1),
		stats.Count,
	)

	assert.InDelta(
		t,
		expectedReward,
		stats.AvgReward,
		1e-9,
	)

	bandit.UpdateReward(
		arm,
		nil,
		secondFeedback,
	)

	require.Equal(
		t,
		int64(2),
		stats.Count,
	)

	require.Equal(
		t,
		int64(2),
		bandit.TotalCounts,
	)

	assert.InDelta(
		t,
		2.0*expectedReward,
		stats.SumRewards,
		1e-9,
	)

	assert.InDelta(
		t,
		expectedReward,
		stats.AvgReward,
		1e-9,
	)
}

func TestLinUCBRewardDependsOnlyOnDuration(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-latency-only-reward-test",
			0.1,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			arm: 0.20,
		},
	}

	feedback := ExecutionFeedback{
		DurationMs:    10.0,
		IsWarmStart:   true,
		NodeName:      "node-a",
		ExecutionNode: "node-a",
	}

	expectedReward :=
		-math.Log(10.0)

	features :=
		bandit.computeFeatures(
			0.20,
		)

	bandit.UpdateReward(
		arm,
		ctx,
		feedback,
	)

	state, ok :=
		bandit.Arms[arm]

	require.True(t, ok)

	for i := 0; i < bandit.Dim; i++ {
		assert.InDelta(
			t,
			expectedReward*
				features.AtVec(i),
			state.b.AtVec(i),
			1e-9,
		)
	}
}

// recordingPolicy captures the values passed by UpdateBandit.
// It allows the body-decoding and context-retrieval path to be tested without
// depending on the internal state of a concrete bandit implementation.
type recordingPolicy struct {
	selectedArm string
	updatedArm  string
	ctx         *Context
	feedback    ExecutionFeedback
	resolutions int
	updates     int
}

func (p *recordingPolicy) SelectArm(
	ctx *Context,
) string {
	return ""
}

func (p *recordingPolicy) SelectArmFrom(
	ctx *Context,
	allowedArms []string,
) string {
	return ""
}

func (p *recordingPolicy) UpdateReward(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	p.updatedArm = arm
	p.ctx = ctx
	p.feedback = feedback
	p.updates++
}

func (p *recordingPolicy) ResolveSelection(
	selectedArm string,
	executionArm string,
	ctx *Context,
	feedback *ExecutionFeedback,
	selectedArmReward *SyntheticReward,
) {
	p.selectedArm =
		selectedArm

	p.resolutions++

	if feedback == nil ||
		executionArm == "" {

		return
	}

	p.UpdateReward(
		executionArm,
		ctx,
		*feedback,
	)
}

func (p *recordingPolicy) InitArm(
	arm string,
) {
}

func (p *recordingPolicy) GetType() BanditType {
	return UCB1
}

func TestUpdateBanditPreservesNodeSpecificFeedback(
	t *testing.T,
) {
	previousManager :=
		GlobalBanditManager

	GlobalDecisionStats.Reset()

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager

		GlobalDecisionStats.Reset()
	})

	recorder :=
		&recordingPolicy{}

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"hello": recorder,
			},
		}

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"shared-ring": 0.40,
		},
	}

	response := function.Response{
		Success: true,
		ExecutionReport: function.ExecutionReport{
			ResponseTime:   0.475,
			Duration:       0.007,
			InitTime:       0.465,
			QueueingTime:   0.002,
			OffloadLatency: 0.001,
			IsWarmStart:    false,
			ExecutionNode:  "node-a",

			KeplerEnergy: &profiling.KeplerInvocationEnergyProfile{
				SchemaVersion: profiling.
					KeplerInvocationEnergyProfileSchemaVersion,

				Available: true,

				ContainerID: "container-a",

				CPUJoulesByZone: map[string]float64{
					"core":    1.25,
					"package": 1.75,
				},
			},
		},
	}

	body, err :=
		json.Marshal(
			response,
		)

	require.NoError(
		t,
		err,
	)

	initialFeedback := ExecutionFeedback{
		NodeName: "node-a",
	}

	err = UpdateBandit(
		body,
		"/invoke/hello",
		"shared-ring",
		DecisionRecord{
			RequestID:    "request-1",
			FunctionName: "hello",
			SelectedArm:  "shared-ring",
			ExecutionArm: "shared-ring",
			Context:      ctx,
		},
		initialFeedback,
	)

	require.NoError(
		t,
		err,
	)

	require.Equal(
		t,
		1,
		recorder.updates,
	)

	assert.Equal(
		t,
		"shared-ring",
		recorder.updatedArm,
	)

	assert.Same(
		t,
		ctx,
		recorder.ctx,
	)

	assert.InDelta(
		t,
		7.0,
		recorder.feedback.DurationMs,
		1e-9,
	)

	assert.InDelta(
		t,
		475.0,
		recorder.feedback.ResponseTimeMs,
		1e-9,
	)

	assert.InDelta(
		t,
		465.0,
		recorder.feedback.InitTimeMs,
		1e-9,
	)

	assert.InDelta(
		t,
		2.0,
		recorder.feedback.QueueingTimeMs,
		1e-9,
	)

	assert.InDelta(
		t,
		1.0,
		recorder.feedback.OffloadLatencyMs,
		1e-9,
	)

	assert.False(
		t,
		recorder.feedback.IsWarmStart,
	)

	assert.Equal(
		t,
		"node-a",
		recorder.feedback.NodeName,
	)

	assert.Equal(
		t,
		"node-a",
		recorder.feedback.ExecutionNode,
	)

	require.NotNil(
		t,
		recorder.feedback.KeplerEnergy,
	)

	assert.True(
		t,
		recorder.feedback.KeplerEnergy.Available,
	)

	assert.Equal(
		t,
		"container-a",
		recorder.feedback.KeplerEnergy.ContainerID,
	)

	assert.InDelta(
		t,
		1.25,
		recorder.feedback.KeplerEnergy.CPUJoulesByZone["core"],
		1e-9,
	)

	assert.InDelta(
		t,
		1.75,
		recorder.feedback.KeplerEnergy.CPUJoulesByZone["package"],
		1e-9,
	)

	assert.Equal(
		t,
		1,
		recorder.resolutions,
	)

	assert.Equal(
		t,
		"shared-ring",
		recorder.selectedArm,
	)

	stats :=
		GlobalDecisionStats.
			Snapshot()

	assert.Equal(
		t,
		int64(1),
		stats.DirectExecutions,
	)

	assert.Zero(
		t,
		stats.FallbackExecutions,
	)

	assert.Zero(
		t,
		stats.CancelledDecisions,
	)

}
