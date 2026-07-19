package mab

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/function"
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

func TestUCB1RewardUsesConcreteExecutionNodeFactors(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.5,
	)

	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.25,
	)

	bandit :=
		NewUCB1Bandit(
			"node-specific-reward-test",
			0.0,
		)

	const arm = "shared-ring"

	bandit.InitArm(arm)

	cheapFeedback := ExecutionFeedback{
		DurationMs:    10.0,
		IsWarmStart:   true,
		NodeName:      "cheap-node",
		ExecutionNode: "cheap-node",
		CostFactor:    1.0,
		EnergyFactor:  1.0,
	}

	expensiveFeedback := ExecutionFeedback{
		DurationMs:    10.0,
		IsWarmStart:   true,
		NodeName:      "expensive-node",
		ExecutionNode: "expensive-node",
		CostFactor:    3.0,
		EnergyFactor:  2.0,
	}

	latencyReward :=
		-math.Log(10.0)

	expectedCheapReward :=
		latencyReward -
			0.5*1.0 -
			0.25*1.0

	expectedExpensiveReward :=
		latencyReward -
			0.5*3.0 -
			0.25*2.0

	bandit.UpdateReward(
		arm,
		nil,
		cheapFeedback,
	)

	stats, ok := bandit.Arms[arm]
	require.True(t, ok)

	require.Equal(
		t,
		int64(1),
		stats.Count,
	)

	assert.InDelta(
		t,
		expectedCheapReward,
		stats.AvgReward,
		1e-9,
	)

	bandit.UpdateReward(
		arm,
		nil,
		expensiveFeedback,
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
		expectedCheapReward+
			expectedExpensiveReward,
		stats.SumRewards,
		1e-9,
	)

	expectedAverageReward :=
		(expectedCheapReward + expectedExpensiveReward) / 2.0

	assert.InDelta(
		t,
		expectedAverageReward,
		stats.AvgReward,
		1e-9,
	)

	assert.Less(
		t,
		expectedExpensiveReward,
		expectedCheapReward,
	)
}

func TestLinUCBRewardUsesConcreteExecutionNodeFactors(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	viper.Set(
		config.MAB_COST_WEIGHT,
		0.5,
	)

	viper.Set(
		config.MAB_ENERGY_WEIGHT,
		0.25,
	)

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-node-specific-reward-test",
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
		NodeName:      "expensive-node",
		ExecutionNode: "expensive-node",
		CostFactor:    3.0,
		EnergyFactor:  2.0,
	}

	expectedReward :=
		-math.Log(10.0) -
			0.5*3.0 -
			0.25*2.0

	features :=
		bandit.computeFeatures(
			0.20,
		)

	bandit.UpdateReward(
		arm,
		ctx,
		feedback,
	)

	state, ok := bandit.Arms[arm]
	require.True(t, ok)

	// The LinUCB b vector starts from zero. After one update:
	//
	//     b = reward * x
	assert.InDelta(
		t,
		expectedReward*
			features.AtVec(0),
		state.b.AtVec(0),
		1e-9,
	)

	assert.InDelta(
		t,
		expectedReward*
			features.AtVec(1),
		state.b.AtVec(1),
		1e-9,
	)
}

// recordingPolicy captures the values passed by UpdateBandit.
// It allows the body-decoding and context-retrieval path to be tested without
// depending on the internal state of a concrete bandit implementation.
type recordingPolicy struct {
	updatedArm string
	ctx        *Context
	feedback   ExecutionFeedback
	updates    int
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

	previousStorage :=
		GlobalContextStorage

	t.Cleanup(func() {
		GlobalBanditManager =
			previousManager

		GlobalContextStorage =
			previousStorage
	})

	recorder :=
		&recordingPolicy{}

	GlobalBanditManager =
		&BanditManager{
			bandits: map[string]Policy{
				"hello": recorder,
			},
		}

	GlobalContextStorage =
		&ContextStorage{}

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"shared-ring": 0.40,
		},
	}

	GlobalContextStorage.Store(
		"request-1",
		ctx,
	)

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
		NodeName:     "node-a",
		CostFactor:   2.5,
		EnergyFactor: 1.7,
	}

	err = UpdateBandit(
		body,
		"/invoke/hello",
		"shared-ring",
		"request-1",
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

	assert.InDelta(
		t,
		2.5,
		recorder.feedback.CostFactor,
		1e-9,
	)

	assert.InDelta(
		t,
		1.7,
		recorder.feedback.EnergyFactor,
		1e-9,
	)

	// UpdateBandit must consume and remove the saved context.
	assert.Nil(
		t,
		GlobalContextStorage.RetrieveAndDelete(
			"request-1",
		),
	)
}
