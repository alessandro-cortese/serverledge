package lb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func isolateNodeMetricsForTest(
	t *testing.T,
) {
	t.Helper()

	previousMetrics := NodeMetrics

	NodeMetrics = &NodeMetricCache{
		metrics: make(map[string]NodeMetric),
	}

	t.Cleanup(func() {
		NodeMetrics = previousMetrics
	})
}

func TestExecutionFeedbackForNodeUsesConcreteNodeProfile(
	t *testing.T,
) {
	isolateNodeMetricsForTest(t)

	// These nodes may belong to the same machine-tag ring, but their
	// structural cost and energy profiles are different.
	NodeMetrics.Update(
		"cheap-node",
		512,
		1024,
		1,
		1.0,
	)

	NodeMetrics.UpdateCostProfile(
		"cheap-node",
		1.0,
		1.2,
	)

	NodeMetrics.Update(
		"expensive-node",
		512,
		1024,
		1,
		1.0,
	)

	NodeMetrics.UpdateCostProfile(
		"expensive-node",
		3.0,
		2.5,
	)

	cheapFeedback :=
		executionFeedbackForNode(
			"cheap-node",
		)

	expensiveFeedback :=
		executionFeedbackForNode(
			"expensive-node",
		)

	assert.Equal(
		t,
		"cheap-node",
		cheapFeedback.NodeName,
	)

	assert.InDelta(
		t,
		1.0,
		cheapFeedback.CostFactor,
		1e-9,
	)

	assert.InDelta(
		t,
		1.2,
		cheapFeedback.EnergyFactor,
		1e-9,
	)

	assert.Equal(
		t,
		"expensive-node",
		expensiveFeedback.NodeName,
	)

	assert.InDelta(
		t,
		3.0,
		expensiveFeedback.CostFactor,
		1e-9,
	)

	assert.InDelta(
		t,
		2.5,
		expensiveFeedback.EnergyFactor,
		1e-9,
	)

	assert.NotEqual(
		t,
		cheapFeedback.CostFactor,
		expensiveFeedback.CostFactor,
	)

	assert.NotEqual(
		t,
		cheapFeedback.EnergyFactor,
		expensiveFeedback.EnergyFactor,
	)
}

func TestExecutionFeedbackForUnknownNodeUsesDefaults(
	t *testing.T,
) {
	isolateNodeMetricsForTest(t)

	feedback :=
		executionFeedbackForNode(
			"unknown-node",
		)

	assert.Equal(
		t,
		"unknown-node",
		feedback.NodeName,
	)

	assert.InDelta(
		t,
		1.0,
		feedback.CostFactor,
		1e-9,
	)

	assert.InDelta(
		t,
		1.0,
		feedback.EnergyFactor,
		1e-9,
	)
}
