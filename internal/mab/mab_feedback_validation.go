package mab

import "math"

func isFiniteNumber(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateExecutionFeedback(policy string, functionName string, arm string, feedback ExecutionFeedback) bool {
	// Every invocation is first registered as observed. It can then be
	// classified as accepted, skipped or invalid.
	GlobalColdStartStats.RecordObserved(functionName, arm, feedback)
	logInvalidObservabilityMetrics(
		policy,
		functionName,
		arm,
		feedback,
	)

	switch {
	case !isFiniteNumber(feedback.DurationMs):
		recordInvalidExecutionFeedback(policy, functionName, arm, "non_finite_duration", feedback)
		return false

	case feedback.DurationMs <= 0:
		recordInvalidExecutionFeedback(policy, functionName, arm, "non_positive_duration", feedback)
		return false
	}

	return true
}

func logInvalidObservabilityMetrics(policy string, functionName string, arm string, feedback ExecutionFeedback) {
	metrics := []struct {
		name  string
		value float64
	}{
		{
			name:  "response_time_ms",
			value: feedback.ResponseTimeMs,
		},
		{
			name:  "init_time_ms",
			value: feedback.InitTimeMs,
		},
		{
			name:  "queueing_time_ms",
			value: feedback.QueueingTimeMs,
		},
		{
			name:  "offload_latency_ms",
			value: feedback.OffloadLatencyMs,
		},
	}

	for _, metric := range metrics {
		if isFiniteNumber(metric.value) && metric.value >= 0 {
			continue
		}

		logMABInvalidObservabilityMetric(
			policy,
			functionName,
			arm,
			metric.name,
			metric.value,
		)
	}
}

func recordInvalidExecutionFeedback(policy string, functionName string, arm string, reason string, feedback ExecutionFeedback) {

	GlobalColdStartStats.RecordInvalid(functionName, arm)
	logMABInvalidExecutionFeedback(
		policy,
		functionName,
		arm,
		reason,
		feedback,
	)

	logCurrentColdStartStats(
		policy,
		functionName,
		arm,
	)
}

func recordAcceptedExecutionFeedback(policy string, functionName string, arm string, feedback ExecutionFeedback) {
	if !feedback.IsWarmStart {
		GlobalColdStartStats.RecordColdAccepted(functionName, arm)
	}

	logCurrentColdStartStats(
		policy,
		functionName,
		arm,
	)
}

func logCurrentColdStartStats(policy string, functionName string, arm string) {
	logMABColdStartStats(
		policy,
		functionName,
		arm,
		GlobalColdStartStats.Snapshot(functionName, arm),
	)
}
