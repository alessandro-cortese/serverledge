package mab

import (
	"bytes"
	"log"
	"strings"
	"testing"
)

func captureLinUCBObservabilityLog(
	t *testing.T,
	fn func(),
) string {
	t.Helper()

	var buffer bytes.Buffer

	oldWriter := log.Writer()
	oldFlags := log.Flags()
	oldPrefix := log.Prefix()

	log.SetOutput(&buffer)
	log.SetFlags(0)
	log.SetPrefix("")

	t.Cleanup(
		func() {
			log.SetOutput(oldWriter)
			log.SetFlags(oldFlags)
			log.SetPrefix(oldPrefix)
		},
	)

	fn()

	return buffer.String()
}

func testLinUCBObservationDecision() DecisionRecord {
	return DecisionRecord{
		RequestID:    "request-test-1",
		FunctionName: "test-function",
		SelectedArm:  "x86",
		ExecutionArm: "x86",
		Context: &Context{
			ArchMemUsage: map[string]float64{
				"x86":   0.50,
				"arm64": 0.25,
			},
		},
		Fallback:       false,
		FallbackReason: "",
	}
}

func testLinUCBObservationFeedback() ExecutionFeedback {
	return ExecutionFeedback{
		DurationMs:       100.0,
		ResponseTimeMs:   110.0,
		InitTimeMs:       0.0,
		QueueingTimeMs:   5.0,
		OffloadLatencyMs: 0.0,
		IsWarmStart:      true,
		NodeName:         "node-x86-1",
		ExecutionNode:    "node-x86-1",
	}
}

func TestLinUCBObservabilityDisabled(t *testing.T) {
	t.Setenv(
		linUCBObservabilityEnv,
		"",
	)

	output :=
		captureLinUCBObservabilityLog(
			t,
			func() {
				logLinUCBRequestObservation(
					LinUCB,
					testLinUCBObservationDecision(),
					testLinUCBObservationFeedback(),
				)
			},
		)

	if output != "" {
		t.Fatalf(
			"expected no LinUCB observability log while disabled, got: %s",
			output,
		)
	}
}

func TestLinUCBObservabilityEnabled(t *testing.T) {
	t.Setenv(
		linUCBObservabilityEnv,
		"1",
	)

	output :=
		captureLinUCBObservabilityLog(
			t,
			func() {
				logLinUCBRequestObservation(
					LinUCB,
					testLinUCBObservationDecision(),
					testLinUCBObservationFeedback(),
				)
			},
		)

	expectedFragments := []string{
		"[LINUCB_OBSERVABILITY]",
		"event=request_observation",
		`request_id="request-test-1"`,
		"policy=LinUCB",
		`function="test-function"`,
		`selected_arm="x86"`,
		`execution_arm="x86"`,
		"fallback=false",
		"context_present=true",
		`context="arm64=0.250000,x86=0.500000"`,
		"selected_context_present=true",
		"selected_utilization=0.500000",
		"execution_context_present=true",
		"execution_utilization=0.500000",
		"duration_ms=100.000000",
		"response_time_ms=110.000000",
		"warm_start=true",
		`node_name="node-x86-1"`,
		`execution_node="node-x86-1"`,
	}

	for _, fragment := range expectedFragments {
		if !strings.Contains(
			output,
			fragment,
		) {
			t.Fatalf(
				"missing fragment %q in log:\n%s",
				fragment,
				output,
			)
		}
	}
}

func TestLinUCBObservabilityIgnoresNonLinUCBPolicy(
	t *testing.T,
) {
	t.Setenv(
		linUCBObservabilityEnv,
		"1",
	)

	output :=
		captureLinUCBObservabilityLog(
			t,
			func() {
				logLinUCBRequestObservation(
					UCB1,
					testLinUCBObservationDecision(),
					testLinUCBObservationFeedback(),
				)
			},
		)

	if output != "" {
		t.Fatalf(
			"expected no LinUCB observation for UCB1, got: %s",
			output,
		)
	}
}

func TestLinUCBObservabilityMarksMissingContext(
	t *testing.T,
) {
	t.Setenv(
		linUCBObservabilityEnv,
		"1",
	)

	decision :=
		testLinUCBObservationDecision()

	decision.Context =
		&Context{
			ArchMemUsage: map[string]float64{
				"arm64": 0.25,
			},
		}

	output :=
		captureLinUCBObservabilityLog(
			t,
			func() {
				logLinUCBRequestObservation(
					LinUCB,
					decision,
					testLinUCBObservationFeedback(),
				)
			},
		)

	if !strings.Contains(
		output,
		"selected_context_present=false",
	) {
		t.Fatalf(
			"missing selected-context diagnostic:\n%s",
			output,
		)
	}

	if !strings.Contains(
		output,
		"selected_utilization=0.000000",
	) {
		t.Fatalf(
			"missing zero-utilization fallback diagnostic:\n%s",
			output,
		)
	}
}
