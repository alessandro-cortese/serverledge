package profiling

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestAggregateColdStartSamplesComputesMeanAndOddMedian(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"request-1",
				"function-a",
				"x86",
				100,
				1000,
			),
			coldStartSampleForTest(
				"request-2",
				"function-a",
				"x86",
				200,
				2000,
			),
			coldStartSampleForTest(
				"request-3",
				"function-a",
				"x86",
				900,
				3000,
			),
		}

	profile, err :=
		AggregateColdStartSamples(
			samples,
		)

	if err != nil {
		t.Fatalf(
			"AggregateColdStartSamples returned error: %v",
			err,
		)
	}

	if profile.SampleCount != 3 {
		t.Fatalf(
			"expected sample count 3, got %d",
			profile.SampleCount,
		)
	}

	// (100 + 200 + 900) / 3 = 400
	assertColdFloatEqual(
		t,
		400,
		profile.InitTime.MeanMs,
	)

	// Sorted values: 100, 200, 900.
	assertColdFloatEqual(
		t,
		200,
		profile.InitTime.MedianMs,
	)

	if profile.SchemaVersion !=
		ColdStartProfileSchemaVersion {

		t.Fatalf(
			"expected schema version %d, got %d",
			ColdStartProfileSchemaVersion,
			profile.SchemaVersion,
		)
	}

	if profile.FunctionName != "function-a" {
		t.Fatalf(
			"unexpected function name %q",
			profile.FunctionName,
		)
	}

	if profile.MachineTag != "x86" {
		t.Fatalf(
			"unexpected machine tag %q",
			profile.MachineTag,
		)
	}
}

func TestAggregateColdStartSamplesComputesEvenMedian(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"request-1",
				"function-a",
				"x86",
				100,
				1000,
			),
			coldStartSampleForTest(
				"request-2",
				"function-a",
				"x86",
				200,
				2000,
			),
			coldStartSampleForTest(
				"request-3",
				"function-a",
				"x86",
				300,
				3000,
			),
			coldStartSampleForTest(
				"request-4",
				"function-a",
				"x86",
				900,
				4000,
			),
		}

	profile, err :=
		AggregateColdStartSamples(
			samples,
		)

	if err != nil {
		t.Fatalf(
			"AggregateColdStartSamples returned error: %v",
			err,
		)
	}

	// (100 + 200 + 300 + 900) / 4 = 375
	assertColdFloatEqual(
		t,
		375,
		profile.InitTime.MeanMs,
	)

	// Sorted values: 100, 200, 300, 900.
	// Median = (200 + 300) / 2 = 250.
	assertColdFloatEqual(
		t,
		250,
		profile.InitTime.MedianMs,
	)
}

func TestAggregateColdStartSamplesRejectsDuplicateRequestID(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"duplicate-request",
				"function-a",
				"x86",
				100,
				1000,
			),
			coldStartSampleForTest(
				"duplicate-request",
				"function-a",
				"x86",
				200,
				2000,
			),
		}

	_, err :=
		AggregateColdStartSamples(
			samples,
		)

	if err == nil {
		t.Fatal(
			"expected duplicate request ID error",
		)
	}

	if !strings.Contains(
		err.Error(),
		"duplicate request ID",
	) {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}
}

func TestAggregateColdStartSamplesRejectsMixedMachineTags(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"request-x86",
				"function-a",
				"x86",
				100,
				1000,
			),
			coldStartSampleForTest(
				"request-arm",
				"function-a",
				"arm64",
				200,
				2000,
			),
		}

	_, err :=
		AggregateColdStartSamples(
			samples,
		)

	if err == nil {
		t.Fatal(
			"expected mixed machine tag error",
		)
	}

	if !strings.Contains(
		err.Error(),
		"mixed machine tags",
	) {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}
}

func TestBuildColdStartProfileFromSamplesIgnoresWarmSamples(
	t *testing.T,
) {
	cold :=
		coldStartSampleForTest(
			"cold-request",
			"function-a",
			"x86",
			150,
			1000,
		)

	warm :=
		BuildInvocationSample(
			InvocationSampleInput{
				Timestamp: time.UnixMilli(
					2000,
				),

				RequestID: "warm-request",

				FunctionName: "function-a",

				MachineTag: "x86",

				NodeName: "node-x86",

				ContainerID: "container-warm",

				ConfiguredCPUs: 1,

				ConfiguredMemoryMB: 128,

				WarmStart: true,

				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					DurationMs: 10,

					ResponseTimeMs: 10,
				},
			},
		)

	if !cold.
		Eligibility.
		ColdStartAnalysis {

		t.Fatal(
			"test setup error: cold sample is not eligible for cold-start analysis",
		)
	}

	if warm.
		Eligibility.
		ColdStartAnalysis {

		t.Fatal(
			"test setup error: warm sample unexpectedly eligible for cold-start analysis",
		)
	}

	profile, err :=
		BuildColdStartProfileFromSamples(
			[]InvocationSample{
				warm,
				cold,
			},
		)

	if err != nil {
		t.Fatalf(
			"BuildColdStartProfileFromSamples returned error: %v",
			err,
		)
	}

	if profile.SampleCount != 1 {
		t.Fatalf(
			"expected one cold sample, got %d",
			profile.SampleCount,
		)
	}

	if len(
		profile.SourceRequestIDs,
	) != 1 {

		t.Fatalf(
			"expected one source request ID, got %d",
			len(
				profile.SourceRequestIDs,
			),
		)
	}

	if profile.SourceRequestIDs[0] !=
		"cold-request" {

		t.Fatalf(
			"expected cold-request, got %q",
			profile.SourceRequestIDs[0],
		)
	}

	assertColdFloatEqual(
		t,
		150,
		profile.InitTime.MeanMs,
	)

	assertColdFloatEqual(
		t,
		150,
		profile.InitTime.MedianMs,
	)
}

// coldStartSampleForTest creates one successful cold invocation through the
// production InvocationSample builder. This also verifies that the existing
// eligibility logic marks the sample for ColdStartAnalysis.
func coldStartSampleForTest(
	requestID string,
	functionName string,
	machineTag string,
	initTimeMs float64,
	timestampMs int64,
) InvocationSample {
	return BuildInvocationSample(
		InvocationSampleInput{
			Timestamp: time.UnixMilli(
				timestampMs,
			),

			RequestID: requestID,

			FunctionName: functionName,

			MachineTag: machineTag,

			NodeName: "node-" +
				machineTag,

			ContainerID: "container-" +
				requestID,

			ConfiguredCPUs: 1,

			ConfiguredMemoryMB: 128,

			WarmStart: false,

			ExecutionSucceeded: true,

			Timing: InvocationTiming{
				InitTimeMs: initTimeMs,
			},
		},
	)
}

func assertColdFloatEqual(
	t *testing.T,
	expected float64,
	actual float64,
) {
	t.Helper()

	const tolerance = 1e-9

	if math.Abs(
		expected-actual,
	) > tolerance {

		t.Fatalf(
			"expected %.12f, got %.12f",
			expected,
			actual,
		)
	}
}
