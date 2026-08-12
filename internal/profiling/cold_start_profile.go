package profiling

import (
	"fmt"
	"strings"
)

const ColdStartProfileSchemaVersion = 1

// ColdStartInitTimeStatistics contains descriptive statistics for the
// initialization time of cold invocations.
//
// Cold-start measurements remain intentionally separate from the warm
// resource feature vector used for clustering.
type ColdStartInitTimeStatistics struct {
	MeanMs   float64 `json:"mean_ms"`
	MedianMs float64 `json:"median_ms"`
}

// ColdStartProfile aggregates cold-start initialization measurements for one
// function, one machine class and one fixed resource configuration.
type ColdStartProfile struct {
	SchemaVersion int `json:"schema_version"`

	FunctionName string `json:"function_name"`
	MachineTag   string `json:"machine_tag"`

	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`

	SampleCount int `json:"sample_count"`

	SourceRequestIDs []string `json:"source_request_ids"`

	InitTime ColdStartInitTimeStatistics `json:"init_time"`
}

// AggregateColdStartSamples aggregates homogeneous samples explicitly marked
// as eligible for cold-start analysis.
func AggregateColdStartSamples(
	samples []InvocationSample,
) (ColdStartProfile, error) {
	if len(samples) == 0 {
		return ColdStartProfile{},
			fmt.Errorf(
				"at least one cold-start sample is required",
			)
	}

	first := samples[0]

	if err := validateColdStartSample(first); err != nil {
		return ColdStartProfile{},
			fmt.Errorf(
				"invalid first cold-start sample: %w",
				err,
			)
	}

	initTimes := make(
		[]float64,
		0,
		len(samples),
	)

	requestIDs := make(
		[]string,
		0,
		len(samples),
	)

	seenRequestIDs := make(
		map[string]struct{},
		len(samples),
	)

	for index, sample := range samples {
		if err := validateColdStartSample(sample); err != nil {
			return ColdStartProfile{},
				fmt.Errorf(
					"invalid cold-start sample at index %d: %w",
					index,
					err,
				)
		}

		if sample.FunctionName != first.FunctionName {
			return ColdStartProfile{},
				fmt.Errorf(
					"mixed function names: %q and %q",
					first.FunctionName,
					sample.FunctionName,
				)
		}

		if sample.MachineTag != first.MachineTag {
			return ColdStartProfile{},
				fmt.Errorf(
					"mixed machine tags for function %q: %q and %q",
					first.FunctionName,
					first.MachineTag,
					sample.MachineTag,
				)
		}

		if !sameFunctionConfiguration(
			sample.FunctionConfiguration,
			first.FunctionConfiguration,
		) {
			return ColdStartProfile{},
				fmt.Errorf(
					"mixed function configurations for function %q on machine tag %q",
					first.FunctionName,
					first.MachineTag,
				)
		}

		requestID :=
			strings.TrimSpace(
				sample.RequestID,
			)

		if _, exists :=
			seenRequestIDs[requestID]; exists {

			return ColdStartProfile{},
				fmt.Errorf(
					"duplicate request ID %q",
					requestID,
				)
		}

		seenRequestIDs[requestID] =
			struct{}{}

		requestIDs =
			append(
				requestIDs,
				requestID,
			)

		initTimes =
			append(
				initTimes,
				sample.Timing.InitTimeMs,
			)
	}

	return ColdStartProfile{
		SchemaVersion: ColdStartProfileSchemaVersion,

		FunctionName: first.FunctionName,

		MachineTag: first.MachineTag,

		FunctionConfiguration: first.FunctionConfiguration,

		SampleCount: len(samples),

		SourceRequestIDs: requestIDs,

		InitTime: ColdStartInitTimeStatistics{
			MeanMs: meanFloat64(
				initTimes,
			),

			MedianMs: medianFloat64(
				initTimes,
			),
		},
	}, nil
}

// BuildColdStartProfileFromSamples ignores samples not marked for cold-start
// analysis and aggregates the remaining eligible measurements.
func BuildColdStartProfileFromSamples(
	samples []InvocationSample,
) (ColdStartProfile, error) {
	eligible :=
		make(
			[]InvocationSample,
			0,
			len(samples),
		)

	for index, sample := range samples {

		if !sample.
			Eligibility.
			ColdStartAnalysis {

			continue
		}

		if err :=
			validateColdStartSample(
				sample,
			); err != nil {

			return ColdStartProfile{},
				fmt.Errorf(
					"invalid eligible cold-start sample at index %d: %w",
					index,
					err,
				)
		}

		eligible =
			append(
				eligible,
				sample,
			)
	}

	return AggregateColdStartSamples(
		eligible,
	)
}

func validateColdStartSample(
	sample InvocationSample,
) error {
	if sample.SchemaVersion !=
		InvocationSampleSchemaVersion {

		return fmt.Errorf(
			"unsupported invocation sample schema version %d",
			sample.SchemaVersion,
		)
	}

	if strings.TrimSpace(
		sample.RequestID,
	) == "" {

		return fmt.Errorf(
			"request ID is empty",
		)
	}

	if strings.TrimSpace(
		sample.FunctionName,
	) == "" {

		return fmt.Errorf(
			"function name is empty",
		)
	}

	if strings.TrimSpace(
		sample.MachineTag,
	) == "" {

		return fmt.Errorf(
			"machine tag is empty",
		)
	}

	if !finitePositive(
		sample.
			FunctionConfiguration.
			ConfiguredCPUs,
	) {
		return fmt.Errorf(
			"configured CPUs are invalid: %v",
			sample.
				FunctionConfiguration.
				ConfiguredCPUs,
		)
	}

	if sample.
		FunctionConfiguration.
		ConfiguredMemoryMB <= 0 {

		return fmt.Errorf(
			"configured memory is invalid: %d MB",
			sample.
				FunctionConfiguration.
				ConfiguredMemoryMB,
		)
	}

	if !sample.
		Eligibility.
		ColdStartAnalysis {

		return fmt.Errorf(
			"sample is not eligible for cold-start analysis",
		)
	}

	if sample.WarmStart {
		return fmt.Errorf(
			"cold-start eligible sample is marked as warm",
		)
	}

	if !sample.ExecutionSucceeded {
		return fmt.Errorf(
			"cold-start eligible sample did not succeed",
		)
	}

	if !finiteNonNegative(
		sample.Timing.InitTimeMs,
	) {
		return fmt.Errorf(
			"init time is invalid: %v",
			sample.Timing.InitTimeMs,
		)
	}

	return nil
}
