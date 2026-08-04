package profiling

import (
	"math"
	"time"
)

const InvocationSampleSchemaVersion = 1

// InvocationTiming contains the timing dimensions associated with one
// invocation. Values are expressed in milliseconds to make the exported
// dataset explicit and independent from Go's time.Duration encoding.
type InvocationTiming struct {
	DurationMs          float64 `json:"duration_ms,omitempty"`
	ResponseTimeMs      float64 `json:"response_time_ms,omitempty"`
	InitTimeMs          float64 `json:"init_time_ms,omitempty"`
	QueueingTimeMs      float64 `json:"queueing_time_ms,omitempty"`
	OffloadLatencyMs    float64 `json:"offload_latency_ms,omitempty"`
	InvocationWaitMs    float64 `json:"invocation_wait_ms,omitempty"`
	ExecutionWallTimeMs float64 `json:"execution_wall_time_ms,omitempty"`
}

// InvocationEligibility keeps the raw sample and the decisions about which
// later analyses may use it separate. A cold sample is excluded from warm
// resource clustering and warm performance analysis, while remaining eligible
// for cold-start analysis.
type InvocationEligibility struct {
	ResourceClustering  bool     `json:"resource_clustering"`
	ColdStartAnalysis   bool     `json:"cold_start_analysis"`
	PerformanceAnalysis bool     `json:"performance_analysis"`
	ExclusionReasons    []string `json:"exclusion_reasons"`
}

// InvocationSample is the versioned raw record written to the JSONL dataset.
//
// Profile deliberately contains the complete InvocationResourceProfile.
// Therefore metrics added later from /proc, /sys, cgroups or Kepler will
// automatically be included in subsequent JSONL records.
type InvocationSample struct {
	SchemaVersion int   `json:"schema_version"`
	TimestampMs   int64 `json:"timestamp_ms"`

	RequestID    string `json:"request_id"`
	FunctionName string `json:"function_name"`
	MachineTag   string `json:"machine_tag"`
	NodeName     string `json:"node_name"`
	ContainerID  string `json:"container_id"`

	WarmStart          bool   `json:"warm_start"`
	ExecutionSucceeded bool   `json:"execution_succeeded"`
	ExecutionError     string `json:"execution_error,omitempty"`

	Timing      InvocationTiming           `json:"timing"`
	Profile     *InvocationResourceProfile `json:"profile,omitempty"`
	Eligibility InvocationEligibility      `json:"eligibility"`
}

// InvocationSampleInput contains the values required to build one raw sample.
type InvocationSampleInput struct {
	Timestamp time.Time

	RequestID    string
	FunctionName string
	MachineTag   string
	NodeName     string
	ContainerID  string

	WarmStart          bool
	ExecutionSucceeded bool
	ExecutionError     string

	Timing  InvocationTiming
	Profile *InvocationResourceProfile
}

// BuildInvocationSample builds one raw, versioned profiling record and assigns
// deterministic analysis eligibility flags.
func BuildInvocationSample(
	input InvocationSampleInput,
) InvocationSample {
	timestamp := input.Timestamp

	if timestamp.IsZero() {
		timestamp = time.Now()
	}

	containerID :=
		input.ContainerID

	if containerID == "" &&
		input.Profile != nil {

		containerID =
			input.Profile.ContainerID
	}

	return InvocationSample{
		SchemaVersion:      InvocationSampleSchemaVersion,
		TimestampMs:        timestamp.UnixMilli(),
		RequestID:          input.RequestID,
		FunctionName:       input.FunctionName,
		MachineTag:         input.MachineTag,
		NodeName:           input.NodeName,
		ContainerID:        containerID,
		WarmStart:          input.WarmStart,
		ExecutionSucceeded: input.ExecutionSucceeded,
		ExecutionError:     input.ExecutionError,
		Timing:             input.Timing,
		Profile:            input.Profile,
		Eligibility: buildInvocationEligibility(
			input.WarmStart,
			input.ExecutionSucceeded,
			input.Timing,
			input.Profile,
		),
	}
}

func buildInvocationEligibility(
	warmStart bool,
	executionSucceeded bool,
	timing InvocationTiming,
	profile *InvocationResourceProfile,
) InvocationEligibility {
	reasons :=
		make(
			[]string,
			0,
			7,
		)

	performanceTimingValid :=
		finiteNonNegative(
			timing.DurationMs,
		) &&
			finiteNonNegative(
				timing.ResponseTimeMs,
			)

	if !executionSucceeded {
		reasons =
			append(
				reasons,
				"execution_failed",
			)
	}

	if !warmStart {
		reasons =
			append(
				reasons,
				"cold_start",
			)
	}

	if warmStart &&
		!performanceTimingValid {

		reasons =
			append(
				reasons,
				"timing_invalid",
			)
	}

	// Il profilo delle risorse è richiesto soltanto per
	// le invocazioni warm.
	profileUsable := false

	if warmStart {
		profileUsable = true

		if profile == nil {
			profileUsable = false

			reasons =
				append(
					reasons,
					"profile_missing",
				)
		} else {
			if !profile.Enabled {
				profileUsable = false

				reasons =
					append(
						reasons,
						"profile_disabled",
					)
			}

			if !profile.Collected {
				profileUsable = false

				reasons =
					append(
						reasons,
						"profile_not_collected",
					)
			}

			if !profile.Valid {
				profileUsable = false

				reasons =
					append(
						reasons,
						"profile_invalid",
					)
			}

			if !profile.ExclusiveContainer {
				profileUsable = false

				reasons =
					append(
						reasons,
						"container_not_exclusive",
					)
			}
		}
	}

	performanceAnalysis :=
		warmStart &&
			executionSucceeded &&
			performanceTimingValid

	coldStartAnalysis :=
		!warmStart &&
			executionSucceeded &&
			finiteNonNegative(
				timing.InitTimeMs,
			)

	resourceClustering :=
		performanceAnalysis &&
			profileUsable

	return InvocationEligibility{
		ResourceClustering:  resourceClustering,
		ColdStartAnalysis:   coldStartAnalysis,
		PerformanceAnalysis: performanceAnalysis,
		ExclusionReasons:    reasons,
	}
}

func finiteNonNegative(
	value float64,
) bool {
	return value >= 0 &&
		!math.IsNaN(
			value,
		) &&
		!math.IsInf(
			value,
			0,
		)
}
