package profiling

import (
	"fmt"
	"math"
)

const (
	PaperRFFeatureVectorSchemaVersion    = 1
	PaperRFInvocationSampleSchemaVersion = 6
)

// PaperRFProfilingFeatures contains the 14 columns that remain after the
// published classifier artifact drops metadata and the seven unused SAAF
// dimensions.
//
// Names and units deliberately follow SAAF/the reference artifact where
// possible:
//   - memory is expressed in kB;
//   - timing and CPU deltas are expressed in ms;
//   - utilizedCPUs is dimensionless and rounded to three decimals, as in SAAF.
type PaperRFProfilingFeatures struct {
	TotalMemory            float64 `json:"totalMemory"`
	FreeMemory             float64 `json:"freeMemory"`
	FrameworkRuntime       float64 `json:"frameworkRuntime"`
	UserRuntime            float64 `json:"userRuntime"`
	CPUUserDelta           float64 `json:"cpuUserDelta"`
	CPUKernelDelta         float64 `json:"cpuKernelDelta"`
	CPUIdleDelta           float64 `json:"cpuIdleDelta"`
	CPUIOWaitDelta         float64 `json:"cpuIOWaitDelta"`
	CPUSoftIRQDelta        float64 `json:"cpuSoftIrqDelta"`
	CPUStealDelta          float64 `json:"cpuStealDelta"`
	PageFaultsDelta        float64 `json:"pageFaultsDelta"`
	UtilizedCPUs           float64 `json:"utilizedCPUs"`
	FrameworkRuntimeDeltas float64 `json:"frameworkRuntimeDeltas"`
	Runtime                float64 `json:"runtime"`
}

// PaperRFFeatureVector keeps identifiers and Serverledge allocation metadata
// outside the classifier feature block.
type PaperRFFeatureVector struct {
	SchemaVersion         int                             `json:"schema_version"`
	RequestID             string                          `json:"request_id"`
	FunctionName          string                          `json:"function_name"`
	MachineTag            string                          `json:"machine_tag"`
	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`
	Features              PaperRFProfilingFeatures        `json:"features"`
}

// PaperRFFeatureNames returns the exact column order consumed by the
// paper-faithful Random Forest adapter.
func PaperRFFeatureNames() []string {
	return []string{
		"totalMemory",
		"freeMemory",
		"frameworkRuntime",
		"userRuntime",
		"cpuUserDelta",
		"cpuKernelDelta",
		"cpuIdleDelta",
		"cpuIOWaitDelta",
		"cpuSoftIrqDelta",
		"cpuStealDelta",
		"pageFaultsDelta",
		"utilizedCPUs",
		"frameworkRuntimeDeltas",
		"runtime",
	}
}

// Values returns the numeric vector in exactly the same order returned by
// PaperRFFeatureNames.
func (f PaperRFProfilingFeatures) Values() []float64 {
	return []float64{
		f.TotalMemory,
		f.FreeMemory,
		f.FrameworkRuntime,
		f.UserRuntime,
		f.CPUUserDelta,
		f.CPUKernelDelta,
		f.CPUIdleDelta,
		f.CPUIOWaitDelta,
		f.CPUSoftIRQDelta,
		f.CPUStealDelta,
		f.PageFaultsDelta,
		f.UtilizedCPUs,
		f.FrameworkRuntimeDeltas,
		f.Runtime,
	}
}

// PaperRFSampleEligible reports whether one schema-6 sample contains all
// measurements needed by the 14-feature adapter.
//
// Cold starts, failed executions, invalid/non-exclusive container profiles and
// incomplete node snapshots are intentionally excluded.
func PaperRFSampleEligible(sample InvocationSample) bool {
	return validatePaperRFSampleInput(sample) == nil
}

// BuildPaperRFFeatureVector maps one Serverledge InvocationSample to the 14
// features used by the published Random Forest artifact.
//
// The mapping is intentionally separate from BuildProfilingFeatureVector: the
// clustering pipeline keeps its existing container-scoped six-feature
// semantics, while this adapter uses the node/procfs semantics closest to SAAF.
func BuildPaperRFFeatureVector(sample InvocationSample) (PaperRFFeatureVector, error) {
	if err := validatePaperRFSampleInput(sample); err != nil {
		return PaperRFFeatureVector{}, err
	}

	profile := sample.Profile
	node := sample.NodeEnvironment

	// SAAF obtains these values from /proc/meminfo, whose native unit is kB.
	totalMemoryKB := float64(node.TotalMemoryBeforeBytes) / 1024.0
	freeMemoryKB := float64(node.FreeMemoryBeforeBytes) / 1024.0

	// The SAAF userRuntime interval is the workload interval after the initial
	// inspector phase. DurationMs is the closest Serverledge quantity because it
	// removes invocationWait from the outer Execute wall-clock interval.
	userRuntimeMs := sample.Timing.DurationMs

	// SAAF 0.7 reads the aggregate /proc/stat counters and converts every
	// jiffy directly with 1000/CLK_TCK. It does not subtract guest time from
	// cpuUser (or guest_nice from cpuNice), so the adapter deliberately uses
	// the raw tick deltas rather than Serverledge's historical normalized-ms
	// fields.
	tickMilliseconds :=
		1000.0 /
			float64(
				node.CPUClockTicksPerSecond,
			)

	cpuUserDeltaMs :=
		float64(
			node.CPUUserDeltaTicks,
		) * tickMilliseconds

	cpuKernelDeltaMs :=
		float64(
			node.CPUKernelDeltaTicks,
		) * tickMilliseconds

	cpuIdleDeltaMs :=
		float64(
			node.CPUIdleDeltaTicks,
		) * tickMilliseconds

	cpuIOWaitDeltaMs :=
		float64(
			node.CPUIOWaitDeltaTicks,
		) * tickMilliseconds

	cpuSoftIRQDeltaMs :=
		float64(
			node.CPUSoftIRQDeltaTicks,
		) * tickMilliseconds

	cpuStealDeltaMs :=
		float64(
			node.CPUStealDeltaTicks,
		) * tickMilliseconds

	// SAAF 0.7 computes utilizedCPUs as (user + kernel) / userRuntime and rounds
	// the result to three decimal places before storing the feature.
	utilizedCPUs :=
		math.Round(
			((cpuUserDeltaMs+cpuKernelDeltaMs)/userRuntimeMs)*1000.0,
		) / 1000.0

	// SAAF measures initial and final inspection phases inside the function.
	// Serverledge performs the corresponding observation externally, therefore
	// the closest available mapping is the sum of the node and container
	// snapshot overheads on each side of the invocation.
	frameworkRuntimeMs :=
		profile.ProfilingStartOverheadMs +
			node.SnapshotStartOverheadMs

	frameworkRuntimeDeltasMs :=
		profile.ProfilingEndOverheadMs +
			node.SnapshotEndOverheadMs

	runtimeMs :=
		frameworkRuntimeMs +
			userRuntimeMs +
			frameworkRuntimeDeltasMs

	features := PaperRFProfilingFeatures{
		TotalMemory:            totalMemoryKB,
		FreeMemory:             freeMemoryKB,
		FrameworkRuntime:       frameworkRuntimeMs,
		UserRuntime:            userRuntimeMs,
		CPUUserDelta:           cpuUserDeltaMs,
		CPUKernelDelta:         cpuKernelDeltaMs,
		CPUIdleDelta:           cpuIdleDeltaMs,
		CPUIOWaitDelta:         cpuIOWaitDeltaMs,
		CPUSoftIRQDelta:        cpuSoftIRQDeltaMs,
		CPUStealDelta:          cpuStealDeltaMs,
		PageFaultsDelta:        float64(node.PageFaultsDelta),
		UtilizedCPUs:           utilizedCPUs,
		FrameworkRuntimeDeltas: frameworkRuntimeDeltasMs,
		Runtime:                runtimeMs,
	}

	names := PaperRFFeatureNames()
	values := features.Values()

	for index, value := range values {
		if !finiteNonNegative(value) {
			return PaperRFFeatureVector{},
				fmt.Errorf(
					"sample %q produced invalid paper RF feature %s=%v",
					sample.RequestID,
					names[index],
					value,
				)
		}
	}

	if !finitePositive(features.TotalMemory) {
		return PaperRFFeatureVector{},
			fmt.Errorf(
				"sample %q produced invalid paper RF feature totalMemory=%v",
				sample.RequestID,
				features.TotalMemory,
			)
	}

	return PaperRFFeatureVector{
		SchemaVersion:         PaperRFFeatureVectorSchemaVersion,
		RequestID:             sample.RequestID,
		FunctionName:          sample.FunctionName,
		MachineTag:            sample.MachineTag,
		FunctionConfiguration: sample.FunctionConfiguration,
		Features:              features,
	}, nil
}

// BuildPaperRFFeatureVectors converts all usable schema-6 warm invocations.
//
// A dataset containing another raw schema is rejected instead of silently
// mixing incompatible measurement contracts. Within a schema-6 dataset,
// unusable measurements are skipped so that collection scripts can keep
// gathering until the requested number of valid examples is reached.
func BuildPaperRFFeatureVectors(
	samples []InvocationSample,
) ([]PaperRFFeatureVector, error) {
	vectors :=
		make(
			[]PaperRFFeatureVector,
			0,
			len(samples),
		)

	for index, sample := range samples {
		if sample.SchemaVersion != PaperRFInvocationSampleSchemaVersion {
			return nil,
				fmt.Errorf(
					"sample at index %d uses InvocationSample schema %d: paper RF export requires schema %d",
					index,
					sample.SchemaVersion,
					PaperRFInvocationSampleSchemaVersion,
				)
		}

		if !PaperRFSampleEligible(sample) {
			continue
		}

		vector, err :=
			BuildPaperRFFeatureVector(
				sample,
			)

		if err != nil {
			return nil,
				fmt.Errorf(
					"failed to build paper RF feature vector from sample at index %d: %w",
					index,
					err,
				)
		}

		vectors =
			append(
				vectors,
				vector,
			)
	}

	return vectors, nil
}

func validatePaperRFSampleInput(
	sample InvocationSample,
) error {
	if sample.SchemaVersion != PaperRFInvocationSampleSchemaVersion {
		return fmt.Errorf(
			"sample %q uses InvocationSample schema %d: expected %d",
			sample.RequestID,
			sample.SchemaVersion,
			PaperRFInvocationSampleSchemaVersion,
		)
	}

	if !sample.WarmStart {
		return fmt.Errorf(
			"sample %q is not a warm invocation",
			sample.RequestID,
		)
	}

	if !sample.ExecutionSucceeded {
		return fmt.Errorf(
			"sample %q did not execute successfully",
			sample.RequestID,
		)
	}

	if sample.Profile == nil {
		return fmt.Errorf(
			"sample %q has no container profile",
			sample.RequestID,
		)
	}

	profile := sample.Profile

	if !profile.Enabled ||
		!profile.Collected ||
		!profile.Valid ||
		!profile.ExclusiveContainer {

		return fmt.Errorf(
			"sample %q has no usable exclusive container profile",
			sample.RequestID,
		)
	}

	if sample.NodeEnvironment == nil {
		return fmt.Errorf(
			"sample %q has no node environment",
			sample.RequestID,
		)
	}

	node := sample.NodeEnvironment

	if !node.Collected ||
		!node.CPUAvailable ||
		!node.MemoryAvailable ||
		!node.VMStatAvailable {

		return fmt.Errorf(
			"sample %q has incomplete node CPU/memory/vmstat measurements",
			sample.RequestID,
		)
	}

	if !finitePositive(sample.Timing.DurationMs) {
		return fmt.Errorf(
			"sample %q has invalid duration_ms %.6f",
			sample.RequestID,
			sample.Timing.DurationMs,
		)
	}

	if !node.CPUClockTicksAvailable ||
		node.CPUClockTicksPerSecond == 0 {

		return fmt.Errorf(
			"sample %q has no usable CLK_TCK value for direct /proc/stat conversion",
			sample.RequestID,
		)
	}

	requiredNonNegative := []struct {
		name  string
		value float64
	}{
		{"profiling_start_overhead_ms", profile.ProfilingStartOverheadMs},
		{"profiling_end_overhead_ms", profile.ProfilingEndOverheadMs},
		{"snapshot_start_overhead_ms", node.SnapshotStartOverheadMs},
		{"snapshot_end_overhead_ms", node.SnapshotEndOverheadMs},
	}

	for _, field := range requiredNonNegative {
		if !finiteNonNegative(field.value) {
			return fmt.Errorf(
				"sample %q has invalid %s=%v",
				sample.RequestID,
				field.name,
				field.value,
			)
		}
	}

	if node.TotalMemoryBeforeBytes == 0 {
		return fmt.Errorf(
			"sample %q has non-positive total node memory",
			sample.RequestID,
		)
	}

	return nil
}
