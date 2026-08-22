package profiling

import (
	"fmt"
	"math"
	"sort"
)

const (
	FunctionProfileSchemaVersion = 1
	MinFunctionProfileSamples    = 10
	MaxFunctionProfileSamples    = 20
)

// FunctionProfile is the aggregated profiling representation of one function
// executed on one machine class with one fixed resource configuration.
//
// Mean and Median are both retained because the experimental phase will compare
// which aggregation is more robust before clustering.
type FunctionProfile struct {
	SchemaVersion         int                             `json:"schema_version"`
	FunctionName          string                          `json:"function_name"`
	MachineTag            string                          `json:"machine_tag"`
	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`
	SampleCount           int                             `json:"sample_count"`
	SourceRequestIDs      []string                        `json:"source_request_ids"`
	Mean                  RandomForestProfilingFeatures   `json:"mean"`
	Median                RandomForestProfilingFeatures   `json:"median"`
}

// MeanValues returns the mean feature vector in the stable order defined by
// RandomForestFeatureNames.
func (profile FunctionProfile) MeanValues() []float64 {
	return profile.Mean.Values()
}

// MedianValues returns the median feature vector in the stable order defined by
// RandomForestFeatureNames.
func (profile FunctionProfile) MedianValues() []float64 {
	return profile.Median.Values()
}

// AggregateProfilingFeatureVectors aggregates between 10 and 20 feature
// vectors belonging to the same function, machine tag and resource
// configuration.
//
// Request IDs must be unique. Mixing executions produced with different
// function configurations is deliberately rejected because those measurements
// would not describe the same experimental condition.
func AggregateProfilingFeatureVectors(vectors []ProfilingFeatureVector) (FunctionProfile, error) {

	if len(vectors) < MinFunctionProfileSamples {
		return FunctionProfile{}, fmt.Errorf("not enough profiling samples: got %d, need at least %d", len(vectors), MinFunctionProfileSamples)
	}

	if len(vectors) > MaxFunctionProfileSamples {
		return FunctionProfile{}, fmt.Errorf("too many profiling samples: got %d, maximum is %d", len(vectors), MaxFunctionProfileSamples)
	}

	first := vectors[0]

	if err := validateVectorIdentity(first); err != nil {
		return FunctionProfile{}, fmt.Errorf("invalid first profiling vector: %w", err)
	}

	featureCount := len(RandomForestFeatureNames())
	featureColumns := make([][]float64, featureCount)

	for index := range featureColumns {
		featureColumns[index] = make([]float64, 0, len(vectors))
	}

	requestIDs := make([]string, 0, len(vectors))
	seenRequestIDs := make(map[string]struct{}, len(vectors))
	names := RandomForestFeatureNames()

	for index, vector := range vectors {
		if err := validateVectorIdentity(vector); err != nil {
			return FunctionProfile{}, fmt.Errorf("invalid profiling vector at index %d: %w", index, err)
		}

		if vector.FunctionName != first.FunctionName {
			return FunctionProfile{}, fmt.Errorf("mixed function names: %q and %q", first.FunctionName, vector.FunctionName)
		}

		if vector.MachineTag != first.MachineTag {
			return FunctionProfile{}, fmt.Errorf("mixed machine tags for function %q: %q and %q", first.FunctionName, first.MachineTag, vector.MachineTag)
		}

		if !sameFunctionConfiguration(vector.FunctionConfiguration, first.FunctionConfiguration) {
			return FunctionProfile{}, fmt.Errorf("mixed function configurations for function %q on machine tag %q", first.FunctionName, first.MachineTag)
		}

		if _, exists := seenRequestIDs[vector.RequestID]; exists {
			return FunctionProfile{}, fmt.Errorf("duplicate request ID %q", vector.RequestID)
		}

		seenRequestIDs[vector.RequestID] = struct{}{}
		requestIDs = append(requestIDs, vector.RequestID)
		values := vector.Features.Values()

		if len(values) != featureCount {
			return FunctionProfile{}, fmt.Errorf("profiling vector %q contains %d features, expected %d", vector.RequestID, len(values), featureCount)
		}

		for featureIndex, value := range values {
			if !finiteNonNegative(value) {
				return FunctionProfile{}, fmt.Errorf("profiling vector %q contains invalid feature %s=%v", vector.RequestID, names[featureIndex], value)
			}

			featureColumns[featureIndex] = append(featureColumns[featureIndex], value)
		}
	}

	meanValues := make([]float64, featureCount)
	medianValues := make([]float64, featureCount)

	for featureIndex, values := range featureColumns {
		meanValues[featureIndex] = meanFloat64(values)
		medianValues[featureIndex] = medianFloat64(values)
	}

	meanFeatures, err := randomForestFeaturesFromValues(meanValues)
	if err != nil {
		return FunctionProfile{}, fmt.Errorf("failed to build mean feature vector: %w", err)
	}

	medianFeatures, err := randomForestFeaturesFromValues(medianValues)
	if err != nil {
		return FunctionProfile{}, fmt.Errorf("failed to build median feature vector: %w", err)
	}

	return FunctionProfile{
		SchemaVersion:         FunctionProfileSchemaVersion,
		FunctionName:          first.FunctionName,
		MachineTag:            first.MachineTag,
		FunctionConfiguration: first.FunctionConfiguration,
		SampleCount:           len(vectors),
		SourceRequestIDs:      requestIDs,
		Mean:                  meanFeatures,
		Median:                medianFeatures,
	}, nil
}

// BuildFunctionProfileFromSamples converts the eligible raw invocation samples
// into feature vectors and aggregates them.
//
// Samples explicitly marked as not eligible for resource clustering are
// ignored. An eligible sample that cannot be converted into a feature vector
// is considered an error because it indicates an inconsistency in the dataset.
func BuildFunctionProfileFromSamples(samples []InvocationSample) (FunctionProfile, error) {

	vectors := make([]ProfilingFeatureVector, 0, len(samples))
	for index, sample := range samples {
		if !sample.Eligibility.ResourceClustering {
			continue
		}

		vector, err := BuildProfilingFeatureVector(sample)
		if err != nil {
			return FunctionProfile{}, fmt.Errorf("failed to build feature vector from eligible sample at index %d: %w", index, err)
		}
		vectors = append(vectors, vector)
	}
	return AggregateProfilingFeatureVectors(vectors)
}

func validateVectorIdentity(vector ProfilingFeatureVector) error {

	if vector.SchemaVersion != ProfilingFeatureVectorSchemaVersion {
		return fmt.Errorf("unsupported feature vector schema version %d", vector.SchemaVersion)
	}

	if vector.RequestID == "" {
		return fmt.Errorf("request ID is empty")
	}

	if vector.FunctionName == "" {
		return fmt.Errorf("function name is empty")
	}

	if vector.MachineTag == "" {
		return fmt.Errorf("machine tag is empty")
	}

	if !finitePositive(vector.FunctionConfiguration.ConfiguredCPUs) {
		return fmt.Errorf("configured CPUs are invalid: %v", vector.FunctionConfiguration.ConfiguredCPUs)
	}

	if vector.FunctionConfiguration.ConfiguredMemoryMB <= 0 {
		return fmt.Errorf("configured memory is invalid: %d MB", vector.FunctionConfiguration.ConfiguredMemoryMB)
	}

	return nil
}

func sameFunctionConfiguration(first InvocationFunctionConfiguration, second InvocationFunctionConfiguration) bool {

	const cpuTolerance = 1e-9
	return math.Abs(first.ConfiguredCPUs-second.ConfiguredCPUs) <= cpuTolerance && first.ConfiguredMemoryMB == second.ConfiguredMemoryMB
}

func meanFloat64(values []float64) float64 {
	var mean float64
	for index, value := range values {
		// Incremental mean avoids unnecessarily large intermediate sums.
		mean += (value - mean) / float64(index+1)
	}
	return mean
}

func medianFloat64(values []float64) float64 {

	sortedValues := append([]float64(nil), values...)
	sort.Float64s(sortedValues)
	middle := len(sortedValues) / 2
	if len(sortedValues)%2 != 0 {
		return sortedValues[middle]
	}
	return (sortedValues[middle-1] + sortedValues[middle]) / 2.0
}

func randomForestFeaturesFromValues(values []float64) (RandomForestProfilingFeatures, error) {
	expectedFeatureCount := len(RandomForestFeatureNames())

	if len(values) != expectedFeatureCount {
		return RandomForestProfilingFeatures{}, fmt.Errorf("expected %d feature values, got %d", expectedFeatureCount, len(values))
	}

	names := RandomForestFeatureNames()
	for index, value := range values {
		if !finiteNonNegative(value) {
			return RandomForestProfilingFeatures{}, fmt.Errorf("invalid feature %s=%v", names[index], value)
		}
	}

	return RandomForestProfilingFeatures{
		PageFaultsDelta:    values[0],
		UtilizedCPUs:       values[1],
		FreeMemoryMB:       values[2],
		CPUUserDeltaMs:     values[3],
		CPUKernelDeltaMs:   values[4],
		FrameworkRuntimeMs: values[5],
	}, nil
}
