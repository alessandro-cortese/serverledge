package profiling

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const DefaultFunctionProfileExportPath = "data/profiling/function-profiles.jsonl"

// FunctionProfileGroupStatus describes whether one homogeneous profiling group
// had enough eligible samples to produce a FunctionProfile.
type FunctionProfileGroupStatus struct {
	FunctionName          string                          `json:"function_name"`
	MachineTag            string                          `json:"machine_tag"`
	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`
	EligibleSampleCount   int                             `json:"eligible_sample_count"`
	SelectedSampleCount   int                             `json:"selected_sample_count"`
	Built                 bool                            `json:"built"`
}

// FunctionProfileBuildResult contains the aggregated profiles and a summary of
// how the raw invocation dataset was partitioned.
type FunctionProfileBuildResult struct {
	Profiles        []FunctionProfile
	Groups          []FunctionProfileGroupStatus
	RawSampleCount  int
	EligibleSamples int
	IgnoredSamples  int
}

type functionProfileGroupKey struct {
	FunctionName       string
	MachineTag         string
	ConfiguredCPUs     float64
	ConfiguredMemoryMB int64
}

type timestampedInvocationSample struct {
	TimestampMs int64
	InputOrder  int
	Sample      InvocationSample
}

// LoadInvocationSamplesJSONL loads the raw append-only profiling dataset.
//
// Every non-empty line must contain one InvocationSample using the current
// schema version.
func LoadInvocationSamplesJSONL(path string) ([]InvocationSample, error) {

	samples := make([]InvocationSample, 0)
	if err := scanJSONLines(path, "profiling sample", func(line []byte, lineNumber int) error {
		var sample InvocationSample
		if err := json.Unmarshal(line, &sample); err != nil {
			return fmt.Errorf("invalid profiling sample JSON at line %d: %w", lineNumber, err)
		}

		if sample.SchemaVersion != InvocationSampleSchemaVersion {
			return fmt.Errorf("unsupported invocation sample schema version %d at line %d: expected %d", sample.SchemaVersion, lineNumber, InvocationSampleSchemaVersion)
		}

		samples = append(samples, sample)
		return nil
	}); err != nil {
		return nil, err
	}

	return samples, nil
}

// LoadInvocationSamplesJSONLFiles loads and merges multiple per-node raw
// profiling datasets. Input paths are sorted to keep deterministic behavior.
//
// Request IDs must be globally unique across all input datasets. This also
// prevents the same invocation from contributing more than once if datasets
// are accidentally duplicated during collection.
func LoadInvocationSamplesJSONLFiles(paths []string) ([]InvocationSample, error) {

	if len(paths) == 0 {
		return nil, fmt.Errorf("at least one profiling sample input dataset is required")
	}

	normalizedPaths := make([]string, 0, len(paths))
	seenPaths := make(map[string]struct{}, len(paths))

	for _, path := range paths {
		path = strings.TrimSpace(path)

		if path == "" {
			return nil, fmt.Errorf("profiling sample input path cannot be empty")
		}

		absPath, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve profiling sample input path %s: %w", path, err)
		}

		cleanPath := filepath.Clean(absPath)

		if _, exists := seenPaths[cleanPath]; exists {
			return nil, fmt.Errorf("duplicate profiling sample input path %s", cleanPath)
		}

		seenPaths[cleanPath] = struct{}{}
		normalizedPaths = append(normalizedPaths, cleanPath)
	}

	// The input order must not depend on the order in which files happened to
	// be supplied by a shell script or discovered in the filesystem.
	sort.Strings(normalizedPaths)
	merged := make([]InvocationSample, 0)
	requestSources := make(map[string]string)
	for _, path := range normalizedPaths {

		samples, err := LoadInvocationSamplesJSONL(path)
		if err != nil {
			return nil, err
		}

		for index, sample := range samples {
			requestID := strings.TrimSpace(sample.RequestID)
			if requestID == "" {
				return nil, fmt.Errorf("profiling sample at index %d in %s has empty request ID", index, path)
			}

			if previousPath, exists := requestSources[requestID]; exists {
				return nil, fmt.Errorf("duplicate request ID %q across profiling datasets %s and %s", requestID, previousPath, path)
			}

			requestSources[requestID] = path
			merged = append(merged, sample)
		}
	}

	return merged, nil
}

// DiscoverInvocationSampleDatasets recursively finds per-node raw profiling
// datasets below root.
//
// The expected file name is the basename of
// DefaultInvocationSampleExportPath, currently profiling-samples.jsonl.
//
// A central collection directory can therefore have the form:
//
//	data/profiling/raw/<run>/
//	├── arm-node-01/profiling-samples.jsonl
//	├── x86-node-01/profiling-samples.jsonl
//	└── x86-node-02/profiling-samples.jsonl
func DiscoverInvocationSampleDatasets(root string) ([]string, error) {

	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("profiling dataset root cannot be empty")
	}

	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect profiling dataset root %s: %w", root, err)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("profiling dataset root is not a directory: %s", root)
	}

	targetName := filepath.Base(DefaultInvocationSampleExportPath)

	paths := make([]string, 0)
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		if entry.IsDir() || entry.Name() != targetName {
			return nil
		}

		absPath, err := filepath.Abs(path)
		if err != nil {
			return err
		}

		paths = append(paths, filepath.Clean(absPath))

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to discover profiling datasets under %s: %w", root, err)
	}

	sort.Strings(paths)

	if len(paths) == 0 {
		return nil, fmt.Errorf("no %s datasets found under %s", targetName, root)
	}

	return paths, nil
}

// BuildFunctionProfilesByGroup groups eligible samples by:
//
//   - function name;
//   - machine tag;
//   - configured CPUs;
//   - configured memory.
//
// For each complete group it selects the most recent N eligible samples and
// builds one FunctionProfile.
//
// Groups with fewer than samplesPerProfile eligible samples are reported but
// skipped. This allows the raw JSONL dataset to contain partially profiled
// functions without preventing complete groups from being processed.
func BuildFunctionProfilesByGroup(samples []InvocationSample, samplesPerProfile int) (FunctionProfileBuildResult, error) {

	if samplesPerProfile < MinFunctionProfileSamples || samplesPerProfile > MaxFunctionProfileSamples {
		return FunctionProfileBuildResult{},
			fmt.Errorf(
				"samples per profile must be between %d and %d, got %d",
				MinFunctionProfileSamples,
				MaxFunctionProfileSamples,
				samplesPerProfile,
			)
	}

	result := FunctionProfileBuildResult{RawSampleCount: len(samples)}
	groups := make(map[functionProfileGroupKey][]timestampedInvocationSample)

	for index, sample := range samples {

		if !sample.Eligibility.ResourceClustering {
			result.IgnoredSamples++
			continue
		}

		// ResourceClustering=true promises that the sample is usable.
		// Validate this contract immediately, even if the corresponding
		// group later turns out to contain too few samples.
		if _, err := BuildProfilingFeatureVector(sample); err != nil {
			return FunctionProfileBuildResult{}, fmt.Errorf("failed to build feature vector from eligible sample at index %d: %w", index, err)
		}
		result.EligibleSamples++

		key :=
			functionProfileGroupKey{
				FunctionName:       sample.FunctionName,
				MachineTag:         sample.MachineTag,
				ConfiguredCPUs:     sample.FunctionConfiguration.ConfiguredCPUs,
				ConfiguredMemoryMB: sample.FunctionConfiguration.ConfiguredMemoryMB,
			}

		groups[key] = append(groups[key], timestampedInvocationSample{
			TimestampMs: sample.TimestampMs,
			InputOrder:  index,
			Sample:      sample,
		})
	}

	keys := make([]functionProfileGroupKey, 0, len(groups))

	for key := range groups {
		keys = append(keys, key)
	}

	// Deterministic output order.
	sort.Slice(keys, func(i int, j int) bool {
		if keys[i].FunctionName != keys[j].FunctionName {
			return keys[i].FunctionName < keys[j].FunctionName
		}

		if keys[i].MachineTag != keys[j].MachineTag {
			return keys[i].MachineTag < keys[j].MachineTag
		}

		if keys[i].ConfiguredCPUs != keys[j].ConfiguredCPUs {
			return keys[i].ConfiguredCPUs < keys[j].ConfiguredCPUs
		}

		return keys[i].ConfiguredMemoryMB < keys[j].ConfiguredMemoryMB
	})

	result.Groups = make([]FunctionProfileGroupStatus, 0, len(keys))
	result.Profiles = make([]FunctionProfile, 0, len(keys))

	for _, key := range keys {

		candidates := groups[key]
		// Order chronologically. Input order is used as deterministic
		// tie-breaker if two samples have the same millisecond timestamp.
		sort.SliceStable(candidates, func(i int, j int) bool {
			if candidates[i].TimestampMs != candidates[j].TimestampMs {
				return candidates[i].TimestampMs < candidates[j].TimestampMs
			}

			return candidates[i].InputOrder < candidates[j].InputOrder
		})

		status := FunctionProfileGroupStatus{
			FunctionName: key.FunctionName,
			MachineTag:   key.MachineTag,
			FunctionConfiguration: InvocationFunctionConfiguration{
				ConfiguredCPUs:     key.ConfiguredCPUs,
				ConfiguredMemoryMB: key.ConfiguredMemoryMB,
			},
			EligibleSampleCount: len(candidates),
		}

		if len(candidates) < samplesPerProfile {
			result.Groups = append(result.Groups, status)
			continue
		}

		// If the dataset contains more samples than required, use the most
		// recent ones. This makes repeated profiling runs deterministic while
		// preferring the latest measurements.
		selectedCandidates := candidates[len(candidates)-samplesPerProfile:]
		selectedSamples := make([]InvocationSample, 0, samplesPerProfile)

		for _, candidate := range selectedCandidates {
			selectedSamples = append(selectedSamples, candidate.Sample)
		}

		// This is the production caller that connects:
		//
		// raw InvocationSample
		//        ↓
		// BuildFunctionProfileFromSamples
		//        ↓
		// BuildProfilingFeatureVector
		//        ↓
		// AggregateProfilingFeatureVectors
		//        ↓
		// FunctionProfile
		profile, err := BuildFunctionProfileFromSamples(selectedSamples)
		if err != nil {
			return FunctionProfileBuildResult{}, fmt.Errorf("failed to aggregate function %q on machine tag %q: %w", key.FunctionName, key.MachineTag, err)
		}

		status.Built = true
		status.SelectedSampleCount = samplesPerProfile
		result.Groups = append(result.Groups, status)
		result.Profiles = append(result.Profiles, profile)
	}

	return result, nil
}

// ExportFunctionProfilesJSONL atomically rewrites the derived dataset with one
// complete FunctionProfile JSON object per line.
//
// Unlike profiling-samples.jsonl, this is a derived artifact. It is rewritten
// instead of appended to so running the aggregation repeatedly does not create
// duplicate profiles.
func ExportFunctionProfilesJSONL(path string, profiles []FunctionProfile) error {

	path = strings.TrimSpace(path)

	if path == "" {
		return fmt.Errorf("function profile output path cannot be empty")
	}

	tempFile, tempPath, err := createAtomicFile(path, ".function-profiles-*.tmp", "function profile dataset")
	if err != nil {
		return err
	}

	cleanupTemp := true
	defer func() {
		if cleanupTemp {
			_ = os.Remove(tempPath)
		}
	}()

	encoder := json.NewEncoder(tempFile)

	for index, profile := range profiles {

		if profile.SchemaVersion != FunctionProfileSchemaVersion {
			_ = tempFile.Close()
			return fmt.Errorf("function profile at index %d has unsupported schema version %d", index, profile.SchemaVersion)
		}

		if err := encoder.Encode(profile); err != nil {
			_ = tempFile.Close()
			return fmt.Errorf("failed to encode function profile at index %d: %w", index, err)
		}
	}

	if err := finalizeAtomicFile(tempFile, tempPath, path, "function profile dataset"); err != nil {
		return err
	}

	cleanupTemp = false
	return nil
}
