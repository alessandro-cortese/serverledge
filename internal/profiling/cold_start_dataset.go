package profiling

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	DefaultColdStartProfileExportPath = "data/profiling/cold-start-profiles.jsonl"

	DefaultColdStartProfileCSVPath = "data/profiling/cold-start-profiles.csv"

	ColdStartProfileCSVSchemaVersion = 1
)

type ColdStartProfileGroupStatus struct {
	FunctionName string `json:"function_name"`

	MachineTag string `json:"machine_tag"`

	FunctionConfiguration InvocationFunctionConfiguration `json:"function_configuration"`

	EligibleSampleCount int `json:"eligible_sample_count"`

	SelectedSampleCount int `json:"selected_sample_count"`

	Built bool `json:"built"`
}

type ColdStartProfileBuildResult struct {
	Profiles []ColdStartProfile

	Groups []ColdStartProfileGroupStatus

	RawSampleCount int

	EligibleSamples int

	IgnoredSamples int
}

// BuildColdStartProfilesByGroup groups cold-start samples by function,
// machine tag and resource configuration.
//
// samplesPerProfile is intentionally explicit. The local smoke test may use
// one sample, while the final experimental protocol can require a larger
// number without changing the implementation.
func BuildColdStartProfilesByGroup(
	samples []InvocationSample,
	samplesPerProfile int,
) (ColdStartProfileBuildResult, error) {
	if samplesPerProfile <= 0 {
		return ColdStartProfileBuildResult{},
			fmt.Errorf(
				"samples per cold-start profile must be positive, got %d",
				samplesPerProfile,
			)
	}

	result := ColdStartProfileBuildResult{RawSampleCount: len(samples)}

	groups := make(map[functionProfileGroupKey][]timestampedInvocationSample)

	for index, sample := range samples {
		if !sample.Eligibility.ColdStartAnalysis {
			result.IgnoredSamples++
			continue
		}

		if err := validateColdStartSample(sample); err != nil {
			return ColdStartProfileBuildResult{},
				fmt.Errorf(
					"invalid eligible cold-start sample at index %d: %w",
					index,
					err,
				)
		}

		result.EligibleSamples++

		key := functionProfileGroupKey{
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

	result.Groups = make([]ColdStartProfileGroupStatus, 0, len(keys))

	result.Profiles = make([]ColdStartProfile, 0, len(keys))

	for _, key := range keys {
		candidates := groups[key]

		sort.SliceStable(candidates, func(i int, j int) bool {
			if candidates[i].TimestampMs != candidates[j].TimestampMs {
				return candidates[i].TimestampMs < candidates[j].TimestampMs
			}

			return candidates[i].InputOrder < candidates[j].InputOrder
		},
		)

		status := ColdStartProfileGroupStatus{
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

		// As for warm profiling, if more observations than requested are
		// available, select the most recent ones deterministically.
		selectedCandidates := candidates[len(candidates)-samplesPerProfile:]

		selectedSamples := make([]InvocationSample, 0, samplesPerProfile)

		for _, candidate := range selectedCandidates {
			selectedSamples = append(selectedSamples, candidate.Sample)
		}

		profile, err := AggregateColdStartSamples(selectedSamples)

		if err != nil {
			return ColdStartProfileBuildResult{}, fmt.Errorf(
				"failed to aggregate cold starts for function %q on machine tag %q: %w",
				key.FunctionName,
				key.MachineTag,
				err,
			)
		}

		status.Built = true
		status.SelectedSampleCount = samplesPerProfile
		result.Groups = append(result.Groups, status)
		result.Profiles = append(result.Profiles, profile)
	}

	return result, nil
}

func ExportColdStartProfilesJSONL(
	path string,
	profiles []ColdStartProfile,
) error {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return fmt.Errorf(
			"cold-start profile output path cannot be empty",
		)
	}

	if len(profiles) == 0 {
		return fmt.Errorf(
			"cannot export an empty cold-start profile dataset",
		)
	}

	for index, profile := range profiles {

		if err :=
			validateColdStartProfile(
				profile,
			); err != nil {

			return fmt.Errorf(
				"invalid cold-start profile at index %d: %w",
				index,
				err,
			)
		}
	}

	if err := validateUniqueColdStartProfiles(profiles); err != nil {
		return err
	}

	directory :=
		filepath.Dir(
			path,
		)

	if directory != "." {
		if err :=
			os.MkdirAll(
				directory,
				0o755,
			); err != nil {

			return fmt.Errorf(
				"failed to create cold-start profile export directory %s: %w",
				directory,
				err,
			)
		}
	}

	tempFile, err :=
		os.CreateTemp(
			directory,
			".cold-start-profiles-*.tmp",
		)

	if err != nil {
		return fmt.Errorf(
			"failed to create temporary cold-start profile dataset: %w",
			err,
		)
	}

	tempPath :=
		tempFile.Name()

	cleanupTemp := true

	defer func() {
		if cleanupTemp {
			_ =
				os.Remove(
					tempPath,
				)
		}
	}()

	encoder :=
		json.NewEncoder(
			tempFile,
		)

	for index, profile := range profiles {

		if err :=
			encoder.Encode(
				profile,
			); err != nil {

			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to encode cold-start profile at index %d: %w",
				index,
				err,
			)
		}
	}

	if err :=
		finalizeColdStartAtomicFile(
			tempFile,
			tempPath,
			path,
		); err != nil {

		return err
	}

	cleanupTemp = false

	return nil
}

func LoadColdStartProfilesJSONL(
	path string,
) ([]ColdStartProfile, error) {
	path =
		strings.TrimSpace(
			path,
		)

	if path == "" {
		return nil,
			fmt.Errorf(
				"cold-start profile input path cannot be empty",
			)
	}

	file, err :=
		os.Open(
			path,
		)

	if err != nil {
		return nil,
			fmt.Errorf(
				"failed to open cold-start profile dataset %s: %w",
				path,
				err,
			)
	}

	defer file.Close()

	scanner :=
		bufio.NewScanner(
			file,
		)

	scanner.Buffer(
		make(
			[]byte,
			64*1024,
		),
		4*1024*1024,
	)

	profiles :=
		make(
			[]ColdStartProfile,
			0,
		)

	lineNumber := 0

	for scanner.Scan() {
		lineNumber++

		line :=
			strings.TrimSpace(
				scanner.Text(),
			)

		if line == "" {
			continue
		}

		var profile ColdStartProfile

		if err :=
			json.Unmarshal(
				[]byte(line),
				&profile,
			); err != nil {

			return nil,
				fmt.Errorf(
					"invalid cold-start profile JSON at line %d: %w",
					lineNumber,
					err,
				)
		}

		if err :=
			validateColdStartProfile(
				profile,
			); err != nil {

			return nil,
				fmt.Errorf(
					"invalid cold-start profile at line %d: %w",
					lineNumber,
					err,
				)
		}

		profiles =
			append(
				profiles,
				profile,
			)
	}

	if err :=
		scanner.Err(); err != nil {

		return nil,
			fmt.Errorf(
				"failed to read cold-start profile dataset %s: %w",
				path,
				err,
			)
	}

	if len(profiles) == 0 {
		return nil,
			fmt.Errorf(
				"cold-start profile dataset %s contains no profiles",
				path,
			)
	}

	if err :=
		validateUniqueColdStartProfiles(
			profiles,
		); err != nil {

		return nil,
			err
	}

	return profiles, nil
}

func ColdStartProfileCSVHeader() []string {
	return []string{
		"csv_schema_version",
		"experiment_id",
		"cold_start_profile_schema_version",
		"function_name",
		"machine_tag",
		"configured_cpus",
		"configured_memory_mb",
		"sample_count",
		"init_time_mean_ms",
		"init_time_median_ms",
	}
}

func ExportColdStartProfilesCSV(
	path string,
	experimentID string,
	profiles []ColdStartProfile,
) error {
	path =
		strings.TrimSpace(
			path,
		)

	experimentID =
		strings.TrimSpace(
			experimentID,
		)

	if path == "" {
		return fmt.Errorf(
			"cold-start CSV output path cannot be empty",
		)
	}

	if experimentID == "" {
		return fmt.Errorf(
			"experiment ID cannot be empty",
		)
	}

	if len(profiles) == 0 {
		return fmt.Errorf(
			"cannot export an empty cold-start profile dataset",
		)
	}

	for index, profile := range profiles {

		if err :=
			validateColdStartProfile(
				profile,
			); err != nil {

			return fmt.Errorf(
				"invalid cold-start profile at index %d: %w",
				index,
				err,
			)
		}
	}

	if err :=
		validateUniqueColdStartProfiles(
			profiles,
		); err != nil {

		return err
	}

	ordered :=
		append(
			[]ColdStartProfile(nil),
			profiles...,
		)

	sort.SliceStable(
		ordered,
		func(
			i int,
			j int,
		) bool {

			if ordered[i].FunctionName !=
				ordered[j].FunctionName {

				return ordered[i].
					FunctionName <
					ordered[j].
						FunctionName
			}

			if ordered[i].MachineTag !=
				ordered[j].MachineTag {

				return ordered[i].
					MachineTag <
					ordered[j].
						MachineTag
			}

			if ordered[i].
				FunctionConfiguration.
				ConfiguredCPUs !=
				ordered[j].
					FunctionConfiguration.
					ConfiguredCPUs {

				return ordered[i].
					FunctionConfiguration.
					ConfiguredCPUs <
					ordered[j].
						FunctionConfiguration.
						ConfiguredCPUs
			}

			return ordered[i].
				FunctionConfiguration.
				ConfiguredMemoryMB <
				ordered[j].
					FunctionConfiguration.
					ConfiguredMemoryMB
		},
	)

	directory :=
		filepath.Dir(
			path,
		)

	if directory != "." {
		if err :=
			os.MkdirAll(
				directory,
				0o755,
			); err != nil {

			return fmt.Errorf(
				"failed to create cold-start CSV directory %s: %w",
				directory,
				err,
			)
		}
	}

	tempFile, err :=
		os.CreateTemp(
			directory,
			".cold-start-profiles-*.csv.tmp",
		)

	if err != nil {
		return fmt.Errorf(
			"failed to create temporary cold-start CSV: %w",
			err,
		)
	}

	tempPath :=
		tempFile.Name()

	cleanupTemp := true

	defer func() {
		if cleanupTemp {
			_ =
				os.Remove(
					tempPath,
				)
		}
	}()

	writer :=
		csv.NewWriter(
			tempFile,
		)

	if err :=
		writer.Write(
			ColdStartProfileCSVHeader(),
		); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to write cold-start CSV header: %w",
			err,
		)
	}

	for index, profile := range ordered {

		record :=
			[]string{
				strconv.Itoa(
					ColdStartProfileCSVSchemaVersion,
				),

				experimentID,

				strconv.Itoa(
					profile.SchemaVersion,
				),

				profile.FunctionName,

				profile.MachineTag,

				strconv.FormatFloat(
					profile.
						FunctionConfiguration.
						ConfiguredCPUs,
					'g',
					-1,
					64,
				),

				strconv.FormatInt(
					profile.
						FunctionConfiguration.
						ConfiguredMemoryMB,
					10,
				),

				strconv.Itoa(
					profile.SampleCount,
				),

				strconv.FormatFloat(
					profile.InitTime.MeanMs,
					'g',
					-1,
					64,
				),

				strconv.FormatFloat(
					profile.InitTime.MedianMs,
					'g',
					-1,
					64,
				),
			}

		if err :=
			writer.Write(
				record,
			); err != nil {

			_ =
				tempFile.Close()

			return fmt.Errorf(
				"failed to write cold-start CSV row %d: %w",
				index,
				err,
			)
		}
	}

	writer.Flush()

	if err :=
		writer.Error(); err != nil {

		_ =
			tempFile.Close()

		return fmt.Errorf(
			"failed to flush cold-start CSV: %w",
			err,
		)
	}

	if err :=
		finalizeColdStartAtomicFile(
			tempFile,
			tempPath,
			path,
		); err != nil {

		return err
	}

	cleanupTemp = false

	return nil
}

func validateColdStartProfile(
	profile ColdStartProfile,
) error {
	if profile.SchemaVersion !=
		ColdStartProfileSchemaVersion {

		return fmt.Errorf(
			"unsupported cold-start profile schema version %d",
			profile.SchemaVersion,
		)
	}

	if strings.TrimSpace(
		profile.FunctionName,
	) == "" {

		return fmt.Errorf(
			"function name is empty",
		)
	}

	if strings.TrimSpace(
		profile.MachineTag,
	) == "" {

		return fmt.Errorf(
			"machine tag is empty",
		)
	}

	if !finitePositive(
		profile.
			FunctionConfiguration.
			ConfiguredCPUs,
	) {
		return fmt.Errorf(
			"configured CPUs are invalid: %v",
			profile.
				FunctionConfiguration.
				ConfiguredCPUs,
		)
	}

	if profile.
		FunctionConfiguration.
		ConfiguredMemoryMB <= 0 {

		return fmt.Errorf(
			"configured memory is invalid: %d MB",
			profile.
				FunctionConfiguration.
				ConfiguredMemoryMB,
		)
	}

	if profile.SampleCount <= 0 {
		return fmt.Errorf(
			"sample count must be positive, got %d",
			profile.SampleCount,
		)
	}

	if len(
		profile.SourceRequestIDs,
	) != profile.SampleCount {

		return fmt.Errorf(
			"source request ID count %d does not match sample count %d",
			len(
				profile.SourceRequestIDs,
			),
			profile.SampleCount,
		)
	}

	seen :=
		make(
			map[string]struct{},
			len(
				profile.SourceRequestIDs,
			),
		)

	for _, requestID := range profile.SourceRequestIDs {

		requestID =
			strings.TrimSpace(
				requestID,
			)

		if requestID == "" {
			return fmt.Errorf(
				"source request ID is empty",
			)
		}

		if _, exists :=
			seen[requestID]; exists {

			return fmt.Errorf(
				"duplicate source request ID %q",
				requestID,
			)
		}

		seen[requestID] =
			struct{}{}
	}

	if !finiteNonNegative(
		profile.InitTime.MeanMs,
	) {
		return fmt.Errorf(
			"mean init time is invalid: %v",
			profile.InitTime.MeanMs,
		)
	}

	if !finiteNonNegative(
		profile.InitTime.MedianMs,
	) {
		return fmt.Errorf(
			"median init time is invalid: %v",
			profile.InitTime.MedianMs,
		)
	}

	return nil
}

type coldStartProfileIdentity struct {
	FunctionName string

	MachineTag string

	ConfiguredCPUs float64

	ConfiguredMemoryMB int64
}

func validateUniqueColdStartProfiles(
	profiles []ColdStartProfile,
) error {
	seen := make(map[coldStartProfileIdentity]struct{}, len(profiles))

	for _, profile := range profiles {
		identity := coldStartProfileIdentity{
			FunctionName:       profile.FunctionName,
			MachineTag:         profile.MachineTag,
			ConfiguredCPUs:     profile.FunctionConfiguration.ConfiguredCPUs,
			ConfiguredMemoryMB: profile.FunctionConfiguration.ConfiguredMemoryMB,
		}

		if _, exists := seen[identity]; exists {
			return fmt.Errorf(
				"duplicate cold-start profile for function %q, machine tag %q, CPUs %g, memory %d MB",
				identity.FunctionName,
				identity.MachineTag,
				identity.ConfiguredCPUs,
				identity.ConfiguredMemoryMB,
			)
		}

		seen[identity] = struct{}{}
	}

	return nil
}

func finalizeColdStartAtomicFile(
	file *os.File,
	tempPath string,
	path string,
) error {
	if err :=
		file.Sync(); err != nil {

		_ =
			file.Close()

		return fmt.Errorf(
			"failed to sync temporary file: %w",
			err,
		)
	}

	if err :=
		file.Chmod(
			0o644,
		); err != nil {

		_ =
			file.Close()

		return fmt.Errorf(
			"failed to set output permissions: %w",
			err,
		)
	}

	if err :=
		file.Close(); err != nil {

		return fmt.Errorf(
			"failed to close temporary file: %w",
			err,
		)
	}

	if err :=
		os.Rename(
			tempPath,
			path,
		); err != nil {

		return fmt.Errorf(
			"failed to replace output %s: %w",
			path,
			err,
		)
	}

	return nil
}
