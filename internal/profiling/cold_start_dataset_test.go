package profiling

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBuildColdStartProfilesByGroupSeparatesMachineTags(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"x86-request",
				"function-a",
				"x86",
				100,
				1000,
			),
			coldStartSampleForTest(
				"arm-request",
				"function-a",
				"arm64",
				200,
				2000,
			),
		}

	result, err :=
		BuildColdStartProfilesByGroup(
			samples,
			1,
		)

	if err != nil {
		t.Fatalf(
			"BuildColdStartProfilesByGroup returned error: %v",
			err,
		)
	}

	if result.RawSampleCount != 2 {
		t.Fatalf(
			"expected 2 raw samples, got %d",
			result.RawSampleCount,
		)
	}

	if result.EligibleSamples != 2 {
		t.Fatalf(
			"expected 2 eligible samples, got %d",
			result.EligibleSamples,
		)
	}

	if result.IgnoredSamples != 0 {
		t.Fatalf(
			"expected 0 ignored samples, got %d",
			result.IgnoredSamples,
		)
	}

	if len(result.Groups) != 2 {
		t.Fatalf(
			"expected 2 groups, got %d",
			len(result.Groups),
		)
	}

	if len(result.Profiles) != 2 {
		t.Fatalf(
			"expected 2 profiles, got %d",
			len(result.Profiles),
		)
	}

	// Groups are deterministically sorted by function and then MachineTag.
	if result.Profiles[0].MachineTag !=
		"arm64" {

		t.Fatalf(
			"expected first profile to be arm64, got %q",
			result.Profiles[0].MachineTag,
		)
	}

	if result.Profiles[1].MachineTag !=
		"x86" {

		t.Fatalf(
			"expected second profile to be x86, got %q",
			result.Profiles[1].MachineTag,
		)
	}
}

func TestBuildColdStartProfilesByGroupIgnoresFailedCold(
	t *testing.T,
) {
	successful :=
		coldStartSampleForTest(
			"successful-request",
			"function-a",
			"x86",
			100,
			1000,
		)

	failed :=
		BuildInvocationSample(
			InvocationSampleInput{
				Timestamp: time.UnixMilli(
					2000,
				),

				RequestID: "failed-request",

				FunctionName: "function-a",

				MachineTag: "x86",

				NodeName: "node-x86",

				ContainerID: "container-failed",

				ConfiguredCPUs: 1,

				ConfiguredMemoryMB: 128,

				WarmStart: false,

				ExecutionSucceeded: false,

				ExecutionError: "expected test failure",

				Timing: InvocationTiming{
					InitTimeMs: 200,
				},
			},
		)

	if failed.
		Eligibility.
		ColdStartAnalysis {

		t.Fatal(
			"test setup error: failed invocation unexpectedly eligible for cold-start analysis",
		)
	}

	result, err :=
		BuildColdStartProfilesByGroup(
			[]InvocationSample{
				successful,
				failed,
			},
			1,
		)

	if err != nil {
		t.Fatalf(
			"BuildColdStartProfilesByGroup returned error: %v",
			err,
		)
	}

	if result.RawSampleCount != 2 {
		t.Fatalf(
			"expected 2 raw samples, got %d",
			result.RawSampleCount,
		)
	}

	if result.EligibleSamples != 1 {
		t.Fatalf(
			"expected 1 eligible cold sample, got %d",
			result.EligibleSamples,
		)
	}

	if result.IgnoredSamples != 1 {
		t.Fatalf(
			"expected 1 ignored sample, got %d",
			result.IgnoredSamples,
		)
	}

	if len(result.Profiles) != 1 {
		t.Fatalf(
			"expected 1 profile, got %d",
			len(result.Profiles),
		)
	}

	if result.Profiles[0].
		SourceRequestIDs[0] !=
		"successful-request" {

		t.Fatalf(
			"unexpected selected request ID %q",
			result.Profiles[0].
				SourceRequestIDs[0],
		)
	}
}

func TestBuildColdStartProfilesByGroupSelectsMostRecentSamples(
	t *testing.T,
) {
	samples :=
		[]InvocationSample{
			coldStartSampleForTest(
				"oldest",
				"function-a",
				"x86",
				10,
				1000,
			),
			coldStartSampleForTest(
				"middle",
				"function-a",
				"x86",
				20,
				2000,
			),
			coldStartSampleForTest(
				"newest",
				"function-a",
				"x86",
				30,
				3000,
			),
		}

	result, err :=
		BuildColdStartProfilesByGroup(
			samples,
			2,
		)

	if err != nil {
		t.Fatalf(
			"BuildColdStartProfilesByGroup returned error: %v",
			err,
		)
	}

	if len(result.Profiles) != 1 {
		t.Fatalf(
			"expected 1 profile, got %d",
			len(result.Profiles),
		)
	}

	profile :=
		result.Profiles[0]

	expectedIDs :=
		[]string{
			"middle",
			"newest",
		}

	if !reflect.DeepEqual(
		expectedIDs,
		profile.SourceRequestIDs,
	) {
		t.Fatalf(
			"expected request IDs %v, got %v",
			expectedIDs,
			profile.SourceRequestIDs,
		)
	}

	if profile.SampleCount != 2 {
		t.Fatalf(
			"expected sample count 2, got %d",
			profile.SampleCount,
		)
	}

	// Most recent values are 20 and 30.
	assertColdFloatEqual(
		t,
		25,
		profile.InitTime.MeanMs,
	)

	assertColdFloatEqual(
		t,
		25,
		profile.InitTime.MedianMs,
	)
}

func TestBuildColdStartProfilesByGroupSkipsIncompleteGroup(
	t *testing.T,
) {
	result, err :=
		BuildColdStartProfilesByGroup(
			[]InvocationSample{
				coldStartSampleForTest(
					"request-1",
					"function-a",
					"x86",
					100,
					1000,
				),
			},
			2,
		)

	if err != nil {
		t.Fatalf(
			"BuildColdStartProfilesByGroup returned error: %v",
			err,
		)
	}

	if len(result.Profiles) != 0 {
		t.Fatalf(
			"expected no complete profiles, got %d",
			len(result.Profiles),
		)
	}

	if len(result.Groups) != 1 {
		t.Fatalf(
			"expected 1 group, got %d",
			len(result.Groups),
		)
	}

	group :=
		result.Groups[0]

	if group.Built {
		t.Fatal(
			"expected incomplete group not to be built",
		)
	}

	if group.EligibleSampleCount != 1 {
		t.Fatalf(
			"expected one eligible sample, got %d",
			group.EligibleSampleCount,
		)
	}

	if group.SelectedSampleCount != 0 {
		t.Fatalf(
			"expected zero selected samples, got %d",
			group.SelectedSampleCount,
		)
	}
}

func TestExportAndLoadColdStartProfilesJSONL(
	t *testing.T,
) {
	profile, err :=
		AggregateColdStartSamples(
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
			},
		)

	if err != nil {
		t.Fatalf(
			"AggregateColdStartSamples returned error: %v",
			err,
		)
	}

	path :=
		filepath.Join(
			t.TempDir(),
			"cold-start-profiles.jsonl",
		)

	if err :=
		ExportColdStartProfilesJSONL(
			path,
			[]ColdStartProfile{
				profile,
			},
		); err != nil {

		t.Fatalf(
			"ExportColdStartProfilesJSONL returned error: %v",
			err,
		)
	}

	loaded, err :=
		LoadColdStartProfilesJSONL(
			path,
		)

	if err != nil {
		t.Fatalf(
			"LoadColdStartProfilesJSONL returned error: %v",
			err,
		)
	}

	if len(loaded) != 1 {
		t.Fatalf(
			"expected one loaded profile, got %d",
			len(loaded),
		)
	}

	if !reflect.DeepEqual(
		profile,
		loaded[0],
	) {
		t.Fatalf(
			"round-trip mismatch:\nexpected: %#v\nactual:   %#v",
			profile,
			loaded[0],
		)
	}
}

func TestExportColdStartProfilesCSV(
	t *testing.T,
) {
	profile, err :=
		AggregateColdStartSamples(
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
			},
		)

	if err != nil {
		t.Fatalf(
			"AggregateColdStartSamples returned error: %v",
			err,
		)
	}

	path :=
		filepath.Join(
			t.TempDir(),
			"cold-start-profiles.csv",
		)

	if err :=
		ExportColdStartProfilesCSV(
			path,
			"experiment-001",
			[]ColdStartProfile{
				profile,
			},
		); err != nil {

		t.Fatalf(
			"ExportColdStartProfilesCSV returned error: %v",
			err,
		)
	}

	file, err :=
		os.Open(
			path,
		)

	if err != nil {
		t.Fatalf(
			"failed to open CSV: %v",
			err,
		)
	}

	defer file.Close()

	records, err :=
		csv.NewReader(
			file,
		).ReadAll()

	if err != nil {
		t.Fatalf(
			"failed to read CSV: %v",
			err,
		)
	}

	if len(records) != 2 {
		t.Fatalf(
			"expected header plus one record, got %d rows",
			len(records),
		)
	}

	if !reflect.DeepEqual(
		ColdStartProfileCSVHeader(),
		records[0],
	) {
		t.Fatalf(
			"unexpected CSV header:\nexpected: %v\nactual:   %v",
			ColdStartProfileCSVHeader(),
			records[0],
		)
	}

	row :=
		records[1]

	if row[0] != "1" {
		t.Fatalf(
			"unexpected CSV schema version %q",
			row[0],
		)
	}

	if row[1] != "experiment-001" {
		t.Fatalf(
			"unexpected experiment ID %q",
			row[1],
		)
	}

	if row[2] != "1" {
		t.Fatalf(
			"unexpected ColdStartProfile schema %q",
			row[2],
		)
	}

	if row[3] != "function-a" {
		t.Fatalf(
			"unexpected function name %q",
			row[3],
		)
	}

	if row[4] != "x86" {
		t.Fatalf(
			"unexpected machine tag %q",
			row[4],
		)
	}

	if row[5] != "1" {
		t.Fatalf(
			"unexpected configured CPUs %q",
			row[5],
		)
	}

	if row[6] != "128" {
		t.Fatalf(
			"unexpected configured memory %q",
			row[6],
		)
	}

	if row[7] != "2" {
		t.Fatalf(
			"unexpected sample count %q",
			row[7],
		)
	}

	if row[8] != "150" {
		t.Fatalf(
			"unexpected mean init time %q",
			row[8],
		)
	}

	if row[9] != "150" {
		t.Fatalf(
			"unexpected median init time %q",
			row[9],
		)
	}
}

func TestExportColdStartProfilesJSONLRejectsDuplicateIdentity(
	t *testing.T,
) {
	first, err :=
		AggregateColdStartSamples(
			[]InvocationSample{
				coldStartSampleForTest(
					"request-first",
					"function-a",
					"x86",
					100,
					1000,
				),
			},
		)

	if err != nil {
		t.Fatalf(
			"failed to build first profile: %v",
			err,
		)
	}

	second, err :=
		AggregateColdStartSamples(
			[]InvocationSample{
				coldStartSampleForTest(
					"request-second",
					"function-a",
					"x86",
					200,
					2000,
				),
			},
		)

	if err != nil {
		t.Fatalf(
			"failed to build second profile: %v",
			err,
		)
	}

	err =
		ExportColdStartProfilesJSONL(
			filepath.Join(
				t.TempDir(),
				"duplicate.jsonl",
			),
			[]ColdStartProfile{
				first,
				second,
			},
		)

	if err == nil {
		t.Fatal(
			"expected duplicate ColdStartProfile identity error",
		)
	}

	if !strings.Contains(
		err.Error(),
		"duplicate cold-start profile",
	) {
		t.Fatalf(
			"unexpected error: %v",
			err,
		)
	}
}
