package profiling

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/serverledge-faas/serverledge/internal/config"
)

func TestExportInvocationSampleDisabledDoesNotCreateFile(
	t *testing.T,
) {
	viper.Reset()
	t.Cleanup(
		viper.Reset,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"samples.jsonl",
		)

	viper.Set(
		config.FUNCTION_PROFILING_ENABLED,
		true,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_ENABLED,
		false,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_PATH,
		path,
	)

	require.NoError(
		t,
		ExportInvocationSample(
			InvocationSample{},
		),
	)

	_, err :=
		os.Stat(
			path,
		)

	assert.ErrorIs(
		t,
		err,
		os.ErrNotExist,
	)
}

func TestExportInvocationSampleCreatesVersionedJSONLine(
	t *testing.T,
) {
	viper.Reset()

	t.Cleanup(
		viper.Reset,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"nested",
			"samples.jsonl",
		)

	viper.Set(
		config.FUNCTION_PROFILING_ENABLED,
		true,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_ENABLED,
		true,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_PATH,
		path,
	)

	sample :=
		BuildInvocationSample(
			InvocationSampleInput{
				RequestID: "request-a",

				FunctionName: "hello",

				WarmStart: false,

				ExecutionSucceeded: true,

				Timing: InvocationTiming{
					DurationMs: 10,

					ResponseTimeMs: 120,

					InitTimeMs: 100,
				},
			},
		)

	require.NoError(
		t,
		ExportInvocationSample(
			sample,
		),
	)

	content, err :=
		os.ReadFile(
			path,
		)

	require.NoError(
		t,
		err,
	)

	lines :=
		splitNonEmptyLines(
			string(
				content,
			),
		)

	require.Len(
		t,
		lines,
		1,
	)

	var decoded InvocationSample

	require.NoError(
		t,
		json.Unmarshal(
			[]byte(
				lines[0],
			),
			&decoded,
		),
	)

	assert.Equal(
		t,
		InvocationSampleSchemaVersion,
		decoded.SchemaVersion,
	)

	assert.Equal(
		t,
		"request-a",
		decoded.RequestID,
	)

	assert.InDelta(
		t,
		100.0,
		decoded.Timing.InitTimeMs,
		1e-9,
	)

	assert.True(
		t,
		decoded.Eligibility.
			ColdStartAnalysis,
	)
}

func TestExportInvocationSampleConcurrentWritesRemainValidJSONLines(
	t *testing.T,
) {
	viper.Reset()

	t.Cleanup(
		viper.Reset,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"samples.jsonl",
		)

	viper.Set(
		config.FUNCTION_PROFILING_ENABLED,
		true,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_ENABLED,
		true,
	)

	viper.Set(
		config.FUNCTION_PROFILING_EXPORT_PATH,
		path,
	)

	const samples = 64

	var waitGroup sync.WaitGroup

	errors :=
		make(
			chan error,
			samples,
		)

	for index := 0; index < samples; index++ {

		waitGroup.Add(
			1,
		)

		go func(
			index int,
		) {
			defer waitGroup.Done()

			errors <- ExportInvocationSample(
				BuildInvocationSample(
					InvocationSampleInput{
						RequestID: fmt.Sprintf(
							"request-%d",
							index,
						),

						WarmStart: true,

						ExecutionSucceeded: true,

						Timing: InvocationTiming{
							DurationMs: 1,

							ResponseTimeMs: 2,
						},

						Profile: &InvocationResourceProfile{
							Enabled: true,

							Collected: true,

							Valid: true,

							ExclusiveContainer: true,
						},
					},
				),
			)
		}(
			index,
		)
	}

	waitGroup.Wait()

	close(
		errors,
	)

	for err := range errors {

		require.NoError(
			t,
			err,
		)
	}

	file, err :=
		os.Open(
			path,
		)

	require.NoError(
		t,
		err,
	)

	defer file.Close()

	seen :=
		make(
			map[string]struct{},
			samples,
		)

	scanner :=
		bufio.NewScanner(
			file,
		)

	for scanner.Scan() {
		var sample InvocationSample

		require.NoError(
			t,
			json.Unmarshal(
				scanner.Bytes(),
				&sample,
			),
		)

		seen[sample.RequestID] = struct{}{}
	}

	require.NoError(
		t,
		scanner.Err(),
	)

	assert.Len(
		t,
		seen,
		samples,
	)
}

func splitNonEmptyLines(
	value string,
) []string {
	result :=
		make(
			[]string,
			0,
		)

	scanner :=
		bufio.NewScanner(
			strings.NewReader(
				value,
			),
		)

	for scanner.Scan() {
		if scanner.Text() != "" {
			result =
				append(
					result,
					scanner.Text(),
				)
		}
	}

	return result
}
