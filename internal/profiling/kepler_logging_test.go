package profiling

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLogKeplerInvocationEnergyProfileLogsAvailableMeasurement(
	t *testing.T,
) {
	var output bytes.Buffer

	previousWriter :=
		log.Writer()

	previousFlags :=
		log.Flags()

	previousPrefix :=
		log.Prefix()

	log.SetOutput(
		&output,
	)

	log.SetFlags(
		0,
	)

	log.SetPrefix(
		"",
	)

	t.Cleanup(
		func() {
			log.SetOutput(
				previousWriter,
			)

			log.SetFlags(
				previousFlags,
			)

			log.SetPrefix(
				previousPrefix,
			)
		},
	)

	LogKeplerInvocationEnergyProfile(
		&KeplerInvocationEnergyProfile{
			SchemaVersion: KeplerInvocationEnergyProfileSchemaVersion,

			Available: true,

			ContainerID: "docker://aaaaaaaaaaaaaaaa",

			PollAttempts: 3,

			RefreshWaitMs: 812.5,

			CPUJoulesByZone: map[string]float64{
				"package": 0.7,
				"core":    0.4,
			},
		},
	)

	logged :=
		output.String()

	require.NotEmpty(
		t,
		logged,
	)

	assert.Contains(
		t,
		logged,
		"event=kepler_invocation_energy",
	)

	assert.Contains(
		t,
		logged,
		"available=true",
	)

	assert.Contains(
		t,
		logged,
		"poll_attempts=3",
	)

	assert.Contains(
		t,
		logged,
		`cpu_joules_by_zone="core=0.400000000,package=0.700000000"`,
	)

	assert.NotContains(
		t,
		logged,
		"total_joules",
	)
}

func TestLogKeplerInvocationEnergyProfileLogsUnavailableMeasurement(
	t *testing.T,
) {
	var output bytes.Buffer

	previousWriter :=
		log.Writer()

	previousFlags :=
		log.Flags()

	previousPrefix :=
		log.Prefix()

	log.SetOutput(
		&output,
	)

	log.SetFlags(
		0,
	)

	log.SetPrefix(
		"",
	)

	t.Cleanup(
		func() {
			log.SetOutput(
				previousWriter,
			)

			log.SetFlags(
				previousFlags,
			)

			log.SetPrefix(
				previousPrefix,
			)
		},
	)

	LogKeplerInvocationEnergyProfile(
		NewInvalidKeplerInvocationEnergyProfile(
			"aaaaaaaaaaaaaaaa",
			"refresh timeout",
		),
	)

	logged :=
		output.String()

	assert.True(
		t,
		strings.Contains(
			logged,
			"available=false",
		),
	)

	assert.Contains(
		t,
		logged,
		`reason="refresh timeout"`,
	)
}

func TestLogKeplerInvocationEnergyProfileIgnoresNil(
	t *testing.T,
) {
	var output bytes.Buffer

	previousWriter :=
		log.Writer()

	log.SetOutput(
		&output,
	)

	t.Cleanup(
		func() {
			log.SetOutput(
				previousWriter,
			)
		},
	)

	LogKeplerInvocationEnergyProfile(
		nil,
	)

	assert.Empty(
		t,
		output.String(),
	)
}
