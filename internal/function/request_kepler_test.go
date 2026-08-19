package function

import (
	"encoding/json"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/profiling"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionReportSerializesKeplerEnergy(
	t *testing.T,
) {
	report :=
		ExecutionReport{
			Duration: 10.0,

			KeplerEnergy: &profiling.KeplerInvocationEnergyProfile{
				SchemaVersion: profiling.
					KeplerInvocationEnergyProfileSchemaVersion,

				Available: true,

				ContainerID: "container-a",

				CPUJoulesByZone: map[string]float64{
					"core":    0.4,
					"package": 0.7,
				},
			},
		}

	encoded,
		err :=
		json.Marshal(
			report,
		)

	require.NoError(
		t,
		err,
	)

	payload :=
		string(
			encoded,
		)

	assert.Contains(
		t,
		payload,
		`"KeplerEnergy"`,
	)

	assert.Contains(
		t,
		payload,
		`"available":true`,
	)

	assert.Contains(
		t,
		payload,
		`"core":0.4`,
	)

	assert.Contains(
		t,
		payload,
		`"package":0.7`,
	)
}

func TestExecutionReportOmitsKeplerEnergyWhenUnavailableByConfiguration(
	t *testing.T,
) {
	report :=
		ExecutionReport{
			Duration: 10.0,
		}

	encoded,
		err :=
		json.Marshal(
			report,
		)

	require.NoError(
		t,
		err,
	)

	assert.NotContains(
		t,
		string(
			encoded,
		),
		`"KeplerEnergy"`,
	)
}
