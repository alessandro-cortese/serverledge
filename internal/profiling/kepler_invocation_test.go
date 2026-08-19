package profiling

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const keplerInvocationTestContainerID = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func TestKeplerWaitForContainerEnergyDeltaWaitsForRefresh(
	t *testing.T,
) {
	var reads atomic.Int64

	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					read :=
						reads.Add(
							1,
						)

					core :=
						1.0

					pkg :=
						2.0

					if read >= 2 {
						core =
							1.4

						pkg =
							2.7
					}

					_,
						_ =
						fmt.Fprint(
							w,
							keplerInvocationMetrics(
								keplerInvocationTestContainerID,
								core,
								pkg,
							),
						)
				},
			),
		)

	defer server.Close()

	client,
		err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	before :=
		KeplerContainerSnapshot{
			ReadAt: time.Now(),

			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"core":    1.0,
				"package": 2.0,
			},
		}

	ctx,
		cancel :=
		context.WithTimeout(
			context.Background(),
			time.Second,
		)

	defer cancel()

	profile,
		err :=
		client.WaitForContainerEnergyDelta(
			ctx,
			before,
			time.Millisecond,
		)

	require.NoError(
		t,
		err,
	)

	require.NotNil(
		t,
		profile,
	)

	assert.True(
		t,
		profile.Available,
	)

	assert.Equal(
		t,
		keplerInvocationTestContainerID,
		profile.ContainerID,
	)

	assert.GreaterOrEqual(
		t,
		profile.PollAttempts,
		2,
	)

	assert.InDelta(
		t,
		0.4,
		profile.CPUJoulesByZone["core"],
		1e-9,
	)

	assert.InDelta(
		t,
		0.7,
		profile.CPUJoulesByZone["package"],
		1e-9,
	)
}

func TestKeplerWaitForContainerEnergyDeltaTimesOutWhenCounterDoesNotAdvance(
	t *testing.T,
) {
	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					_,
						_ =
						fmt.Fprint(
							w,
							keplerInvocationMetrics(
								keplerInvocationTestContainerID,
								1.0,
								2.0,
							),
						)
				},
			),
		)

	defer server.Close()

	client,
		err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	before :=
		KeplerContainerSnapshot{
			ReadAt: time.Now(),

			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"core":    1.0,
				"package": 2.0,
			},
		}

	ctx,
		cancel :=
		context.WithTimeout(
			context.Background(),
			25*time.Millisecond,
		)

	defer cancel()

	_,
		err =
		client.WaitForContainerEnergyDelta(
			ctx,
			before,
			5*time.Millisecond,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"timed out waiting for Kepler refresh",
	)
}

func TestKeplerEnergyDeltaRejectsRegressingCounter(
	t *testing.T,
) {
	before :=
		KeplerContainerSnapshot{
			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"package": 5.0,
			},
		}

	after :=
		KeplerContainerSnapshot{
			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"package": 4.5,
			},
		}

	_,
		_,
		err :=
		keplerEnergyDelta(
			before,
			after,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"counter regressed",
	)
}

func TestKeplerEnergyDeltaRejectsChangedZoneSet(
	t *testing.T,
) {
	before :=
		KeplerContainerSnapshot{
			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"core":    1.0,
				"package": 2.0,
			},
		}

	after :=
		KeplerContainerSnapshot{
			ContainerID: keplerInvocationTestContainerID,

			CPUJoulesByZone: map[string]float64{
				"package": 2.5,
			},
		}

	_,
		_,
		err :=
		keplerEnergyDelta(
			before,
			after,
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"energy-zone set changed",
	)
}

func TestNewInvalidKeplerInvocationEnergyProfileDoesNotPretendZeroEnergy(
	t *testing.T,
) {
	profile :=
		NewInvalidKeplerInvocationEnergyProfile(
			"docker://"+
				keplerInvocationTestContainerID,
			"refresh timeout",
		)

	require.NotNil(
		t,
		profile,
	)

	assert.False(
		t,
		profile.Available,
	)

	assert.Equal(
		t,
		keplerInvocationTestContainerID,
		profile.ContainerID,
	)

	assert.Equal(
		t,
		"refresh timeout",
		profile.InvalidReason,
	)

	assert.Nil(
		t,
		profile.CPUJoulesByZone,
	)
}

func keplerInvocationMetrics(
	containerID string,
	core float64,
	pkg float64,
) string {
	return strings.TrimSpace(
		fmt.Sprintf(
			`
# TYPE kepler_container_cpu_joules_total counter
kepler_container_cpu_joules_total{container_id=%q,container_name="test",runtime="docker",state="running",zone="core",pod_id=""} %.9f
kepler_container_cpu_joules_total{container_id=%q,container_name="test",runtime="docker",state="running",zone="package",pod_id=""} %.9f
`,
			containerID,
			core,
			containerID,
			pkg,
		),
	) + "\n"
}
