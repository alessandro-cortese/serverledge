package profiling

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKeplerClientReadsContainerSnapshot(
	t *testing.T,
) {
	metrics := `
# TYPE kepler_container_cpu_joules_total counter
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",container_name="target",runtime="docker",state="running",zone="package",pod_id=""} 12.5
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",container_name="target",runtime="docker",state="running",zone="dram",pod_id=""} 2.25
kepler_container_cpu_joules_total{container_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",container_name="other",runtime="docker",state="running",zone="package",pod_id=""} 99
`

	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					w.Header().Set(
						"Content-Type",
						"text/plain; version=0.0.4",
					)

					_,
						_ =
						w.Write(
							[]byte(
								metrics,
							),
						)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	snapshot, err :=
		client.ReadContainerSnapshot(
			context.Background(),
			"aaaaaaaaaaaa",
		)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		strings.Repeat(
			"a",
			64,
		),
		snapshot.ContainerID,
	)

	assert.Equal(
		t,
		12.5,
		snapshot.CPUJoulesByZone["package"],
	)

	assert.Equal(
		t,
		2.25,
		snapshot.CPUJoulesByZone["dram"],
	)

	assert.False(
		t,
		snapshot.ReadAt.IsZero(),
	)
}

func TestKeplerClientAcceptsRuntimePrefixedContainerID(
	t *testing.T,
) {
	metrics := `
# TYPE kepler_container_cpu_joules_total counter
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",zone="package"} 3
`

	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					_,
						_ =
						w.Write(
							[]byte(
								metrics,
							),
						)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	_,
		err =
		client.ReadContainerSnapshot(
			context.Background(),
			"docker://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		)

	require.NoError(
		t,
		err,
	)
}

func TestKeplerClientRejectsMissingContainer(
	t *testing.T,
) {
	metrics := `
# TYPE kepler_container_cpu_joules_total counter
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",zone="package"} 3
`

	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					_,
						_ =
						w.Write(
							[]byte(
								metrics,
							),
						)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	_,
		err =
		client.ReadContainerSnapshot(
			context.Background(),
			strings.Repeat(
				"b",
				64,
			),
		)

	require.ErrorContains(
		t,
		err,
		"is not present",
	)
}

func TestKeplerClientRejectsMissingMetric(
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
						w.Write(
							[]byte(
								"# TYPE unrelated counter\nunrelated 1\n",
							),
						)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	_,
		err =
		client.ReadContainerSnapshot(
			context.Background(),
			strings.Repeat(
				"a",
				64,
			),
		)

	require.ErrorContains(
		t,
		err,
		KeplerContainerCPUJoulesMetric,
	)
}

func TestKeplerClientRejectsHTTPFailure(
	t *testing.T,
) {
	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					http.Error(
						w,
						"unavailable",
						http.StatusServiceUnavailable,
					)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	_,
		err =
		client.ReadContainerSnapshot(
			context.Background(),
			strings.Repeat(
				"a",
				64,
			),
		)

	require.ErrorContains(
		t,
		err,
		"HTTP 503",
	)
}

func TestKeplerClientRejectsAmbiguousContainerPrefix(
	t *testing.T,
) {
	metrics := `
# TYPE kepler_container_cpu_joules_total counter
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaa1111111111111111111111111111111111111111111111111111",zone="package"} 3
kepler_container_cpu_joules_total{container_id="aaaaaaaaaaaa2222222222222222222222222222222222222222222222222222",zone="package"} 4
`

	server :=
		httptest.NewServer(
			http.HandlerFunc(
				func(
					w http.ResponseWriter,
					_ *http.Request,
				) {
					_,
						_ =
						w.Write(
							[]byte(
								metrics,
							),
						)
				},
			),
		)

	defer server.Close()

	client, err :=
		NewKeplerClient(
			server.URL,
			time.Second,
		)

	require.NoError(
		t,
		err,
	)

	_,
		err =
		client.ReadContainerSnapshot(
			context.Background(),
			"aaaaaaaaaaaa",
		)

	require.ErrorContains(
		t,
		err,
		"matches multiple",
	)
}
