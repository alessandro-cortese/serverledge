package profiling

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
)

const (
	KeplerContainerCPUJoulesMetric = "kepler_container_cpu_joules_total"
)

// KeplerContainerSnapshot is a point-in-time view of the cumulative
// container-scoped counters exposed by Kepler.
//
// CPUJoulesByZone deliberately preserves Kepler's zone dimension rather than
// immediately summing it. Different power meters / architectures may expose
// different hardware zones and summing them blindly could make later analysis
// ambiguous.
type KeplerContainerSnapshot struct {
	ReadAt time.Time

	ContainerID string

	CPUJoulesByZone map[string]float64
}

// KeplerClient reads the Prometheus exposition endpoint exported by one
// node-local Kepler instance.
//
// A Prometheus server is not required: Serverledge reads Kepler's exporter
// endpoint directly.
type KeplerClient struct {
	metricsURL string

	httpClient *http.Client
}

// NewKeplerClient creates a reader for one Kepler exporter.
func NewKeplerClient(
	metricsURL string,
	timeout time.Duration,
) (
	*KeplerClient,
	error,
) {
	metricsURL =
		strings.TrimSpace(
			metricsURL,
		)

	if metricsURL == "" {
		return nil,
			fmt.Errorf(
				"Kepler metrics URL cannot be empty",
			)
	}

	parsedURL, err :=
		url.Parse(
			metricsURL,
		)

	if err != nil {
		return nil,
			fmt.Errorf(
				"invalid Kepler metrics URL %q: %w",
				metricsURL,
				err,
			)
	}

	if parsedURL.Scheme != "http" &&
		parsedURL.Scheme != "https" {

		return nil,
			fmt.Errorf(
				"Kepler metrics URL must use http or https, got %q",
				parsedURL.Scheme,
			)
	}

	if parsedURL.Host == "" {
		return nil,
			fmt.Errorf(
				"Kepler metrics URL must contain a host",
			)
	}

	if timeout <= 0 {
		return nil,
			fmt.Errorf(
				"Kepler HTTP timeout must be positive",
			)
	}

	return &KeplerClient{
		metricsURL: metricsURL,

		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// ReadContainerSnapshot reads the current cumulative Kepler counters for one
// container.
//
// The caller may provide either the full runtime container ID or a sufficiently
// long unique prefix. Runtime prefixes such as docker:// are normalized.
func (c *KeplerClient) ReadContainerSnapshot(
	ctx context.Context,
	containerID string,
) (
	KeplerContainerSnapshot,
	error,
) {
	if c == nil ||
		c.httpClient == nil {

		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"Kepler client is not initialized",
			)
	}

	containerID =
		normalizeKeplerContainerID(
			containerID,
		)

	if containerID == "" {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"container ID cannot be empty",
			)
	}

	req, err :=
		http.NewRequestWithContext(
			ctx,
			http.MethodGet,
			c.metricsURL,
			nil,
		)

	if err != nil {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"build Kepler metrics request: %w",
				err,
			)
	}

	resp, err :=
		c.httpClient.Do(
			req,
		)

	if err != nil {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"read Kepler metrics: %w",
				err,
			)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"Kepler metrics endpoint returned HTTP %d",
				resp.StatusCode,
			)
	}

	var parser expfmt.TextParser

	families, err :=
		parser.TextToMetricFamilies(
			resp.Body,
		)

	if err != nil {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"parse Kepler metrics: %w",
				err,
			)
	}

	joulesFamily, ok :=
		families[KeplerContainerCPUJoulesMetric]

	if !ok {
		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"Kepler metric %s is not exposed",
				KeplerContainerCPUJoulesMetric,
			)
	}

	if joulesFamily.GetType() !=
		dto.MetricType_COUNTER {

		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"Kepler metric %s is not a counter",
				KeplerContainerCPUJoulesMetric,
			)
	}

	matchedContainerID :=
		""

	joulesByZone :=
		make(
			map[string]float64,
		)

	for _, metric := range joulesFamily.Metric {

		labels :=
			prometheusMetricLabels(
				metric.Label,
			)

		candidateID :=
			normalizeKeplerContainerID(
				labels["container_id"],
			)

		if !keplerContainerIDMatches(
			containerID,
			candidateID,
		) {
			continue
		}

		if matchedContainerID == "" {

			matchedContainerID =
				candidateID

		} else if matchedContainerID !=
			candidateID {

			return KeplerContainerSnapshot{},
				fmt.Errorf(
					"container ID prefix %q matches multiple Kepler containers",
					containerID,
				)
		}

		zone :=
			strings.TrimSpace(
				labels["zone"],
			)

		if zone == "" {
			zone =
				"default"
		}

		if _,
			exists :=
			joulesByZone[zone]; exists {

			return KeplerContainerSnapshot{},
				fmt.Errorf(
					"Kepler exposes multiple %s series for container %q and zone %q",
					KeplerContainerCPUJoulesMetric,
					candidateID,
					zone,
				)
		}

		counter :=
			metric.GetCounter()

		if counter == nil {
			return KeplerContainerSnapshot{},
				fmt.Errorf(
					"Kepler metric %s is not a counter",
					KeplerContainerCPUJoulesMetric,
				)
		}

		value :=
			counter.GetValue()

		if !finiteKeplerCounter(
			value,
		) {
			return KeplerContainerSnapshot{},
				fmt.Errorf(
					"Kepler metric %s contains invalid value %v for container %q zone %q",
					KeplerContainerCPUJoulesMetric,
					value,
					candidateID,
					zone,
				)
		}

		joulesByZone[zone] = value
	}

	if matchedContainerID == "" ||
		len(
			joulesByZone,
		) == 0 {

		return KeplerContainerSnapshot{},
			fmt.Errorf(
				"container %q is not present in Kepler metric %s",
				containerID,
				KeplerContainerCPUJoulesMetric,
			)
	}

	return KeplerContainerSnapshot{
		ReadAt: time.Now(),

		ContainerID: matchedContainerID,

		CPUJoulesByZone: joulesByZone,
	}, nil
}

func prometheusMetricLabels(
	labels []*dto.LabelPair,
) map[string]string {

	result :=
		make(
			map[string]string,
			len(
				labels,
			),
		)

	for _, label := range labels {

		if label == nil {
			continue
		}

		result[label.GetName()] = label.GetValue()
	}

	return result
}

func normalizeKeplerContainerID(
	value string,
) string {

	value =
		strings.TrimSpace(
			value,
		)

	for _, prefix := range []string{
		"docker://",
		"containerd://",
		"cri-o://",
	} {

		value =
			strings.TrimPrefix(
				value,
				prefix,
			)
	}

	return value
}

func keplerContainerIDMatches(
	requested string,
	candidate string,
) bool {

	requested =
		normalizeKeplerContainerID(
			requested,
		)

	candidate =
		normalizeKeplerContainerID(
			candidate,
		)

	if requested == "" ||
		candidate == "" {

		return false
	}

	if requested ==
		candidate {

		return true
	}

	const minimumSafePrefixLength = 12

	if len(
		requested,
	) >= minimumSafePrefixLength &&
		strings.HasPrefix(
			candidate,
			requested,
		) {

		return true
	}

	if len(
		candidate,
	) >= minimumSafePrefixLength &&
		strings.HasPrefix(
			requested,
			candidate,
		) {

		return true
	}

	return false
}

func finiteKeplerCounter(
	value float64,
) bool {

	return value >= 0 &&
		!math.IsNaN(
			value,
		) &&
		!math.IsInf(
			value,
			0,
		)
}
