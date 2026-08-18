package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/serverledge-faas/serverledge/internal/profiling"
)

const probeSchemaVersion = 1

type probeOutput struct {
	SchemaVersion int `json:"schema_version"`

	MetricsURL string `json:"metrics_url"`

	ReadAt time.Time `json:"read_at"`

	ContainerID string `json:"container_id"`

	CPUJoulesByZone map[string]float64 `json:"cpu_joules_by_zone"`
}

func main() {
	var (
		metricsURL string

		containerID string

		timeout time.Duration
	)

	flag.StringVar(
		&metricsURL,
		"url",
		"http://127.0.0.1:28282/metrics",
		"Kepler Prometheus metrics endpoint",
	)

	flag.StringVar(
		&containerID,
		"container-id",
		"",
		"full container ID or unique prefix",
	)

	flag.DurationVar(
		&timeout,
		"timeout",
		3*time.Second,
		"maximum duration for the Kepler HTTP request",
	)

	flag.Parse()

	containerID =
		strings.TrimSpace(
			containerID,
		)

	if containerID == "" {
		fail(
			"-container-id is required",
		)
	}

	client, err :=
		profiling.NewKeplerClient(
			metricsURL,
			timeout,
		)

	if err != nil {
		fail(
			err.Error(),
		)
	}

	ctx, cancel :=
		context.WithTimeout(
			context.Background(),
			timeout,
		)

	defer cancel()

	snapshot, err :=
		client.ReadContainerSnapshot(
			ctx,
			containerID,
		)

	if err != nil {
		fail(
			err.Error(),
		)
	}

	output :=
		probeOutput{
			SchemaVersion: probeSchemaVersion,

			MetricsURL: metricsURL,

			ReadAt: snapshot.ReadAt,

			ContainerID: snapshot.ContainerID,

			CPUJoulesByZone: snapshot.CPUJoulesByZone,
		}

	encoder :=
		json.NewEncoder(
			os.Stdout,
		)

	encoder.SetIndent(
		"",
		"  ",
	)

	if err :=
		encoder.Encode(
			output,
		); err != nil {

		fail(
			fmt.Sprintf(
				"encode probe result: %v",
				err,
			),
		)
	}
}

func fail(
	message string,
) {
	fmt.Fprintf(
		os.Stderr,
		"kepler-probe: %s\n",
		message,
	)

	os.Exit(
		1,
	)
}
