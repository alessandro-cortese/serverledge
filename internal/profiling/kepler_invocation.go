package profiling

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const KeplerInvocationEnergyProfileSchemaVersion = 1

// KeplerInvocationEnergyProfile contains the energy attributed by Kepler to one
// profiling window of a container.
//
// The values remain separated by Kepler energy zone. No attempt is made here to
// sum package/core or other hardware domains because their relationship may
// differ across machines and architectures.
type KeplerInvocationEnergyProfile struct {
	SchemaVersion   int                `json:"schema_version"`
	Available       bool               `json:"available"`
	InvalidReason   string             `json:"invalid_reason,omitempty"`
	ContainerID     string             `json:"container_id"`
	BeforeReadAt    time.Time          `json:"before_read_at,omitempty"`
	AfterReadAt     time.Time          `json:"after_read_at,omitempty"`
	RefreshWaitMs   float64            `json:"refresh_wait_ms"`
	PollAttempts    int                `json:"poll_attempts"`
	CPUJoulesByZone map[string]float64 `json:"cpu_joules_by_zone,omitempty"`
}

// NewInvalidKeplerInvocationEnergyProfile records why an invocation-level
// Kepler measurement could not be collected.
//
// A missing measurement is deliberately represented as unavailable rather than
// as zero Joules: zero energy and unavailable energy have different semantics.
func NewInvalidKeplerInvocationEnergyProfile(containerID string, reason string) *KeplerInvocationEnergyProfile {

	return &KeplerInvocationEnergyProfile{
		SchemaVersion: KeplerInvocationEnergyProfileSchemaVersion,
		Available:     false,
		InvalidReason: strings.TrimSpace(reason),
		ContainerID:   normalizeKeplerContainerID(containerID),
	}
}

// WaitForContainerEnergyDelta waits until Kepler exposes a snapshot newer than
// the supplied pre-invocation snapshot.
//
// "Newer" means that:
//   - the same container is present;
//   - the same energy zones are present;
//   - counters do not regress;
//   - at least one energy-zone counter has advanced.
//
// The caller controls the overall refresh deadline through ctx. The caller is
// expected to hold the container profiling lock for the whole operation so that
// another invocation cannot contaminate the measurement.
func (c *KeplerClient) WaitForContainerEnergyDelta(ctx context.Context, before KeplerContainerSnapshot, pollInterval time.Duration) (*KeplerInvocationEnergyProfile, error) {

	if c == nil || c.httpClient == nil {
		return nil, fmt.Errorf("Kepler client is not initialized")
	}

	if pollInterval <= 0 {
		return nil, fmt.Errorf("Kepler poll interval must be positive")
	}

	containerID := normalizeKeplerContainerID(before.ContainerID)
	if containerID == "" {
		return nil, fmt.Errorf("Kepler before snapshot has an empty container ID")
	}

	if len(before.CPUJoulesByZone) == 0 {
		return nil, fmt.Errorf("Kepler before snapshot for container %q has no energy zones", containerID)
	}

	for zone, value := range before.CPUJoulesByZone {

		if strings.TrimSpace(zone) == "" {
			return nil, fmt.Errorf("Kepler before snapshot for container %q contains an empty energy zone", containerID)
		}

		if !finiteKeplerCounter(value) {
			return nil, fmt.Errorf("Kepler before snapshot for container %q contains invalid counter %v for zone %q", containerID, value, zone)
		}
	}

	refreshStartedAt := time.Now()
	attempts := 0
	var lastReadErr error

	for {
		if err := ctx.Err(); err != nil {
			if lastReadErr != nil {
				return nil, fmt.Errorf("timed out waiting for Kepler refresh for container %q after %d attempts: last read error: %v: %w", containerID, attempts, lastReadErr, err)
			}

			return nil, fmt.Errorf("timed out waiting for Kepler refresh for container %q after %d attempts: %w", containerID, attempts, err)
		}

		attempts++

		after, err := c.ReadContainerSnapshot(ctx, containerID)

		if err != nil {
			lastReadErr = err
		} else {
			lastReadErr = nil
			deltas, advanced, err := keplerEnergyDelta(before, after)

			if err != nil {
				return nil, err
			}

			if advanced {
				return &KeplerInvocationEnergyProfile{
					SchemaVersion:   KeplerInvocationEnergyProfileSchemaVersion,
					Available:       true,
					ContainerID:     after.ContainerID,
					BeforeReadAt:    before.ReadAt,
					AfterReadAt:     after.ReadAt,
					RefreshWaitMs:   float64(time.Since(refreshStartedAt)) / float64(time.Millisecond),
					PollAttempts:    attempts,
					CPUJoulesByZone: deltas,
				}, nil
			}
		}

		timer := time.NewTimer(pollInterval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}

		case <-timer.C:
		}
	}
}

// keplerEnergyDelta compares two cumulative Kepler snapshots.
//
// The function intentionally requires the zone set to remain stable during one
// invocation measurement. A zone that appears or disappears mid-window makes
// the measurement ambiguous and is therefore rejected.
func keplerEnergyDelta(before KeplerContainerSnapshot, after KeplerContainerSnapshot) (map[string]float64, bool, error) {

	beforeContainerID := normalizeKeplerContainerID(before.ContainerID)
	afterContainerID := normalizeKeplerContainerID(after.ContainerID)

	if beforeContainerID == "" || afterContainerID == "" {
		return nil, false, fmt.Errorf("Kepler energy snapshots must contain container IDs")
	}

	if beforeContainerID != afterContainerID {
		return nil, false, fmt.Errorf("Kepler container changed during energy measurement: before=%q after=%q", beforeContainerID, afterContainerID)
	}

	if len(before.CPUJoulesByZone) != len(after.CPUJoulesByZone) {
		return nil, false, fmt.Errorf("Kepler energy-zone set changed for container %q", beforeContainerID)
	}

	deltas := make(map[string]float64, len(before.CPUJoulesByZone))
	advanced := false

	for zone, beforeValue := range before.CPUJoulesByZone {

		afterValue, ok := after.CPUJoulesByZone[zone]
		if !ok {
			return nil, false, fmt.Errorf("Kepler energy zone %q disappeared for container %q", zone, beforeContainerID)
		}

		if !finiteKeplerCounter(beforeValue) || !finiteKeplerCounter(afterValue) {
			return nil, false, fmt.Errorf("Kepler energy zone %q contains an invalid counter", zone)
		}

		if afterValue < beforeValue {
			return nil, false, fmt.Errorf("Kepler energy counter regressed for container %q zone %q: before=%f after=%f", beforeContainerID, zone, beforeValue, afterValue)
		}

		delta := afterValue - beforeValue
		deltas[zone] = delta
		if delta > 0 {
			advanced = true
		}
	}

	return deltas, advanced, nil
}
