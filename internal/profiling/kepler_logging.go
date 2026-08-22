package profiling

import (
	"fmt"
	"log"
	"sort"
	"strings"
)

// LogKeplerInvocationEnergyProfile emits one deterministic structured log line
// for an invocation-level Kepler measurement.
//
// Energy zones remain separate. This function deliberately does not calculate
// a total because Kepler zones such as package and core must not be assumed to
// be additive.
func LogKeplerInvocationEnergyProfile(profile *KeplerInvocationEnergyProfile) {

	if profile == nil {
		return
	}

	containerID := normalizeKeplerContainerID(profile.ContainerID)
	if !profile.Available {
		log.Printf("[PROFILING] event=kepler_invocation_energy available=false container_id=%s reason=%q", containerID, profile.InvalidReason)
		return
	}

	zones := make([]string, 0, len(profile.CPUJoulesByZone))
	for zone := range profile.CPUJoulesByZone {
		zones = append(zones, zone)
	}

	sort.Strings(zones)
	zoneValues := make([]string, 0, len(zones))
	for _, zone := range zones {
		zoneValues = append(zoneValues, fmt.Sprintf("%s=%.9f", zone, profile.CPUJoulesByZone[zone]))
	}

	log.Printf(
		"[PROFILING] event=kepler_invocation_energy available=true container_id=%s poll_attempts=%d refresh_wait_ms=%.6f cpu_joules_by_zone=%q",
		containerID,
		profile.PollAttempts,
		profile.RefreshWaitMs,
		strings.Join(zoneValues, ","),
	)
}
