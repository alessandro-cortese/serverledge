package mab

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/config"
)

// KeplerRewardInput is the energy measurement selected from the Kepler
// execution feedback for use by a future energy-based reward calculator.
//
// This type deliberately represents one Kepler zone only. Different hardware
// energy domains are not summed automatically.
type KeplerRewardInput struct {
	Zone   string
	Joules float64
}

// keplerEnergyReferenceJoules makes the logarithm argument dimensionless:
//
//	reward = -ln(E / E_ref)
//
// with:
//
//	E_ref = 1 J
//
// Since KeplerRewardInput.Joules is expressed in joules, the numerical
// calculation is equivalent to -ln(E_joules), while keeping the physical
// meaning of the transformation explicit.
const keplerEnergyReferenceJoules = 1.0

// ConfiguredKeplerRewardZone returns the Kepler zone explicitly selected for
// the future energy reward.
//
// There is intentionally no default zone. Choosing package, core, or another
// hardware domain is an experimental decision and must therefore be explicit.
func ConfiguredKeplerRewardZone() (string, error) {

	zone := strings.ToLower(strings.TrimSpace(config.GetString(config.MAB_REWARD_KEPLER_ZONE, "")))
	if zone == "" {
		return "", fmt.Errorf("Kepler reward zone is not configured: set %s", config.MAB_REWARD_KEPLER_ZONE)
	}

	return zone, nil
}

// SelectKeplerRewardInput extracts exactly one configured Kepler zone from a
// real execution feedback.
//
// It does not convert the measurement into a MAB reward. Its only purpose is
// to provide a validated energy input for the future reward calculator.
func SelectKeplerRewardInput(feedback ExecutionFeedback) (KeplerRewardInput, error) {

	if feedback.KeplerEnergy == nil {
		return KeplerRewardInput{}, fmt.Errorf("Kepler energy feedback is missing")
	}

	if !feedback.KeplerEnergy.Available {
		reason := strings.TrimSpace(feedback.KeplerEnergy.InvalidReason)
		if reason == "" {
			reason = "unspecified Kepler collection failure"
		}

		return KeplerRewardInput{}, fmt.Errorf("Kepler energy feedback is unavailable: %s", reason)
	}

	zone, err := ConfiguredKeplerRewardZone()

	if err != nil {
		return KeplerRewardInput{},
			err
	}

	if len(
		feedback.KeplerEnergy.CPUJoulesByZone,
	) == 0 {

		return KeplerRewardInput{}, fmt.Errorf("Kepler energy feedback contains no CPU energy zones")
	}

	joules, ok := feedback.KeplerEnergy.CPUJoulesByZone[zone]

	// Kepler currently exposes lower-case zone labels, but keep the lookup
	// tolerant to case differences without changing the configured semantic
	// domain.
	if !ok {
		for availableZone, availableJoules := range feedback.KeplerEnergy.CPUJoulesByZone {
			if strings.EqualFold(strings.TrimSpace(availableZone), zone) {
				joules = availableJoules
				ok = true
				break
			}
		}
	}

	if !ok {
		availableZones := make([]string, 0, len(feedback.KeplerEnergy.CPUJoulesByZone))
		for availableZone := range feedback.KeplerEnergy.CPUJoulesByZone {
			availableZones = append(availableZones, availableZone)
		}

		sort.Strings(availableZones)
		return KeplerRewardInput{}, fmt.Errorf("configured Kepler reward zone %q is unavailable; available zones: %v", zone, availableZones)
	}

	if math.IsNaN(joules) || math.IsInf(joules, 0) {
		return KeplerRewardInput{}, fmt.Errorf("Kepler energy for zone %q must be finite, got %v", zone, joules)
	}

	if joules < 0 {
		return KeplerRewardInput{},
			fmt.Errorf("Kepler energy for zone %q cannot be negative, got %.9f J", zone, joules)
	}

	return KeplerRewardInput{
		Zone:   zone,
		Joules: joules,
	}, nil
}

// calculateKeplerEnergyReward converts the explicitly selected Kepler energy
// domain into a scalar reward.
//
// Lower energy consumption must result in a larger reward, therefore:
//
//	reward = -ln(E / 1 J)
//
// The selector accepts zero because selecting/validating a measurement is
// independent from the reward transformation. The logarithmic reward,
// however, requires strictly positive energy.
func calculateKeplerEnergyReward(feedback ExecutionFeedback) (RewardResult, error) {

	input, err := SelectKeplerRewardInput(feedback)
	if err != nil {
		return RewardResult{}, err
	}

	if input.Joules <= 0 {
		return RewardResult{}, fmt.Errorf("Kepler energy for reward zone %q must be positive for logarithmic reward, got %.9f J", input.Zone, input.Joules)
	}

	normalizedEnergy := input.Joules / keplerEnergyReferenceJoules
	reward := -math.Log(normalizedEnergy)
	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		return RewardResult{}, fmt.Errorf("Kepler logarithmic reward for zone %q is not finite: energy=%.9f J reward=%v", input.Zone, input.Joules, reward)
	}

	return RewardResult{
		Mode:       RewardModeKeplerEnergy,
		Value:      reward,
		InputName:  "kepler_energy_" + input.Zone,
		InputValue: input.Joules,
		InputUnit:  "J",
	}, nil
}
