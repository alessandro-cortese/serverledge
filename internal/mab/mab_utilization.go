package mab

import (
	"math"

	"github.com/serverledge-faas/serverledge/internal/config"
)

const defaultUtilizationThreshold = 0.70

// UtilizationScoreBreakdown describes how the current ring utilization
// changes an arm's selection score.
//
// Utilization is dynamic system state, so it is applied at decision time
// instead of being accumulated inside the arm's historical mean reward.
type UtilizationScoreBreakdown struct {
	BaseScore            float64
	Utilization          float64
	UtilizationThreshold float64
	UtilizationPenalty   float64
	UtilizationWeight    float64
	UtilizationTerm      float64
	FinalScore           float64
}

func clampUnitInterval(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0.0
	}
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

// armUtilization returns the ring-level memory utilization observed when the
// routing decision was made. Missing context is treated as zero utilization so
// that unavailable telemetry does not block an arm during startup.
func armUtilization(ctx *Context, arm string) float64 {
	if ctx == nil || ctx.ArchMemUsage == nil {
		return 0.0
	}

	utilization, ok := ctx.ArchMemUsage[arm]
	if !ok {
		return 0.0
	}

	return clampUnitInterval(utilization)
}

// utilizationPenalty is zero up to threshold and grows quadratically from
// threshold to 1.0. This avoids reacting aggressively to normal load while
// strongly discouraging rings that are close to saturation.
func utilizationPenalty(
	utilization float64,
	threshold float64,
) float64 {
	u := clampUnitInterval(utilization)
	t := clampUnitInterval(threshold)

	if t >= 1.0 || u <= t {
		return 0.0
	}

	normalized := (u - t) / (1.0 - t)
	return normalized * normalized
}

func configuredUCB1UtilizationThreshold() float64 {
	return clampUnitInterval(
		config.GetFloat(
			config.MAB_UCB1_UTILIZATION_THRESHOLD,
			defaultUtilizationThreshold,
		),
	)
}

func configuredUCB1UtilizationWeight() float64 {
	weight := config.GetFloat(
		config.MAB_UCB1_UTILIZATION_WEIGHT,
		0.0,
	)

	if math.IsNaN(weight) ||
		math.IsInf(weight, 0) ||
		weight < 0.0 {

		return 0.0
	}

	return weight
}

func buildUtilizationScoreBreakdown(
	arm string,
	baseScore float64,
	ctx *Context,
	utilizationWeight float64,
) UtilizationScoreBreakdown {

	utilization := armUtilization(ctx, arm)

	// UtilizationScoreBreakdown describes how the optional
	// UCB1UtilizationAware policy changes an arm's selection score. Plain UCB1
	// does not call this helper, while LinUCB uses utilization natively as a
	// contextual feature without this additional deterministic penalty.
	threshold := configuredUCB1UtilizationThreshold()
	penalty := utilizationPenalty(
		utilization,
		threshold,
	)
	term := utilizationWeight * penalty

	return UtilizationScoreBreakdown{
		BaseScore:            baseScore,
		Utilization:          utilization,
		UtilizationThreshold: threshold,
		UtilizationPenalty:   penalty,
		UtilizationWeight:    utilizationWeight,
		UtilizationTerm:      term,
		FinalScore:           baseScore - term,
	}
}
