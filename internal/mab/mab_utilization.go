package mab

import "math"

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
//
// This helper is used by LinUCB. Classic UCB1 remains context-free and does not
// read the utilization context.
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
