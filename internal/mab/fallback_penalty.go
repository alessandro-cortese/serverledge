package mab

import (
	"log"

	"github.com/serverledge-faas/serverledge/internal/config"
)

// defaultFallbackPenalty is the synthetic reward assigned to an arm that could
// not serve the request and had to fall back to another ring.
//
// The latency reward is -ln(DurationMs), so a penalty value corresponds to an
// equivalent duration of e^(-penalty) milliseconds. The inherited value of -12
// corresponds to roughly 162 seconds: with functions that exceed one hundred
// seconds on the slower architecture, a fallback would score within half a
// point of a legitimate but slow execution, and the bandit could not tell the
// two apart.
//
// -20 corresponds to an equivalent duration several orders of magnitude beyond
// any real execution, so a fallback is unambiguously the worst outcome an arm
// can produce, whatever the workload.
const defaultFallbackPenalty = -20.0

func configuredFallbackPenalty() float64 {
	penalty := config.GetFloat(config.MAB_FALLBACK_PENALTY, defaultFallbackPenalty)

	if !isFiniteNumber(penalty) || penalty >= 0 {
		log.Printf(
			"%s event=invalid_fallback_penalty ts=%d configured_penalty=%f fallback_penalty=%f\n",
			mabLogPrefix,
			nowMillis(),
			penalty,
			defaultFallbackPenalty,
		)

		return defaultFallbackPenalty
	}

	return penalty
}

func fallbackSyntheticReward(decision DecisionRecord, reason string) *SyntheticReward {
	requiresPenalty := decision.Fallback || reason == DecisionFailureReasonNoCandidateAfterFallback
	if !requiresPenalty {
		return nil
	}

	penaltyReason := decision.FallbackReason
	if penaltyReason == "" {
		penaltyReason = reason
	}

	if penaltyReason == "" {
		penaltyReason = FallbackReasonObservedExecutionDiffers
	}

	return &SyntheticReward{
		RequestID: decision.RequestID,
		Value:     configuredFallbackPenalty(),
		Reason:    penaltyReason,
	}
}
