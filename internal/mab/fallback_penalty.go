package mab

import (
	"log"

	"github.com/serverledge-faas/serverledge/internal/config"
)

const defaultFallbackPenalty = -12.0

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
