package mab

import (
	"log"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/config"
)

// ColdStartMode controls whether cold-start executions contribute to MAB
// learning.
//
// Initialization overhead is never added to the reward: execution mode uses
// only ExecutionFeedback.DurationMs.
type ColdStartMode string

const (
	// ColdStartModeSkip preserves the behavior inherited from the previous
	// implementation: cold-start samples do not update the MAB.
	ColdStartModeSkip ColdStartMode = "skip"

	// ColdStartModeExecution updates the MAB using the execution duration even
	// when the invocation required a cold start.
	//
	// InitTimeMs, QueueingTimeMs, OffloadLatencyMs and ResponseTimeMs remain
	// excluded from the reward.
	ColdStartModeExecution ColdStartMode = "execution"
)

func configuredColdStartMode() ColdStartMode {
	configuredMode := strings.ToLower(strings.TrimSpace(config.GetString(config.MAB_COLD_START_MODE, string(ColdStartModeSkip))))

	switch ColdStartMode(configuredMode) {
	case ColdStartModeSkip:
		return ColdStartModeSkip

	case ColdStartModeExecution:
		return ColdStartModeExecution

	default:
		log.Printf(
			"%s event=invalid_cold_start_mode ts=%d configured_mode=%s fallback_mode=%s\n",
			mabLogPrefix,
			nowMillis(),
			configuredMode,
			ColdStartModeSkip,
		)

		return ColdStartModeSkip
	}
}

// shouldUpdateRewardFromFeedback determines whether a valid feedback sample
// can update the selected policy.
//
// Validation must be performed before calling this function.
func shouldUpdateRewardFromFeedback(policy string, functionName string, arm string, feedback ExecutionFeedback) bool {

	if feedback.IsWarmStart {
		return true
	}

	mode := configuredColdStartMode()
	if mode == ColdStartModeExecution {
		logMABUseColdStartSample(
			policy,
			functionName,
			arm,
			feedback.DurationMs,
		)

		return true
	}

	GlobalColdStartStats.RecordColdSkipped(functionName, arm)
	logMABSkipColdStart(
		policy,
		functionName,
		arm,
		feedback.DurationMs,
	)

	logCurrentColdStartStats(
		policy,
		functionName,
		arm,
	)

	return false
}
