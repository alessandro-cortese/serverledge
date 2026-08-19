package mab

import (
	"fmt"
	"math"
	"strings"

	"github.com/serverledge-faas/serverledge/internal/config"
)

// RewardMode identifies the objective used to transform real execution
// feedback into the scalar reward consumed by a MAB policy.
//
// Reward modes affect only real execution feedback. Synthetic rewards, such as
// the fallback penalty, follow their existing independent path.
type RewardMode string

const (
	// RewardModeLatency is the current Serverledge baseline:
	//
	//	reward = -ln(DurationMs)
	RewardModeLatency      RewardMode = "latency"
	RewardModeKeplerEnergy RewardMode = "kepler_energy"
)

// RewardResult describes both the scalar reward and the measurement from which
// it was derived. Keeping this information together makes reward logging
// independent from the concrete objective.
type RewardResult struct {
	Mode RewardMode

	Value float64

	InputName  string
	InputValue float64
	InputUnit  string
}

// RewardCalculator converts execution feedback into a scalar MAB reward.
type RewardCalculator func(
	feedback ExecutionFeedback,
) (
	RewardResult,
	error,
)

// rewardCalculators is the registry of supported reward objectives.
//
// Adding a new objective must not require modifications to UCB1 or LinUCB:
// the new calculator only needs to be registered here.
var rewardCalculators = map[RewardMode]RewardCalculator{
	RewardModeLatency:      calculateLatencyReward,
	RewardModeKeplerEnergy: calculateKeplerEnergyReward,
}

// ConfiguredRewardMode returns the reward objective selected through
// mab.reward.mode.
//
// An empty or missing configuration retains the historical latency baseline.
// Unknown non-empty values are rejected rather than silently falling back to
// latency, because silently changing the experimental objective would make
// measurements unreliable.
func ConfiguredRewardMode() (
	RewardMode,
	error,
) {
	rawMode :=
		strings.ToLower(
			strings.TrimSpace(
				config.GetString(
					config.MAB_REWARD_MODE,
					string(
						RewardModeLatency,
					),
				),
			),
		)

	if rawMode == "" {
		rawMode =
			string(
				RewardModeLatency,
			)
	}

	mode :=
		RewardMode(
			rawMode,
		)

	if _,
		ok :=
		rewardCalculators[mode]; !ok {

		return "",
			fmt.Errorf(
				"unsupported MAB reward mode %q",
				rawMode,
			)
	}

	return mode,
		nil
}

// ValidateRewardConfiguration can be used during application startup to reject
// an unsupported reward objective before an experiment begins.
func ValidateRewardConfiguration() error {
	mode,
		err :=
		ConfiguredRewardMode()

	if err != nil {
		return err
	}

	if mode ==
		RewardModeKeplerEnergy {

		_,
			err :=
			ConfiguredKeplerRewardZone()

		if err != nil {
			return err
		}
	}

	return nil
}

// CalculateExecutionReward selects the configured reward strategy and applies
// it to one real execution feedback.
//
// UCB1 and LinUCB must use this function instead of embedding a reward formula
// directly in the policy implementation.
func CalculateExecutionReward(
	feedback ExecutionFeedback,
) (
	RewardResult,
	error,
) {
	mode,
		err :=
		ConfiguredRewardMode()

	if err != nil {
		return RewardResult{},
			err
	}

	calculator :=
		rewardCalculators[mode]

	return calculator(
		feedback,
	)
}

// calculateLatencyReward implements the current baseline objective.
func calculateLatencyReward(
	feedback ExecutionFeedback,
) (
	RewardResult,
	error,
) {
	if !isFiniteNumber(
		feedback.DurationMs,
	) {
		return RewardResult{},
			fmt.Errorf(
				"duration must be finite, got %v",
				feedback.DurationMs,
			)
	}

	if feedback.DurationMs <= 0 {
		return RewardResult{},
			fmt.Errorf(
				"duration must be positive, got %.6f ms",
				feedback.DurationMs,
			)
	}

	value :=
		-math.Log(
			feedback.DurationMs,
		)

	return RewardResult{
		Mode: RewardModeLatency,

		Value: value,

		InputName: "duration_ms",

		InputValue: feedback.DurationMs,

		InputUnit: "ms",
	}, nil
}
