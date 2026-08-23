package mab

import (
	"fmt"
	"math"
)

const WeakMABPriorSchemaVersion = 1

const (
	WeakPriorSkipNoRealObservations           = "no_real_observations"
	WeakPriorSkipInsufficientRealObservations = "insufficient_real_observations"
)

// WeakMABPriorConfig defines how much donor knowledge can be represented by
// the weak prior.
//
// EquivalentObservationWeight is expressed as an equivalent number of donor
// observations. In the first implementation it is deliberately constrained to
// (0, 1] so that one arm can never receive more than one equivalent donor
// observation.
//
// MinRealObservationsPerArm prevents a donor arm with too little real evidence
// from being used as a transfer source.
type WeakMABPriorConfig struct {
	EquivalentObservationWeight float64 `json:"equivalent_observation_weight"`
	MinRealObservationsPerArm   int64   `json:"min_real_observations_per_arm"`
}

// WeakMABPrior is a donor-derived, attenuated knowledge artifact.
//
// It is deliberately independent from the live state of the target function.
// Building this object does not modify UCB1 or LinUCB.
type WeakMABPrior struct {
	SchemaVersion                           int                        `json:"schema_version"`
	DonorFunctionName                       string                     `json:"donor_function_name"`
	Policy                                  BanditType                 `json:"policy"`
	Config                                  WeakMABPriorConfig         `json:"config"`
	HasPrior                                bool                       `json:"has_prior"`
	SourceRealObservationCount              int64                      `json:"source_real_observation_count"`
	SourceExcludedSyntheticObservationCount int64                      `json:"source_excluded_synthetic_observation_count"`
	ArmCount                                int                        `json:"arm_count"`
	TransferredArmCount                     int                        `json:"transferred_arm_count"`
	SkippedArmCount                         int                        `json:"skipped_arm_count"`
	Arms                                    map[string]WeakMABArmPrior `json:"arms"`
}

// WeakMABArmPrior contains the attenuated prior for one arm.
type WeakMABArmPrior struct {
	SourceRealObservationCount              int64               `json:"source_real_observation_count"`
	SourceExcludedSyntheticObservationCount int64               `json:"source_excluded_synthetic_observation_count"`
	Transferred                             bool                `json:"transferred"`
	SkipReason                              string              `json:"skip_reason,omitempty"`
	AppliedEquivalentObservationWeight      float64             `json:"applied_equivalent_observation_weight"`
	AttenuationScale                        float64             `json:"attenuation_scale"`
	UCB1                                    *WeakUCB1ArmPrior   `json:"ucb1,omitempty"`
	LinUCB                                  *WeakLinUCBArmPrior `json:"linucb,omitempty"`
}

// WeakUCB1ArmPrior represents an attenuated UCB1 reward prior.
//
// ObservationWeight is intentionally float64 because it represents statistical
// prior weight, not a count of real completed executions.
type WeakUCB1ArmPrior struct {
	ObservationWeight float64 `json:"observation_weight"`
	RewardSum         float64 `json:"reward_sum"`
	MeanReward        float64 `json:"mean_reward"`
}

// WeakLinUCBArmPrior contains the attenuated contextual contribution that can
// later be added to the target LinUCB model.
type WeakLinUCBArmPrior struct {
	ObservationWeight float64     `json:"observation_weight"`
	Dim               int         `json:"dim"`
	AContribution     [][]float64 `json:"a_contribution"`
	BContribution     []float64   `json:"b_contribution"`
}

// BuildWeakMABPrior transforms transferable real-feedback knowledge into a
// deliberately weak donor prior.
//
// This function is pure with respect to the live MABs: it does not mutate the
// donor and does not apply anything to a target function.
func BuildWeakMABPrior(source TransferableMABKnowledge, config WeakMABPriorConfig) (WeakMABPrior, error) {

	if err := validateWeakMABPriorConfig(config); err != nil {
		return WeakMABPrior{}, err
	}

	if err := validateTransferableKnowledgeForPrior(source); err != nil {
		return WeakMABPrior{}, err
	}

	arms := make(map[string]WeakMABArmPrior, len(source.Arms))
	transferredArmCount := 0
	for arm, sourceArm := range source.Arms {
		priorArm := WeakMABArmPrior{
			SourceRealObservationCount:              sourceArm.RealObservationCount,
			SourceExcludedSyntheticObservationCount: sourceArm.ExcludedSyntheticObservationCount,
		}

		if sourceArm.RealObservationCount == 0 {
			priorArm.SkipReason = WeakPriorSkipNoRealObservations
			arms[arm] = priorArm
			continue
		}

		if sourceArm.RealObservationCount < config.MinRealObservationsPerArm {
			priorArm.SkipReason = WeakPriorSkipInsufficientRealObservations
			arms[arm] = priorArm
			continue
		}

		appliedWeight := math.Min(config.EquivalentObservationWeight, float64(sourceArm.RealObservationCount))
		scale := appliedWeight / float64(sourceArm.RealObservationCount)
		priorArm.Transferred = true
		priorArm.AppliedEquivalentObservationWeight = appliedWeight
		priorArm.AttenuationScale = scale

		switch source.Policy {
		case UCB1:
			sourceUCB := sourceArm.UCB1
			rewardSum := sourceUCB.RealSumRewards * scale
			meanReward := rewardSum / appliedWeight
			priorArm.UCB1 = &WeakUCB1ArmPrior{
				ObservationWeight: appliedWeight,
				RewardSum:         rewardSum,
				MeanReward:        meanReward,
			}

		case LinUCB:
			sourceLin := sourceArm.LinUCB
			priorArm.LinUCB = &WeakLinUCBArmPrior{
				ObservationWeight: appliedWeight,
				Dim:               sourceLin.Dim,
				AContribution:     scaleNestedMatrix(sourceLin.AContribution, scale),
				BContribution:     scaleVector(sourceLin.BContribution, scale),
			}

		default:
			return WeakMABPrior{},
				fmt.Errorf("unsupported MAB policy %q", source.Policy)
		}

		transferredArmCount++
		arms[arm] = priorArm
	}

	return WeakMABPrior{
			SchemaVersion:                           WeakMABPriorSchemaVersion,
			DonorFunctionName:                       source.FunctionName,
			Policy:                                  source.Policy,
			Config:                                  config,
			HasPrior:                                transferredArmCount > 0,
			SourceRealObservationCount:              source.RealObservationCount,
			SourceExcludedSyntheticObservationCount: source.ExcludedSyntheticObservationCount,
			ArmCount:                                len(source.Arms),
			TransferredArmCount:                     transferredArmCount,
			SkippedArmCount:                         len(source.Arms) - transferredArmCount,
			Arms:                                    arms,
		},
		nil
}

func validateWeakMABPriorConfig(config WeakMABPriorConfig) error {

	weight := config.EquivalentObservationWeight
	if !isFiniteNumber(weight) {
		return fmt.Errorf("equivalent observation weight must be finite")
	}

	if weight <= 0.0 || weight > 1.0 {
		return fmt.Errorf("equivalent observation weight must be in (0, 1]")
	}

	if config.MinRealObservationsPerArm < 1 {
		return fmt.Errorf("minimum real observations per arm must be at least 1")
	}

	return nil
}

func validateTransferableKnowledgeForPrior(source TransferableMABKnowledge) error {

	if source.SchemaVersion != TransferableMABKnowledgeSchemaVersion {
		return fmt.Errorf("unsupported transferable MAB knowledge schema version %d", source.SchemaVersion)
	}

	if source.FunctionName == "" {
		return fmt.Errorf("donor function name cannot be empty")
	}

	switch source.Policy {
	case UCB1, LinUCB:
		// Supported.

	default:
		return fmt.Errorf("unsupported MAB policy %q", source.Policy)
	}

	var totalReal int64
	var totalSynthetic int64

	for arm, sourceArm := range source.Arms {
		if arm == "" {
			return fmt.Errorf("arm name cannot be empty")
		}

		if sourceArm.RealObservationCount < 0 {
			return fmt.Errorf("arm %q has negative real observation count", arm)
		}

		if sourceArm.ExcludedSyntheticObservationCount < 0 {
			return fmt.Errorf("arm %q has negative synthetic observation count", arm)
		}

		totalReal += sourceArm.RealObservationCount
		totalSynthetic += sourceArm.ExcludedSyntheticObservationCount

		switch source.Policy {

		case UCB1:
			if sourceArm.UCB1 == nil || sourceArm.LinUCB != nil {
				return fmt.Errorf("arm %q has invalid UCB1 transferable payload", arm)
			}

			if err := validateTransferableUCB1Arm(arm, sourceArm.RealObservationCount, sourceArm.UCB1); err != nil {
				return err
			}

		case LinUCB:
			if sourceArm.LinUCB == nil || sourceArm.UCB1 != nil {
				return fmt.Errorf("arm %q has invalid LinUCB transferable payload", arm)
			}

			if err := validateTransferableLinUCBArm(arm, sourceArm.LinUCB); err != nil {
				return err
			}
		}
	}

	if totalReal != source.RealObservationCount {
		return fmt.Errorf("transferable real observation summary mismatch: arms=%d summary=%d", totalReal, source.RealObservationCount)
	}

	if totalSynthetic != source.ExcludedSyntheticObservationCount {
		return fmt.Errorf("transferable synthetic observation summary mismatch: arms=%d summary=%d", totalSynthetic, source.ExcludedSyntheticObservationCount)
	}

	if source.HasRealKnowledge != (totalReal > 0) {
		return fmt.Errorf("has_real_knowledge does not match real observation count")
	}

	return nil
}

func validateTransferableUCB1Arm(arm string, realCount int64, source *TransferableUCB1ArmKnowledge) error {
	if !isFiniteNumber(source.RealSumRewards) || !isFiniteNumber(source.RealAvgReward) {
		return fmt.Errorf("arm %q has non-finite UCB1 transferable reward statistics", arm)
	}

	if realCount == 0 {
		if source.RealSumRewards != 0.0 || source.RealAvgReward != 0.0 {
			return fmt.Errorf("arm %q has UCB1 reward statistics without real observations", arm)
		}

		return nil
	}

	expectedAverage := source.RealSumRewards / float64(realCount)
	if !weakPriorAlmostEqual(expectedAverage, source.RealAvgReward) {
		return fmt.Errorf("arm %q has inconsistent UCB1 average reward", arm)
	}

	return nil
}

func validateTransferableLinUCBArm(arm string, source *TransferableLinUCBArmKnowledge) error {
	if source.Dim < 1 {
		return fmt.Errorf("arm %q has invalid LinUCB dimension", arm)
	}

	if len(source.AContribution) != source.Dim {
		return fmt.Errorf("arm %q has invalid LinUCB A row count", arm)
	}

	for i, row := range source.AContribution {
		if len(row) != source.Dim {
			return fmt.Errorf("arm %q has invalid LinUCB A column count at row %d", arm, i)
		}

		for _, value := range row {
			if !isFiniteNumber(value) {
				return fmt.Errorf("arm %q has non-finite LinUCB A contribution", arm)
			}
		}
	}

	if len(source.BContribution) != source.Dim {
		return fmt.Errorf("arm %q has invalid LinUCB b dimension", arm)
	}

	for _, value := range source.BContribution {
		if !isFiniteNumber(value) {
			return fmt.Errorf("arm %q has non-finite LinUCB b contribution", arm)
		}
	}

	return nil
}

func scaleNestedMatrix(source [][]float64, scale float64) [][]float64 {

	result := make([][]float64, len(source))

	for i, row := range source {
		result[i] = make([]float64, len(row))

		for j, value := range row {
			result[i][j] = value * scale
		}
	}

	return result
}

func scaleVector(source []float64, scale float64) []float64 {

	result := make([]float64, len(source))

	for i, value := range source {
		result[i] = value * scale
	}

	return result
}

func weakPriorAlmostEqual(left float64, right float64) bool {
	difference := math.Abs(left - right)
	scale := math.Max(1.0, math.Max(math.Abs(left), math.Abs(right)))
	return difference <= 1e-9*scale
}
