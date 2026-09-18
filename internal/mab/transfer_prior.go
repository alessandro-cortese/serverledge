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

// UCB1ReferenceAnchorConfig describes the reference-architecture measurement
// already collected for a new target during profiling.
//
// The target is profiled on ReferenceArm before transfer. For UCB1-family
// policies, Serverledge can then transfer only the donor's relative
// architecture effect instead of copying the donor's absolute reward scale:
//
//	priorMean(reference) = TargetReferenceMeanReward
//	priorMean(arm) = TargetReferenceMeanReward
//	                 + (donorMean(arm) - donorMean(reference))
//
// With the latency reward r=-ln(duration_ms), the transferred difference is
// the logarithm of the donor's relative speedup/slowdown between architectures.
// No target observation from any non-reference architecture is required.
type UCB1ReferenceAnchorConfig struct {
	Enabled                   bool    `json:"enabled"`
	ReferenceArm              string  `json:"reference_arm,omitempty"`
	TargetReferenceMeanReward float64 `json:"target_reference_mean_reward,omitempty"`
}

// WeakMABPriorConfig defines how much donor knowledge can be represented by
// the weak prior.
//
// Legacy/coupled UCB1 and LinUCB use EquivalentObservationWeight.
//
// UCB1Decoupled instead uses two independent weights:
//   - RewardObservationWeight (w_R): strength of transferred knowledge in the
//     exploitation mean;
//   - ExplorationObservationWeight (w_E): pseudo-count used only by the UCB1
//     exploration term.
//
// All weights are expressed as equivalent observations and are deliberately
// constrained to (0, 1]. A value of 1 therefore means at most one equivalent
// prior observation per transferred arm. For UCB1Decoupled the legacy
// EquivalentObservationWeight must be left at zero to make the experiment
// unambiguous and reproducible.
//
// UCB1ReferenceAnchor is optional and applies only to UCB1-family policies.
// When enabled, the prior reward means are anchored to the target's measured
// reference-architecture mean reward as described above.
//
// MinRealObservationsPerArm prevents a donor arm with too little real evidence
// from being used as a transfer source.
type WeakMABPriorConfig struct {
	EquivalentObservationWeight  float64                    `json:"equivalent_observation_weight,omitempty"`
	RewardObservationWeight      float64                    `json:"reward_observation_weight,omitempty"`
	ExplorationObservationWeight float64                    `json:"exploration_observation_weight,omitempty"`
	MinRealObservationsPerArm    int64                      `json:"min_real_observations_per_arm"`
	UCB1ReferenceAnchor          *UCB1ReferenceAnchorConfig `json:"ucb1_reference_anchor,omitempty"`
}

// WeakMABPrior is a donor-derived, attenuated knowledge artifact.
//
// It is deliberately independent from the live state of the target function.
// Building this object does not modify UCB1, UCB1Decoupled or LinUCB.
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
	SourceRealObservationCount              int64  `json:"source_real_observation_count"`
	SourceExcludedSyntheticObservationCount int64  `json:"source_excluded_synthetic_observation_count"`
	Transferred                             bool   `json:"transferred"`
	SkipReason                              string `json:"skip_reason,omitempty"`

	// AppliedEquivalentObservationWeight is retained for backwards-compatible
	// provenance and represents the reward/exploitation prior weight. For the
	// legacy coupled formulation it is also the exploration pseudo-count.
	AppliedEquivalentObservationWeight float64 `json:"applied_equivalent_observation_weight"`

	// AppliedExplorationObservationWeight records the exploration-only
	// pseudo-count used by UCB1-family policies. It equals
	// AppliedEquivalentObservationWeight for legacy UCB1 and can differ for
	// UCB1Decoupled. LinUCB leaves it at zero.
	AppliedExplorationObservationWeight float64 `json:"applied_exploration_observation_weight,omitempty"`

	AttenuationScale float64             `json:"attenuation_scale"`
	UCB1             *WeakUCB1ArmPrior   `json:"ucb1,omitempty"`
	LinUCB           *WeakLinUCBArmPrior `json:"linucb,omitempty"`
}

// WeakUCB1ArmPrior represents an attenuated UCB1-family reward prior.
//
// ObservationWeight is w_R: statistical prior weight used by the exploitation
// mean. ExplorationObservationWeight is w_E: pseudo-count used only by the
// exploration term. For legacy UCB1 the two values are identical.
type WeakUCB1ArmPrior struct {
	ObservationWeight            float64 `json:"observation_weight"`
	ExplorationObservationWeight float64 `json:"exploration_observation_weight"`
	RewardSum                    float64 `json:"reward_sum"`
	MeanReward                   float64 `json:"mean_reward"`
}

// WeakLinUCBArmPrior contains the attenuated contextual contribution that can
// later be added to the target LinUCB model.
type WeakLinUCBArmPrior struct {
	ObservationWeight float64     `json:"observation_weight"`
	Dim               int         `json:"dim"`
	AContribution     [][]float64 `json:"a_contribution"`
	BContribution     []float64   `json:"b_contribution"`
}

type resolvedWeakPriorWeights struct {
	reward      float64
	exploration float64
}

type resolvedUCB1ReferenceAnchor struct {
	referenceArm              string
	targetReferenceMeanReward float64
	donorReferenceMeanReward  float64
}

// BuildWeakMABPrior transforms transferable real-feedback knowledge into a
// deliberately weak donor prior using the donor policy itself as the target
// policy. This preserves the historical API used by the existing unit tests
// and by callers that do not need cross-variant UCB1 transfer.
//
// For runtime experiments that reuse the same UCB1-family donor knowledge for
// both the coupled and decoupled target formulas, use
// BuildWeakMABPriorForTarget instead.
func BuildWeakMABPrior(source TransferableMABKnowledge, config WeakMABPriorConfig) (WeakMABPrior, error) {
	return BuildWeakMABPriorForTarget(source, source.Policy, config)
}

// BuildWeakMABPriorForTarget transforms donor knowledge into a prior whose
// semantics are defined by targetPolicy.
//
// The donor policy identifies the representation of the stored knowledge.
// UCB1 and UCB1Decoupled deliberately share the same transferable real-feedback
// representation (real reward count/sum/mean), therefore knowledge collected
// by either UCB1 variant can initialize either UCB1 target variant. The target
// policy decides how that common knowledge is weighted:
//   - UCB1: one coupled equivalent-observation weight;
//   - UCB1Decoupled: independent reward/exploitation and exploration weights.
//
// LinUCB remains a separate knowledge family and is only transferable to a
// LinUCB target.
//
// This function is pure with respect to the live MABs: it does not mutate the
// donor and does not apply anything to a target function.
func BuildWeakMABPriorForTarget(source TransferableMABKnowledge, targetPolicy BanditType, config WeakMABPriorConfig) (WeakMABPrior, error) {
	if err := validateWeakMABPriorConfig(config); err != nil {
		return WeakMABPrior{}, err
	}

	if err := validateTransferableKnowledgeForPrior(source); err != nil {
		return WeakMABPrior{}, err
	}

	if err := validateTransferPolicyCompatibility(source.Policy, targetPolicy); err != nil {
		return WeakMABPrior{}, err
	}

	if err := validateWeakMABPriorConfigForPolicy(config, targetPolicy); err != nil {
		return WeakMABPrior{}, err
	}

	weights, err := resolveWeakPriorWeights(config, targetPolicy)
	if err != nil {
		return WeakMABPrior{}, err
	}

	anchor, err := resolveUCB1ReferenceAnchor(source, config)
	if err != nil {
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

		appliedRewardWeight := math.Min(weights.reward, float64(sourceArm.RealObservationCount))
		appliedExplorationWeight := 0.0
		if isUCB1FamilyPolicy(targetPolicy) {
			appliedExplorationWeight = math.Min(weights.exploration, float64(sourceArm.RealObservationCount))
		}

		scale := appliedRewardWeight / float64(sourceArm.RealObservationCount)
		priorArm.Transferred = true
		priorArm.AppliedEquivalentObservationWeight = appliedRewardWeight
		priorArm.AppliedExplorationObservationWeight = appliedExplorationWeight
		priorArm.AttenuationScale = scale

		switch targetPolicy {
		case UCB1, UCB1Decoupled:
			sourceUCB := sourceArm.UCB1
			meanReward := sourceUCB.RealAvgReward

			if anchor != nil {
				meanReward = anchor.targetReferenceMeanReward +
					(sourceUCB.RealAvgReward - anchor.donorReferenceMeanReward)
			}

			rewardSum := meanReward * appliedRewardWeight
			priorArm.UCB1 = &WeakUCB1ArmPrior{
				ObservationWeight:            appliedRewardWeight,
				ExplorationObservationWeight: appliedExplorationWeight,
				RewardSum:                    rewardSum,
				MeanReward:                   meanReward,
			}

		case LinUCB:
			sourceLin := sourceArm.LinUCB
			priorArm.LinUCB = &WeakLinUCBArmPrior{
				ObservationWeight: appliedRewardWeight,
				Dim:               sourceLin.Dim,
				AContribution:     scaleNestedMatrix(sourceLin.AContribution, scale),
				BContribution:     scaleVector(sourceLin.BContribution, scale),
			}

		default:
			return WeakMABPrior{}, fmt.Errorf("unsupported target MAB policy %q", targetPolicy)
		}

		transferredArmCount++
		arms[arm] = priorArm
	}

	return WeakMABPrior{
			SchemaVersion:                           WeakMABPriorSchemaVersion,
			DonorFunctionName:                       source.FunctionName,
			Policy:                                  targetPolicy,
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

func weakPriorUsesDecoupledWeights(config WeakMABPriorConfig) bool {
	return config.RewardObservationWeight != 0.0 || config.ExplorationObservationWeight != 0.0
}

func validateObservationWeight(name string, weight float64) error {
	if !isFiniteNumber(weight) {
		return fmt.Errorf("%s must be finite", name)
	}

	if weight <= 0.0 || weight > 1.0 {
		return fmt.Errorf("%s must be in (0, 1]", name)
	}

	return nil
}

func validateWeakMABPriorConfig(config WeakMABPriorConfig) error {
	if weakPriorUsesDecoupledWeights(config) {
		if config.EquivalentObservationWeight != 0.0 {
			return fmt.Errorf("equivalent observation weight must be zero when decoupled UCB1 weights are configured")
		}

		if err := validateObservationWeight("reward observation weight", config.RewardObservationWeight); err != nil {
			return err
		}

		if err := validateObservationWeight("exploration observation weight", config.ExplorationObservationWeight); err != nil {
			return err
		}
	} else {
		if err := validateObservationWeight("equivalent observation weight", config.EquivalentObservationWeight); err != nil {
			return err
		}
	}

	if config.MinRealObservationsPerArm < 1 {
		return fmt.Errorf("minimum real observations per arm must be at least 1")
	}

	if anchor := config.UCB1ReferenceAnchor; anchor != nil && anchor.Enabled {
		if anchor.ReferenceArm == "" {
			return fmt.Errorf("UCB1 reference anchor arm cannot be empty")
		}

		if !isFiniteNumber(anchor.TargetReferenceMeanReward) {
			return fmt.Errorf("UCB1 target reference mean reward must be finite")
		}
	}

	return nil
}

func validateWeakMABPriorConfigForPolicy(config WeakMABPriorConfig, policy BanditType) error {
	usesDecoupled := weakPriorUsesDecoupledWeights(config)
	anchorEnabled := config.UCB1ReferenceAnchor != nil && config.UCB1ReferenceAnchor.Enabled

	switch policy {
	case UCB1:
		if usesDecoupled {
			return fmt.Errorf("decoupled reward/exploration weights require policy %s", UCB1Decoupled)
		}

	case UCB1Decoupled:
		if !usesDecoupled {
			return fmt.Errorf("policy %s requires reward_observation_weight and exploration_observation_weight", UCB1Decoupled)
		}

	case LinUCB:
		if usesDecoupled {
			return fmt.Errorf("decoupled UCB1 weights are not valid for LinUCB")
		}
		if anchorEnabled {
			return fmt.Errorf("UCB1 reference anchoring is not valid for LinUCB")
		}

	default:
		return fmt.Errorf("unsupported MAB policy %q", policy)
	}

	return nil
}

func isUCB1FamilyPolicy(policy BanditType) bool {
	return policy == UCB1 || policy == UCB1Decoupled
}

func validateTransferPolicyCompatibility(sourcePolicy BanditType, targetPolicy BanditType) error {
	switch sourcePolicy {
	case UCB1, UCB1Decoupled, LinUCB:
		// Supported donor policy.
	default:
		return fmt.Errorf("unsupported donor MAB policy %q", sourcePolicy)
	}

	switch targetPolicy {
	case UCB1, UCB1Decoupled, LinUCB:
		// Supported target policy.
	default:
		return fmt.Errorf("unsupported target MAB policy %q", targetPolicy)
	}

	if sourcePolicy == targetPolicy {
		return nil
	}

	if isUCB1FamilyPolicy(sourcePolicy) && isUCB1FamilyPolicy(targetPolicy) {
		return nil
	}

	return fmt.Errorf(
		"incompatible transfer policies: donor=%s target=%s",
		sourcePolicy,
		targetPolicy,
	)
}

func resolveWeakPriorWeights(config WeakMABPriorConfig, policy BanditType) (resolvedWeakPriorWeights, error) {
	if err := validateWeakMABPriorConfigForPolicy(config, policy); err != nil {
		return resolvedWeakPriorWeights{}, err
	}

	if policy == UCB1Decoupled {
		return resolvedWeakPriorWeights{
			reward:      config.RewardObservationWeight,
			exploration: config.ExplorationObservationWeight,
		}, nil
	}

	return resolvedWeakPriorWeights{
		reward:      config.EquivalentObservationWeight,
		exploration: config.EquivalentObservationWeight,
	}, nil
}

func resolveUCB1ReferenceAnchor(source TransferableMABKnowledge, config WeakMABPriorConfig) (*resolvedUCB1ReferenceAnchor, error) {
	anchorConfig := config.UCB1ReferenceAnchor
	if anchorConfig == nil || !anchorConfig.Enabled {
		return nil, nil
	}

	if !isUCB1FamilyPolicy(source.Policy) {
		return nil, fmt.Errorf("UCB1 reference anchoring requires a UCB1-family donor")
	}

	referenceArm, exists := source.Arms[anchorConfig.ReferenceArm]
	if !exists {
		return nil, fmt.Errorf("UCB1 reference anchor arm %q is not present in donor knowledge", anchorConfig.ReferenceArm)
	}

	if referenceArm.RealObservationCount < config.MinRealObservationsPerArm {
		return nil, fmt.Errorf("UCB1 reference anchor arm %q has insufficient real donor observations", anchorConfig.ReferenceArm)
	}

	if referenceArm.UCB1 == nil {
		return nil, fmt.Errorf("UCB1 reference anchor arm %q has no UCB1 reward statistics", anchorConfig.ReferenceArm)
	}

	return &resolvedUCB1ReferenceAnchor{
		referenceArm:              anchorConfig.ReferenceArm,
		targetReferenceMeanReward: anchorConfig.TargetReferenceMeanReward,
		donorReferenceMeanReward:  referenceArm.UCB1.RealAvgReward,
	}, nil
}

func validateTransferableKnowledgeForPrior(source TransferableMABKnowledge) error {
	if source.SchemaVersion != TransferableMABKnowledgeSchemaVersion {
		return fmt.Errorf("unsupported transferable MAB knowledge schema version %d", source.SchemaVersion)
	}

	if source.FunctionName == "" {
		return fmt.Errorf("donor function name cannot be empty")
	}

	switch source.Policy {
	case UCB1, UCB1Decoupled, LinUCB:
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
		case UCB1, UCB1Decoupled:
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
