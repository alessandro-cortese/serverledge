package mab

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// ApplyWeakMABPrior applies an already-built weak donor prior to a fresh target
// policy.
//
// The operation only changes the target's decision state:
//   - UCB1 keeps prior weight/reward separate from Count/SumRewards/Real*;
//   - LinUCB adds the prior contribution to live A and b while keeping it
//     separate from RealAContribution/RealBContribution.
//
// Consequently, a received prior can influence the target's early decisions
// without ever being re-exported as real target knowledge.
func ApplyWeakMABPrior(target Policy, prior WeakMABPrior) error {

	if target == nil {
		return fmt.Errorf("target MAB policy cannot be nil")
	}

	if err := validateWeakMABPriorForApplication(prior); err != nil {
		return err
	}

	if target.GetType() != prior.Policy {
		return fmt.Errorf("weak prior policy %q does not match target policy %q", prior.Policy, target.GetType())
	}

	switch typed := target.(type) {
	case *UCB1Bandit:
		if typed == nil {
			return fmt.Errorf("target UCB1 policy cannot be nil")
		}

		if typed.FunctionName == prior.DonorFunctionName {
			return fmt.Errorf("weak prior donor and target function must be different")
		}

		return typed.applyWeakPrior(prior)

	case *LinUCBDisjointPolicy:
		if typed == nil {
			return fmt.Errorf("target LinUCB policy cannot be nil")
		}

		if typed.FunctionName == prior.DonorFunctionName {
			return fmt.Errorf("weak prior donor and target function must be different")
		}

		return typed.applyWeakPrior(prior)

	default:
		return fmt.Errorf("unsupported target MAB policy type %T", target)
	}
}

func (b *UCB1Bandit) applyWeakPrior(prior WeakMABPrior) error {

	b.mu.Lock()
	defer b.mu.Unlock()

	if err := b.validateFreshTargetForWeakPriorLocked(); err != nil {
		return err
	}

	for arm, priorArm := range prior.Arms {
		if !priorArm.Transferred {
			continue
		}

		if _, exists := b.Arms[arm]; !exists {
			return fmt.Errorf("weak prior arm %q is not initialized in target UCB1 policy", arm)
		}
	}

	if !prior.HasPrior {
		return nil
	}

	for arm, priorArm := range prior.Arms {
		if !priorArm.Transferred {
			continue
		}

		payload := priorArm.UCB1
		stats := b.Arms[arm]

		stats.PriorObservationWeight = payload.ObservationWeight
		stats.PriorRewardSum = payload.RewardSum
	}

	b.PriorDonorFunctionName = prior.DonorFunctionName

	return nil
}

func (b *UCB1Bandit) validateFreshTargetForWeakPriorLocked() error {
	if b.PriorDonorFunctionName != "" {
		return fmt.Errorf("target UCB1 policy already has a weak prior from donor %q", b.PriorDonorFunctionName)
	}

	if b.TotalCounts != 0 || b.TotalInFlight != 0 {
		return fmt.Errorf("weak prior can only be applied to a fresh UCB1 target")
	}

	for arm, stats := range b.Arms {
		if stats == nil {
			return fmt.Errorf("target UCB1 arm %q has nil state", arm)
		}

		if stats.Count != 0 ||
			stats.SumRewards != 0.0 ||
			stats.AvgReward != 0.0 ||
			stats.InFlight != 0 ||
			stats.RealCount != 0 ||
			stats.RealSumRewards != 0.0 ||
			stats.RealAvgReward != 0.0 ||
			stats.SyntheticCount != 0 ||
			stats.PriorObservationWeight != 0.0 ||
			stats.PriorRewardSum != 0.0 {

			return fmt.Errorf(
				"weak prior can only be applied to a fresh UCB1 target: arm %q already contains learning state",
				arm,
			)
		}
	}

	return nil
}

func (p *LinUCBDisjointPolicy) applyWeakPrior(prior WeakMABPrior) error {

	p.mu.Lock()
	defer p.mu.Unlock()

	if err := p.validateFreshTargetForWeakPriorLocked(); err != nil {
		return err
	}

	for arm, priorArm := range prior.Arms {
		if !priorArm.Transferred {
			continue
		}

		if _, exists := p.Arms[arm]; !exists {
			return fmt.Errorf("weak prior arm %q is not initialized in target LinUCB policy", arm)
		}

		if priorArm.LinUCB.Dim != p.Dim {
			return fmt.Errorf("weak prior arm %q dimension %d does not match target LinUCB dimension %d", arm, priorArm.LinUCB.Dim, p.Dim)
		}
	}

	if !prior.HasPrior {
		return nil
	}

	for arm, priorArm := range prior.Arms {
		if !priorArm.Transferred {
			continue
		}

		payload := priorArm.LinUCB
		state := p.Arms[arm]

		for i := 0; i < p.Dim; i++ {
			for j := 0; j < p.Dim; j++ {
				value := payload.AContribution[i][j]
				state.A.Set(i, j, state.A.At(i, j)+value)
				state.PriorAContribution.Set(i, j, value)
			}

			value := payload.BContribution[i]
			state.b.SetVec(i, state.b.AtVec(i)+value)
			state.PriorBContribution.SetVec(i, value)
		}

		state.PriorObservationWeight = payload.ObservationWeight
	}

	p.PriorDonorFunctionName = prior.DonorFunctionName

	return nil
}

func (p *LinUCBDisjointPolicy) validateFreshTargetForWeakPriorLocked() error {
	if p.PriorDonorFunctionName != "" {
		return fmt.Errorf("target LinUCB policy already has a weak prior from donor %q", p.PriorDonorFunctionName)
	}

	if p.TotalInFlight != 0 {
		return fmt.Errorf("weak prior can only be applied to a fresh LinUCB target")
	}

	for arm, state := range p.Arms {
		if state == nil ||
			state.A == nil ||
			state.b == nil ||
			state.RealAContribution == nil ||
			state.RealBContribution == nil ||
			state.PriorAContribution == nil ||
			state.PriorBContribution == nil {

			return fmt.Errorf("target LinUCB arm %q has incomplete state", arm)
		}

		if state.InFlight != 0 ||
			state.RealObservationCount != 0 ||
			state.SyntheticObservationCount != 0 ||
			state.PriorObservationWeight != 0.0 ||
			!denseIsIdentity(state.A, p.Dim) ||
			!vectorIsZero(state.b, p.Dim) ||
			!denseIsZero(state.RealAContribution, p.Dim, p.Dim) ||
			!vectorIsZero(state.RealBContribution, p.Dim) ||
			!denseIsZero(state.PriorAContribution, p.Dim, p.Dim) ||
			!vectorIsZero(state.PriorBContribution, p.Dim) {

			return fmt.Errorf(
				"weak prior can only be applied to a fresh LinUCB target: arm %q already contains learning state",
				arm,
			)
		}
	}

	return nil
}

func validateWeakMABPriorForApplication(prior WeakMABPrior) error {

	if prior.SchemaVersion != WeakMABPriorSchemaVersion {
		return fmt.Errorf("unsupported weak MAB prior schema version %d", prior.SchemaVersion)
	}

	if prior.DonorFunctionName == "" {
		return fmt.Errorf("weak prior donor function name cannot be empty")
	}

	if err := validateWeakMABPriorConfig(prior.Config); err != nil {
		return err
	}

	switch prior.Policy {
	case UCB1, LinUCB:
		// Supported.
	default:
		return fmt.Errorf("unsupported weak prior policy %q", prior.Policy)
	}

	if prior.SourceRealObservationCount < 0 || prior.SourceExcludedSyntheticObservationCount < 0 {
		return fmt.Errorf("weak prior source observation summaries cannot be negative")
	}

	if prior.ArmCount != len(prior.Arms) {
		return fmt.Errorf("weak prior arm summary mismatch: arms=%d summary=%d", len(prior.Arms), prior.ArmCount)
	}

	var totalReal int64
	var totalSynthetic int64
	transferred := 0
	skipped := 0

	for arm, priorArm := range prior.Arms {
		if arm == "" {
			return fmt.Errorf("weak prior arm name cannot be empty")
		}

		if priorArm.SourceRealObservationCount < 0 || priorArm.SourceExcludedSyntheticObservationCount < 0 {
			return fmt.Errorf("weak prior arm %q has negative source observation counts", arm)
		}

		totalReal += priorArm.SourceRealObservationCount
		totalSynthetic += priorArm.SourceExcludedSyntheticObservationCount

		if priorArm.Transferred {
			transferred++
			if err := validateTransferredWeakPriorArm(arm, prior.Policy, prior.Config, priorArm); err != nil {
				return err
			}

			continue
		}

		skipped++

		if err := validateSkippedWeakPriorArm(arm, prior.Config, priorArm); err != nil {
			return err
		}
	}

	if totalReal != prior.SourceRealObservationCount {
		return fmt.Errorf("weak prior source real observation summary mismatch: arms=%d summary=%d", totalReal, prior.SourceRealObservationCount)
	}

	if totalSynthetic != prior.SourceExcludedSyntheticObservationCount {
		return fmt.Errorf("weak prior source synthetic observation summary mismatch: arms=%d summary=%d", totalSynthetic, prior.SourceExcludedSyntheticObservationCount)
	}

	if transferred != prior.TransferredArmCount || skipped != prior.SkippedArmCount || transferred+skipped != prior.ArmCount {
		return fmt.Errorf("weak prior transferred/skipped arm summary mismatch")
	}

	if prior.HasPrior != (transferred > 0) {
		return fmt.Errorf("weak prior has_prior does not match transferred arm count")
	}

	return nil
}

func validateTransferredWeakPriorArm(arm string, policy BanditType, config WeakMABPriorConfig, priorArm WeakMABArmPrior) error {
	if priorArm.SkipReason != "" {
		return fmt.Errorf("transferred weak prior arm %q cannot have a skip reason", arm)
	}

	weight := priorArm.AppliedEquivalentObservationWeight
	if !isFiniteNumber(weight) || weight <= 0.0 || weight > config.EquivalentObservationWeight || weight > float64(priorArm.SourceRealObservationCount) {
		return fmt.Errorf("weak prior arm %q has invalid applied observation weight", arm)
	}

	if priorArm.SourceRealObservationCount < config.MinRealObservationsPerArm {
		return fmt.Errorf("weak prior arm %q was transferred without sufficient real evidence", arm)
	}

	expectedScale := weight / float64(priorArm.SourceRealObservationCount)
	if !isFiniteNumber(priorArm.AttenuationScale) || !weakPriorAlmostEqual(expectedScale, priorArm.AttenuationScale) {
		return fmt.Errorf("weak prior arm %q has inconsistent attenuation scale", arm)
	}

	switch policy {
	case UCB1:
		if priorArm.UCB1 == nil || priorArm.LinUCB != nil {
			return fmt.Errorf("weak prior arm %q has invalid UCB1 payload", arm)
		}

		payload := priorArm.UCB1
		if !isFiniteNumber(payload.ObservationWeight) || !isFiniteNumber(payload.RewardSum) || !isFiniteNumber(payload.MeanReward) || !weakPriorAlmostEqual(payload.ObservationWeight, weight) {
			return fmt.Errorf("weak prior arm %q has invalid UCB1 statistics", arm)
		}

		expectedMean := payload.RewardSum / payload.ObservationWeight
		if !weakPriorAlmostEqual(expectedMean, payload.MeanReward) {
			return fmt.Errorf("weak prior arm %q has inconsistent UCB1 mean reward", arm)
		}

	case LinUCB:
		if priorArm.LinUCB == nil || priorArm.UCB1 != nil {
			return fmt.Errorf("weak prior arm %q has invalid LinUCB payload", arm)
		}

		payload := priorArm.LinUCB
		if payload.Dim <= 0 ||
			!isFiniteNumber(payload.ObservationWeight) || !weakPriorAlmostEqual(payload.ObservationWeight, weight) {
			return fmt.Errorf("weak prior arm %q has invalid LinUCB metadata", arm)
		}

		if len(payload.AContribution) != payload.Dim || len(payload.BContribution) != payload.Dim {
			return fmt.Errorf("weak prior arm %q has invalid LinUCB dimensions", arm)
		}

		for i := 0; i < payload.Dim; i++ {
			if len(payload.AContribution[i]) != payload.Dim || !isFiniteNumber(payload.BContribution[i]) {
				return fmt.Errorf("weak prior arm %q has invalid LinUCB dimensions", arm)
			}

			for j := 0; j < payload.Dim; j++ {
				if !isFiniteNumber(payload.AContribution[i][j]) {
					return fmt.Errorf("weak prior arm %q has non-finite LinUCB contribution", arm)
				}
			}
		}
	}

	return nil
}

func validateSkippedWeakPriorArm(arm string, config WeakMABPriorConfig, priorArm WeakMABArmPrior) error {
	if priorArm.UCB1 != nil || priorArm.LinUCB != nil || priorArm.AppliedEquivalentObservationWeight != 0.0 || priorArm.AttenuationScale != 0.0 {
		return fmt.Errorf("skipped weak prior arm %q contains transferable payload", arm)
	}

	switch priorArm.SkipReason {
	case WeakPriorSkipNoRealObservations:
		if priorArm.SourceRealObservationCount != 0 {
			return fmt.Errorf("weak prior arm %q has inconsistent no-real-observations skip reason", arm)
		}

	case WeakPriorSkipInsufficientRealObservations:
		if priorArm.SourceRealObservationCount <= 0 || priorArm.SourceRealObservationCount >= config.MinRealObservationsPerArm {
			return fmt.Errorf("weak prior arm %q has inconsistent insufficient-evidence skip reason", arm)
		}

	default:
		return fmt.Errorf("skipped weak prior arm %q has invalid skip reason %q", arm, priorArm.SkipReason)
	}

	return nil
}

func denseIsIdentity(matrix *mat.Dense, dim int) bool {

	rows, cols := matrix.Dims()
	if rows != dim || cols != dim {
		return false
	}

	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}

			if math.Abs(matrix.At(i, j)-expected) > 1e-12 {
				return false
			}
		}
	}

	return true
}

func denseIsZero(matrix *mat.Dense, rows int, cols int) bool {

	actualRows, actualCols := matrix.Dims()
	if actualRows != rows || actualCols != cols {
		return false
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(matrix.At(i, j)) > 1e-12 {
				return false
			}
		}
	}

	return true
}

func vectorIsZero(vector *mat.VecDense, dim int) bool {

	if vector.Len() != dim {
		return false
	}

	for i := 0; i < dim; i++ {
		if math.Abs(vector.AtVec(i)) > 1e-12 {
			return false
		}
	}

	return true
}
