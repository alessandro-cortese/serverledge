package mab

import "fmt"

const (
	RuntimeTransferReasonApplied             = "applied"
	RuntimeTransferReasonNoTransferablePrior = "no_transferable_prior"
)

// RuntimeTransferResult describes the result of initializing one target MAB
// from an already-selected donor.
//
// Donor selection itself remains outside this package: 10C.1 receives the
// donor identity chosen by the similarity pipeline and connects the existing
// 10B stages at runtime.
type RuntimeTransferResult struct {
	TargetFunctionName string
	DonorFunctionName  string
	Policy             BanditType
	Applied            bool
	Reason             string
	Prior              WeakMABPrior
}

// InitializeTargetFromDonor atomically creates a new target bandit and, when
// the donor contains enough transferable real evidence, initializes it with a
// weak prior.
//
// The target is not published in BanditManager until prior construction and
// application have completed. This prevents request handling from observing a
// partially initialized target.
//
// The donor must already exist in the manager because transferable knowledge
// is currently extracted from live MAB state. Persisted donor knowledge and
// automatic ingestion of the 10A donor-selection artifact are intentionally
// left to the next runtime-integration step.
func (bm *BanditManager) InitializeTargetFromDonor(targetFunctionName string, donorFunctionName string, priorConfig WeakMABPriorConfig) (RuntimeTransferResult, error) {

	if bm == nil {
		return RuntimeTransferResult{}, fmt.Errorf("bandit manager cannot be nil")
	}

	if targetFunctionName == "" {
		return RuntimeTransferResult{}, fmt.Errorf("target function name cannot be empty")
	}

	if donorFunctionName == "" {
		return RuntimeTransferResult{}, fmt.Errorf("donor function name cannot be empty")
	}

	if targetFunctionName == donorFunctionName {
		return RuntimeTransferResult{}, fmt.Errorf("donor and target function must be different")
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()

	if _, exists := bm.bandits[targetFunctionName]; exists {
		return RuntimeTransferResult{}, fmt.Errorf("target function %q already has a MAB policy", targetFunctionName)
	}

	donor, exists := bm.bandits[donorFunctionName]
	if !exists {
		return RuntimeTransferResult{},
			fmt.Errorf("donor function %q does not have a live MAB policy", donorFunctionName)
	}

	source, ok := transferableKnowledgeFromPolicy(donor)
	if !ok {
		return RuntimeTransferResult{}, fmt.Errorf("donor function %q uses an unsupported MAB policy type %T", donorFunctionName, donor)
	}

	prior, err := BuildWeakMABPrior(source, priorConfig)
	if err != nil {
		return RuntimeTransferResult{}, fmt.Errorf("build weak prior from donor %q: %w", donorFunctionName, err)
	}

	target := bm.newBanditLocked(targetFunctionName)
	if target.GetType() != source.Policy {
		return RuntimeTransferResult{}, fmt.Errorf("runtime transfer policy mismatch: donor=%s target=%s", source.Policy, target.GetType())
	}

	result := RuntimeTransferResult{
		TargetFunctionName: targetFunctionName,
		DonorFunctionName:  donorFunctionName,
		Policy:             source.Policy,
		Applied:            false,
		Reason:             RuntimeTransferReasonNoTransferablePrior,
		Prior:              prior,
	}

	if prior.HasPrior {
		if err := ApplyWeakMABPrior(target, prior); err != nil {
			return RuntimeTransferResult{}, fmt.Errorf("apply weak prior to target %q: %w", targetFunctionName, err)
		}

		result.Applied = true
		result.Reason = RuntimeTransferReasonApplied
	}

	// Publish only after the target is fully initialized.
	//
	// If no transferable prior was available, the ordinary fresh policy is
	// still published so that the function can immediately continue with
	// normal online learning from zero.
	bm.bandits[targetFunctionName] = target
	logMABRuntimeTransfer(result)

	return result, nil
}

func transferableKnowledgeFromPolicy(policy Policy) (TransferableMABKnowledge, bool) {
	switch typed := policy.(type) {
	case *UCB1Bandit:
		return typed.TransferableKnowledge(), true

	case *LinUCBDisjointPolicy:
		return typed.TransferableKnowledge(), true

	default:
		return TransferableMABKnowledge{}, false
	}
}
