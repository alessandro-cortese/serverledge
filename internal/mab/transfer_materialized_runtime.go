package mab

import "fmt"

// InitializeTargetFromMaterializedPrior atomically creates a fresh target MAB
// and applies an already-materialized weak prior.
//
// Unlike InitializeTargetFromDonor, this method does not require the donor to
// exist in the current BanditManager. This is intentionally useful for
// controlled experiments where donor knowledge is collected in one runtime,
// materialized as a WeakMABPrior, and then replayed into a fresh runtime with
// different target-side MAB parameters.
//
// In particular, this allows experiments where:
//  1. donor knowledge is collected with UCB1 c=0.8;
//  2. the resulting weak prior is frozen;
//  3. fresh target runtimes receive exactly that prior with c=0.8 or c=0.
//
// The target is published only after successful prior application.
func (bm *BanditManager) InitializeTargetFromMaterializedPrior(
	targetFunctionName string,
	prior WeakMABPrior,
) (RuntimeTransferResult, error) {

	if bm == nil {
		return RuntimeTransferResult{},
			fmt.Errorf("bandit manager cannot be nil")
	}

	if targetFunctionName == "" {
		return RuntimeTransferResult{},
			fmt.Errorf("target function name cannot be empty")
	}

	if prior.DonorFunctionName == "" {
		return RuntimeTransferResult{},
			fmt.Errorf("materialized weak prior donor function name cannot be empty")
	}

	if targetFunctionName == prior.DonorFunctionName {
		return RuntimeTransferResult{},
			fmt.Errorf("donor and target function must be different")
	}

	if !prior.HasPrior {
		return RuntimeTransferResult{},
			fmt.Errorf("materialized weak prior does not contain transferable knowledge")
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()

	if _, exists := bm.bandits[targetFunctionName]; exists {
		return RuntimeTransferResult{},
			fmt.Errorf(
				"target function %q already has a MAB policy",
				targetFunctionName,
			)
	}

	target := bm.newBanditLocked(targetFunctionName)
	targetPolicy := target.GetType()

	if prior.Policy != targetPolicy {
		return RuntimeTransferResult{},
			fmt.Errorf(
				"materialized weak prior policy %q does not match target policy %q",
				prior.Policy,
				targetPolicy,
			)
	}

	if err := ApplyWeakMABPrior(target, prior); err != nil {
		return RuntimeTransferResult{},
			fmt.Errorf(
				"apply materialized weak prior to target %q: %w",
				targetFunctionName,
				err,
			)
	}

	result := RuntimeTransferResult{
		TargetFunctionName: targetFunctionName,
		DonorFunctionName:  prior.DonorFunctionName,
		Policy:             targetPolicy,
		Applied:            true,
		Reason:             RuntimeTransferReasonApplied,
		Prior:              prior,
	}

	// Publish only after the target has been fully initialized.
	bm.bandits[targetFunctionName] = target

	logMABRuntimeTransfer(result)

	return result, nil
}
