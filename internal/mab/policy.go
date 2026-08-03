package mab

import "sort"

type BanditType string

const (
	UCB1   BanditType = "UCB1"
	LinUCB BanditType = "LinUCB"
)

// Context carries the system-state snapshot captured at decision time.
// LinUCB uses it to build its feature vector. The context does not directly
// change the scalar reward returned by an execution.
type Context struct {
	// ArchMemUsage keeps the aggregate memory-utilization snapshot for each
	// MAB arm/machine-tag ring. The historical field name is retained for
	// compatibility with the previous implementation.
	ArchMemUsage map[string]float64

	// ArmCostFactor keeps the normalized cost factor for each arm/tag at decision time.
	// ArmCostFactor map[string]float64

	// ArmEnergyFactor keeps the normalized energy factor for each arm/tag at decision time.
	// ArmEnergyFactor map[string]float64
}

// ExecutionFeedback contains the measurements associated with the node that
// actually executed an invocation. Cost and energy are node-level structural
// properties and must not be replaced with averages computed over the arm/ring.
type ExecutionFeedback struct {
	DurationMs       float64
	ResponseTimeMs   float64
	InitTimeMs       float64
	QueueingTimeMs   float64
	OffloadLatencyMs float64

	IsWarmStart bool

	NodeName      string
	ExecutionNode string

	CostFactor   float64
	EnergyFactor float64
}

// SyntheticReward is a policy-level learning observation that is not derived
// from execution timing. It is used to penalize a selected arm that could not
// serve the request and therefore forced a load-balancer fallback.
type SyntheticReward struct {
	RequestID string
	Value     float64
	Reason    string
}

// Policy is the interface that any Bandit algorithm must implement.
type Policy interface {
	// SelectArm chooses the best arm among all arms known by the policy.
	SelectArm(ctx *Context) string

	// SelectArmFrom chooses the best arm only among allowedArms.
	//
	// Semantics:
	//   - nil means that every known arm is allowed;
	//   - a non-nil empty slice means that no arm is allowed;
	//   - otherwise only the listed known arms can be selected.
	SelectArmFrom(ctx *Context, allowedArms []string) string

	// UpdateReward updates the internal model of the policy based on the feedback.
	// It requires the context that was present when the decision was made (if the MAB has a context).
	UpdateReward(arm string, ctx *Context, feedback ExecutionFeedback)

	// InitArm initializes a new arm before it is used. So it will be easier to implement more than 2 arms for new architectures.
	InitArm(arm string)

	// ResolveSelection atomically closes one pending selection and optionally
	// applies its execution feedback. selectedArm is the arm returned by the MAB;
	// executionArm is the arm that actually executed the request and can differ
	// after a load-balancer fallback. A nil feedback releases the pending
	// selection without applying real execution feedback. selectedArmReward, when
	// present, is applied to the selected arm before processing the real feedback.
	ResolveSelection(selectedArm string, executionArm string, ctx *Context, feedback *ExecutionFeedback, selectedArmReward *SyntheticReward)

	// GetType returns the type of the bandit policy.
	GetType() BanditType
}

// filterAllowedArms returns the known arms that the policy is allowed to consider.
//
// Semantics:
//   - allowedArms == nil: no action mask, therefore all known arms are returned;
//   - allowedArms != nil: only known arms explicitly listed in allowedArms are returned;
//   - duplicates and unknown arms are ignored.
//
// When no mask is supplied, the returned list is sorted to keep experiments
// reproducible. When a mask is supplied, its order is preserved
func filterAllowedArms[T any](
	knownArms map[string]T,
	allowedArms []string,
) []string {
	if allowedArms == nil {
		arms := make([]string, 0, len(knownArms))

		for arm := range knownArms {
			arms = append(arms, arm)
		}

		sort.Strings(arms)
		return arms
	}

	arms := make([]string, 0, len(allowedArms))
	seen := make(map[string]struct{}, len(allowedArms))

	for _, arm := range allowedArms {
		if _, duplicate := seen[arm]; duplicate {
			continue
		}
		seen[arm] = struct{}{}

		if _, exists := knownArms[arm]; !exists {
			continue
		}

		arms = append(arms, arm)
	}

	return arms
}
