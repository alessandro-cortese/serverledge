package mab

type BanditType string

const (
	UCB1   BanditType = "UCB1"
	LinUCB BanditType = "LinUCB"
)

// Context carries the state of the system at the time of decision.
// Currently, it holds memory usage, but can be extended (i.e.: we could also add % of cpu load)
// It is used only by contextual MABs, obviously. The UCB1 doesn't need this since it works without context.
type Context struct {
	// ArmMemUsage keeps the memory utilization snapshot for each arm/tag.
	// The historical name is kept for compatibility with the existing LinUCB code,
	// but the key is now the MAB arm, i.e. usually a machine_tag/ring.
	ArchMemUsage map[string]float64

	// ArmCostFactor keeps the normalized cost factor for each arm/tag at decision time.
	ArmCostFactor map[string]float64

	// ArmEnergyFactor keeps the normalized energy factor for each arm/tag at decision time.
	ArmEnergyFactor map[string]float64
}

// Policy is the interface that any Bandit algorithm must implement.
type Policy interface {
	// SelectArm chooses the best arm based on the policy logic and optional context.
	SelectArm(ctx *Context) string

	// UpdateReward updates the internal model of the policy based on the feedback.
	// It requires the context that was present when the decision was made (if the MAB has a context).
	UpdateReward(arm string, ctx *Context, isWarmStart bool, durationMs float64)

	// InitArm initializes a new arm before it is used. So it will be easier to implement more than 2 arms for new architectures.
	InitArm(arm string)

	// GetType returns the type of the bandit policy.
	GetType() BanditType
}
