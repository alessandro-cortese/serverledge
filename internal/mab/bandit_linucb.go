package mab

import (
	"log"
	"math"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat" // for matrix operations
)

const linUCBUtilizationEpsilon = 0.01

// LinUCBDisjointPolicy implements LinUCB with one disjoint linear model for
// each MAB arm/machine-tag ring.
//
// The contextual feature is the aggregate memory utilization of the ring at
// decision time. Utilization is not subtracted from the scalar reward: the
// policy learns its relationship with execution performance through A and b.
//
// Reference: Li et al., "A Contextual-Bandit Approach to Personalized News
// Article Recommendation", Algorithm 1.
type LinUCBDisjointPolicy struct {
	FunctionName string
	Alpha        float64 // Exploration parameter

	// Maps each arm to its disjoint model state (A, b).
	Arms map[string]*LinUCBArmState
	mu   sync.RWMutex

	// Dimension of the feature vector (d)
	// Bias (1) + MemoryFeature (1) = 2
	Dim int
}

// LinUCBArmState holds the matrix A and vector b for a specific arm.
// A represents the design matrix (d x d)
// b represents the reward mapping (d x 1)
type LinUCBArmState struct {
	A *mat.Dense
	b *mat.VecDense
}

// NewLinUCBDisjointPolicy creates a new instance of the policy.
func NewLinUCBDisjointPolicy(functionName string, alpha float64) *LinUCBDisjointPolicy {
	return &LinUCBDisjointPolicy{
		FunctionName: functionName,
		Alpha:        alpha,
		Arms:         make(map[string]*LinUCBArmState),
		Dim:          2, // Currently: Bias + Memory Usage
	}
}

// InitArm initializes the matrices for a new architecture.
func (p *LinUCBDisjointPolicy) InitArm(arm string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.Arms[arm]; exists {
		return
	}

	// Initialize A as Identity Matrix (d x d) as per the paper
	A := mat.NewDense(p.Dim, p.Dim, nil)
	for i := 0; i < p.Dim; i++ {
		A.Set(i, i, 1.0)
	}

	// Initialize b as Zero Vector (d)
	b := mat.NewVecDense(p.Dim, nil)

	p.Arms[arm] = &LinUCBArmState{
		A: A,
		b: b,
	}

	logMABArmAdded(
		string(p.GetType()),
		arm,
		p.FunctionName,
		formatArmsFromMap(p.Arms),
	)
}

// SelectArm chooses among all arms known by the policy.
// It delegates to SelectArmFrom with no action mask.
func (p *LinUCBDisjointPolicy) SelectArm(ctx *Context) string {
	return p.SelectArmFrom(ctx, nil)
}

// SelectArmFrom calculates the LinUCB score only for the supplied action mask.
//
// allowedArms semantics:
//   - nil: consider every arm known by the policy;
//   - non-nil: consider only the listed, known arms.
func (p *LinUCBDisjointPolicy) SelectArmFrom(
	ctx *Context,
	allowedArms []string,
) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	candidateArms := filterAllowedArms(
		p.Arms,
		allowedArms,
	)

	if len(candidateArms) == 0 {
		log.Printf(
			"[MAB] event=no_allowed_arms ts=%d policy=%s function=%s\n",
			nowMillis(),
			string(p.GetType()),
			p.FunctionName,
		)

		return ""
	}

	bestArm := ""
	bestScore := -math.MaxFloat64

	for _, arm := range candidateArms {
		state := p.Arms[arm]

		// Missing context is interpreted as zero utilization.
		utilization := armUtilization(
			ctx,
			arm,
		)

		x := p.computeFeatures(utilization)

		var AInv mat.Dense

		if err := AInv.Inverse(state.A); err != nil {
			log.Printf(
				"[LinUCB] Error inverting matrix for arm %s: %v",
				arm,
				err,
			)

			return ""
		}

		// theta = A^-1 * b
		var theta mat.VecDense
		theta.MulVec(
			&AInv,
			state.b,
		)

		// Expected reward: x^T * theta
		expectedReward :=
			mat.Dot(
				x,
				&theta,
			)

		// Confidence: alpha * sqrt(x^T * A^-1 * x)
		var tempVec mat.VecDense
		tempVec.MulVec(
			&AInv,
			x,
		)

		variance :=
			mat.Dot(
				x,
				&tempVec,
			)

		confidence :=
			p.Alpha *
				math.Sqrt(variance)

		score :=
			expectedReward +
				confidence

		logMABContextualArmScore(
			string(p.GetType()),
			p.FunctionName,
			arm,
			score,
			expectedReward,
			confidence,
			utilization,
		)

		if score > bestScore {
			bestScore = score
			bestArm = arm
		}
	}

	if bestArm == "" {
		log.Printf(
			"[MAB] event=no_arm_selected ts=%d policy=%s function=%s allowed_arms=%v\n",
			nowMillis(),
			string(p.GetType()),
			p.FunctionName,
			candidateArms,
		)

		return ""
	}

	logMABContextualSelectArm(
		string(p.GetType()),
		p.FunctionName,
		bestArm,
		"linucb_score",
		bestScore,
		strings.Join(candidateArms, ","),
	)

	return bestArm
}

// UpdateReward updates A and b for the chosen arm after a feedback sample
// accepted by the configured cold-start policy.
//
// The context must be the utilization snapshot captured when the decision was
// made. Utilization changes the feature vector used to update A and b, but it
// is not subtracted from the scalar reward. The latency component of the reward
// is based on DurationMs; any cost or energy terms remain controlled by their
// separate configured weights.
func (p *LinUCBDisjointPolicy) UpdateReward(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	p.mu.Lock()
	defer p.mu.Unlock()

	policy := string(p.GetType())

	state, ok := p.Arms[arm]

	if !ok {
		log.Printf("[LinUCB] Warning: Trying to update unknown arm %s", arm)
		panic(3)
	}

	if !validateExecutionFeedback(
		policy,
		p.FunctionName,
		arm,
		feedback,
	) {
		return
	}

	if !shouldUpdateRewardFromFeedback(
		policy,
		p.FunctionName,
		arm,
		feedback,
	) {
		return
	}

	if ctx == nil {
		log.Printf(
			"[LinUCB] Warning: Context is nil for arm %s",
			arm,
		)

		panic(4)
	}

	utilization :=
		armUtilization(
			ctx,
			arm,
		)

	latencyReward :=
		-math.Log(
			feedback.DurationMs,
		)

	breakdown :=
		buildCostBreakdown(
			latencyReward,
			feedback,
		)

	reward :=
		breakdown.FinalReward

	if !isFiniteNumber(reward) {
		recordInvalidExecutionFeedback(
			policy,
			p.FunctionName,
			arm,
			"non_finite_reward",
			feedback,
		)

		return
	}

	logMABRewardBreakdown(
		policy,
		p.FunctionName,
		arm,
		feedback.DurationMs,
		breakdown,
	)

	x :=
		p.computeFeatures(
			utilization,
		)

	// A = A + x * x^T
	var outerProduct mat.Dense

	outerProduct.Outer(
		1.0,
		x,
		x,
	)

	state.A.Add(
		state.A,
		&outerProduct,
	)

	// b = b + reward * x
	var scaledX mat.VecDense

	scaledX.ScaleVec(
		reward,
		x,
	)

	state.b.AddVec(
		state.b,
		&scaledX,
	)

	recordAcceptedExecutionFeedback(
		policy,
		p.FunctionName,
		arm,
		feedback,
	)

	logMABContextualUpdateReward(
		policy,
		p.FunctionName,
		arm,
		feedback.DurationMs,
		feedback.IsWarmStart,
		utilization,
		reward,
	)
}

// computeFeatures transforms raw context data into the feature vector
// [1, sigma(u)].
func (p *LinUCBDisjointPolicy) computeFeatures(utilization float64) *mat.VecDense {
	bias := 1.0

	// Legacy non-linear contextual feature from the previous thesis:
	// 1 / (1 - utilization + epsilon).
	//
	// It amplifies differences near saturation, while the learned model decides
	// whether that context is associated with better or worse rewards.
	u := clampUnitInterval(utilization)
	sigma := 1.0 / (1.0 - u + linUCBUtilizationEpsilon)

	return mat.NewVecDense(
		p.Dim,
		[]float64{
			bias,
			sigma,
		},
	)
}

func (p *LinUCBDisjointPolicy) GetType() BanditType {
	return LinUCB
}
