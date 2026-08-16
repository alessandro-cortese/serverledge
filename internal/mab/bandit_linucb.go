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
	Arms          map[string]*LinUCBArmState
	TotalInFlight int64
	mu            sync.RWMutex

	// Dimension of the feature vector (d)
	// Bias (1) + MemoryFeature (1) = 2
	Dim int

	// PriorDonorFunctionName records the donor used to initialize this target.
	// The value is provenance only and does not change the LinUCB formula.
	PriorDonorFunctionName string
}

// LinUCBArmState holds the matrix A and vector b for a specific arm.
// A represents the design matrix (d x d)
// b represents the reward mapping (d x 1)
type LinUCBArmState struct {
	A        *mat.Dense
	b        *mat.VecDense
	InFlight int64

	// RealAContribution and RealBContribution contain only the additive
	// contribution produced by accepted execution feedback.
	//
	// They deliberately exclude:
	//   - the identity regularizer used to initialize A;
	//   - synthetic fallback penalties.
	//
	// A and b above remain the live LinUCB model and continue to include every
	// accepted learning observation exactly as before.
	RealAContribution    *mat.Dense
	RealBContribution    *mat.VecDense
	RealObservationCount int64

	// SyntheticObservationCount is diagnostic only.
	//
	// Synthetic fallback observations continue to update the live A and b but
	// are excluded from the transferable contextual knowledge.
	SyntheticObservationCount int64

	// Prior* contains donor-derived weak contextual evidence applied before the
	// target function starts learning. It contributes to the live A and b used
	// for selection, but it is deliberately excluded from Real* so that a
	// received prior can never be re-exported as target experience.
	PriorAContribution     *mat.Dense
	PriorBContribution     *mat.VecDense
	PriorObservationWeight float64
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

	realAContribution :=
		mat.NewDense(
			p.Dim,
			p.Dim,
			nil,
		)

	realBContribution :=
		mat.NewVecDense(
			p.Dim,
			nil,
		)

	priorAContribution :=
		mat.NewDense(
			p.Dim,
			p.Dim,
			nil,
		)

	priorBContribution :=
		mat.NewVecDense(
			p.Dim,
			nil,
		)

	p.Arms[arm] = &LinUCBArmState{
		A:                  A,
		b:                  b,
		RealAContribution:  realAContribution,
		RealBContribution:  realBContribution,
		PriorAContribution: priorAContribution,
		PriorBContribution: priorBContribution,
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
			state.InFlight,
			p.TotalInFlight,
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

	p.markSelectionLocked(
		bestArm,
	)

	logMABContextualSelectArm(
		string(p.GetType()),
		p.FunctionName,
		bestArm,
		"linucb_score",
		bestScore,
		p.TotalInFlight,
		strings.Join(
			candidateArms,
			",",
		),
	)

	return bestArm
}

// UpdateReward applies a standalone feedback sample. It does not resolve an
// in-flight selection; production request handling uses ResolveSelection.
func (p *LinUCBDisjointPolicy) UpdateReward(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.updateRewardLocked(
		arm,
		ctx,
		feedback,
	)
}

func (p *LinUCBDisjointPolicy) ResolveSelection(
	selectedArm string,
	executionArm string,
	ctx *Context,
	feedback *ExecutionFeedback,
	selectedArmReward *SyntheticReward,
) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.completeSelectionLocked(
		selectedArm,
	) {
		return
	}

	if selectedArmReward != nil {
		p.updateSyntheticRewardLocked(
			selectedArm,
			ctx,
			*selectedArmReward,
		)
	}

	if feedback == nil ||
		executionArm == "" {

		return
	}

	p.updateRewardLocked(
		executionArm,
		ctx,
		*feedback,
	)
}

func (p *LinUCBDisjointPolicy) updateSyntheticRewardLocked(
	arm string,
	ctx *Context,
	synthetic SyntheticReward,
) {
	state, ok :=
		p.Arms[arm]

	if !ok {
		log.Printf(
			"[MAB] event=unknown_synthetic_reward_arm policy=%s function=%s arm=%s reason=%s\n",
			string(p.GetType()),
			p.FunctionName,
			arm,
			synthetic.Reason,
		)

		return
	}

	if ctx == nil {
		log.Printf(
			"[MAB] event=invalid_synthetic_reward policy=%s function=%s arm=%s reason=missing_decision_context synthetic_reason=%s\n",
			string(p.GetType()),
			p.FunctionName,
			arm,
			synthetic.Reason,
		)

		return
	}

	if !isFiniteNumber(
		synthetic.Value,
	) {
		log.Printf(
			"[MAB] event=invalid_synthetic_reward policy=%s function=%s arm=%s reward=%f reason=%s\n",
			string(p.GetType()),
			p.FunctionName,
			arm,
			synthetic.Value,
			synthetic.Reason,
		)

		return
	}

	utilization :=
		armUtilization(
			ctx,
			arm,
		)

	x :=
		p.computeFeatures(
			utilization,
		)

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

	var scaledX mat.VecDense

	scaledX.ScaleVec(
		synthetic.Value,
		x,
	)

	state.b.AddVec(
		state.b,
		&scaledX,
	)

	state.SyntheticObservationCount++

	logMABSyntheticReward(
		string(p.GetType()),
		p.FunctionName,
		arm,
		synthetic,
		utilization,
		0,
		0.0,
		0,
	)
}

func (p *LinUCBDisjointPolicy) markSelectionLocked(
	arm string,
) {
	state, ok :=
		p.Arms[arm]

	if !ok {
		return
	}

	state.InFlight++
	p.TotalInFlight++

	logMABInFlightChanged(
		string(p.GetType()),
		p.FunctionName,
		arm,
		"started",
		state.InFlight,
		p.TotalInFlight,
	)
}

func (p *LinUCBDisjointPolicy) completeSelectionLocked(
	arm string,
) bool {
	state, ok :=
		p.Arms[arm]

	if !ok ||
		state.InFlight <= 0 {

		logMABInFlightIgnored(
			string(p.GetType()),
			p.FunctionName,
			arm,
			"no_pending_selection",
		)

		return false
	}

	state.InFlight--

	if p.TotalInFlight > 0 {
		p.TotalInFlight--
	}

	logMABInFlightChanged(
		string(p.GetType()),
		p.FunctionName,
		arm,
		"resolved",
		state.InFlight,
		p.TotalInFlight,
	)

	return true
}

func (p *LinUCBDisjointPolicy) updateRewardLocked(
	arm string,
	ctx *Context,
	feedback ExecutionFeedback,
) {
	policy :=
		string(
			p.GetType(),
		)

	state, ok :=
		p.Arms[arm]

	if !ok {
		log.Printf(
			"[LinUCB] Warning: trying to update unknown arm %s",
			arm,
		)

		return
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
		recordInvalidExecutionFeedback(
			policy,
			p.FunctionName,
			arm,
			"missing_decision_context",
			feedback,
		)

		return
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

	if !isFiniteNumber(
		reward,
	) {
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

	var scaledX mat.VecDense

	scaledX.ScaleVec(
		reward,
		x,
	)

	state.b.AddVec(
		state.b,
		&scaledX,
	)

	state.RealAContribution.Add(
		state.RealAContribution,
		&outerProduct,
	)

	state.RealBContribution.AddVec(
		state.RealBContribution,
		&scaledX,
	)

	state.RealObservationCount++

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
