package mab

import (
	"log"
	"math"
	"strings"
	"sync"

	"github.com/serverledge-faas/serverledge/internal/config"
	"gonum.org/v1/gonum/mat" // for matrix operations
)

// LinUCBDisjointPolicy implements the LinUCB algorithm with disjoint linear models.
// Reference: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", Algorithm 1.
type LinUCBDisjointPolicy struct {
	FunctionName string  // TODO: Ask if I can do this
	Alpha        float64 // Exploration parameter

	// Maps each arm to its features (A, b)
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

	candidateArms := filterAllowedArms(p.Arms, allowedArms)

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

		// Construct the feature vector for this arm.
		// A missing context is interpreted as zero utilization.
		memUsage := 0.0

		if ctx != nil && ctx.ArchMemUsage != nil {
			if usage, ok := ctx.ArchMemUsage[arm]; ok {
				memUsage = usage
			}
		}

		x := p.computeFeatures(memUsage)

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
		theta.MulVec(&AInv, state.b)

		// Expected reward: x^T * theta
		expectedReward := mat.Dot(x, &theta)

		// Confidence term: alpha * sqrt(x^T * A^-1 * x)
		var tempVec mat.VecDense
		tempVec.MulVec(&AInv, x)

		variance := mat.Dot(x, &tempVec)
		confidence := p.Alpha * math.Sqrt(variance)

		score := expectedReward + confidence

		logMABContextualArmScore(
			string(p.GetType()),
			p.FunctionName,
			arm,
			score,
			expectedReward,
			confidence,
			memUsage,
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

// UpdateReward updates A and b for the chosen arm. Context is necessary to keep track of the memory usage AT THE MOMENT
// the decision was taken. So it has to be a "snapshot" of memory at that given time.
func (p *LinUCBDisjointPolicy) UpdateReward(arm string, ctx *Context, isWarmStart bool, durationMs float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !isWarmStart {
		logMABSkipColdStart(
			string(p.GetType()),
			p.FunctionName,
			arm,
			durationMs,
		)
		return // likely an outlier, skip update
	}

	state, ok := p.Arms[arm]
	if !ok {
		log.Printf("[LinUCB] Warning: Trying to update unknown arm %s", arm)
		panic(3) // should never happen if correctly used
	}

	// Reconstruct the feature vector x_t used at decision time.
	memUsage := 0.0
	if ctx != nil {
		memUsage = ctx.ArchMemUsage[arm]
	} else {
		log.Printf("[LinUCB] Warning: Context is nil for arm %s", arm)
		panic(4) // should never happen
	}

	if durationMs <= 0 {
		log.Printf(
			"[MAB] event=skip_invalid_reward ts=%d policy=%s function=%s arm=%s duration_ms=%.6f reason=non_positive_duration\n",
			nowMillis(),
			string(p.GetType()),
			p.FunctionName,
			arm,
			durationMs,
		)
		return
	}

	lambda := config.GetFloat(config.MAB_LINUCB_LAMBDA, 0.0)
	memoryPenalty := memPenalty(memUsage)
	latencyReward := -math.Log(durationMs)

	breakdown := buildCostBreakdown(
		arm,
		latencyReward,
		ctx,
		memoryPenalty,
		lambda,
	)
	reward := breakdown.FinalReward

	logMABRewardBreakdown(
		string(p.GetType()),
		p.FunctionName,
		arm,
		durationMs,
		breakdown,
	)

	x := p.computeFeatures(memUsage)

	// Update A: A = A + x * x^T
	var outerProduct mat.Dense
	outerProduct.Outer(1.0, x, x)
	state.A.Add(state.A, &outerProduct)

	// Update b: b = b + reward * x
	var scaledX mat.VecDense
	scaledX.ScaleVec(reward, x)
	state.b.AddVec(state.b, &scaledX)

	logMABContextualUpdateReward(
		string(p.GetType()),
		p.FunctionName,
		arm,
		durationMs,
		isWarmStart,
		memUsage,
		lambda,
		reward,
	)
}

func memPenalty(memUsage float64) float64 {
	// Grows from 0 at 0.75 utilization to 1 at 1.0 utilization
	penalty := (memUsage - 0.75) / 0.25 // (memUsage - 0.75) / (1 - 0.75)
	return max(0.0, penalty)
}

// computeFeatures transforms raw context data into the feature vector [1, sigma(u)].
func (p *LinUCBDisjointPolicy) computeFeatures(memUsage float64) *mat.VecDense {
	// Bias term
	bias := 1.0

	// Non-linear penalty (sigma) as suggested: 1 / (1 - u + epsilon)
	// epsilon prevents division by zero if usage is 100%
	epsilon := 0.01
	sigma := 1.0 / (1.0 - memUsage + epsilon)

	return mat.NewVecDense(p.Dim, []float64{bias, sigma})
}

func (p *LinUCBDisjointPolicy) GetType() BanditType {
	return LinUCB
}
