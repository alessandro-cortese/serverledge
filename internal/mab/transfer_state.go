package mab

const TransferableMABKnowledgeSchemaVersion = 1

// TransferableMABKnowledge is a read-only snapshot of the learning state that
// can be considered for transfer to another function.
//
// In-flight selections and synthetic fallback rewards are deliberately excluded
// from the transferable statistics. The number of excluded synthetic
// observations is retained only for diagnostics and provenance.
type TransferableMABKnowledge struct {
	SchemaVersion int `json:"schema_version"`

	FunctionName string     `json:"function_name"`
	Policy       BanditType `json:"policy"`

	HasRealKnowledge bool `json:"has_real_knowledge"`

	RealObservationCount int64 `json:"real_observation_count"`

	ExcludedSyntheticObservationCount int64 `json:"excluded_synthetic_observation_count"`

	Arms map[string]TransferableArmKnowledge `json:"arms"`
}

// TransferableArmKnowledge contains the transferable state for one MAB arm.
//
// Exactly one between UCB1 and LinUCB is populated according to the policy
// associated with the function.
type TransferableArmKnowledge struct {
	RealObservationCount int64 `json:"real_observation_count"`

	ExcludedSyntheticObservationCount int64 `json:"excluded_synthetic_observation_count"`

	UCB1 *TransferableUCB1ArmKnowledge `json:"ucb1,omitempty"`

	LinUCB *TransferableLinUCBArmKnowledge `json:"linucb,omitempty"`
}

// TransferableUCB1ArmKnowledge contains only reward statistics produced by
// accepted real execution feedback.
type TransferableUCB1ArmKnowledge struct {
	RealSumRewards float64 `json:"real_sum_rewards"`
	RealAvgReward  float64 `json:"real_avg_reward"`
}

// TransferableLinUCBArmKnowledge contains only the contribution produced by
// accepted real execution feedback.
//
// AContribution therefore excludes both:
//
//   - the identity regularizer;
//   - synthetic fallback penalties.
type TransferableLinUCBArmKnowledge struct {
	Dim int `json:"dim"`

	AContribution [][]float64 `json:"a_contribution"`

	BContribution []float64 `json:"b_contribution"`
}

// TransferableKnowledge returns a deep read-only snapshot of the real-feedback
// knowledge accumulated by UCB1.
func (b *UCB1Bandit) TransferableKnowledge() TransferableMABKnowledge {
	b.mu.RLock()
	defer b.mu.RUnlock()

	arms :=
		make(
			map[string]TransferableArmKnowledge,
			len(b.Arms),
		)

	var totalReal int64
	var totalSynthetic int64

	for arm, stats := range b.Arms {

		totalReal +=
			stats.RealCount

		totalSynthetic +=
			stats.SyntheticCount

		arms[arm] =
			TransferableArmKnowledge{
				RealObservationCount: stats.RealCount,

				ExcludedSyntheticObservationCount: stats.SyntheticCount,

				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: stats.RealSumRewards,

					RealAvgReward: stats.RealAvgReward,
				},
			}
	}

	return TransferableMABKnowledge{
		SchemaVersion: TransferableMABKnowledgeSchemaVersion,

		FunctionName: b.FunctionName,

		Policy: UCB1,

		HasRealKnowledge: totalReal > 0,

		RealObservationCount: totalReal,

		ExcludedSyntheticObservationCount: totalSynthetic,

		Arms: arms,
	}
}

// TransferableKnowledge returns a deep read-only snapshot of the real-feedback
// contextual knowledge accumulated by LinUCB.
func (p *LinUCBDisjointPolicy) TransferableKnowledge() TransferableMABKnowledge {
	p.mu.RLock()
	defer p.mu.RUnlock()

	arms :=
		make(
			map[string]TransferableArmKnowledge,
			len(p.Arms),
		)

	var totalReal int64
	var totalSynthetic int64

	for arm, state := range p.Arms {

		totalReal +=
			state.RealObservationCount

		totalSynthetic +=
			state.SyntheticObservationCount

		arms[arm] =
			TransferableArmKnowledge{
				RealObservationCount: state.RealObservationCount,

				ExcludedSyntheticObservationCount: state.SyntheticObservationCount,

				LinUCB: &TransferableLinUCBArmKnowledge{
					Dim: p.Dim,

					AContribution: denseToNestedSlice(
						state.RealAContribution,
						p.Dim,
						p.Dim,
					),

					BContribution: vectorToSlice(
						state.RealBContribution,
						p.Dim,
					),
				},
			}
	}

	return TransferableMABKnowledge{
		SchemaVersion: TransferableMABKnowledgeSchemaVersion,

		FunctionName: p.FunctionName,

		Policy: LinUCB,

		HasRealKnowledge: totalReal > 0,

		RealObservationCount: totalReal,

		ExcludedSyntheticObservationCount: totalSynthetic,

		Arms: arms,
	}
}

func denseToNestedSlice(
	matrix interface {
		At(
			int,
			int,
		) float64
	},
	rows int,
	cols int,
) [][]float64 {

	result :=
		make(
			[][]float64,
			rows,
		)

	for i := 0; i < rows; i++ {
		result[i] =
			make(
				[]float64,
				cols,
			)

		for j := 0; j < cols; j++ {
			result[i][j] =
				matrix.At(
					i,
					j,
				)
		}
	}

	return result
}

func vectorToSlice(
	vector interface {
		AtVec(
			int,
		) float64
	},
	dim int,
) []float64 {

	result :=
		make(
			[]float64,
			dim,
		)

	for i := 0; i < dim; i++ {
		result[i] =
			vector.AtVec(
				i,
			)
	}

	return result
}

// SnapshotTransferableKnowledge returns the transferable state associated with
// an already existing function bandit.
//
// Unlike GetBandit, this method deliberately does not create a new policy when
// functionName is unknown.
func (bm *BanditManager) SnapshotTransferableKnowledge(
	functionName string,
) (
	TransferableMABKnowledge,
	bool,
) {
	bm.mu.RLock()

	bandit, exists :=
		bm.bandits[functionName]

	bm.mu.RUnlock()

	if !exists {
		return TransferableMABKnowledge{},
			false
	}

	return transferableKnowledgeFromPolicy(
		bandit,
	)
}
