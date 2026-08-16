package mab

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyWeakUCB1PriorInfluencesSelectionWithoutCreatingRealExperience(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		0.5,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
			"arm64": {
				RealSumRewards: -4.0,
				RealAvgReward:  -1.0,
			},
		},
	)

	target := NewUCB1Bandit("target-ucb", 0.0)
	target.InitArm("x86")
	target.InitArm("arm64")

	require.NoError(t, ApplyWeakMABPrior(target, prior))

	assert.Equal(t, "donor-ucb", target.PriorDonorFunctionName)
	assert.Zero(t, target.TotalCounts)
	assert.Zero(t, target.Arms["x86"].Count)
	assert.Zero(t, target.Arms["arm64"].Count)

	assert.InDelta(
		t,
		0.5,
		target.Arms["x86"].PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-1.0,
		target.Arms["x86"].PriorRewardSum,
		1e-12,
	)

	assert.InDelta(
		t,
		0.5,
		target.Arms["arm64"].PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-0.5,
		target.Arms["arm64"].PriorRewardSum,
		1e-12,
	)

	snapshot := target.TransferableKnowledge()

	assert.False(t, snapshot.HasRealKnowledge)
	assert.Zero(t, snapshot.RealObservationCount)
	assert.Zero(t, snapshot.Arms["x86"].RealObservationCount)
	assert.Zero(t, snapshot.Arms["arm64"].RealObservationCount)

	selected := target.SelectArmFrom(
		nil,
		[]string{"x86", "arm64"},
	)

	assert.Equal(t, "arm64", selected)
}

func TestUCB1RealFeedbackDominatesSeparatelyFromAppliedPrior(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target := NewUCB1Bandit("target-ucb-real", 0.0)
	target.InitArm("x86")

	require.NoError(t, ApplyWeakMABPrior(target, prior))

	target.UpdateReward(
		"x86",
		nil,
		ExecutionFeedback{
			DurationMs:   10.0,
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	realReward := -math.Log(10.0)
	stats := target.Arms["x86"]

	assert.Equal(t, int64(1), stats.Count)
	assert.Equal(t, int64(1), stats.RealCount)

	assert.InDelta(
		t,
		realReward,
		stats.SumRewards,
		1e-9,
	)

	assert.InDelta(
		t,
		realReward,
		stats.RealSumRewards,
		1e-9,
	)

	assert.InDelta(
		t,
		1.0,
		stats.PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		stats.PriorRewardSum,
		1e-12,
	)

	target.mu.RLock()
	effectiveMean :=
		target.effectiveArmAverageRewardLocked(stats)
	target.mu.RUnlock()

	assert.InDelta(
		t,
		(-2.0+realReward)/2.0,
		effectiveMean,
		1e-9,
	)

	snapshot := target.TransferableKnowledge()

	assert.True(t, snapshot.HasRealKnowledge)
	assert.Equal(t, int64(1), snapshot.RealObservationCount)

	assert.InDelta(
		t,
		realReward,
		snapshot.Arms["x86"].UCB1.RealSumRewards,
		1e-9,
	)

	assert.InDelta(
		t,
		realReward,
		snapshot.Arms["x86"].UCB1.RealAvgReward,
		1e-9,
	)
}

func TestApplyWeakLinUCBPriorInfluencesSelectionWithoutCreatingRealExperience(
	t *testing.T,
) {
	prior := buildLinUCBApplicationPrior(
		t,
		2,
		map[string]TransferableLinUCBArmKnowledge{
			"x86": {
				Dim: 2,
				AContribution: [][]float64{
					{2.0, 0.0},
					{0.0, 2.0},
				},
				BContribution: []float64{0.5, 0.0},
			},
			"arm64": {
				Dim: 2,
				AContribution: [][]float64{
					{2.0, 0.0},
					{0.0, 2.0},
				},
				BContribution: []float64{2.0, 0.0},
			},
		},
	)

	target := NewLinUCBDisjointPolicy(
		"target-lin",
		0.0,
	)

	target.InitArm("x86")
	target.InitArm("arm64")

	require.NoError(
		t,
		ApplyWeakMABPrior(target, prior),
	)

	assert.Equal(
		t,
		"donor-lin",
		target.PriorDonorFunctionName,
	)

	assert.InDelta(
		t,
		1.0,
		target.Arms["x86"].PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		1.0,
		target.Arms["arm64"].PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		2.0,
		target.Arms["x86"].A.At(0, 0),
		1e-12,
	)

	assert.InDelta(
		t,
		0.25,
		target.Arms["x86"].b.AtVec(0),
		1e-12,
	)

	assert.InDelta(
		t,
		2.0,
		target.Arms["arm64"].A.At(0, 0),
		1e-12,
	)

	assert.InDelta(
		t,
		1.0,
		target.Arms["arm64"].b.AtVec(0),
		1e-12,
	)

	snapshot := target.TransferableKnowledge()

	assert.False(t, snapshot.HasRealKnowledge)
	assert.Zero(t, snapshot.RealObservationCount)

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"x86":   0.0,
			"arm64": 0.0,
		},
	}

	selected := target.SelectArmFrom(
		ctx,
		[]string{"x86", "arm64"},
	)

	assert.Equal(t, "arm64", selected)
}

func TestLinUCBRealFeedbackRemainsOnlyTransferableKnowledgeAfterPrior(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	prior := buildLinUCBApplicationPrior(
		t,
		2,
		map[string]TransferableLinUCBArmKnowledge{
			"x86": {
				Dim: 2,
				AContribution: [][]float64{
					{2.0, 0.0},
					{0.0, 2.0},
				},
				BContribution: []float64{2.0, 1.0},
			},
		},
	)

	target := NewLinUCBDisjointPolicy(
		"target-lin-real",
		0.0,
	)

	target.InitArm("x86")

	require.NoError(
		t,
		ApplyWeakMABPrior(target, prior),
	)

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"x86": 0.25,
		},
	}

	target.UpdateReward(
		"x86",
		ctx,
		ExecutionFeedback{
			DurationMs:   10.0,
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	snapshot := target.TransferableKnowledge()
	arm := snapshot.Arms["x86"]

	assert.True(t, snapshot.HasRealKnowledge)
	assert.Equal(t, int64(1), snapshot.RealObservationCount)
	assert.Equal(t, int64(1), arm.RealObservationCount)

	features :=
		target.computeFeatures(0.25)

	reward :=
		-math.Log(10.0)

	for i := 0; i < target.Dim; i++ {
		assert.InDelta(
			t,
			reward*features.AtVec(i),
			arm.LinUCB.BContribution[i],
			1e-9,
		)

		for j := 0; j < target.Dim; j++ {
			assert.InDelta(
				t,
				features.AtVec(i)*
					features.AtVec(j),
				arm.LinUCB.AContribution[i][j],
				1e-9,
			)
		}
	}
}

func TestApplyWeakPriorRejectsPolicyMismatch(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target :=
		NewLinUCBDisjointPolicy(
			"target-policy-mismatch",
			0.0,
		)

	target.InitArm("x86")

	assert.Error(
		t,
		ApplyWeakMABPrior(target, prior),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.True(
		t,
		denseIsIdentity(
			target.Arms["x86"].A,
			target.Dim,
		),
	)
}

func TestApplyWeakPriorRejectsNonFreshTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target :=
		NewUCB1Bandit(
			"target-non-fresh",
			0.0,
		)

	target.InitArm("x86")

	target.UpdateReward(
		"x86",
		nil,
		ExecutionFeedback{
			DurationMs:   10.0,
			IsWarmStart:  true,
			CostFactor:   1.0,
			EnergyFactor: 1.0,
		},
	)

	assert.Error(
		t,
		ApplyWeakMABPrior(target, prior),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.Arms["x86"].
			PriorObservationWeight,
	)
}

func TestApplyWeakPriorRejectsUnknownTransferredArmAtomically(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target :=
		NewUCB1Bandit(
			"target-missing-arm",
			0.0,
		)

	target.InitArm("arm64")

	assert.Error(
		t,
		ApplyWeakMABPrior(target, prior),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.Arms["arm64"].
			PriorObservationWeight,
	)

	assert.Zero(
		t,
		target.Arms["arm64"].
			PriorRewardSum,
	)
}

func TestApplyWeakLinUCBPriorRejectsDimensionMismatch(
	t *testing.T,
) {
	prior := buildLinUCBApplicationPrior(
		t,
		3,
		map[string]TransferableLinUCBArmKnowledge{
			"x86": {
				Dim: 3,
				AContribution: [][]float64{
					{2.0, 0.0, 0.0},
					{0.0, 2.0, 0.0},
					{0.0, 0.0, 2.0},
				},
				BContribution: []float64{
					1.0,
					0.0,
					0.0,
				},
			},
		},
	)

	target :=
		NewLinUCBDisjointPolicy(
			"target-dim-mismatch",
			0.0,
		)

	target.InitArm("x86")

	assert.Error(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.True(
		t,
		denseIsIdentity(
			target.Arms["x86"].A,
			target.Dim,
		),
	)

	assert.True(
		t,
		vectorIsZero(
			target.Arms["x86"].b,
			target.Dim,
		),
	)
}

func TestApplyWeakPriorRejectsSecondApplication(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target :=
		NewUCB1Bandit(
			"target-single-prior",
			0.0,
		)

	target.InitArm("x86")

	require.NoError(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.Error(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.InDelta(
		t,
		1.0,
		target.Arms["x86"].
			PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		target.Arms["x86"].
			PriorRewardSum,
		1e-12,
	)
}

func TestApplyWeakPriorWithoutTransferredArmsIsNoOp(
	t *testing.T,
) {
	source := TransferableMABKnowledge{
		SchemaVersion: TransferableMABKnowledgeSchemaVersion,

		FunctionName: "donor-empty",

		Policy: UCB1,

		HasRealKnowledge: false,

		RealObservationCount: 0,

		ExcludedSyntheticObservationCount: 0,

		Arms: map[string]TransferableArmKnowledge{
			"x86": {
				RealObservationCount: 0,

				ExcludedSyntheticObservationCount: 0,

				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: 0.0,
					RealAvgReward:  0.0,
				},
			},
		},
	}

	prior, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.NoError(t, err)
	require.False(t, prior.HasPrior)

	target :=
		NewUCB1Bandit(
			"target-no-prior",
			0.0,
		)

	target.InitArm("x86")

	require.NoError(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.Arms["x86"].
			PriorObservationWeight,
	)

	assert.Zero(
		t,
		target.Arms["x86"].
			PriorRewardSum,
	)
}

func TestApplyWeakPriorRejectsCorruptedArtifactBeforeMutation(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	prior.TransferredArmCount = 2

	target :=
		NewUCB1Bandit(
			"target-corrupted-prior",
			0.0,
		)

	target.InitArm("x86")

	assert.Error(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.Arms["x86"].
			PriorObservationWeight,
	)

	assert.Zero(
		t,
		target.Arms["x86"].
			PriorRewardSum,
	)
}

func TestApplyWeakPriorRejectsSelfTransfer(
	t *testing.T,
) {
	prior := buildUCB1ApplicationPrior(
		t,
		1.0,
		map[string]TransferableUCB1ArmKnowledge{
			"x86": {
				RealSumRewards: -8.0,
				RealAvgReward:  -2.0,
			},
		},
	)

	target :=
		NewUCB1Bandit(
			"donor-ucb",
			0.0,
		)

	target.InitArm("x86")

	assert.Error(
		t,
		ApplyWeakMABPrior(
			target,
			prior,
		),
	)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)
}

func buildUCB1ApplicationPrior(
	t *testing.T,
	weight float64,
	arms map[string]TransferableUCB1ArmKnowledge,
) WeakMABPrior {
	t.Helper()

	sourceArms :=
		make(
			map[string]TransferableArmKnowledge,
			len(arms),
		)

	var totalReal int64

	for arm, payload := range arms {
		const realCount int64 = 4

		payloadCopy :=
			payload

		sourceArms[arm] =
			TransferableArmKnowledge{
				RealObservationCount: realCount,

				UCB1: &payloadCopy,
			}

		totalReal +=
			realCount
	}

	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor-ucb",

			Policy: UCB1,

			HasRealKnowledge: totalReal > 0,

			RealObservationCount: totalReal,

			Arms: sourceArms,
		}

	prior, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: weight,

				MinRealObservationsPerArm: 1,
			},
		)

	require.NoError(
		t,
		err,
	)

	return prior
}

func buildLinUCBApplicationPrior(
	t *testing.T,
	dim int,
	arms map[string]TransferableLinUCBArmKnowledge,
) WeakMABPrior {
	t.Helper()

	sourceArms :=
		make(
			map[string]TransferableArmKnowledge,
			len(arms),
		)

	var totalReal int64

	for arm, payload := range arms {
		const realCount int64 = 2

		payloadCopy :=
			payload

		sourceArms[arm] =
			TransferableArmKnowledge{
				RealObservationCount: realCount,

				LinUCB: &payloadCopy,
			}

		totalReal +=
			realCount
	}

	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor-lin",

			Policy: LinUCB,

			HasRealKnowledge: totalReal > 0,

			RealObservationCount: totalReal,

			Arms: sourceArms,
		}

	prior, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.NoError(
		t,
		err,
	)

	for _, arm := range prior.Arms {
		require.NotNil(
			t,
			arm.LinUCB,
		)

		require.Equal(
			t,
			dim,
			arm.LinUCB.Dim,
		)
	}

	return prior
}
