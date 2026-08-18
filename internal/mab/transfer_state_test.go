package mab

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUCB1TransferableKnowledgeKeepsOnlyRealFeedback(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewUCB1Bandit(
			"ucb-transfer-source",
			0.0,
		)

	bandit.InitArm(
		"x86",
	)

	bandit.UpdateReward(
		"x86",
		nil,
		ExecutionFeedback{
			DurationMs: 10.0,

			IsWarmStart: true,
		},
	)

	bandit.mu.Lock()

	bandit.updateSyntheticRewardLocked(
		"x86",
		SyntheticReward{
			RequestID: "fallback-1",

			Value: -12.0,

			Reason: FallbackReasonSelectedArmNoCandidate,
		},
	)

	bandit.mu.Unlock()

	snapshot :=
		bandit.
			TransferableKnowledge()

	arm :=
		snapshot.Arms["x86"]

	require.NotNil(
		t,
		arm.UCB1,
	)

	require.Nil(
		t,
		arm.LinUCB,
	)

	assert.Equal(
		t,
		int64(1),
		arm.RealObservationCount,
	)

	assert.Equal(
		t,
		int64(1),
		arm.ExcludedSyntheticObservationCount,
	)

	assert.InDelta(
		t,
		-math.Log(10.0),
		arm.UCB1.RealSumRewards,
		1e-9,
	)

	assert.InDelta(
		t,
		-math.Log(10.0),
		arm.UCB1.RealAvgReward,
		1e-9,
	)

	assert.Equal(
		t,
		int64(1),
		snapshot.RealObservationCount,
	)

	assert.Equal(
		t,
		int64(1),
		snapshot.ExcludedSyntheticObservationCount,
	)

	assert.True(
		t,
		snapshot.HasRealKnowledge,
	)

	assert.Equal(
		t,
		int64(2),
		bandit.Arms["x86"].Count,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["x86"].RealCount,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["x86"].SyntheticCount,
	)
}

func TestUCB1TransferableKnowledgeExcludesInFlight(
	t *testing.T,
) {
	bandit :=
		NewUCB1Bandit(
			"ucb-transfer-inflight",
			0.0,
		)

	bandit.InitArm(
		"x86",
	)

	selected :=
		bandit.SelectArmFrom(
			nil,
			[]string{
				"x86",
			},
		)

	require.Equal(
		t,
		"x86",
		selected,
	)

	require.Equal(
		t,
		int64(1),
		bandit.Arms["x86"].
			InFlight,
	)

	snapshot :=
		bandit.
			TransferableKnowledge()

	arm :=
		snapshot.Arms["x86"]

	assert.Zero(
		t,
		arm.RealObservationCount,
	)

	assert.Zero(
		t,
		arm.ExcludedSyntheticObservationCount,
	)

	assert.False(
		t,
		snapshot.HasRealKnowledge,
	)

	assert.Zero(
		t,
		snapshot.RealObservationCount,
	)
}

func TestLinUCBTransferableKnowledgeKeepsOnlyRealContributions(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-transfer-source",
			0.0,
		)

	bandit.InitArm(
		"x86",
	)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				"x86": 0.25,
			},
		}

	bandit.UpdateReward(
		"x86",
		ctx,
		ExecutionFeedback{
			DurationMs: 10.0,

			IsWarmStart: true,
		},
	)

	bandit.mu.Lock()

	bandit.updateSyntheticRewardLocked(
		"x86",
		ctx,
		SyntheticReward{
			RequestID: "fallback-1",

			Value: -12.0,

			Reason: FallbackReasonSelectedArmNoCandidate,
		},
	)

	bandit.mu.Unlock()

	snapshot :=
		bandit.
			TransferableKnowledge()

	arm :=
		snapshot.Arms["x86"]

	require.NotNil(
		t,
		arm.LinUCB,
	)

	require.Nil(
		t,
		arm.UCB1,
	)

	features :=
		bandit.
			computeFeatures(
				0.25,
			)

	expectedReward :=
		-math.Log(
			10.0,
		)

	assert.Equal(
		t,
		int64(1),
		arm.RealObservationCount,
	)

	assert.Equal(
		t,
		int64(1),
		arm.ExcludedSyntheticObservationCount,
	)

	assert.Equal(
		t,
		bandit.Dim,
		arm.LinUCB.Dim,
	)

	for i := 0; i < bandit.Dim; i++ {
		assert.InDelta(
			t,
			expectedReward*
				features.AtVec(i),
			arm.LinUCB.
				BContribution[i],
			1e-9,
		)

		for j := 0; j < bandit.Dim; j++ {
			assert.InDelta(
				t,
				features.AtVec(i)*
					features.AtVec(j),
				arm.LinUCB.
					AContribution[i][j],
				1e-9,
			)
		}
	}

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["x86"].
			RealObservationCount,
	)

	assert.Equal(
		t,
		int64(1),
		bandit.Arms["x86"].
			SyntheticObservationCount,
	)
}

func TestLinUCBTransferableKnowledgeExcludesIdentityRegularizer(
	t *testing.T,
) {
	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-empty-transfer",
			0.0,
		)

	bandit.InitArm(
		"x86",
	)

	snapshot :=
		bandit.
			TransferableKnowledge()

	arm :=
		snapshot.Arms["x86"]

	require.NotNil(
		t,
		arm.LinUCB,
	)

	for i := 0; i < bandit.Dim; i++ {
		assert.Zero(
			t,
			arm.LinUCB.
				BContribution[i],
		)

		for j := 0; j < bandit.Dim; j++ {
			assert.Zero(
				t,
				arm.LinUCB.
					AContribution[i][j],
			)
		}
	}

	for i := 0; i < bandit.Dim; i++ {
		assert.InDelta(
			t,
			1.0,
			bandit.Arms["x86"].
				A.At(
				i,
				i,
			),
			1e-9,
		)
	}

	assert.False(
		t,
		snapshot.HasRealKnowledge,
	)
}

func TestTransferableKnowledgeIsDeepCopied(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(t)

	bandit :=
		NewLinUCBDisjointPolicy(
			"linucb-copy-transfer",
			0.0,
		)

	bandit.InitArm(
		"x86",
	)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				"x86": 0.10,
			},
		}

	bandit.UpdateReward(
		"x86",
		ctx,
		ExecutionFeedback{
			DurationMs: 10.0,

			IsWarmStart: true,
		},
	)

	snapshot :=
		bandit.
			TransferableKnowledge()

	snapshot.
		Arms["x86"].
		LinUCB.
		AContribution[0][0] =
		999.0

	snapshot.
		Arms["x86"].
		LinUCB.
		BContribution[0] =
		999.0

	second :=
		bandit.
			TransferableKnowledge()

	assert.NotEqual(
		t,
		999.0,
		second.
			Arms["x86"].
			LinUCB.
			AContribution[0][0],
	)

	assert.NotEqual(
		t,
		999.0,
		second.
			Arms["x86"].
			LinUCB.
			BContribution[0],
	)
}

func TestBanditManagerSnapshotDoesNotCreateUnknownBandit(
	t *testing.T,
) {
	manager :=
		&BanditManager{
			bandits: make(
				map[string]Policy,
			),
		}

	_, ok :=
		manager.
			SnapshotTransferableKnowledge(
				"unknown-function",
			)

	assert.False(
		t,
		ok,
	)

	assert.Empty(
		t,
		manager.bandits,
	)
}

func TestBanditManagerSnapshotsBothSupportedPolicies(
	t *testing.T,
) {
	ucb :=
		NewUCB1Bandit(
			"ucb",
			0.0,
		)

	ucb.InitArm(
		"x86",
	)

	lin :=
		NewLinUCBDisjointPolicy(
			"lin",
			0.0,
		)

	lin.InitArm(
		"x86",
	)

	manager :=
		&BanditManager{
			bandits: map[string]Policy{
				"ucb": ucb,

				"lin": lin,
			},
		}

	ucbSnapshot, ok :=
		manager.
			SnapshotTransferableKnowledge(
				"ucb",
			)

	require.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		UCB1,
		ucbSnapshot.Policy,
	)

	linSnapshot, ok :=
		manager.
			SnapshotTransferableKnowledge(
				"lin",
			)

	require.True(
		t,
		ok,
	)

	assert.Equal(
		t,
		LinUCB,
		linSnapshot.Policy,
	)
}
