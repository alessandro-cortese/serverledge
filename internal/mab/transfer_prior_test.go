package mab

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildWeakUCB1PriorUsesOneEquivalentObservation(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor-ucb",

			Policy: UCB1,

			HasRealKnowledge: true,

			RealObservationCount: 4,

			ExcludedSyntheticObservationCount: 2,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 4,

					ExcludedSyntheticObservationCount: 2,

					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: -8.0,

						RealAvgReward: -2.0,
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

	require.NoError(
		t,
		err,
	)

	assert.True(
		t,
		prior.HasPrior,
	)

	assert.Equal(
		t,
		1,
		prior.TransferredArmCount,
	)

	arm :=
		prior.Arms["x86"]

	require.True(
		t,
		arm.Transferred,
	)

	require.NotNil(
		t,
		arm.UCB1,
	)

	require.Nil(
		t,
		arm.LinUCB,
	)

	assert.InDelta(
		t,
		1.0,
		arm.
			AppliedEquivalentObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		0.25,
		arm.
			AttenuationScale,
		1e-12,
	)

	assert.InDelta(
		t,
		1.0,
		arm.UCB1.
			ObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		arm.UCB1.
			RewardSum,
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		arm.UCB1.
			MeanReward,
		1e-12,
	)

	assert.Equal(
		t,
		int64(2),
		arm.
			SourceExcludedSyntheticObservationCount,
	)
}

func TestBuildWeakUCB1PriorSupportsFractionalObservationWeight(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor-ucb",

			Policy: UCB1,

			HasRealKnowledge: true,

			RealObservationCount: 4,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 4,

					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: -8.0,

						RealAvgReward: -2.0,
					},
				},
			},
		}

	prior, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 0.5,

				MinRealObservationsPerArm: 1,
			},
		)

	require.NoError(
		t,
		err,
	)

	arm :=
		prior.Arms["x86"]

	assert.InDelta(
		t,
		0.5,
		arm.UCB1.
			ObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		-1.0,
		arm.UCB1.
			RewardSum,
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		arm.UCB1.
			MeanReward,
		1e-12,
	)
}

func TestBuildWeakPriorSkipsArmWithInsufficientRealEvidence(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor",

			Policy: UCB1,

			HasRealKnowledge: true,

			RealObservationCount: 1,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 1,

					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: -2.0,

						RealAvgReward: -2.0,
					},
				},
			},
		}

	prior, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 2,
			},
		)

	require.NoError(
		t,
		err,
	)

	assert.False(
		t,
		prior.HasPrior,
	)

	assert.Zero(
		t,
		prior.TransferredArmCount,
	)

	assert.Equal(
		t,
		1,
		prior.SkippedArmCount,
	)

	arm :=
		prior.Arms["x86"]

	assert.False(
		t,
		arm.Transferred,
	)

	assert.Equal(
		t,
		WeakPriorSkipInsufficientRealObservations,
		arm.SkipReason,
	)

	assert.Nil(
		t,
		arm.UCB1,
	)
}

func TestBuildWeakLinUCBPriorScalesRealContributions(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor-linucb",

			Policy: LinUCB,

			HasRealKnowledge: true,

			RealObservationCount: 4,

			ExcludedSyntheticObservationCount: 3,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 4,

					ExcludedSyntheticObservationCount: 3,

					LinUCB: &TransferableLinUCBArmKnowledge{
						Dim: 2,

						AContribution: [][]float64{
							{
								4.0,
								2.0,
							},
							{
								2.0,
								8.0,
							},
						},

						BContribution: []float64{
							-8.0,
							-4.0,
						},
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

	require.NoError(
		t,
		err,
	)

	arm :=
		prior.Arms["x86"]

	require.True(
		t,
		arm.Transferred,
	)

	require.NotNil(
		t,
		arm.LinUCB,
	)

	require.Nil(
		t,
		arm.UCB1,
	)

	assert.InDelta(
		t,
		0.25,
		arm.AttenuationScale,
		1e-12,
	)

	assert.InDelta(
		t,
		1.0,
		arm.LinUCB.
			AContribution[0][0],
		1e-12,
	)

	assert.InDelta(
		t,
		0.5,
		arm.LinUCB.
			AContribution[0][1],
		1e-12,
	)

	assert.InDelta(
		t,
		0.5,
		arm.LinUCB.
			AContribution[1][0],
		1e-12,
	)

	assert.InDelta(
		t,
		2.0,
		arm.LinUCB.
			AContribution[1][1],
		1e-12,
	)

	assert.InDelta(
		t,
		-2.0,
		arm.LinUCB.
			BContribution[0],
		1e-12,
	)

	assert.InDelta(
		t,
		-1.0,
		arm.LinUCB.
			BContribution[1],
		1e-12,
	)

	assert.Equal(
		t,
		int64(3),
		arm.
			SourceExcludedSyntheticObservationCount,
	)
}

func TestBuildWeakPriorRejectsInvalidConfig(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor",

			Policy: UCB1,

			HasRealKnowledge: false,

			Arms: map[string]TransferableArmKnowledge{},
		}

	tests :=
		[]struct {
			name   string
			config WeakMABPriorConfig
		}{
			{
				name: "zero weight",

				config: WeakMABPriorConfig{
					EquivalentObservationWeight: 0.0,

					MinRealObservationsPerArm: 1,
				},
			},
			{
				name: "negative weight",

				config: WeakMABPriorConfig{
					EquivalentObservationWeight: -0.1,

					MinRealObservationsPerArm: 1,
				},
			},
			{
				name: "weight greater than one",

				config: WeakMABPriorConfig{
					EquivalentObservationWeight: 1.1,

					MinRealObservationsPerArm: 1,
				},
			},
			{
				name: "NaN weight",

				config: WeakMABPriorConfig{
					EquivalentObservationWeight: math.NaN(),

					MinRealObservationsPerArm: 1,
				},
			},
			{
				name: "invalid minimum observations",

				config: WeakMABPriorConfig{
					EquivalentObservationWeight: 1.0,

					MinRealObservationsPerArm: 0,
				},
			},
		}

	for _, test := range tests {

		t.Run(
			test.name,
			func(
				t *testing.T,
			) {
				_, err :=
					BuildWeakMABPrior(
						source,
						test.config,
					)

				require.Error(
					t,
					err,
				)
			},
		)
	}
}

func TestBuildWeakPriorRejectsTransferableSummaryMismatch(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "donor",

			Policy: UCB1,

			HasRealKnowledge: true,

			// Wrong on purpose.
			RealObservationCount: 2,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 1,

					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: -2.0,

						RealAvgReward: -2.0,
					},
				},
			},
		}

	_, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"summary mismatch",
	)
}

func TestBuildWeakPriorWithoutRealKnowledgeProducesNoPrior(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "empty-donor",

			Policy: UCB1,

			HasRealKnowledge: false,

			RealObservationCount: 0,

			ExcludedSyntheticObservationCount: 3,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 0,

					ExcludedSyntheticObservationCount: 3,

					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: 0.0,

						RealAvgReward: 0.0,
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

	require.NoError(
		t,
		err,
	)

	assert.False(
		t,
		prior.HasPrior,
	)

	assert.Zero(
		t,
		prior.TransferredArmCount,
	)

	assert.Equal(
		t,
		1,
		prior.SkippedArmCount,
	)

	assert.Equal(
		t,
		WeakPriorSkipNoRealObservations,
		prior.Arms["x86"].
			SkipReason,
	)

	assert.Equal(
		t,
		int64(3),
		prior.
			SourceExcludedSyntheticObservationCount,
	)
}

func TestBuildWeakPriorRejectsPolicyPayloadMismatch(
	t *testing.T,
) {
	source :=
		TransferableMABKnowledge{
			SchemaVersion: TransferableMABKnowledgeSchemaVersion,

			FunctionName: "invalid-donor",

			Policy: UCB1,

			HasRealKnowledge: true,

			RealObservationCount: 1,

			Arms: map[string]TransferableArmKnowledge{
				"x86": {
					RealObservationCount: 1,

					// Wrong payload on purpose.
					LinUCB: &TransferableLinUCBArmKnowledge{
						Dim: 2,

						AContribution: [][]float64{
							{
								1.0,
								0.0,
							},
							{
								0.0,
								1.0,
							},
						},

						BContribution: []float64{
							0.0,
							0.0,
						},
					},
				},
			},
		}

	_, err :=
		BuildWeakMABPrior(
			source,
			WeakMABPriorConfig{
				EquivalentObservationWeight: 1.0,

				MinRealObservationsPerArm: 1,
			},
		)

	require.Error(
		t,
		err,
	)

	assert.Contains(
		t,
		err.Error(),
		"invalid UCB1 transferable payload",
	)
}
