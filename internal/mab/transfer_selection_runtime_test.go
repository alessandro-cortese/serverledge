package mab

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelectionArtifactRuntimeTransferUCB1UsesSelectedDonor(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	donor :=
		manager.
			GetBandit(
				"selected-donor",
			).(*UCB1Bandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			nil,
			ExecutionFeedback{
				DurationMs: 10.0,

				IsWarmStart: true,

				CostFactor: 1.0,

				EnergyFactor: 1.0,
			},
		)
	}

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-ucb1",
			"target-from-selection",
			DonorSelectionStatusSelected,
			"",
			"selected-donor",
			0.25,
			1.0,
			false,
			nil,
		)

	result, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-from-selection",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 0.5,

					MinRealObservationsPerArm: 2,
				},
			)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		"selection-run-ucb1",
		result.SelectionRunID,
	)

	assert.Equal(
		t,
		DonorSelectionStatusSelected,
		result.SelectionStatus,
	)

	assert.Equal(
		t,
		"selected-donor",
		result.SelectedDonorFunctionName,
	)

	assert.True(
		t,
		result.TransferAttempted,
	)

	assert.True(
		t,
		result.TransferApplied,
	)

	assert.Equal(
		t,
		RuntimeTransferReasonApplied,
		result.RuntimeReason,
	)

	assert.True(
		t,
		result.Prior.HasPrior,
	)

	target :=
		manager.
			GetBandit(
				"target-from-selection",
			).(*UCB1Bandit)

	assert.Equal(
		t,
		"selected-donor",
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.TotalCounts,
	)

	assert.Zero(
		t,
		target.Arms["x86-local"].RealCount,
	)

	assert.InDelta(
		t,
		0.5,
		target.Arms["x86-local"].PriorObservationWeight,
		1e-12,
	)
}

func TestSelectionArtifactRuntimeTransferLinUCBUsesSelectedDonor(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"LinUCB",
	)

	viper.Set(
		config.MAB_LINUCB_ALPHA,
		0.1,
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	donor :=
		manager.
			GetBandit(
				"selected-lin-donor",
			).(*LinUCBDisjointPolicy)

	ctx :=
		&Context{
			ArchMemUsage: map[string]float64{
				"x86-local": 0.35,
			},
		}

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			ctx,
			ExecutionFeedback{
				DurationMs: 12.0,

				IsWarmStart: true,

				CostFactor: 1.0,

				EnergyFactor: 1.0,
			},
		)
	}

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-linucb",
			"target-lin-selection",
			DonorSelectionStatusSelected,
			"",
			"selected-lin-donor",
			0.20,
			1.0,
			false,
			nil,
		)

	result, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-lin-selection",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 0.5,

					MinRealObservationsPerArm: 2,
				},
			)

	require.NoError(
		t,
		err,
	)

	assert.True(
		t,
		result.TransferAttempted,
	)

	assert.True(
		t,
		result.TransferApplied,
	)

	target :=
		manager.
			GetBandit(
				"target-lin-selection",
			).(*LinUCBDisjointPolicy)

	state :=
		target.Arms["x86-local"]

	assert.Equal(
		t,
		"selected-lin-donor",
		target.PriorDonorFunctionName,
	)

	assert.InDelta(
		t,
		0.5,
		state.PriorObservationWeight,
		1e-12,
	)

	assert.Zero(
		t,
		state.RealObservationCount,
	)

	assert.False(
		t,
		denseIsZero(
			state.PriorAContribution,
			target.Dim,
			target.Dim,
		),
	)

	assert.False(
		t,
		vectorIsZero(
			state.PriorBContribution,
			target.Dim,
		),
	)
}

func TestSelectionArtifactNoTransferCreatesFreshTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-far",
			"far-target",
			DonorSelectionStatusNoTransfer,
			"distance_threshold_exceeded",
			"",
			0.0,
			1.0,
			false,
			nil,
		)

	result, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"far-target",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 0.5,

					MinRealObservationsPerArm: 2,
				},
			)

	require.NoError(
		t,
		err,
	)

	assert.Equal(
		t,
		DonorSelectionStatusNoTransfer,
		result.SelectionStatus,
	)

	assert.Equal(
		t,
		"distance_threshold_exceeded",
		result.SelectionReason,
	)

	assert.False(
		t,
		result.TransferAttempted,
	)

	assert.False(
		t,
		result.TransferApplied,
	)

	assert.Equal(
		t,
		SelectionRuntimeReasonSelectionNoTransfer,
		result.RuntimeReason,
	)

	target :=
		manager.
			GetBandit(
				"far-target",
			).(*UCB1Bandit)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.TotalCounts,
	)

	assert.Zero(
		t,
		target.Arms["x86-local"].PriorObservationWeight,
	)
}

func TestSelectionArtifactSelectedDonorWithoutRuntimeEvidenceUsesFreshTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	// Similarity selected the donor, but the donor has not accumulated
	// sufficient real MAB knowledge.
	manager.GetBandit(
		"empty-selected-donor",
	)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-empty-donor",
			"target-empty-donor",
			DonorSelectionStatusSelected,
			"",
			"empty-selected-donor",
			0.10,
			1.0,
			false,
			nil,
		)

	result, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-empty-donor",
				artifactPath,
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
		result.TransferAttempted,
	)

	assert.False(
		t,
		result.TransferApplied,
	)

	assert.Equal(
		t,
		RuntimeTransferReasonNoTransferablePrior,
		result.RuntimeReason,
	)

	target :=
		manager.
			GetBandit(
				"target-empty-donor",
			).(*UCB1Bandit)

	assert.Empty(
		t,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.Arms["x86-local"].PriorObservationWeight,
	)
}

func TestSelectionArtifactTargetMismatchDoesNotPublishTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-mismatch",
			"artifact-target",
			DonorSelectionStatusNoTransfer,
			"distance_threshold_exceeded",
			"",
			0.0,
			1.0,
			false,
			nil,
		)

	_, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"different-target",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 1.0,

					MinRealObservationsPerArm: 1,
				},
			)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["different-target"]

	assert.False(
		t,
		exists,
	)

	_, exists =
		manager.bandits["artifact-target"]

	assert.False(
		t,
		exists,
	)
}

func TestSelectionArtifactUnknownSelectedDonorDoesNotPublishTarget(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-missing-donor",
			"target-missing-donor",
			DonorSelectionStatusSelected,
			"",
			"missing-donor",
			0.10,
			1.0,
			false,
			nil,
		)

	_, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-missing-donor",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 1.0,

					MinRealObservationsPerArm: 1,
				},
			)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["target-missing-donor"]

	assert.False(
		t,
		exists,
	)
}

func TestSelectionArtifactRejectsMaterializedBanditPrior(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-invalid-prior",
			"target-invalid-prior",
			DonorSelectionStatusSelected,
			"",
			"donor",
			0.10,
			1.0,
			true,
			map[string]any{
				"unexpected": true,
			},
		)

	_, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-invalid-prior",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 1.0,

					MinRealObservationsPerArm: 1,
				},
			)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["target-invalid-prior"]

	assert.False(
		t,
		exists,
	)
}

func TestSelectionArtifactRejectsSelectedDonorOutsideThreshold(
	t *testing.T,
) {
	resetExecutionFeedbackConfig(
		t,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	manager :=
		newRuntimeTransferTestManager(
			"x86-local",
		)

	artifactPath :=
		writeRuntimeSelectionArtifact(
			t,
			"selection-run-invalid-distance",
			"target-invalid-distance",
			DonorSelectionStatusSelected,
			"",
			"donor",
			2.0,
			1.0,
			false,
			nil,
		)

	_, err :=
		manager.
			InitializeTargetFromDonorSelectionFile(
				"target-invalid-distance",
				artifactPath,
				WeakMABPriorConfig{
					EquivalentObservationWeight: 1.0,

					MinRealObservationsPerArm: 1,
				},
			)

	require.Error(
		t,
		err,
	)

	_, exists :=
		manager.bandits["target-invalid-distance"]

	assert.False(
		t,
		exists,
	)
}

func writeRuntimeSelectionArtifact(
	t *testing.T,
	selectionRunID string,
	targetFunctionName string,
	status string,
	reason string,
	donorFunctionName string,
	donorDistance float64,
	maxDistance float64,
	banditPriorMaterialized bool,
	banditPrior any,
) string {
	t.Helper()

	var selectedDonor any

	candidateCount :=
		0

	if status ==
		DonorSelectionStatusSelected {

		selectedDonor =
			map[string]any{
				"function_name": donorFunctionName,

				"distance": donorDistance,
			}

		candidateCount =
			1
	}

	document :=
		map[string]any{
			"schema_version": DonorSelectionArtifactSchemaVersion,

			"selection_run_id": selectionRunID,

			"status": status,

			"reason": reason,

			"query": map[string]any{
				"schema_version": DonorSelectionQuerySchemaVersion,

				"query_id": "query-" +
					targetFunctionName,

				"function_name": targetFunctionName,
			},

			"selection_policy": map[string]any{
				"distance": "euclidean",

				"max_distance": maxDistance,

				"configuration_match_required": true,

				"require_same_cluster": false,

				"bandit_prior_materialized": banditPriorMaterialized,
			},

			"selected_donor": selectedDonor,

			"candidate_count": candidateCount,

			"ranking": []any{},

			"bandit_prior": banditPrior,
		}

	content, err :=
		json.MarshalIndent(
			document,
			"",
			"  ",
		)

	require.NoError(
		t,
		err,
	)

	path :=
		filepath.Join(
			t.TempDir(),
			"selection.json",
		)

	require.NoError(
		t,
		os.WriteFile(
			path,
			content,
			0o600,
		),
	)

	return path
}
