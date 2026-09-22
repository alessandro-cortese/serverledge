package lb

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/labstack/echo/v4"
	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/mab"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTransferControlInitializesSelectedUCB1Donor(
	t *testing.T,
) {
	oldManager :=
		mab.GlobalBanditManager

	viper.Reset()

	mab.InitBanditManager()

	mab.GlobalBanditManager.
		AddArmToAll(
			"x86-local",
		)

	t.Cleanup(
		func() {
			viper.Reset()

			mab.GlobalBanditManager =
				oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		MAB,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	donor :=
		mab.GlobalBanditManager.
			GetBandit(
				"control-donor",
			).(*mab.UCB1Bandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			nil,
			mab.ExecutionFeedback{
				DurationMs: 10.0,

				IsWarmStart: true,
			},
		)
	}

	body :=
		transferControlRequestBody(
			t,
			"control-target",
			"control-donor",
			mab.DonorSelectionStatusSelected,
			"",
		)

	e := echo.New()

	RegisterTransferControlRoutes(
		e,
	)

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			bytes.NewReader(
				body,
			),
		)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec :=
		httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	require.Equal(
		t,
		http.StatusOK,
		rec.Code,
		rec.Body.String(),
	)

	var result mab.SelectionRuntimeTransferResult

	require.NoError(
		t,
		json.Unmarshal(
			rec.Body.Bytes(),
			&result,
		),
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
		"control-donor",
		result.SelectedDonorFunctionName,
	)

	assert.Equal(
		t,
		mab.RuntimeTransferReasonApplied,
		result.RuntimeReason,
	)

	target :=
		mab.GlobalBanditManager.
			GetBandit(
				"control-target",
			).(*mab.UCB1Bandit)

	assert.Equal(
		t,
		"control-donor",
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

func TestTransferControlNoTransferCreatesFreshTarget(
	t *testing.T,
) {
	oldManager :=
		mab.GlobalBanditManager

	viper.Reset()

	mab.InitBanditManager()

	mab.GlobalBanditManager.
		AddArmToAll(
			"x86-local",
		)

	t.Cleanup(
		func() {
			viper.Reset()

			mab.GlobalBanditManager =
				oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		MAB,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	body :=
		transferControlRequestBody(
			t,
			"fresh-control-target",
			"",
			mab.DonorSelectionStatusNoTransfer,
			"distance_threshold_exceeded",
		)

	e := echo.New()

	RegisterTransferControlRoutes(
		e,
	)

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			bytes.NewReader(
				body,
			),
		)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec :=
		httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	require.Equal(
		t,
		http.StatusOK,
		rec.Code,
		rec.Body.String(),
	)

	var result mab.SelectionRuntimeTransferResult

	require.NoError(
		t,
		json.Unmarshal(
			rec.Body.Bytes(),
			&result,
		),
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
		mab.SelectionRuntimeReasonSelectionNoTransfer,
		result.RuntimeReason,
	)

	target :=
		mab.GlobalBanditManager.
			GetBandit(
				"fresh-control-target",
			).(*mab.UCB1Bandit)

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

func TestTransferControlRejectsNonMABLoadBalancerMode(
	t *testing.T,
) {
	oldManager :=
		mab.GlobalBanditManager

	viper.Reset()

	mab.InitBanditManager()

	t.Cleanup(
		func() {
			viper.Reset()

			mab.GlobalBanditManager =
				oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		RR,
	)

	body :=
		transferControlRequestBody(
			t,
			"target",
			"",
			mab.DonorSelectionStatusNoTransfer,
			"distance_threshold_exceeded",
		)

	e := echo.New()

	RegisterTransferControlRoutes(
		e,
	)

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			bytes.NewReader(
				body,
			),
		)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec :=
		httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	assert.Equal(
		t,
		http.StatusConflict,
		rec.Code,
	)
}

func TestTransferControlRouteIsNotProxied(
	t *testing.T,
) {
	e := echo.New()

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			nil,
		)

	rec :=
		httptest.NewRecorder()

	ctx :=
		e.NewContext(
			req,
			rec,
		)

	assert.True(
		t,
		transferControlProxySkipper(
			ctx,
		),
	)

	otherReq :=
		httptest.NewRequest(
			http.MethodPost,
			"/invoke/test",
			nil,
		)

	otherRec :=
		httptest.NewRecorder()

	otherCtx :=
		e.NewContext(
			otherReq,
			otherRec,
		)

	assert.False(
		t,
		transferControlProxySkipper(
			otherCtx,
		),
	)
}

func TestTransferControlDisabledDoesNotRegisterRoute(
	t *testing.T,
) {
	viper.Reset()

	t.Cleanup(
		viper.Reset,
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		false,
	)

	e := echo.New()

	RegisterTransferControlRoutes(
		e,
	)

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			bytes.NewReader(
				[]byte(`{}`),
			),
		)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec :=
		httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	assert.Equal(
		t,
		http.StatusNotFound,
		rec.Code,
	)
}

func transferControlRequestBody(
	t *testing.T,
	target string,
	donor string,
	status string,
	reason string,
) []byte {
	t.Helper()

	var selectedDonor any

	candidateCount := 0

	if status ==
		mab.DonorSelectionStatusSelected {

		selectedDonor =
			map[string]any{
				"function_name": donor,

				"distance": 0.25,
			}

		candidateCount = 1
	}

	artifact :=
		map[string]any{
			"schema_version": mab.
				DonorSelectionArtifactSchemaVersion,

			"selection_run_id": "control-selection-run",

			"status": status,

			"reason": reason,

			"query": map[string]any{
				"schema_version": mab.
					DonorSelectionQuerySchemaVersion,

				"query_id": "query-" +
					target,

				"function_name": target,
			},

			"selection_policy": map[string]any{
				"distance": "euclidean",

				"max_distance": 1.0,

				"configuration_match_required": true,

				"require_same_cluster": false,

				"bandit_prior_materialized": false,
			},

			"selected_donor": selectedDonor,

			"candidate_count": candidateCount,

			"bandit_prior": nil,
		}

	request :=
		map[string]any{
			"target_function_name": target,

			"selection_artifact": artifact,

			"prior_config": map[string]any{
				"equivalent_observation_weight": 0.5,

				"min_real_observations_per_arm": 2,
			},
		}

	body, err :=
		json.Marshal(
			request,
		)

	require.NoError(
		t,
		err,
	)

	return body
}

func TestMaterializedTransferControlReplaysFrozenUCB1Prior(
	t *testing.T,
) {
	oldManager :=
		mab.GlobalBanditManager

	viper.Reset()

	t.Cleanup(
		func() {
			viper.Reset()

			mab.GlobalBanditManager =
				oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		MAB,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1",
	)

	// ------------------------------------------------------------
	// Phase A:
	// build a real materialized prior from donor observations.
	// ------------------------------------------------------------

	viper.Set(
		config.MAB_UCB1_C,
		0.8,
	)

	mab.InitBanditManager()

	mab.GlobalBanditManager.
		AddArmToAll(
			"x86-local",
		)

	donor :=
		mab.GlobalBanditManager.
			GetBandit(
				"materialized-control-donor",
			).(*mab.UCB1Bandit)

	for i := 0; i < 4; i++ {
		donor.UpdateReward(
			"x86-local",
			nil,
			mab.ExecutionFeedback{
				DurationMs:  10.0,
				IsWarmStart: true,
			},
		)
	}

	firstBody :=
		transferControlRequestBody(
			t,
			"temporary-source-target",
			"materialized-control-donor",
			mab.DonorSelectionStatusSelected,
			"",
		)

	firstEcho :=
		echo.New()

	RegisterTransferControlRoutes(
		firstEcho,
	)

	firstReq :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializePath,
			bytes.NewReader(
				firstBody,
			),
		)

	firstReq.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	firstRec :=
		httptest.NewRecorder()

	firstEcho.ServeHTTP(
		firstRec,
		firstReq,
	)

	require.Equal(
		t,
		http.StatusOK,
		firstRec.Code,
		firstRec.Body.String(),
	)

	var originalResult mab.SelectionRuntimeTransferResult

	require.NoError(
		t,
		json.Unmarshal(
			firstRec.Body.Bytes(),
			&originalResult,
		),
	)

	require.True(
		t,
		originalResult.TransferApplied,
	)

	require.True(
		t,
		originalResult.Prior.HasPrior,
	)

	frozenPrior :=
		originalResult.Prior

	// ------------------------------------------------------------
	// Phase B:
	// completely fresh BanditManager with c=0.
	//
	// The original donor no longer exists in this runtime.
	// Only the materialized prior is transferred.
	// ------------------------------------------------------------

	mab.InitBanditManager()

	mab.GlobalBanditManager.
		AddArmToAll(
			"x86-local",
		)

	viper.Set(
		config.MAB_UCB1_C,
		0.0,
	)

	requestBody, err :=
		json.Marshal(
			map[string]any{
				"target_function_name": "materialized-control-target",

				"prior": frozenPrior,
			},
		)

	require.NoError(
		t,
		err,
	)

	secondEcho :=
		echo.New()

	RegisterTransferControlRoutes(
		secondEcho,
	)

	secondReq :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializeMaterializedPath,
			bytes.NewReader(
				requestBody,
			),
		)

	secondReq.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	secondRec :=
		httptest.NewRecorder()

	secondEcho.ServeHTTP(
		secondRec,
		secondReq,
	)

	require.Equal(
		t,
		http.StatusOK,
		secondRec.Code,
		secondRec.Body.String(),
	)

	var result transferControlInitializeMaterializedResponse

	require.NoError(
		t,
		json.Unmarshal(
			secondRec.Body.Bytes(),
			&result,
		),
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
		"materialized_prior",
		result.InitializationSource,
	)

	assert.Equal(
		t,
		frozenPrior.DonorFunctionName,
		result.DonorFunctionName,
	)

	assert.Equal(
		t,
		frozenPrior,
		result.Prior,
	)

	target :=
		mab.GlobalBanditManager.
			GetBandit(
				"materialized-control-target",
			).(*mab.UCB1Bandit)

	assert.Equal(
		t,
		frozenPrior.DonorFunctionName,
		target.PriorDonorFunctionName,
	)

	assert.Zero(
		t,
		target.TotalCounts,
	)

	assert.Zero(
		t,
		target.Arms["x86-local"].
			RealCount,
	)

	assert.InDelta(
		t,
		frozenPrior.
			Arms["x86-local"].
			UCB1.
			ObservationWeight,
		target.
			Arms["x86-local"].
			PriorObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		frozenPrior.
			Arms["x86-local"].
			UCB1.
			RewardSum,
		target.
			Arms["x86-local"].
			PriorRewardSum,
		1e-12,
	)
}

func TestMaterializedTransferControlReplaysFrozenUCB1DecoupledPrior(
	t *testing.T,
) {
	oldManager := mab.GlobalBanditManager

	viper.Reset()

	t.Cleanup(
		func() {
			viper.Reset()
			mab.GlobalBanditManager = oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		MAB,
	)

	viper.Set(
		config.MAB_POLICY,
		"UCB1Decoupled",
	)

	viper.Set(
		config.MAB_UCB1_C,
		0.8,
	)

	mab.InitBanditManager()

	mab.GlobalBanditManager.AddArmToAll("amd64")
	mab.GlobalBanditManager.AddArmToAll("arm64")

	/*
		This prior mirrors the structure used by the R02 decoupled
		materialized bundle:

		    reward weight      = 0.25
		    exploration weight = 1.0

		The reward-side values are those of the frozen coupled R02
		prior. Only the exploration pseudo-count differs.
	*/
	requestBody, err := json.Marshal(
		map[string]any{
			"target_function_name": "materialized-decoupled-target",
			"prior": map[string]any{
				"schema_version":      1,
				"donor_function_name": "compression",
				"policy":              "UCB1Decoupled",
				"config": map[string]any{
					"reward_observation_weight":      0.25,
					"exploration_observation_weight": 1.0,
					"min_real_observations_per_arm":  10,
					"ucb1_reference_anchor": map[string]any{
						"enabled":                      true,
						"reference_arm":                "amd64",
						"target_reference_mean_reward": -7.364532382702029,
					},
				},
				"has_prior":                                   true,
				"source_real_observation_count":               30,
				"source_excluded_synthetic_observation_count": 0,
				"arm_count":                                   2,
				"transferred_arm_count":                       2,
				"skipped_arm_count":                           0,
				"arms": map[string]any{
					"amd64": map[string]any{
						"source_real_observation_count":               16,
						"source_excluded_synthetic_observation_count": 0,
						"transferred":                            true,
						"applied_equivalent_observation_weight":  0.25,
						"applied_exploration_observation_weight": 1.0,
						"attenuation_scale":                      0.015625,
						"ucb1": map[string]any{
							"observation_weight":             0.25,
							"exploration_observation_weight": 1.0,
							"mean_reward":                    -7.364532382702029,
							"reward_sum":                     -1.8411330956755072,
						},
					},
					"arm64": map[string]any{
						"source_real_observation_count":               14,
						"source_excluded_synthetic_observation_count": 0,
						"transferred":                            true,
						"applied_equivalent_observation_weight":  0.25,
						"applied_exploration_observation_weight": 1.0,
						"attenuation_scale":                      0.017857142857142856,
						"ucb1": map[string]any{
							"observation_weight":             0.25,
							"exploration_observation_weight": 1.0,
							"mean_reward":                    -7.401677954614536,
							"reward_sum":                     -1.850419488653634,
						},
					},
				},
			},
		},
	)

	require.NoError(t, err)

	e := echo.New()

	RegisterTransferControlRoutes(e)

	req := httptest.NewRequest(
		http.MethodPost,
		TransferControlInitializeMaterializedPath,
		bytes.NewReader(requestBody),
	)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec := httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	require.Equal(
		t,
		http.StatusOK,
		rec.Code,
		rec.Body.String(),
	)

	var result transferControlInitializeMaterializedResponse

	require.NoError(
		t,
		json.Unmarshal(
			rec.Body.Bytes(),
			&result,
		),
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
		"materialized_prior",
		result.InitializationSource,
	)

	assert.Equal(
		t,
		mab.UCB1Decoupled,
		result.Policy,
	)

	assert.Equal(
		t,
		"compression",
		result.DonorFunctionName,
	)

	assert.Equal(
		t,
		mab.UCB1Decoupled,
		result.Prior.Policy,
	)

	assert.InDelta(
		t,
		0.25,
		result.Prior.Config.RewardObservationWeight,
		1e-12,
	)

	assert.InDelta(
		t,
		1.0,
		result.Prior.Config.ExplorationObservationWeight,
		1e-12,
	)

	target := mab.GlobalBanditManager.
		GetBandit(
			"materialized-decoupled-target",
		).(*mab.UCB1DecoupledBandit)

	assert.Equal(
		t,
		"compression",
		target.PriorDonorFunctionName,
	)

	// The frozen prior must not become fake real target experience.
	assert.Zero(
		t,
		target.TotalCounts,
	)

	for _, arm := range []string{"amd64", "arm64"} {
		stats := target.Arms[arm]
		frozen := result.Prior.Arms[arm].UCB1

		assert.Zero(
			t,
			stats.Count,
		)

		assert.Zero(
			t,
			stats.RealCount,
		)

		assert.InDelta(
			t,
			0.25,
			stats.PriorObservationWeight,
			1e-12,
		)

		assert.InDelta(
			t,
			1.0,
			stats.PriorExplorationObservationWeight,
			1e-12,
		)

		assert.InDelta(
			t,
			frozen.RewardSum,
			stats.PriorRewardSum,
			1e-12,
		)
	}

	assert.InDelta(
		t,
		-1.8411330956755072,
		target.Arms["amd64"].PriorRewardSum,
		1e-12,
	)

	assert.InDelta(
		t,
		-1.850419488653634,
		target.Arms["arm64"].PriorRewardSum,
		1e-12,
	)
}

func TestMaterializedTransferControlRejectsInvalidRequest(
	t *testing.T,
) {
	oldManager :=
		mab.GlobalBanditManager

	viper.Reset()

	mab.InitBanditManager()

	t.Cleanup(
		func() {
			viper.Reset()

			mab.GlobalBanditManager =
				oldManager
		},
	)

	viper.Set(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		true,
	)

	viper.Set(
		config.LB_MODE,
		MAB,
	)

	e :=
		echo.New()

	RegisterTransferControlRoutes(
		e,
	)

	req :=
		httptest.NewRequest(
			http.MethodPost,
			TransferControlInitializeMaterializedPath,
			bytes.NewReader(
				[]byte(`{not-json}`),
			),
		)

	req.Header.Set(
		echo.HeaderContentType,
		echo.MIMEApplicationJSON,
	)

	rec :=
		httptest.NewRecorder()

	e.ServeHTTP(
		rec,
		req,
	)

	assert.Equal(
		t,
		http.StatusBadRequest,
		rec.Code,
	)
}
