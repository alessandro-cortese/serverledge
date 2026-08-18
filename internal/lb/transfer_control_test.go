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
