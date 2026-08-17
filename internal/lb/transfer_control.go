package lb

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/mab"
)

const TransferControlInitializePath = "/mab/transfer/initialize"

type transferControlInitializeRequest struct {
	TargetFunctionName string                 `json:"target_function_name"`
	SelectionArtifact  json.RawMessage        `json:"selection_artifact"`
	PriorConfig        mab.WeakMABPriorConfig `json:"prior_config"`
}

// RegisterTransferControlRoutes registers the experimental control surface used
// by the shell-based bootstrap workflow. The endpoint is disabled by default
// and must be explicitly enabled through mab.transfer.control.enabled.
//
// This endpoint does not alter UCB1 or LinUCB. It only invokes the already
// validated donor-selection -> runtime initialization path inside the
// live load-balancer process, where GlobalBanditManager resides.
func RegisterTransferControlRoutes(e *echo.Echo) {
	if e == nil {
		return
	}

	if !config.GetBool(
		config.MAB_TRANSFER_CONTROL_ENABLED,
		false,
	) {
		log.Printf(
			"[LB][MAB] transfer control API disabled (%s=false)\n",
			config.MAB_TRANSFER_CONTROL_ENABLED,
		)

		return
	}

	e.POST(
		TransferControlInitializePath,
		initializeTargetFromSelection,
	)

	log.Printf(
		"[LB][MAB] transfer control API enabled: POST %s\n",
		TransferControlInitializePath,
	)
}

func transferControlProxySkipper(c echo.Context) bool {
	if c == nil {
		return false
	}

	return strings.HasPrefix(
		c.Request().URL.Path,
		"/mab/transfer/",
	)
}

func initializeTargetFromSelection(c echo.Context) error {
	if !strings.EqualFold(
		strings.TrimSpace(
			config.GetString(
				config.LB_MODE,
				RR,
			),
		),
		MAB,
	) {
		return c.JSON(
			http.StatusConflict,
			echo.Map{
				"error": "transfer initialization requires lb.mode=MAB",
			},
		)
	}

	if mab.GlobalBanditManager == nil {
		return c.JSON(
			http.StatusServiceUnavailable,
			echo.Map{
				"error": "MAB manager is not initialized",
			},
		)
	}

	var request transferControlInitializeRequest

	decoder :=
		json.NewDecoder(
			c.Request().Body,
		)

	if err :=
		decoder.Decode(
			&request,
		); err != nil {

		return c.JSON(
			http.StatusBadRequest,
			echo.Map{
				"error": "invalid transfer control request: " + err.Error(),
			},
		)
	}

	request.TargetFunctionName =
		strings.TrimSpace(
			request.TargetFunctionName,
		)

	if request.TargetFunctionName == "" {
		return c.JSON(
			http.StatusBadRequest,
			echo.Map{
				"error": "target_function_name cannot be empty",
			},
		)
	}

	artifact, err :=
		mab.DecodeDonorSelectionArtifact(
			request.SelectionArtifact,
		)

	if err != nil {
		return c.JSON(
			http.StatusBadRequest,
			echo.Map{
				"error": err.Error(),
			},
		)
	}

	result, err :=
		mab.GlobalBanditManager.
			InitializeTargetFromDonorSelectionArtifact(
				request.TargetFunctionName,
				artifact,
				request.PriorConfig,
			)

	if err != nil {
		return c.JSON(
			http.StatusConflict,
			echo.Map{
				"error": err.Error(),
			},
		)
	}

	return c.JSON(
		http.StatusOK,
		result,
	)
}
