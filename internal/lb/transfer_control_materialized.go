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

const TransferControlInitializeMaterializedPath = "/mab/transfer/initialize-materialized"

type transferControlInitializeMaterializedRequest struct {
	TargetFunctionName string           `json:"target_function_name"`
	Prior              mab.WeakMABPrior `json:"prior"`
}

type transferControlInitializeMaterializedResponse struct {
	TargetFunctionName   string           `json:"target_function_name"`
	DonorFunctionName    string           `json:"donor_function_name"`
	Policy               mab.BanditType   `json:"policy"`
	InitializationSource string           `json:"initialization_source"`
	TransferAttempted    bool             `json:"transfer_attempted"`
	TransferApplied      bool             `json:"transfer_applied"`
	RuntimeReason        string           `json:"runtime_reason"`
	Prior                mab.WeakMABPrior `json:"prior"`
}

func registerMaterializedTransferControlRoute(
	e *echo.Echo,
) {
	e.POST(
		TransferControlInitializeMaterializedPath,
		initializeTargetFromMaterializedPrior,
	)

	log.Printf(
		"[LB][MAB] materialized transfer control API enabled: POST %s\n",
		TransferControlInitializeMaterializedPath,
	)
}

func initializeTargetFromMaterializedPrior(
	c echo.Context,
) error {

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
				"error": "materialized transfer initialization requires lb.mode=MAB",
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

	var request transferControlInitializeMaterializedRequest

	decoder := json.NewDecoder(
		c.Request().Body,
	)

	if err := decoder.Decode(&request); err != nil {
		return c.JSON(
			http.StatusBadRequest,
			echo.Map{
				"error": "invalid materialized transfer control request: " + err.Error(),
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

	result, err :=
		mab.GlobalBanditManager.
			InitializeTargetFromMaterializedPrior(
				request.TargetFunctionName,
				request.Prior,
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
		transferControlInitializeMaterializedResponse{
			TargetFunctionName:   result.TargetFunctionName,
			DonorFunctionName:    result.DonorFunctionName,
			Policy:               result.Policy,
			InitializationSource: "materialized_prior",
			TransferAttempted:    true,
			TransferApplied:      result.Applied,
			RuntimeReason:        result.Reason,
			Prior:                result.Prior,
		},
	)
}
