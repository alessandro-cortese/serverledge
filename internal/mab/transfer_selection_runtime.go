package mab

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

const (
	DonorSelectionArtifactSchemaVersion       = 1
	DonorSelectionQuerySchemaVersion          = 1
	DonorSelectionStatusSelected              = "selected"
	DonorSelectionStatusNoTransfer            = "no-transfer"
	SelectionRuntimeReasonSelectionNoTransfer = "selection_no_transfer"
)

// DonorSelectionArtifact contains the runtime-facing subset of the JSON
// produced by analysis/profiling/similarity_selection.py.
//
// Analysis-only fields such as ranking, architecture preference and durations
// are intentionally ignored by the runtime.
type DonorSelectionArtifact struct {
	SchemaVersion   int                          `json:"schema_version"`
	SelectionRunID  string                       `json:"selection_run_id"`
	Status          string                       `json:"status"`
	Reason          string                       `json:"reason"`
	Query           DonorSelectionQueryArtifact  `json:"query"`
	SelectionPolicy DonorSelectionPolicyArtifact `json:"selection_policy"`
	SelectedDonor   *SelectedDonorArtifact       `json:"selected_donor"`
	CandidateCount  int                          `json:"candidate_count"`
	BanditPrior     json.RawMessage              `json:"bandit_prior"`
}

type DonorSelectionQueryArtifact struct {
	SchemaVersion int    `json:"schema_version"`
	QueryID       string `json:"query_id"`
	FunctionName  string `json:"function_name"`
}

type DonorSelectionPolicyArtifact struct {
	Distance                   string  `json:"distance"`
	MaxDistance                float64 `json:"max_distance"`
	ConfigurationMatchRequired *bool   `json:"configuration_match_required"`
	RequireSameCluster         bool    `json:"require_same_cluster"`
	BanditPriorMaterialized    *bool   `json:"bandit_prior_materialized"`
}

type SelectedDonorArtifact struct {
	FunctionName string  `json:"function_name"`
	Distance     float64 `json:"distance"`
}

// SelectionRuntimeTransferResult keeps the similarity-selection result
// separate from the actual runtime-transfer result.
//
// A donor may be selected by similarity but still fail to produce a weak
// prior when its runtime MAB has insufficient real observations.
type SelectionRuntimeTransferResult struct {
	SelectionRunID            string `json:"selection_run_id"`
	TargetFunctionName        string `json:"target_function_name"`
	SelectionStatus           string `json:"selection_status"`
	SelectionReason           string `json:"selection_reason"`
	SelectedDonorFunctionName string `json:"selected_donor_function_name,omitempty"`

	// SelectionRequiredSameCluster reports whether the offline selection was
	// restricted to donors in the target's own cluster.
	//
	// The runtime does not require the restriction — an unrestricted selection
	// is simply a weaker condition — but the value must reach the log, because
	// without it a transfer cannot be traced back to the conditions under which
	// its donor was chosen.
	SelectionRequiredSameCluster bool         `json:"selection_required_same_cluster"`
	TransferAttempted            bool         `json:"transfer_attempted"`
	TransferApplied              bool         `json:"transfer_applied"`
	RuntimeReason                string       `json:"runtime_reason"`
	Prior                        WeakMABPrior `json:"prior"`
}

// DecodeDonorSelectionArtifact decodes and validates one JSON artifact produced
// by similarity_selection.py. It is used both by the file-based integration and
// by the experimental load-balancer control API, which receives the artifact
// directly in the request body.
func DecodeDonorSelectionArtifact(content []byte) (DonorSelectionArtifact, error) {

	if len(bytes.TrimSpace(content)) == 0 {
		return DonorSelectionArtifact{},
			fmt.Errorf("donor selection artifact cannot be empty")
	}

	var artifact DonorSelectionArtifact
	if err := json.Unmarshal(content, &artifact); err != nil {
		return DonorSelectionArtifact{}, fmt.Errorf("decode donor selection artifact: %w", err)
	}

	if err := validateDonorSelectionArtifact(artifact); err != nil {
		return DonorSelectionArtifact{}, fmt.Errorf("invalid donor selection artifact: %w", err)
	}

	return artifact, nil
}

// LoadDonorSelectionArtifact reads and validates one JSON artifact produced by
// similarity_selection.py.
func LoadDonorSelectionArtifact(path string) (DonorSelectionArtifact, error) {

	path = strings.TrimSpace(path)
	if path == "" {
		return DonorSelectionArtifact{}, fmt.Errorf("donor selection artifact path cannot be empty")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return DonorSelectionArtifact{}, fmt.Errorf("read donor selection artifact %q: %w", path, err)
	}

	artifact, err := DecodeDonorSelectionArtifact(content)

	if err != nil {
		return DonorSelectionArtifact{}, fmt.Errorf("load donor selection artifact %q: %w", path, err)
	}

	return artifact, nil
}

// InitializeTargetFromDonorSelectionFile connects a selection JSON stored on
// disk to the runtime donor->target initialization.
func (bm *BanditManager) InitializeTargetFromDonorSelectionFile(targetFunctionName string, selectionArtifactPath string, priorConfig WeakMABPriorConfig) (SelectionRuntimeTransferResult, error) {

	artifact, err := LoadDonorSelectionArtifact(selectionArtifactPath)
	if err != nil {
		return SelectionRuntimeTransferResult{}, err
	}

	return bm.InitializeTargetFromDonorSelectionArtifact(targetFunctionName, artifact, priorConfig)
}

// InitializeTargetFromDonorSelectionArtifact performs the same runtime
// integration as InitializeTargetFromDonorSelectionFile but receives an
// already decoded selection artifact. This is the control surface needed by
// the experimental bootstrap workflow: the shell script can send the JSON to
// the running load balancer without relying on a shared filesystem path.
func (bm *BanditManager) InitializeTargetFromDonorSelectionArtifact(targetFunctionName string, artifact DonorSelectionArtifact, priorConfig WeakMABPriorConfig) (SelectionRuntimeTransferResult, error) {

	if bm == nil {
		return SelectionRuntimeTransferResult{}, fmt.Errorf("bandit manager cannot be nil")
	}

	targetFunctionName = strings.TrimSpace(targetFunctionName)

	if targetFunctionName == "" {
		return SelectionRuntimeTransferResult{}, fmt.Errorf("target function name cannot be empty")
	}

	if err := validateWeakMABPriorConfig(priorConfig); err != nil {
		return SelectionRuntimeTransferResult{}, err
	}

	if err := validateDonorSelectionArtifact(artifact); err != nil {
		return SelectionRuntimeTransferResult{}, fmt.Errorf("invalid donor selection artifact: %w", err)
	}

	if artifact.Query.FunctionName != targetFunctionName {
		return SelectionRuntimeTransferResult{},
			fmt.Errorf("donor selection target mismatch: artifact query is for %q, requested target is %q", artifact.Query.FunctionName, targetFunctionName)
	}

	result := SelectionRuntimeTransferResult{
		SelectionRunID:               artifact.SelectionRunID,
		TargetFunctionName:           targetFunctionName,
		SelectionStatus:              artifact.Status,
		SelectionRequiredSameCluster: artifact.SelectionPolicy.RequireSameCluster,
		SelectionReason:              artifact.Reason,
	}

	switch artifact.Status {

	case DonorSelectionStatusSelected:
		donorFunctionName := artifact.SelectedDonor.FunctionName
		runtimeResult, err := bm.InitializeTargetFromDonor(targetFunctionName, donorFunctionName, priorConfig)
		if err != nil {
			return SelectionRuntimeTransferResult{}, fmt.Errorf("initialize target %q from selected donor %q: %w", targetFunctionName, donorFunctionName, err)
		}

		result.SelectedDonorFunctionName = donorFunctionName
		result.TransferAttempted = true
		result.TransferApplied = runtimeResult.Applied
		result.RuntimeReason = runtimeResult.Reason
		result.Prior = runtimeResult.Prior

	case DonorSelectionStatusNoTransfer:
		if err := bm.initializeFreshTargetWithoutTransfer(targetFunctionName); err != nil {
			return SelectionRuntimeTransferResult{}, err
		}

		result.TransferAttempted = false
		result.TransferApplied = false
		result.RuntimeReason = SelectionRuntimeReasonSelectionNoTransfer

	default:
		return SelectionRuntimeTransferResult{}, fmt.Errorf("unsupported donor selection status %q", artifact.Status)
	}

	logMABSelectionRuntimeTransfer(result)

	return result, nil
}

// initializeFreshTargetWithoutTransfer publishes a normal MAB initialized from
// zero when the similarity pipeline explicitly decided not to transfer.
func (bm *BanditManager) initializeFreshTargetWithoutTransfer(targetFunctionName string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if _, exists := bm.bandits[targetFunctionName]; exists {
		return fmt.Errorf("target function %q already has a MAB policy", targetFunctionName)
	}

	target := bm.newBanditLocked(targetFunctionName)
	bm.bandits[targetFunctionName] = target

	return nil
}

func validateDonorSelectionArtifact(artifact DonorSelectionArtifact) error {

	if artifact.SchemaVersion != DonorSelectionArtifactSchemaVersion {
		return fmt.Errorf("unsupported donor selection schema version %d", artifact.SchemaVersion)
	}

	if strings.TrimSpace(artifact.SelectionRunID) == "" {
		return fmt.Errorf("selection run ID cannot be empty")
	}

	if artifact.Query.SchemaVersion != DonorSelectionQuerySchemaVersion {
		return fmt.Errorf("unsupported transfer query schema version %d", artifact.Query.SchemaVersion)
	}

	if strings.TrimSpace(artifact.Query.QueryID) == "" {
		return fmt.Errorf("selection query ID cannot be empty")
	}

	if strings.TrimSpace(artifact.Query.FunctionName) == "" {
		return fmt.Errorf("selection query function name cannot be empty")
	}

	if artifact.SelectionPolicy.Distance != "euclidean" {
		return fmt.Errorf("unsupported donor selection distance %q", artifact.SelectionPolicy.Distance)
	}

	if !isFiniteNumber(artifact.SelectionPolicy.MaxDistance) || artifact.SelectionPolicy.MaxDistance <= 0.0 {
		return fmt.Errorf("selection max_distance must be finite and positive")
	}

	if artifact.SelectionPolicy.ConfigurationMatchRequired == nil || !*artifact.SelectionPolicy.ConfigurationMatchRequired {
		return fmt.Errorf("selection must explicitly require configuration matching")
	}

	if artifact.SelectionPolicy.BanditPriorMaterialized == nil {
		return fmt.Errorf("selection must explicitly report bandit_prior_materialized")
	}

	if *artifact.SelectionPolicy.BanditPriorMaterialized {
		return fmt.Errorf("10A selection artifact must not materialize a bandit prior")
	}

	if len(artifact.BanditPrior) == 0 || !bytes.Equal(bytes.TrimSpace(artifact.BanditPrior), []byte("null")) {
		return fmt.Errorf("10A selection artifact bandit_prior must be null")
	}

	if artifact.CandidateCount < 0 {
		return fmt.Errorf("candidate_count cannot be negative")
	}

	switch artifact.Status {

	case DonorSelectionStatusSelected:
		if strings.TrimSpace(artifact.Reason) != "" {
			return fmt.Errorf("selected donor artifact cannot contain a no-transfer reason")
		}

		if artifact.CandidateCount == 0 {
			return fmt.Errorf("selected donor artifact must contain at least one candidate")
		}

		if artifact.SelectedDonor == nil {
			return fmt.Errorf("selected donor artifact is missing selected_donor")
		}

		donorName := strings.TrimSpace(artifact.SelectedDonor.FunctionName)
		if donorName == "" {
			return fmt.Errorf("selected donor function name cannot be empty")
		}

		if donorName == artifact.Query.FunctionName {
			return fmt.Errorf("selected donor and query function must be different")
		}

		distance := artifact.SelectedDonor.Distance
		if !isFiniteNumber(distance) || distance < 0.0 {
			return fmt.Errorf("selected donor distance must be finite and non-negative")
		}

		if !selectionDistanceWithinThreshold(distance, artifact.SelectionPolicy.MaxDistance) {
			return fmt.Errorf("selected donor distance %.12f exceeds max_distance %.12f", distance, artifact.SelectionPolicy.MaxDistance)
		}

	case DonorSelectionStatusNoTransfer:
		if strings.TrimSpace(artifact.Reason) == "" {
			return fmt.Errorf("no-transfer artifact must contain a reason")
		}

		if artifact.SelectedDonor != nil {
			return fmt.Errorf("no-transfer artifact cannot contain selected_donor")
		}

	default:
		return fmt.Errorf("unsupported donor selection status %q", artifact.Status)
	}

	return nil
}

// similarity_selection.py accepts equality with the threshold using a very
// small numerical tolerance. Keep the Go validation compatible with it.
func selectionDistanceWithinThreshold(distance float64, maxDistance float64) bool {
	if distance <= maxDistance {
		return true
	}

	tolerance := math.Max(1e-12, 1e-12*math.Max(math.Abs(distance), math.Abs(maxDistance)))
	return math.Abs(distance-maxDistance) <= tolerance
}
