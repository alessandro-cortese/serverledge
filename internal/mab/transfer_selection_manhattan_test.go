package mab

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDonorSelectionArtifactAcceptsManhattanWithinClusterRanking(t *testing.T) {
	required := true
	materialized := false

	artifact := DonorSelectionArtifact{
		SchemaVersion:  DonorSelectionArtifactSchemaVersion,
		SelectionRunID: "manhattan-test",
		Status:         DonorSelectionStatusSelected,
		Query: DonorSelectionQueryArtifact{
			SchemaVersion: DonorSelectionQuerySchemaVersion,
			QueryID:       "query-1",
			FunctionName:  "target",
		},
		SelectionPolicy: DonorSelectionPolicyArtifact{
			Distance:                   "manhattan",
			MaxDistance:                10.0,
			ConfigurationMatchRequired: &required,
			RequireSameCluster:         true,
			BanditPriorMaterialized:    &materialized,
		},
		SelectedDonor: &SelectedDonorArtifact{
			FunctionName: "donor",
			Distance:     1.25,
		},
		CandidateCount: 3,
		BanditPrior:    json.RawMessage("null"),
	}

	require.NoError(t, validateDonorSelectionArtifact(artifact))
}
