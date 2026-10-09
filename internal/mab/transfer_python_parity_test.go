package mab

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The JSON fixture is created from the frozen Python build_prior + NumericTLPolicy.
// This test deliberately checks the real Go BuildWeakMABPriorForTarget,
// ApplyWeakMABPrior and SelectArmFrom implementations. To keep the replay
// synchronous and independent of integration-test infrastructure, each accepted
// -ln(duration_ms) reward is applied directly to the Go statistics after
// ResolveSelection clears the in-flight counter. Live feedback acceptance is
// tested separately in the existing MAB tests.
type pythonParityFixture struct {
	SchemaVersion int `json:"schema_version"`
	Cases         []struct {
		Mode   string             `json:"mode"`
		WR     float64            `json:"w_r"`
		WE     float64            `json:"w_e"`
		C      float64            `json:"c"`
		Anchor float64            `json:"anchor_mean_reward"`
		Donor  map[string]float64 `json:"donor_mean_reward"`
		Prior  map[string]float64 `json:"prior_mean_reward"`
		DonorN int64              `json:"donor_n_per_arm"`
		Steps  []struct {
			T        int                `json:"t"`
			Before   map[string]float64 `json:"scores_before"`
			Selected string             `json:"selected_arm"`
			Duration float64            `json:"duration_ms"`
			Reward   float64            `json:"reward"`
			After    map[string]float64 `json:"scores_after"`
			Counts   map[string]int64   `json:"counts_after"`
		} `json:"steps"`
	} `json:"cases"`
}

func parityClose(got, want float64) bool {
	return math.Abs(got-want) <= 1e-9*math.Max(1, math.Abs(want))
}

func parityGoScores(b *UCB1DecoupledBandit) map[string]float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	priorTotal := b.totalPriorExplorationObservationWeightLocked()
	total := float64(b.TotalCounts+b.TotalInFlight) + priorTotal
	if total < 1 {
		total = 1
	}
	scores := make(map[string]float64, len(b.Arms))
	for name, stats := range b.Arms {
		n := b.explorationArmObservationWeightLocked(stats)
		if n <= 0 {
			panic("expected positive exploration count")
		}
		bonus := 0.0
		if b.c != 0 {
			bonus = b.c * math.Sqrt(math.Log(total)/n)
		}
		scores[name] = b.effectiveArmAverageRewardLocked(stats) + bonus
	}
	return scores
}

func assertParityScores(t *testing.T, step int, stage string, actual, expected map[string]float64) {
	t.Helper()
	for _, arm := range []string{"amd64", "arm64"} {
		if !parityClose(actual[arm], expected[arm]) {
			t.Fatalf("step=%d %s %s score: Go %.16g, Python %.16g", step, stage, arm, actual[arm], expected[arm])
		}
	}
}

func TestPythonUCB1Parity(t *testing.T) {
	path := filepath.Join("testdata", "ucb1_python_parity.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing Python fixture (%s); generate it with generate_ucb1_parity_fixture.py: %v", path, err)
	}
	var fixture pythonParityFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SchemaVersion != 1 || len(fixture.Cases) != 3 {
		t.Fatal("wrong fixture schema/case count")
	}

	for i, test := range fixture.Cases {
		t.Run(fmt.Sprintf("%d_%s_wR%g_wE%g", i, test.Mode, test.WR, test.WE), func(t *testing.T) {
			donorArms := map[string]TransferableArmKnowledge{}
			for _, arm := range []string{"amd64", "arm64"} {
				mu := test.Donor[arm]
				donorArms[arm] = TransferableArmKnowledge{
					RealObservationCount: test.DonorN,
					UCB1: &TransferableUCB1ArmKnowledge{
						RealSumRewards: float64(test.DonorN) * mu,
						RealAvgReward:  mu,
					},
				}
			}
			donor := TransferableMABKnowledge{
				SchemaVersion: TransferableMABKnowledgeSchemaVersion,
				FunctionName:  "fixture-donor", Policy: UCB1Decoupled,
				HasRealKnowledge:     true,
				RealObservationCount: 2 * test.DonorN,
				Arms:                 donorArms,
			}
			cfg := WeakMABPriorConfig{
				RewardObservationWeight:      test.WR,
				ExplorationObservationWeight: test.WE,
				MinRealObservationsPerArm:    1,
				UCB1ReferenceAnchor: &UCB1ReferenceAnchorConfig{
					Enabled: true, ReferenceArm: "amd64",
					TargetReferenceMeanReward: test.Anchor,
					Mode:                      UCB1ReferenceAnchorMode(test.Mode),
				},
			}
			prior, err := BuildWeakMABPriorForTarget(donor, UCB1Decoupled, cfg)
			if err != nil {
				t.Fatal(err)
			}
			target := NewUCB1DecoupledBandit("fixture-target", test.C)
			for _, arm := range []string{"amd64", "arm64"} {
				target.InitArm(arm)
			}
			if err := ApplyWeakMABPrior(target, prior); err != nil {
				t.Fatal(err)
			}
			for arm, want := range test.Prior {
				if !parityClose(prior.Arms[arm].UCB1.MeanReward, want) {
					t.Fatalf("%s prior Go %.16g Python %.16g", arm, prior.Arms[arm].UCB1.MeanReward, want)
				}
			}
			for _, step := range test.Steps {
				assertParityScores(t, step.T, "before", parityGoScores(target), step.Before)
				arm := target.SelectArmFrom(nil, []string{"amd64", "arm64"})
				if arm != step.Selected {
					t.Fatalf("step %d: Go chose %s, Python chose %s", step.T, arm, step.Selected)
				}
				target.ResolveSelection(arm, "", nil, nil, nil)
				reward := -math.Log(step.Duration)
				if !parityClose(reward, step.Reward) {
					t.Fatal("reward mismatch")
				}
				target.mu.Lock()
				stats := target.Arms[arm]
				stats.Count++
				stats.RealCount++
				stats.SumRewards += reward
				stats.AvgReward = stats.SumRewards / float64(stats.Count)
				stats.RealSumRewards += reward
				stats.RealAvgReward = stats.RealSumRewards / float64(stats.RealCount)
				target.TotalCounts++
				target.mu.Unlock()
				for arm, want := range step.Counts {
					if target.Arms[arm].Count != want {
						t.Fatalf("step %d: %s count Go=%d Python=%d", step.T, arm, target.Arms[arm].Count, want)
					}
				}
				assertParityScores(t, step.T, "after", parityGoScores(target), step.After)
			}
			if target.TransferableKnowledge().RealObservationCount != int64(len(test.Steps)) {
				t.Fatal("prior leaked into donor's real observations or feedback count mismatch")
			}
		})
	}
}
