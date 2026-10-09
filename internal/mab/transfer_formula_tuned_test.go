package mab

import (
	"encoding/json"
	"math"
	"testing"
)

// These fixtures represent donor reward means derived from real warm feedback:
// 12 observations per architecture. The target anchor is measured only on x86.
// The test deliberately uses reward means, not duration ratios.
func tunedFormulaTestDonor(refMean, otherMean float64) TransferableMABKnowledge {
	return TransferableMABKnowledge{
		SchemaVersion:        TransferableMABKnowledgeSchemaVersion,
		FunctionName:         "known-donor",
		Policy:               UCB1Decoupled,
		HasRealKnowledge:     true,
		RealObservationCount: 24,
		Arms: map[string]TransferableArmKnowledge{
			"amd64": {
				RealObservationCount: 12,
				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: 12 * refMean,
					RealAvgReward:  refMean,
				},
			},
			"arm64": {
				RealObservationCount: 12,
				UCB1: &TransferableUCB1ArmKnowledge{
					RealSumRewards: 12 * otherMean,
					RealAvgReward:  otherMean,
				},
			},
		},
	}
}

func tunedFormulaTestConfig(mode UCB1ReferenceAnchorMode, wR, wE float64) WeakMABPriorConfig {
	return WeakMABPriorConfig{
		RewardObservationWeight:      wR,
		ExplorationObservationWeight: wE,
		MinRealObservationsPerArm:    1,
		UCB1ReferenceAnchor: &UCB1ReferenceAnchorConfig{
			Enabled:                   true,
			ReferenceArm:              "amd64",
			TargetReferenceMeanReward: -3.0,
			Mode:                      mode,
		},
	}
}

func TestTunedUCB1PriorDifferenceAndRatio(t *testing.T) {
	donor := tunedFormulaTestDonor(-7.0, -6.0)
	cases := []struct {
		name        string
		mode        UCB1ReferenceAnchorMode
		wR          float64
		wE          float64
		expectedARM float64
	}{
		{"difference_final", UCB1AnchorDifference, 0.05, 64, -2.0},
		{"ratio_final", UCB1AnchorRatio, 0.01, 64, -18.0 / 7.0},
		{"legacy_omitted_mode", "", 0.25, 1, -2.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tunedFormulaTestConfig(tc.mode, tc.wR, tc.wE)
			prior, err := BuildWeakMABPriorForTarget(donor, UCB1Decoupled, cfg)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			if err := validateWeakMABPriorForApplication(prior); err != nil {
				t.Fatalf("validate: %v", err)
			}
			for arm, expected := range map[string]float64{"amd64": -3.0, "arm64": tc.expectedARM} {
				a := prior.Arms[arm]
				if math.Abs(a.UCB1.MeanReward-expected) > 1e-12 {
					t.Errorf("%s mean got %.16g, want %.16g", arm, a.UCB1.MeanReward, expected)
				}
				if a.UCB1.ObservationWeight != tc.wR || a.UCB1.ExplorationObservationWeight != tc.wE {
					t.Errorf("%s weights got (%.3g, %.3g), want (%.3g, %.3g)", arm,
						a.UCB1.ObservationWeight, a.UCB1.ExplorationObservationWeight, tc.wR, tc.wE)
				}
				if a.AppliedExplorationObservationWeight != tc.wE {
					t.Errorf("%s exploration cap incorrectly applied: %.3g != %.3g", arm, a.AppliedExplorationObservationWeight, tc.wE)
				}
				if math.Abs(a.UCB1.RewardSum-tc.wR*expected) > 1e-12 {
					t.Errorf("%s reward sum mismatch", arm)
				}
			}

			// The materialized JSON must preserve the formula choice and large w_E.
			payload, err := json.Marshal(prior)
			if err != nil {
				t.Fatal(err)
			}
			var roundtrip WeakMABPrior
			if err := json.Unmarshal(payload, &roundtrip); err != nil {
				t.Fatal(err)
			}
			if roundtrip.Config.UCB1ReferenceAnchor.Mode != tc.mode {
				t.Fatalf("mode lost in JSON roundtrip: %q", roundtrip.Config.UCB1ReferenceAnchor.Mode)
			}

			target := NewUCB1DecoupledBandit("new-target", 0.2)
			target.InitArm("amd64")
			target.InitArm("arm64")
			if err := ApplyWeakMABPrior(target, roundtrip); err != nil {
				t.Fatalf("apply: %v", err)
			}
			for _, arm := range []string{"amd64", "arm64"} {
				state := target.Arms[arm]
				if state.PriorExplorationObservationWeight != tc.wE || state.PriorObservationWeight != tc.wR {
					t.Fatalf("%s target prior weights differ from materialized prior", arm)
				}
				if state.RealCount != 0 || state.Count != 0 {
					t.Fatalf("%s prior incorrectly counted as real/live feedback", arm)
				}
			}
			if selected := target.SelectArmFrom(nil, []string{"amd64", "arm64"}); selected != "arm64" {
				t.Fatalf("first selection = %q; want arm64", selected)
			}
			// A transferred prior is not re-exported as real donor experience.
			if knowledge := target.TransferableKnowledge(); knowledge.RealObservationCount != 0 {
				t.Fatalf("transferred prior leaked into real knowledge: %d", knowledge.RealObservationCount)
			}
		})
	}
}

func TestTunedUCB1RatioRejectsZeroDonorReference(t *testing.T) {
	donor := tunedFormulaTestDonor(0.0, -6.0)
	_, err := BuildWeakMABPriorForTarget(donor, UCB1Decoupled,
		tunedFormulaTestConfig(UCB1AnchorRatio, 0.01, 64))
	if err == nil {
		t.Fatal("ratio should fail when donor reference reward mean is zero")
	}
	// Unlike Ratio, Difference is still defined if the donor mean is zero.
	if _, err := BuildWeakMABPriorForTarget(donor, UCB1Decoupled,
		tunedFormulaTestConfig(UCB1AnchorDifference, 0.05, 64)); err != nil {
		t.Fatalf("difference should work with zero donor reference: %v", err)
	}
}

func TestTunedUCB1PriorValidationRejectsInvalidModesAndWeights(t *testing.T) {
	donor := tunedFormulaTestDonor(-7, -6)
	cases := []struct {
		name string
		cfg  WeakMABPriorConfig
	}{
		{"unknown_mode", tunedFormulaTestConfig("speedup", 0.05, 64)},
		{"nonpositive_exploration", tunedFormulaTestConfig(UCB1AnchorDifference, 0.05, 0)},
		{"infinite_exploration", tunedFormulaTestConfig(UCB1AnchorDifference, 0.05, math.Inf(1))},
		{"excessive_reward", tunedFormulaTestConfig(UCB1AnchorDifference, 1.1, 64)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := BuildWeakMABPriorForTarget(donor, UCB1Decoupled, tc.cfg); err == nil {
				t.Fatalf("invalid config unexpectedly accepted")
			}
		})
	}
}

func TestTunedUCB1MaterializedRejectsTamperedExploration(t *testing.T) {
	prior, err := BuildWeakMABPriorForTarget(tunedFormulaTestDonor(-7, -6),
		UCB1Decoupled, tunedFormulaTestConfig(UCB1AnchorDifference, 0.05, 64))
	if err != nil {
		t.Fatal(err)
	}
	arm := prior.Arms["amd64"]
	arm.AppliedExplorationObservationWeight = 65
	arm.UCB1.ExplorationObservationWeight = 65
	prior.Arms["amd64"] = arm
	if err := validateWeakMABPriorForApplication(prior); err == nil {
		t.Fatal("tampered w_E greater than configured value accepted")
	}
}
