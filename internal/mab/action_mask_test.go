package mab

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests explicitly verify that:
// - UCB1 does not select an arm excluded by the mask;
// - a single eligible arm results in a deterministic choice;
// - SelectArm without a mask continues to explore all arms;
// - an empty mask results in no selection;
// - LinUCB adheres to the same action mask

func newUCB1ForActionMaskTest() *UCB1Bandit {
	bandit := &UCB1Bandit{
		FunctionName: "action-mask-test",
		Arms:         make(map[string]*ArmStats),
		c:            0.8,
	}

	bandit.InitArm("x86-tiny")
	bandit.InitArm("arm-tiny")
	bandit.InitArm("x86-tiny/gpu/nvidia")

	return bandit
}

func TestUCB1SelectArmFromNeverSelectsMaskedArm(t *testing.T) {
	bandit := newUCB1ForActionMaskTest()

	// The GPU arm is untried and would be selected first without a mask.
	bandit.Arms["x86-tiny"].Count = 1
	bandit.Arms["x86-tiny"].AvgReward = 2.0

	bandit.Arms["arm-tiny"].Count = 1
	bandit.Arms["arm-tiny"].AvgReward = 1.0

	bandit.TotalCounts = 2

	selected := bandit.SelectArmFrom(
		nil,
		[]string{"x86-tiny", "arm-tiny"},
	)

	assert.Equal(t, "x86-tiny", selected)
	assert.NotEqual(t, "x86-tiny/gpu/nvidia", selected)
}

func TestUCB1SelectArmFromWithSingleAllowedArmIsDeterministic(
	t *testing.T,
) {
	bandit := newUCB1ForActionMaskTest()

	selected := bandit.SelectArmFrom(
		nil,
		[]string{"x86-tiny/gpu/nvidia"},
	)

	assert.Equal(t, "x86-tiny/gpu/nvidia", selected)
}

func TestUCB1SelectArmWithoutMaskStillExploresAllKnownArms(
	t *testing.T,
) {
	bandit := newUCB1ForActionMaskTest()

	bandit.Arms["arm-tiny"].Count = 1
	bandit.Arms["x86-tiny"].Count = 1
	bandit.TotalCounts = 2

	selected := bandit.SelectArm(nil)

	assert.Equal(t, "x86-tiny/gpu/nvidia", selected)
}

func TestSelectArmFromWithEmptyMaskReturnsNoArm(t *testing.T) {
	bandit := newUCB1ForActionMaskTest()

	selected := bandit.SelectArmFrom(
		nil,
		[]string{},
	)

	assert.Empty(t, selected)
}

func TestLinUCBSelectArmFromUsesOnlyAllowedArms(t *testing.T) {
	bandit := NewLinUCBDisjointPolicy(
		"linucb-action-mask-test",
		0.1,
	)

	bandit.InitArm("x86-tiny")
	bandit.InitArm("arm-tiny")
	bandit.InitArm("x86-tiny/gpu/nvidia")

	ctx := &Context{
		ArchMemUsage: map[string]float64{
			"x86-tiny":            0.20,
			"arm-tiny":            0.30,
			"x86-tiny/gpu/nvidia": 0.10,
		},
	}

	selected := bandit.SelectArmFrom(
		ctx,
		[]string{"arm-tiny"},
	)

	require.Equal(t, "arm-tiny", selected)
}
