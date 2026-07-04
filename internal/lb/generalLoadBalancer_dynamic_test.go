package lb

import (
	"net/url"
	"testing"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/serverledge-faas/serverledge/internal/container"
	"github.com/serverledge-faas/serverledge/internal/function"
	"github.com/serverledge-faas/serverledge/internal/mab"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newDynamicTarget(name string, arch string, machineTag string) *middleware.ProxyTarget {
	return &middleware.ProxyTarget{
		Name: name,
		URL:  &url.URL{Scheme: "http", Host: name},
		Meta: echo.Map{
			"arch":        arch,
			"machine_tag": machineTag,
		},
	}
}

func newGeneralLbForTest(targets ...*middleware.ProxyTarget) *GeneralLoadBalancer {
	mab.InitBanditManager()
	return NewGeneralLoadBalancer(targets)
}

func TestGeneralLoadBalancerCreatesDynamicRingsFromMachineTags(t *testing.T) {
	b := newGeneralLbForTest(
		newDynamicTarget("arm1", container.ARM, "arm"),
		newDynamicTarget("x86_1", container.X86, "x86"),
		newDynamicTarget("gpu1", container.X86, "x86/gpu/nvidia"),
		newDynamicTarget("gpu2", container.X86, "x86/gpu/nvidia"),
	)

	require.Contains(t, b.rings, "arm")
	require.Contains(t, b.rings, "x86")
	require.Contains(t, b.rings, "x86/gpu/nvidia")

	assert.Equal(t, 1, b.rings["arm"].Size())
	assert.Equal(t, 1, b.rings["x86"].Size())
	assert.Equal(t, 2, b.rings["x86/gpu/nvidia"].Size())
	assert.ElementsMatch(t, []string{"arm", "x86", "x86/gpu/nvidia"}, b.architectures)
}

func TestGeneralLoadBalancerCompatibleTagsUseWildcardTagPattern(t *testing.T) {
	b := newGeneralLbForTest(
		newDynamicTarget("arm1", container.ARM, "arm"),
		newDynamicTarget("x86_1", container.X86, "x86"),
		newDynamicTarget("nvidia1", container.X86, "x86/gpu/nvidia"),
		newDynamicTarget("amd1", container.ARM, "arm/gpu/amd"),
	)

	fun := &function.Function{
		Name:           "gpuGeneric",
		MemoryMB:       128,
		SupportedArchs: []string{container.ARM, container.X86},
		TagPattern:     "*/gpu/*",
	}

	compatibleTags := b.compatibleTagsForFunction(fun)

	assert.ElementsMatch(t, []string{"x86/gpu/nvidia", "arm/gpu/amd"}, compatibleTags)
}

func TestGeneralLoadBalancerHardwarePatternCanRestrictToOneVendor(t *testing.T) {
	b := newGeneralLbForTest(
		newDynamicTarget("nvidia1", container.X86, "x86/gpu/nvidia"),
		newDynamicTarget("amd1", container.ARM, "arm/gpu/amd"),
	)

	fun := &function.Function{
		Name:           "gpuNvidiaOnly",
		MemoryMB:       128,
		SupportedArchs: []string{container.ARM, container.X86},
		TagPattern:     "*/gpu/nvidia",
	}

	compatibleTags := b.compatibleTagsForFunction(fun)

	assert.Equal(t, []string{"x86/gpu/nvidia"}, compatibleTags)
}

func TestGeneralLoadBalancerAddsNewMABArmWhenNewMachineTagAppears(t *testing.T) {
	b := newGeneralLbForTest(
		newDynamicTarget("x86_1", container.X86, "x86"),
	)

	policy := mab.GlobalBanditManager.GetBandit("dynamicFunction")
	ucb, ok := policy.(*mab.UCB1Bandit)
	require.True(t, ok, "default MAB policy should be UCB1 in this test")

	require.Contains(t, ucb.Arms, "x86")
	require.NotContains(t, ucb.Arms, "arm")

	added := b.AddTarget(newDynamicTarget("arm1", container.ARM, "arm"))
	require.True(t, added)

	assert.Contains(t, b.rings, "arm")
	assert.Contains(t, ucb.Arms, "arm", "AddTarget should call AddArmToAll for the newly discovered machine_tag")
}

func TestGeneralLoadBalancerMABCanSelectNewlyAddedArmForExploration(t *testing.T) {
	b := newGeneralLbForTest(
		newDynamicTarget("x86_1", container.X86, "x86"),
	)
	b.mode = MAB

	funcName := "mabDynamicFunction"
	fun := &function.Function{
		Name:           funcName,
		MemoryMB:       128,
		SupportedArchs: []string{container.ARM, container.X86},
	}

	policy := mab.GlobalBanditManager.GetBandit(funcName)
	ucb, ok := policy.(*mab.UCB1Bandit)
	require.True(t, ok, "default MAB policy should be UCB1 in this test")

	// Simulate that x86 has already been tried, then dynamically add arm.
	ucb.Arms["x86"].Count = 1
	ucb.TotalCounts = 1

	added := b.AddTarget(newDynamicTarget("arm1", container.ARM, "arm"))
	require.True(t, added)
	require.Contains(t, ucb.Arms, "arm")
	require.Equal(t, int64(0), ucb.Arms["arm"].Count)

	compatibleTags := b.compatibleTagsForFunction(fun)
	require.ElementsMatch(t, []string{"x86", "arm"}, compatibleTags)

	selected := b.selectTargetTag(funcName, fun, compatibleTags, nil)

	assert.Equal(t, "arm", selected, "UCB1 should force exploration of the newly added, untried arm")
}
