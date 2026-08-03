package container

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProfilingLockSerializesOneContainer(
	t *testing.T,
) {
	cont :=
		&Container{
			ID: "container-a",
		}

	cont.profilingMu.Lock()

	assert.False(
		t,
		cont.profilingMu.TryLock(),
		"a second profiling section must not enter the same container",
	)

	cont.profilingMu.Unlock()

	reacquired :=
		cont.profilingMu.TryLock()

	require.True(
		t,
		reacquired,
		"the profiling lock must be available after the active section ends",
	)

	cont.profilingMu.Unlock()
}

func TestProfilingLockIsPerContainer(
	t *testing.T,
) {
	first :=
		&Container{
			ID: "container-a",
		}

	second :=
		&Container{
			ID: "container-b",
		}

	first.profilingMu.Lock()

	acquired :=
		second.profilingMu.TryLock()

	require.True(
		t,
		acquired,
		"profiling one container must not block profiling another container",
	)

	second.profilingMu.Unlock()
	first.profilingMu.Unlock()
}
