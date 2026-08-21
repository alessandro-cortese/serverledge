package profiling

import (
	"bytes"
	"log"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLogInvocationResourceProfileQuotesRequestID(
	t *testing.T,
) {
	previousWriter := log.Writer()
	previousFlags := log.Flags()
	previousPrefix := log.Prefix()

	defer func() {
		log.SetOutput(previousWriter)
		log.SetFlags(previousFlags)
		log.SetPrefix(previousPrefix)
	}()

	var output bytes.Buffer

	log.SetOutput(&output)
	log.SetFlags(0)
	log.SetPrefix("")

	LogInvocationResourceProfile(
		"function-- x861234",
		"function",
		"x86",
		"node-a",
		true,
		&InvocationResourceProfile{
			Collected:                true,
			Valid:                    true,
			ProfilingStartOverheadMs: 12.5,
		},
	)

	line := output.String()

	assert.Contains(
		t,
		line,
		"profiling_start_overhead_ms=12.500000",
	)

	assert.True(
		t,
		strings.Contains(
			line,
			`request_id="function-- x861234"`,
		),
	)
}
