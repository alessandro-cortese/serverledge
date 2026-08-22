package profiling

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/serverledge-faas/serverledge/internal/config"
)

const DefaultInvocationSampleExportPath = "data/profiling/profiling-samples.jsonl"

var invocationSampleExportMu sync.Mutex

// ExportInvocationSample appends one complete JSON object followed by a newline
// to the configured dataset.
//
// Export failures are returned to the caller but never alter the result of the
// function invocation.
func ExportInvocationSample(sample InvocationSample) error {

	if !config.GetBool(config.FUNCTION_PROFILING_ENABLED, false) || !config.GetBool(config.FUNCTION_PROFILING_EXPORT_ENABLED, false) {
		return nil
	}

	path := strings.TrimSpace(config.GetString(config.FUNCTION_PROFILING_EXPORT_PATH, DefaultInvocationSampleExportPath))
	if path == "" {
		return fmt.Errorf("profiling export path cannot be empty")
	}

	payload, err := json.Marshal(sample)

	if err != nil {
		return fmt.Errorf("failed to marshal profiling sample: %w", err)
	}

	line := append(payload, '\n')
	invocationSampleExportMu.Lock()

	defer invocationSampleExportMu.Unlock()

	directory := filepath.Dir(path)
	if directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return fmt.Errorf("failed to create profiling export directory %s: %w", directory, err)
		}
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("failed to open profiling export file %s: %w", path, err)
	}

	defer file.Close()

	written, err := file.Write(line)
	if err != nil {
		return fmt.Errorf("failed to append profiling sample to %s: %w", path, err)
	}

	if written != len(line) {
		return fmt.Errorf("short profiling sample write to %s: wrote %d of %d bytes", path, written, len(line))
	}

	return nil
}
