package profiling

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Shared file level helpers for the profiling datasets.
//
// The package writes three kinds of artifact raw invocation samples, function
// profiles and cold start profiles in two formats. Reading and publishing
// them follows the same two idioms everywhere, so both live here in a single
// implementation instead of being repeated per artifact.

const (
	// jsonLineInitialBufferBytes is the starting size of the scanner buffer.
	jsonLineInitialBufferBytes = 64 * 1024
	// jsonLineMaxBytes raises the bufio.Scanner limit above its 64 KiB default:
	// a profiling record contains many nested metrics and a valid JSONL line
	// would otherwise be rejected.
	jsonLineMaxBytes = 4 * 1024 * 1024
)

// scanJSONLines opens a JSONL dataset and invokes handle once per non-empty
// line, passing the 1-based line number so that callers can report errors
// against the source file.
//
// label names the artifact in the error messages ("profiling sample",
// "function profile", "cold start profile"). Errors returned by handle are
// propagated unchanged: the caller owns deserialization and validation.
func scanJSONLines(path string, label string, handle func(line []byte, lineNumber int) error) error {

	path = strings.TrimSpace(path)

	if path == "" {
		return fmt.Errorf("%s input path cannot be empty", label)
	}

	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open %s dataset %s: %w", label, path, err)
	}

	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, jsonLineInitialBufferBytes), jsonLineMaxBytes)
	lineNumber := 0

	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if err := handle([]byte(line), lineNumber); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("failed to read %s dataset %s: %w", label, path, err)
	}

	return nil
}

// createAtomicFile prepares the destination directory and opens a temporary
// file next to path, so that the export can be published with a rename.
//
// The temporary file is created in the destination directory on purpose: a
// rename is atomic only within the same filesystem.
func createAtomicFile(path string, pattern string, label string) (*os.File, string, error) {

	directory := filepath.Dir(path)
	if directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return nil, "", fmt.Errorf("failed to create %s directory %s: %w", label, directory, err)
		}
	}

	file, err := os.CreateTemp(directory, pattern)

	if err != nil {
		return nil, "", fmt.Errorf("failed to create temporary %s: %w", label, err)
	}

	return file, file.Name(), nil
}

// finalizeAtomicFile flushes the temporary file, hardens its permissions and
// publishes it atomically over path.
//
// On any failure before the rename the file is closed and the temporary copy is
// left on disk for the caller's deferred cleanup, so a partial export never
// replaces a valid dataset.
func finalizeAtomicFile(file *os.File, tempPath string, path string, label string) error {

	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("failed to sync temporary %s: %w", label, err)
	}

	if err := file.Chmod(0o644); err != nil {
		_ = file.Close()
		return fmt.Errorf("failed to set %s permissions: %w", label, err)
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("failed to close temporary %s: %w", label, err)
	}

	if err := os.Rename(tempPath, path); err != nil {
		return fmt.Errorf("failed to replace %s %s: %w", label, path, err)
	}

	return nil
}
