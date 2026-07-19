package mab

import "sync"

// ColdStartStats contains observability data for one function/arm pair.
//
// These values never influence arm selection or reward calculation.
type ColdStartStats struct {
	TotalInvocations int64

	ColdObserved int64
	WarmObserved int64

	ColdAccepted int64
	ColdSkipped  int64

	InvalidFeedback int64

	ColdDurationSamples int64
	WarmDurationSamples int64
	ColdInitTimeSamples int64

	ColdDurationSumMs float64
	WarmDurationSumMs float64
	ColdInitTimeSumMs float64
}

// ColdStartStatsSnapshot is an immutable copy of the collected statistics,
// including the derived averages.
type ColdStartStatsSnapshot struct {
	ColdStartStats

	AvgColdDurationMs float64
	AvgWarmDurationMs float64
	AvgColdInitTimeMs float64
}

// ColdStartStatsStore maintains statistics grouped by function and arm.
type ColdStartStatsStore struct {
	mu sync.RWMutex

	stats map[string]map[string]*ColdStartStats
}

func NewColdStartStatsStore() *ColdStartStatsStore {
	return &ColdStartStatsStore{
		stats: make(
			map[string]map[string]*ColdStartStats,
		),
	}
}

var GlobalColdStartStats = NewColdStartStatsStore()

func (s *ColdStartStatsStore) getOrCreateLocked(
	functionName string,
	arm string,
) *ColdStartStats {
	if s.stats == nil {
		s.stats =
			make(
				map[string]map[string]*ColdStartStats,
			)
	}

	if s.stats[functionName] == nil {
		s.stats[functionName] =
			make(
				map[string]*ColdStartStats,
			)
	}

	if s.stats[functionName][arm] == nil {
		s.stats[functionName][arm] =
			&ColdStartStats{}
	}

	return s.stats[functionName][arm]
}

// RecordObserved records an invocation regardless of whether it will later be
// accepted, skipped or rejected as invalid.
func (s *ColdStartStatsStore) RecordObserved(
	functionName string,
	arm string,
	feedback ExecutionFeedback,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stats :=
		s.getOrCreateLocked(
			functionName,
			arm,
		)

	stats.TotalInvocations++

	if feedback.IsWarmStart {
		stats.WarmObserved++

		if isFiniteNumber(
			feedback.DurationMs,
		) && feedback.DurationMs > 0 {

			stats.WarmDurationSamples++
			stats.WarmDurationSumMs +=
				feedback.DurationMs
		}

		return
	}

	stats.ColdObserved++

	if isFiniteNumber(
		feedback.DurationMs,
	) && feedback.DurationMs > 0 {

		stats.ColdDurationSamples++
		stats.ColdDurationSumMs +=
			feedback.DurationMs
	}

	if isFiniteNumber(
		feedback.InitTimeMs,
	) && feedback.InitTimeMs >= 0 {

		stats.ColdInitTimeSamples++
		stats.ColdInitTimeSumMs +=
			feedback.InitTimeMs
	}
}

func (s *ColdStartStatsStore) RecordColdAccepted(
	functionName string,
	arm string,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stats :=
		s.getOrCreateLocked(
			functionName,
			arm,
		)

	stats.ColdAccepted++
}

func (s *ColdStartStatsStore) RecordColdSkipped(
	functionName string,
	arm string,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stats :=
		s.getOrCreateLocked(
			functionName,
			arm,
		)

	stats.ColdSkipped++
}

func (s *ColdStartStatsStore) RecordInvalid(
	functionName string,
	arm string,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stats :=
		s.getOrCreateLocked(
			functionName,
			arm,
		)

	stats.InvalidFeedback++
}

func (s *ColdStartStatsStore) Snapshot(
	functionName string,
	arm string,
) ColdStartStatsSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()

	functionStats :=
		s.stats[functionName]

	if functionStats == nil ||
		functionStats[arm] == nil {

		return ColdStartStatsSnapshot{}
	}

	raw :=
		*functionStats[arm]

	snapshot :=
		ColdStartStatsSnapshot{
			ColdStartStats: raw,
		}

	if raw.ColdDurationSamples > 0 {
		snapshot.AvgColdDurationMs =
			raw.ColdDurationSumMs /
				float64(
					raw.ColdDurationSamples,
				)
	}

	if raw.WarmDurationSamples > 0 {
		snapshot.AvgWarmDurationMs =
			raw.WarmDurationSumMs /
				float64(
					raw.WarmDurationSamples,
				)
	}

	if raw.ColdInitTimeSamples > 0 {
		snapshot.AvgColdInitTimeMs =
			raw.ColdInitTimeSumMs /
				float64(
					raw.ColdInitTimeSamples,
				)
	}

	return snapshot
}

// Reset clears all collected process-local statistics.
// It is primarily useful for isolated tests and experiment restarts.
func (s *ColdStartStatsStore) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.stats =
		make(
			map[string]map[string]*ColdStartStats,
		)
}
