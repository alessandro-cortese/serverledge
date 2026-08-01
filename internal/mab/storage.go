package mab

import (
	"sync"
)

const (
	// FallbackReasonSelectedArmNoCandidate indicates that the ring selected by
	// the MAB could not provide a concrete execution candidate.
	FallbackReasonSelectedArmNoCandidate = "selected_arm_no_candidate"

	// FallbackReasonObservedExecutionDiffers is used defensively when the
	// execution tag returned by the node differs from the selected arm and no
	// more specific fallback reason was recorded by the load balancer.
	FallbackReasonObservedExecutionDiffers = "observed_execution_differs_from_selected"
)

// DecisionRecord tracks the complete lifecycle of one MAB decision.
//
// SelectedArm is the arm chosen by the policy. ExecutionArm is the ring chosen
// by the load balancer after candidate lookup and any fallback. The response
// header is still treated as the authoritative execution arm when feedback is
// processed.
type DecisionRecord struct {
	RequestID      string
	FunctionName   string
	SelectedArm    string
	ExecutionArm   string
	Context        *Context
	Fallback       bool
	FallbackReason string
}

type DecisionStorage struct {
	data sync.Map
}

var GlobalDecisionStorage = &DecisionStorage{}

func (s *DecisionStorage) Store(
	reqID string,
	decision DecisionRecord,
) {
	if reqID == "" {
		return
	}

	decision.RequestID =
		reqID

	s.data.Store(
		reqID,
		decision,
	)

	logMABDecisionCreated(
		decision,
	)
}

// SetExecutionPlan completes the routing part of a decision after the load
// balancer has selected a concrete candidate.
func (s *DecisionStorage) SetExecutionPlan(
	reqID string,
	executionArm string,
	fallbackReason string,
) (DecisionRecord, bool) {
	if reqID == "" {
		return DecisionRecord{}, false
	}

	value, ok :=
		s.data.Load(
			reqID,
		)

	if !ok {
		return DecisionRecord{}, false
	}

	decision, ok :=
		value.(DecisionRecord)

	if !ok {
		return DecisionRecord{}, false
	}

	decision.ExecutionArm =
		executionArm

	decision.Fallback =
		executionArm != "" &&
			decision.SelectedArm !=
				executionArm

	if decision.Fallback {
		if fallbackReason == "" {
			fallbackReason =
				FallbackReasonObservedExecutionDiffers
		}

		decision.FallbackReason =
			fallbackReason
	} else {
		decision.FallbackReason =
			""
	}

	s.data.Store(
		reqID,
		decision,
	)

	logMABDecisionPlanned(
		decision,
	)

	return decision, true
}

func (s *DecisionStorage) RetrieveAndDelete(
	reqID string,
) (DecisionRecord, bool) {
	if reqID == "" {
		return DecisionRecord{}, false
	}

	value, ok :=
		s.data.LoadAndDelete(
			reqID,
		)

	if !ok {
		return DecisionRecord{}, false
	}

	decision, ok :=
		value.(DecisionRecord)

	if !ok {
		return DecisionRecord{}, false
	}

	return decision, true
}

func (s *DecisionStorage) Reset() {
	s.data.Range(
		func(
			key any,
			_ any,
		) bool {
			s.data.Delete(
				key,
			)

			return true
		},
	)
}

type DecisionStatsSnapshot struct {
	DirectExecutions   int64
	FallbackExecutions int64
	CancelledDecisions int64
}

type DecisionStats struct {
	mu sync.RWMutex

	directExecutions   int64
	fallbackExecutions int64
	cancelledDecisions int64
}

var GlobalDecisionStats = &DecisionStats{}

func (s *DecisionStats) RecordResolved(
	fallback bool,
) DecisionStatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	if fallback {
		s.fallbackExecutions++
	} else {
		s.directExecutions++
	}

	return s.snapshotLocked()
}

func (s *DecisionStats) RecordCancelled() DecisionStatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cancelledDecisions++

	return s.snapshotLocked()
}

func (s *DecisionStats) Snapshot() DecisionStatsSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.snapshotLocked()
}

func (s *DecisionStats) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.directExecutions = 0
	s.fallbackExecutions = 0
	s.cancelledDecisions = 0
}

func (s *DecisionStats) snapshotLocked() DecisionStatsSnapshot {
	return DecisionStatsSnapshot{
		DirectExecutions: s.directExecutions,

		FallbackExecutions: s.fallbackExecutions,

		CancelledDecisions: s.cancelledDecisions,
	}
}

// ResolveDecisionWithoutFeedback closes the selected arm without updating any
// learned reward. It is used for missing candidates, proxy failures and
// malformed or unusable responses.
func ResolveDecisionWithoutFeedback(
	decision DecisionRecord,
	reason string,
) bool {
	if reason != "" {
		decision.FallbackReason =
			reason
	}

	resolved := false

	if GlobalBanditManager != nil {
		bandit :=
			GlobalBanditManager.
				GetBandit(
					decision.FunctionName,
				)

		bandit.ResolveSelection(
			decision.SelectedArm,
			"",
			decision.Context,
			nil,
		)

		resolved = true
	}

	stats :=
		GlobalDecisionStats.
			RecordCancelled()

	logMABDecisionCancelled(
		decision,
		reason,
		stats,
		GlobalBanditManager != nil,
	)

	return resolved
}

// ResolveDecisionWithFeedback closes the selected arm and applies the real
// feedback to the arm reported by the execution node.
func ResolveDecisionWithFeedback(
	decision DecisionRecord,
	observedExecutionArm string,
	feedback ExecutionFeedback,
) bool {
	if observedExecutionArm == "" {
		return false
	}

	if decision.ExecutionArm != "" &&
		decision.ExecutionArm !=
			observedExecutionArm {

		logMABExecutionArmMismatch(
			decision,
			observedExecutionArm,
		)
	}

	// The response header is authoritative because it identifies the node/ring
	// that actually executed the invocation.
	decision.ExecutionArm =
		observedExecutionArm

	decision.Fallback =
		decision.SelectedArm !=
			decision.ExecutionArm

	if decision.Fallback {
		if decision.FallbackReason == "" {
			decision.FallbackReason =
				FallbackReasonObservedExecutionDiffers
		}
	} else {
		decision.FallbackReason =
			""
	}

	if GlobalBanditManager == nil {
		return false
	}

	bandit :=
		GlobalBanditManager.
			GetBandit(
				decision.FunctionName,
			)

	bandit.ResolveSelection(
		decision.SelectedArm,
		decision.ExecutionArm,
		decision.Context,
		&feedback,
	)

	stats :=
		GlobalDecisionStats.
			RecordResolved(
				decision.Fallback,
			)

	logMABDecisionResolved(
		decision,
		stats,
	)

	return true
}

// CancelDecision removes a stored decision and closes its pending selection
// without applying a reward. Calling it repeatedly for the same request is safe.
func CancelDecision(
	reqID string,
	reason string,
) bool {
	decision, ok :=
		GlobalDecisionStorage.
			RetrieveAndDelete(
				reqID,
			)

	if !ok {
		return false
	}

	ResolveDecisionWithoutFeedback(
		decision,
		reason,
	)

	return true
}
