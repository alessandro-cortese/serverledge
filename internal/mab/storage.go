package mab

import (
	"sync"
)

// ContextStorage temporarily stores the decision-time utilization snapshot
// until the corresponding execution completes.
//
// This allows contextual policies to update their model using exactly the
// same system state that was observed when the arm was selected.
type ContextStorage struct {
	// sync.Map is safe for concurrent use. So we use it since the LB can handle multiple request simultaneously.
	data sync.Map
}

var GlobalContextStorage = &ContextStorage{}

func (s *ContextStorage) Store(reqID string, ctx *Context) {
	s.data.Store(reqID, ctx)
}

func (s *ContextStorage) RetrieveAndDelete(reqID string) *Context {
	val, ok := s.data.LoadAndDelete(reqID)
	if !ok {
		return nil
	}
	return val.(*Context)
}
