package context

import (
	"context"

	"github.com/lace-ai/gai"
)

type HistoryState struct {
	Turns   []Turn
	Summary *Summary
}

type HistoryStateStore interface {
	GetHistoryState(ctx context.Context, turnID string) (*HistoryState, error)
	SaveHistoyrState(ctx context.Context, endTurnCount int, summary Summary) error
}

type HistorySource struct {
	store SessionStore
	id    string
	debug gai.DebugSink
}

func NewHistory(store SessionStore, id string) *HistorySource {
	return &HistorySource{
		store: store,
		id:    id,
	}
}

func (s *HistorySource) DebugSink(debug gai.DebugSink) {
	s.debug = debug
}

func (s *HistorySource) BuildParts(ctx context.Context) (parts []Part, err error) {
	return parts, nil
}
