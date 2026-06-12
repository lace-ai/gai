package context

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type HistoryState struct {
	Turns   []Turn
	Summary *Summary
}

type HistoryStore interface {
	GetLastHistoryState(ctx context.Context, sessionID string) (*HistoryState, error)
	SaveHistoyrState(ctx context.Context, endTurnCount int, summary Summary) error
}

type HistorySource struct {
	historyStateStore HistoryStore
	sessionID         string

	debug     gai.DebugSink
	tokenizer ai.Tokenizer
}

func NewHistory(sessionId string, historyStateStore HistoryStore) *HistorySource {
	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
	}
}

func (s *HistorySource) SetTokenizer(tokenizer ai.Tokenizer) {
	s.tokenizer = tokenizer
}

func (s *HistorySource) DebugSink(debug gai.DebugSink, conv Conversation) {
	s.debug = debug
}

type HistoryPart struct{}

func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (Part, error) {
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		return nil, err
	}
	if lastHistoryState != nil {
	}
	return nil, nil
}
