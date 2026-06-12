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

type HistoryStateStore interface {
	GetHistoryState(ctx context.Context, turnID string) (*HistoryState, error)
	SaveHistoyrState(ctx context.Context, endTurnCount int, summary Summary) error
}

type HistorySource struct {
	store     SessionStore
	id        string
	debug     gai.DebugSink
	tokenizer ai.Tokenizer
}

func NewHistory(store SessionStore, id string) *HistorySource {
	return &HistorySource{
		store: store,
		id:    id,
	}
}

func (s *HistorySource) SetTokenizer(tokenizer ai.Tokenizer) {
	s.tokenizer = tokenizer
}

func (s *HistorySource) DebugSink(debug gai.DebugSink, conv Conversation) {
	s.debug = debug
}

func (s *HistorySource) BuildParts(ctx context.Context) ([]Part, error) {
	return nil, nil
}
