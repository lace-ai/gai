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

	TurnTokenStore
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

type HistoryPart struct {
	Contents   []Content
	TokenCount map[string]int
}

func (p *HistoryPart) Name() string {
	return "history"
}

func (p *HistoryPart) Marshal(ctx context.Context) ([]byte, error) {
	return nil, nil
}

func (p *HistoryPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if tokenizer == nil {
		return 0, ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := p.TokenCount[tokenizerID]; ok {
		return count, nil
	}

	count := 0
	for _, content := range p.Contents {
		tokens, err := tokenizer.CountTokens(ctx, content.String())
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	p.saveTokens(tokenizerID, count)
	return count, nil
}

func (p *HistoryPart) saveTokens(tokenizerID string, tokens int) {
	if p.TokenCount == nil {
		p.TokenCount = make(map[string]int)
	}
	p.TokenCount[tokenizerID] = tokens
}

func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (Part, error) {
	if s.tokenizer == nil {
		return nil, ErrTokenizerNotFound
	}
	tokenizerID := s.tokenizer.ID()
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		return nil, err
	}
	var part HistoryPart
	tokenCount := 0
	if lastHistoryState != nil {
		part.Contents = append(part.Contents, lastHistoryState.Summary.Content)
		tokenCount += lastHistoryState.Summary.TokenCount[tokenizerID]
		for _, turn := range lastHistoryState.Turns {
			part.Contents = append(part.Contents, turn.UserMessage.Content)
			for _, message := range turn.Messages {
				part.Contents = append(part.Contents, message.Content)
			}
			tokens, err := turn.Tokenize(ctx, s.tokenizer, s.historyStateStore)
			if err != nil {
				return nil, err
			}
			tokenCount += tokens
			if tokenCount > tokenBudget {
				break
			}
		}
	}
	part.saveTokens(tokenizerID, tokenCount)
	return &part, nil
}
