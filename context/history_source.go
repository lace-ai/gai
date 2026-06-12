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
	s.emit(ctx, "history_source_build_started", map[string]any{
		"session_id":   s.sessionID,
		"token_budget": tokenBudget,
	}, nil)

	if s.historyStateStore == nil {
		s.emit(ctx, "history_source_store_missing", map[string]any{
			"session_id": s.sessionID,
		}, ErrSessionStoreNotFound)
		return nil, ErrSessionStoreNotFound
	}
	if s.tokenizer == nil {
		s.emit(ctx, "history_source_tokenizer_missing", map[string]any{
			"session_id": s.sessionID,
		}, ErrTokenizerNotFound)
		return nil, ErrTokenizerNotFound
	}
	tokenizerID := s.tokenizer.ID()
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		s.emit(ctx, "history_source_state_load_failed", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, err)
		return nil, err
	}
	var part HistoryPart
	tokenCount := 0
	turnCount := 0
	messageCount := 0
	if lastHistoryState == nil {
		s.emit(ctx, "history_source_state_missing", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, nil)
	} else {
		if lastHistoryState.Summary != nil {
			part.Contents = append(part.Contents, lastHistoryState.Summary.Content)
			tokenCount += lastHistoryState.Summary.TokenCount[tokenizerID]
			summaryFields := map[string]any{
				"session_id":          s.sessionID,
				"tokenizer_id":        tokenizerID,
				"summary_tokens":      lastHistoryState.Summary.TokenCount[tokenizerID],
				"summary_start_turn":  lastHistoryState.Summary.StartTurnID,
				"summary_end_turn":    lastHistoryState.Summary.EndTurnID,
				"summary_start_count": lastHistoryState.Summary.StartTurnCount,
				"summary_end_count":   lastHistoryState.Summary.EndTurnCount,
			}
			if s.debug != nil && s.debug.IncludeSensitiveData() {
				summaryFields["summary_content"] = lastHistoryState.Summary.Content.String()
			}
			s.emit(ctx, "history_source_summary_included", summaryFields, nil)
		} else {
			s.emit(ctx, "history_source_summary_missing", map[string]any{
				"session_id":   s.sessionID,
				"tokenizer_id": tokenizerID,
			}, nil)
		}

		for _, turn := range lastHistoryState.Turns {
			turnCount++
			turnMessages := 0
			if turn.UserMessage != nil {
				part.Contents = append(part.Contents, turn.UserMessage.Content)
				messageCount++
				turnMessages++
			}
			for _, message := range turn.Messages {
				part.Contents = append(part.Contents, message.Content)
				messageCount++
				turnMessages++
			}
			tokens, err := turn.Tokenize(ctx, s.tokenizer, s.historyStateStore)
			if err != nil {
				s.emit(ctx, "history_source_turn_tokenize_failed", map[string]any{
					"session_id":   s.sessionID,
					"tokenizer_id": tokenizerID,
					"turn_id":      turn.ID,
					"turn_count":   turn.Count,
				}, err)
				return nil, err
			}
			tokenCount += tokens
			s.emit(ctx, "history_source_turn_included", map[string]any{
				"session_id":    s.sessionID,
				"tokenizer_id":  tokenizerID,
				"turn_id":       turn.ID,
				"turn_count":    turn.Count,
				"turn_messages": turnMessages,
				"turn_tokens":   tokens,
				"total_tokens":  tokenCount,
			}, nil)
			if tokenCount > tokenBudget {
				s.emit(ctx, "history_source_token_budget_reached", map[string]any{
					"session_id":    s.sessionID,
					"tokenizer_id":  tokenizerID,
					"token_budget":  tokenBudget,
					"total_tokens":  tokenCount,
					"last_turn_id":  turn.ID,
					"last_turn_cnt": turn.Count,
				}, nil)
				break
			}
		}
	}
	part.saveTokens(tokenizerID, tokenCount)
	s.emit(ctx, "history_source_build_finished", map[string]any{
		"session_id":    s.sessionID,
		"tokenizer_id":  tokenizerID,
		"token_budget":  tokenBudget,
		"total_tokens":  tokenCount,
		"turn_count":    turnCount,
		"message_count": messageCount,
		"content_count": len(part.Contents),
	}, nil)
	return &part, nil
}

func (s *HistorySource) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if s == nil || s.debug == nil {
		return
	}
	s.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:HistorySource",
		Fields: fields,
		Err:    err,
	})
}
