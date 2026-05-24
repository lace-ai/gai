package context

import (
	stdcontext "context"
	"strconv"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type HistorySource struct {
	store SessionStore
	id    int
	debug gai.DebugSink
}

func History(store SessionStore, id int) Source {
	return &HistorySource{
		store: store,
		id:    id,
	}
}

func (s *HistorySource) DebugSink(debug gai.DebugSink) {
	s.debug = debug
}

func (s *HistorySource) BuildParts(ctx stdcontext.Context, view PromptView, budget SourceBudget) ([]Part, error) {
	if s == nil || s.store == nil {
		return nil, ErrSessionStoreNotFound
	}
	if budget.Tokenizer == nil {
		return nil, ErrTokenizerNotFound
	}

	limit := budget.ContentLimit()
	if limit == unlimitedTokens {
		limit = budget.RemainingPromptTokens
	}
	if limit < 0 {
		limit = 0
	}

	tokens := 0
	parts := []Part{}
	historyOffset := 0
	for tokens < limit {
		messages, err := s.store.GetMessages(ctx, s.id, 1, historyOffset)
		if err != nil {
			return nil, err
		}
		if len(messages) == 0 {
			break
		}
		historyOffset += len(messages)

		rendered := renderMessages(messages)
		messageTokens, err := s.countRenderedMessages(ctx, s.store, budget.Tokenizer, messages)
		if err != nil {
			return nil, err
		}
		if tokens+messageTokens > limit {
			break
		}

		tokens += messageTokens
		part := newHistoryPart("history-"+strconv.Itoa(len(parts)), rendered, messageTokens, budget.Required)
		parts = append(parts, part)
	}

	return parts, nil
}

func (s *HistorySource) countRenderedMessages(ctx stdcontext.Context, store SessionStore, tokenizer ai.Tokenizer, messages []Message) (int, error) {
	if tokens, ok := storedMessageTokens(messages, tokenizer.ID()); ok {
		return tokens, nil
	}
	var totalTokens int
	for _, message := range messages {
		tokens, err := countMessageContentTokens(ctx, tokenizer, []Message{message})
		if err != nil {
			return 0, err
		}
		go func(message Message, tokens int) {
			innerCtx, cancel := stdcontext.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			err := store.UpdateMessageTokens(innerCtx, message.ID, tokenizer.ID(), tokens)
			if err != nil {
				if s.debug != nil {
					s.debug.Emit(ctx, gai.DebugEvent{
						Name:   "HistorySource",
						Source: "token_count_update_error",
						Fields: map[string]any{
							"message_id":   message.ID,
							"tokenizer_id": tokenizer.ID(),
							"error":        err.Error(),
						},
						Err: err,
					})
				}
			}
		}(message, tokens)
		totalTokens += tokens
	}
	return totalTokens, nil
}

func countMessageContentTokens(ctx stdcontext.Context, tokenizer ai.Tokenizer, messages []Message) (int, error) {
	var totalTokens int
	for _, message := range messages {
		tokens, err := tokenizer.CountTokens(ctx, message.Content.String())
		if err != nil {
			return 0, err
		}
		totalTokens += tokens
	}
	return totalTokens, nil
}

func storedMessageTokens(messages []Message, tokenizerID string) (int, bool) {
	tokens := 0
	for _, message := range messages {
		messageTokens, ok := message.TokenCount[tokenizerID]
		if !ok || messageTokens < 0 {
			return 0, false
		}
		tokens += messageTokens
	}
	return tokens, true
}

func newHistoryPart(id, text string, tokens int, required bool) Part {
	opts := []EntryOption{Tokens(tokens)}
	if required {
		opts = append(opts, Required())
	}
	return NewPart(id, text, opts...)
}
