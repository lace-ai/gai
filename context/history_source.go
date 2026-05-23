package context

import (
	stdcontext "context"
	"strconv"

	"github.com/lace-ai/gai/ai"
)

type HistorySource struct {
	store SessionStore
	id    int
}

func History(store SessionStore, id int) Source {
	return &HistorySource{
		store: store,
		id:    id,
	}
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
	convParts := []Part{}
	var conv Conversation
	if view != nil {
		conv = view.Conversation()
	}
	if conv != nil {
		convMessages := conv.Messages()
		renderedConv := renderMessages(convMessages)
		if renderedConv != "" {
			convTokens, err := countRenderedMessages(ctx, budget.Tokenizer, convMessages, renderedConv)
			if err != nil {
				return nil, err
			}
			if convTokens > limit {
				if budget.Required && budget.Summarizer != nil {
					summary, err := budget.Summarizer.Summarize(ctx, SummaryRequest{
						ID:        "current-loop",
						Text:      renderedConv,
						MaxTokens: limit,
						Required:  true,
					})
					if err != nil {
						return nil, err
					}
					summaryTokens, err := budget.Tokenizer.CountTokens(ctx, summary)
					if err != nil {
						return nil, err
					}
					if summaryTokens > limit {
						return nil, promptBudgetError("current-loop", summaryTokens, limit)
					}
					convTokens = summaryTokens
					renderedConv = summary
				} else if budget.Required {
					return nil, promptBudgetError("current-loop", convTokens, limit)
				} else {
					return nil, nil
				}
			}
			tokens += convTokens
			convParts = append(convParts, newHistoryPart("current-loop", renderedConv, convTokens, budget.Required))
		}
	}

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
		messageTokens, err := countRenderedMessages(ctx, budget.Tokenizer, messages, rendered)
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

	parts = append(parts, convParts...)
	return parts, nil
}

func countRenderedMessages(ctx stdcontext.Context, tokenizer ai.Tokenizer, messages []Message, rendered string) (int, error) {
	if tokens, ok := storedMessageTokens(messages, tokenizer.ID()); ok {
		return tokens, nil
	}
	return tokenizer.CountTokens(ctx, rendered)
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
