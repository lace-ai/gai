package context

import (
	stdcontext "context"
	"strconv"

	"github.com/lace-ai/gai/ai"
)

type HistorySource struct {
	store      SessionStore
	id         int
	tokenLimit int
	tokenizer  ai.Tokenizer
}

func History(store SessionStore, id, tokenLimit int, tokenizer ai.Tokenizer) Source {
	return &HistorySource{
		store:      store,
		id:         id,
		tokenLimit: tokenLimit,
		tokenizer:  tokenizer,
	}
}

func (s *HistorySource) BuildParts(ctx stdcontext.Context, conv Conversation) ([]Part, error) {
	if s == nil || s.store == nil {
		return nil, ErrSessionStoreNotFound
	}
	if s.tokenizer == nil {
		return nil, ErrTokenizerNotFound
	}

	tokens := 0

	convParts := []Part{}
	if conv != nil {
		renderedConv := renderMessages(conv.Messages())
		if renderedConv != "" {
			convTokens, err := s.tokenizer.CountTokens(ctx, renderedConv)
			if err != nil {
				return nil, err
			}
			tokens += convTokens
			convParts = append(convParts, StaticPart("current-loop", renderedConv).RequiredPart().WithTokens(convTokens))
		}
	}

	parts := []Part{}
	historyOffset := 0
	for tokens < s.tokenLimit {
		messages, err := s.store.GetMessages(s.id, 1, historyOffset)
		if err != nil {
			return nil, err
		}
		if len(messages) == 0 {
			break
		}
		historyOffset += len(messages)

		rendered := renderMessages(messages)
		messageTokens, err := s.tokenizer.CountTokens(ctx, rendered)
		if err != nil {
			return nil, err
		}
		if tokens+messageTokens > s.tokenLimit {
			break
		}

		tokens += messageTokens
		part := StaticPart("history-"+strconv.Itoa(len(parts)), rendered).RequiredPart().WithTokens(messageTokens)
		parts = append(parts, part)
	}

	parts = append(parts, convParts...)
	return parts, nil
}
