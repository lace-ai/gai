package context

import (
	stdcontext "context"
	"strings"
)

type HistorySource struct {
	store SessionStore
	id    int
	limit int
}

func History(store SessionStore, id int, limit int) Source {
	return &HistorySource{
		store: store,
		id:    id,
		limit: limit,
	}
}

func (s *HistorySource) BuildParts(ctx stdcontext.Context, conv Conversation) ([]Part, error) {
	if s == nil || s.store == nil {
		return nil, ErrSessionStoreNotFound
	}

	messages, err := s.store.GetMessages(s.id, s.limit, 0)
	if err != nil {
		return nil, err
	}

	parts := []Part{
		StaticPart("history", renderMessages(messages)),
	}
	if conv != nil {
		parts = append(parts, StaticPart("current-loop", renderMessages(conv.Messages())))
	}

	return parts, nil
}

func renderMessages(messages []Message) string {
	var builder strings.Builder
	RenderMessages(messages, &builder)
	return builder.String()
}
