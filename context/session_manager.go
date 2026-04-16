package context

import (
	"strings"
)

type SessionManager struct {
	store SessionStore
	id    int
}

func NewSessionManager(store SessionStore, id int) *SessionManager {
	return &SessionManager{
		store: store,
		id:    id,
	}
}

func (s *SessionManager) BuildContext(conv Conversation) (string, error) {
	var builder strings.Builder

	messages, err := s.store.GetMessages(s.id, 5, 0)
	if err != nil {
		return "", err
	}
	RenderMessages(messages, &builder)
	builder.WriteString("\nCurrent Loop:\n")
	RenderMessages(conv.Messages(), &builder)

	return builder.String(), nil
}
