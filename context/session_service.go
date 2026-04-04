package context

import (
	"strings"
)

type Session struct {
	ID           int
	SessionStore SessionStore
}

func NewSession(id int, sessionStore SessionStore) *Session {
	return &Session{
		ID:           id,
		SessionStore: sessionStore,
	}
}

func (s Session) BuildHistory() (string, error) {
	var builder strings.Builder

	messages, err := s.SessionStore.GetMessages(s.ID, 5, 0)
	if err != nil {
		return "", err
	}

	RenderMessages(messages, &builder)
	return builder.String(), nil
}
