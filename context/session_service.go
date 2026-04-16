package context

import (
	"strings"
)

type Session struct {
	ID           int
	SessionStore SessionStore
}

func (s Session) Validate() error {
	if s.ID < 0 {
		return ErrInvalidSessionID
	}
	if s.SessionStore == nil {
		return ErrSessionStoreNotFound
	}
	return nil
}

func NewSession(id int, sessionStore SessionStore) *Session {
	return &Session{
		ID:           id,
		SessionStore: sessionStore,
	}
}

func (s Session) BuildHistory() (string, error) {
	if err := s.Validate(); err != nil {
		return "", err
	}
	var builder strings.Builder

	messages, err := s.SessionStore.GetMessages(s.ID, 5, 0)
	if err != nil {
		return "", err
	}

	RenderMessages(messages, &builder)
	return builder.String(), nil
}
