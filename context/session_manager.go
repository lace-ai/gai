package context

import "strings"

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

func (s *SessionManager) BuildContext(iterations []Message) string {
	var builder strings.Builder

	for _, msg := range iterations {
		builder.WriteString(msg.Content.String())
		builder.WriteString("\n")
	}

	return builder.String()
}
