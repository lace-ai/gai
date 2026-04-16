package context

type SessionStore interface {
	GetSession(sessionID int) error
	// GetMessages returns messages in a session, ordered by created_at desc
	GetMessages(sessionID int, limit int, offset int) ([]Message, error)
	CreateSession() (int, error)
	AddMessages(sessionID int, messages []Message) ([]Message, error)
	AddMessage(sessionID int, message Message) (Message, error)
}
