package context

type SessionStore interface {
	GetSession(sessionID int) error
	GetMessages(sessionID int, limit int, offset int) ([]Message, error)
	CreateSession() (int, error)
}
