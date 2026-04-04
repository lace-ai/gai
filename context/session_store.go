package context

type SessionStore interface {
	GetSession(sessionID int) (*Session, error)
	GetMessages(sessionID int, limit int, offset int) ([]Message, error)
}
