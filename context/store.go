package context

import "context"

type SessionStore interface {
	GetSession(ctx context.Context, sessionID string) error
	// GetMessages returns newest messages in a session, ordered by created_at desc
	GetMessages(ctx context.Context, sessionID string, limit int, offset int) ([]Message, error)
	CreateSession(ctx context.Context) (string, error)
	AddMessages(ctx context.Context, sessionID string, messages []Message) ([]Message, error)
	AddMessage(ctx context.Context, sessionID string, message Message) (Message, error)
	UpdateMessageTokens(ctx context.Context, messageID string, tokenizer string, tokens int) error
}
