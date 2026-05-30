package context

import "context"

type SessionStore interface {
	GetSession(ctx context.Context, sessionID int) error
	// GetMessages returns newest messages in a session, ordered by created_at desc
	GetMessages(ctx context.Context, sessionID int, limit int, offset int) ([]Message, error)
	CreateSession(ctx context.Context) (int, error)
	AddMessages(ctx context.Context, sessionID int, messages []Message) ([]Message, error)
	AddMessage(ctx context.Context, sessionID int, message Message) (Message, error)
	UpdateMessageTokens(ctx context.Context, messageID int, tokenizer string, tokens int) error
}

type RAGStore interface {
	// GetRelevantDocuments returns relevant documents for a query, ordered by relevance desc
	GetRelevantDocuments(ctx context.Context, query string, limit int) ([]Document, error)
	AddDocument(ctx context.Context, content string) (int, error)
	UpdateDocumentTokens(ctx context.Context, documentID int, tokenizer string, tokens int) error
}
