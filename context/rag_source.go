package context

import (
	"context"
)

type RAGStore interface {
	// GetRelevantDocuments returns relevant documents for a query, ordered by relevance desc
	GetRelevantDocuments(ctx context.Context, query string, limit int) ([]Document, error)
	AddDocument(ctx context.Context, content string) (string, error)
	UpdateDocumentTokens(ctx context.Context, documentID string, tokenizer string, tokens int) error
}

type Document struct {
	ID         string
	Content    string
	TokenCount map[string]int
}
