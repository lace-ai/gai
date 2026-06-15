package context

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

type Part interface {
	Name() string
	Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error)
	Marshal(ctx context.Context) ([]byte, error)
}

type TextPart struct {
	Content string
	tokens  map[string]int
}

func NewTextPart(content string) TextPart {
	return TextPart{
		Content: content,
		tokens:  make(map[string]int),
	}
}

func (t TextPart) Name() string {
	return "text"
}

func (t TextPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if t.tokens == nil {
		t.tokens = make(map[string]int)
	}
	if count, exists := t.tokens[tokenizer.ID()]; exists {
		return count, nil
	}
	count, err := tokenizer.CountTokens(ctx, t.Content)
	if err != nil {
		return 0, err
	}
	t.tokens[tokenizer.ID()] = count
	return count, nil
}

func (t TextPart) Marshal(ctx context.Context) ([]byte, error) {
	return []byte(t.Content), nil
}

type MessagePart struct {
	Role    string
	Content string
	tokens  map[string]int
}

func NewMessagePart(role, content string) MessagePart {
	return MessagePart{
		Role:    role,
		Content: content,
		tokens:  make(map[string]int),
	}
}

func (m MessagePart) Name() string {
	return "message"
}

func (m MessagePart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if m.tokens == nil {
		m.tokens = make(map[string]int)
	}
	if count, exists := m.tokens[tokenizer.ID()]; exists {
		return count, nil
	}
	count, err := tokenizer.CountTokens(ctx, m.Content)
	if err != nil {
		return 0, err
	}
	m.tokens[tokenizer.ID()] = count
	return count, nil
}

func (m MessagePart) Marshal(ctx context.Context) ([]byte, error) {
	return []byte(m.Role + ": " + m.Content), nil
}
