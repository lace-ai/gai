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
