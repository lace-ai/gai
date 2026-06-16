package context

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

type Part interface {
	Name() string
	Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error)
	Render(ctx context.Context) (RenderNode, error)
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

func (t TextPart) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{Type: "text", Value: t.Content}, nil
}

type MessagePart struct {
	Role    Role
	Content Content
	tokens  map[string]int
}

func NewMessagePart(role Role, content Content) MessagePart {
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
	content := ""
	if m.Content != nil {
		content = m.Content.String()
	}
	count, err := tokenizer.CountTokens(ctx, content)
	if err != nil {
		return 0, err
	}
	m.tokens[tokenizer.ID()] = count
	return count, nil
}

func (m MessagePart) Render(ctx context.Context) (RenderNode, error) {
	node := RenderNode{
		Type:   "message",
		Fields: []RenderField{{Key: "role", Value: string(m.Role)}},
	}
	if m.Content == nil {
		return node, nil
	}
	child, err := m.Content.Render(ctx)
	if err != nil {
		return RenderNode{}, err
	}
	node.Children = []RenderNode{child}
	return node, nil
}
