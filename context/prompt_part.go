package context

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

// Part is a token-countable unit that can produce a renderer-neutral node.
type Part interface {
	Name() string
	Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error)
	Render(ctx context.Context) (RenderNode, error)
}

// TextPart is a plain-text prompt part.
type TextPart struct {
	Content string
	tokens  map[string]int
}

// NewTextPart creates a text part with an empty token-count cache.
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

// MessagePart associates message content with a conversation role.
type MessagePart struct {
	Role    Role
	Content Content
	tokens  map[string]int
}

// NewMessagePart creates a role-aware message part.
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
		Type: roleRenderType(m.Role),
	}
	if m.Content == nil {
		return node, nil
	}
	child, err := m.Content.Render(ctx)
	if err != nil {
		return RenderNode{}, err
	}
	if child.Type == ContentTypeText && len(child.Fields) == 0 && len(child.Children) == 0 {
		node.Value = child.Value
		return node, nil
	}
	node.Children = []RenderNode{child}
	return node, nil
}

func roleRenderType(role Role) string {
	if IsValidRole(role) {
		return string(role)
	}
	return "message"
}

// SystemPart groups multiple parts under one system-instruction node.
type SystemPart struct {
	Instructions []Part
}

// NewSystemPart creates a grouped system-instruction part.
func NewSystemPart(instructions []Part) SystemPart {
	return SystemPart{
		Instructions: instructions,
	}
}

func (i SystemPart) Name() string {
	return "system"
}

func (i SystemPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	count := 0
	for _, part := range i.Instructions {
		tokens, err := part.Tokens(ctx, tokenizer)
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	return count, nil
}

func (i SystemPart) Render(ctx context.Context) (RenderNode, error) {
	node := RenderNode{Type: "instructions"}
	for _, part := range i.Instructions {
		child, err := part.Render(ctx)
		if err != nil {
			return RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
}
