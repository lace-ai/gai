package context

import (
	"context"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Turn struct {
	ID          string
	Count       int
	UserMessage *Message
	Messages    []Message
	TokenCount  map[string]int
}

type Message struct {
	ID        string
	SessionID string
	TurnID    string
	Role      Role
	Content   Content
	// TokenCount key: tokenizer.ID, value: token count for content
	TokenCount map[string]int
}

type TurnTokenStore interface {
	UpdateTurnTokens(ctx context.Context, turnID string, tokenizer string, tokens int) error
}

func (t *Turn) Tokenize(ctx context.Context, tokenizer ai.Tokenizer, store TurnTokenStore) (int, error) {
	if t == nil {
		return 0, ErrMessageNotFound
	}
	if tokenizer == nil {
		return 0, ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := t.TokenCount[tokenizerID]; ok {
		return count, nil
	}

	messages := t.messages()
	if count, ok := messagesTokenCount(messages, tokenizerID); ok {
		return t.saveTokens(ctx, store, tokenizerID, count)
	}

	count, err := tokenizer.CountTokens(ctx, combinedMessageContent(messages))
	if err != nil {
		return 0, err
	}
	return t.saveTokens(ctx, store, tokenizerID, count)
}

func (t *Turn) saveTokens(ctx context.Context, store TurnTokenStore, tokenizerID string, count int) (int, error) {
	if t.TokenCount == nil {
		t.TokenCount = make(map[string]int)
	}
	t.TokenCount[tokenizerID] = count
	if store == nil || t.ID == "" {
		return count, nil
	}
	if err := store.UpdateTurnTokens(ctx, t.ID, tokenizerID, count); err != nil {
		return 0, err
	}
	return count, nil
}

func (t *Turn) messages() []Message {
	if t == nil {
		return nil
	}
	messages := make([]Message, 0, len(t.Messages)+1)
	if t.UserMessage != nil {
		messages = append(messages, *t.UserMessage)
	}
	messages = append(messages, t.Messages...)
	return messages
}

func messagesTokenCount(messages []Message, tokenizerID string) (int, bool) {
	total := 0
	for _, message := range messages {
		count, ok := message.TokenCount[tokenizerID]
		if !ok {
			return 0, false
		}
		total += count
	}
	return total, true
}

func combinedMessageContent(messages []Message) string {
	var builder strings.Builder
	for i, message := range messages {
		if i > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(message.Content.String())
	}
	return builder.String()
}

func IsValidRole(role Role) bool {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return true
	default:
		return false
	}
}

func (m Message) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if count, ok := m.TokenCount[tokenizer.ID()]; ok {
		return count, nil
	}
	count, err := tokenizer.CountTokens(ctx, m.Content.String())
	if err != nil {
		return 0, err
	}
	if m.TokenCount == nil {
		m.TokenCount = make(map[string]int)
	}
	m.TokenCount[tokenizer.ID()] = count
	return count, nil
}

func (m Message) Marshal(ctx context.Context) ([]byte, error) {
	return m.Content.Marshal()
}

func (m Message) Name() string {
	return string(m.Role)
}
