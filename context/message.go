package context

import (
	"context"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

// Role identifies the participant represented by a message.
type Role string

const (
	// RoleSystem identifies system instructions.
	RoleSystem Role = "system"
	// RoleUser identifies user input.
	RoleUser Role = "user"
	// RoleAssistant identifies model output and tool requests.
	RoleAssistant Role = "assistant"
	// RoleTool identifies tool results returned to the model.
	RoleTool Role = "tool"
)

// Turn groups one user message with the assistant and tool messages it caused.
type Turn struct {
	ID          string
	Count       int
	UserMessage *Message
	Messages    []Message
	TokenCount  map[string]int
	debugSink   gai.DebugSink
}

// Message is one role-labelled conversation entry.
type Message struct {
	ID        string
	SessionID string
	TurnID    string
	Role      Role
	Content   Content
	// TokenCount key: tokenizer.ID, value: token count for content
	TokenCount map[string]int
}

// TurnTokenStore persists calculated token counts for a turn.
type TurnTokenStore interface {
	UpdateTurnTokens(ctx context.Context, turnID string, tokenizer string, tokens int) error
}

// Tokenize returns the turn token count, using cached message or turn counts
// when available and optionally persisting a newly calculated value.
func (t *Turn) Tokenize(ctx context.Context, tokenizer ai.Tokenizer, store TurnTokenStore) (int, error) {
	if t == nil {
		return 0, ErrMessageNotFound
	}
	if tokenizer == nil {
		return 0, ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := t.TokenCount[tokenizerID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(t.TokenCount, tokenizerID)
	}

	messages := t.messages()
	if count, ok := messagesTokenCount(messages, tokenizerID); ok {
		return t.saveTokens(ctx, store, tokenizerID, count)
	}

	count, err := tokenizer.CountTokens(ctx, combinedMessageContent(messages))
	if err != nil {
		return 0, err
	}
	_, err = t.saveTokens(ctx, store, tokenizerID, count)
	if err != nil {
		if t.debugSink != nil {
			t.debugSink.Emit(ctx, gai.DebugEvent{
				Name:   "turn_token_save_failed",
				Source: "context:Turn.Tokenize",
				Fields: map[string]any{
					"turn_id":      t.ID,
					"turn_count":   t.Count,
					"tokenizer_id": tokenizerID,
					"token_count":  count,
				},
				Err: err,
			})
		}
	}
	return count, nil
}

// SetDebugSink configures diagnostics for non-fatal turn operations.
func (t *Turn) SetDebugSink(sink gai.DebugSink) {
	t.debugSink = sink
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
		if !ok || count < 0 {
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
		if message.Content != nil {
			builder.WriteString(message.Content.String())
		}
	}
	return builder.String()
}

// IsValidRole reports whether role is one of the built-in roles.
func IsValidRole(role Role) bool {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return true
	default:
		return false
	}
}

// Tokens returns the message token count for tokenizer, caching the result.
func (m Message) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if tokenizer == nil {
		return 0, ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := m.TokenCount[tokenizerID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(m.TokenCount, tokenizerID)
	}
	content := ""
	if m.Content != nil {
		content = m.Content.String()
	}
	count, err := tokenizer.CountTokens(ctx, content)
	if err != nil {
		return 0, err
	}
	if m.TokenCount == nil {
		m.TokenCount = make(map[string]int)
	}
	m.TokenCount[tokenizerID] = count
	return count, nil
}
