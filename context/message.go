package context

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/lace-ai/gai/ai"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	ID        int
	SessionID int
	Role      Role
	Content   Content
	// TokenCount key: tokenizer.ID, value: token count for content
	TokenCount map[string]int
	CreatedAt  time.Time
}

func IsValidRole(role Role) bool {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return true
	default:
		return false
	}
}

func RenderMessages(messages []Message, builder *strings.Builder) {
	for i, m := range messages {
		builder.WriteString("<")
		builder.WriteString(string(m.Role))
		builder.WriteString(" key=")
		builder.WriteString(strconv.Itoa(i))
		builder.WriteString(">\n")
		builder.WriteString(m.Content.String())
		builder.WriteString("\n")
		builder.WriteString("</")
		builder.WriteString(string(m.Role))
		builder.WriteString(">")
	}
}

func renderMessages(messages []Message) string {
	var builder strings.Builder
	RenderMessages(messages, &builder)
	return builder.String()
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
