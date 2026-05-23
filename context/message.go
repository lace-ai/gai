package context

import (
	"strconv"
	"strings"
	"time"
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
	CreatedAt time.Time
	Content   Content
	// TokenCount key: tokenizer.ID, value: token count for content
	TokenCount map[string]int
	Role       Role
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
