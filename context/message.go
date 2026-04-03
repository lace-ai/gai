package context

import (
	"strconv"
	"strings"
	"time"

	"agent-backend/gai/loop"
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
	Role      Role
}

type Content struct {
	Text       string
	Iterations []loop.Iteration
}

func (c Content) String() string {
	var builder strings.Builder
	if c.Text != "" {
		builder.WriteString(c.Text)
	}

	for _, iter := range c.Iterations {
		builder.WriteString("\n")
		builder.WriteString(iter.String())
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
