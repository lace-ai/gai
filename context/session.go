package context

import (
	"strings"
)

type Session struct {
	ID       int
	Messages []Message
}

func (s Session) BuildHistory() string {
	var builder strings.Builder
	RenderMessages(s.Messages[len(s.Messages)-3:], &builder)
	return builder.String()
}
