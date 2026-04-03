package context

import (
	"fmt"
	"strings"
)

type MemoryService struct {
	repo *Repository
}

func (m *MemoryService) AddMessage(content string, role Role, sessionID int) (Message, error) {
	if strings.TrimSpace(content) == "" {
		return Message{}, fmt.Errorf("%w: content is empty", ErrMessageInvalid)
	}
	if !IsValidRole(role) {
		return Message{}, fmt.Errorf("%w: role %s is invalid", ErrMessageInvalid, role)
	}
	return m.repo.AddMessage(content, role, sessionID)
}

func (m *MemoryService) GetMessages(sessionID int) ([]Message, error) {
	return m.repo.GetMessagesBySession(sessionID)
}

func (m *MemoryService) EnrichPrompt(sessionID int) (string, error) {
	messages, err := m.repo.GetMessagesBySession(sessionID)
	if err != nil {
		return "", err
	}

	var builder strings.Builder

	builder.WriteString("<conversation>")
	RenderMessages(messages, &builder)
	builder.WriteString("</conversation>")

	return builder.String(), nil
}
