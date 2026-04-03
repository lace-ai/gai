package context

import "strconv"

type MemorySystem struct {
	Session Session
	Service MemoryService
}

func (m *MemorySystem) SessionID() string {
	return strconv.Itoa(m.Session.ID)
}

func (m *MemorySystem) AddMessage(content string, role Role) (Message, error) {
	return m.Service.AddMessage(content, role, m.Session.ID)
}

func (m *MemorySystem) GetMessages(limit int) ([]Message, error) {
	messages, err := m.Service.GetMessages(m.Session.ID)
	if err != nil {
		return nil, err
	}

	if limit > 0 && len(messages) > limit {
		return messages[len(messages)-limit:], nil
	}

	return messages, nil
}

func (m *MemorySystem) EnrichPrompt(prompt string) (string, error) {
	return m.Service.EnrichPrompt(m.Session.ID)
}
