package memory

type Memory interface {
	SessionID() string
	AddMessage(content string, role Role) (Message, error)
	GetMessages(limit int) ([]Message, error)
	EnrichPrompt(prompt string) (string, error)
}

func NewMemory(sessionID int) (Memory, error) {
	if sessionID <= 0 {
		return nil, ErrSessionIDInvalid
	}
	session := Session{ID: sessionID}
	repo := &Repository{}
	service := MemoryService{
		repo: repo,
	}
	memorySystem := MemorySystem{
		Session: session,
		Service: service,
	}
	return &memorySystem, nil
}
