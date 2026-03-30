package memory

import (
	"sync"
	"time"
)

type Repository struct {
	mu       sync.RWMutex
	counter  int
	messages []Message
}

func (r *Repository) GetMessagesBySession(sessionID int) ([]Message, error) {
	if sessionID <= 0 {
		return nil, ErrSessionIDInvalid
	}

	var res []Message
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, message := range r.messages {
		if message.SessionID == sessionID {
			res = append(res, message)
		}
	}

	return res, nil
}

func (r *Repository) AddMessage(content string, role Role, sessionID int) (Message, error) {
	if sessionID <= 0 {
		return Message{}, ErrSessionIDInvalid
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	message := Message{
		ID:        r.counter,
		SessionID: sessionID,
		CreatedAt: time.Now(),
		Content:   content,
		Role:      role,
	}
	r.counter++

	r.messages = append(r.messages, message)
	return message, nil
}
