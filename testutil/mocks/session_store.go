package mocks

import (
	"context"
	"sync"

	gaictx "github.com/lace-ai/gai/context"
)

var _ gaictx.SessionStore = (*MockSessionStore)(nil)

type GetMessagesCall struct {
	Context   context.Context
	SessionID int
	Limit     int
	Offset    int
}

type GetSessionCall struct {
	Context   context.Context
	SessionID int
}

type CreateSessionCall struct {
	Context context.Context
}

type AddMessagesCall struct {
	Context   context.Context
	SessionID int
	Messages  []gaictx.Message
}

type AddMessageCall struct {
	Context   context.Context
	SessionID int
	Message   gaictx.Message
}

type UpdateMessageTokensCall struct {
	Context   context.Context
	MessageID int
	Tokenizer string
	Tokens    int
}

type MockSessionStore struct {
	Messages                 []gaictx.Message
	Err                      error
	CreateSessionID          int
	GetSessionCalls          []GetSessionCall
	GetMessagesCalls         []GetMessagesCall
	CreateCalls              []CreateSessionCall
	AddMessagesCalls         []AddMessagesCall
	AddMessageCalls          []AddMessageCall
	UpdateMessageTokensCalls []UpdateMessageTokensCall

	mu sync.Mutex
}

func (s *MockSessionStore) GetSession(ctx context.Context, sessionID int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.GetSessionCalls = append(s.GetSessionCalls, GetSessionCall{Context: ctx, SessionID: sessionID})
	return s.Err
}

func (s *MockSessionStore) GetMessages(ctx context.Context, sessionID int, limit int, offset int) ([]gaictx.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.GetMessagesCalls = append(s.GetMessagesCalls, GetMessagesCall{
		Context:   ctx,
		SessionID: sessionID,
		Limit:     limit,
		Offset:    offset,
	})
	if s.Err != nil {
		return nil, s.Err
	}
	if offset >= len(s.Messages) {
		return nil, nil
	}

	end := min(offset+limit, len(s.Messages))
	messages := make([]gaictx.Message, end-offset)
	copy(messages, s.Messages[offset:end])
	return messages, nil
}

func (s *MockSessionStore) CreateSession(ctx context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.CreateCalls = append(s.CreateCalls, CreateSessionCall{Context: ctx})
	if s.Err != nil {
		return 0, s.Err
	}
	return s.CreateSessionID, nil
}

func (s *MockSessionStore) AddMessages(ctx context.Context, sessionID int, messages []gaictx.Message) ([]gaictx.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := make([]gaictx.Message, len(messages))
	copy(copied, messages)
	s.AddMessagesCalls = append(s.AddMessagesCalls, AddMessagesCall{
		Context:   ctx,
		SessionID: sessionID,
		Messages:  copied,
	})
	if s.Err != nil {
		return nil, s.Err
	}
	s.Messages = append(s.Messages, copied...)

	added := make([]gaictx.Message, len(copied))
	copy(added, copied)
	return added, nil
}

func (s *MockSessionStore) AddMessage(ctx context.Context, sessionID int, message gaictx.Message) (gaictx.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.AddMessageCalls = append(s.AddMessageCalls, AddMessageCall{
		Context:   ctx,
		SessionID: sessionID,
		Message:   message,
	})
	if s.Err != nil {
		return gaictx.Message{}, s.Err
	}
	s.Messages = append(s.Messages, message)
	return message, nil
}

func (s *MockSessionStore) UpdateMessageTokens(ctx context.Context, messageID int, tokenizer string, tokens int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.UpdateMessageTokensCalls = append(s.UpdateMessageTokensCalls, UpdateMessageTokensCall{
		Context:   ctx,
		MessageID: messageID,
		Tokenizer: tokenizer,
		Tokens:    tokens,
	})
	if s.Err != nil {
		return s.Err
	}
	for i := range s.Messages {
		if s.Messages[i].ID == messageID {
			if s.Messages[i].TokenCount == nil {
				s.Messages[i].TokenCount = map[string]int{}
			}
			s.Messages[i].TokenCount[tokenizer] = tokens
			break
		}
	}
	return nil
}
