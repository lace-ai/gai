package context_test

import (
	stdcontext "context"
	"testing"
	"time"

	aicontext "github.com/lace-ai/gai/context"
)

func TestHistorySourceBuildsStoredAndCurrentLoopParts(t *testing.T) {
	t.Parallel()

	store := &fakeSessionStore{
		messages: []aicontext.Message{
			{
				ID:        1,
				SessionID: 7,
				CreatedAt: time.Now(),
				Role:      aicontext.RoleUser,
				Content:   aicontext.NewTextContent("stored user"),
			},
			{
				ID:        2,
				SessionID: 7,
				CreatedAt: time.Now(),
				Role:      aicontext.RoleAssistant,
				Content:   aicontext.NewTextContent("stored assistant"),
			},
		},
	}
	conv := fakeConversation{
		messages: []aicontext.Message{
			{
				Role:    aicontext.RoleTool,
				Content: aicontext.NewTextContent("tool output"),
			},
		},
	}

	parts, err := aicontext.NewSessionManager(store, 7, 3).BuildParts(stdcontext.Background(), conv)
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}

	if store.gotSessionID != 7 || store.gotLimit != 3 || store.gotOffset != 0 {
		t.Fatalf("unexpected GetMessages args: session=%d limit=%d offset=%d", store.gotSessionID, store.gotLimit, store.gotOffset)
	}
	if len(parts) != 2 {
		t.Fatalf("expected history and current-loop parts, got %d", len(parts))
	}
	if parts[0].Name != "history" || parts[1].Name != "current-loop" {
		t.Fatalf("unexpected part names: %+v", parts)
	}
	if parts[0].Text == "" || parts[1].Text == "" {
		t.Fatalf("expected rendered messages in both parts: %+v", parts)
	}
}

type fakeConversation struct {
	messages []aicontext.Message
}

func (c fakeConversation) Messages() []aicontext.Message {
	return c.messages
}

type fakeSessionStore struct {
	messages []aicontext.Message

	gotSessionID int
	gotLimit     int
	gotOffset    int
}

func (s *fakeSessionStore) GetSession(sessionID int) error {
	return nil
}

func (s *fakeSessionStore) GetMessages(sessionID int, limit int, offset int) ([]aicontext.Message, error) {
	s.gotSessionID = sessionID
	s.gotLimit = limit
	s.gotOffset = offset
	return s.messages, nil
}

func (s *fakeSessionStore) CreateSession() (int, error) {
	return 0, nil
}

func (s *fakeSessionStore) AddMessages(sessionID int, messages []aicontext.Message) ([]aicontext.Message, error) {
	return messages, nil
}

func (s *fakeSessionStore) AddMessage(sessionID int, message aicontext.Message) (aicontext.Message, error) {
	return message, nil
}
