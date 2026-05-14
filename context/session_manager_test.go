package context_test

import (
	stdcontext "context"
	"errors"
	"strings"
	"testing"
	"time"

	aicontext "github.com/lace-ai/gai/context"
)

func TestHistorySourceBuildsPartsWithinTokenBudget(t *testing.T) {
	t.Parallel()

	store := &fakeSessionStore{
		messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
			sessionMessage(2, aicontext.RoleAssistant, "stored two"),
			sessionMessage(3, aicontext.RoleUser, "stored message that does not fit"),
		},
	}
	tokenizer := whitespaceTokenizer{}
	conv := fakeConversation{
		messages: []aicontext.Message{
			sessionMessage(0, aicontext.RoleUser, "current question"),
		},
	}

	parts, err := aicontext.History(store, 7, 17, tokenizer).BuildParts(stdcontext.Background(), conv)
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}

	rendered := joinPartText(parts)
	assertHistoryContainsAll(t, rendered, "current question", "stored one", "stored two")
	assertHistoryContainsNone(t, rendered, "stored message that does not fit")

	if len(parts) != 3 {
		t.Fatalf("expected current loop plus fitting history parts, got %d: %+v", len(parts), parts)
	}
	for _, part := range parts {
		if !part.Required {
			t.Fatalf("expected all produced parts to be required: %+v", parts)
		}
		wantTokens := mustCountTokens(t, tokenizer, part.Text)
		if part.Tokens != wantTokens {
			t.Fatalf("part %q has token count %d, want %d", part.Name, part.Tokens, wantTokens)
		}
	}
	assertHistoryStoreQueries(t, store.calls, 7)
}

func TestHistorySourceDoesNotLoadStoredMessagesWhenCurrentLoopUsesBudget(t *testing.T) {
	t.Parallel()

	store := &fakeSessionStore{
		messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}
	conv := fakeConversation{
		messages: []aicontext.Message{
			sessionMessage(0, aicontext.RoleUser, "current message already fills budget"),
		},
	}

	parts, err := aicontext.History(store, 7, 6, whitespaceTokenizer{}).BuildParts(stdcontext.Background(), conv)
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}

	rendered := joinPartText(parts)
	assertHistoryContainsAll(t, rendered, "current message already fills budget")
	assertHistoryContainsNone(t, rendered, "stored one")
	if len(store.calls) != 0 {
		t.Fatalf("expected no store calls when current loop reaches the budget, got %+v", store.calls)
	}
}

func TestHistorySourcePropagatesStoreErrors(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("read failed")
	store := &fakeSessionStore{err: wantErr}

	_, err := aicontext.History(store, 7, 100, whitespaceTokenizer{}).BuildParts(stdcontext.Background(), fakeConversation{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected store error %v, got %v", wantErr, err)
	}
}

func TestHistorySourcePropagatesTokenizerErrors(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("count failed")
	store := &fakeSessionStore{
		messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}

	_, err := aicontext.History(store, 7, 100, whitespaceTokenizer{err: wantErr}).BuildParts(stdcontext.Background(), fakeConversation{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected tokenizer error %v, got %v", wantErr, err)
	}
}

func TestHistorySourceRequiresStore(t *testing.T) {
	t.Parallel()

	_, err := aicontext.History(nil, 7, 100, whitespaceTokenizer{}).BuildParts(stdcontext.Background(), fakeConversation{})
	if !errors.Is(err, aicontext.ErrSessionStoreNotFound) {
		t.Fatalf("expected ErrSessionStoreNotFound, got %v", err)
	}
}

func TestHistorySourceRequiresTokenizer(t *testing.T) {
	t.Parallel()

	store := &fakeSessionStore{}

	_, err := aicontext.History(store, 7, 100, nil).BuildParts(stdcontext.Background(), fakeConversation{})
	if !errors.Is(err, aicontext.ErrTokenizerNotFound) {
		t.Fatalf("expected ErrTokenizerNotFound, got %v", err)
	}
}

type whitespaceTokenizer struct {
	err error
}

func (whitespaceTokenizer) Tokenize(ctx stdcontext.Context, text string) []string {
	return strings.Fields(text)
}

func (t whitespaceTokenizer) CountTokens(ctx stdcontext.Context, text string) (int, error) {
	if t.err != nil {
		return 0, t.err
	}
	return len(t.Tokenize(ctx, text)), nil
}

type fakeConversation struct {
	messages []aicontext.Message
}

func (c fakeConversation) Messages() []aicontext.Message {
	return c.messages
}

type getMessagesCall struct {
	sessionID int
	limit     int
	offset    int
}

type fakeSessionStore struct {
	messages []aicontext.Message
	err      error
	calls    []getMessagesCall
}

func (s *fakeSessionStore) GetSession(sessionID int) error {
	return nil
}

func (s *fakeSessionStore) GetMessages(sessionID int, limit int, offset int) ([]aicontext.Message, error) {
	s.calls = append(s.calls, getMessagesCall{sessionID: sessionID, limit: limit, offset: offset})
	if s.err != nil {
		return nil, s.err
	}
	if offset >= len(s.messages) {
		return nil, nil
	}
	end := min(offset+limit, len(s.messages))
	return s.messages[offset:end], nil
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

func sessionMessage(id int, role aicontext.Role, text string) aicontext.Message {
	return aicontext.Message{
		ID:        id,
		SessionID: 7,
		CreatedAt: time.Now(),
		Role:      role,
		Content:   aicontext.NewTextContent(text),
	}
}

func joinPartText(parts []aicontext.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(part.Text)
		builder.WriteByte('\n')
	}
	return builder.String()
}

func assertHistoryContainsAll(t *testing.T, text string, values ...string) {
	t.Helper()
	for _, value := range values {
		if !strings.Contains(text, value) {
			t.Fatalf("expected %q to contain %q", text, value)
		}
	}
}

func assertHistoryContainsNone(t *testing.T, text string, values ...string) {
	t.Helper()
	for _, value := range values {
		if strings.Contains(text, value) {
			t.Fatalf("expected %q not to contain %q", text, value)
		}
	}
}

func assertHistoryStoreQueries(t *testing.T, calls []getMessagesCall, sessionID int) {
	t.Helper()
	if len(calls) == 0 {
		t.Fatal("expected history source to query the session store")
	}
	for _, call := range calls {
		if call.sessionID != sessionID {
			t.Fatalf("unexpected session id in store call: got %+v want session %d", call, sessionID)
		}
		if call.limit < 1 {
			t.Fatalf("expected positive history query limit, got %+v", call)
		}
		if call.offset < 0 {
			t.Fatalf("expected non-negative history query offset, got %+v", call)
		}
	}
}

func mustCountTokens(t *testing.T, tokenizer whitespaceTokenizer, text string) int {
	t.Helper()
	tokens, err := tokenizer.CountTokens(stdcontext.Background(), text)
	if err != nil {
		t.Fatalf("CountTokens failed: %v", err)
	}
	return tokens
}
