package context_test

import (
	stdcontext "context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	aicontext "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestHistorySourceBuildsPartsWithinTokenBudget(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []aicontext.Message{
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

	parts, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: conv}, historyBudget(17, tokenizer))
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
			t.Fatalf("part %q has token count %d, want %d", part.ID, part.Tokens, wantTokens)
		}
	}
	assertHistoryStoreQueries(t, store.GetMessagesCalls, 7)
}

func TestHistorySourceFailsWhenRequiredCurrentLoopExceedsBudget(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}
	conv := fakeConversation{
		messages: []aicontext.Message{
			sessionMessage(0, aicontext.RoleUser, "current message already fills budget"),
		},
	}

	_, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: conv}, historyBudget(6, whitespaceTokenizer{}))
	if !errors.Is(err, aicontext.ErrPromptBudget) {
		t.Fatalf("expected ErrPromptBudget, got %v", err)
	}
	if len(store.GetMessagesCalls) != 0 {
		t.Fatalf("expected no store calls when current loop reaches the budget, got %+v", store.GetMessagesCalls)
	}
}

func TestHistorySourceUsesEntryRequiredness(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}
	conv := fakeConversation{
		messages: []aicontext.Message{
			sessionMessage(0, aicontext.RoleUser, "current question"),
		},
	}

	parts, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: conv}, aicontext.SourceBudget{
		Tokenizer:             whitespaceTokenizer{},
		MaxTokens:             20,
		RemainingPromptTokens: 20,
		Required:              false,
	})
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}
	if len(parts) == 0 {
		t.Fatal("expected optional history parts")
	}
	for _, part := range parts {
		if part.Required {
			t.Fatalf("optional history source should not mark emitted parts required: %+v", parts)
		}
	}
}

func TestHistorySourceSkipsEmptyCurrentLoop(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}

	parts, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, rejectingEmptyTokenizer{}))
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}

	if len(parts) != 1 {
		t.Fatalf("expected only stored history, got %d parts: %+v", len(parts), parts)
	}
	if parts[0].ID != "history-0" {
		t.Fatalf("expected stored history part, got %q", parts[0].ID)
	}
}

func TestHistorySourcePropagatesStoreErrors(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("read failed")
	store := &mocks.MockSessionStore{Err: wantErr}

	_, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected store error %v, got %v", wantErr, err)
	}
}

func TestHistorySourcePropagatesTokenizerErrors(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("count failed")
	store := &mocks.MockSessionStore{
		Messages: []aicontext.Message{
			sessionMessage(1, aicontext.RoleUser, "stored one"),
		},
	}

	_, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{err: wantErr}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected tokenizer error %v, got %v", wantErr, err)
	}
}

func TestHistorySourceRequiresStore(t *testing.T) {
	t.Parallel()

	_, err := aicontext.History(nil, 7).BuildParts(stdcontext.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{}))
	if !errors.Is(err, aicontext.ErrSessionStoreNotFound) {
		t.Fatalf("expected ErrSessionStoreNotFound, got %v", err)
	}
}

func TestHistorySourceRequiresTokenizer(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{}

	_, err := aicontext.History(store, 7).BuildParts(stdcontext.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, nil))
	if !errors.Is(err, aicontext.ErrTokenizerNotFound) {
		t.Fatalf("expected ErrTokenizerNotFound, got %v", err)
	}
}

type whitespaceTokenizer struct {
	err error
}

type rejectingEmptyTokenizer struct{}

func (t rejectingEmptyTokenizer) Tokenize(ctx stdcontext.Context, text string) ([]string, error) {
	return whitespaceTokenizer{}.Tokenize(ctx, text)
}

func (t rejectingEmptyTokenizer) CountTokens(ctx stdcontext.Context, text string) (int, error) {
	if text == "" {
		return 0, errors.New("empty text")
	}
	return whitespaceTokenizer{}.CountTokens(ctx, text)
}

func (t whitespaceTokenizer) Tokenize(ctx stdcontext.Context, text string) ([]string, error) {
	if t.err != nil {
		return nil, t.err
	}
	return strings.Fields(text), nil
}

func (t whitespaceTokenizer) CountTokens(ctx stdcontext.Context, text string) (int, error) {
	tokens, err := t.Tokenize(ctx, text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

func historyBudget(maxTokens int, tokenizer ai.Tokenizer) aicontext.SourceBudget {
	return aicontext.SourceBudget{
		Tokenizer:             tokenizer,
		MaxTokens:             maxTokens,
		RemainingPromptTokens: maxTokens,
		Required:              true,
	}
}

type fakeConversation struct {
	messages []aicontext.Message
}

func (c fakeConversation) Messages() []aicontext.Message {
	return c.messages
}

type testPromptView struct {
	conv aicontext.Conversation
}

func (v testPromptView) Conversation() aicontext.Conversation {
	return v.conv
}

func (v testPromptView) Entries() []aicontext.EntryView {
	return nil
}

func (v testPromptView) SectionEntries(section aicontext.Section) []aicontext.EntryView {
	return nil
}

func (v testPromptView) Entry(id string) (aicontext.EntryView, bool) {
	return aicontext.EntryView{}, false
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

func assertHistoryStoreQueries(t *testing.T, calls []mocks.GetMessagesCall, sessionID int) {
	t.Helper()
	if len(calls) == 0 {
		t.Fatal("expected history source to query the session store")
	}
	for _, call := range calls {
		if call.SessionID != sessionID {
			t.Fatalf("unexpected session id in store call: got %+v want session %d", call, sessionID)
		}
		if call.Limit < 1 {
			t.Fatalf("expected positive history query limit, got %+v", call)
		}
		if call.Offset < 0 {
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
