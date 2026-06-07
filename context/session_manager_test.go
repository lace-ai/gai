package context_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestHistorySourceBuildsPartsWithinTokenBudget(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []gaictx.Message{
			sessionMessage(1, gaictx.RoleUser, "stored one"),
			sessionMessage(2, gaictx.RoleAssistant, "stored two"),
			sessionMessage(3, gaictx.RoleUser, "stored message that does not fit"),
		},
	}
	tokenizer := whitespaceTokenizer{}
	conv := fakeConversation{
		messages: []gaictx.Message{
			sessionMessage(0, gaictx.RoleUser, "current question"),
		},
	}

	parts, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: conv}, historyBudget(4, tokenizer))
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}

	rendered := joinPartText(parts)
	assertHistoryContainsAll(t, rendered, "stored one", "stored two")
	assertHistoryContainsNone(t, rendered, "current question", "stored message that does not fit")

	if len(parts) != 2 {
		t.Fatalf("expected fitting stored history parts, got %d: %+v", len(parts), parts)
	}
	wantTokens := map[string]int{
		"history-0": 2,
		"history-1": 2,
	}
	for _, part := range parts {
		if !part.Required {
			t.Fatalf("expected all produced parts to be required: %+v", parts)
		}
		if part.Tokens != wantTokens[part.ID] {
			t.Fatalf("part %q has token count %d, want %d", part.ID, part.Tokens, wantTokens[part.ID])
		}
	}
	assertHistoryStoreQueries(t, store.GetMessagesCalls, 7)
}

func TestHistorySourceUsesStoredMessageTokenCounts(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	store := &mocks.MockSessionStore{
		Messages: []gaictx.Message{
			sessionMessageWithTokens(1, gaictx.RoleUser, "stored one", tokenizer.ID(), 2),
			sessionMessageWithTokens(2, gaictx.RoleAssistant, "stored two three", tokenizer.ID(), 3),
		},
	}
	conv := fakeConversation{
		messages: []gaictx.Message{
			sessionMessageWithTokens(0, gaictx.RoleUser, "current question", tokenizer.ID(), 2),
		},
	}

	parts, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: conv}, historyBudget(10, tokenizer))
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}
	if tokenizer.CountCalls != 0 {
		t.Fatalf("expected stored token counts to avoid tokenizer calls, got %d", tokenizer.CountCalls)
	}
	if len(parts) != 2 {
		t.Fatalf("expected stored history parts, got %d: %+v", len(parts), parts)
	}

	wantTokens := map[string]int{
		"history-0": 2,
		"history-1": 3,
	}
	for _, part := range parts {
		if part.Tokens != wantTokens[part.ID] {
			t.Fatalf("part %q has token count %d, want %d", part.ID, part.Tokens, wantTokens[part.ID])
		}
	}
}

func TestHistorySourceIgnoresCurrentLoopBudget(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []gaictx.Message{
			sessionMessage(1, gaictx.RoleUser, "stored one"),
		},
	}
	conv := fakeConversation{
		messages: []gaictx.Message{
			sessionMessage(0, gaictx.RoleUser, "current message already fills budget"),
		},
	}

	parts, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: conv}, historyBudget(6, whitespaceTokenizer{}))
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}
	rendered := joinPartText(parts)
	if strings.Contains(rendered, "current message already fills budget") {
		t.Fatalf("history source should not include current loop messages: %q", rendered)
	}
}

func TestHistorySourceUsesEntryRequiredness(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{
		Messages: []gaictx.Message{
			sessionMessage(1, gaictx.RoleUser, "stored one"),
		},
	}
	conv := fakeConversation{
		messages: []gaictx.Message{
			sessionMessage(0, gaictx.RoleUser, "current question"),
		},
	}

	parts, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: conv}, gaictx.SourceBudget{
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
		Messages: []gaictx.Message{
			sessionMessage(1, gaictx.RoleUser, "stored one"),
		},
	}

	parts, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, rejectingEmptyTokenizer{}))
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

	_, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected store error %v, got %v", wantErr, err)
	}
}

func TestHistorySourcePropagatesTokenizerErrors(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("count failed")
	store := &mocks.MockSessionStore{
		Messages: []gaictx.Message{
			sessionMessage(1, gaictx.RoleUser, "stored one"),
		},
	}

	_, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{err: wantErr}))
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected tokenizer error %v, got %v", wantErr, err)
	}
}

func TestHistorySourceRequiresStore(t *testing.T) {
	t.Parallel()

	_, err := gaictx.History(nil, 7).BuildParts(context.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, whitespaceTokenizer{}))
	if !errors.Is(err, gaictx.ErrSessionStoreNotFound) {
		t.Fatalf("expected ErrSessionStoreNotFound, got %v", err)
	}
}

func TestHistorySourceRequiresTokenizer(t *testing.T) {
	t.Parallel()

	store := &mocks.MockSessionStore{}

	_, err := gaictx.History(store, 7).BuildParts(context.Background(), testPromptView{conv: fakeConversation{}}, historyBudget(100, nil))
	if !errors.Is(err, gaictx.ErrTokenizerNotFound) {
		t.Fatalf("expected ErrTokenizerNotFound, got %v", err)
	}
}

type whitespaceTokenizer struct {
	err error
}

type rejectingEmptyTokenizer struct{}

func (t rejectingEmptyTokenizer) ID() string {
	return "test.rejecting-empty"
}

func (t rejectingEmptyTokenizer) Tokenize(ctx context.Context, text string) ([]string, error) {
	return whitespaceTokenizer{}.Tokenize(ctx, text)
}

func (t rejectingEmptyTokenizer) CountTokens(ctx context.Context, text string) (int, error) {
	if text == "" {
		return 0, errors.New("empty text")
	}
	return whitespaceTokenizer{}.CountTokens(ctx, text)
}

func (t whitespaceTokenizer) Tokenize(ctx context.Context, text string) ([]string, error) {
	if t.err != nil {
		return nil, t.err
	}
	return strings.Fields(text), nil
}

func (t whitespaceTokenizer) ID() string {
	return "test.whitespace"
}

func (t whitespaceTokenizer) CountTokens(ctx context.Context, text string) (int, error) {
	tokens, err := t.Tokenize(ctx, text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

func historyBudget(maxTokens int, tokenizer ai.Tokenizer) gaictx.SourceBudget {
	return gaictx.SourceBudget{
		Tokenizer:             tokenizer,
		MaxTokens:             maxTokens,
		RemainingPromptTokens: maxTokens,
		Required:              true,
	}
}

type fakeConversation struct {
	messages []gaictx.Message
}

func (c fakeConversation) Messages() []gaictx.Message {
	return c.messages
}

type testPromptView struct {
	conv gaictx.Conversation
}

func (v testPromptView) Conversation() gaictx.Conversation {
	return v.conv
}

func (v testPromptView) Entries() []gaictx.EntryView {
	return nil
}

func (v testPromptView) SectionEntries(section gaictx.Section) []gaictx.EntryView {
	return nil
}

func (v testPromptView) Entry(id string) (gaictx.EntryView, bool) {
	return gaictx.EntryView{}, false
}

func sessionMessage(id int, role gaictx.Role, text string) gaictx.Message {
	return gaictx.Message{
		ID:        id,
		CreatedAt: time.Now(),
		Role:      role,
		Content:   gaictx.NewTextContent(text),
	}
}

func sessionMessageWithTokens(id int, role gaictx.Role, text, tokenizerID string, tokens int) gaictx.Message {
	message := sessionMessage(id, role, text)
	message.TokenCount = map[string]int{tokenizerID: tokens}
	return message
}

func joinPartText(parts []gaictx.Part) string {
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
	tokens, err := tokenizer.CountTokens(context.Background(), text)
	if err != nil {
		t.Fatalf("CountTokens failed: %v", err)
	}
	return tokens
}
