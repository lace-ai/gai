package context_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lace-ai/gai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

type turnTokenUpdate struct {
	turnID    string
	tokenizer string
	tokens    int
}

type turnTokenStore struct {
	updates []turnTokenUpdate
	err     error
}

func (s *turnTokenStore) UpdateTurnTokens(ctx context.Context, turnID string, tokenizer string, tokens int) error {
	s.updates = append(s.updates, turnTokenUpdate{
		turnID:    turnID,
		tokenizer: tokenizer,
		tokens:    tokens,
	})
	return s.err
}

func TestTurnTokenizeUsesExistingTurnCount(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID:         "turn-1",
		TokenCount: map[string]int{"mock.tokenizer": 7},
		Messages: []gaictx.Message{
			{Content: gaictx.NewTextContent("should not be counted")},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), tokenizer, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 7 {
		t.Fatalf("expected cached turn tokens, got %d", tokens)
	}
	if tokenizer.CountCalls != 0 {
		t.Fatalf("expected tokenizer not to be called, got %d calls", tokenizer.CountCalls)
	}
	if len(store.updates) != 0 {
		t.Fatalf("expected cached count not to be saved again, got %+v", store.updates)
	}
}

func TestTurnTokenizeSumsExistingMessageCounts(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID: "turn-1",
		UserMessage: &gaictx.Message{
			Content:    gaictx.NewTextContent("hello"),
			TokenCount: map[string]int{"mock.tokenizer": 1},
		},
		Messages: []gaictx.Message{
			{
				Content:    gaictx.NewTextContent("assistant response"),
				TokenCount: map[string]int{"mock.tokenizer": 2},
			},
			{
				Content:    gaictx.NewToolResultContent("tool", "result text", false, ""),
				TokenCount: map[string]int{"mock.tokenizer": 3},
			},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), tokenizer, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 6 {
		t.Fatalf("expected summed message tokens, got %d", tokens)
	}
	if tokenizer.CountCalls != 0 {
		t.Fatalf("expected tokenizer not to be called, got %d calls", tokenizer.CountCalls)
	}
	if turn.TokenCount["mock.tokenizer"] != 6 {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 {
		t.Fatalf("expected one turn token update, got %+v", store.updates)
	}
	if store.updates[0].turnID != "turn-1" || store.updates[0].tokens != 6 {
		t.Fatalf("unexpected turn token update: %+v", store.updates[0])
	}
}

func TestTurnTokenizeCountsCombinedMessagesWithoutUpdatingMessages(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID: "turn-1",
		UserMessage: &gaictx.Message{
			Content: gaictx.NewTextContent("hello user"),
		},
		Messages: []gaictx.Message{
			{Content: gaictx.NewTextContent("assistant response")},
			{Content: gaictx.NewToolResultContent("tool", "tool result", false, "")},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), tokenizer, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens == 0 {
		t.Fatal("expected combined messages to be counted")
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected one combined tokenizer call, got %d calls", tokenizer.CountCalls)
	}
	if turn.UserMessage.TokenCount != nil {
		t.Fatalf("expected user message token count to stay untouched, got %+v", turn.UserMessage.TokenCount)
	}
	for _, message := range turn.Messages {
		if message.TokenCount != nil {
			t.Fatalf("expected message token count to stay untouched, got %+v", message.TokenCount)
		}
	}
	if turn.TokenCount["mock.tokenizer"] != tokens {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 {
		t.Fatalf("expected one turn token update, got %+v", store.updates)
	}
}

func TestTurnTokenizeHandlesNilMessageContent(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	turn := gaictx.Turn{
		ID:          "turn-1",
		UserMessage: &gaictx.Message{Role: gaictx.RoleUser},
		Messages:    []gaictx.Message{{Role: gaictx.RoleAssistant}},
	}

	tokens, err := turn.Tokenize(context.Background(), tokenizer, nil)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 0 {
		t.Fatalf("expected nil content to contribute no tokens, got %d", tokens)
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected one tokenizer call, got %d", tokenizer.CountCalls)
	}
}

func TestTurnTokenizeRequiresTokenizer(t *testing.T) {
	t.Parallel()

	turn := gaictx.Turn{ID: "turn-1"}
	_, err := turn.Tokenize(context.Background(), nil, nil)
	if !errors.Is(err, gaictx.ErrTokenizerNotFound) {
		t.Fatalf("expected ErrTokenizerNotFound, got %v", err)
	}
}

func TestMessageTokensRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{Count: 4}
	message := gaictx.Message{
		Content:    gaictx.NewTextContent("hello world"),
		TokenCount: map[string]int{"mock.tokenizer": -1},
	}

	tokens, err := message.Tokens(context.Background(), tokenizer)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 4 {
		t.Fatalf("expected tokenizer to recount invalid cached value, got %d", tokens)
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected one tokenizer call, got %d", tokenizer.CountCalls)
	}
	if message.TokenCount["mock.tokenizer"] != 4 {
		t.Fatalf("expected cache to be updated, got %+v", message.TokenCount)
	}
}

func TestMessageTokensHandlesNilContent(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	message := gaictx.Message{}

	tokens, err := message.Tokens(context.Background(), tokenizer)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 0 {
		t.Fatalf("expected nil content to contribute no tokens, got %d", tokens)
	}
}

func TestTurnTokenizeIgnoresNegativeCachedMessageCounts(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{}
	store := &turnTokenStore{}
	turn := gaictx.Turn{
		ID: "turn-1",
		UserMessage: &gaictx.Message{
			Content:    gaictx.NewTextContent("hello"),
			TokenCount: map[string]int{"mock.tokenizer": -1},
		},
		Messages: []gaictx.Message{
			{
				Content:    gaictx.NewTextContent("assistant response"),
				TokenCount: map[string]int{"mock.tokenizer": 2},
			},
		},
	}

	tokens, err := turn.Tokenize(context.Background(), tokenizer, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 3 {
		t.Fatalf("expected turn tokens to be recounted from messages, got %d", tokens)
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected tokenizer to be called once, got %d calls", tokenizer.CountCalls)
	}
	if turn.TokenCount["mock.tokenizer"] != 3 {
		t.Fatalf("expected turn token count to be cached, got %+v", turn.TokenCount)
	}
	if len(store.updates) != 1 || store.updates[0].tokens != 3 {
		t.Fatalf("expected turn token update with repaired count, got %+v", store.updates)
	}
}

func TestTurnTokenizeEmitsDebugEventWhenSavingTokensFails(t *testing.T) {
	t.Parallel()

	saveErr := errors.New("save tokens")
	store := &turnTokenStore{err: saveErr}
	turn := gaictx.Turn{
		ID:    "turn-1",
		Count: 2,
		Messages: []gaictx.Message{
			{Content: gaictx.NewTextContent("three token message")},
		},
	}
	var event gai.DebugEvent
	turn.SetDebugSink(gai.DebugSinkFunc(func(_ context.Context, emitted gai.DebugEvent) {
		event = emitted
	}))

	tokens, err := turn.Tokenize(context.Background(), &mocks.MockTokenizer{}, store)
	if err != nil {
		t.Fatalf("Tokenize failed: %v", err)
	}
	if tokens != 3 {
		t.Fatalf("expected calculated token count despite save failure, got %d", tokens)
	}
	if event.Name != "turn_token_save_failed" {
		t.Fatalf("expected token save failure event, got %+v", event)
	}
	if event.Source != "context:Turn.Tokenize" {
		t.Fatalf("unexpected event source: %q", event.Source)
	}
	if !errors.Is(event.Err, saveErr) {
		t.Fatalf("expected save error on event, got %v", event.Err)
	}
	if event.Fields["turn_id"] != "turn-1" || event.Fields["turn_count"] != 2 ||
		event.Fields["tokenizer_id"] != "mock.tokenizer" || event.Fields["token_count"] != 3 {
		t.Fatalf("unexpected event fields: %+v", event.Fields)
	}
}
