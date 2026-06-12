package context_test

import (
	"context"
	"errors"
	"testing"

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

func TestTurnTokenizeRequiresTokenizer(t *testing.T) {
	t.Parallel()

	turn := gaictx.Turn{ID: "turn-1"}
	_, err := turn.Tokenize(context.Background(), nil, nil)
	if !errors.Is(err, gaictx.ErrTokenizerNotFound) {
		t.Fatalf("expected ErrTokenizerNotFound, got %v", err)
	}
}
