package openai

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestNewTokenizerResolvesKnownOpenAIModels(t *testing.T) {
	tests := []struct {
		model  string
		wantID string
		text   string
		want   int
	}{
		{model: "gpt-5", wantID: "openai.tiktoken-go/v0.8.1:o200k_base", text: "hello", want: 1},
		{model: "o1", wantID: "openai.tiktoken-go/v0.8.1:o200k_base", text: "hello", want: 1},
		{model: GPT41, wantID: "openai.tiktoken-go/v0.8.1:o200k_base", text: "hello", want: 1},
		{model: GPT4oMini, wantID: "openai.tiktoken-go/v0.8.1:o200k_base", text: "hello", want: 1},
		{model: "gpt-4-0125-preview", wantID: "openai.tiktoken-go/v0.8.1:cl100k_base", text: `{"tool":"weather","city":"München"}`, want: 10},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			tokenizer, err := NewTokenizer(tt.model)
			if err != nil {
				t.Fatalf("NewTokenizer(%q) error: %v", tt.model, err)
			}
			if tokenizer.ID() != tt.wantID {
				t.Fatalf("ID() = %q, want %q", tokenizer.ID(), tt.wantID)
			}
			got, err := tokenizer.CountTokens(context.Background(), tt.text)
			if err != nil || got != tt.want {
				t.Fatalf("CountTokens(%q) = %d, %v; want %d, nil", tt.text, got, err, tt.want)
			}
			tokens, err := tokenizer.Tokenize(context.Background(), tt.text)
			if err != nil {
				t.Fatalf("Tokenize(%q) error: %v", tt.text, err)
			}
			if len(tokens) != got {
				t.Fatalf("len(Tokenize(%q)) = %d, want CountTokens result %d", tt.text, len(tokens), got)
			}
		})
	}
}

func TestNewTokenizerRejectsUnknownModelWithTypedUnavailability(t *testing.T) {
	_, err := NewTokenizer("gpt-future-999")
	if err == nil {
		t.Fatal("NewTokenizer() error = nil, want unavailable error")
	}
	var unavailable *TokenizerUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("NewTokenizer() error = %T %v, want *TokenizerUnavailableError", err, err)
	}
	if unavailable.Model != "gpt-future-999" {
		t.Fatalf("unavailable model = %q, want %q", unavailable.Model, "gpt-future-999")
	}
	if !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("NewTokenizer() error = %v, want ai.ErrTokenizerUnsupported", err)
	}
}

func TestModelTokenizerUsesExplicitResolverAndPreservesUnavailableFallback(t *testing.T) {
	if tokenizer := (&Model{name: GPT41}).Tokenizer(); tokenizer == nil {
		t.Fatal("Tokenizer() = nil for a supported model")
	}
	if tokenizer := (&Model{name: "gpt-future-999"}).Tokenizer(); tokenizer != nil {
		t.Fatalf("Tokenizer() = %T for an unsupported model, want nil", tokenizer)
	}
}

func TestTokenizerHonorsCanceledContextWithoutProviderIO(t *testing.T) {
	tokenizer, err := NewTokenizer(GPT41)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.CountTokens(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("CountTokens() error = %v, want context.Canceled", err)
	}
	if _, err := tokenizer.Tokenize(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Tokenize() error = %v, want context.Canceled", err)
	}
}

func TestTokenizerTokenizesRepresentativeToolPayloadDeterministically(t *testing.T) {
	tokenizer, err := NewTokenizer(GPT41)
	if err != nil {
		t.Fatal(err)
	}
	payload := `{"name":"weather","description":"Get weather for München","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}`
	first, err := tokenizer.Tokenize(context.Background(), payload)
	if err != nil {
		t.Fatal(err)
	}
	second, err := tokenizer.Tokenize(context.Background(), payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("Tokenize() is not deterministic: first=%q second=%q", first, second)
	}
	count, err := tokenizer.CountTokens(context.Background(), payload)
	if err != nil || count != len(first) {
		t.Fatalf("CountTokens() = %d, %v; want %d, nil", count, err, len(first))
	}
}
