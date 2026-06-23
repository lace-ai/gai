package ai

import "context"

// Model is a text-generation model exposed by an AI provider.
//
// Implementations must close the channel returned by GenerateStream when the
// request finishes. Close releases resources owned by the model.
type Model interface {
	// Name returns the provider-specific model identifier.
	Name() string
	// Generate executes a request and returns its complete response.
	Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
	// GenerateStream executes a request and emits response tokens incrementally.
	GenerateStream(ctx context.Context, req AIRequest) <-chan Token
	// Close releases resources owned by the model.
	Close() error
	// Tokenizer returns the tokenizer associated with the model.
	Tokenizer() Tokenizer
}
