package ai

import "context"

// Tokenizer converts text to model tokens and estimates prompt size.
type Tokenizer interface {
	// Tokenize splits text into the tokenizer's token representation.
	Tokenize(ctx context.Context, text string) ([]string, error)
	// CountTokens returns the number of tokens required for text.
	CountTokens(ctx context.Context, text string) (int, error)
	// ID returns a stable identifier for the tokenizer implementation.
	ID() string
}
