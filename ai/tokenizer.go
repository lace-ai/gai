package ai

import "context"

type Tokenizer interface {
	Tokenize(ctx context.Context, text string) []string
	CountTokens(ctx context.Context, text string) (int, error)
}
