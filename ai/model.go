package ai

import "context"

type Model interface {
	Name() string
	Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
	GenerateStream(ctx context.Context, req AIRequest) <-chan Token
	Close() error
}
