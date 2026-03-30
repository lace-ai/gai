package ai

import "context"

type Model interface {
	Name() string
	Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
	Close() error
}
