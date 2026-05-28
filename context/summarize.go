package context

import "context"

type SummaryRequest struct {
	ID        string
	Text      string
	MaxTokens int
	Required  bool
	Meta      map[string]any
}

type Summarizer interface {
	Summarize(ctx context.Context, req SummaryRequest) (string, error)
}
