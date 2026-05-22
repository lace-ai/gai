package context

import stdcontext "context"

type SummaryRequest struct {
	ID        string
	Text      string
	MaxTokens int
	Required  bool
	Meta      map[string]any
}

type Summarizer interface {
	Summarize(ctx stdcontext.Context, req SummaryRequest) (string, error)
}
