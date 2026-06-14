package history

import "errors"

var (
	ErrInvalidSummaryAmount = errors.New("summary amount must be between 0 and 1")
	ErrSummarizerRequired   = errors.New("summarizer is required when summary is enabled")
)
