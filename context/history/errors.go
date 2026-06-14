package history

import "errors"

var (
	// ErrInvalidSummaryAmount is returned when summary amount is outside 0..1.
	ErrInvalidSummaryAmount = errors.New("summary amount must be between 0 and 1")

	// ErrSummarizerRequired is returned when summarization is enabled without a model or summarizer.
	ErrSummarizerRequired = errors.New("summarizer is required when summary is enabled")

	// ErrHistoryStateRequired is returned when summarization is requested without history state.
	ErrHistoryStateRequired = errors.New("history state is required")

	// ErrHistorySourceNil is returned when a nil HistorySource is asked to summarize.
	ErrHistorySourceNil = errors.New("history source is nil")

	// ErrSummarizerMissing is returned when a HistorySource has no configured summarizer.
	ErrSummarizerMissing = errors.New("summarizer not configured for history source")
)
