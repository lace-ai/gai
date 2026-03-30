package loop

import "errors"

var (
	ErrNilAgent            = errors.New("agent is nil")
	ErrModelNotConfigured  = errors.New("model is not configured")
	ErrMemoryNotConfigured = errors.New("memory is not configured")
	ErrEmptyPrompt         = errors.New("prompt is empty")
	ErrNilResponseBuilder  = errors.New("response builder is nil")
	ErrToolCallMalformed   = errors.New("tool call payload is malformed")
	ErrToolCallType        = errors.New("tool call type must be function")
	ErrToolCallID          = errors.New("tool call id is required")
	ErrToolArgsMissing     = errors.New("tool call arguments are required")
	ErrInvalidToolRequest  = errors.New("invalid tool request")
	ErrToolNotFound        = errors.New("tool not found")
	ErrMaxIterations       = errors.New("max loop iterations exceeded")
	ErrPromptPathEmpty     = errors.New("prompt path is empty")
	ErrPromptFileType      = errors.New("prompt file must be .md or .txt")
	ErrPromptMissing       = errors.New("prompt file is missing")
	ErrArgsDecodeTarget    = errors.New("tool args decode target is nil")
)
