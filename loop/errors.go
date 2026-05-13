package loop

import "errors"

var (
	ErrNilAgent            = errors.New("agent is nil")
	ErrModelNotConfigured  = errors.New("model is not configured")
	ErrPromptNotConfigured = errors.New("prompt builder is not configured")
	ErrToolReqValidation   = errors.New("invalid tool call")
	ErrToolCallMalformed   = errors.New("tool call payload is malformed")
	ErrToolNotFound        = errors.New("tool not found")
	ErrMaxIterations       = errors.New("max loop iterations exceeded")
	ErrPromptPathEmpty     = errors.New("prompt path is empty")
	ErrPromptFileType      = errors.New("prompt file must be .md or .txt")
	ErrPromptMissing       = errors.New("prompt file is missing")
	ErrArgsDecodeTarget    = errors.New("tool args decode target is nil")
	ErrPreProcessToolRes   = errors.New("pre-process tool response error")
	ErrBuildPrompt         = errors.New("build prompt error")
	ErrMaxRetries          = errors.New("max retries exceeded")
)
