package loop

import "errors"

var (
	ErrNilAgent           = errors.New("agent is nil")
	ErrModelNotConfigured = errors.New("model is not configured")
	ErrToolReqValidation  = errors.New("invalid tool call")
	ErrToolCallMalformed  = errors.New("tool call payload is malformed")
	ErrToolNotFound       = errors.New("tool not found")
	ErrMaxIterations      = errors.New("max loop iterations exceeded")
	ErrPromptPathEmpty    = errors.New("prompt path is empty")
	ErrPromptFileType     = errors.New("prompt file must be .md or .txt")
	ErrPromptMissing      = errors.New("prompt file is missing")
	ErrArgsDecodeTarget   = errors.New("tool args decode target is nil")
	ErrPreProcessToolRes  = errors.New("pre-process tool response error")
	ErrBuildContext       = errors.New("build context error")
)
