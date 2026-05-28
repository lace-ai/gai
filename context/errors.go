package context

import "errors"

var (
	ErrSessionNotFound      = errors.New("session not found")
	ErrInvalidSessionID     = errors.New("invalid session ID")
	ErrSessionStoreNotFound = errors.New("session store not found")
	ErrRAGStoreNotFound     = errors.New("rag store not found")

	ErrPromptPathEmpty = errors.New("prompt path is empty")
	ErrPromptFileType  = errors.New("prompt file must be .md or .txt")
	ErrPromptMissing   = errors.New("prompt file is missing")

	ErrPromptBuilderNil  = errors.New("prompt builder is nil")
	ErrPromptEntryID     = errors.New("prompt entry ID error")
	ErrPromptSource      = errors.New("prompt source error")
	ErrPromptBudget      = errors.New("prompt budget exceeded")
	ErrUserPromptEmpty   = errors.New("user prompt is empty")
	ErrTokenizerNotFound = errors.New("tokenizer not found")
)
