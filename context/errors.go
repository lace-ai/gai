package context

import "errors"

var (
	ErrSessionNotFound      = errors.New("session not found")
	ErrInvalidSessionID     = errors.New("invalid session ID")
	ErrSessionStoreNotFound = errors.New("session store not found")

	ErrPromptPathEmpty = errors.New("prompt path is empty")
	ErrPromptFileType  = errors.New("prompt file must be .md or .txt")
	ErrPromptMissing   = errors.New("prompt file is missing")
)
