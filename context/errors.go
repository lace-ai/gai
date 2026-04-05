package context

import "errors"

var (
	ErrSessionNotFound      = errors.New("session not found")
	ErrInvalidSessionID     = errors.New("invalid session ID")
	ErrSessionStoreNotFound = errors.New("session store not found")
)
