package context

import "errors"

var (
	ErrRepositoryInvalid error = errors.New("repository is invalid")
	ErrSessionIDInvalid  error = errors.New("session ID is invalid")
	ErrMessageInvalid    error = errors.New("message is invalid")
)
