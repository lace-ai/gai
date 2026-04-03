package context

import "errors"

var (
	ErrSessionIDInvalid    = errors.New("session id must be greater than zero")
	ErrMessageContentEmpty = errors.New("message content cannot be empty")
	ErrRoleInvalid         = errors.New("message role is invalid")
)
