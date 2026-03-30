package ai

import "errors"

var (
	ErrModelNotFound = errors.New("model not found")
	ErrProviderDown  = errors.New("provider unavailable")
)
