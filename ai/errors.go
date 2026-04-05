package ai

import "errors"

var (
	ErrModelNotFound         = errors.New("model not found")
	ErrProviderDown          = errors.New("provider unavailable")
	ErrProviderNotFound      = errors.New("provider not found")
	ErrProviderInvalid       = errors.New("provider is invalid")
	ErrProviderAlreadyExists = errors.New("provider already exists")
	ErrNilProvider           = errors.New("provider is nil")
	ErrNilModelRepository    = errors.New("model repository is nil")
)
