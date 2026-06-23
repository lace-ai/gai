package ai

import "errors"

var (
	// ErrModelNotFound indicates that a requested model is unavailable.
	ErrModelNotFound = errors.New("model not found")
	// ErrProviderDown indicates that the provider cannot currently serve requests.
	ErrProviderDown = errors.New("provider unavailable")
	// ErrProviderNotFound indicates that a requested provider is not registered.
	ErrProviderNotFound = errors.New("provider not found")
	// ErrProviderInvalid indicates that provider configuration is invalid.
	ErrProviderInvalid = errors.New("provider is invalid")
	// ErrProviderAlreadyExists indicates that a provider name is already registered.
	ErrProviderAlreadyExists = errors.New("provider already exists")
	// ErrNilProvider indicates that a provider argument is nil.
	ErrNilProvider = errors.New("provider is nil")
	// ErrNilModelRepository indicates an operation on a nil repository.
	ErrNilModelRepository = errors.New("model repository is nil")
	// ErrInvalidToolCall indicates a malformed tool call.
	ErrInvalidToolCall = errors.New("invalid tool call")
	// ErrTokenizerUnsupported indicates that a tokenizer does not support an operation.
	ErrTokenizerUnsupported = errors.New("tokenizer operation unsupported")
)
