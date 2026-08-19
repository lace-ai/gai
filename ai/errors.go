package ai

import (
	"errors"
	"fmt"
	"time"
)

// ProviderErrorKind is a provider-neutral classification used by retry policy.
type ProviderErrorKind string

const (
	ProviderErrorUnknown        ProviderErrorKind = "unknown"
	ProviderErrorRateLimited    ProviderErrorKind = "rate_limited"
	ProviderErrorTransient      ProviderErrorKind = "transient"
	ProviderErrorAuthentication ProviderErrorKind = "authentication"
	ProviderErrorInvalidRequest ProviderErrorKind = "invalid_request"
	ProviderErrorUnsupported    ProviderErrorKind = "unsupported"
)

// ProviderError carries safe, provider-neutral failure metadata. Provider
// adapters should wrap vendor errors in this type before returning them.
type ProviderError struct {
	Kind       ProviderErrorKind
	StatusCode int
	Code       string
	RequestID  string
	RetryAfter time.Duration
	Err        error
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Err != nil {
		return fmt.Sprintf("provider %s error: %v", e.Kind, e.Err)
	}
	return fmt.Sprintf("provider %s error", e.Kind)
}

func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

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
	// ErrInvalidToolDefinition indicates a malformed tool definition.
	ErrInvalidToolDefinition = errors.New("invalid tool definition")
	// ErrInvalidResponseFormat indicates a malformed structured response request.
	ErrInvalidResponseFormat = errors.New("invalid response format")
	// ErrUnsupportedCapability indicates that a provider cannot satisfy a request feature.
	ErrUnsupportedCapability = errors.New("unsupported provider capability")
	// ErrTokenizerUnsupported indicates that a tokenizer does not support an operation.
	ErrTokenizerUnsupported = errors.New("tokenizer operation unsupported")
)
