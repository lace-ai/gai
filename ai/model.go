package ai

import "context"

// Model is a text-generation model exposed by an AI provider.
//
// Implementations must close the channel returned by GenerateStream when the
// request finishes or its context is canceled. Cancellation may be represented
// by closing the channel without emitting an error token.
type Model interface {
	// Name returns the provider-specific model identifier.
	Name() string
	// Generate executes a request and returns its complete response.
	Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
	// GenerateStream executes a request and emits response tokens incrementally.
	// Implementations should use SendToken so cancellation cannot block a sender.
	GenerateStream(ctx context.Context, req AIRequest) <-chan Token
	// Close releases resources owned by the model.
	Close() error
	// Tokenizer returns the tokenizer associated with the model.
	Tokenizer() Tokenizer
}

// NativeToolModel is optionally implemented by legacy models that send tool
// definitions through their provider's native tool-calling API. New models
// should implement ModelDescriber and report ModelDescriptor.NativeTools
// instead. Agent consults NativeToolModel only when ModelDescriber is absent.
//
// It is deliberately separate from Model so existing custom Model
// implementations retain the text-based compatibility protocol by default.
//
// Deprecated: implement ModelDescriber instead.
type NativeToolModel interface {
	NativeTools() bool
}
