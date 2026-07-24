// Package ai defines provider-independent interfaces and value types for text
// generation.
//
// A Provider exposes named Models. A Model accepts an AIRequest and either
// returns a complete AIResponse or streams Tokens. ModelRepository provides a
// small registry for selecting models by provider and model name.
//
// Provider-specific implementations live in subpackages such as anthropic,
// gemini, and mistral.
package ai
