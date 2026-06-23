package ai

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ToolDefinition describes a function tool in a provider-neutral form.
type ToolDefinition struct {
	// Type identifies the tool kind. The only built-in kind is "function".
	Type string
	// Name is the function name exposed to the model.
	Name string
	// Description explains when and how the model should use the tool.
	Description string
	// Parameters contains the function parameters as JSON Schema.
	Parameters json.RawMessage
}

// ToolChoiceMode controls provider-side function calling.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNone     ToolChoiceMode = "none"
)

// ToolChoice controls whether and how the model may call tools.
type ToolChoice struct {
	Mode  ToolChoiceMode
	Names []string
}

// ResponseFormatType identifies the requested model response format.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// ResponseFormat requests text or structured JSON output.
type ResponseFormat struct {
	Type   ResponseFormatType
	Name   string
	Schema json.RawMessage
}

// ReasoningEffort is a provider-neutral reasoning intensity hint.
type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// ReasoningConfig configures model reasoning/thinking behavior when supported.
type ReasoningConfig struct {
	Enabled         bool
	IncludeThoughts bool
	BudgetTokens    int
	Effort          ReasoningEffort
}

// NewToolDefinition validates and returns a provider-neutral function tool.
func NewToolDefinition(name, description string, parameters json.RawMessage) (ToolDefinition, error) {
	def := ToolDefinition{
		Type:        "function",
		Name:        strings.TrimSpace(name),
		Description: strings.TrimSpace(description),
		Parameters:  append(json.RawMessage(nil), parameters...),
	}
	if err := def.Validate(); err != nil {
		return ToolDefinition{}, err
	}
	return def, nil
}

// Validate checks that def is a usable function tool definition.
func (def ToolDefinition) Validate() error {
	if def.Type == "" {
		def.Type = "function"
	}
	if def.Type != "function" {
		return fmt.Errorf("%w: unsupported tool type %q", ErrInvalidToolDefinition, def.Type)
	}
	if strings.TrimSpace(def.Name) == "" {
		return fmt.Errorf("%w: name empty", ErrInvalidToolDefinition)
	}
	if strings.TrimSpace(def.Description) == "" {
		return fmt.Errorf("%w: description empty", ErrInvalidToolDefinition)
	}
	if len(def.Parameters) == 0 {
		return fmt.Errorf("%w: parameters empty", ErrInvalidToolDefinition)
	}
	if !json.Valid(def.Parameters) {
		return fmt.Errorf("%w: parameters invalid JSON", ErrInvalidToolDefinition)
	}
	var schema map[string]any
	if err := json.Unmarshal(def.Parameters, &schema); err != nil {
		return fmt.Errorf("%w: parameters schema: %w", ErrInvalidToolDefinition, err)
	}
	if typ, ok := schema["type"].(string); ok && typ != "object" {
		return fmt.Errorf("%w: parameters type must be object", ErrInvalidToolDefinition)
	}
	return nil
}

// Validate checks that format is internally consistent.
func (format ResponseFormat) Validate() error {
	switch format.Type {
	case "", ResponseFormatText, ResponseFormatJSONObject:
		return nil
	case ResponseFormatJSONSchema:
		if strings.TrimSpace(format.Name) == "" {
			return fmt.Errorf("%w: schema name empty", ErrInvalidResponseFormat)
		}
		if len(format.Schema) == 0 {
			return fmt.Errorf("%w: schema empty", ErrInvalidResponseFormat)
		}
		if !json.Valid(format.Schema) {
			return fmt.Errorf("%w: schema invalid JSON", ErrInvalidResponseFormat)
		}
		return nil
	default:
		return fmt.Errorf("%w: unsupported response format %q", ErrInvalidResponseFormat, format.Type)
	}
}
