package ai_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestNewToolDefinitionValidatesSchema(t *testing.T) {
	def, err := ai.NewToolDefinition(
		"search",
		"Searches documents.",
		json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
	)
	if err != nil {
		t.Fatalf("NewToolDefinition error: %v", err)
	}
	if def.Type != "function" {
		t.Fatalf("expected function type, got %q", def.Type)
	}
	if def.Name != "search" {
		t.Fatalf("unexpected name: %q", def.Name)
	}
}

func TestNewToolDefinitionRejectsInvalidSchema(t *testing.T) {
	_, err := ai.NewToolDefinition("search", "Searches documents.", json.RawMessage(`{"type":"string"}`))
	if !errors.Is(err, ai.ErrInvalidToolDefinition) {
		t.Fatalf("expected ErrInvalidToolDefinition, got %v", err)
	}
}

func TestResponseFormatValidation(t *testing.T) {
	format := ai.ResponseFormat{
		Type:   ai.ResponseFormatJSONSchema,
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
	}
	if err := format.Validate(); err != nil {
		t.Fatalf("Validate error: %v", err)
	}

	format.Schema = json.RawMessage(`{`)
	if err := format.Validate(); !errors.Is(err, ai.ErrInvalidResponseFormat) {
		t.Fatalf("expected ErrInvalidResponseFormat, got %v", err)
	}
}
