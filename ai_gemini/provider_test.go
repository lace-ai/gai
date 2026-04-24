package gemini

import (
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestProviderModelValidation(t *testing.T) {
	p := New("test-key")

	model, err := p.Model("   ")
	if !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("expected ErrModelNotFound, got %v", err)
	}
	if model != nil {
		t.Fatalf("expected nil model on empty name")
	}
}

func TestProviderModelAndListModels(t *testing.T) {
	p := New("test-key")

	model, err := p.Model("gemini-3-flash-preview")
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if model == nil {
		t.Fatalf("expected non-nil model")
	}
	if model.Name() != "gemini-3-flash-preview" {
		t.Fatalf("unexpected model name: %q", model.Name())
	}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("expected at least one model")
	}
}
