package mistral

import (
	"agent-backend/gai/ai"
	"errors"
	"testing"
)

func TestProviderModelValidation(t *testing.T) {
	p := New("test-key")

	model, err := p.Model("   ")
	if !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("expected ErrModelNotFound, got %w", err)
	}
	if model != nil {
		t.Fatalf("expected nil model on empty name")
	}
}

func TestProviderModelAndListModels(t *testing.T) {
	p := New("test-key")

	model, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model returned error: %w", err)
	}
	if model == nil {
		t.Fatalf("expected non-nil model")
	}
	if model.Name() != MistralSmallLatest {
		t.Fatalf("unexpected model name: %q", model.Name())
	}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %w", err)
	}
	if len(models) == 0 {
		t.Fatalf("expected at least one model")
	}
}

func TestProviderValidate(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %w", err)
	}

	if err := New("   ").Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %w", err)
	}

	if err := New("test-key").Validate(); err != nil {
		t.Fatalf("expected no error, got %w", err)
	}
}
