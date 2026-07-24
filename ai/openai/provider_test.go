package openai

import (
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestProviderModelAndStaticList(t *testing.T) {
	p := New("test-key", nil)

	model, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if model == nil || model.Name() != GPT41Mini {
		t.Fatalf("unexpected model: %#v", model)
	}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected static models")
	}
	models[0] = "mutated"
	again, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if again[0] == "mutated" {
		t.Fatal("ListModels must return a copy")
	}
}

func TestProviderValidation(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}
	if err := New("   ", nil).Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %v", err)
	}
	if model, err := New("test-key", nil).Model("unknown"); !errors.Is(err, ai.ErrModelNotFound) || model != nil {
		t.Fatalf("expected unknown model rejection, got model=%#v err=%v", model, err)
	}
}
