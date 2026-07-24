package anthropic

import (
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestProviderValidationAndModels(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("Validate(nil) error = %v, want ErrNilProvider", err)
	}
	if err := New("   ", nil).Validate(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("Validate(empty) error = %v, want ErrInvalidAPIKey", err)
	}

	p := New("test-key", nil)
	if p.httpClient.Timeout != 0 {
		t.Fatalf("client timeout = %s, want context-controlled lifetime", p.httpClient.Timeout)
	}
	if p.Name() != "anthropic" {
		t.Fatalf("Name() = %q", p.Name())
	}
	if _, err := p.Model("unknown"); !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("Model(unknown) error = %v, want ErrModelNotFound", err)
	}
	models, err := p.ListModels()
	if err != nil || len(models) == 0 {
		t.Fatalf("ListModels() = %v, %v", models, err)
	}
	model, err := p.Model(models[0])
	if err != nil || model.Name() != models[0] {
		t.Fatalf("Model(%q) = %v, %v", models[0], model, err)
	}
}
