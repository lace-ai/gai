package anthropic

import (
	"errors"
	"reflect"
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

	p := New(" test-key ", nil)
	if p.apiKey != "test-key" {
		t.Fatalf("New() stored API key %q, want trimmed key", p.apiKey)
	}
	if p.httpClient.Timeout != 0 {
		t.Fatalf("client timeout = %s, want context-controlled lifetime", p.httpClient.Timeout)
	}
	if p.Name() != "anthropic" {
		t.Fatalf("Name() = %q", p.Name())
	}
	if _, err := p.Model("unknown"); !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("Model(unknown) error = %v, want ErrModelNotFound", err)
	}
	got, err := p.ListModels()
	if err != nil || !reflect.DeepEqual(got, models) {
		t.Fatalf("ListModels() = %v, %v; want %v", got, err, models)
	}
	for _, name := range models {
		model, err := p.Model(name)
		if err != nil || model.Name() != name {
			t.Fatalf("Model(%q) = %v, %v", name, model, err)
		}
	}
}
