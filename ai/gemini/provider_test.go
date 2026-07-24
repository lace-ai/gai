package gemini

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
)

type errorRoundTripper struct{}

func (errorRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("model discovery unavailable")
}

type handlerRoundTripper func(*http.Request) (*http.Response, error)

func (h handlerRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return h(r) }

func response(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

func TestProviderDynamicallyListsModelsAndAcceptsThem(t *testing.T) {
	var requests []*http.Request
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(r *http.Request) (*http.Response, error) {
		requests = append(requests, r.Clone(r.Context()))
		return response(http.StatusOK, `{"models":[{"name":"models/gemini-dynamic"}]}`), nil
	})}
	p.baseURL = "https://models.test"

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 1 || models[0] != "gemini-dynamic" {
		t.Fatalf("unexpected models: %#v", models)
	}

	model, err := p.Model("gemini-dynamic")
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	if model.Name() != "gemini-dynamic" {
		t.Fatalf("unexpected model name: %q", model.Name())
	}
	if len(requests) != 1 {
		t.Fatalf("expected one discovery request, got %d", len(requests))
	}
	for _, request := range requests {
		if request.Method != http.MethodGet || request.URL.Path != "/v1beta/models" {
			t.Fatalf("unexpected request: %s %s", request.Method, request.URL.String())
		}
		if request.Header.Get("x-goog-api-key") != "test-key" {
			t.Fatalf("unexpected API key header: %q", request.Header.Get("x-goog-api-key"))
		}
	}
}

func TestProviderFallsBackWhenModelDiscoveryFails(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		return response(http.StatusServiceUnavailable, "unavailable"), nil
	})}

	models, err := p.ListModels()
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if !containsModel(models, Gemini3FlashPreview) {
		t.Fatalf("expected hard-coded fallback models, got %#v", models)
	}
	if _, err := p.Model(Gemini3FlashPreview); err != nil {
		t.Fatalf("Model did not accept fallback model: %v", err)
	}
}

func TestProviderDoesNotDiscoverModelsWithInvalidAPIKey(t *testing.T) {
	hits := 0
	p := New("   ", nil)
	p.httpClient = &http.Client{Transport: handlerRoundTripper(func(*http.Request) (*http.Response, error) {
		hits++
		return response(http.StatusOK, `{}`), nil
	})}
	if _, err := p.ListModels(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey, got %v", err)
	}
	if hits != 0 {
		t.Fatalf("expected no discovery requests, got %d", hits)
	}
}

func TestProviderModelValidation(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

	model, err := p.Model("   ")
	if !errors.Is(err, ai.ErrModelNotFound) {
		t.Fatalf("expected ErrModelNotFound, got %v", err)
	}
	if model != nil {
		t.Fatalf("expected nil model on empty name")
	}

	model, err = p.Model("unknown-model")
	if err != nil {
		t.Fatalf("expected dynamically discoverable model name to be accepted, got %v", err)
	}
	if model == nil || model.Name() != "unknown-model" {
		t.Fatalf("unexpected model for dynamic name: %#v", model)
	}
}

func TestProviderModelAndListModels(t *testing.T) {
	p := New("test-key", nil)
	p.httpClient = &http.Client{Transport: errorRoundTripper{}}

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

func TestProviderValidate(t *testing.T) {
	if err := (*Provider)(nil).Validate(); !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}

	if _, err := New("   ", nil).ListModels(); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey from ListModels, got %v", err)
	}

	if _, err := New("   ", nil).Model(Gemini3FlashPreview); !errors.Is(err, ErrInvalidAPIKey) {
		t.Fatalf("expected ErrInvalidAPIKey from Model, got %v", err)
	}
}
