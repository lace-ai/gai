package exa_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop/tools/exa"
)

func TestSearchTool(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("x-api-key"); got != "secret" {
			t.Errorf("x-api-key = %q, want secret", got)
		}

		var body struct {
			Query      string `json:"query"`
			Type       string `json:"type"`
			NumResults int    `json:"numResults"`
			Contents   struct {
				Highlights bool `json:"highlights"`
			} `json:"contents"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if body.Query != "current Go release" || body.Type != "fast" || body.NumResults != 3 || !body.Contents.Highlights {
			t.Errorf("unexpected request: %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"requestId":"request-1","results":[{"title":"Go","url":"https://go.dev","highlights":["Current release"]}]}`))
	}))
	defer server.Close()

	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL), exa.WithSearchType("fast"), exa.WithNumResults(3))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	if tool.Name() != "web_search" {
		t.Fatalf("Name = %q, want web_search", tool.Name())
	}

	response := tool.Function(context.Background(), &ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":" current Go release "}`),
	})
	if response.Err != nil {
		t.Fatalf("Function: %v", response.Err)
	}
	if !json.Valid([]byte(response.Text)) {
		t.Fatalf("tool response is not JSON: %s", response.Text)
	}
}

func TestSearchToolReturnsAPIError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"rate limited"}`, http.StatusTooManyRequests)
	}))
	defer server.Close()

	tool, err := exa.NewSearchTool("secret", exa.WithEndpoint(server.URL))
	if err != nil {
		t.Fatalf("NewSearchTool: %v", err)
	}
	response := tool.Function(context.Background(), &ai.ToolCall{
		ID: "call-1", Type: "function", Name: tool.Name(), Args: json.RawMessage(`{"query":"news"}`),
	})
	if response.Err == nil {
		t.Fatal("expected API error")
	}
}

func TestNewSearchToolValidatesConfiguration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		apiKey  string
		options []exa.Option
		wantErr error
	}{
		{name: "missing API key", wantErr: exa.ErrAPIKeyMissing},
		{name: "unsupported type", apiKey: "secret", options: []exa.Option{exa.WithSearchType("neural")}, wantErr: exa.ErrInvalidOption},
		{name: "too many results", apiKey: "secret", options: []exa.Option{exa.WithNumResults(101)}, wantErr: exa.ErrInvalidOption},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := exa.NewSearchTool(test.apiKey, test.options...)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
		})
	}
}
