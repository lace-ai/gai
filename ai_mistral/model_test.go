package mistral

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestModelGenerate(t *testing.T) {
	var gotAuth string
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`))
	}))
	defer ts.Close()

	p := New("test-key")
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	res, err := m.Generate(context.Background(), ai.AIRequest{
		Prompt:    ai.Prompt{System: "sys", Prompt: "hello", Context: "ctx"},
		MaxTokens: 42,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if gotAuth != "Bearer test-key" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
	if gotReq.Model != MistralSmallLatest {
		t.Fatalf("unexpected model: %q", gotReq.Model)
	}
	if len(gotReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(gotReq.Messages))
	}
	if gotReq.Messages[0].Role != "user" {
		t.Fatalf("unexpected role: %q", gotReq.Messages[0].Role)
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 42 {
		t.Fatalf("expected max_tokens=42, got %+v", gotReq.MaxTokens)
	}

	if res.Text != "ok" {
		t.Fatalf("unexpected response text: %q", res.Text)
	}
	if res.InputTokens != 11 || res.OutputTokens != 7 {
		t.Fatalf("unexpected usage mapping: %+v", res)
	}
}

func TestModelGenerateNoChoices(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key")
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = m.Generate(context.Background(), ai.AIRequest{Prompt: ai.Prompt{Prompt: "hello"}})
	if err != ErrNoChoices {
		t.Fatalf("expected ErrNoChoices, got %v", err)
	}
}
