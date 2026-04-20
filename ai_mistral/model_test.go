package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/HecoAI/gai/ai"
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

func TestModelGenerateStream(t *testing.T) {
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"hel"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"lo"}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key")
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt:    ai.Prompt{Prompt: "hello"},
		MaxTokens: 55,
	})

	var gotText string
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		if tok.Type != ai.TokenTypeText {
			t.Fatalf("unexpected token type: %s", tok.Type)
		}
		gotText += tok.String()
	}

	if !gotReq.Stream {
		t.Fatalf("expected stream=true in payload")
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 55 {
		t.Fatalf("expected max_tokens=55, got %+v", gotReq.MaxTokens)
	}
	if gotText != "hello" {
		t.Fatalf("unexpected streamed text: %q", gotText)
	}
}
