package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestModelGenerateMapsCapabilitiesAndResponse(t *testing.T) {
	var got map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if gotAuth := r.Header.Get("Authorization"); gotAuth != "Bearer test-key" {
			t.Fatalf("unexpected auth: %q", gotAuth)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","choices":[{"message":{"content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"query\":\"go\"}"}}]}}],"usage":{"prompt_tokens":11,"completion_tokens":7,"completion_tokens_details":{"reasoning_tokens":3}}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	res, err := m.Generate(context.Background(), ai.AIRequest{
		Prompt:         "hello",
		MaxTokens:      42,
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)},
		Reasoning:      ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh},
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got["model"] != GPT41Mini || got["max_completion_tokens"] != float64(42) || got["reasoning_effort"] != "high" {
		t.Fatalf("unexpected request mapping: %#v", got)
	}
	if len(got["tools"].([]any)) != 1 || got["tool_choice"].(map[string]any)["function"].(map[string]any)["name"] != "search" {
		t.Fatalf("unexpected tool mapping: %#v", got)
	}
	if got["response_format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("unexpected response format: %#v", got)
	}
	if res.Text != "ok" || res.InputTokens != 11 || res.OutputTokens != 7 || res.ReasoningTokens != 3 {
		t.Fatalf("unexpected response: %#v", res)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "call_1" || res.ToolCalls[0].Name != "search" || string(res.ToolCalls[0].Args) != `{"query":"go"}` {
		t.Fatalf("unexpected tool calls: %#v", res.ToolCalls)
	}
}

func TestModelGenerateStreamMapsTextAndToolCalls(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"go\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 2 || tokens[0].Type != ai.TokenTypeText || tokens[0].Text != "hello " {
		t.Fatalf("unexpected stream tokens: %#v", tokens)
	}
	if call := tokens[1].ToolCall; tokens[1].Type != ai.TokenTypeToolCall || call == nil || call.ID != "call_1" || call.Name != "search" || string(call.Args) != `{"q":"go"}` {
		t.Fatalf("unexpected tool token: %#v", tokens[1])
	}
}
