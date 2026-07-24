package openai

import (
	"context"
	"encoding/json"
	"errors"
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
	m, err := p.Model(O3)
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
	if got["model"] != O3 || got["max_completion_tokens"] != float64(42) || got["reasoning_effort"] != "high" {
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

func TestNativeMessagesMapUserPayload(t *testing.T) {
	messages, err := mapNativeMessages([]ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: "initial request"}})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	var payload []map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload) != 1 || payload[0]["role"] != "user" || payload[0]["content"] != "initial request" {
		t.Fatalf("payload = %#v", payload)
	}
}

func TestNativeMessagesMapToolErrorPayload(t *testing.T) {
	messages, err := mapNativeMessages([]ai.RequestMessage{{
		Role: ai.RequestMessageRoleTool,
		ToolResult: &ai.RequestToolResult{
			ToolCallID: "call_1",
			Name:       "search",
			Content:    "upstream unavailable",
			IsError:    true,
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	var payload []map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload) != 1 || payload[0]["role"] != "tool" || payload[0]["tool_call_id"] != "call_1" || payload[0]["content"] != `{"error":"upstream unavailable"}` {
		t.Fatalf("payload = %#v", payload)
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

func TestModelGenerateStreamEmitsToolCallsByIndex(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}},{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(GPT41Mini)
	if err != nil {
		t.Fatalf("Model returned error: %v", err)
	}
	var calls []*ai.ToolCall
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		if token.Type == ai.TokenTypeToolCall {
			calls = append(calls, token.ToolCall)
		}
	}
	if len(calls) != 2 || calls[0].ID != "call_1" || calls[1].ID != "call_2" {
		t.Fatalf("unexpected tool-call order: %#v", calls)
	}
}

func TestBuildChatCompletionParamsRejectsReasoningEffortForNonReasoningModels(t *testing.T) {
	for _, model := range []string{GPT41, GPT41Mini, GPT41Nano, GPT4o, GPT4oMini} {
		_, err := buildChatCompletionParams(model, ai.AIRequest{Prompt: "hello", Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}, false)
		if !errors.Is(err, ai.ErrUnsupportedCapability) {
			t.Fatalf("expected unsupported capability error for %q, got %v", model, err)
		}
	}
	if _, err := buildChatCompletionParams(O3, ai.AIRequest{Prompt: "hello", Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortHigh}}, false); err != nil {
		t.Fatalf("expected reasoning model to accept effort: %v", err)
	}
	if _, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{Prompt: "hello"}, false); err != nil {
		t.Fatalf("expected non-reasoning model to accept empty effort: %v", err)
	}
}

func TestApplyToolsRestrictsRequiredToolNames(t *testing.T) {
	definitions := []ai.ToolDefinition{
		{Type: "function", Name: "first", Description: "First", Parameters: json.RawMessage(`{"type":"object"}`)},
		{Type: "function", Name: "second", Description: "Second", Parameters: json.RawMessage(`{"type":"object"}`)},
		{Type: "function", Name: "excluded", Description: "Excluded", Parameters: json.RawMessage(`{"type":"object"}`)},
	}
	params, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{
		Prompt: "hello", Tools: definitions,
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"second", "first"}},
	}, false)
	if err != nil {
		t.Fatalf("buildChatCompletionParams returned error: %v", err)
	}
	if len(params.Tools) != 2 || params.Tools[0].Function.Name != "first" || params.Tools[1].Function.Name != "second" {
		t.Fatalf("unexpected restricted tools: %#v", params.Tools)
	}

	if _, err := buildChatCompletionParams(GPT41Mini, ai.AIRequest{
		Prompt: "hello", Tools: definitions,
		ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"missing"}},
	}, false); err == nil {
		t.Fatal("expected an undefined required tool to be rejected")
	}
}

func TestModelGenerateValidatesToolCallArguments(t *testing.T) {
	for _, tt := range []struct {
		name      string
		arguments string
		wantArgs  string
		wantErr   bool
	}{
		{name: "empty defaults to object", arguments: "", wantArgs: "{}"},
		{name: "invalid JSON is rejected", arguments: "not-json", wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":` + mustMarshalJSON(t, tt.arguments) + `}}]}}]}`))
			}))
			defer ts.Close()

			p := New("test-key", nil)
			p.baseURL = ts.URL
			m, err := p.Model(GPT41Mini)
			if err != nil {
				t.Fatalf("Model returned error: %v", err)
			}
			res, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "hello"})
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected Generate to reject invalid tool-call arguments")
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate returned error: %v", err)
			}
			if len(res.ToolCalls) != 1 || string(res.ToolCalls[0].Args) != tt.wantArgs {
				t.Fatalf("unexpected tool calls: %#v", res.ToolCalls)
			}
		})
	}
}

func mustMarshalJSON(t *testing.T, value string) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON: %v", err)
	}
	return string(raw)
}
