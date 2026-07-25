package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	res, err := m.Generate(context.Background(), ai.AIRequest{
		Prompt:    "sys\n\nctx\n\nhello",
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

func TestNativeMessagesMapUserPayload(t *testing.T) {
	messages := mapNativeMessages([]ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: "initial request"}})
	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content != "initial request" {
		t.Fatalf("payload = %#v", messages)
	}
}

func TestNativeMessagesMapToolErrorPayload(t *testing.T) {
	messages := mapNativeMessages([]ai.RequestMessage{{
		Role: ai.RequestMessageRoleTool,
		ToolResult: &ai.RequestToolResult{
			ToolCallID: "call_1",
			Name:       "search",
			Content:    "upstream unavailable",
			IsError:    true,
		},
	}})
	if len(messages) != 1 || messages[0].Role != "tool" || messages[0].ToolCallID != "call_1" || messages[0].Content != `{"error":"upstream unavailable"}` {
		t.Fatalf("payload = %#v", messages)
	}
}

func TestModelGenerateMapsRequestCapabilities(t *testing.T) {
	var gotReq chatCompletionRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = m.Generate(context.Background(), ai.AIRequest{
		Prompt: "hello",
		Tools: []ai.ToolDefinition{
			{
				Type:        "function",
				Name:        "search",
				Description: "Searches documents.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
		},
		ToolChoice: ai.ToolChoice{
			Mode:  ai.ToolChoiceRequired,
			Names: []string{"search"},
		},
		ResponseFormat: ai.ResponseFormat{
			Type:   ai.ResponseFormatJSONSchema,
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
		},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if len(gotReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(gotReq.Tools))
	}
	if gotReq.Tools[0].Type != "function" || gotReq.Tools[0].Function.Name != "search" {
		t.Fatalf("unexpected tool mapping: %#v", gotReq.Tools[0])
	}
	choice, ok := gotReq.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("expected named tool choice map, got %#v", gotReq.ToolChoice)
	}
	function, ok := choice["function"].(map[string]any)
	if !ok || function["name"] != "search" {
		t.Fatalf("unexpected tool choice: %#v", gotReq.ToolChoice)
	}
	if gotReq.ResponseFormat == nil || gotReq.ResponseFormat.Type != "json_schema" {
		t.Fatalf("unexpected response format: %#v", gotReq.ResponseFormat)
	}
	if gotReq.ResponseFormat.JSONSchema == nil || gotReq.ResponseFormat.JSONSchema.Name != "answer" {
		t.Fatalf("unexpected response schema: %#v", gotReq.ResponseFormat.JSONSchema)
	}
}

func TestBuildChatCompletionRequestRejectsUnsupportedToolChoiceMode(t *testing.T) {
	_, err := buildChatCompletionRequest(ai.AIRequest{
		Prompt: "hello",
		Tools: []ai.ToolDefinition{
			{
				Type:        "function",
				Name:        "search",
				Description: "Searches documents.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
		},
		ToolChoice: ai.ToolChoice{
			Mode: "sometimes",
		},
	}, MistralSmallLatest, false)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), `unsupported mistral tool choice mode "sometimes"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestModelGenerateMapsMultipleResponseToolCalls(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{
				"message":{
					"content":"",
					"tool_calls":[
						{"id":"call_1","type":"function","function":{"name":"first_tool","arguments":"{\"value\":1}"}},
						{"id":"call_2","type":"function","function":{"name":"second_tool","arguments":"{\"value\":2}"}}
					]
				}
			}],
			"usage":{"prompt_tokens":3,"completion_tokens":4}
		}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	res, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "call tools"})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if len(res.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %#v", res.ToolCalls)
	}
	if res.ToolCalls[0].ID != "call_1" || res.ToolCalls[0].Name != "first_tool" || string(res.ToolCalls[0].Args) != `{"value":1}` {
		t.Fatalf("unexpected first tool call: %#v", res.ToolCalls[0])
	}
	if res.ToolCalls[1].ID != "call_2" || res.ToolCalls[1].Name != "second_tool" || string(res.ToolCalls[1].Args) != `{"value":2}` {
		t.Fatalf("unexpected second tool call: %#v", res.ToolCalls[1])
	}
}

func TestMapChatResponseToolCallsRejectsMalformedEntries(t *testing.T) {
	tests := []struct {
		name    string
		raw     json.RawMessage
		wantErr string
	}{
		{
			name:    "missing tool name",
			raw:     json.RawMessage(`[{"id":"call_1","type":"function","function":{"arguments":"{\"value\":1}"}}]`),
			wantErr: "map tool_calls[0]: missing tool name",
		},
		{
			name:    "invalid JSON arguments",
			raw:     json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"value\""}}]`),
			wantErr: `map tool_calls[0]: invalid JSON arguments for tool "search"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := mapChatResponseToolCalls(tt.raw)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestModelGenerateNoChoices(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = m.Generate(context.Background(), ai.AIRequest{Prompt: "hello"})
	if err != ErrNoChoices {
		t.Fatalf("expected ErrNoChoices, got %v", err)
	}
}

func TestModelTokenizerTokenizeUnsupported(t *testing.T) {
	p := New("test-key", nil)

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	tokens, err := m.Tokenizer().Tokenize(context.Background(), "hello")
	if !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("expected ErrTokenizerUnsupported, got %v", err)
	}
	if tokens != nil {
		t.Fatalf("expected no tokens for unsupported tokenizer, got %#v", tokens)
	}
}

func TestModelTokenizerCountTokens(t *testing.T) {
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
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":0}}`))
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	got, err := m.Tokenizer().(*Tokenizer).CountTokens(context.Background(), "hello")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
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
	if gotReq.Messages[0].Content != "hello" {
		t.Fatalf("unexpected content: %q", gotReq.Messages[0].Content)
	}
	if gotReq.MaxTokens == nil || *gotReq.MaxTokens != 1 {
		t.Fatalf("expected max_tokens=1, got %+v", gotReq.MaxTokens)
	}
	if got != 9 {
		t.Fatalf("expected 9 tokens, got %d", got)
	}
}

func TestModelTokenizerID(t *testing.T) {
	p := New("test-key", nil)
	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	tokenizer := m.Tokenizer()
	if tokenizer.ID() != "mistral."+MistralSmallLatest {
		t.Fatalf("unexpected tokenizer ID: %q", tokenizer.ID())
	}
}

func TestModelTokenizerCountTokensHTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	_, err = m.Tokenizer().(*Tokenizer).CountTokens(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "mistral token count failed (status 400)") {
		t.Fatalf("unexpected error: %v", err)
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

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt:    "hello",
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

func TestModelGenerateStreamToolCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		toolCallJSON := `[{"id":"call_abc","type":"function","function":{"name":"my_tool","arguments":"{\"param\":\"value\"}"}}]`
		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"tool_calls":` + toolCallJSON + `}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var tokens []ai.Token
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		tokens = append(tokens, tok)
	}

	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	tok := tokens[0]
	if tok.Type != ai.TokenTypeToolCall {
		t.Fatalf("expected TokenTypeToolCall, got %s", tok.Type)
	}
	if tok.ToolCall == nil {
		t.Fatal("expected ToolCall to be populated, got nil")
	}
	if !strings.HasPrefix(tok.ToolCall.ID, "call_my_tool_") {
		t.Fatalf("expected generated ToolCall.ID for my_tool, got %q", tok.ToolCall.ID)
	}
	if tok.ToolCall.Type != "function" {
		t.Fatalf("expected ToolCall.Type=function, got %q", tok.ToolCall.Type)
	}
	if tok.ToolCall.Name != "my_tool" {
		t.Fatalf("expected ToolCall.Name=my_tool, got %q", tok.ToolCall.Name)
	}
	wantArgs := `{"param":"value"}`
	if string(tok.ToolCall.Args) != wantArgs {
		t.Fatalf("expected ToolCall.Args=%s, got %s", wantArgs, string(tok.ToolCall.Args))
	}
}

func TestModelGenerateStreamToolCallDeltas(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","type":"function","function":{"name":"my_","arguments":"{\"param\""}}]}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"abc","function":{"name":"tool","arguments":":\"value\"}"}}]}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var tokens []ai.Token
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}
		tokens = append(tokens, tok)
	}

	if len(tokens) != 1 {
		t.Fatalf("expected 1 token, got %d", len(tokens))
	}
	tok := tokens[0]
	if tok.Type != ai.TokenTypeToolCall {
		t.Fatalf("expected TokenTypeToolCall, got %s", tok.Type)
	}
	if tok.ToolCall == nil {
		t.Fatal("expected ToolCall to be populated, got nil")
	}
	if tok.ToolCall.Name != "my_tool" {
		t.Fatalf("expected ToolCall.Name=my_tool, got %q", tok.ToolCall.Name)
	}
	if string(tok.ToolCall.Args) != `{"param":"value"}` {
		t.Fatalf("expected accumulated ToolCall.Args, got %s", string(tok.ToolCall.Args))
	}
}

func TestModelGenerateStreamDetectsTextEncodedToolCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer is not a flusher")
		}

		for _, chunk := range []string{
			`data: {"choices":[{"delta":{"content":"\n"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"\n"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"{\"id\":\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"echo\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"type\":\"function\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"name\":\"echo\",\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"arguments\":{\""}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"text\":\"try"}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":" the echo tool\"}}"}}]}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write stream chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	p := New("test-key", nil)
	p.baseURL = ts.URL

	m, err := p.Model(MistralSmallLatest)
	if err != nil {
		t.Fatalf("Model error: %v", err)
	}

	stream := m.GenerateStream(context.Background(), ai.AIRequest{
		Prompt: "call a tool",
	})

	var (
		gotToolCall bool
		gotText     string
	)
	for tok := range stream {
		if tok.Err != nil {
			t.Fatalf("unexpected stream error: %v", tok.Err)
		}

		if tok.Type == ai.TokenTypeToolCall {
			if tok.ToolCall == nil {
				t.Fatal("expected ToolCall to be populated, got nil")
			}
			if !strings.HasPrefix(tok.ToolCall.ID, "call_echo_") {
				t.Fatalf("expected generated ToolCall.ID for echo, got %q", tok.ToolCall.ID)
			}
			if tok.ToolCall.Type != "function" {
				t.Fatalf("expected ToolCall.Type=function, got %q", tok.ToolCall.Type)
			}
			if tok.ToolCall.Name != "echo" {
				t.Fatalf("expected ToolCall.Name=echo, got %q", tok.ToolCall.Name)
			}
			if string(tok.ToolCall.Args) != `{"text":"try the echo tool"}` {
				t.Fatalf("unexpected ToolCall.Args: %s", string(tok.ToolCall.Args))
			}
			gotToolCall = true
			continue
		}

		if tok.Type == ai.TokenTypeText {
			gotText += tok.String()
		}
	}

	if !gotToolCall {
		t.Fatal("expected to detect a tool call from text stream, got none")
	}
	if gotText != "" {
		t.Fatalf("expected no text tokens after tool-call detection, got %q", gotText)
	}
}
