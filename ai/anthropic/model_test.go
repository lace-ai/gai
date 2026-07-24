package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
)

func testModel(t *testing.T, handler http.HandlerFunc) *Model {
	t.Helper()
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	p := New("test-key", nil)
	p.baseURL = ts.URL
	m, err := p.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	return m.(*Model)
}

func TestGenerateSendsAnthropicRequestAndMapsBlocksAndUsage(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Fatalf("unexpected Anthropic headers: %#v", r.Header)
		}
		var body messagesRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != ClaudeSonnet4_6 || body.MaxTokens != 4096 || body.Stream || len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.Messages[0].Content != "hello" {
			t.Fatalf("unexpected request: %#v", body)
		}
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"},{"type":"thinking","thinking":"reason"},{"type":"tool_use","id":"toolu_1","name":"search","input":{"q":"x"}}],"usage":{"input_tokens":10,"cache_creation_input_tokens":2,"cache_read_input_tokens":3,"output_tokens":4,"output_tokens_details":{"thinking_tokens":2}}}`))
	})

	got, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != "answer" || got.Reasoning != "reason" || got.InputTokens != 15 || got.OutputTokens != 4 || got.ReasoningTokens != 2 || len(got.ToolCalls) != 1 {
		t.Fatalf("unexpected response: %#v", got)
	}
	if got.ToolCalls[0].ID != "toolu_1" || got.ToolCalls[0].Type != "function" || got.ToolCalls[0].Name != "search" || string(got.ToolCalls[0].Args) != `{"q":"x"}` {
		t.Fatalf("unexpected tool call: %#v", got.ToolCalls[0])
	}
}

func TestGenerateMapsCapabilitiesAndRejectsUnsupportedResponseFormat(t *testing.T) {
	var got messagesRequest
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-beta") != structuredOutputBeta {
			t.Fatalf("anthropic-beta = %q", r.Header.Get("anthropic-beta"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
	})
	_, err := m.Generate(context.Background(), ai.AIRequest{
		Prompt: "hello", MaxTokens: 2048,
		Tools:          []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice:     ai.ToolChoice{Mode: ai.ToolChoiceAuto},
		ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONSchema, Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
		Reasoning:      ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.MaxTokens != 2048 || len(got.Tools) != 1 || got.Tools[0].InputSchema == nil || got.ToolChoice.Type != "auto" || got.OutputConfig == nil || got.OutputConfig.Format == nil || got.OutputConfig.Format.Type != "json_schema" || got.Thinking == nil || got.Thinking.BudgetTokens != 1024 {
		t.Fatalf("unexpected capability mapping: %#v", got)
	}
	_, err = m.Generate(context.Background(), ai.AIRequest{Prompt: "x", ResponseFormat: ai.ResponseFormat{Type: ai.ResponseFormatJSONObject}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("JSON object error = %v, want unsupported capability", err)
	}
	_, err = m.Generate(context.Background(), ai.AIRequest{Prompt: "x", MaxTokens: 1024, Reasoning: ai.ReasoningConfig{Enabled: true, BudgetTokens: 1024}})
	if !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("invalid thinking budget error = %v", err)
	}
}

func TestBuildMessagesRequestMapsAdaptiveReasoning(t *testing.T) {
	request, err := buildMessagesRequest(ai.AIRequest{
		Prompt: "hello",
		Reasoning: ai.ReasoningConfig{
			Enabled:         true,
			IncludeThoughts: false,
			Effort:          ai.ReasoningEffortHigh,
		},
	}, ClaudeSonnet4_6, false)
	if err != nil {
		t.Fatalf("buildMessagesRequest error: %v", err)
	}
	if request.Thinking == nil || request.Thinking.Type != "adaptive" || request.Thinking.Display != "omitted" {
		t.Fatalf("unexpected thinking configuration: %#v", request.Thinking)
	}
	if request.OutputConfig == nil || request.OutputConfig.Effort != string(ai.ReasoningEffortHigh) {
		t.Fatalf("unexpected output configuration: %#v", request.OutputConfig)
	}
}

func TestMapToolChoice(t *testing.T) {
	tests := []struct {
		choice ai.ToolChoice
		want   toolChoiceRequest
	}{
		{choice: ai.ToolChoice{}, want: toolChoiceRequest{Type: "auto"}},
		{choice: ai.ToolChoice{Mode: ai.ToolChoiceNone}, want: toolChoiceRequest{Type: "none"}},
		{choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired}, want: toolChoiceRequest{Type: "any"}},
		{choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}}, want: toolChoiceRequest{Type: "tool", Name: "search"}},
	}
	for _, tt := range tests {
		got, err := mapToolChoice(tt.choice)
		if err != nil || *got != tt.want {
			t.Fatalf("mapToolChoice(%#v) = %#v, %v; want %#v", tt.choice, got, err, tt.want)
		}
	}
	if _, err := mapToolChoice(ai.ToolChoice{Mode: ai.ToolChoiceAuto, Names: []string{"search"}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("restricted auto error = %v", err)
	}
}

func TestGenerateReturnsProviderError(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}`))
	})
	_, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "hello"})
	var providerErr *Error
	if !errors.As(err, &providerErr) || providerErr.StatusCode != http.StatusBadRequest || providerErr.Type != "invalid_request_error" || providerErr.Message != "bad input" {
		t.Fatalf("unexpected error: %#v", err)
	}
}

func TestGenerateStreamMapsInterleavedBlocksAndToolJSON(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Fatalf("Accept = %q", r.Header.Get("Accept"))
		}
		var body messagesRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if !body.Stream {
			t.Fatal("stream was not requested")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"why\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"x\\\"}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 3 || tokens[0].Type != ai.TokenTypeText || tokens[0].Text != "hi" || tokens[1].Type != ai.TokenTypeThought || tokens[1].Text != "why" || tokens[2].Type != ai.TokenTypeToolCall || tokens[2].ToolCall == nil || string(tokens[2].ToolCall.Args) != `{"q":"x"}` {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamErrorAndCancellation(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type != ai.TokenTypeErr || tokens[0].Err == nil || !strings.Contains(tokens[0].Err.Error(), "busy") {
		t.Fatalf("unexpected stream error: %#v", tokens)
	}

	started := make(chan struct{})
	release := make(chan struct{})
	m = testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	ch := m.GenerateStream(ctx, ai.AIRequest{Prompt: "hello"})
	<-started
	cancel()
	defer close(release)
	select {
	case _, ok := <-ch:
		if ok {
			for range ch {
			}
		}
	case <-time.After(time.Second):
		t.Fatal("stream did not close after cancellation")
	}
}

func TestGenerateStreamRejectsTruncatedToolBlock(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type != ai.TokenTypeErr || tokens[0].Err == nil || !strings.Contains(tokens[0].Err.Error(), "open content") {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestConsumeSSERejectsOversizedEvent(t *testing.T) {
	stream := "data: " + strings.Repeat("x", maxSSEEvent+1) + "\n\n"
	err := consumeSSE(strings.NewReader(stream), func(ai.Token) bool { return true })
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("consumeSSE oversized event error = %v", err)
	}
}

func TestCountTokens(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages/count_tokens" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Fatal("missing Anthropic headers")
		}
		var body countTokensRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != ClaudeSonnet4_6 || len(body.Messages) != 1 || body.Messages[0].Content != "hello" {
			t.Fatalf("body = %#v", body)
		}
		_, _ = w.Write([]byte(`{"input_tokens":42}`))
	})
	if got, err := m.Tokenizer().CountTokens(context.Background(), "hello"); err != nil || got != 42 {
		t.Fatalf("CountTokens = %d, %v", got, err)
	}
	if _, err := m.Tokenizer().Tokenize(context.Background(), "hello"); !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("Tokenize error = %v", err)
	}
}
