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

func decodeRequest(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return body
}

func object(t *testing.T, value any) map[string]any {
	t.Helper()
	got, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON object", value)
	}
	return got
}

func array(t *testing.T, value any) []any {
	t.Helper()
	got, ok := value.([]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON array", value)
	}
	return got
}

func TestGenerateSendsAnthropicRequestAndMapsBlocksAndUsage(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Fatalf("unexpected Anthropic headers: %#v", r.Header)
		}
		body := decodeRequest(t, r)
		messages := array(t, body["messages"])
		content := array(t, object(t, messages[0])["content"])
		if body["model"] != ClaudeSonnet4_6 || body["max_tokens"] != float64(4096) || body["stream"] != nil || len(messages) != 1 || object(t, messages[0])["role"] != "user" || len(content) != 1 || object(t, content[0])["type"] != "text" || object(t, content[0])["text"] != "hello" {
			t.Fatalf("unexpected request: %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
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
	var got map[string]any
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-beta") != "" {
			t.Fatalf("anthropic-beta = %q", r.Header.Get("anthropic-beta"))
		}
		got = decodeRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
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
	tools := array(t, got["tools"])
	tool := object(t, tools[0])
	outputConfig := object(t, got["output_config"])
	format := object(t, outputConfig["format"])
	thinking := object(t, got["thinking"])
	if got["max_tokens"] != float64(2048) || len(tools) != 1 || tool["input_schema"] == nil || object(t, got["tool_choice"])["type"] != "auto" || format["type"] != "json_schema" || thinking["budget_tokens"] != float64(1024) {
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

func TestGenerateMapsAdaptiveReasoningAndToolChoice(t *testing.T) {
	tests := []struct {
		name    string
		choice  ai.ToolChoice
		want    map[string]any
		wantErr bool
	}{
		{name: "auto", choice: ai.ToolChoice{}, want: map[string]any{"type": "auto"}},
		{name: "none", choice: ai.ToolChoice{Mode: ai.ToolChoiceNone}, want: map[string]any{"type": "none"}},
		{name: "any", choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired}, wantErr: true},
		{name: "named", choice: ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"search"}}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
				body := decodeRequest(t, r)
				if got := object(t, body["tool_choice"]); !equalJSON(got, tt.want) {
					t.Fatalf("tool_choice = %#v, want %#v", got, tt.want)
				}
				thinking := object(t, body["thinking"])
				if thinking["type"] != "adaptive" || thinking["display"] != "omitted" || object(t, body["output_config"])["effort"] != "high" {
					t.Fatalf("unexpected adaptive reasoning: %#v", body)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"content":[],"usage":{}}`))
			})
			_, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "hello", Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: tt.choice, Reasoning: ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh}})
			if tt.wantErr {
				if !errors.Is(err, ai.ErrUnsupportedCapability) {
					t.Fatalf("Generate error = %v, want unsupported capability", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
	m := testModel(t, func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected request") })
	if _, err := m.Generate(context.Background(), ai.AIRequest{Prompt: "hello", Tools: []ai.ToolDefinition{{Type: "function", Name: "search", Description: "Search", Parameters: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: ai.ToolChoice{Mode: ai.ToolChoiceAuto, Names: []string{"search"}}}); !errors.Is(err, ai.ErrUnsupportedCapability) {
		t.Fatalf("restricted auto error = %v", err)
	}
}

func equalJSON(got, want map[string]any) bool {
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	return string(gotJSON) == string(wantJSON)
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
		body := decodeRequest(t, r)
		if body["stream"] != true {
			t.Fatal("stream was not requested")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"why\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"x\\\"}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 3 || tokens[0].Type != ai.TokenTypeText || string(tokens[0].Data) != "hi" || tokens[1].Type != ai.TokenTypeThought || tokens[1].Text != "why" || tokens[2].Type != ai.TokenTypeToolCall || tokens[2].ToolCall == nil || string(tokens[2].ToolCall.Args) != `{"q":"x"}` {
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

func TestGenerateStreamDetectsTextFallbackToolCall(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"{\\\"type\\\":\\\"function\\\",\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\\\"name\\\":\\\"search\\\",\\\"arguments\\\":{}}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
	})

	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type != ai.TokenTypeToolCall || tokens[0].ToolCall == nil || tokens[0].ToolCall.Name != "search" {
		t.Fatalf("unexpected tokens: %#v", tokens)
	}
}

func TestGenerateStreamRejectsTruncatedToolBlock(t *testing.T) {
	m := testModel(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"search\"}}\n\n"))
	})
	var tokens []ai.Token
	for token := range m.GenerateStream(context.Background(), ai.AIRequest{Prompt: "hello"}) {
		tokens = append(tokens, token)
	}
	if len(tokens) != 1 || tokens[0].Type != ai.TokenTypeErr || tokens[0].Err == nil || !strings.Contains(tokens[0].Err.Error(), "open content") {
		t.Fatalf("unexpected tokens: %#v", tokens)
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
		body := decodeRequest(t, r)
		messages := array(t, body["messages"])
		content := array(t, object(t, messages[0])["content"])
		if body["model"] != ClaudeSonnet4_6 || len(messages) != 1 || len(content) != 1 || object(t, content[0])["type"] != "text" || object(t, content[0])["text"] != "hello" {
			t.Fatalf("body = %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"input_tokens":42}`))
	})
	if got, err := m.Tokenizer().CountTokens(context.Background(), "hello"); err != nil || got != 42 {
		t.Fatalf("CountTokens = %d, %v", got, err)
	}
	if _, err := m.Tokenizer().Tokenize(context.Background(), "hello"); !errors.Is(err, ai.ErrTokenizerUnsupported) {
		t.Fatalf("Tokenize error = %v", err)
	}
}
