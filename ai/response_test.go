package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestWrapStreamDetectsLeadingToolCallAndPassesThroughRemainder(t *testing.T) {
	in := make(chan ai.Token, 4)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(" \n\t{")}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`"id":"call-1","name":"function","arguments":{"x":1}}`)}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(" trailing text")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 2 {
		t.Fatalf("expected 2 output tokens, got %d", len(out))
	}

	if out[0].Type != ai.TokenTypeToolCall {
		t.Fatalf("expected first token type %q, got %q", ai.TokenTypeToolCall, out[0].Type)
	}
	if out[0].ToolCall == nil {
		t.Fatalf("expected tool call metadata, got nil")
	}
	if out[0].ToolCall.ID != "call-1" {
		t.Fatalf("unexpected tool call id: %q", out[0].ToolCall.ID)
	}
	if out[0].ToolCall.Name != "function" {
		t.Fatalf("unexpected tool call name: %q", out[0].ToolCall.Name)
	}
	if normalizeJSON(out[0].ToolCall.Args) != `{"x":1}` {
		t.Fatalf("unexpected tool call arguments: %s", string(out[0].ToolCall.Args))
	}

	if out[1].Type != ai.TokenTypeText || string(out[1].Data) != " trailing text" {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
}

func TestWrapStreamPassesThroughNonJSONLeadingText(t *testing.T) {
	in := make(chan ai.Token, 2)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte("hello")}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(" world")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 2 {
		t.Fatalf("expected 2 output tokens, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != "hello" {
		t.Fatalf("unexpected first token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
	if out[1].Type != ai.TokenTypeText || string(out[1].Data) != " world" {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
}

func TestWrapStreamReplaysPendingWhenNonTextArrivesBeforeDecision(t *testing.T) {
	in := make(chan ai.Token, 3)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte("  ")}
	in <- ai.Token{Type: ai.TokenTypeErr, Data: []byte("boom")}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte("after")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 3 {
		t.Fatalf("expected 3 output tokens, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != "  " {
		t.Fatalf("unexpected first token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
	if out[1].Type != ai.TokenTypeErr || string(out[1].Data) != "boom" {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
	if out[2].Type != ai.TokenTypeText || string(out[2].Data) != "after" {
		t.Fatalf("unexpected third token: type=%q data=%q", out[2].Type, string(out[2].Data))
	}
}

func TestWrapStreamReplaysWhenJSONIsNotValidToolCall(t *testing.T) {
	in := make(chan ai.Token, 2)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","name":"not-function"}`)}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte("tail")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 2 {
		t.Fatalf("expected 2 output tokens, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != `{"id":"call-1","name":"not-function"}` {
		t.Fatalf("unexpected first token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
	if out[1].Type != ai.TokenTypeText || string(out[1].Data) != "tail" {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
}

func TestWrapStreamReplaysUnclosedJSONAtEndOfStream(t *testing.T) {
	in := make(chan ai.Token, 1)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","name":"function"`)}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 1 {
		t.Fatalf("expected 1 output token, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != `{"id":"call-1","name":"function"` {
		t.Fatalf("unexpected token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
}

func TestWrapStreamHandlesBracesInsideStrings(t *testing.T) {
	in := make(chan ai.Token, 2)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-2","name":"function","arguments":{"msg":"{\\\"a\\\":1}"`)}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`,"items":[1,2,3]}}`)}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 1 {
		t.Fatalf("expected 1 output token, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeToolCall || out[0].ToolCall == nil {
		t.Fatalf("expected tool call token, got type=%q", out[0].Type)
	}
	if out[0].ToolCall.ID != "call-2" {
		t.Fatalf("unexpected tool call id: %q", out[0].ToolCall.ID)
	}
	if normalizeJSON(out[0].ToolCall.Args) != `{"items":[1,2,3],"msg":"{\\\"a\\\":1}"}` {
		t.Fatalf("unexpected tool call arguments: %s", string(out[0].ToolCall.Args))
	}
}

func TestWrapStreamDefaultsMissingArgumentsToEmptyObject(t *testing.T) {
	in := make(chan ai.Token, 1)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-3","name":"function"}`)}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 1 {
		t.Fatalf("expected 1 output token, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeToolCall || out[0].ToolCall == nil {
		t.Fatalf("expected tool call token, got type=%q", out[0].Type)
	}
	if normalizeJSON(out[0].ToolCall.Args) != `{}` {
		t.Fatalf("unexpected default arguments: %s", string(out[0].ToolCall.Args))
	}
}

func TestWrapStreamDetectsToolCallFromProductionLikeChunkedJSON(t *testing.T) {
	chunks := []string{
		"\n",
		"\n",
		`{"id":"`,
		`echo","`,
		`name":"function","`,
		`arguments":{"`,
		`text":"try`,
		` the echo tool"}}`,
	}

	in := make(chan ai.Token, len(chunks))
	for _, c := range chunks {
		in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(c)}
	}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 1 {
		t.Fatalf("expected exactly 1 output token, got %d", len(out))
	}

	if out[0].Type != ai.TokenTypeToolCall {
		t.Fatalf("expected output type %q, got %q", ai.TokenTypeToolCall, out[0].Type)
	}
	if out[0].ToolCall == nil {
		t.Fatal("expected tool call metadata, got nil")
	}
	if out[0].ToolCall.ID != "echo" {
		t.Fatalf("unexpected tool call id: %q", out[0].ToolCall.ID)
	}
	if out[0].ToolCall.Name != "function" {
		t.Fatalf("unexpected tool call name: %q", out[0].ToolCall.Name)
	}
	if normalizeJSON(out[0].ToolCall.Args) != `{"text":"try the echo tool"}` {
		t.Fatalf("unexpected tool call arguments: %s", string(out[0].ToolCall.Args))
	}
}

func TestWrapStreamDoesNotDetectToolCallWhenTextPrefixExists(t *testing.T) {
	in := make(chan ai.Token, 3)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte("Sure, I can help. ")}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-9","name":"function","arguments":{"x":1}}`)}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(" done")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 3 {
		t.Fatalf("expected 3 output tokens, got %d", len(out))
	}

	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != "Sure, I can help. " {
		t.Fatalf("unexpected first token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
	if out[1].Type != ai.TokenTypeText || string(out[1].Data) != `{"id":"call-9","name":"function","arguments":{"x":1}}` {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
	if out[2].Type != ai.TokenTypeText || string(out[2].Data) != " done" {
		t.Fatalf("unexpected third token: type=%q data=%q", out[2].Type, string(out[2].Data))
	}
}

func TestWrapStreamPreservesNonToolJSONObject(t *testing.T) {
	in := make(chan ai.Token, 2)
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(`{"kind":"event","value":123}`)}
	in <- ai.Token{Type: ai.TokenTypeText, Data: []byte(" tail")}
	close(in)

	out := collectTokens(ai.WrapStream(in))
	if len(out) != 2 {
		t.Fatalf("expected 2 output tokens, got %d", len(out))
	}
	if out[0].Type != ai.TokenTypeText || string(out[0].Data) != `{"kind":"event","value":123}` {
		t.Fatalf("unexpected first token: type=%q data=%q", out[0].Type, string(out[0].Data))
	}
	if out[1].Type != ai.TokenTypeText || string(out[1].Data) != " tail" {
		t.Fatalf("unexpected second token: type=%q data=%q", out[1].Type, string(out[1].Data))
	}
}

func collectTokens(in <-chan ai.Token) []ai.Token {
	var out []ai.Token
	for t := range in {
		out = append(out, t)
	}
	return out
}

func normalizeJSON(v []byte) string {
	var payload any
	if err := json.Unmarshal(v, &payload); err != nil {
		return string(v)
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return string(v)
	}
	return string(b)
}
