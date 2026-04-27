package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/lace-ai/gai/ai"
)

type expectedWrapToken struct {
	typ          ai.TokenType
	data         string
	checkData    bool
	toolType     string
	toolName     string
	toolArgsJSON string
}

func TestWrapStream(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  []ai.Token
		output []expectedWrapToken
	}{
		{
			name: "Detects leading tool call and passes through remainder",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(" \n\t{")},
				{Type: ai.TokenTypeText, Data: []byte(`"id":"call-1","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte(" trailing text")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing text", checkData: true},
			},
		},
		{
			name: "Passes through non-JSON leading text",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("hello")},
				{Type: ai.TokenTypeText, Data: []byte(" world")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "hello", checkData: true},
				{typ: ai.TokenTypeText, data: " world", checkData: true},
			},
		},
		{
			name: "Replays pending when non-text arrives before decision",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("  ")},
				{Type: ai.TokenTypeErr, Data: []byte("boom")},
				{Type: ai.TokenTypeText, Data: []byte("after")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "  ", checkData: true},
				{typ: ai.TokenTypeErr, data: "boom", checkData: true},
				{typ: ai.TokenTypeText, data: "after", checkData: true},
			},
		},
		{
			name: "Replays when JSON is not a valid tool call",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","type":"not-function","name":"echo"}`)},
				{Type: ai.TokenTypeText, Data: []byte("tail")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"not-function","name":"echo"}`, checkData: true},
				{typ: ai.TokenTypeText, data: "tail", checkData: true},
			},
		},
		{
			name: "Replays unclosed JSON at end of stream",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-1","type":"function","name":"echo"`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"id":"call-1","type":"function","name":"echo"`, checkData: true},
			},
		},
		{
			name: "Handles braces inside strings",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-2","type":"function","name":"echo","arguments":{"msg":"{\\\"a\\\":1}"`)},
				{Type: ai.TokenTypeText, Data: []byte(`,"items":[1,2,3]}}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"items":[1,2,3],"msg":"{\\\"a\\\":1}"}`},
			},
		},
		{
			name: "Defaults missing arguments to empty object",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-3","type":"function","name":"echo"}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{}`},
			},
		},
		{
			name: "Detects tool call and preserves trailing text in same token",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-10","type":"function","name":"echo","arguments":{"x":1}} trailing`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeText, data: " trailing", checkData: true},
			},
		},
		{
			name: "Detects adjacent tool calls in same token",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-11","type":"function","name":"echo","arguments":{"x":1}}{"id":"call-12","type":"function","name":"echo","arguments":{"y":2}} tail`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"x":1}`},
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"y":2}`},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
		{
			name: "Detects tool call from production-like chunked JSON",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("\n")},
				{Type: ai.TokenTypeText, Data: []byte("\n")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"`)},
				{Type: ai.TokenTypeText, Data: []byte(`echo","`)},
				{Type: ai.TokenTypeText, Data: []byte(`type":"function","`)},
				{Type: ai.TokenTypeText, Data: []byte(`name":"echo","`)},
				{Type: ai.TokenTypeText, Data: []byte(`arguments":{"`)},
				{Type: ai.TokenTypeText, Data: []byte(`text":"try`)},
				{Type: ai.TokenTypeText, Data: []byte(` the echo tool"}}`)},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeToolCall, toolType: "function", toolName: "echo", toolArgsJSON: `{"text":"try the echo tool"}`},
			},
		},
		{
			name: "Does not detect tool call when text prefix exists",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte("Sure, I can help. ")},
				{Type: ai.TokenTypeText, Data: []byte(`{"id":"call-9","type":"function","name":"echo","arguments":{"x":1}}`)},
				{Type: ai.TokenTypeText, Data: []byte(" done")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: "Sure, I can help. ", checkData: true},
				{typ: ai.TokenTypeText, data: `{"id":"call-9","type":"function","name":"echo","arguments":{"x":1}}`, checkData: true},
				{typ: ai.TokenTypeText, data: " done", checkData: true},
			},
		},
		{
			name: "Preserves non-tool JSON object",
			input: []ai.Token{
				{Type: ai.TokenTypeText, Data: []byte(`{"kind":"event","value":123}`)},
				{Type: ai.TokenTypeText, Data: []byte(" tail")},
			},
			output: []expectedWrapToken{
				{typ: ai.TokenTypeText, data: `{"kind":"event","value":123}`, checkData: true},
				{typ: ai.TokenTypeText, data: " tail", checkData: true},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			in := make(chan ai.Token, len(tt.input))
			for _, tok := range tt.input {
				in <- tok
			}
			close(in)

			out := collectTokens(ai.WrapStream(t.Context(), in, nil))
			if len(out) != len(tt.output) {
				t.Fatalf("expected %d output tokens, got %d", len(tt.output), len(out))
			}

			for i, expected := range tt.output {
				got := out[i]
				if got.Type != expected.typ {
					t.Fatalf("token %d unexpected type: got=%q want=%q", i, got.Type, expected.typ)
				}

				if expected.checkData && string(got.Data) != expected.data {
					t.Fatalf("token %d unexpected data: got=%q want=%q", i, string(got.Data), expected.data)
				}

				if expected.typ == ai.TokenTypeToolCall {
					if got.ToolCall == nil {
						t.Fatalf("token %d expected tool call metadata, got nil", i)
					}
					if got.ToolCall.ID == "" {
						t.Fatalf("token %d unexpected empty tool call id", i)
					}
					if got.ToolCall.Type != expected.toolType {
						t.Fatalf("token %d unexpected tool call type: got=%q want=%q", i, got.ToolCall.Type, expected.toolType)
					}
					if got.ToolCall.Name != expected.toolName {
						t.Fatalf("token %d unexpected tool call name: got=%q want=%q", i, got.ToolCall.Name, expected.toolName)
					}
					if normalizeJSON(got.ToolCall.Args) != expected.toolArgsJSON {
						t.Fatalf("token %d unexpected tool call arguments: got=%s want=%s", i, string(got.ToolCall.Args), expected.toolArgsJSON)
					}
				}
			}
		})
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
