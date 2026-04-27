package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

func TestMapFunctionCall(t *testing.T) {
	got, err := mapFunctionCall(&genai.FunctionCall{
		ID:   "call_1",
		Name: "echo_tool",
		Args: map[string]any{
			"query": "hello",
		},
	})
	if err != nil {
		t.Fatalf("mapFunctionCall error: %v", err)
	}

	if !strings.HasPrefix(got.ID, "call_echo_tool_") {
		t.Fatalf("expected generated tool id for echo_tool, got %q", got.ID)
	}
	if got.Type != "function" {
		t.Fatalf("expected tool call type=function, got %q", got.Type)
	}
	if got.Name != "echo_tool" {
		t.Fatalf("expected tool name to be function name, got %q", got.Name)
	}

	var args map[string]any
	if err := json.Unmarshal(got.Args, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["query"] != "hello" {
		t.Fatalf("unexpected args: %#v", args)
	}
}

func TestMapFunctionCallEmptyName(t *testing.T) {
	if _, err := mapFunctionCall(&genai.FunctionCall{ID: "call_1"}); err == nil {
		t.Fatal("expected error for empty function name")
	}
}

func TestMarshalArgsNilDefaultsToObject(t *testing.T) {
	raw, err := marshalArgs(nil)
	if err != nil {
		t.Fatalf("marshalArgs error: %v", err)
	}
	if string(raw) != "{}" {
		t.Fatalf("expected {}, got %s", string(raw))
	}
}

func TestBuildTextToken(t *testing.T) {
	tok := buildTextToken(&genai.Part{Text: "hello"})
	if tok.Type != ai.TokenTypeText {
		t.Fatalf("expected text token, got %s", tok.Type)
	}
	if string(tok.Data) != "hello" {
		t.Fatalf("expected token data to be plain text, got %q", string(tok.Data))
	}
	if tok.Text != "hello" {
		t.Fatalf("expected token text to be set, got %q", tok.Text)
	}
}

func TestBuildThoughtToken(t *testing.T) {
	tok := buildTextToken(&genai.Part{Text: "thinking", Thought: true})
	if tok.Type != ai.TokenTypeThought {
		t.Fatalf("expected thought token, got %s", tok.Type)
	}
	if string(tok.Data) != "thinking" {
		t.Fatalf("expected token data to be plain text, got %q", string(tok.Data))
	}
}
