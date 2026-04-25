package gemini

import (
	"encoding/json"
	"testing"

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

	if got.ID != "echo_tool" {
		t.Fatalf("expected tool id to be function name, got %q", got.ID)
	}
	if got.Name != "function" {
		t.Fatalf("expected tool call type=function, got %q", got.Name)
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
