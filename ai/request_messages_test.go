package ai_test

import (
	"encoding/json"
	"github.com/lace-ai/gai/ai"
	"testing"
)

func TestAIRequestValidatesAndCopiesNativeMessages(t *testing.T) {
	r := ai.AIRequest{Messages: []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: "initial request"}, {Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_1", Name: "search", Arguments: json.RawMessage(`{"q":"x"}`), ThoughtSignature: []byte("opaque-signature")}}}, {Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "search", Content: "ok"}}}}
	if err := r.ValidateMessages(); err != nil {
		t.Fatal(err)
	}
	c := r.Copy()
	c.Messages[1].ToolCalls[0].Arguments[2] = 'X'
	c.Messages[1].ToolCalls[0].ThoughtSignature[0] = 'X'
	if string(r.Messages[1].ToolCalls[0].Arguments) == string(c.Messages[1].ToolCalls[0].Arguments) {
		t.Fatal("aliased arguments")
	}
	if string(r.Messages[1].ToolCalls[0].ThoughtSignature) == string(c.Messages[1].ToolCalls[0].ThoughtSignature) {
		t.Fatal("aliased thought signature")
	}
}

func TestAIRequestRejectsToolResultWithMismatchedName(t *testing.T) {
	r := ai.AIRequest{Messages: []ai.RequestMessage{
		{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_1", Name: "search", Arguments: json.RawMessage(`{"q":"x"}`)}}},
		{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "other", Content: "ok"}},
	}}
	if err := r.ValidateMessages(); err == nil {
		t.Fatal("expected mismatched tool result name to be rejected")
	}
}

func TestAIRequestRejectsDuplicateToolResult(t *testing.T) {
	r := ai.AIRequest{Messages: []ai.RequestMessage{
		{Role: ai.RequestMessageRoleAssistant, ToolCalls: []ai.RequestToolCall{{ID: "call_1", Name: "search", Arguments: json.RawMessage(`{"q":"x"}`)}}},
		{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "search", Content: "first"}},
		{Role: ai.RequestMessageRoleTool, ToolResult: &ai.RequestToolResult{ToolCallID: "call_1", Name: "search", Content: "duplicate"}},
	}}
	if err := r.ValidateMessages(); err == nil {
		t.Fatal("expected duplicate tool result to be rejected")
	}
}
