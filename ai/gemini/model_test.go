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

func TestBuildGenerateContentConfigMapsCapabilities(t *testing.T) {
	cfg, err := buildGenerateContentConfig(ai.AIRequest{
		MaxTokens: 64,
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
		Reasoning: ai.ReasoningConfig{
			Enabled:         true,
			IncludeThoughts: true,
			BudgetTokens:    128,
			Effort:          ai.ReasoningEffortHigh,
		},
	})
	if err != nil {
		t.Fatalf("buildGenerateContentConfig error: %v", err)
	}

	if cfg.MaxOutputTokens != 64 {
		t.Fatalf("unexpected max output tokens: %d", cfg.MaxOutputTokens)
	}
	if len(cfg.Tools) != 1 || len(cfg.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected one function declaration, got %#v", cfg.Tools)
	}
	if cfg.Tools[0].FunctionDeclarations[0].Name != "search" {
		t.Fatalf("unexpected function declaration: %#v", cfg.Tools[0].FunctionDeclarations[0])
	}
	if cfg.ToolConfig == nil || cfg.ToolConfig.FunctionCallingConfig == nil {
		t.Fatal("expected function calling config")
	}
	if cfg.ToolConfig.FunctionCallingConfig.Mode != genai.FunctionCallingConfigModeAny {
		t.Fatalf("unexpected function calling mode: %s", cfg.ToolConfig.FunctionCallingConfig.Mode)
	}
	if len(cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) != 1 || cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0] != "search" {
		t.Fatalf("unexpected allowed functions: %#v", cfg.ToolConfig.FunctionCallingConfig.AllowedFunctionNames)
	}
	if cfg.ResponseMIMEType != "application/json" || cfg.ResponseJsonSchema == nil {
		t.Fatalf("expected JSON response schema, got mime=%q schema=%#v", cfg.ResponseMIMEType, cfg.ResponseJsonSchema)
	}
	if cfg.ThinkingConfig == nil || !cfg.ThinkingConfig.IncludeThoughts {
		t.Fatalf("expected thinking config with included thoughts, got %#v", cfg.ThinkingConfig)
	}
	if cfg.ThinkingConfig.ThinkingBudget == nil || *cfg.ThinkingConfig.ThinkingBudget != 128 {
		t.Fatalf("unexpected thinking budget: %#v", cfg.ThinkingConfig.ThinkingBudget)
	}
	if cfg.ThinkingConfig.ThinkingLevel != genai.ThinkingLevelHigh {
		t.Fatalf("unexpected thinking level: %s", cfg.ThinkingConfig.ThinkingLevel)
	}
}

func TestMapGenerateContentResponseSeparatesTextReasoningAndToolCalls(t *testing.T) {
	text, reasoning, toolCalls, err := mapGenerateContentResponse(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "visible"},
						{Text: "private", Thought: true},
						{FunctionCall: &genai.FunctionCall{Name: "search", Args: map[string]any{"query": "x"}}},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("mapGenerateContentResponse error: %v", err)
	}
	if text != "visible" {
		t.Fatalf("unexpected text: %q", text)
	}
	if reasoning != "private" {
		t.Fatalf("unexpected reasoning: %q", reasoning)
	}
	if len(toolCalls) != 1 || toolCalls[0].Name != "search" {
		t.Fatalf("unexpected tool calls: %#v", toolCalls)
	}
}

func TestModelTokenizer(t *testing.T) {
	m := &Model{name: "gemini-2.5-flash"}
	tokenizer := m.Tokenizer()
	if tokenizer == nil {
		t.Fatal("expected tokenizer")
	}
	if tokenizer.ID() != "gemini.gemini-2.5-flash" {
		t.Fatalf("unexpected tokenizer ID: %q", tokenizer.ID())
	}
}
