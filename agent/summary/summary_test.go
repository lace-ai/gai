package summary_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestSummarizerRunsSummaryAgentThroughLoop(t *testing.T) {
	t.Parallel()

	model := &recordingModel{response: "short summary"}
	summarizer := summary.New(model)

	got, err := summarizer.Summarize(context.Background(), gaictx.SummaryRequest{
		ID:        "history",
		Text:      "long input",
		MaxTokens: 7,
	})
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}
	if got != "short summary" {
		t.Fatalf("unexpected summary: %q", got)
	}
	if model.request.MaxTokens != 7 {
		t.Fatalf("expected summary max tokens on loop request, got %d", model.request.MaxTokens)
	}
	if !strings.Contains(model.request.Prompt.System, "Summarize the provided context") {
		t.Fatalf("expected embedded summary system prompt: %q", model.request.Prompt.System)
	}
	if !strings.Contains(model.request.Prompt.Prompt, "long input") {
		t.Fatalf("expected summary input in user prompt: %q", model.request.Prompt.Prompt)
	}
}

func TestDefinitionAllowsSystemPromptOverride(t *testing.T) {
	t.Parallel()

	model := &recordingModel{response: "short summary"}
	summarizer := summary.Summarizer{
		Definition: summary.Definition(model, summary.WithSystemPrompt("custom summary system")),
	}

	_, err := summarizer.Summarize(context.Background(), gaictx.SummaryRequest{Text: "input"})
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}
	if !strings.Contains(model.request.Prompt.System, "custom summary system") {
		t.Fatalf("expected custom system prompt: %q", model.request.Prompt.System)
	}
}

func TestSummarizerDrainsLoopErrorsWhileWaitingForTokens(t *testing.T) {
	t.Parallel()

	model := toolCallModel{
		tokens: []ai.Token{
			{
				Type: ai.TokenTypeToolCall,
				ToolCall: &ai.ToolCall{
					ID:   "call_1",
					Type: "function",
					Name: "lookup",
					Args: []byte(`{"query":"first"}`),
				},
			},
			{
				Type: ai.TokenTypeToolCall,
				ToolCall: &ai.ToolCall{
					ID:   "call_2",
					Type: "function",
					Name: "lookup",
					Args: []byte(`{"query":"second"}`),
				},
			},
		},
	}
	summarizer := summary.New(
		model,
		summary.WithTools(staticTool{name: "lookup"}),
		summary.WithPreProcessor(failingPreProcessor{err: errors.New("preprocess failed")}),
	)

	done := make(chan error, 1)
	go func() {
		_, err := summarizer.Summarize(context.Background(), gaictx.SummaryRequest{Text: "input"})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected preprocessing error")
		}
		if !strings.Contains(err.Error(), "preprocess failed") {
			t.Fatalf("expected preprocessing error, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Summarize blocked while loop errors were waiting to be drained")
	}
}

type recordingModel struct {
	response string
	request  ai.AIRequest
}

func (m *recordingModel) Name() string {
	return "recording"
}

func (m *recordingModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	m.request = req
	return &ai.AIResponse{Text: m.response}, nil
}

func (m *recordingModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		m.request = req
		out <- ai.Token{Type: ai.TokenTypeText, Text: m.response}
	}()
	return out
}

func (m *recordingModel) Close() error {
	return nil
}

func (m *recordingModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}

type toolCallModel struct {
	tokens []ai.Token
}

func (m toolCallModel) Name() string {
	return "tool-call"
}

func (m toolCallModel) Generate(context.Context, ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m toolCallModel) GenerateStream(context.Context, ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, len(m.tokens))
	go func() {
		defer close(out)
		for _, token := range m.tokens {
			out <- token
		}
	}()
	return out
}

func (m toolCallModel) Close() error {
	return nil
}

func (m toolCallModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}

type staticTool struct {
	name string
}

func (t staticTool) Name() string {
	return t.name
}

func (t staticTool) Description() string {
	return "static test tool"
}

func (t staticTool) Params() string {
	return "{}"
}

func (t staticTool) Function(*ai.ToolCall) *loop.ToolResponse {
	return &loop.ToolResponse{Text: "ok"}
}

type failingPreProcessor struct {
	err error
}

func (p failingPreProcessor) Process(ai.ToolCall, *loop.ToolResponse) error {
	return p.err
}
