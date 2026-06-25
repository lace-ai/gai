package agent_test

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type scriptedWorkflowModel struct {
	mu       sync.Mutex
	requests []ai.AIRequest
	scripts  [][]ai.Token
}

func (m *scriptedWorkflowModel) Name() string {
	return "scripted-workflow-model"
}

func (m *scriptedWorkflowModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *scriptedWorkflowModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	m.mu.Lock()
	call := len(m.requests)
	m.requests = append(m.requests, req)
	var script []ai.Token
	if call < len(m.scripts) {
		script = append([]ai.Token(nil), m.scripts[call]...)
	}
	m.mu.Unlock()

	go func() {
		defer close(out)
		for _, token := range script {
			select {
			case out <- token:
			case <-ctx.Done():
				select {
				case out <- ai.Token{Type: ai.TokenTypeErr, Err: ctx.Err()}:
				default:
				}
				return
			}
		}
	}()
	return out
}

func (m *scriptedWorkflowModel) Close() error {
	return nil
}

func (m *scriptedWorkflowModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}

func (m *scriptedWorkflowModel) Requests() []ai.AIRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.AIRequest(nil), m.requests...)
}

func TestAgentWorkflowEndToEndWithToolCall(t *testing.T) {
	model := &scriptedWorkflowModel{
		scripts: [][]ai.Token{
			{
				{Type: ai.TokenTypeThought, Text: "checking tool"},
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call_1",
						Type: "function",
						Name: "echo",
						Args: []byte(`{"text":"tool says hi"}`),
					},
				},
			},
			{{Type: ai.TokenTypeText, Text: "final answer"}},
		},
	}
	assistant := agent.New(agent.Definition{
		Name:  "e2e",
		Model: model,
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{
				Renderer: &gaictx.SimpleRenderer{},
			}), nil
		},
		Limits: agent.Limits{
			MaxLoopIterations: 3,
			MaxTokens:         64,
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "final answer" {
		t.Fatalf("unexpected final token text: %q", got)
	}
	if len(consumed.statuses) != 2 {
		t.Fatalf("expected two loop statuses, got %#v", consumed.statuses)
	}

	result := workflow.Result()
	if !result.Complete {
		t.Fatalf("workflow result was not marked complete: %+v", result)
	}
	if result.Text != "final answer" || result.Primary.Text != "final answer" {
		t.Fatalf("unexpected workflow text: %+v", result)
	}
	if result.Reasoning != "checking tool" || result.Primary.Reasoning != "checking tool" {
		t.Fatalf("unexpected reasoning capture: %+v", result)
	}
	if len(result.Primary.Iterations) != 2 {
		t.Fatalf("expected two iterations, got %+v", result.Primary.Iterations)
	}
	first := result.Primary.Iterations[0]
	var toolPart *loop.IterationPart
	for i := range first.Parts {
		if first.Parts[i].ToolReq != nil {
			toolPart = &first.Parts[i]
			break
		}
	}
	if toolPart == nil || toolPart.ToolResp == nil {
		t.Fatalf("expected first iteration to contain executed tool call, got %+v", first.Parts)
	}
	if toolPart.ToolResp.TextValue() != "tool says hi" {
		t.Fatalf("unexpected tool response: %+v", toolPart.ToolResp)
	}
	if len(result.Primary.Messages) != 4 {
		t.Fatalf("expected user, tool call, tool result, and final assistant messages, got %+v", result.Primary.Messages)
	}
	if result.Primary.Messages[1].Content.Type() != gaictx.ContentTypeToolCall ||
		result.Primary.Messages[2].Content.Type() != gaictx.ContentTypeToolResult ||
		result.Primary.Messages[3].Content.String() != "final answer" {
		t.Fatalf("unexpected reconstructed messages: %+v", result.Primary.Messages)
	}

	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected two model requests, got %d", len(requests))
	}
	if requests[0].MaxTokens != 64 || len(requests[0].Tools) != 1 || requests[0].Tools[0].Name != "echo" {
		t.Fatalf("first request did not include limits and tools: %+v", requests[0])
	}
	if !strings.Contains(requests[1].Prompt, "tool res: tool says hi") {
		t.Fatalf("second prompt did not include tool result:\n%s", requests[1].Prompt)
	}
}

func TestAgentWorkflowEndToEndWithAppendMiddleware(t *testing.T) {
	main := agent.New(agent.Definition{
		Name:  "main",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "answer"}}}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
		},
		Middleware: []agent.Middleware{
			agent.NewAgentMiddleware(agent.New(agent.Definition{
				Name:  "audit",
				Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: " audited"}}}},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}}), nil
				},
			}), agent.AgentMiddlewareConfig{
				Name:   "audit",
				Output: agent.AppendOutput,
			}),
		},
	})

	workflow, err := main.NewRun(context.Background(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("unexpected workflow errors: %v", consumed.errs)
	}
	if got := tokensText(consumed.tokens); got != "answer audited" {
		t.Fatalf("unexpected transformed output: %q", got)
	}

	result := workflow.Result()
	if !result.Complete || result.Primary.Text != "answer" || result.Text != "answer audited" {
		t.Fatalf("unexpected workflow result: %+v", result)
	}
	if len(result.Stages) != 1 || result.Stages[0].Name != "audit" || result.Stages[0].Result.Text != " audited" {
		t.Fatalf("unexpected middleware stages: %+v", result.Stages)
	}
}
