package agent_test

import (
	"context"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type testPromptBuilder struct {
	prompt    string
	tokenizer ai.Tokenizer
}

func (b *testPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *testPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *testPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *testPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	return nil, nil
}

func (b *testPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	return b.prompt, nil
}

func (b *testPromptBuilder) GetUserPrompt() string {
	return b.prompt
}

func (b *testPromptBuilder) SetUserPrompt(prompt string) {
	b.prompt = prompt
}

func (b *testPromptBuilder) SetTokenizer(tokenizer ai.Tokenizer) {
	b.tokenizer = tokenizer
}

func TestAgentNewRunCreatesLoop(t *testing.T) {
	t.Parallel()

	model := &mocks.MockModel{}
	tool := loop.NewEchoTool()
	var builder *testPromptBuilder

	assistant := agent.New(agent.Definition{
		Name:  "test-agent",
		Model: model,
		Tools: []loop.Tool{tool},
		Prompt: func(input agent.RunInput) gaictx.PromptBuilder {
			builder = &testPromptBuilder{prompt: input.Text}
			return builder
		},
		Limits: agent.Limits{
			MaxLoopIterations: 2,
			RetryCount:        1,
			MaxTokens:         9,
		},
	})

	run, err := assistant.NewRun(agent.RunInput{Text: "input"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if run.Model != model {
		t.Fatal("expected configured model")
	}
	if len(run.Tools) != 1 || run.Tools[0] != tool {
		t.Fatalf("expected configured tools, got %+v", run.Tools)
	}
	if run.MaxLoopIterations != 2 {
		t.Fatalf("expected max iterations 2, got %d", run.MaxLoopIterations)
	}
	if run.RetryCount != 1 {
		t.Fatalf("expected retry count 1, got %d", run.RetryCount)
	}
	if run.MaxTokens != 9 {
		t.Fatalf("expected max tokens 9, got %d", run.MaxTokens)
	}
	if builder == nil || builder.tokenizer == nil {
		t.Fatal("expected model tokenizer to be set on prompt builder")
	}
}

func TestAgentNewRunUsesInputMaxTokens(t *testing.T) {
	t.Parallel()

	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Prompt: func(input agent.RunInput) gaictx.PromptBuilder {
			return &testPromptBuilder{prompt: input.Text}
		},
		Limits: agent.Limits{
			MaxTokens: 9,
		},
	})

	run, err := assistant.NewRun(agent.RunInput{Text: "input", MaxTokens: 3})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if run.MaxTokens != 3 {
		t.Fatalf("expected input max tokens 3, got %d", run.MaxTokens)
	}
}

func TestAgentNewRunRequiresModelAndPrompt(t *testing.T) {
	t.Parallel()

	_, err := agent.New(agent.Definition{}).NewRun(agent.RunInput{})
	if err != loop.ErrModelNotConfigured {
		t.Fatalf("expected ErrModelNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{Model: &mocks.MockModel{}}).NewRun(agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{
		Model:  &mocks.MockModel{},
		Prompt: func(agent.RunInput) gaictx.PromptBuilder { return nil },
	}).NewRun(agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured for nil builder, got %v", err)
	}
}
