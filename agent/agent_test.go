package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type testPromptBuilder struct {
	prompt    string
	input     gaictx.PromptInput
	tokenizer ai.Tokenizer
}

type testContextSource struct{ name string }

func (s testContextSource) Name() string { return s.name }
func (s testContextSource) Function(context.Context, int) (gaictx.Part, error) {
	return gaictx.NewTextPart(s.name), nil
}

func (b *testPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
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

func (b *testPromptBuilder) Input() gaictx.PromptInput {
	return b.input.Clone()
}

func (b *testPromptBuilder) SetInput(input gaictx.PromptInput) {
	b.input = input.Clone()
	b.prompt = ""
	if input.User != nil {
		b.prompt = input.User.String()
		return
	}
	if len(input.Context) > 0 && input.Context[0] != nil {
		if node, err := input.Context[0].Render(context.Background()); err == nil {
			b.prompt = node.Value
		}
	}
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
		Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			builder = &testPromptBuilder{}
			return builder, nil
		},
		Limits: agent.Limits{
			MaxLoopIterations: 2,
			RetryCount:        1,
			MaxTokens:         9,
		},
	})

	run, err := assistant.NewRun(context.Background(), textRunInput("input"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if run.Loop.Model != model {
		t.Fatal("expected configured model")
	}
	if len(run.Loop.Tools) != 1 || run.Loop.Tools[0] != tool {
		t.Fatalf("expected configured tools, got %+v", run.Loop.Tools)
	}
	if run.Loop.MaxLoopIterations != 2 {
		t.Fatalf("expected max iterations 2, got %d", run.Loop.MaxLoopIterations)
	}
	if run.Loop.RetryCount != 1 {
		t.Fatalf("expected retry count 1, got %d", run.Loop.RetryCount)
	}
	if run.Loop.MaxTokens != 9 {
		t.Fatalf("expected max tokens 9, got %d", run.Loop.MaxTokens)
	}
	if builder == nil || builder.tokenizer == nil {
		t.Fatal("expected model tokenizer to be set on prompt builder")
	}
}

func TestAgentToolsAutomaticallyAddPromptContract(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{testContextSource{name: "application_context"}},
	})
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	run, err := assistant.NewRun(context.Background(), textRunInput("remember my name"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 2 || builder.ContextSources[0].Name() != "tool_definitions" || builder.ContextSources[1].Name() != "application_context" {
		t.Fatalf("tool definitions were not prepended: %+v", builder.ContextSources)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	for _, expected := range []string{
		"tool: echo",
		`{"type":"function","name":"<tool-name>","arguments":{...}}`,
	} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("automatic tool prompt missing %q:\n%s", expected, prompt)
		}
	}
}

func TestAgentToolDefinitionOptionsCustomizeAutomaticPromptContract(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model:                 &mocks.MockModel{},
		Tools:                 []loop.Tool{loop.NewEchoTool()},
		ToolDefinitionOptions: []tooldefinitions.Option{tooldefinitions.WithUsageProtocol("Use tools only after asking for confirmation.")},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	run, err := assistant.NewRun(context.Background(), textRunInput("remember my name"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(prompt, "Use tools only after asking for confirmation.") {
		t.Fatalf("custom tool definition protocol missing:\n%s", prompt)
	}
	if strings.Contains(prompt, `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("default tool definition protocol still present:\n%s", prompt)
	}
}

func TestAgentDoesNotDuplicateExistingToolDefinitions(t *testing.T) {
	t.Parallel()

	tool := loop.NewEchoTool()
	source, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []loop.Tool{tool}, nil)
	if err != nil {
		t.Fatalf("new tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{source},
	})
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Tools: []loop.Tool{tool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	run, err := assistant.NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if got := strings.Count(prompt, "tool: echo"); got != 1 {
		t.Fatalf("tool definitions rendered %d times:\n%s", got, prompt)
	}
}

func TestAgentNewRunUsesInputMaxTokens(t *testing.T) {
	t.Parallel()

	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Limits: agent.Limits{
			MaxTokens: 9,
		},
	})

	input := textRunInput("input")
	input.MaxTokens = 3
	run, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if run.Loop.MaxTokens != 3 {
		t.Fatalf("expected input max tokens 3, got %d", run.Loop.MaxTokens)
	}
}

func textRunInput(text string) agent.RunInput {
	return agent.RunInput{Prompt: gaictx.PromptInput{User: gaictx.NewTextContent(text)}}
}

func promptUserText(input agent.RunInput) string {
	if input.Prompt.User == nil {
		return ""
	}
	return input.Prompt.User.String()
}

func promptContextValue(input agent.RunInput, name string) string {
	for _, part := range input.Prompt.Context {
		if part == nil || part.Name() != name {
			continue
		}
		node, err := part.Render(context.Background())
		if err == nil {
			return node.Value
		}
	}
	return ""
}

func TestAgentNewRunUsesConfiguredTokenizerOverride(t *testing.T) {
	t.Parallel()

	modelTokenizer := &mocks.MockTokenizer{IDValue: "model"}
	overrideTokenizer := &mocks.MockTokenizer{IDValue: "override"}
	builder := &testPromptBuilder{}
	assistant := agent.New(agent.Definition{
		Model:     &mocks.MockModel{TokenizerValue: modelTokenizer},
		Tokenizer: overrideTokenizer,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	if _, err := assistant.NewRun(context.Background(), agent.RunInput{}); err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if builder.tokenizer != overrideTokenizer {
		t.Fatalf("expected configured tokenizer override, got %v", builder.tokenizer)
	}
}

func TestAgentNewRunFallsBackToModelTokenizer(t *testing.T) {
	t.Parallel()

	modelTokenizer := &mocks.MockTokenizer{IDValue: "model"}
	builder := &testPromptBuilder{}
	assistant := agent.New(agent.Definition{
		Model: &mocks.MockModel{TokenizerValue: modelTokenizer},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	if _, err := assistant.NewRun(context.Background(), agent.RunInput{}); err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if builder.tokenizer != modelTokenizer {
		t.Fatalf("expected model tokenizer fallback, got %v", builder.tokenizer)
	}
}

func TestAgentNewRunRequiresModelAndPrompt(t *testing.T) {
	t.Parallel()

	_, err := agent.New(agent.Definition{}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrModelNotConfigured {
		t.Fatalf("expected ErrModelNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{Model: &mocks.MockModel{}}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured, got %v", err)
	}

	_, err = agent.New(agent.Definition{
		Model:  &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return nil, nil },
	}).NewRun(context.Background(), agent.RunInput{})
	if err != loop.ErrPromptNotConfigured {
		t.Fatalf("expected ErrPromptNotConfigured for nil builder, got %v", err)
	}
}

func TestAgentNewRunReturnsPromptError(t *testing.T) {
	t.Parallel()

	promptErr := errors.New("prompt failed")
	_, err := agent.New(agent.Definition{
		Model: &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return nil, promptErr
		},
	}).NewRun(context.Background(), agent.RunInput{})
	if !errors.Is(err, promptErr) {
		t.Fatalf("expected prompt error, got %v", err)
	}
}
