package agent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
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

type nativeToolWorkflowModel struct {
	*scriptedWorkflowModel
}

func (nativeToolWorkflowModel) NativeTools() bool { return true }

type disabledNativeToolWorkflowModel struct {
	*scriptedWorkflowModel
}

func (disabledNativeToolWorkflowModel) NativeTools() bool { return false }

type describedToolWorkflowModel struct {
	*scriptedWorkflowModel
	descriptor        ai.ModelDescriptor
	legacyNativeTools bool
}

func (m describedToolWorkflowModel) Descriptor() ai.ModelDescriptor { return m.descriptor }
func (m describedToolWorkflowModel) NativeTools() bool              { return m.legacyNativeTools }
func (m describedToolWorkflowModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	if err := m.descriptor.ValidateRequest(req); err != nil {
		out := make(chan ai.Token, 1)
		out <- ai.Token{Type: ai.TokenTypeErr, Err: err}
		close(out)
		return out
	}
	return m.scriptedWorkflowModel.GenerateStream(ctx, req)
}

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

func (b *testPromptBuilder) BuildRequest(ctx context.Context, conv gaictx.Conversation) (string, []ai.RequestMessage, error) {
	return b.prompt, []ai.RequestMessage{{Role: ai.RequestMessageRoleUser, Text: b.prompt}}, nil
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
		Reasoning: ai.ReasoningConfig{
			Enabled:         true,
			IncludeThoughts: true,
			BudgetTokens:    128,
			Effort:          ai.ReasoningEffortHigh,
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
	if run.Loop.Reasoning != (ai.ReasoningConfig{Enabled: true, IncludeThoughts: true, BudgetTokens: 128, Effort: ai.ReasoningEffortHigh}) {
		t.Fatalf("expected configured reasoning, got %+v", run.Loop.Reasoning)
	}
	if builder == nil || builder.tokenizer == nil {
		t.Fatal("expected model tokenizer to be set on prompt builder")
	}
}

func TestAgentNewRunPreservesNilAndEmptyExecutionToolOverrides(t *testing.T) {
	t.Parallel()

	definitionTool := loop.NewEchoTool()
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})

	inherited, err := assistant.NewRun(context.Background(), textRunInput("inherit tools"))
	if err != nil {
		t.Fatalf("NewRun with nil tools failed: %v", err)
	}
	if len(inherited.Loop.Tools) != 1 || inherited.Loop.Tools[0] != definitionTool {
		t.Fatalf("nil execution tools = %#v, want definition tools", inherited.Loop.Tools)
	}

	disabledInput := textRunInput("disable tools")
	disabledInput.Execution.Tools = []loop.Tool{}
	disabled, err := assistant.NewRun(context.Background(), disabledInput)
	if err != nil {
		t.Fatalf("NewRun with empty tools failed: %v", err)
	}
	if disabled.Loop.Tools == nil {
		t.Fatal("empty execution tools became nil, which would restore definition tools when reused")
	}
	if len(disabled.Loop.Tools) != 0 {
		t.Fatalf("empty execution tools = %#v, want no tools", disabled.Loop.Tools)
	}
}

func TestAgentNewRunExecutionOverridesSnapshotToolsAndConfiguration(t *testing.T) {
	t.Parallel()

	definitionTool := loop.NewEchoTool()
	runTool := namedTool{name: "run_tool"}
	assistant := agent.New(agent.Definition{
		Model:     nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{}},
		Tools:     []loop.Tool{definitionTool},
		Reasoning: ai.ReasoningConfig{Effort: ai.ReasoningEffortLow},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	input := textRunInput("hello")
	input.Execution = agent.ExecutionConfig{
		Tools:      []loop.Tool{runTool},
		ToolChoice: &ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"run_tool"}},
		Reasoning:  &ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh},
	}

	workflow, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	input.Execution.Tools[0] = definitionTool
	input.Execution.ToolChoice.Names[0] = "changed"
	input.Execution.Reasoning.Effort = ai.ReasoningEffortLow

	if len(workflow.Loop.Tools) != 1 || workflow.Loop.Tools[0] != runTool {
		t.Fatalf("loop tools = %#v, want execution tool snapshot", workflow.Loop.Tools)
	}
	if got := workflow.Loop.ToolChoice; !reflect.DeepEqual(got, ai.ToolChoice{Mode: ai.ToolChoiceRequired, Names: []string{"run_tool"}}) {
		t.Fatalf("tool choice = %#v", got)
	}
	if got := workflow.Loop.Reasoning; got != (ai.ReasoningConfig{Enabled: true, Effort: ai.ReasoningEffortHigh}) {
		t.Fatalf("reasoning = %#v", got)
	}
}

func TestAgentNewRunRejectsDuplicateExecutionToolNamesBeforeRun(t *testing.T) {
	t.Parallel()

	tool := loop.NewEchoTool()
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{}},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	input := textRunInput("hello")
	input.Execution.Tools = []loop.Tool{tool, tool}

	_, err := assistant.NewRun(context.Background(), input)
	if err == nil || !strings.Contains(err.Error(), "duplicate tool name") {
		t.Fatalf("NewRun error = %v, want duplicate tool name", err)
	}
}

func TestAgentConcurrentRunsUseIndependentExecutionToolSets(t *testing.T) {
	t.Parallel()

	model := nativeToolWorkflowModel{scriptedWorkflowModel: &scriptedWorkflowModel{
		scripts: [][]ai.Token{{{Type: ai.TokenTypeText, Text: "done"}}, {{Type: ai.TokenTypeText, Text: "done"}}},
	}}
	assistant := agent.New(agent.Definition{
		Model: model,
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, name := range []string{"first", "second"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			input := textRunInput("hello")
			input.Execution.Tools = []loop.Tool{namedTool{name: name}}
			workflow, err := assistant.NewRun(context.Background(), input)
			if err == nil {
				consumeWorkflow(t, workflow)
			}
			errs <- err
		}(name)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("NewRun failed: %v", err)
		}
	}

	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	names := map[string]bool{}
	for _, request := range requests {
		if len(request.Tools) != 1 {
			t.Fatalf("request tools = %#v", request.Tools)
		}
		names[request.Tools[0].Name] = true
	}
	if !names["first"] || !names["second"] {
		t.Fatalf("request tool names = %#v", names)
	}
}

type namedTool struct{ name string }

func (t namedTool) Name() string        { return t.name }
func (t namedTool) Description() string { return "test tool" }
func (t namedTool) Params() ai.ToolParameters {
	return ai.ToolParameters{}
}
func (t namedTool) Function(context.Context, *ai.ToolCall) *loop.ToolResponse {
	return loop.NewToolSuccess("ok")
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

func TestAgentNativeToolModelOmitsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{testContextSource{name: "application_context"}},
	})
	model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
	assistant := agent.New(agent.Definition{
		Model: nativeToolWorkflowModel{model},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	workflow, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 1 || builder.ContextSources[0].Name() != "application_context" {
		t.Fatalf("native tool model added prompt tool protocol: %+v", builder.ContextSources)
	}

	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}
	requests := model.Requests()
	if len(requests) != 1 {
		t.Fatalf("model received %d requests, want 1", len(requests))
	}
	if len(requests[0].Tools) != 1 || requests[0].Tools[0].Name != "echo" {
		t.Fatalf("native tool model request did not include tool definition: %+v", requests[0])
	}
	if strings.Contains(requests[0].Prompt, `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("native tool model request included prompt tool protocol:\n%s", requests[0].Prompt)
	}
}

func TestAgentNativeToolModelWithoutSupportAddsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{loop.NewEchoTool()},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	run, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 1 || builder.ContextSources[0].Name() != "tool_definitions" {
		t.Fatalf("disabled native tool model did not add prompt tool protocol: %+v", builder.ContextSources)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(prompt, `{"type":"function","name":"<tool-name>","arguments":{...}}`) {
		t.Fatalf("disabled native tool model prompt missing tool protocol:\n%s", prompt)
	}
}

func TestAgentModelDescriberControlsPromptToolProtocol(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name               string
		nativeTools        ai.FeatureSupport
		legacyNativeTools  bool
		wantPromptProtocol bool
	}{
		{name: "supported overrides legacy false", nativeTools: ai.FeatureSupportSupported},
		{name: "unknown ignores legacy support", nativeTools: ai.FeatureSupportUnknown, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "unsupported ignores legacy support", nativeTools: ai.FeatureSupportUnsupported, legacyNativeTools: true, wantPromptProtocol: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
			assistant := agent.New(agent.Definition{
				Model: describedToolWorkflowModel{
					scriptedWorkflowModel: &scriptedWorkflowModel{},
					descriptor:            ai.ModelDescriptor{NativeTools: tt.nativeTools},
					legacyNativeTools:     tt.legacyNativeTools,
				},
				Tools: []loop.Tool{loop.NewEchoTool()},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return builder, nil
				},
			})

			if _, err := assistant.NewRun(context.Background(), textRunInput("use echo")); err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			hasPromptProtocol := len(builder.ContextSources) == 1 && builder.ContextSources[0].Name() == "tool_definitions"
			if hasPromptProtocol != tt.wantPromptProtocol {
				t.Fatalf("prompt protocol = %t, want %t; sources: %+v", hasPromptProtocol, tt.wantPromptProtocol, builder.ContextSources)
			}
		})
	}
}

func TestAgentResolvesToolTransportForPromptAndRequest(t *testing.T) {
	tests := []struct {
		name               string
		described          bool
		nativeTools        ai.FeatureSupport
		legacyNativeTools  bool
		wantPromptProtocol bool
		wantRequestTools   bool
	}{
		{name: "descriptor supported", described: true, nativeTools: ai.FeatureSupportSupported, wantRequestTools: true},
		{name: "descriptor unknown", described: true, nativeTools: ai.FeatureSupportUnknown, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "descriptor unsupported", described: true, nativeTools: ai.FeatureSupportUnsupported, legacyNativeTools: true, wantPromptProtocol: true},
		{name: "legacy native model", legacyNativeTools: true, wantRequestTools: true},
		{name: "legacy text model", wantPromptProtocol: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
			model := &scriptedWorkflowModel{scripts: [][]ai.Token{{}}}
			var configuredModel ai.Model = model
			if tt.described {
				configuredModel = describedToolWorkflowModel{
					scriptedWorkflowModel: model,
					descriptor:            ai.ModelDescriptor{NativeTools: tt.nativeTools},
					legacyNativeTools:     tt.legacyNativeTools,
				}
			} else if tt.legacyNativeTools {
				configuredModel = nativeToolWorkflowModel{model}
			}

			assistant := agent.New(agent.Definition{
				Model: configuredModel,
				Tools: []loop.Tool{loop.NewEchoTool()},
				Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
					return builder, nil
				},
			})
			workflow, err := assistant.NewRun(context.Background(), textRunInput("use echo"))
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			consumed := consumeWorkflow(t, workflow)
			if len(consumed.errs) != 0 {
				t.Fatalf("workflow errors: %v", consumed.errs)
			}
			if got := len(builder.ContextSources) == 1 && builder.ContextSources[0].Name() == "tool_definitions"; got != tt.wantPromptProtocol {
				t.Fatalf("prompt protocol = %t, want %t; sources: %+v", got, tt.wantPromptProtocol, builder.ContextSources)
			}
			requests := model.Requests()
			if len(requests) != 1 {
				t.Fatalf("requests = %d, want 1", len(requests))
			}
			if got := len(requests[0].Tools) > 0; got != tt.wantRequestTools {
				t.Fatalf("request tools = %#v, want present=%t", requests[0].Tools, tt.wantRequestTools)
			}
			if len(workflow.Loop.Tools) != 1 || workflow.Loop.Tools[0].Name() != "echo" {
				t.Fatalf("loop lost executable tools: %#v", workflow.Loop.Tools)
			}
		})
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

func TestAgentTextTransportRequiresSelectedToolInPrompt(t *testing.T) {
	t.Parallel()

	builder := gaictx.New(gaictx.Definition{Renderer: &gaictx.SimpleRenderer{}})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{
			namedTool{name: "search"},
			namedTool{name: "weather"},
		},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("find the weather")
	input.Execution.ToolChoice = &ai.ToolChoice{
		Mode:  ai.ToolChoiceRequired,
		Names: []string{"weather"},
	}
	run, err := assistant.NewRun(context.Background(), input)
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
	if !strings.Contains(prompt, "You must make at least one tool call before producing a normal response.") {
		t.Fatalf("prompt missing required-tool instruction:\n%s", prompt)
	}
	if !strings.Contains(prompt, "You may call only these selected tools: weather.") {
		t.Fatalf("prompt missing selected-tool restriction:\n%s", prompt)
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

func TestAgentExecutionToolsReplaceExistingPromptToolDefinitions(t *testing.T) {
	t.Parallel()

	definitionTool := namedTool{name: "definition_tool"}
	runTool := namedTool{name: "run_tool"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []loop.Tool{definitionTool}, nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{staleSource},
	})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("use the run tool")
	input.Execution.Tools = []loop.Tool{runTool}
	run, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 1 || builder.ContextSources[0].Name() != "tool_definitions" {
		t.Fatalf("context sources = %+v, want one tool-definitions source", builder.ContextSources)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: definition_tool") {
		t.Fatalf("prompt retained definition-level tool:\n%s", prompt)
	}
	if !strings.Contains(prompt, "tool: run_tool") {
		t.Fatalf("prompt missing run-level tool:\n%s", prompt)
	}
}

func TestAgentExecutionEmptyToolsRemoveExistingPromptToolDefinitions(t *testing.T) {
	t.Parallel()

	definitionTool := namedTool{name: "definition_tool"}
	staleSource, err := tooldefinitions.New(&gaictx.SimpleRenderer{}, []loop.Tool{definitionTool}, nil)
	if err != nil {
		t.Fatalf("new stale tool source: %v", err)
	}
	builder := gaictx.New(gaictx.Definition{
		Renderer:       &gaictx.SimpleRenderer{},
		ContextSources: []gaictx.ContextSource{staleSource},
	})
	assistant := agent.New(agent.Definition{
		Model: disabledNativeToolWorkflowModel{&scriptedWorkflowModel{}},
		Tools: []loop.Tool{definitionTool},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) {
			return builder, nil
		},
	})

	input := textRunInput("do not use tools")
	input.Execution.Tools = []loop.Tool{}
	run, err := assistant.NewRun(context.Background(), input)
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	if len(builder.ContextSources) != 0 {
		t.Fatalf("context sources = %+v, want no tool definitions", builder.ContextSources)
	}
	if _, err := run.Loop.PromptBuilder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	prompt, err := run.Loop.PromptBuilder.BuildPrompt(context.Background(), nil)
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt, "tool: definition_tool") {
		t.Fatalf("prompt retained definition-level tool:\n%s", prompt)
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
