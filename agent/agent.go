package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

// RunInput contains the application input for one agent run.
type RunInput struct {
	ID string
	// TraceContext explicitly selects approved trace-wide dimensions. Nil
	// inherits a trace context already present in ctx; non-nil replaces it.
	TraceContext *gai.TraceContext
	// Prompt separates genuine user content from structured machine context.
	Prompt gaictx.PromptInput
	// MaxTokens overrides Definition.Limits.MaxTokens when it is positive.
	MaxTokens int
	// ResponseFormat requests the output shape for every model call in this run.
	ResponseFormat ai.ResponseFormat
	// Execution overrides selected definition-level execution settings for this run.
	// Tools replaces Definition.Tools when non-nil; an empty non-nil slice disables
	// definition-level tools. Nil inherits Definition.Tools.
	Execution ExecutionConfig
	// Meta carries application data such as user, session, or request IDs.
	Meta map[string]any
}

// ExecutionConfig provides optional configuration for one workflow.
//
// Tool slices are copied when NewRun is called, preserving tool membership and
// order for that workflow. Tool implementations themselves must remain safe for
// concurrent use.
type ExecutionConfig struct {
	Tools []loop.Tool
	// ToolChoice overrides the provider tool-choice setting when non-nil.
	ToolChoice *ai.ToolChoice
	// Reasoning overrides Definition.Reasoning when non-nil.
	Reasoning *ai.ReasoningConfig
}

// Prompt creates the prompt builder used by one run.
type Prompt func(ctx context.Context, input RunInput) (gaictx.PromptBuilder, error)

// Limits controls loop retries, iterations, and model output size.
type Limits struct {
	// MaxLoopIterations limits model/tool iterations. Zero uses the loop default.
	MaxLoopIterations int
	// RetryCount limits model retries. Zero uses the loop default.
	RetryCount int
	// MaxTokens is the default model output limit for the agent.
	MaxTokens int
}

// Definition describes a reusable agent and its workflow middleware.
type Definition struct {
	// Name identifies the agent in diagnostics and is the default name used when
	// the agent is adapted into middleware.
	Name string
	// Model performs the agent's model calls.
	Model ai.Model
	// Tools are available to the model during loop execution. When the model's
	// ai.ModelDescriber descriptor reports native tools are supported, their
	// definitions and text-based invocation protocol are not added to the
	// prompt. Models without a descriptor use ai.NativeToolModel as a legacy
	// fallback. Otherwise, the protocol is added as the first prompt context
	// source unless its builder already contains a tool_definitions source.
	Tools []loop.Tool
	// ToolDefinitionOptions configure the auto-prepended tool-definitions prompt
	// source used for Tools.
	ToolDefinitionOptions []tooldefinitions.Option
	// Prompt builds run-specific instructions and context.
	Prompt Prompt
	// Limits configures loop execution defaults.
	Limits Limits
	// Reasoning configures model reasoning/thinking behavior for every model call.
	Reasoning ai.ReasoningConfig
	// Tokenizer overrides Model.Tokenizer when it is non-nil.
	Tokenizer ai.Tokenizer
	// ToolResponseProcessor can transform tool responses before they enter the transcript.
	ToolResponseProcessor loop.ToolResponseProcessor
	// DebugSink receives agent and workflow lifecycle events.
	DebugSink gai.DebugSink
	// Middleware transforms the run stream in declaration order.
	Middleware []Middleware
}

// Agent is a reusable definition that creates independent workflows.
type Agent struct {
	def Definition
}

// New creates an agent from def. Configuration is validated by NewRun.
func New(def Definition) *Agent {
	return &Agent{def: def}
}

// NewRun builds a single-use workflow for input.
//
// Prompt construction happens before NewRun returns. Model execution and
// middleware processing begin when Workflow.Run is called.
func (a *Agent) NewRun(ctx context.Context, input RunInput) (*Workflow, error) {
	ctx, input = resolveRunTraceContext(ctx, input)
	ctx, obs := newRunCreationObserver(ctx, a, input)
	if a != nil {
		if err := validateMiddleware(a.def.Middleware); err != nil {
			obs.Failed(ctx, "middleware_validation", err)
			obs.Finish(err)
			return nil, err
		}
	}
	l, err := a.newLoop(ctx, input)
	if err != nil {
		obs.Failed(ctx, "loop_creation", err)
		obs.Finish(err)
		return nil, err
	}
	workflow := newWorkflow(input, l, a.name(), a.debugSink(), a.middleware())
	obs.Created(ctx)
	obs.Finish(nil)
	return workflow, nil
}

func resolveRunTraceContext(ctx context.Context, input RunInput) (context.Context, RunInput) {
	if input.TraceContext != nil {
		ctx = gai.WithTraceContext(ctx, *input.TraceContext)
	}
	if traceContext, ok := gai.TraceContextFromContext(ctx); ok {
		input.TraceContext = &traceContext
	}
	return ctx, input
}

func (a *Agent) name() string {
	if a == nil {
		return ""
	}
	return a.def.Name
}

func (a *Agent) debugSink() gai.DebugSink {
	if a == nil {
		return nil
	}
	return a.def.DebugSink
}

func (a *Agent) middleware() []Middleware {
	if a == nil {
		return nil
	}
	return a.def.Middleware
}

func (a *Agent) newLoop(ctx context.Context, input RunInput) (*loop.Loop, error) {
	if a == nil {
		return nil, loop.ErrNilLoop
	}
	if a.def.Model == nil {
		return nil, loop.ErrModelNotConfigured
	}
	if a.def.Prompt == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	tools := cloneTools(a.def.Tools)
	if input.Execution.Tools != nil {
		tools = cloneTools(input.Execution.Tools)
	}
	if _, err := loop.ToolDefinitions(tools); err != nil {
		return nil, err
	}

	promptBuilder, err := a.def.Prompt(ctx, input)
	if err != nil {
		return nil, err
	}
	if promptBuilder == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	if cloner, ok := promptBuilder.(gaictx.PromptBuilderCloner); ok {
		promptBuilder = cloner.ClonePromptBuilder()
		if promptBuilder == nil {
			return nil, loop.ErrPromptNotConfigured
		}
	}
	promptBuilder.SetInput(input.Prompt)
	nativeTools := usesNativeTools(a.def.Model)
	textToolsDisabled := !nativeTools && input.Execution.ToolChoice != nil && input.Execution.ToolChoice.Mode == ai.ToolChoiceNone
	if !nativeTools && input.Execution.ToolChoice != nil {
		if err := validateTextToolChoice(*input.Execution.ToolChoice, tools); err != nil {
			return nil, err
		}
		tools = selectedTextTools(*input.Execution.ToolChoice, tools)
	}
	if !nativeTools {
		hasToolDefinitions := hasContextSource(promptBuilder, "tool_definitions")
		if (input.Execution.Tools != nil || textToolsDisabled) && hasToolDefinitions && len(tools) == 0 {
			remover, ok := promptBuilder.(gaictx.ContextSourceRemover)
			if !ok {
				return nil, fmt.Errorf("prompt builder cannot remove tool definitions for execution tool override")
			}
			if err := remover.RemoveContextSource(ctx, "tool_definitions"); err != nil {
				return nil, err
			}
		} else if len(tools) > 0 && (!hasToolDefinitions || input.Execution.Tools != nil || (input.Execution.ToolChoice != nil && input.Execution.ToolChoice.Mode == ai.ToolChoiceRequired && len(input.Execution.ToolChoice.Names) > 0)) {
			toolSource, err := tooldefinitions.New(nil, tools, a.def.DebugSink, a.def.ToolDefinitionOptions...)
			if err != nil {
				return nil, err
			}
			if hasToolDefinitions {
				replacer, ok := promptBuilder.(gaictx.ContextSourceReplacer)
				if !ok {
					return nil, fmt.Errorf("prompt builder cannot replace tool definitions for execution tool override")
				}
				if err := replacer.ReplaceContextSource(ctx, "tool_definitions", toolSource); err != nil {
					return nil, err
				}
			} else if err := promptBuilder.PrependContextSource(ctx, toolSource); err != nil {
				return nil, err
			}
		}
		if input.Execution.ToolChoice != nil {
			if instruction := textToolChoiceInstruction(*input.Execution.ToolChoice); instruction != "" {
				if err := promptBuilder.AppendSystemInstructions(ctx, gaictx.NewTextPart(instruction)); err != nil {
					return nil, err
				}
			}
		}
	}
	if setter, ok := promptBuilder.(gaictx.TokenizerSetter); ok {
		tokenizer := a.def.Tokenizer
		if tokenizer == nil {
			tokenizer = a.def.Model.Tokenizer()
		}
		if tokenizer != nil {
			setter.SetTokenizer(tokenizer)
		}
	}

	l := loop.New(a.def.Model, tools, promptBuilder, a.def.ToolResponseProcessor)
	if !nativeTools {
		l.ToolTransport = loop.ToolTransportText
	}
	if a.def.Limits.MaxLoopIterations > 0 {
		l.MaxLoopIterations = a.def.Limits.MaxLoopIterations
	}
	if a.def.Limits.RetryCount > 0 {
		l.RetryCount = a.def.Limits.RetryCount
	}
	if input.MaxTokens > 0 {
		l.MaxTokens = input.MaxTokens
	} else {
		l.MaxTokens = a.def.Limits.MaxTokens
	}
	l.ResponseFormat = cloneResponseFormat(input.ResponseFormat)
	l.Reasoning = a.def.Reasoning
	if input.Execution.ToolChoice != nil {
		l.ToolChoice = cloneToolChoice(*input.Execution.ToolChoice)
	}
	if input.Execution.Reasoning != nil {
		l.Reasoning = *input.Execution.Reasoning
	}
	return l, nil
}

func cloneTools(tools []loop.Tool) []loop.Tool {
	if tools == nil {
		return nil
	}
	cloned := make([]loop.Tool, len(tools))
	copy(cloned, tools)
	return cloned
}

func cloneToolChoice(choice ai.ToolChoice) ai.ToolChoice {
	cloned := choice
	cloned.Names = append([]string(nil), choice.Names...)
	return cloned
}

func textToolChoiceInstruction(choice ai.ToolChoice) string {
	if choice.Mode != ai.ToolChoiceRequired {
		return ""
	}
	if len(choice.Names) == 0 {
		return "You must make at least one tool call before producing a normal response."
	}
	return "You must make at least one tool call before producing a normal response. You may call only these selected tools: " + strings.Join(choice.Names, ", ") + "."
}

func validateTextToolChoice(choice ai.ToolChoice, tools []loop.Tool) error {
	if choice.Mode != ai.ToolChoiceRequired {
		return nil
	}
	if len(tools) == 0 {
		return fmt.Errorf("required text tool choice requires at least one tool")
	}
	available := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		available[tool.Name()] = struct{}{}
	}
	for _, name := range choice.Names {
		if _, ok := available[name]; !ok {
			return fmt.Errorf("required text tool %q is not configured", name)
		}
	}
	return nil
}

func selectedTextTools(choice ai.ToolChoice, tools []loop.Tool) []loop.Tool {
	if choice.Mode == ai.ToolChoiceNone {
		return []loop.Tool{}
	}
	if choice.Mode != ai.ToolChoiceRequired || len(choice.Names) == 0 {
		return tools
	}
	selected := make(map[string]struct{}, len(choice.Names))
	for _, name := range choice.Names {
		selected[name] = struct{}{}
	}
	filtered := make([]loop.Tool, 0, len(choice.Names))
	for _, tool := range tools {
		if _, ok := selected[tool.Name()]; ok {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func usesNativeTools(model ai.Model) bool {
	if describer, ok := model.(ai.ModelDescriber); ok {
		return describer.Descriptor().SupportsNativeTools()
	}
	native, ok := model.(ai.NativeToolModel)
	return ok && native.NativeTools()
}

type contextSourceLookup interface {
	HasContextSource(name string) bool
}

func hasContextSource(builder gaictx.PromptBuilder, name string) bool {
	lookup, ok := builder.(contextSourceLookup)
	return ok && lookup.HasContextSource(name)
}
