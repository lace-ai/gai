package agent

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/tooldefinitions"
	"github.com/lace-ai/gai/loop"
)

// RunInput contains the application input for one agent run.
type RunInput struct {
	ID string
	// Prompt separates genuine user content from structured machine context.
	Prompt gaictx.PromptInput
	// MaxTokens overrides Definition.Limits.MaxTokens when it is positive.
	MaxTokens int
	// Meta carries application data such as user, session, or request IDs.
	Meta map[string]any
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
	// Tools are available to the model during loop execution. Their definitions
	// and text-based invocation protocol are added as the first prompt context
	// source unless its builder already contains a tool_definitions source.
	Tools []loop.Tool
	// ToolDefinitionOptions configure the auto-prepended tool-definitions prompt
	// source used for Tools.
	ToolDefinitionOptions []tooldefinitions.Option
	// Prompt builds run-specific instructions and context.
	Prompt Prompt
	// Limits configures loop execution defaults.
	Limits Limits
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

	promptBuilder, err := a.def.Prompt(ctx, input)
	if err != nil {
		return nil, err
	}
	if promptBuilder == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	promptBuilder.SetInput(input.Prompt)
	if len(a.def.Tools) > 0 && !hasContextSource(promptBuilder, "tool_definitions") {
		toolSource, err := tooldefinitions.New(nil, a.def.Tools, a.def.DebugSink, a.def.ToolDefinitionOptions...)
		if err != nil {
			return nil, err
		}
		if err := promptBuilder.PrependContextSource(ctx, toolSource); err != nil {
			return nil, err
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

	l := loop.New(a.def.Model, a.def.Tools, promptBuilder, a.def.ToolResponseProcessor)
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
	return l, nil
}

type contextSourceLookup interface {
	HasContextSource(name string) bool
}

func hasContextSource(builder gaictx.PromptBuilder, name string) bool {
	lookup, ok := builder.(contextSourceLookup)
	return ok && lookup.HasContextSource(name)
}
