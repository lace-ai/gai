package agent

import (
	"context"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

// RunInput contains the application input for one agent run.
//
// When an agent is executed through AgentMiddleware, Result contains a snapshot
// of the upstream workflow. Result is nil for the primary agent.
type RunInput struct {
	ID string
	// Text is the user input for a primary agent and the current visible output
	// for an agent running as middleware.
	Text string
	// MaxTokens overrides Definition.Limits.MaxTokens when it is positive.
	MaxTokens int
	// Meta carries application data such as user, session, or request IDs.
	Meta map[string]any
	// Result gives middleware agents typed access to the original input, primary
	// transcript, current output, errors, and results from earlier stages.
	Result *WorkflowResult
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
	// Tools are available to the model during loop execution.
	Tools []loop.Tool
	// Prompt builds run-specific instructions and context.
	Prompt Prompt
	// Limits configures loop execution defaults.
	Limits Limits
	// Tokenizer overrides Model.Tokenizer when it is non-nil.
	Tokenizer ai.Tokenizer
	// Preprocessor can transform tool responses before they enter the transcript.
	Preprocessor loop.ToolResPreProcessor
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
	if a != nil {
		if err := validateMiddleware(a.def.Middleware); err != nil {
			return nil, err
		}
	}
	l, err := a.newLoop(ctx, input)
	if err != nil {
		return nil, err
	}
	return newWorkflow(input, l, a.def.Middleware), nil
}

func (a *Agent) newLoop(ctx context.Context, input RunInput) (*loop.Loop, error) {
	if a == nil {
		return nil, loop.ErrNilAgent
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
	if setter, ok := promptBuilder.(gaictx.TokenizerSetter); ok {
		tokenizer := a.def.Tokenizer
		if tokenizer == nil {
			tokenizer = a.def.Model.Tokenizer()
		}
		if tokenizer != nil {
			setter.SetTokenizer(tokenizer)
		}
	}

	l := loop.New(a.def.Model, a.def.Tools, promptBuilder, a.def.Preprocessor)
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
