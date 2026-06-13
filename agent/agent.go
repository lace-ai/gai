package agent

import (
	"context"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

type RunInput struct {
	ID        string
	Text      string
	MaxTokens int
	Meta      map[string]any
}

type Prompt func(ctx context.Context, input RunInput) gaictx.PromptBuilder

type Limits struct {
	MaxLoopIterations int
	RetryCount        int
	MaxTokens         int
}

type Definition struct {
	Name         string
	Model        ai.Model
	Tools        []loop.Tool
	Prompt       Prompt
	Limits       Limits
	Tokenizer    ai.Tokenizer
	Preprocessor loop.ToolResPreProcessor
}

type Agent struct {
	def Definition
}

func New(def Definition) *Agent {
	return &Agent{def: def}
}

func (a *Agent) NewRun(ctx context.Context, input RunInput) (*loop.Loop, error) {
	if a == nil {
		return nil, loop.ErrNilAgent
	}
	if a.def.Model == nil {
		return nil, loop.ErrModelNotConfigured
	}
	if a.def.Prompt == nil {
		return nil, loop.ErrPromptNotConfigured
	}

	promptBuilder := a.def.Prompt(ctx, input)
	if promptBuilder == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	if setter, ok := promptBuilder.(gaictx.TokenizerSetter); ok {
		setter.SetTokenizer(a.def.Model.Tokenizer())
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
