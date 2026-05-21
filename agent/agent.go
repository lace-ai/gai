package agent

import (
	"github.com/lace-ai/gai/ai"
	aicontext "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

type RunInput struct {
	ID        string
	Text      string
	MaxTokens int
	Meta      map[string]any
}

type PromptBuilderFactory func(input RunInput) aicontext.PromptBuilder

type Definition struct {
	Model                ai.Model
	Tools                []loop.Tool
	PromptBuilderFactory PromptBuilderFactory
	PreProcessToolRes    loop.ToolResPreProcessor
	MaxLoopIterations    int
	RetryCount           int
	MaxTokens            int
}

func NewLoop(def Definition, input RunInput) (*loop.Loop, error) {
	if def.Model == nil {
		return nil, loop.ErrModelNotConfigured
	}
	if def.PromptBuilderFactory == nil {
		return nil, loop.ErrPromptNotConfigured
	}
	promptBuilder := def.PromptBuilderFactory(input)
	if promptBuilder == nil {
		return nil, loop.ErrPromptNotConfigured
	}

	l := loop.New(def.Model, def.Tools, promptBuilder, def.PreProcessToolRes)
	if def.MaxLoopIterations > 0 {
		l.MaxLoopIterations = def.MaxLoopIterations
	}
	if def.RetryCount > 0 {
		l.RetryCount = def.RetryCount
	}
	if input.MaxTokens > 0 {
		l.MaxTokens = input.MaxTokens
	} else {
		l.MaxTokens = def.MaxTokens
	}
	return l, nil
}
