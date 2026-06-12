package context

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type Part interface {
	Name() string
	Tokens(ctx context.Context, tokenizer ai.Tokenizer) int
	Marshal(ctx context.Context) ([]byte, error)
}

type ContextSource interface {
	Name() string
	Function(ctx context.Context, TokenBudget int) (Part, error)
}

type PromptBuilder interface {
	AppendContextSource(ctx context.Context, source ContextSource) error
	AppendContextSources(ctx context.Context, sources ...ContextSource) error
	AppendSystemInstructions(ctx context.Context, instructions ...Part) error
	BuildContext(ctx context.Context) ([]Part, error)
	BuildPrompt(ctx context.Context, conv Conversation) (string, error)
}

type TokenBudget interface {
	SetTokenLimit(limit int) error
	GetRemainingTokens() (int, error)
}

type Builder struct {
	SystemInstructions []Part
	ContextSources     []ContextSource
	Iteration          []Part
	TokenBudget        int
	Renderer           Renderer
	debugSink          gai.DebugSinkFunc
}

func NewBuilder(renderer Renderer, tokenBudget int) *Builder {
	return &Builder{
		SystemInstructions: []Part{},
		ContextSources:     []ContextSource{},
		Iteration:          []Part{},
		TokenBudget:        tokenBudget,
		Renderer:           renderer,
	}
}

func (b *Builder) SetDebugSink(debugSink gai.DebugSinkFunc) {
	b.debugSink = debugSink
}

func (b *Builder) AppendContextSource(ctx context.Context, source ContextSource) error {
	b.ContextSources = append(b.ContextSources, source)
	return nil
}

func (b *Builder) AppendContextSources(ctx context.Context, sources ...ContextSource) error {
	for _, source := range sources {
		if err := b.AppendContextSource(ctx, source); err != nil {
			return err
		}
	}
	return nil
}

func (b *Builder) AppendSystemInstructions(ctx context.Context, instructions ...Part) error {
	b.SystemInstructions = append(b.SystemInstructions, instructions...)
	return nil
}

func (b *Builder) BuildContext(ctx context.Context) ([]Part, error) {
	var contextParts []Part
	contextParts = append(contextParts, b.SystemInstructions...)
	for _, source := range b.ContextSources {
		part, err := source.Function(ctx, b.TokenBudget)
		if err != nil {
			return nil, err
		}
		contextParts = append(contextParts, part)
	}
	return contextParts, nil
}

func (b *Builder) BuildPrompt(ctx context.Context, conv Conversation) (string, error) {
	var parts []Part
	parts = append(parts, b.SystemInstructions...)
	contextParts, err := b.BuildContext(ctx)
	if err != nil {
		return "", err
	}
	parts = append(parts, contextParts...)
	for _, message := range conv.Messages() {
		parts = append(parts, message)
	}
	return b.Renderer.Render(ctx, parts)
}
