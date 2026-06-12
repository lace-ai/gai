package context

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type ContextSource interface {
	Name() string
	Function(ctx context.Context, TokenBudget int) (Part, error)
}

type TokenizerSetter interface {
	SetTokenizer(tokenizer ai.Tokenizer)
}

type PromptBuilder interface {
	AppendContextSource(ctx context.Context, source ContextSource) error
	AppendContextSources(ctx context.Context, sources ...ContextSource) error
	AppendSystemInstructions(ctx context.Context, instructions ...Part) error
	BuildContext(ctx context.Context) ([]Part, error)
	BuildPrompt(ctx context.Context, conv Conversation) (string, error)
	GetUserPrompt() string
	SetUserPrompt(prompt string)
}

type TokenBudget interface {
	SetTokenLimit(limit int) error
	GetRemainingTokens() (int, error)
}

type Definition struct {
	Renderer           Renderer
	SystemInstructions []Part
	ContextSources     []ContextSource
	UserPrompt         string
	TokenBudget        int
	DebugSink          gai.DebugSinkFunc
}

type Builder struct {
	SystemInstructions []Part
	ContextSources     []ContextSource
	ContextParts       []Part
	Iteration          []Part
	TokenBudget        int
	Renderer           Renderer
	debugSink          gai.DebugSinkFunc
	userPrompt         string
}

func New(def Definition) *Builder {
	renderer := def.Renderer
	if renderer == nil {
		renderer = XMLRenderer{}
	}
	return &Builder{
		SystemInstructions: append([]Part{}, def.SystemInstructions...),
		ContextSources:     append([]ContextSource{}, def.ContextSources...),
		ContextParts:       []Part{},
		Iteration:          []Part{},
		TokenBudget:        def.TokenBudget,
		Renderer:           renderer,
		debugSink:          def.DebugSink,
		userPrompt:         def.UserPrompt,
	}
}

func NewBuilder(renderer Renderer, tokenBudget int) *Builder {
	return New(Definition{
		Renderer:    renderer,
		TokenBudget: tokenBudget,
	})
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
	for _, source := range b.ContextSources {
		part, err := source.Function(ctx, b.TokenBudget)
		if err != nil {
			return nil, err
		}
		contextParts = append(contextParts, part)
	}
	b.ContextParts = contextParts
	return contextParts, nil
}

func (b *Builder) BuildPrompt(ctx context.Context, conv Conversation) (string, error) {
	var parts []Part
	parts = append(parts, b.SystemInstructions...)
	parts = append(parts, b.ContextParts...)
	if b.userPrompt != "" {
		parts = append(parts, NewTextPart(b.userPrompt))
	}
	for _, message := range conv.Messages() {
		parts = append(parts, message)
	}
	return b.Renderer.Render(ctx, parts)
}

func (b *Builder) GetUserPrompt() string {
	return b.userPrompt
}

func (b *Builder) SetTokenLimit(limit int) error {
	if limit < 0 {
		return ErrInvalidTokenLimit
	}
	b.TokenBudget = limit
	return nil
}

func (b *Builder) SetUserPrompt(prompt string) {
	b.userPrompt = prompt
}
