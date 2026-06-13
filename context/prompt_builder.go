package context

import (
	"context"
	"fmt"

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
	SetOutputTokenReserve(reserve int) error
	GetRemainingTokens() (int, error)
	Tokenizer() ai.Tokenizer
}

type Definition struct {
	Renderer           Renderer
	SystemInstructions []Part
	ContextSources     []ContextSource
	UserPrompt         string
	TokenBudget        int
	OutputTokenReserve int
	Tokenizer          ai.Tokenizer
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
	tokenizer          ai.Tokenizer
	OutputTokenReserve int
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
		OutputTokenReserve: def.OutputTokenReserve,
		Renderer:           renderer,
		debugSink:          def.DebugSink,
		userPrompt:         def.UserPrompt,
		tokenizer:          def.Tokenizer,
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
	if setter, ok := source.(TokenizerSetter); ok && b.tokenizer != nil {
		setter.SetTokenizer(b.tokenizer)
	}
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
	remainingTokens := b.TokenBudget - b.OutputTokenReserve - b.SystemInstructionsTokens(ctx)
	for _, source := range b.ContextSources {
		if setter, ok := source.(TokenizerSetter); ok && b.tokenizer != nil {
			setter.SetTokenizer(b.tokenizer)
		}
		part, err := source.Function(ctx, remainingTokens)
		if err != nil {
			return nil, err
		}
		if part != nil {
			contextParts = append(contextParts, part)
			tokens, err := part.Tokens(ctx, b.tokenizer)
			if err != nil {
				if b.debugSink != nil {
				}
				continue
			}
			remainingTokens -= tokens
		}
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
	if conv != nil {
		for _, message := range conv.Messages() {
			parts = append(parts, message)
		}
	}
	return b.Renderer.Render(ctx, parts)
}

func (b *Builder) GetUserPrompt() string {
	return b.userPrompt
}

func (b *Builder) SetTokenLimit(limit int) error {
	if limit < 0 {
		return fmt.Errorf("%w: %d", ErrInvalideTokenLimit, limit)
	}
	b.TokenBudget = limit
	return nil
}

func (b *Builder) SetUserPrompt(prompt string) {
	b.userPrompt = prompt
}

func (b *Builder) Tokenizer() ai.Tokenizer {
	return b.tokenizer
}

func (b *Builder) SetTokenizer(tokenizer ai.Tokenizer) {
	b.tokenizer = tokenizer
}

func (b *Builder) SetOutputTokenReserve(reserve int) error {
	if reserve < 0 {
		return fmt.Errorf("%w: %d", ErrInvalidOutputReserve, reserve)
	}
	b.OutputTokenReserve = reserve
	return nil
}

func (b *Builder) SystemInstructionsTokens(ctx context.Context) int {
	count := 0
	for _, part := range b.SystemInstructions {
		tokens, err := part.Tokens(ctx, b.tokenizer)
		if err != nil {
			if b.debugSink != nil {
			}
			continue
		}
		count += tokens
	}
	return count
}
