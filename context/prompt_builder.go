package context

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type Part interface {
	Name() string
	Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error)
	Marshal(ctx context.Context) ([]byte, error)
}

type TextPart struct {
	Content string
	tokens  map[string]int
}

func NewTextPart(content string) TextPart {
	return TextPart{
		Content: content,
		tokens:  make(map[string]int),
	}
}

func (t TextPart) Name() string {
	return "text"
}

func (t TextPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if t.tokens == nil {
		t.tokens = make(map[string]int)
	}
	if count, exists := t.tokens[tokenizer.ID()]; exists {
		return count, nil
	}
	count, err := tokenizer.CountTokens(ctx, t.Content)
	if err != nil {
		return 0, err
	}
	t.tokens[tokenizer.ID()] = count
	return count, nil
}

func (t TextPart) Marshal(ctx context.Context) ([]byte, error) {
	_ = ctx
	return []byte(t.Content), nil
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
	GetUserPrompt() string
}

type TokenBudget interface {
	SetTokenLimit(limit int) error
	GetRemainingTokens() (int, error)
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
	b.ContextParts = contextParts
	return contextParts, nil
}

func (b *Builder) BuildPrompt(ctx context.Context, conv Conversation) (string, error) {
	var parts []Part
	parts = append(parts, b.SystemInstructions...)
	parts = append(parts, b.ContextParts...)
	for _, message := range conv.Messages() {
		parts = append(parts, message)
	}
	return b.Renderer.Render(ctx, parts)
}

func (b *Builder) GetUserPrompt() string {
	return b.userPrompt
}
