package context

import (
	"context"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

const (
	contextTracerName       = "github.com/lace-ai/gai/context"
	promptDebugFullLimit    = 4000
	promptDebugPreviewLimit = 160
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
	DebugSink          gai.DebugSink
}

type Builder struct {
	SystemInstructions []Part
	ContextSources     []ContextSource
	ContextParts       []Part
	Iteration          []Part
	TokenBudget        int
	Renderer           Renderer
	debugSink          gai.DebugSink
	userPrompt         string
	tokenizer          ai.Tokenizer
	OutputTokenReserve int
}

func New(def Definition) *Builder {
	renderer := def.Renderer
	if renderer == nil {
		renderer = XMLRenderer{def.DebugSink, 100}
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

func (b *Builder) SetDebugSink(debugSink gai.DebugSink) {
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

func (b *Builder) BuildContext(ctx context.Context) (contextParts []Part, err error) {
	ctx, obs := newPromptBuilderContextObserver(ctx, b)
	stats := promptContextBuildStats{
		SourceCount:            len(b.ContextSources),
		SystemInstructionCount: len(b.SystemInstructions),
		TokenBudget:            b.TokenBudget,
		OutputTokenReserve:     b.OutputTokenReserve,
		TokenizerPresent:       b.tokenizer != nil,
	}
	defer func() {
		stats.ContextPartCount = len(contextParts)
		obs.FinishContext(err, stats)
	}()

	if b.TokenBudget > 0 {
		stats.SystemTokens = b.SystemInstructionsTokens(ctx)
		stats.RemainingTokens = b.TokenBudget - b.OutputTokenReserve - stats.SystemTokens
	} else {
		obs.TokenBudgetSkipped(ctx)
		stats.RemainingTokens = 1000000 // effectively unlimited
	}
	obs.BuildStarted(ctx, stats)

	for _, source := range b.ContextSources {
		if source == nil {
			obs.SourceSkipped(ctx, "<nil>")
			continue
		}
		if setter, ok := source.(TokenizerSetter); ok && b.tokenizer != nil {
			setter.SetTokenizer(b.tokenizer)
		}
		part, err := source.Function(ctx, stats.RemainingTokens)
		if err != nil {
			obs.SourceFailed(ctx, source.Name(), stats.RemainingTokens, err)
			return nil, err
		}
		if part != nil {
			contextParts = append(contextParts, part)
			stats.IncludedSourceCount++
			tokens, ok := b.partTokens(ctx, part, map[string]any{
				"source": source.Name(),
				"part":   part.Name(),
			})
			if ok && b.TokenBudget > 0 {
				stats.RemainingTokens -= tokens
			}
			obs.SourceIncluded(ctx, source.Name(), part.Name(), promptPartTokenStats{
				Tokens:        tokens,
				TokensCounted: ok,
			}, stats.RemainingTokens)
		}
	}
	b.ContextParts = contextParts
	obs.BuildFinished(ctx, stats)
	return contextParts, nil
}

func (b *Builder) BuildPrompt(ctx context.Context, conv Conversation) (prompt string, err error) {
	ctx, obs := newPromptBuilderRenderObserver(ctx, b)
	stats := promptRenderStats{
		SystemPartCount:  len(b.SystemInstructions),
		ContextPartCount: len(b.ContextParts),
		HasUserPrompt:    b.userPrompt != "",
	}
	defer func() {
		stats.PromptChars = len(prompt)
		obs.FinishRender(err, stats)
	}()

	var parts []Part
	parts = append(parts, NewSystemPart(b.SystemInstructions))
	parts = append(parts, b.ContextParts...)
	if b.userPrompt != "" {
		parts = append(parts, NewMessagePart(RoleUser, NewTextContent(b.userPrompt)))
	}
	if conv != nil {
		messages := conv.Messages()
		stats.ConversationMessageCount = len(messages)
		for _, message := range messages {
			if message.Role != RoleUser {
				parts = append(parts, NewMessagePart(message.Role, message.Content))
			}
		}
	}
	stats.PartCount = len(parts)
	renderCtx, finishRendererRender := obs.StartRendererRender(ctx, stats.PartCount)
	prompt, err = b.Renderer.Render(renderCtx, parts)
	stats.PromptChars = len(prompt)
	finishRendererRender(err, stats.PromptChars)
	if err != nil {
		obs.RenderFailed(ctx, stats, err)
		return "", err
	}
	obs.RenderFinished(ctx, stats, promptDebugFields(ctx, parts, prompt))
	return prompt, nil
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
	if b.tokenizer == nil {
		if len(b.SystemInstructions) > 0 {
			newPromptBuilderDebugObserver(b).TokenCountSkipped(ctx, map[string]any{
				"reason": "tokenizer_missing",
				"scope":  "system_instructions",
				"parts":  len(b.SystemInstructions),
			})
		}
		return count
	}
	for _, part := range b.SystemInstructions {
		tokens, ok := b.partTokens(ctx, part, map[string]any{
			"scope": "system_instructions",
			"part":  part.Name(),
		})
		if !ok {
			continue
		}
		count += tokens
	}
	return count
}

func (b *Builder) partTokens(ctx context.Context, part Part, fields map[string]any) (int, bool) {
	obs := newPromptBuilderDebugObserver(b)
	if b.tokenizer == nil {
		obs.TokenCountSkipped(ctx, mergeDebugFields(fields, map[string]any{
			"reason": "tokenizer_missing",
		}))
		return 0, false
	}
	tokens, err := part.Tokens(ctx, b.tokenizer)
	if err != nil {
		obs.TokenCountFailed(ctx, fields, err)
		return 0, false
	}
	return tokens, true
}

func mergeDebugFields(base map[string]any, extra map[string]any) map[string]any {
	merged := make(map[string]any, len(base)+len(extra))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}
