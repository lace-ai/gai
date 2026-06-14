package context

import (
	"context"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
)

const contextTracerName = "github.com/lace-ai/gai/context"
const promptDebugFullLimit = 4000
const promptDebugPreviewLimit = 160

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
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "build_context",
		attribute.Int("context.source_count", len(b.ContextSources)),
		attribute.Int("context.system_instruction_count", len(b.SystemInstructions)),
		attribute.Int("context.token_budget", b.TokenBudget),
		attribute.Int("context.output_token_reserve", b.OutputTokenReserve),
		attribute.Bool("context.tokenizer_present", b.tokenizer != nil),
	)
	systemTokens := 0
	remainingTokens := 0
	includedSourceCount := 0
	defer func() {
		span.SetAttributes(
			attribute.Int("context.system_tokens", systemTokens),
			attribute.Int("context.remaining_tokens", remainingTokens),
			attribute.Int("context.context_parts", len(contextParts)),
			attribute.Int("context.included_source_count", includedSourceCount),
		)
		gai.EndSpan(span, err)
	}()

	if b.TokenBudget > 0 {
		systemTokens = b.SystemInstructionsTokens(ctx)
		remainingTokens = b.TokenBudget - b.OutputTokenReserve - systemTokens
	} else {
		b.emit(ctx, "prompt_builder_token_budget_skipped", map[string]any{
			"reason": "token_budget_not_set",
		}, nil)
		remainingTokens = 1000000 // effectively unlimited
	}
	span.SetAttributes(
		attribute.Int("context.system_tokens", systemTokens),
		attribute.Int("context.remaining_tokens", remainingTokens),
	)
	b.emit(ctx, "prompt_builder_context_build_started", map[string]any{
		"source_count":             len(b.ContextSources),
		"system_instruction_count": len(b.SystemInstructions),
		"system_tokens":            systemTokens,
		"token_budget":             b.TokenBudget,
		"output_token_reserve":     b.OutputTokenReserve,
		"remaining_tokens":         remainingTokens,
		"tokenizer_present":        b.tokenizer != nil,
	}, nil)

	for _, source := range b.ContextSources {
		if setter, ok := source.(TokenizerSetter); ok && b.tokenizer != nil {
			setter.SetTokenizer(b.tokenizer)
		}
		part, err := source.Function(ctx, remainingTokens)
		if err != nil {
			span.SetAttributes(attribute.String("context.failed_source", source.Name()))
			b.emit(ctx, "prompt_builder_source_failed", map[string]any{
				"source":           source.Name(),
				"remaining_tokens": remainingTokens,
			}, err)
			return nil, err
		}
		if part != nil {
			contextParts = append(contextParts, part)
			includedSourceCount++
			tokens, ok := b.partTokens(ctx, part, map[string]any{
				"source": source.Name(),
				"part":   part.Name(),
			})
			if ok && b.TokenBudget > 0 {
				remainingTokens -= tokens
			}
			b.emit(ctx, "prompt_builder_source_included", map[string]any{
				"source":           source.Name(),
				"part":             part.Name(),
				"tokens":           tokens,
				"tokens_counted":   ok,
				"remaining_tokens": remainingTokens,
			}, nil)
		}
	}
	b.ContextParts = contextParts
	b.emit(ctx, "prompt_builder_context_build_finished", map[string]any{
		"source_count":     len(b.ContextSources),
		"context_parts":    len(contextParts),
		"remaining_tokens": remainingTokens,
	}, nil)
	return contextParts, nil
}

func (b *Builder) BuildPrompt(ctx context.Context, conv Conversation) (prompt string, err error) {
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "render_prompt",
		attribute.Int("context.system_parts", len(b.SystemInstructions)),
		attribute.Int("context.context_parts", len(b.ContextParts)),
		attribute.Bool("context.has_user_prompt", b.userPrompt != ""),
	)
	conversationMessages := 0
	partCount := 0
	defer func() {
		span.SetAttributes(
			attribute.Int("context.conversation_messages", conversationMessages),
			attribute.Int("context.part_count", partCount),
			attribute.Int("context.prompt_chars", len(prompt)),
		)
		gai.EndSpan(span, err)
	}()

	var parts []Part
	parts = append(parts, b.SystemInstructions...)
	parts = append(parts, b.ContextParts...)
	if b.userPrompt != "" {
		parts = append(parts, NewTextPart(b.userPrompt))
	}
	if conv != nil {
		messages := conv.Messages()
		conversationMessages = len(messages)
		for _, message := range messages {
			parts = append(parts, message)
		}
	}
	partCount = len(parts)
	renderCtx, renderSpan := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "renderer_render",
		attribute.Int("context.part_count", partCount),
	)
	prompt, err = b.Renderer.Render(renderCtx, parts)
	renderSpan.SetAttributes(attribute.Int("context.prompt_chars", len(prompt)))
	gai.EndSpan(renderSpan, err)
	fields := map[string]any{
		"part_count":            len(parts),
		"system_parts":          len(b.SystemInstructions),
		"context_parts":         len(b.ContextParts),
		"has_user_prompt":       b.userPrompt != "",
		"conversation_messages": conversationMessages,
	}
	if err != nil {
		b.emit(ctx, "prompt_builder_render_failed", fields, err)
		return "", err
	}
	fields["prompt_chars"] = len(prompt)
	if b.debugSink != nil && b.debugSink.IncludeSensitiveData() {
		for key, value := range b.promptDebugFields(ctx, parts, prompt) {
			fields[key] = value
		}
	}
	b.emit(ctx, "prompt_builder_render_finished", fields, nil)
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
			b.emit(ctx, "prompt_builder_token_count_skipped", map[string]any{
				"reason": "tokenizer_missing",
				"scope":  "system_instructions",
				"parts":  len(b.SystemInstructions),
			}, nil)
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
	if b.tokenizer == nil {
		b.emit(ctx, "prompt_builder_token_count_skipped", mergeDebugFields(fields, map[string]any{
			"reason": "tokenizer_missing",
		}), nil)
		return 0, false
	}
	tokens, err := part.Tokens(ctx, b.tokenizer)
	if err != nil {
		b.emit(ctx, "prompt_builder_token_count_failed", fields, err)
		return 0, false
	}
	return tokens, true
}

func (b *Builder) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if b == nil || b.debugSink == nil {
		return
	}
	b.debugSink.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:Builder",
		Fields: fields,
		Err:    err,
	})
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

func (b *Builder) promptDebugFields(ctx context.Context, parts []Part, prompt string) map[string]any {
	fields := map[string]any{
		"prompt_render_mode": "full",
	}
	if len(prompt) <= promptDebugFullLimit {
		fields["prompt"] = prompt
		return fields
	}

	fields["prompt_render_mode"] = "structured"
	fields["prompt_head"] = clippedPrompt(prompt, 1000, false)
	fields["prompt_tail"] = clippedPrompt(prompt, 1000, true)
	fields["prompt_structure"] = b.promptStructure(ctx, parts)
	return fields
}

func (b *Builder) promptStructure(ctx context.Context, parts []Part) []map[string]any {
	structure := make([]map[string]any, 0, len(parts))
	for i, part := range parts {
		entry := map[string]any{
			"index": i,
		}
		if part == nil {
			entry["name"] = "<nil>"
			structure = append(structure, entry)
			continue
		}
		entry["name"] = part.Name()
		raw, err := part.Marshal(ctx)
		if err != nil {
			entry["marshal_error"] = err.Error()
			structure = append(structure, entry)
			continue
		}
		content := string(raw)
		entry["chars"] = len(content)
		entry["preview"] = clippedPrompt(content, promptDebugPreviewLimit, false)
		if len(content) > promptDebugPreviewLimit {
			entry["preview_tail"] = clippedPrompt(content, promptDebugPreviewLimit, true)
		}
		structure = append(structure, entry)
	}
	return structure
}

func clippedPrompt(text string, limit int, tail bool) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	if tail {
		return "..." + text[len(text)-limit:]
	}
	return text[:limit] + "..."
}
