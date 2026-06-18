package context

import (
	"context"
	"strings"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type promptContextBuildStats struct {
	SourceCount            int
	SystemInstructionCount int
	SystemTokens           int
	TokenBudget            int
	OutputTokenReserve     int
	RemainingTokens        int
	ContextPartCount       int
	IncludedSourceCount    int
	TokenizerPresent       bool
}

type promptRenderStats struct {
	SystemPartCount          int
	ContextPartCount         int
	ConversationMessageCount int
	PartCount                int
	PromptChars              int
	HasUserPrompt            bool
}

type promptPartTokenStats struct {
	Tokens        int
	TokensCounted bool
}

type promptBuilderObserver struct {
	debug gai.DebugSink
	span  trace.Span
}

func newPromptBuilderDebugObserver(b *Builder) *promptBuilderObserver {
	if b == nil {
		return &promptBuilderObserver{}
	}
	return &promptBuilderObserver{debug: b.debugSink}
}

func newPromptBuilderContextObserver(ctx context.Context, b *Builder) (context.Context, *promptBuilderObserver) {
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "build_context",
		attribute.Int("context.source_count", len(b.ContextSources)),
		attribute.Int("context.system_instruction_count", len(b.SystemInstructions)),
		attribute.Int("context.token_budget", b.TokenBudget),
		attribute.Int("context.output_token_reserve", b.OutputTokenReserve),
		attribute.Bool("context.tokenizer_present", b.tokenizer != nil),
	)
	return ctx, &promptBuilderObserver{
		debug: b.debugSink,
		span:  span,
	}
}

func newPromptBuilderRenderObserver(ctx context.Context, b *Builder) (context.Context, *promptBuilderObserver) {
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "render_prompt",
		attribute.Int("context.system_parts", len(b.SystemInstructions)),
		attribute.Int("context.context_parts", len(b.ContextParts)),
		attribute.Bool("context.has_user_prompt", b.userPrompt != ""),
	)
	return ctx, &promptBuilderObserver{
		debug: b.debugSink,
		span:  span,
	}
}

func (o *promptBuilderObserver) FinishContext(err error, stats promptContextBuildStats) {
	if o == nil || o.span == nil {
		return
	}
	o.span.SetAttributes(
		attribute.Int("context.system_tokens", stats.SystemTokens),
		attribute.Int("context.remaining_tokens", stats.RemainingTokens),
		attribute.Int("context.context_parts", stats.ContextPartCount),
		attribute.Int("context.included_source_count", stats.IncludedSourceCount),
	)
	gai.EndSpan(o.span, err)
}

func (o *promptBuilderObserver) FinishRender(err error, stats promptRenderStats) {
	if o == nil || o.span == nil {
		return
	}
	o.span.SetAttributes(
		attribute.Int("context.conversation_messages", stats.ConversationMessageCount),
		attribute.Int("context.part_count", stats.PartCount),
		attribute.Int("context.prompt_chars", stats.PromptChars),
	)
	gai.EndSpan(o.span, err)
}

func (o *promptBuilderObserver) StartRendererRender(ctx context.Context, partCount int) (context.Context, func(error, int)) {
	renderCtx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.prompt_builder", "context.operation", "renderer_render",
		attribute.Int("context.part_count", partCount),
	)
	return renderCtx, func(err error, promptChars int) {
		span.SetAttributes(attribute.Int("context.prompt_chars", promptChars))
		gai.EndSpan(span, err)
	}
}

func (o *promptBuilderObserver) TokenBudgetSkipped(ctx context.Context) {
	o.emit(ctx, "prompt_builder_token_budget_skipped", map[string]any{
		"reason": "token_budget_not_set",
	}, nil)
}

func (o *promptBuilderObserver) BuildStarted(ctx context.Context, stats promptContextBuildStats) {
	if o == nil {
		return
	}
	if o.span != nil {
		o.span.SetAttributes(
			attribute.Int("context.system_tokens", stats.SystemTokens),
			attribute.Int("context.remaining_tokens", stats.RemainingTokens),
		)
	}
	o.emit(ctx, "prompt_builder_context_build_started", map[string]any{
		"source_count":             stats.SourceCount,
		"system_instruction_count": stats.SystemInstructionCount,
		"system_tokens":            stats.SystemTokens,
		"token_budget":             stats.TokenBudget,
		"output_token_reserve":     stats.OutputTokenReserve,
		"remaining_tokens":         stats.RemainingTokens,
		"tokenizer_present":        stats.TokenizerPresent,
	}, nil)
}

func (o *promptBuilderObserver) SourceFailed(ctx context.Context, source string, remainingTokens int, err error) {
	if o == nil {
		return
	}
	if o.span != nil {
		o.span.SetAttributes(attribute.String("context.failed_source", source))
	}
	o.emit(ctx, "prompt_builder_source_failed", map[string]any{
		"source":           source,
		"remaining_tokens": remainingTokens,
	}, err)
}

func (o *promptBuilderObserver) SourceSkipped(ctx context.Context, source string) {
	o.emit(ctx, "prompt_builder_source_skipped", map[string]any{
		"source": source,
	}, nil)
}

func (o *promptBuilderObserver) SourceIncluded(ctx context.Context, source string, part string, stats promptPartTokenStats, remainingTokens int) {
	o.emit(ctx, "prompt_builder_source_included", map[string]any{
		"source":           source,
		"part":             part,
		"tokens":           stats.Tokens,
		"tokens_counted":   stats.TokensCounted,
		"remaining_tokens": remainingTokens,
	}, nil)
}

func (o *promptBuilderObserver) BuildFinished(ctx context.Context, stats promptContextBuildStats) {
	o.emit(ctx, "prompt_builder_context_build_finished", map[string]any{
		"source_count":     stats.SourceCount,
		"context_parts":    stats.ContextPartCount,
		"remaining_tokens": stats.RemainingTokens,
	}, nil)
}

func (o *promptBuilderObserver) RenderFailed(ctx context.Context, stats promptRenderStats, err error) {
	o.emit(ctx, "prompt_builder_render_failed", map[string]any{
		"part_count":            stats.PartCount,
		"system_parts":          stats.SystemPartCount,
		"context_parts":         stats.ContextPartCount,
		"has_user_prompt":       stats.HasUserPrompt,
		"conversation_messages": stats.ConversationMessageCount,
	}, err)
}

func (o *promptBuilderObserver) RenderFinished(ctx context.Context, stats promptRenderStats, sensitiveFields map[string]any) {
	fields := map[string]any{
		"part_count":            stats.PartCount,
		"system_parts":          stats.SystemPartCount,
		"context_parts":         stats.ContextPartCount,
		"has_user_prompt":       stats.HasUserPrompt,
		"conversation_messages": stats.ConversationMessageCount,
		"prompt_chars":          stats.PromptChars,
	}
	if o.includeSensitiveData() {
		for key, value := range sensitiveFields {
			fields[key] = value
		}
	}
	o.emit(ctx, "prompt_builder_render_finished", fields, nil)
}

func (o *promptBuilderObserver) TokenCountSkipped(ctx context.Context, fields map[string]any) {
	o.emit(ctx, "prompt_builder_token_count_skipped", fields, nil)
}

func (o *promptBuilderObserver) TokenCountFailed(ctx context.Context, fields map[string]any, err error) {
	o.emit(ctx, "prompt_builder_token_count_failed", fields, err)
}

func (o *promptBuilderObserver) includeSensitiveData() bool {
	return o != nil && o.debug != nil && o.debug.IncludeSensitiveData()
}

func (o *promptBuilderObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:Builder",
		Fields: fields,
		Err:    err,
	})
}

func promptDebugFields(ctx context.Context, parts []Part, prompt string) map[string]any {
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
	fields["prompt_structure"] = promptStructure(ctx, parts)
	return fields
}

func promptStructure(ctx context.Context, parts []Part) []map[string]any {
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
		node, err := part.Render(ctx)
		if err != nil {
			entry["render_error"] = err.Error()
			structure = append(structure, entry)
			continue
		}
		entry["node"] = renderNodeStructure(node)
		content := renderNodeText(node)
		entry["chars"] = len(content)
		entry["preview"] = clippedPrompt(content, promptDebugPreviewLimit, false)
		if len(content) > promptDebugPreviewLimit {
			entry["preview_tail"] = clippedPrompt(content, promptDebugPreviewLimit, true)
		}
		structure = append(structure, entry)
	}
	return structure
}

func renderNodeStructure(node RenderNode) map[string]any {
	entry := map[string]any{
		"type": node.Type,
	}
	if len(node.Fields) > 0 {
		fields := make([]map[string]string, 0, len(node.Fields))
		for _, field := range node.Fields {
			fields = append(fields, map[string]string{
				"key":   field.Key,
				"value": field.Value,
			})
		}
		entry["fields"] = fields
	}
	if node.Value != "" {
		entry["value_chars"] = len(node.Value)
		entry["value_preview"] = clippedPrompt(node.Value, promptDebugPreviewLimit, false)
	}
	if len(node.Children) > 0 {
		children := make([]map[string]any, 0, len(node.Children))
		for _, child := range node.Children {
			children = append(children, renderNodeStructure(child))
		}
		entry["children"] = children
	}
	return entry
}

func renderNodeText(node RenderNode) string {
	var builder strings.Builder
	appendRenderNodeText(&builder, node)
	return builder.String()
}

func appendRenderNodeText(builder *strings.Builder, node RenderNode) {
	if node.Value != "" {
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(node.Value)
	}
	for _, child := range node.Children {
		appendRenderNodeText(builder, child)
	}
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
