package tooldefinitions

import (
	"context"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/lace-ai/gai/context/tooldefinitions"

type observer struct {
	debug       gai.DebugSink
	span        trace.Span
	toolCount   int
	tokenBudget int
}

func newObserver(ctx context.Context, debug gai.DebugSink, toolCount int, tokenBudget int) (context.Context, *observer) {
	ctx, span := gai.StartOperationSpan(ctx, tracerName, "context.tool_definitions", "context.operation", "build",
		attribute.String("context.source", "tool_definitions"),
		attribute.Int("context.tool_count", toolCount),
		attribute.Int("context.token_budget", tokenBudget),
	)
	return ctx, &observer{debug: debug, span: span, toolCount: toolCount, tokenBudget: tokenBudget}
}

func (o *observer) Finish(err error) {
	if o == nil || o.span == nil {
		return
	}
	gai.EndSpan(o.span, err)
}

func (o *observer) Started(ctx context.Context) {
	o.emit(ctx, "tool_definitions_build_started", o.fields(), nil)
}

func (o *observer) Succeeded(ctx context.Context, toolNames []string) {
	fields := o.fields()
	fields["tool_names"] = toolNames
	if o != nil && o.span != nil {
		o.span.SetAttributes(attribute.StringSlice("context.tool_names", toolNames))
	}
	o.emit(ctx, "tool_definitions_build_finished", fields, nil)
}

func (o *observer) Failed(ctx context.Context, stage string, err error) {
	fields := o.fields()
	fields["stage"] = stage
	o.emit(ctx, "tool_definitions_build_failed", fields, err)
}

func (o *observer) fields() map[string]any {
	fields := map[string]any{}
	if o == nil {
		return fields
	}
	fields["tool_count"] = o.toolCount
	fields["token_budget"] = o.tokenBudget
	return fields
}

func (o *observer) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context/tooldefinitions:Source.Function",
		Fields: fields,
		Err:    err,
	})
}
