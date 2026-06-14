package gai

import (
	"context"
	"math"
)

type DebugEvent struct {
	Name   string
	Source string
	Fields map[string]any
	Err    error
}

type DebugSink interface {
	Emit(ctx context.Context, e DebugEvent)
	IncludeSensitiveData() bool
}

type DebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f DebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		event := EnrichDebugEvent(ctx, e)
		RecordDebugEvent(ctx, event)
		f(ctx, event)
	}
}

func (f DebugSinkFunc) IncludeSensitiveData() bool {
	return false
}

type SensitiveDebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f SensitiveDebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		event := EnrichDebugEvent(ctx, e)
		RecordDebugEvent(ctx, event)
		f(ctx, event)
	}
}

func (f SensitiveDebugSinkFunc) IncludeSensitiveData() bool {
	return true
}

func EnrichDebugEvent(ctx context.Context, e DebugEvent) DebugEvent {
	traceID, spanID, err := SpanContextIDs(ctx)
	if err != nil {
		return e
	}

	capHint := 0
	if len(e.Fields) <= math.MaxInt-2 {
		capHint = len(e.Fields) + 2
	}
	fields := make(map[string]any, capHint)
	for key, value := range e.Fields {
		fields[key] = value
	}
	fields["otel"] = map[string]any{
		"trace_id": traceID,
		"span_id":  spanID,
	}
	fields["trace_id"] = traceID
	fields["span_id"] = spanID
	e.Fields = fields
	return e
}
