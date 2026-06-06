package gai

import "context"

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
		f(ctx, EnrichDebugEvent(ctx, e))
	}
}

func (f DebugSinkFunc) IncludeSensitiveData() bool {
	return false
}

type SensitiveDebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f SensitiveDebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		f(ctx, EnrichDebugEvent(ctx, e))
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

	fields := make(map[string]any, len(e.Fields)+2)
	for key, value := range e.Fields {
		fields[key] = value
	}
	fields["trace_id"] = traceID
	fields["span_id"] = spanID
	e.Fields = fields
	return e
}
