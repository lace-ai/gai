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
}

type DebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f DebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		f(ctx, e)
	}
}
