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
	IncludeSencitiveData() bool
}

type DebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f DebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		f(ctx, e)
	}
}

func (f DebugSinkFunc) IncludeSencitiveData() bool {
	return false
}

type SensitiveDebugSinkFunc func(ctx context.Context, e DebugEvent)

func (f SensitiveDebugSinkFunc) Emit(ctx context.Context, e DebugEvent) {
	if f != nil {
		f(ctx, e)
	}
}

func (f SensitiveDebugSinkFunc) IncludeSencitiveData() bool {
	return true
}
