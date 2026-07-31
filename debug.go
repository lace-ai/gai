package gai

import (
	"context"
	"encoding/json"
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
	// IncludeSensitiveData enables the legacy unbounded, all-or-nothing
	// content path when no ContentCapturePolicy is installed on ctx.
	// Prefer WithContentCapturePolicy for new applications.
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

// SensitiveDebugSinkFunc enables the legacy raw, unbounded content path.
//
// Deprecated: use DebugSinkFunc with WithContentCapturePolicy.
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

// DebugContentEnabled reports whether a library-managed debug field of kind
// would be captured for ctx and sink. It is useful for avoiding expensive
// serialization before calling AddDebugContent.
func DebugContentEnabled(ctx context.Context, sink DebugSink, kind ContentKind) bool {
	if sink == nil {
		return false
	}
	if policy, hasPolicy := ContentCapturePolicyFromContext(ctx); hasPolicy {
		return policy.captureMode(kind) == CaptureEnabled
	}
	return sink.IncludeSensitiveData()
}

// AddDebugContent adds one library-managed content field after applying the
// request-scoped ContentCapturePolicy. An installed policy is authoritative,
// including for legacy sensitive sinks. With no installed policy, the original
// IncludeSensitiveData behavior and field shape are preserved.
func AddDebugContent(ctx context.Context, sink DebugSink, fields map[string]any, field string, kind ContentKind, value any) {
	if sink == nil || fields == nil || field == "" {
		return
	}
	if _, hasPolicy := ContentCapturePolicyFromContext(ctx); !hasPolicy {
		if DebugContentEnabled(ctx, sink, kind) {
			fields[field] = value
		}
		return
	}

	raw, ok := debugContentBytes(value)
	if !ok {
		return
	}
	captured, ok := CaptureContent(ctx, kind, raw)
	if !ok {
		return
	}
	fields[field] = string(captured.Value)
	fields[field+"_original_bytes"] = captured.OriginalBytes
	fields[field+"_captured_bytes"] = captured.CapturedBytes
	fields[field+"_truncated"] = captured.Truncated
	fields[field+"_redaction_applied"] = captured.RedactionApplied
	fields[field+"_content_kind"] = string(kind)
}

func debugContentBytes(value any) (raw []byte, ok bool) {
	ok = false
	defer func() {
		if recover() != nil {
			raw = nil
			ok = false
		}
	}()
	switch value := value.(type) {
	case string:
		return []byte(value), true
	case []byte:
		return append([]byte(nil), value...), true
	case json.RawMessage:
		return append([]byte(nil), value...), true
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, false
		}
		return encoded, true
	}
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
