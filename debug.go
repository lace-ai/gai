package gai

import (
	"context"
	"encoding/json"
	"reflect"
	"time"
)

// Observation is a finalized, best-effort observability projection. It is not
// an ordered workflow execution event; applications needing replay or transport
// semantics should consume agent.Workflow.RunEvents instead.
type Observation struct {
	Name       string
	Source     string
	OccurredAt time.Time
	RunID      string
	TraceID    string
	SpanID     string
	Fields     map[string]any
	Err        error
}

// ObservationSink receives finalized observations synchronously. Implementations
// must return promptly and should enqueue internally when persistence is slow.
// GAI does not retry or wait for external persistence.
type ObservationSink interface {
	Emit(context.Context, Observation)
}

// ObservationSinkFunc adapts a function into an ObservationSink.
type ObservationSinkFunc func(context.Context, Observation)

func (f ObservationSinkFunc) Emit(ctx context.Context, observation Observation) {
	if f != nil {
		f(ctx, observation)
	}
}

// ObservationContentEnabled reports whether a library-managed content field of
// kind would be captured under ctx's ContentCapturePolicy.
func ObservationContentEnabled(ctx context.Context, sink ObservationSink, kind ContentKind) bool {
	if sink == nil {
		if _, _, err := SpanContextIDs(ctx); err != nil {
			return false
		}
	}
	policy, ok := ContentCapturePolicyFromContext(ctx)
	return ok && policy.captureMode(kind) == CaptureEnabled
}

// AddObservationContent adds one library-managed content field after applying
// the request-scoped ContentCapturePolicy. With no installed policy, content
// capture is disabled.
func AddObservationContent(ctx context.Context, sink ObservationSink, fields map[string]any, field string, kind ContentKind, value any) {
	if fields == nil || field == "" {
		return
	}
	if !ObservationContentEnabled(ctx, sink, kind) {
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

// EnrichObservation stamps time, run and trace correlation, and copies Fields
// so callers cannot mutate an emitted observation after the fact.
func EnrichObservation(ctx context.Context, observation Observation) Observation {
	if observation.OccurredAt.IsZero() {
		observation.OccurredAt = time.Now().UTC()
	}
	if observation.RunID == "" {
		observation.RunID, _ = ObservationRunIDFromContext(ctx)
	}
	if traceID, spanID, err := SpanContextIDs(ctx); err == nil {
		observation.TraceID = traceID
		observation.SpanID = spanID
	}
	fields := make(map[string]any, len(observation.Fields))
	for key, value := range observation.Fields {
		fields[key] = value
	}
	if observation.Err != nil {
		delete(fields, "error")
		fields["outcome"] = "error"
		fields["error_type"] = observationErrorType(observation.Err)
		// Errors may include provider responses, prompts, or tool output. Raw
		// errors remain available to control flow, but never leave it through a
		// finalized observation unless explicitly captured as managed content.
		observation.Err = nil
	}
	observation.Fields = fields
	return observation
}

func observationErrorType(err error) string {
	const maxErrorTypeBytes = 256
	typeName := reflect.TypeOf(err).String()
	if len(typeName) == 0 || len(typeName) > maxErrorTypeBytes {
		return "error"
	}
	return typeName
}

// EmitObservation finalizes and projects one semantic occurrence to OpenTelemetry
// and the optional application sink. Domain observers should use this helper
// rather than calling a sink directly.
func EmitObservation(ctx context.Context, sink ObservationSink, observation Observation) {
	observation = EnrichObservation(ctx, observation)
	RecordObservation(ctx, observation)
	if sink != nil {
		sink.Emit(ctx, observation)
	}
}

type observationRunIDContextKey struct{}

// WithObservationRunID attaches workflow correlation to derived observations.
func WithObservationRunID(ctx context.Context, runID string) context.Context {
	return context.WithValue(ctx, observationRunIDContextKey{}, runID)
}

// ObservationRunIDFromContext returns the workflow correlation ID when present.
func ObservationRunIDFromContext(ctx context.Context) (string, bool) {
	if ctx == nil {
		return "", false
	}
	runID, ok := ctx.Value(observationRunIDContextKey{}).(string)
	return runID, ok && runID != ""
}
